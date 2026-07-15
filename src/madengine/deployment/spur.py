#!/usr/bin/env python3
"""
Spur (Crusoe) deployment backend.

Spur is an "AI-native" scheduler that exposes SLURM-compatible CLI shims
(sbatch/srun/squeue/sacct/scontrol/...) but differs from stock SLURM in ways
that break the standard multi-node flow:

  * `srun` cannot fan out tasks across nodes: any `srun [-N -n] [--mpi ...]`
    invocation runs the command once on the head node, and SLURM_PROCID is
    empty inside srun. The stock madengine template relies on
    `srun bash task_script` launching one task per node with a unique
    SLURM_PROCID, so only rank 0 would ever start.
  * `scontrol show hostname[s]` is unsupported (SLURM_NODELIST is already an
    expanded comma list).
  * The control plane is Raft-based / eventually consistent: sbatch can
    transiently fail ("not the Raft leader") and squeue/sacct states flap.

Strategy: reuse the SLURM template and orchestration, but drive multi-node
execution with a job ARRAY of single-node tasks (one array task per node).
`SLURM_ARRAY_TASK_ID` is the node rank; the tasks self-form the cluster via the
model launcher's TCP rendezvous (rank 0 publishes its transport IP to a shared
filesystem, peers read it as MASTER_ADDR). The spur-specific branches live in
`templates/slurm/job.sh.j2` under `{% if scheduler == 'spur' %}` and are enabled
purely by the template context produced here.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from .base import DeploymentConfig, DeploymentResult, DeploymentStatus
from .slurm import SlurmDeployment


class SpurDeployment(SlurmDeployment):
    """SLURM-compatible deployment for the spur scheduler (job-array fan-out)."""

    DEPLOYMENT_TYPE = "spur"
    # spur ships slurm-compatible shims. scontrol exists but is only partially
    # implemented; the spur flow does not depend on it, so we don't require it.
    REQUIRED_TOOLS = ["sbatch", "squeue", "sacct"]

    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        # Rendezvous root MUST be on a shared (NFS) filesystem visible to every
        # node: rank 0 writes MASTER_ADDR here and peers read it. output_dir is
        # under the (shared) submission/run directory.
        self.rendezvous_dir = str(self.output_dir.resolve() / "spur_rendezvous")

    def _prepare_template_context(self, model_info: Dict) -> Dict[str, Any]:
        context = super()._prepare_template_context(model_info)
        context["scheduler"] = "spur"
        context["rendezvous_dir"] = self.rendezvous_dir
        return context

    def _model_job_name(self) -> str:
        """The #SBATCH --job-name used by the template (madengine-<model name>)."""
        try:
            models = self.manifest.get("built_models") or {}
            first = next(iter(models.values()), {})
            name = first.get("name") or next(iter(models), "")
            return f"madengine-{name}"
        except Exception:
            return "madengine-"

    def _live_task_count(self, job_name: str) -> int:
        """Count my not-yet-finished array tasks in the queue (best-effort).

        Used only as a liveness guard so monitor() does not wait forever if a
        task dies before writing its completion marker. squeue is eventually
        consistent on spur, so a transient 0 is tolerated by the caller.
        """
        try:
            result = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return -1  # unknown
            live = 0
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                name, _, state = line.partition("|")
                if name == job_name and state.upper() in (
                    "PENDING",
                    "RUNNING",
                    "CONFIGURING",
                    "COMPLETING",
                    "RESIZING",
                    "SUSPENDED",
                ):
                    live += 1
            return live
        except Exception:
            return -1  # unknown

    # Number of consecutive polls, AFTER the tasks were first seen alive, with no
    # completion markers AND no live tasks before we conclude the array died
    # without reporting. ~poll interval (30s) times this many => grace window.
    # The "seen alive first" gate is essential on spur: for the first ~1-2 min
    # after sbatch, squeue does not yet list the array tasks (registration lag /
    # eventual consistency), so a fresh, healthy run reports 0 live tasks.
    _SPUR_DEAD_POLLS = 4

    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Marker-based completion detection for the spur job array.

        Each array task writes ``done_rank<rank>`` (its exit code) into
        ``<rendezvous_dir>/<array_job_id>/`` on the shared filesystem. We treat
        those markers as the source of truth because spur's ``sacct -j`` does not
        filter by job id and ``squeue`` is eventually consistent.
        """
        marker_dir = Path(self.rendezvous_dir) / str(deployment_id)
        n = int(self.nodes)

        codes: Dict[int, int] = {}
        if marker_dir.is_dir():
            for rank in range(n):
                f = marker_dir / f"done_rank{rank}"
                if f.exists():
                    try:
                        codes[rank] = int((f.read_text().strip() or "1"))
                    except ValueError:
                        codes[rank] = 1

        if len(codes) >= n:
            failed = {r: c for r, c in codes.items() if c != 0}
            if not failed:
                self._show_log_summary(deployment_id, success=True)
                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=deployment_id,
                    message=f"All {n} array tasks completed successfully",
                )
            self._show_log_summary(deployment_id, success=False)
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id=deployment_id,
                message=f"Array task(s) failed (rank:exit) {failed}",
            )

        # Not all ranks done yet. Guard against a task that died without writing a
        # marker, but only AFTER we have seen the tasks alive at least once: right
        # after sbatch, spur's squeue does not yet list the array tasks, so a fresh
        # healthy run legitimately reports 0 live tasks for the first ~1-2 min.
        live = self._live_task_count(self._model_job_name())
        if live > 0:
            self._spur_seen_live = True
            self._spur_empty_polls = 0
        elif live == 0 and getattr(self, "_spur_seen_live", False) and len(codes) < n:
            # Tasks were running earlier and now none are queued and not all
            # ranks reported: a transient empty squeue is possible, so require
            # several consecutive empty polls before declaring failure.
            self._spur_empty_polls = getattr(self, "_spur_empty_polls", 0) + 1
            if self._spur_empty_polls >= self._SPUR_DEAD_POLLS:
                self._show_log_summary(deployment_id, success=False)
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=deployment_id,
                    message=(
                        f"Only {len(codes)}/{n} ranks reported completion and no "
                        f"array tasks remain in the queue"
                    ),
                )
        else:
            # live == -1 (squeue unavailable/unknown) or still in startup grace.
            self._spur_empty_polls = 0

        return DeploymentResult(
            status=DeploymentStatus.RUNNING,
            deployment_id=deployment_id,
            message=f"{len(codes)}/{n} ranks done (live tasks: {live})",
        )
