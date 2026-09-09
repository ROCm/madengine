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
from typing import Any, Dict, List

from .base import DeploymentConfig, DeploymentResult, DeploymentStatus
from .slurm import SlurmDeployment


# Seconds a non-zero rank waits for rank 0 to publish MASTER_ADDR. A job array
# carries no gang-scheduling guarantee, so tasks can start minutes apart (and
# with --exclusive they may even start serially); override per site with
# slurm.rendezvous_timeout.
DEFAULT_RENDEZVOUS_TIMEOUT = 900


def render_rendezvous_block(rendezvous_dir: str, timeout: int) -> List[str]:
    """Bash lines resolving MASTER_ADDR via a shared-filesystem rendezvous.

    Rank 0 publishes its transport IP to ``<rendezvous_dir>/<array job
    id>/master_addr``; the other ranks poll for it. Requires ``SLURM_PROCID``
    to already hold the array task id; exports ``_MAD_REND_DIR`` (reused for
    the per-rank ``done_rank`` markers) and ``MASTER_ADDR``.

    A peer that times out fails fast rather than continuing with an empty
    MASTER_ADDR (which fails obscurely inside the launcher): it prints a
    diagnostic, writes a non-zero ``done_rank`` marker so monitor() reports the
    failure immediately, and exits non-zero.

    Args:
        rendezvous_dir: Shared-filesystem root, visible from every node.
        timeout: Seconds a non-zero rank waits for rank 0.

    Returns:
        The bash lines, one per list element.
    """
    return [
        f'_MAD_REND_DIR="{rendezvous_dir}/${{SLURM_ARRAY_JOB_ID:-${{SLURM_JOB_ID}}}}"',
        'mkdir -p "$_MAD_REND_DIR" 2>/dev/null || true',
        '_MAD_IFACE="${NCCL_SOCKET_IFNAME:-ens3}"; _MAD_IFACE="${_MAD_IFACE%%,*}"',
        "_MAD_MY_IP=\"$(ip -4 -o addr show \"$_MAD_IFACE\" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n1)\"",
        "[ -z \"$_MAD_MY_IP\" ] && _MAD_MY_IP=\"$(hostname -I | awk '{print $1}')\"",
        'if [ "${SLURM_PROCID}" = "0" ]; then',
        '    echo "$_MAD_MY_IP" > "$_MAD_REND_DIR/master_addr"',
        '    export MASTER_ADDR="$_MAD_MY_IP"',
        "else",
        f"    _MAD_REND_TIMEOUT={int(timeout)}",
        '    for _i in $(seq 1 "$_MAD_REND_TIMEOUT"); do [ -s "$_MAD_REND_DIR/master_addr" ] && break; sleep 1; done',
        '    export MASTER_ADDR="$(cat "$_MAD_REND_DIR/master_addr" 2>/dev/null || true)"',
        '    if [ -z "$MASTER_ADDR" ]; then',
        '        echo "[spur-rendezvous] ERROR: rank ${SLURM_PROCID} on $(hostname) timed out after ${_MAD_REND_TIMEOUT}s waiting for $_MAD_REND_DIR/master_addr" >&2',
        '        echo "[spur-rendezvous] Rank 0 never started (array tasks are not gang-scheduled), died early, or the rendezvous dir is not on a shared filesystem." >&2',
        '        echo "[spur-rendezvous] Raise slurm.rendezvous_timeout above ${_MAD_REND_TIMEOUT}s if the queue wait is simply longer than that." >&2',
        '        echo "1" > "$_MAD_REND_DIR/done_rank${SLURM_PROCID}"',
        "        exit 1",
        "    fi",
        "fi",
        'echo "[spur-rendezvous] rank=${SLURM_PROCID} node=$(hostname) my_ip=$_MAD_MY_IP MASTER_ADDR=${MASTER_ADDR}"',
    ]


class SpurDeployment(SlurmDeployment):
    """SLURM-compatible deployment for the spur scheduler (job-array fan-out)."""

    DEPLOYMENT_TYPE = "spur"
    # spur ships slurm-compatible shims. scontrol exists but is only partially
    # implemented; the spur flow does not depend on it, so we don't require it.
    REQUIRED_TOOLS = ["sbatch", "squeue", "sacct"]
    # Drives the spur-specific branches in the inherited SLURM code paths
    # (template rendering and the slurm_multi launcher): job-array fan-out
    # instead of srun.
    IS_SPUR = True

    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        # Rendezvous root MUST be on a shared (NFS) filesystem visible to every
        # node: rank 0 writes MASTER_ADDR here and peers read it. output_dir is
        # under the (shared) submission/run directory.
        self.rendezvous_dir = str(self.output_dir.resolve() / "spur_rendezvous")
        self.rendezvous_timeout = int(
            self.slurm_config.get("rendezvous_timeout", DEFAULT_RENDEZVOUS_TIMEOUT)
        )

    def _prepare_template_context(self, model_info: Dict) -> Dict[str, Any]:
        context = super()._prepare_template_context(model_info)
        context["scheduler"] = "spur"
        context["rendezvous_dir"] = self.rendezvous_dir
        # Rendered as bash (not escaped) into the spur branch of job.sh.j2; the
        # slurm_multi launcher script emits the same block.
        context["spur_rendezvous_block"] = "\n".join(
            render_rendezvous_block(self.rendezvous_dir, self.rendezvous_timeout)
        )
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

    def _live_task_count(self, deployment_id: str, job_name: str) -> int:
        """Count my not-yet-finished array tasks in the queue (best-effort).

        Used only as a liveness guard so monitor() does not wait forever if a
        task dies before writing its completion marker. squeue is eventually
        consistent on spur, so a transient 0 is tolerated by the caller.

        Args:
            deployment_id: Array job id returned by sbatch.
            job_name: #SBATCH --job-name, used only if no row carries our id.

        Returns:
            Number of live tasks, or -1 if squeue could not be queried.
        """
        try:
            # NOTE: spur's squeue ignores custom -o delimiters (e.g. "%j|%T"
            # renders as "<name> <state>", space-separated), so parse by
            # whitespace. Job names produced by the template contain no spaces.
            cmd = ["squeue", "-h", "-o", "%i %j %T"]
            user = os.environ.get("USER", "")
            if user:
                # Omit -u entirely when USER is unset: `squeue -u ""` is an error.
                cmd[1:1] = ["-u", user]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return -1  # unknown
            live_states = {
                "PENDING",
                "RUNNING",
                "CONFIGURING",
                "COMPLETING",
                "RESIZING",
                "SUSPENDED",
            }
            by_id = 0
            by_name = 0
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                task_id, name, state = parts[0], parts[1], parts[-1]
                if state.upper() not in live_states:
                    continue
                # Array tasks are listed as "<array job id>_<index>" (or as a
                # pending range, "<array job id>_[1-3]"). Matching the id keeps a
                # concurrent run of the same model from inflating the count.
                if task_id == deployment_id or task_id.startswith(f"{deployment_id}_"):
                    by_id += 1
                elif name == job_name:
                    by_name += 1
            # Fall back to name matching only if squeue reported no row for our
            # job id at all (i.e. it does not label array tasks the way we expect).
            return by_id if by_id else by_name
        except Exception:
            return -1  # unknown

    # Number of consecutive polls, AFTER the tasks were first seen alive, with no
    # completion markers AND no live tasks before we conclude the array died
    # without reporting. ~poll interval (30s) times this many => grace window.
    # The "seen alive first" gate is essential on spur: for the first ~1-2 min
    # after sbatch, squeue does not yet list the array tasks (registration lag /
    # eventual consistency), so a fresh, healthy run reports 0 live tasks.
    _SPUR_DEAD_POLLS = 4

    # Number of consecutive polls where squeue could not be queried at all before
    # giving up. Completion markers still win if they appear, so this only bounds
    # the case where the control plane stays unreachable and the markers never
    # arrive; ~poll interval (30s) times this many => ~10 minutes.
    _SPUR_UNKNOWN_POLLS = 20

    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Marker-based completion detection for the spur job array.

        Each array task writes ``done_rank<rank>`` (its exit code) into
        ``<rendezvous_dir>/<array_job_id>/`` on the shared filesystem. We treat
        those markers as the source of truth because spur's ``sacct -j`` does not
        filter by job id and ``squeue`` is eventually consistent.
        """
        marker_dir = Path(self.rendezvous_dir) / str(deployment_id)
        n = int(self.nodes)
        live_output = self.config.additional_context.get("live_output", False)

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
            self._report_logs(deployment_id, success=not failed, live_output=live_output)
            if not failed:
                return DeploymentResult(
                    status=DeploymentStatus.SUCCESS,
                    deployment_id=deployment_id,
                    message=f"All {n} array tasks completed successfully",
                )
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                deployment_id=deployment_id,
                message=f"Array task(s) failed (rank:exit) {failed}",
            )

        # Not all ranks done yet. Guard against a task that died without writing a
        # marker, but only AFTER we have seen the tasks alive at least once: right
        # after sbatch, spur's squeue does not yet list the array tasks, so a fresh
        # healthy run legitimately reports 0 live tasks for the first ~1-2 min.
        live = self._live_task_count(deployment_id, self._model_job_name())
        if live > 0:
            self._spur_seen_live = True
            self._spur_empty_polls = 0
            self._spur_unknown_polls = 0
        elif live == 0 and getattr(self, "_spur_seen_live", False) and len(codes) < n:
            # Tasks were running earlier and now none are queued and not all
            # ranks reported: a transient empty squeue is possible, so require
            # several consecutive empty polls before declaring failure.
            self._spur_unknown_polls = 0
            self._spur_empty_polls = getattr(self, "_spur_empty_polls", 0) + 1
            if self._spur_empty_polls >= self._SPUR_DEAD_POLLS:
                self._report_logs(deployment_id, success=False, live_output=live_output)
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    deployment_id=deployment_id,
                    message=(
                        f"Only {len(codes)}/{n} ranks reported completion and no "
                        f"array tasks remain in the queue"
                    ),
                )
        elif live < 0:
            # squeue could not be queried. Bound this too: otherwise a control
            # plane that stays down leaves monitor() returning RUNNING forever
            # (the caller polls without a timeout).
            self._spur_empty_polls = 0
            self._spur_unknown_polls = getattr(self, "_spur_unknown_polls", 0) + 1
            if self._spur_unknown_polls >= self._SPUR_UNKNOWN_POLLS:
                self._report_logs(deployment_id, success=False, live_output=live_output)
                return DeploymentResult(
                    status=DeploymentStatus.UNKNOWN,
                    deployment_id=deployment_id,
                    message=(
                        f"squeue unavailable for {self._spur_unknown_polls} consecutive "
                        f"polls and only {len(codes)}/{n} ranks reported completion"
                    ),
                )
        else:
            # Still in the startup grace window (tasks not yet registered).
            self._spur_empty_polls = 0
            self._spur_unknown_polls = 0

        if live_output:
            self._stream_job_output(deployment_id)

        return DeploymentResult(
            status=DeploymentStatus.RUNNING,
            deployment_id=deployment_id,
            message=f"{len(codes)}/{n} ranks done (live tasks: {live})",
        )

    def _report_logs(self, deployment_id: str, success: bool, live_output: bool) -> None:
        """Emit final logs the same way SlurmDeployment.monitor() does.

        Args:
            deployment_id: Array job id.
            success: Whether the run succeeded (only used for the summary).
            live_output: Whether the user asked for streamed output.
        """
        if live_output:
            self._stream_job_output(deployment_id, final=True)
        else:
            self._show_log_summary(deployment_id, success=success)
