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

from typing import Any, Dict

from .base import DeploymentConfig
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
