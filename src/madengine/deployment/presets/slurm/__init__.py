"""
SLURM deployment presets.

Layered configuration system:
1. defaults.json - Base SLURM defaults
2. profiles/*.json - Workload-specific profiles (single-node, multi-node)
3. clusters/*.json - Facts about the cluster the job lands on, selected by
   slurm.cluster_profile; see ../cluster_profiles.py
4. User configuration - Highest priority

Convention over Configuration:
- Presence of "slurm" field → SLURM deployment
- No explicit "deploy" field needed

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
