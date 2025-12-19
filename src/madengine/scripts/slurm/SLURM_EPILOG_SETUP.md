# SLURM Epilog Script Setup Guide

This guide explains how to install and configure the SLURM epilog script to automatically clean up GPU processes after each job completes or fails.

## Problem Statement

In multi-node GPU jobs, when a job fails or is cancelled:
- Ray worker processes may continue running in Docker containers on compute nodes
- These "zombie" processes hold GPU memory (100-180 GB per GPU)
- Subsequent jobs fail with "insufficient GPU memory" errors
- Manual cleanup is required on each node

## Solution: SLURM Epilog Script

The epilog script runs **automatically after every job** (success or failure) on each compute node to:
1. Kill Ray worker processes
2. Kill vLLM processes  
3. Clean up Docker containers
4. Kill any remaining GPU processes
5. Optionally reset GPU state

---

## Installation

### 1. Copy Script to Compute Nodes

On **each SLURM compute node**, copy the epilog script:

```bash
sudo cp src/madengine/scripts/slurm/epilog.sh /etc/slurm/epilog.sh
sudo chmod +x /etc/slurm/epilog.sh
sudo chown root:root /etc/slurm/epilog.sh
```

### 2. Create Log Directory

```bash
sudo mkdir -p /var/log/slurm
sudo chmod 755 /var/log/slurm
```

### 3. Configure SLURM

Edit `/etc/slurm/slurm.conf` on the **SLURM controller** and add:

```conf
# Epilog script to clean up GPU processes after each job
Epilog=/etc/slurm/epilog.sh

# Optional: Set timeout for epilog script (default: 60 seconds)
EpilogMsgTime=30
```

### 4. Restart SLURM Services

On **compute nodes**:
```bash
sudo systemctl restart slurmd
```

On **controller**:
```bash
sudo systemctl restart slurmctld
```

---

## Verification

### 1. Submit a Test Job

```bash
sbatch --nodes=1 --gpus-per-node=1 --time=00:01:00 --wrap="python3 -c 'import time; time.sleep(30)'"
```

### 2. Check Epilog Logs

On the compute node where the job ran:

```bash
sudo tail -f /var/log/slurm/epilog.log
```

You should see entries like:
```
[2025-12-17 12:34:56] [Job 12345] === Epilog script starting ===
[2025-12-17 12:34:56] [Job 12345] Checking for GPU processes...
[2025-12-17 12:34:56] [Job 12345] No GPU processes found
[2025-12-17 12:34:56] [Job 12345] === Epilog script completed ===
```

### 3. Test GPU Cleanup After Failed Job

Submit a job that will fail:
```bash
sbatch --nodes=2 --gpus-per-node=4 <<EOF
#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --partition=amd-rccl

# Start Ray cluster but don't clean up (simulate failure)
ray start --head --num-gpus=4 &
sleep 60
exit 1  # Simulate failure
EOF
```

After the job fails, check if Ray processes were cleaned:
```bash
# On the compute node
ps aux | grep ray
amd-smi process  # Should show no processes
```

---

## Configuration Options

### Enable GPU Reset (Optional)

If you want to reset GPU state after each job (may cause brief GPU unavailability):

Edit `/etc/slurm/epilog.sh` and uncomment:
```bash
# Step 4: Reset GPU state (optional, may cause brief GPU unavailability)
# Uncomment if needed:
reset_gpu_state
```

**Warning**: GPU reset may cause a brief period where GPUs are unavailable to other jobs.

### Adjust Logging Level

To reduce log verbosity, edit the `log_message` function in `/etc/slurm/epilog.sh`:

```bash
# Only log errors and warnings
log_message() {
    if [ "$2" = "ERROR" ] || [ "$2" = "WARN" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$2] [Job ${SLURM_JOB_ID:-unknown}] $1" >> "$LOG_FILE"
    fi
}
```

### Exclude Specific Jobs

To skip cleanup for certain jobs (e.g., debugging), check the job name:

```bash
# At the start of epilog.sh
if [ "$SLURM_JOB_NAME" = "debug_session" ]; then
    log_message "Skipping cleanup for debug session"
    exit 0
fi
```

---

## Troubleshooting

### Epilog Script Not Running

**Symptom**: No entries in `/var/log/slurm/epilog.log` after jobs complete

**Solutions**:
1. Verify script permissions:
   ```bash
   ls -la /etc/slurm/epilog.sh
   # Should be: -rwxr-xr-x root root
   ```

2. Check SLURM configuration:
   ```bash
   grep Epilog /etc/slurm/slurm.conf
   # Should show: Epilog=/etc/slurm/epilog.sh
   ```

3. Check SLURM logs:
   ```bash
   sudo tail -f /var/log/slurm/slurmd.log
   ```

### Epilog Script Times Out

**Symptom**: SLURM logs show "Epilog timed out"

**Solution**: Increase timeout in `slurm.conf`:
```conf
EpilogMsgTime=60
```

### GPU Processes Still Present

**Symptom**: After epilog runs, GPU processes still exist

**Solution**: 
1. Check if processes are in Docker containers:
   ```bash
   docker ps -a | grep container_rocm
   ```

2. Add more aggressive Docker cleanup to epilog script:
   ```bash
   # In cleanup_docker_containers()
   docker ps -q | xargs -r docker kill
   docker ps -aq | xargs -r docker rm -f
   ```

### Permissions Errors

**Symptom**: Epilog log shows "Permission denied" errors

**Solution**: Epilog runs as root by default. If issues persist:
1. Check SELinux status: `getenforce`
2. Add SELinux policy or disable: `sudo setenforce 0`

---

## Best Practices

### 1. Monitor Epilog Logs

Set up log rotation for epilog logs:

```bash
sudo cat > /etc/logrotate.d/slurm-epilog <<EOF
/var/log/slurm/epilog.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0644 root root
}
EOF
```

### 2. Test Changes in Staging

Before deploying to production:
1. Test on a staging compute node
2. Run manual cleanup: `/etc/slurm/epilog.sh`
3. Verify no issues with running jobs

### 3. Coordinate with Prolog

If you also use a prolog script, ensure they don't conflict:
- Prolog: Setup/verification before job
- Epilog: Cleanup after job

Example prolog to verify GPUs are clean:
```bash
#!/bin/bash
# /etc/slurm/prolog.sh
if amd-smi process 2>/dev/null | grep -q '[0-9]'; then
    echo "ERROR: GPUs not clean before job start"
    exit 1
fi
```

---

## Integration with madengine

The epilog script is designed to work seamlessly with madengine's `run.sh` cleanup:

1. **During Job**: `run.sh` trap handler cleans up on script exit
2. **After Job**: SLURM epilog catches any missed processes
3. **Defense in Depth**: Two layers of cleanup ensure robustness

This dual-layer approach ensures GPU resources are always released, even if:
- The job is killed with SIGKILL
- Docker containers fail to stop
- Ray workers don't respond to shutdown signals

---

## References

- [SLURM Prolog/Epilog Documentation](https://slurm.schedmd.com/prolog_epilog.html)
- [Ray Cluster Cleanup Best Practices](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)

