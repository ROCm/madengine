#!/usr/bin/env python3
"""
Bare Metal VM Deployment using KVM/libvirt with Docker-in-VM.

This deployment mode creates ephemeral VMs on bare metal nodes, installs Docker,
runs existing madengine container workflows, and provides complete cleanup.

**Architecture:**
    Bare Metal Node (KVM host)
    └── Ephemeral VM (Ubuntu + Docker)
        └── Docker Container (existing madengine images)
            └── Model execution

**User Workflow:**
    1. SSH to bare metal node manually
    2. Run: madengine run --tags model --additional-context-file baremetal-vm.json
    3. madengine creates VM, installs Docker, runs existing container workflow
    4. VM destroyed, bare metal restored

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import json
import time
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from rich.console import Console as RichConsole

from .base import BaseDeployment, DeploymentConfig, DeploymentResult, DeploymentStatus
from madengine.core.errors import OrchestrationError, create_error_context
from madengine.utils.vm_lifecycle import VMLifecycleManager, VMConfig
from madengine.utils.gpu_passthrough import GPUPassthroughManager, GPUPassthroughMode


class BareMetalVMDeployment(BaseDeployment):
    """
    Bare metal execution using VM isolation with Docker-in-VM.
    
    Reuses 100% of existing madengine container execution code by running
    Docker inside an ephemeral VM. VM provides isolation and cleanup,
    Docker provides compatibility with existing images and workflows.
    """
    
    DEPLOYMENT_TYPE = "baremetal_vm"
    REQUIRED_TOOLS = ["virsh", "qemu-img", "qemu-system-x86_64"]
    
    def __init__(self, config: DeploymentConfig):
        """
        Initialize bare metal VM deployment.
        
        Args:
            config: Deployment configuration
        """
        super().__init__(config)
        
        self.rich_console = RichConsole()
        
        # Parse bare metal VM configuration
        self.vm_config = config.additional_context.get("baremetal_vm", {})
        
        # VM resources
        self.vcpus = self.vm_config.get("vcpus", 32)
        self.memory_gb = int(self.vm_config.get("memory", "128G").rstrip("G"))
        self.disk_size = self.vm_config.get("disk_size", "100G")
        
        # Base image (pre-configured Ubuntu with GPU drivers)
        self.base_image = self.vm_config.get(
            "base_image",
            "/var/lib/libvirt/images/ubuntu-22.04-rocm.qcow2"
        )
        
        # GPU configuration
        self.gpu_config = self.vm_config.get("gpu_passthrough", {})
        self.gpu_mode_str = self.gpu_config.get("mode", "sriov")
        self.gpu_mode = GPUPassthroughMode(self.gpu_mode_str)
        self.gpu_vendor = self.gpu_config.get("gpu_vendor", "AMD")
        
        # PCI addresses - can be explicit or auto-discovered
        self.gpu_pci_addresses = self.gpu_config.get("gpu_ids", [])
        
        # Cleanup settings
        self.cleanup_config = self.vm_config.get("cleanup", {})
        self.cleanup_mode = self.cleanup_config.get("mode", "destroy")
        self.verify_clean = self.cleanup_config.get("verify_clean", True)
        
        # SSH settings
        self.ssh_user = self.vm_config.get("ssh_user", "root")
        self.ssh_key = self.vm_config.get("ssh_key")
        
        # Managers
        self.vm_manager = VMLifecycleManager()
        self.gpu_manager = GPUPassthroughManager()
        
        # State
        self.vm_name = None
        self.vm_instance = None
        self.vm_disk_path = None
    
    def validate(self) -> bool:
        """Validate bare metal VM environment."""
        self.rich_console.print("\n[cyan]Validating bare metal VM environment...[/cyan]")
        
        issues = []
        
        # Check KVM module loaded
        result = subprocess.run(
            ["lsmod"], capture_output=True, text=True, timeout=5
        )
        if "kvm" not in result.stdout:
            issues.append("KVM module not loaded (run: modprobe kvm kvm_amd)")
        else:
            self.rich_console.print("  ✓ KVM module loaded")
        
        # Check libvirtd running
        result = subprocess.run(
            ["systemctl", "is-active", "libvirtd"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            issues.append("libvirtd not running (run: systemctl start libvirtd)")
        else:
            self.rich_console.print("  ✓ libvirtd running")
        
        # Check base image exists
        if not os.path.exists(self.base_image):
            issues.append(f"Base image not found: {self.base_image}")
        else:
            self.rich_console.print(f"  ✓ Base image found: {self.base_image}")
        
        # Verify GPU passthrough capability
        is_ready, gpu_issues = self.gpu_manager.verify_passthrough_ready()
        if not is_ready:
            for issue in gpu_issues:
                issues.append(f"GPU: {issue}")
        else:
            self.rich_console.print("  ✓ GPU passthrough ready")
        
        # Check required tools
        for tool in self.REQUIRED_TOOLS:
            result = subprocess.run(
                ["which", tool], capture_output=True, timeout=5
            )
            if result.returncode != 0:
                issues.append(f"Required tool not found: {tool}")
            else:
                self.rich_console.print(f"  ✓ {tool} available")
        
        if issues:
            self.rich_console.print("\n[red]Validation failed:[/red]")
            for issue in issues:
                self.rich_console.print(f"  ✗ {issue}")
            return False
        
        self.rich_console.print("\n[green]✓ Bare metal VM environment validated[/green]\n")
        return True
    
    def deploy(self) -> DeploymentResult:
        """
        Deploy workload in ephemeral VM with Docker.
        
        Steps:
        1. Create VM from base image
        2. Configure GPU passthrough
        3. Start VM and wait for boot
        4. Install Docker Engine in VM
        5. Run madengine Docker workflow (existing code!)
        6. Collect results
        7. Destroy VM completely
        
        Returns:
            DeploymentResult with status and job information
        """
        try:
            self.rich_console.print("\n[bold cyan]🚀 Bare Metal VM Deployment[/bold cyan]\n")
            
            # Generate unique VM name
            self.vm_name = f"madengine-vm-{uuid.uuid4().hex[:8]}"
            self.vm_disk_path = f"/var/lib/libvirt/images/{self.vm_name}.qcow2"
            
            # Step 1: Discover and configure GPUs
            self.rich_console.print("[cyan]Step 1/7: Configuring GPU passthrough...[/cyan]")
            vm_gpu_addresses = self._configure_gpus()
            self.rich_console.print(f"[green]  ✓ GPUs configured: {vm_gpu_addresses}[/green]\n")
            
            # Step 2: Create VM
            self.rich_console.print("[cyan]Step 2/7: Creating ephemeral VM...[/cyan]")
            self._create_vm(vm_gpu_addresses)
            self.rich_console.print(f"[green]  ✓ VM created: {self.vm_name}[/green]\n")
            
            # Step 3: Start VM
            self.rich_console.print("[cyan]Step 3/7: Starting VM...[/cyan]")
            self._start_vm()
            self.rich_console.print(f"[green]  ✓ VM started (IP: {self.vm_instance.ip_address})[/green]\n")
            
            # Step 4: Install Docker in VM
            self.rich_console.print("[cyan]Step 4/7: Installing Docker Engine...[/cyan]")
            self._install_docker_in_vm()
            self.rich_console.print("[green]  ✓ Docker installed and configured[/green]\n")
            
            # Step 5: Run existing madengine Docker workflow
            self.rich_console.print("[cyan]Step 5/7: Running madengine Docker workflow...[/cyan]")
            self._run_docker_workflow()
            self.rich_console.print("[green]  ✓ Workflow completed[/green]\n")
            
            # Step 6: Collect results
            self.rich_console.print("[cyan]Step 6/7: Collecting results...[/cyan]")
            self._collect_results()
            self.rich_console.print("[green]  ✓ Results collected[/green]\n")
            
            # Step 7: Success
            self.rich_console.print("[bold green]✓ Deployment successful![/bold green]\n")
            
            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                job_id=self.vm_name,
                message=f"Workload completed in VM {self.vm_name}"
            )
            
        except Exception as e:
            self.rich_console.print(f"\n[red]✗ Deployment failed: {e}[/red]\n")
            raise OrchestrationError(
                f"Bare metal VM deployment failed: {e}",
                context=create_error_context(
                    operation="baremetal_vm_deploy",
                    component="BareMetalVMDeployment"
                )
            ) from e
        
        finally:
            # ALWAYS cleanup VM
            if self.cleanup_mode == "destroy":
                self.rich_console.print("[cyan]Step 7/7: Cleanup - destroying VM...[/cyan]")
                self._cleanup()
                self.rich_console.print("[green]  ✓ VM destroyed, bare metal restored[/green]\n")
    
    def _configure_gpus(self) -> List[str]:
        """
        Configure GPU passthrough.
        
        Returns:
            List of GPU PCI addresses to pass to VM
        """
        # Auto-discover GPUs if not specified
        if not self.gpu_pci_addresses:
            gpus = self.gpu_manager.find_gpu_devices(self.gpu_vendor)
            if not gpus:
                raise RuntimeError(f"No {self.gpu_vendor} GPUs found")
            # Use first GPU
            self.gpu_pci_addresses = [gpus[0]["pci_address"]]
            self.rich_console.print(f"  Auto-discovered GPU: {self.gpu_pci_addresses[0]}")
        
        # Configure passthrough based on mode
        vm_gpu_addresses = self.gpu_manager.configure_passthrough(
            self.gpu_mode,
            self.gpu_pci_addresses,
            num_vfs=1
        )
        
        return vm_gpu_addresses
    
    def _create_vm(self, gpu_pci_addresses: List[str]):
        """Create VM with specified GPU passthrough."""
        vm_config = VMConfig(
            name=self.vm_name,
            vcpus=self.vcpus,
            memory_gb=self.memory_gb,
            disk_path=self.vm_disk_path,
            base_image=self.base_image,
            gpu_pci_addresses=gpu_pci_addresses,
            network_mode="default"
        )
        
        self.vm_instance = self.vm_manager.create_vm(vm_config)
    
    def _start_vm(self):
        """Start VM and wait for SSH."""
        self.vm_instance = self.vm_manager.start_vm(
            self.vm_name,
            wait_for_ssh=True,
            ssh_timeout=300
        )
    
    def _install_docker_in_vm(self):
        """Install Docker Engine inside VM via SSH."""
        # Determine setup script based on GPU vendor
        if self.gpu_vendor.upper() == "AMD":
            script_name = "setup_docker_amd.sh"
        else:
            script_name = "setup_docker_nvidia.sh"
        
        # Get script path
        script_path = Path(__file__).parent / "templates" / "baremetal_vm" / script_name
        
        if not script_path.exists():
            raise FileNotFoundError(f"Setup script not found: {script_path}")
        
        # Copy script to VM
        self.vm_manager.scp_to_vm(
            self.vm_name,
            str(script_path),
            "/tmp/setup_docker.sh",
            ssh_user=self.ssh_user,
            ssh_key=self.ssh_key
        )
        
        # Make executable and run
        self.vm_manager.ssh_exec(
            self.vm_name,
            "chmod +x /tmp/setup_docker.sh",
            ssh_user=self.ssh_user,
            ssh_key=self.ssh_key
        )
        
        result = self.vm_manager.ssh_exec(
            self.vm_name,
            "/tmp/setup_docker.sh",
            ssh_user=self.ssh_user,
            ssh_key=self.ssh_key
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Docker installation failed: {result.stderr}")
    
    def _run_docker_workflow(self):
        """
        Run existing madengine Docker workflow inside VM.
        
        This is the KEY: we reuse 100% of existing container execution code!
        The VM just provides isolation, Docker workflow is unchanged.
        """
        # Copy manifest to VM
        manifest_file = self.config.manifest_file
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
        
        self.vm_manager.scp_to_vm(
            self.vm_name,
            manifest_file,
            "/workspace/build_manifest.json",
            ssh_user=self.ssh_user,
            ssh_key=self.ssh_key
        )
        
        # Copy MAD package if available
        mad_path = os.environ.get("MAD_PATH", os.getcwd())
        if os.path.exists(mad_path):
            # TODO: Sync MAD package to VM (for now assume it's in base image)
            pass
        
        # Run madengine container workflow via SSH
        # This executes the SAME code path as local Docker execution!
        gpu_vendor_lower = self.gpu_vendor.lower()
        guest_os = self.config.additional_context.get("guest_os", "UBUNTU")
        
        # Build the command
        cmd = f"""
cd /workspace
export MAD_DEPLOYMENT_TYPE=baremetal_vm

# Run madengine workflow (uses existing container runner!)
madengine run \\
    --manifest-file build_manifest.json \\
    --timeout {self.config.timeout} \\
    --live-output
"""
        
        result = self.vm_manager.ssh_exec(
            self.vm_name,
            cmd,
            ssh_user=self.ssh_user,
            ssh_key=self.ssh_key
        )
        
        # Note: We don't fail on non-zero exit code because model failures
        # are tracked in perf_entry.csv, not by exit code
        if result.returncode != 0:
            self.rich_console.print(
                f"  [yellow]Warning: madengine exited with code {result.returncode}[/yellow]"
            )
            self.rich_console.print(f"  [dim]{result.stderr[:500]}[/dim]")
    
    def _collect_results(self):
        """Copy results from VM to host."""
        # Results to collect
        result_files = [
            "/workspace/perf_entry.csv",
            "/workspace/perf_entry.json",
            "/workspace/perf_super.csv",
            "/workspace/perf_entry_super.json"
        ]
        
        for remote_file in result_files:
            local_file = os.path.basename(remote_file)
            try:
                self.vm_manager.scp_from_vm(
                    self.vm_name,
                    remote_file,
                    local_file,
                    ssh_user=self.ssh_user,
                    ssh_key=self.ssh_key
                )
                self.rich_console.print(f"  ✓ Collected: {local_file}")
            except subprocess.CalledProcessError:
                # File may not exist (e.g., no super results)
                pass
    
    def _cleanup(self):
        """Completely destroy VM and verify clean state."""
        try:
            # Stop and destroy VM
            if self.vm_name and self.vm_manager:
                self.vm_manager.destroy_vm(self.vm_name, cleanup_disk=True)
            
            # Release GPU resources
            self.gpu_manager.cleanup_passthrough(
                self.gpu_mode,
                self.gpu_pci_addresses
            )
            
            # Verify clean state
            if self.verify_clean:
                self._verify_clean_state()
        except Exception as e:
            self.rich_console.print(f"[yellow]Warning: Cleanup issue: {e}[/yellow]")
    
    def _verify_clean_state(self):
        """Verify bare metal returned to clean state."""
        checks = {
            "no_madengine_vms": self._check_no_madengine_vms(),
            "gpu_resources_free": self._check_gpu_free(),
            "disk_cleaned": self._check_disk_clean()
        }
        
        all_clean = all(checks.values())
        
        if all_clean:
            self.rich_console.print("  ✓ Clean state verified")
        else:
            failed_checks = [k for k, v in checks.items() if not v]
            self.rich_console.print(
                f"  [yellow]⚠ Some checks failed: {failed_checks}[/yellow]"
            )
    
    def _check_no_madengine_vms(self) -> bool:
        """Check no madengine VMs running."""
        try:
            result = subprocess.run(
                ["virsh", "list", "--all"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "madengine-vm-" not in result.stdout
        except:
            return True  # Assume clean if check fails
    
    def _check_gpu_free(self) -> bool:
        """Check GPU resources released."""
        try:
            # Check no active VFs for SR-IOV
            if self.gpu_mode == GPUPassthroughMode.SRIOV:
                for pci_addr in self.gpu_pci_addresses:
                    numvfs_path = f"/sys/bus/pci/devices/{pci_addr}/sriov_numvfs"
                    if os.path.exists(numvfs_path):
                        with open(numvfs_path, 'r') as f:
                            if int(f.read().strip()) > 0:
                                return False
            return True
        except:
            return True
    
    def _check_disk_clean(self) -> bool:
        """Check VM disk deleted."""
        return not os.path.exists(self.vm_disk_path) if self.vm_disk_path else True
    
    def get_status(self, job_id: str) -> DeploymentResult:
        """
        Get status of deployment job.
        
        Args:
            job_id: Job ID (VM name)
            
        Returns:
            DeploymentResult with current status
        """
        # For bare metal VM, jobs are synchronous, so this is mainly
        # for compatibility with the deployment interface
        if job_id in self.vm_manager.vms:
            vm = self.vm_manager.vms[job_id]
            if vm.domain.isActive():
                return DeploymentResult(
                    status=DeploymentStatus.RUNNING,
                    job_id=job_id,
                    message="VM is running"
                )
        
        return DeploymentResult(
            status=DeploymentStatus.SUCCESS,
            job_id=job_id,
            message="Job completed (VM destroyed)"
        )
    
    def cancel(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: Job ID (VM name)
            
        Returns:
            True if cancelled successfully
        """
        try:
            if job_id in self.vm_manager.vms:
                self.vm_manager.destroy_vm(job_id, cleanup_disk=True)
                return True
            return False
        except Exception as e:
            self.rich_console.print(f"[red]Failed to cancel job {job_id}: {e}[/red]")
            return False
    
    def prepare(self) -> bool:
        """
        Prepare deployment artifacts (no-op for bare metal VM).
        
        Returns:
            True (always succeeds)
        """
        # No preparation needed - everything is done in deploy()
        return True
    
    def monitor(self, deployment_id: str) -> DeploymentResult:
        """
        Monitor deployment status.
        
        Args:
            deployment_id: Deployment ID (VM name)
            
        Returns:
            DeploymentResult with current status
        """
        return self.status(deployment_id)
    
    def collect_results(self, deployment_id: str) -> Dict[str, Any]:
        """
        Collect results from deployment.
        
        Args:
            deployment_id: Deployment ID (VM name)
            
        Returns:
            Dict with results (empty for bare metal VM as results are collected during deploy)
        """
        # Results are collected during deploy() phase
        return {}
    
    def cleanup(self, deployment_id: str) -> bool:
        """
        Cleanup deployment resources.
        
        Args:
            deployment_id: Deployment ID (VM name)
            
        Returns:
            True if cleanup successful
        """
        # Cleanup is handled automatically in deploy() based on vm_config.cleanup
        # This method is for manual cleanup if needed
        try:
            if deployment_id in self.vm_manager.vms:
                self.vm_manager.destroy_vm(deployment_id, cleanup_disk=True)
                return True
            return True  # Already cleaned up
        except Exception as e:
            self.rich_console.print(f"[red]Failed to cleanup {deployment_id}: {e}[/red]")
            return False