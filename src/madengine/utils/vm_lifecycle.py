#!/usr/bin/env python3
"""
VM Lifecycle Management for KVM/libvirt.

Handles creation, startup, shutdown, and cleanup of ephemeral VMs
for bare metal execution with madengine.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import time
import socket
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    import libvirt
    LIBVIRT_AVAILABLE = True
except ImportError:
    LIBVIRT_AVAILABLE = False


@dataclass
class VMConfig:
    """Configuration for a VM instance."""
    name: str
    vcpus: int
    memory_gb: int
    disk_path: str
    base_image: str
    gpu_pci_addresses: List[str]
    network_mode: str = "default"
    
    @property
    def memory_kib(self) -> int:
        """Convert memory from GB to KiB for libvirt."""
        return self.memory_gb * 1024 * 1024


@dataclass
class VMInstance:
    """Represents a running VM instance."""
    name: str
    domain: Any  # libvirt domain object
    ip_address: Optional[str] = None
    disk_path: Optional[str] = None


class VMLifecycleManager:
    """
    Manages VM lifecycle operations using libvirt.
    
    Supports:
    - Creating VMs from base images
    - GPU passthrough (SR-IOV, VFIO)
    - Network configuration
    - SSH access management
    - Complete cleanup and verification
    """
    
    def __init__(self, libvirt_uri: str = "qemu:///system"):
        """
        Initialize VM lifecycle manager.
        
        Args:
            libvirt_uri: libvirt connection URI
        """
        if not LIBVIRT_AVAILABLE:
            raise ImportError(
                "libvirt-python not installed. Install with:\n"
                "pip install libvirt-python"
            )
        
        self.libvirt_uri = libvirt_uri
        self.conn: Optional[Any] = None
        self.vms: Dict[str, VMInstance] = {}
    
    def connect(self):
        """Connect to libvirt hypervisor."""
        if not self.conn:
            self.conn = libvirt.open(self.libvirt_uri)
            if not self.conn:
                raise RuntimeError(f"Failed to connect to libvirt: {self.libvirt_uri}")
    
    def disconnect(self):
        """Disconnect from libvirt hypervisor."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def create_vm(self, config: VMConfig) -> VMInstance:
        """
        Create and define a new VM from base image.
        
        Args:
            config: VM configuration
            
        Returns:
            VMInstance object
        """
        self.connect()
        
        # Create ephemeral disk from base image
        self._create_ephemeral_disk(config.base_image, config.disk_path)
        
        # Generate VM XML definition
        vm_xml = self._generate_vm_xml(config)
        
        # Define VM in libvirt
        domain = self.conn.defineXML(vm_xml)
        
        # Store VM instance
        vm_instance = VMInstance(
            name=config.name,
            domain=domain,
            disk_path=config.disk_path
        )
        self.vms[config.name] = vm_instance
        
        return vm_instance
    
    def start_vm(self, vm_name: str, wait_for_ssh: bool = True, 
                 ssh_timeout: int = 300) -> VMInstance:
        """
        Start a VM and optionally wait for SSH.
        
        Args:
            vm_name: Name of VM to start
            wait_for_ssh: Whether to wait for SSH availability
            ssh_timeout: Timeout for SSH wait in seconds
            
        Returns:
            VMInstance with IP address populated
        """
        vm = self.vms.get(vm_name)
        if not vm:
            raise ValueError(f"VM not found: {vm_name}")
        
        # Start the VM
        vm.domain.create()
        
        # Wait for boot
        time.sleep(10)
        
        # Get IP address
        vm.ip_address = self._get_vm_ip(vm.domain)
        
        # Wait for SSH if requested
        if wait_for_ssh:
            self._wait_for_ssh(vm.ip_address, timeout=ssh_timeout)
        
        return vm
    
    def stop_vm(self, vm_name: str, force: bool = False):
        """
        Stop a running VM.
        
        Args:
            vm_name: Name of VM to stop
            force: If True, force destroy; if False, graceful shutdown
        """
        vm = self.vms.get(vm_name)
        if not vm:
            raise ValueError(f"VM not found: {vm_name}")
        
        if vm.domain.isActive():
            if force:
                vm.domain.destroy()  # Force stop
            else:
                vm.domain.shutdown()  # Graceful shutdown
                # Wait for shutdown (up to 60s)
                for _ in range(60):
                    if not vm.domain.isActive():
                        break
                    time.sleep(1)
                # Force if still running
                if vm.domain.isActive():
                    vm.domain.destroy()
    
    def destroy_vm(self, vm_name: str, cleanup_disk: bool = True):
        """
        Completely destroy a VM and clean up resources.
        
        Args:
            vm_name: Name of VM to destroy
            cleanup_disk: Whether to delete the VM disk
        """
        vm = self.vms.get(vm_name)
        if not vm:
            return  # Already destroyed or never created
        
        # Stop VM if running
        if vm.domain.isActive():
            vm.domain.destroy()
        
        # Undefine (delete) VM
        try:
            vm.domain.undefine()
        except libvirt.libvirtError:
            pass  # Already undefined
        
        # Delete disk
        if cleanup_disk and vm.disk_path and os.path.exists(vm.disk_path):
            os.remove(vm.disk_path)
        
        # Remove from tracking
        del self.vms[vm_name]
    
    def _create_ephemeral_disk(self, base_image: str, disk_path: str):
        """
        Create ephemeral disk from base image using qemu-img.
        
        Creates a copy-on-write disk backed by the base image.
        """
        if not os.path.exists(base_image):
            raise FileNotFoundError(f"Base image not found: {base_image}")
        
        # Create backing image (copy-on-write)
        subprocess.run([
            "qemu-img", "create",
            "-f", "qcow2",
            "-F", "qcow2",
            "-b", base_image,
            disk_path
        ], check=True, capture_output=True)
    
    def _generate_vm_xml(self, config: VMConfig) -> str:
        """
        Generate libvirt XML definition for VM.
        
        Args:
            config: VM configuration
            
        Returns:
            XML string for libvirt
        """
        # Generate GPU passthrough devices
        gpu_devices = ""
        for gpu_pci in config.gpu_pci_addresses:
            parts = gpu_pci.replace("0000:", "").split(":")
            if len(parts) == 2:
                bus = parts[0]
                slot_func = parts[1].split(".")
                slot = slot_func[0]
                func = slot_func[1] if len(slot_func) > 1 else "0"
            else:
                continue
            
            gpu_devices += f"""
    <hostdev mode='subsystem' type='pci' managed='yes'>
      <source>
        <address domain='0x0000' bus='0x{bus}' slot='0x{slot}' function='0x{func}'/>
      </source>
    </hostdev>"""
        
        xml = f"""<domain type='kvm'>
  <name>{config.name}</name>
  <memory unit='KiB'>{config.memory_kib}</memory>
  <currentMemory unit='KiB'>{config.memory_kib}</currentMemory>
  <vcpu placement='static'>{config.vcpus}</vcpu>
  <os>
    <type arch='x86_64' machine='q35'>hvm</type>
    <boot dev='hd'/>
  </os>
  <features>
    <acpi/>
    <apic/>
  </features>
  <cpu mode='host-passthrough' check='none'>
    <topology sockets='1' cores='{config.vcpus}' threads='1'/>
  </cpu>
  <clock offset='utc'>
    <timer name='rtc' tickpolicy='catchup'/>
    <timer name='pit' tickpolicy='delay'/>
    <timer name='hpet' present='no'/>
  </clock>
  <on_poweroff>destroy</on_poweroff>
  <on_reboot>restart</on_reboot>
  <on_crash>destroy</on_crash>
  <devices>
    <emulator>/usr/bin/qemu-system-x86_64</emulator>
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2' cache='none' io='native'/>
      <source file='{config.disk_path}'/>
      <target dev='vda' bus='virtio'/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x04' function='0x0'/>
    </disk>
    <interface type='network'>
      <source network='{config.network_mode}'/>
      <model type='virtio'/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x03' function='0x0'/>
    </interface>
    <console type='pty'>
      <target type='serial' port='0'/>
    </console>
    <channel type='unix'>
      <target type='virtio' name='org.qemu.guest_agent.0'/>
      <address type='virtio-serial' controller='0' bus='0' port='1'/>
    </channel>
    <input type='tablet' bus='usb'>
      <address type='usb' bus='0' port='1'/>
    </input>
    <input type='mouse' bus='ps2'/>
    <input type='keyboard' bus='ps2'/>
    <graphics type='vnc' port='-1' autoport='yes' listen='127.0.0.1'>
      <listen type='address' address='127.0.0.1'/>
    </graphics>
    <video>
      <model type='vga' vram='16384' heads='1' primary='yes'/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x02' function='0x0'/>
    </video>
    <memballoon model='virtio'>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x05' function='0x0'/>
    </memballoon>{gpu_devices}
  </devices>
</domain>"""
        
        return xml
    
    def _get_vm_ip(self, domain) -> str:
        """
        Get VM IP address from libvirt.
        
        Args:
            domain: libvirt domain object
            
        Returns:
            IP address string
        """
        # Try multiple methods to get IP
        
        # Method 1: Query domain interfaces
        try:
            interfaces = domain.interfaceAddresses(
                libvirt.VIR_DOMAIN_INTERFACE_ADDRESSES_SRC_LEASE
            )
            for iface_name, iface in interfaces.items():
                if iface['addrs']:
                    for addr in iface['addrs']:
                        if addr['type'] == libvirt.VIR_IP_ADDR_TYPE_IPV4:
                            return addr['addr']
        except:
            pass
        
        # Method 2: Query DHCP leases from network
        try:
            net = self.conn.networkLookupByName('default')
            leases = net.DHCPLeases()
            for lease in leases:
                if lease['hostname'] == domain.name():
                    return lease['ipaddr']
        except:
            pass
        
        # Method 3: Parse from domain XML
        try:
            import xml.etree.ElementTree as ET
            xml_desc = domain.XMLDesc()
            root = ET.fromstring(xml_desc)
            # This is a fallback - may not always work
        except:
            pass
        
        raise RuntimeError(f"Could not determine IP address for VM: {domain.name()}")
    
    def _wait_for_ssh(self, ip_address: str, port: int = 22, timeout: int = 300):
        """
        Wait for SSH to become available on VM.
        
        Args:
            ip_address: VM IP address
            port: SSH port (default 22)
            timeout: Timeout in seconds
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((ip_address, port))
                sock.close()
                
                if result == 0:
                    # SSH port is open, wait a bit more for SSH daemon
                    time.sleep(5)
                    return
            except socket.error:
                pass
            
            time.sleep(5)
        
        raise TimeoutError(
            f"SSH not available on {ip_address}:{port} after {timeout}s"
        )
    
    def ssh_exec(self, vm_name: str, command: str, 
                 ssh_user: str = "root", ssh_key: Optional[str] = None) -> subprocess.CompletedProcess:
        """
        Execute command in VM via SSH.
        
        Args:
            vm_name: Name of VM
            command: Command to execute
            ssh_user: SSH username
            ssh_key: Path to SSH private key (optional)
            
        Returns:
            subprocess.CompletedProcess result
        """
        vm = self.vms.get(vm_name)
        if not vm:
            raise ValueError(f"VM not found: {vm_name}")
        
        if not vm.ip_address:
            raise ValueError(f"VM {vm_name} has no IP address")
        
        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
        
        if ssh_key:
            ssh_cmd.extend(["-i", ssh_key])
        
        ssh_cmd.append(f"{ssh_user}@{vm.ip_address}")
        ssh_cmd.append(command)
        
        return subprocess.run(ssh_cmd, capture_output=True, text=True)
    
    def scp_to_vm(self, vm_name: str, local_path: str, remote_path: str,
                  ssh_user: str = "root", ssh_key: Optional[str] = None):
        """
        Copy file to VM via SCP.
        
        Args:
            vm_name: Name of VM
            local_path: Local file path
            remote_path: Remote file path
            ssh_user: SSH username
            ssh_key: Path to SSH private key (optional)
        """
        vm = self.vms.get(vm_name)
        if not vm:
            raise ValueError(f"VM not found: {vm_name}")
        
        scp_cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
        
        if ssh_key:
            scp_cmd.extend(["-i", ssh_key])
        
        scp_cmd.append(local_path)
        scp_cmd.append(f"{ssh_user}@{vm.ip_address}:{remote_path}")
        
        subprocess.run(scp_cmd, check=True)
    
    def scp_from_vm(self, vm_name: str, remote_path: str, local_path: str,
                    ssh_user: str = "root", ssh_key: Optional[str] = None):
        """
        Copy file from VM via SCP.
        
        Args:
            vm_name: Name of VM
            remote_path: Remote file path
            local_path: Local file path
            ssh_user: SSH username
            ssh_key: Path to SSH private key (optional)
        """
        vm = self.vms.get(vm_name)
        if not vm:
            raise ValueError(f"VM not found: {vm_name}")
        
        scp_cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
        
        if ssh_key:
            scp_cmd.extend(["-i", ssh_key])
        
        scp_cmd.append(f"{ssh_user}@{vm.ip_address}:{remote_path}")
        scp_cmd.append(local_path)
        
        subprocess.run(scp_cmd, check=True)
    
    def cleanup_all_vms(self):
        """Clean up all managed VMs."""
        vm_names = list(self.vms.keys())
        for vm_name in vm_names:
            try:
                self.destroy_vm(vm_name, cleanup_disk=True)
            except Exception as e:
                print(f"Warning: Failed to cleanup VM {vm_name}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup all VMs."""
        self.cleanup_all_vms()
        self.disconnect()
