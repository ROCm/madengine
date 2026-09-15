#!/usr/bin/env python3
"""
Native RDMA environment recommender for cluster deployments.

Produces a stable JSON contract plus optional KEY=VALUE env file that can be
consumed by SLURM/Kubernetes runtime scripts.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _run(cmd: List[str], timeout_sec: int = 8) -> str:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _pci_from_device(device_path: str) -> str:
    link = os.path.join(device_path, "device")
    if os.path.islink(link):
        try:
            return os.path.basename(os.readlink(link))
        except OSError:
            return "UNKNOWN_PCI"
    return "UNKNOWN_PCI"


def _netdev_for_pci(target_pci: str) -> str:
    for netdev in glob.glob("/sys/class/net/*"):
        link = os.path.join(netdev, "device")
        if not os.path.islink(link):
            continue
        try:
            if os.path.basename(os.readlink(link)) == target_pci:
                return os.path.basename(netdev)
        except OSError:
            continue
    return "NO_NETDEV"


def _vendor_from_pci(pci_addr: str) -> str:
    if not pci_addr or pci_addr == "UNKNOWN_PCI":
        return "UNKNOWN"
    out = _run(["lspci", "-s", pci_addr, "-nn"]).lower()
    if "pensando" in out:
        return "AINIC"
    if "broadcom" in out:
        return "BNXT"
    if "mellanox" in out or "nvidia" in out:
        return "MLNX"
    return "UNKNOWN"


def _ibv_devinfo(device: str) -> str:
    return _run(["ibv_devinfo", "-d", device, "-v"], timeout_sec=10)


def _firmware_from_ibv(output: str) -> str:
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("fw_ver:"):
            return line.split("fw_ver:", 1)[1].strip()
    return "UNKNOWN"


def _gid_from_ibv(output: str) -> Tuple[str, str]:
    for line in output.splitlines():
        if "::ffff:" not in line or "GID[" not in line:
            continue
        idx = re.search(r"GID\[\s*(\d+)\]", line)
        ip = re.search(r"(::ffff:[0-9.]+)", line)
        if idx and ip:
            return idx.group(1), ip.group(1)
    return "-", "N/A"


def _socket_ifname() -> str:
    route = _run(["bash", "-lc", "ip route show default | awk '{print $5}'"])
    if not route:
        return "eth0"
    ifnames = list(dict.fromkeys(route.splitlines()))
    return ifnames[0] if ifnames else "eth0"


def discover_rdma_devices() -> List[Dict[str, str]]:
    devices: List[Dict[str, str]] = []
    for path in sorted(glob.glob("/sys/class/infiniband/*")):
        rdma_name = os.path.basename(path)
        pci = _pci_from_device(path)
        netdev = _netdev_for_pci(pci)
        ibv = _ibv_devinfo(rdma_name)
        gid_index, gid_value = _gid_from_ibv(ibv)
        devices.append(
            {
                "rdma": rdma_name,
                "pci": pci,
                "netdev": netdev,
                "firmware": _firmware_from_ibv(ibv),
                "gid_index": gid_index,
                "gid_value": gid_value,
                "vendor": _vendor_from_pci(pci),
            }
        )
    return devices


def recommend_env_vars(devices: List[Dict[str, str]]) -> Dict[str, str]:
    if not devices:
        return {"NCCL_IB_DISABLE": "1"}

    env: Dict[str, str] = {
        "NCCL_IB_DISABLE": "0",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_SOCKET_IFNAME": _socket_ifname(),
        "GLOO_SOCKET_IFNAME": _socket_ifname(),
    }

    gid_indexes = [d["gid_index"] for d in devices if str(d["gid_index"]).isdigit()]
    if gid_indexes:
        env["NCCL_IB_GID_INDEX"] = str(max(int(g) for g in gid_indexes))

    prioritized = sorted(
        devices,
        key=lambda d: (
            d.get("firmware", ""),
            1 if str(d.get("gid_index", "")).isdigit() else 0,
        ),
        reverse=True,
    )
    hca = [d["rdma"] for d in prioritized if d.get("rdma")]
    if hca:
        env["NCCL_IB_HCA"] = ",".join(hca)
    return env


def build_recommendation() -> Dict[str, Any]:
    devices = discover_rdma_devices()
    status = "ok" if devices else "no_rdma"
    confidence = "high" if devices else "low"
    env = recommend_env_vars(devices)
    return {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "confidence": confidence,
        "errors": [],
        "devices": devices,
        "recommended_env": env,
    }


def write_artifact(payload: Dict[str, Any], output_path: Optional[str]) -> None:
    if not output_path:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_env_file(env: Dict[str, str], env_file: Optional[str]) -> None:
    if not env_file:
        return
    out = Path(env_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in env.items()]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cli() -> int:
    parser = argparse.ArgumentParser(description="Generate RDMA env recommendations.")
    parser.add_argument("--output", default="", help="JSON artifact path")
    parser.add_argument("--env-file", default="", help="KEY=VALUE env output path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when no valid RDMA setup is detected",
    )
    args = parser.parse_args()

    payload = build_recommendation()
    write_artifact(payload, args.output or None)
    write_env_file(payload.get("recommended_env", {}), args.env_file or None)

    if args.strict and payload.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
