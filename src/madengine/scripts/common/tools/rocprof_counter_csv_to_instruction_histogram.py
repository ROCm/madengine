#!/usr/bin/env python3
"""
Generate instruction_histogram.json from rocprofv3 counter_collection or domain_stats CSV.

Reads rocprofv3_output_counter_collection.csv or rocprofv3_output_domain_stats.csv
from the given directory, maps PMC columns (SQ_INSTS_VALU, SQ_INSTS_SALU, etc.) to
instruction classes (VALU, SALU, VMEM, WAITCNT), and writes instruction_histogram.json
so MAD-agent can consume real instruction mix data from madengine runs.

Usage:
  python3 rocprof_counter_csv_to_instruction_histogram.py <output_dir>
"""

import csv
import json
import sys
from pathlib import Path

# PMC column name patterns -> instruction class (aligned with MAD-agent rocprof_parser)
PMC_TO_CLASS = [
    (("sq_insts_valu", "sq_active_inst_valu"), "VALU"),
    (("sq_insts_salu",), "SALU"),
    (("sq_insts_vmem", "sq_insts_vmem_rd", "sq_insts_vmem_wr"), "VMEM"),
    (("sq_insts_smem",), "SMEM"),
    (("sq_wait_inst", "sq_wait_inst_any"), "WAITCNT"),
    (("sq_insts_branch",), "BRANCH"),
]


def _normalize_key(key: str) -> str:
    return (key or "").strip().lower().replace(" ", "").replace("-", "_")


def _safe_int(val) -> int:
    if val is None:
        return 0
    s = (val or "").strip().strip('"').strip("'")
    if not s:
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def parse_csv(path: Path) -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                k = _normalize_key(key)
                if not k:
                    continue
                count = _safe_int(value)
                for patterns, cls in PMC_TO_CLASS:
                    if any(k.startswith(p) or p in k for p in patterns):
                        out[cls] = out.get(cls, 0) + count
                        break
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: rocprof_counter_csv_to_instruction_histogram.py <output_dir>", file=sys.stderr)
        return 1
    out_dir = Path(sys.argv[1])
    if not out_dir.is_dir():
        return 0  # no dir, skip silently
    aggregated = {}
    for name in ("rocprofv3_output_counter_collection.csv", "rocprofv3_output_domain_stats.csv"):
        path = out_dir / name
        if not path.exists():
            continue
        parsed = parse_csv(path)
        for cls, count in parsed.items():
            aggregated[cls] = aggregated.get(cls, 0) + count
    if not aggregated:
        return 0
    dest = out_dir / "instruction_histogram.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=0)
    print(f"Wrote {dest} (classes: {list(aggregated.keys())})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
