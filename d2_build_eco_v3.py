#!/usr/bin/env python3
# [OBSOLETE 25/08/2026] Périmètre 27B abandonné (prod = Qwen3.6-35B-A3B-D2-MOE).
# Raisons : cible Qwen3.8-27B inexistante ; --override-tensor inexistant dans ce fork.
# Conservé pour historique — NE PAS EXÉCUTER.
# -*- coding: utf-8 -*-
"""
D2 BUILD ECO V3 - Generate COMPLETE tensor-type file from KLD scanner
======================================================================
Uses gguf library for metadata, KLD report for overrides.

Usage:
  python d2_build_eco_v3.py
  python d2_build_eco_v3.py --dry-run
"""

import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

GGML_ID_TO_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "IQ4_NL",
}

# KLD-based mapping: tensor_name -> new_quant_type
KLD_MAP = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=os.path.join(HERE, "d2_kld_layer_report.json"))
    ap.add_argument("--gguf", default=os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO.gguf"))
    ap.add_argument("--output", default=os.path.join(HERE, "d2_tensor_types_eco_v3.txt"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  D2 BUILD ECO V3")
    print("=" * 70)

    # Load KLD report
    if os.path.exists(args.report):
        with open(args.report) as f:
            data = json.load(f)
        for t in data.get("tensors", []):
            snr = t["snr_db"]
            name = t["gguf_name"]
            if snr < 8.0:
                KLD_MAP[name] = "Q4_K"
            elif snr < 18.0:
                KLD_MAP[name] = "Q3_K"
            else:
                KLD_MAP[name] = "Q2_K"
        print(f"  KLD overrides: {len(KLD_MAP)} tensors")
    else:
        print(f"  [!] No KLD report found")

    # Load GGUF metadata via gguf library
    from gguf import GGUFReader
    print(f"  Reading GGUF: {args.gguf}")
    r = GGUFReader(args.gguf)

    lines = []
    stats = defaultdict(int)
    n_changed = 0

    for t in r.tensors:
        name = t.name
        tid = t.tensor_type
        current = GGML_ID_TO_NAME.get(tid, "F32")

        # Skip F32
        if tid == 0:
            continue

        # Use KLD override
        if name in KLD_MAP:
            new_type = KLD_MAP[name]
        else:
            # Keep current
            new_type = current

        if new_type != current:
            lines.append(f"{name} {new_type}")
            stats[f"{current}->{new_type}"] += 1
            n_changed += 1
        else:
            stats[f"keep_{current}"] += 1

    # Sort
    def sk(line):
        n = line.split()[0]
        if n.startswith("blk."):
            return (int(n.split(".")[1]), n)
        return (999, n)
    lines.sort(key=sk)

    if not args.dry_run:
        with open(args.output, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n  [+] {args.output} ({len(lines)} overrides, {n_changed} changed)")
    else:
        print(f"\n  DRY RUN: {len(lines)} overrides, {n_changed} changed")

    print("\n  CHANGES:")
    for k in sorted(stats.keys()):
        print(f"    {k:20s}: {stats[k]}")

    print(f"\n  Build command:")
    print(f"    _backup_cuda12_prebuilt/llama-quantize.exe \\")
    print(f"      --allow-requantize \\")
    print(f"      models/Qwen3.8-27B-D2-ECO.gguf \\")
    print(f"      models/Qwen3.8-27B-D2-ECO-v3.gguf \\")
    print(f"      Q4_K \\")
    print(f"      --tensor-type-file {os.path.basename(args.output)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
