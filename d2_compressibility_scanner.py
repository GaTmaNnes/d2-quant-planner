#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 COMPRESSIBILITY SCANNER — Rank tensors by SNR and generate optimal tensor-type file
=======================================================================================

Reads d2_kld_layer_report.json and produces:
  1. Ranked tensor list by sensitivity
  2. Optimal Q2/Q3/Q4 allocation
  3. tensor-type file for llama-quantize

Usage:
  python d2_compressibility_scanner.py
  python d2_compressibility_scanner.py --budget 12.5
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPORT = os.path.join(HERE, "d2_kld_layer_report.json")
# [CORRIGÉ 25/08/2026] l'ancien défaut pointait Qwen3.8-27B-D2-ECO.gguf
# (modèle inexistant) ; prod = 35B D2-MOE.
DEFAULT_GGUF = os.path.join(HERE, "models", "Qwen3.6-35B-A3B-D2-MOE.gguf")


def main():
    ap = argparse.ArgumentParser(description="D2 Compressibility Scanner")
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--budget", type=float, default=12.5, help="Target size in GiB")
    ap.add_argument("--output-types", default=os.path.join(HERE, "d2_tensor_types_eco_v3.txt"))
    args = ap.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  D2 COMPRESSIBILITY SCANNER")
    print("=" * 70)

    # Load KLD report
    if not os.path.exists(args.report):
        print(f"  [!] Report not found: {args.report}")
        print(f"  Run d2_kld_layer_profiler.py first")
        return

    with open(args.report) as f:
        data = json.load(f)

    tensors = data.get("tensors", [])
    print(f"  Loaded {len(tensors)} tensors from {args.report}")

    # Rank by SNR (lowest = most sensitive)
    ranked = sorted(tensors, key=lambda x: x["snr_db"])

    # Classification
    SENSITIVE_THRESHOLD = 8.0   # Q2 is bad here
    MODERATE_THRESHOLD = 13.0   # Q3 is OK
    GOOD_THRESHOLD = 18.0       # Q4 is fine

    print()
    print("  ╔════════════════════════════════════════════════════════════════╗")
    print("  ║          TENSOR SENSITIVITY CLASSIFICATION                    ║")
    print("  ╠════════════════════════════════════════════════════════════════╣")

    critical = []   # SNR < 8 → Q4 needed
    sensitive = []  # SNR 8-13 → Q3 needed
    moderate = []   # SNR 13-18 → Q3 OK, Q2 risky
    robust = []     # SNR > 18 → Q2 fine

    for t in ranked:
        snr = t["snr_db"]
        if snr < SENSITIVE_THRESHOLD:
            critical.append(t)
        elif snr < MODERATE_THRESHOLD:
            sensitive.append(t)
        elif snr < GOOD_THRESHOLD:
            moderate.append(t)
        else:
            robust.append(t)

    print(f"  ║  CRITICAL (SNR < {SENSITIVE_THRESHOLD:.0f} dB)  → Q4 mandatory     : {len(critical):4d} tensors ║")
    print(f"  ║  SENSITIVE (SNR {SENSITIVE_THRESHOLD:.0f}-{MODERATE_THRESHOLD:.0f} dB) → Q3 mandatory     : {len(sensitive):4d} tensors ║")
    print(f"  ║  MODERATE (SNR {MODERATE_THRESHOLD:.0f}-{GOOD_THRESHOLD:.0f} dB) → Q3 preferred     : {len(moderate):4d} tensors ║")
    print(f"  ║  ROBUST   (SNR > {GOOD_THRESHOLD:.0f} dB)  → Q2 OK            : {len(robust):4d} tensors ║")
    print(f"  ║  TOTAL                       {len(ranked):4d} tensors                       ║")
    print("  ╚════════════════════════════════════════════════════════════════╝")

    # Budget allocation
    # Current: Q2_K everywhere → 11.64 GiB
    # Strategy: upgrade critical/sensitive tensors to Q3/Q4
    # Each Q2→Q3 adds ~0.7 bits/weight
    # Each Q2→Q4 adds ~1.8 bits/weight

    print()
    print("  BUDGET ALLOCATION:")

    # Estimate sizes
    Q2_BPW = 2.7
    Q3_BPW = 3.4
    Q4_BPW = 4.5

    total_weights = 0
    for t in tensors:
        # Estimate weight count from shape
        shape = t.get("shape", [1])
        n = 1
        for s in shape:
            n *= s
        total_weights += n

    current_size_gib = 11.64
    current_bpw = current_size_gib * 8 * 1024**3 / total_weights

    # Calculate savings from upgrades
    upgrade_to_q3 = len(critical) + len(sensitive)  # critical + sensitive → Q3
    upgrade_to_q4 = len(critical)  # critical → Q4

    # Recount: critical → Q4, sensitive → Q3, moderate → Q3, robust → Q2
    q4_count = len(critical)
    q3_count = len(sensitive) + len(moderate)
    q2_count = len(robust)

    # Estimate new BPW
    q4_weights = sum(1 for t in critical for _ in [1])  # simplified
    q3_weights = sum(1 for t in sensitive for _ in [1]) + sum(1 for t in moderate for _ in [1])
    q2_weights = sum(1 for t in robust for _ in [1])

    # Better estimate: use actual tensor sizes from report
    total_q4_bytes = sum(t.get("hf_mb", 0) * 1e6 * Q4_BPW / 32 for t in critical)
    total_q3_bytes = sum(t.get("hf_mb", 0) * 1e6 * Q3_BPW / 32 for t in sensitive + moderate)
    total_q2_bytes = sum(t.get("hf_mb", 0) * 1e6 * Q2_BPW / 32 for t in robust)
    new_total_bytes = total_q4_bytes + total_q3_bytes + total_q2_bytes
    new_size_gib = new_total_bytes / (1024**3)

    print(f"    Q4: {q4_count:4d} tensors (critical)  → ~{total_q4_bytes/1e6:.0f} MB")
    print(f"    Q3: {q3_count:4d} tensors (sensitive+moderate) → ~{total_q3_bytes/1e6:.0f} MB")
    print(f"    Q2: {q2_count:4d} tensors (robust)     → ~{total_q2_bytes/1e6:.0f} MB")
    print(f"    Estimated new size: {new_size_gib:.2f} GiB (current: {current_size_gib:.2f} GiB)")

    # Show top critical tensors
    print()
    print("  TOP 20 CRITICAL TENSORS (need Q4):")
    print(f"  {'Layer':6s} {'Tensor':40s} {'SNR':8s} {'Current':8s} {'New':6s}")
    print("  " + "-" * 70)
    for t in critical[:20]:
        name = t["gguf_name"]
        if name.startswith("blk."):
            layer = int(name.split(".")[1])
            comp = name.split(".", 2)[2]
        else:
            layer = -1
            comp = name
        print(f"  L{layer:02d}    {comp:40s} {t['snr_db']:6.1f}dB  {t['quant_type']:6s}  → Q4_K")

    # Show top sensitive tensors
    print()
    print("  TOP 20 SENSITIVE TENSORS (need Q3):")
    print(f"  {'Layer':6s} {'Tensor':40s} {'SNR':8s} {'Current':8s} {'New':6s}")
    print("  " + "-" * 70)
    for t in sensitive[:20]:
        name = t["gguf_name"]
        if name.startswith("blk."):
            layer = int(name.split(".")[1])
            comp = name.split(".", 2)[2]
        else:
            layer = -1
            comp = name
        print(f"  L{layer:02d}    {comp:40s} {t['snr_db']:6.1f}dB  {t['quant_type']:6s}  → Q3_K")

    # Generate tensor-type file
    print()
    print("  Generating tensor-type file...")

    # Build mapping: tensor_name -> recommended quant type
    rec_map = {}
    for t in critical:
        rec_map[t["gguf_name"]] = "Q4_K"
    for t in sensitive + moderate:
        rec_map[t["gguf_name"]] = "Q3_K"
    for t in robust:
        rec_map[t["gguf_name"]] = "Q2_K"

    # Write tensor-type file
    # [CORRIGÉ 25/08/2026] parse_tensor_type exige le format "nom=TYPE" ;
    # l'espace rendait chaque ligne invalide.
    lines = []
    for name, qtype in sorted(rec_map.items()):
        lines.append(f"{name}={qtype}")

    with open(args.output_types, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  [+] {args.output_types} ({len(lines)} entries)")

    # Summary
    print()
    print("=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)
    print(f"  D2-ECO v3 allocation:")
    print(f"    Q4_K: {q4_count} tensors (SNR < {SENSITIVE_THRESHOLD} dB)")
    print(f"    Q3_K: {q3_count} tensors (SNR {SENSITIVE_THRESHOLD}-{GOOD_THRESHOLD} dB)")
    print(f"    Q2_K: {q2_count} tensors (SNR > {GOOD_THRESHOLD} dB)")
    print(f"    Estimated size: ~{new_size_gib:.1f} GiB")
    print()
    print(f"  Build command:")
    # [CORRIGÉ 25/08/2026] exemple aligné sur le modèle de prod (35B D2-MOE)
    print(f"    llama-quantize.exe --allow-requantize \\")
    print(f"      models/Qwen3.6-35B-A3B-D2-MOE.gguf \\")
    print(f"      models/Qwen3.6-35B-A3B-D2-ECO-v3.gguf \\")
    print(f"      Q4_K \\")
    print(f"      --tensor-type-file d2_tensor_types_eco_v3.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
