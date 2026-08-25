#!/usr/bin/env python3
# [OBSOLETE 25/08/2026] Périmètre 27B abandonné (prod = Qwen3.6-35B-A3B-D2-MOE).
# Raisons : cible Qwen3.8-27B inexistante ; --override-tensor inexistant dans ce fork.
# Conservé pour historique — NE PAS EXÉCUTER.
# -*- coding: utf-8 -*-
"""
Generate D2-ECO-v2 tensor-type file from ECO baseline with targeted upgrades.

Strategy based on override benchmarks:
- ffn_down: Q4_K (biggest tensor, +2.8% faster, stable)
- ffn_gate: Q3_K (+1.2% faster)
- ffn_up: Q3_K (+1.1% faster)
- attn_v: Q3_K (neutral to +0.5%)
- attn_q: Q3_K (neutral)
- attn_o: Q3_K (neutral)
- ssm_out: Q4_K (keep current)
- attn_k: Q4_K (keep current - only on attention layers)
- attn_qkv: Q2_K (GDN layers - keep current)
- attn_gate: Q3_K (upgrade from Q2_K where present)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "d2_tensor_types_eco_v2.txt")

# ECO-v2 upgrade rules
# For each tensor type, specify the target quant
TENSOR_QUANTS = {
    "ffn_down.weight":  "q4_K",   # biggest tensor, fastest kernel
    "ffn_gate.weight":  "q3_K",   # +1.2% faster
    "ffn_up.weight":    "q3_K",   # +1.1% faster
    "attn_v.weight":    "q3_K",   # neutral to +0.5%
    "attn_q.weight":    "q3_K",   # neutral
    "attn_o.weight":    "q3_K",   # neutral
    "attn_output.weight": "q3_K", # same as attn_o
    "attn_gate.weight": "q3_K",   # upgrade from q2_K
    "ssm_out.weight":   "q4_K",   # keep current
    "attn_k.weight":    "q4_K",   # keep current (only attention layers)
    "attn_qkv.weight":  "q2_K",   # keep current (GDN layers)
}

# Number of layers in Qwen3.8-27B
N_LAYERS = 64

# Attention layers (every 4th layer starting from 0)
ATTENTION_LAYERS = set(range(0, N_LAYERS, 4))

# GDN layers (all others)
GDN_LAYERS = set(range(N_LAYERS)) - ATTENTION_LAYERS


def main():
    lines = []
    
    for layer in range(N_LAYERS):
        is_attn = layer in ATTENTION_LAYERS
        
        if is_attn:
            # Attention layer: has attn_q, attn_k, attn_v, attn_o, attn_gate
            tensors = [
                ("attn_q.weight", TENSOR_QUANTS["attn_q.weight"]),
                ("attn_k.weight", TENSOR_QUANTS["attn_k.weight"]),
                ("attn_v.weight", TENSOR_QUANTS["attn_v.weight"]),
                ("attn_o.weight", TENSOR_QUANTS["attn_o.weight"]),
                ("attn_gate.weight", TENSOR_QUANTS["attn_gate.weight"]),
            ]
        else:
            # GDN layer: has attn_qkv, ffn_down, ffn_gate, ffn_up, ssm_out
            tensors = [
                ("attn_qkv.weight", TENSOR_QUANTS["attn_qkv.weight"]),
                ("ffn_down.weight", TENSOR_QUANTS["ffn_down.weight"]),
                ("ffn_gate.weight", TENSOR_QUANTS["ffn_gate.weight"]),
                ("ffn_up.weight", TENSOR_QUANTS["ffn_up.weight"]),
                ("ssm_out.weight", TENSOR_QUANTS["ssm_out.weight"]),
            ]
        
        for tensor_name, quant in tensors:
            lines.append(f"blk.{layer}.{tensor_name}={quant}")
    
    # Write output
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    
    print(f"Generated {OUTPUT}")
    print(f"  Lines: {len(lines)}")
    print(f"  Layers: {N_LAYERS}")
    print(f"  Attention layers: {len(ATTENTION_LAYERS)}")
    print(f"  GDN layers: {len(GDN_LAYERS)}")
    
    # Summary
    print("\nECO-v2 quantization map:")
    for tensor, quant in sorted(TENSOR_QUANTS.items()):
        print(f"  {tensor:25s} -> {quant}")


if __name__ == "__main__":
    main()
