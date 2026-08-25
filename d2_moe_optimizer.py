#!/usr/bin/env python3
"""
D2 Axis #11: MoE Optimization Formulator
Solves: MAXIMIZE quality(PPL)
        SUBJECT TO: VRAM ≤ 8 GB
                    t/s ≥ threshold

Uses measured data from axes #1 and #8.
"""
import sys, io, json
from dataclasses import dataclass
from typing import Dict, List, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================================
# MEASURED DATA (from axis #1 experiments)
# ============================================================

@dataclass
class TensorGroup:
    """A group of tensors that can be quantized together."""
    name: str
    n_tensors: int
    size_iq4nl_mb: float  # size in IQ4_NL format
    size_q3k_mb: float    # size in Q3_K format
    size_q2k_mb: float    # size in Q2_K format
    size_iq3xxs_mb: float # size in IQ3_XXS format
    ppl_delta_q3k: float  # PPL change when quantizing this group to Q3_K
    ppl_delta_q2k: float  # PPL change when quantizing this group to Q2_K
    ppl_delta_iq3xxs: float
    speed_factor: float    # relative speed (1.0 = baseline)

# Measured from axis #1 experiments
# Total model: 19.77 GB IQ4_NL, PPL baseline = 7.579
TENSOR_GROUPS = [
    TensorGroup(
        name="expert_gate_up",
        n_tensors=80,
        size_iq4nl_mb=12610,   # 12.61 GB
        size_q3k_mb=9230,      # estimated from Q3_K bpw ratio
        size_q2k_mb=7600,      # estimated
        size_iq3xxs_mb=8220,   # from Unsloth Q3_K_M analysis
        ppl_delta_q3k=+0.190,  # MEASURED: variant C
        ppl_delta_q2k=+0.500,  # extrapolated
        ppl_delta_iq3xxs=+0.350,
        speed_factor=1.0,
    ),
    TensorGroup(
        name="expert_down",
        n_tensors=40,
        size_iq4nl_mb=6300,    # 6.30 GB
        size_q3k_mb=4610,      # Q3_K
        size_q2k_mb=3800,      # Q2_K
        size_iq3xxs_mb=4100,   # IQ3_XXS
        ppl_delta_q3k=+0.014,  # MEASURED: variant B
        ppl_delta_q2k=+0.300,  # MEASURED: variant E
        ppl_delta_iq3xxs=+0.102,# MEASURED: variant F
        speed_factor=1.0,
    ),
    TensorGroup(
        name="attn_qkv",
        n_tensors=40,
        size_iq4nl_mb=350,     # 0.35 GB (Q5_K in original)
        size_q3k_mb=250,       # Q3_K
        size_q2k_mb=200,       # Q2_K
        size_iq3xxs_mb=220,    # IQ3_XXS
        ppl_delta_q3k=+0.050,  # estimated (not measured)
        ppl_delta_q2k=+0.150,
        ppl_delta_iq3xxs=+0.080,
        speed_factor=1.0,
    ),
    TensorGroup(
        name="output_head",
        n_tensors=12,
        size_iq4nl_mb=420,     # 0.42 GB (Q6_K in original)
        size_q3k_mb=260,       # Q3_K
        size_q2k_mb=200,       # Q2_K
        size_iq3xxs_mb=220,
        ppl_delta_q3k=+0.020,  # estimated
        ppl_delta_q2k=+0.080,
        ppl_delta_iq3xxs=+0.040,
        speed_factor=1.0,
    ),
    TensorGroup(
        name="ssm_conv",
        n_tensors=210,
        size_iq4nl_mb=74,      # 0.074 GB
        size_q3k_mb=54,        # Q3_K
        size_q2k_mb=44,
        size_iq3xxs_mb=48,
        ppl_delta_q3k=+0.005,  # very small
        ppl_delta_q2k=+0.020,
        ppl_delta_iq3xxs=+0.010,
        speed_factor=1.0,
    ),
    TensorGroup(
        name="norms_embed",
        n_tensors=413,
        size_iq4nl_mb=89,      # 0.089 GB (F32 in original)
        size_q3k_mb=89,        # norms stay F32
        size_q2k_mb=89,
        size_iq3xxs_mb=89,
        ppl_delta_q3k=0.0,     # not quantized
        ppl_delta_q2k=0.0,
        ppl_delta_iq3xxs=0.0,
        speed_factor=1.0,
    ),
]

VRAM_BUDGET_MB = 8150
NGL = 15
N_LAYERS = 40
GPU_FRACTION = NGL / N_LAYERS  # 0.375

BASE_PPL = 7.579
# [CORRIGÉ 25/08/2026] BASE_TG32=28.12 était une mesure historique NON réplicable
# (~20-25 t/s mesuré le 24/08 soir, bridage GPU variable). Défaut conservateur : 22.
BASELINE_TG32_HISTORIQUE = 28.12  # valeur historique, NON réplicable
BASE_TG32 = 22  # défaut conservateur actuel
BASE_SIZE_MB = 19770  # 19.77 GB


def solve_greedy():
    """Greedy optimizer: for each group, pick the cheapest format that
    saves the most VRAM per PPL unit lost."""
    
    print("=" * 80)
    print("  D2-MoE OPTIMIZER — Greedy Allocation")
    print("=" * 80)
    
    print(f"\n  Constraints:")
    print(f"    VRAM budget: {VRAM_BUDGET_MB} MiB")
    print(f"    ngl: {NGL} ({GPU_FRACTION*100:.0f}% layers GPU)")
    print(f"    Baseline PPL: {BASE_PPL}")
    # [CORRIGÉ 25/08/2026] baseline tg32 historique non réplicable
    print(f"    Baseline tg32: {BASE_TG32} t/s (défaut conservateur ; "
          f"{BASELINE_TG32_HISTORIQUE} historique NON réplicable)")
    
    # Compute GPU weight for each group at each format
    print(f"\n  --- Per-Group Analysis ---")
    print(f"  {'Group':<18} {'IQ4 MB':>8} {'Q3K MB':>8} {'Q2K MB':>8} {'dPPL/Q3':>8} {'dPPL/Q2':>8} {'Efficiency':>10}")
    print(f"  {'-'*70}")
    
    for g in TENSOR_GROUPS:
        gpu_iq4 = g.size_iq4nl_mb * GPU_FRACTION
        gpu_q3 = g.size_q3k_mb * GPU_FRACTION
        gpu_q2 = g.size_q2k_mb * GPU_FRACTION
        
        # Efficiency = VRAM saved / PPL cost
        eff_q3 = (gpu_iq4 - gpu_q3) / g.ppl_delta_q3k if g.ppl_delta_q3k > 0 else float('inf')
        eff_q2 = (gpu_iq4 - gpu_q2) / g.ppl_delta_q2k if g.ppl_delta_q2k > 0 else float('inf')
        
        print(f"  {g.name:<18} {gpu_iq4:>7.0f} {gpu_q3:>7.0f} {gpu_q2:>7.0f} {g.ppl_delta_q3k:>+8.3f} {g.ppl_delta_q2k:>+8.3f} {eff_q3:>8.0f}/{eff_q2:>8.0f}")
    
    # Greedy allocation
    print(f"\n  --- Greedy Allocation ---")
    
    # Start with all IQ4_NL
    allocation = {}
    total_gpu_mb = 0
    total_ppl_delta = 0
    
    for g in TENSOR_GROUPS:
        gpu_size = g.size_iq4nl_mb * GPU_FRACTION
        allocation[g.name] = {"format": "IQ4_NL", "gpu_mb": gpu_size, "ppl_delta": 0}
        total_gpu_mb += gpu_size
    
    print(f"  Initial GPU weight: {total_gpu_mb:.0f} MiB")
    print(f"  Budget remaining: {VRAM_BUDGET_MB - total_gpu_mb:.0f} MiB")
    
    # Greedily downscale groups with best efficiency
    candidates = []
    for g in TENSOR_GROUPS:
        if g.name == "norms_embed":
            continue  # Skip F32 norms
        
        # Try Q3_K
        saved_q3 = (g.size_iq4nl_mb - g.size_q3k_mb) * GPU_FRACTION
        if g.ppl_delta_q3k > 0:
            candidates.append((saved_q3 / g.ppl_delta_q3k, g, "Q3_K", saved_q3, g.ppl_delta_q3k))
        
        # Try Q2_K
        saved_q2 = (g.size_iq4nl_mb - g.size_q2k_mb) * GPU_FRACTION
        if g.ppl_delta_q2k > 0:
            candidates.append((saved_q2 / g.ppl_delta_q2k, g, "Q2_K", saved_q2, g.ppl_delta_q2k))
        
        # Try IQ3_XXS
        saved_iq3 = (g.size_iq4nl_mb - g.size_iq3xxs_mb) * GPU_FRACTION
        if g.ppl_delta_iq3xxs > 0:
            candidates.append((saved_iq3 / g.ppl_delta_iq3xxs, g, "IQ3_XXS", saved_iq3, g.ppl_delta_iq3xxs))
    
    # Sort by efficiency (highest first)
    candidates.sort(key=lambda x: -x[0])
    
    print(f"\n  Priority queue (efficiency = MB saved / PPL cost):")
    for eff, g, fmt, saved, ppl_cost in candidates[:10]:
        print(f"    {g.name:<18} → {fmt:<10} save {saved:>6.0f} MB, cost +{ppl_cost:.3f} PPL, eff={eff:>6.0f}")
    
    # Apply greedy allocation
    print(f"\n  --- Applied Allocation ---")
    remaining_budget = VRAM_BUDGET_MB - total_gpu_mb
    
    for eff, g, fmt, saved, ppl_cost in candidates:
        if saved <= 0:
            continue
        if remaining_budget <= 0:
            break
        
        # Check if this group is already at a better format
        current = allocation[g.name]
        if current["format"] != "IQ4_NL":
            continue  # Already downgraded
        
        # Apply downgrade
        if fmt == "Q3_K":
            new_gpu = g.size_q3k_mb * GPU_FRACTION
        elif fmt == "Q2_K":
            new_gpu = g.size_q2k_mb * GPU_FRACTION
        elif fmt == "IQ3_XXS":
            new_gpu = g.size_iq3xxs_mb * GPU_FRACTION
        else:
            continue
        
        delta_gpu = new_gpu - current["gpu_mb"]
        if abs(delta_gpu) > remaining_budget:
            continue
        
        allocation[g.name] = {"format": fmt, "gpu_mb": new_gpu, "ppl_delta": ppl_cost}
        total_ppl_delta += ppl_cost
        remaining_budget -= abs(delta_gpu)
        
        print(f"    {g.name}: IQ4_NL → {fmt} (save {abs(delta_gpu):.0f} MB, +{ppl_cost:.3f} PPL)")
    
    # Final results
    total_gpu = sum(a["gpu_mb"] for a in allocation.values())
    final_ppl = BASE_PPL + total_ppl_delta
    
    print(f"\n  --- FINAL ALLOCATION ---")
    print(f"  {'Group':<18} {'Format':<10} {'GPU MB':>8} {'PPL Δ':>8}")
    print(f"  {'-'*44}")
    for name, a in allocation.items():
        print(f"  {name:<18} {a['format']:<10} {a['gpu_mb']:>7.0f} {a['ppl_delta']:>+8.3f}")
    print(f"  {'-'*44}")
    print(f"  {'TOTAL':<18} {'':10} {total_gpu:>7.0f} {total_ppl_delta:>+8.3f}")
    
    print(f"\n  --- COMPARISON WITH MEASURED VARIANTS ---")
    print(f"  {'Variant':<25} {'Size':>8} {'PPL':>8} {'tg32':>8}")
    print(f"  {'-'*49}")
    # [CORRIGÉ 25/08/2026] tg32 historiques (build antérieur, NON réplicables)
    print(f"  {'A: IQ4_NL baseline':<25} {'18.85 GB':>8} {BASE_PPL:>8.3f} {BASE_TG32:>8.1f}")
    # [CORRIGÉ 25/08/2026] BUG : OPTIMAL appliquait GPU_FRACTION une 2e fois
    # (total_gpu est DÉJÀ la part GPU) + un ×2 magique. Taille modèle complet
    # correcte = total_gpu / GPU_FRACTION.
    total_model_gb = total_gpu / GPU_FRACTION / 1024
    print(f"  {'B: down=Q3_K':<25} {'17.50 GB':>8} {'7.593':>8} {'27.42*':>8}")
    print(f"  {'C: gate=Q3_K':<25} {'16.14 GB':>8} {'7.769':>8} {'28.39*':>8}")
    print(f"  {'E: down=Q2_K':<25} {'16.46 GB':>8} {'7.879':>8} {'28.57*':>8}")
    print(f"  {'F: down=IQ3_XXS':<25} {'17.02 GB':>8} {'7.681':>8} {'26.47*':>8}")
    print(f"  {'OPTIMAL (greedy)':<25} {f'{total_model_gb:.1f} GB':>8} {f'{final_ppl:.3f}':>8} {'~22':>8}")
    print(f"  (* tg32 = mesures historiques, NON réplicables sur build actuel)")
    
    # VRAM breakdown
    print(f"\n  --- VRAM BREAKDOWN (estimated) ---")
    weight_gpu = total_gpu
    compute_mb = 2000  # estimated compute buffer for 25 CPU layers
    kv_mb = 80         # MoE KV cache is tiny
    overhead_mb = 500  # other overhead
    total_vram = weight_gpu + compute_mb + kv_mb + overhead_mb
    
    print(f"  Weights on GPU:  {weight_gpu:>7.0f} MiB ({weight_gpu/VRAM_BUDGET_MB*100:.1f}%)")
    print(f"  Compute buffers: {compute_mb:>7.0f} MiB ({compute_mb/VRAM_BUDGET_MB*100:.1f}%)")
    print(f"  KV cache:        {kv_mb:>7.0f} MiB ({kv_mb/VRAM_BUDGET_MB*100:.1f}%)")
    print(f"  Overhead:        {overhead_mb:>7.0f} MiB ({overhead_mb/VRAM_BUDGET_MB*100:.1f}%)")
    print(f"  TOTAL:           {total_vram:>7.0f} MiB ({total_vram/VRAM_BUDGET_MB*100:.1f}%)")
    print(f"  Headroom:        {VRAM_BUDGET_MB - total_vram:>7.0f} MiB")
    
    # Save results
    results = {
        "allocation": {k: v for k, v in allocation.items()},
        "total_gpu_mb": total_gpu,
        "total_ppl_delta": total_ppl_delta,
        "final_ppl": final_ppl,
        "vram_breakdown": {
            "weights": weight_gpu,
            "compute": compute_mb,
            "kv": kv_mb,
            "overhead": overhead_mb,
            "total": total_vram,
        }
    }
    
    outpath = "C:/Users/videl/Desktop/lama 1080-5070/models/35b_exp/d2_moe_optimal.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {outpath}")


if __name__ == '__main__':
    solve_greedy()
