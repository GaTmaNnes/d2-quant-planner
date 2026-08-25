#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 35B EXPERT ALLOCATION — Plan de quantification optimal par expert.
Croise importance routeur (gate norm) × SNR par format × coût mémoire.
Calcule la taille GGUF, RAM, VRAM pour chaque stratégie.
Ne nécessite PAS de reprofiler les 256×40 experts (SNR uniforme confirmé).
"""

import json
import os
import sys
import math
from collections import defaultdict, Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Données mesurées
ROUTER_REPORT = os.path.join(HERE, "d2_router_static_report.json")
EXPERT_REPORT = os.path.join(HERE, "d2_expert_profiler_report.json")
OUT_ALLOC = os.path.join(HERE, "d2_35b_expert_plan.json")

# Architecture 35B MoE (Qwen3.5-35B-A3B)
N_LAYERS = 40
N_EXPERTS = 256
N_ACTIVE = 8
HIDDEN = 2048
MOE_INTERM = 512              # par expert
FULL_ATTN_INTERVAL = 4        # toutes les 4 couches
N_FULL_ATTN = 10              # 40/4
N_LINEAR_ATTN = 30

# Tailles par composant (BF16 natif = référence)
GATE_UP_ELEMS = HIDDEN * MOE_INTERM * 2  # gate_proj + up_proj fusionnés
DOWN_ELEMS = MOE_INTERM * HIDDEN
EXPERT_ELEMS = GATE_UP_ELEMS + DOWN_ELEMS  # 1024*2048*2 + 512*2048

BYTES_PER_FMT = {"FP16": 2.0, "FP8": 1.0, "INT8": 1.0, "NVFP4": 0.5,
                 "Q4": 0.5, "Q3": 0.375, "Q2": 0.25}

# SNR par format (moyenne mesurée, stable sur tous les experts)
SNR_PER_FMT = {
    "FP16": 147.2,
    "FP8": 31.7,
    "INT8": 42.3,
    "NVFP4": 10.1,
    "Q4": 17.1,
    "Q3": 9.8,
    "Q2": 1.4,
}

# Poids non-experts
# [CORRIGÉ 25/08/2026] CONTRADICTION HARMONISÉE : ce fichier disait ~6.3 GB BF16
# alors que d2_moe_alloc.py mesure ~3.1 GB BF16 (safetensors, 40 couches hors
# experts). Chiffre RETENU : 3.1 GB BF16 → 1.55 GB Q8 (aligné sur d2_moe_alloc.py).
NON_EXPERT_GB_BF16 = 3.1
NON_EXPERT_GB_Q8 = NON_EXPERT_GB_BF16 / 2  # quantifié en Q8 = moitié


def load_router_importance():
    """Charge importance relative par expert depuis le routeur statique."""
    with open(ROUTER_REPORT) as f:
        rdata = json.load(f)
    
    experts = rdata.get("experts", [])
    result = {}
    for e in experts:
        result[e["expert"]] = {
            "importance": e.get("importance_rel", 0.5),
            "gate_norm_mean": e.get("gate_norm_mean", 0.0),
            "gate_norm_std": e.get("gate_norm_std", 0.0),
        }
    return result


def allocate_experts(router, strategy="balanced"):
    """
    Alloue un format de quantification à chaque expert.
    
    Stratégies:
    - "uniform_int8": tout en INT8 (baseline qualité)
    - "uniform_q4": tout en Q4 (baseline taille)
    - "balanced": top 25% INT8, mid 50% Q4, bottom 25% Q3
    - "aggressive": top 10% INT8, mid 40% Q4, bottom 50% Q2
    - "conservative": top 50% INT8, mid 30% Q4, bottom 20% Q3
    - "weighted": allocation continue basée sur importance normalisée
    """
    sorted_experts = sorted(router.items(), key=lambda x: x[1]["importance"], reverse=True)
    n = len(sorted_experts)
    
    allocation = {}
    
    if strategy == "uniform_int8":
        for eid, _ in sorted_experts:
            allocation[eid] = "INT8"
            
    elif strategy == "uniform_q4":
        for eid, _ in sorted_experts:
            allocation[eid] = "Q4"
            
    elif strategy == "balanced":
        # Top 25% → INT8, mid 50% → Q4, bottom 25% → Q3
        t1 = int(n * 0.25)
        t2 = int(n * 0.75)
        for i, (eid, _) in enumerate(sorted_experts):
            if i < t1:
                allocation[eid] = "INT8"
            elif i < t2:
                allocation[eid] = "Q4"
            else:
                allocation[eid] = "Q3"
                
    elif strategy == "aggressive":
        t1 = int(n * 0.10)
        t2 = int(n * 0.50)
        for i, (eid, _) in enumerate(sorted_experts):
            if i < t1:
                allocation[eid] = "INT8"
            elif i < t2:
                allocation[eid] = "Q4"
            else:
                allocation[eid] = "Q2"
                
    elif strategy == "conservative":
        t1 = int(n * 0.50)
        t2 = int(n * 0.80)
        for i, (eid, _) in enumerate(sorted_experts):
            if i < t1:
                allocation[eid] = "INT8"
            elif i < t2:
                allocation[eid] = "Q4"
            else:
                allocation[eid] = "Q3"
                
    elif strategy == "weighted":
        # Allocation continue : imp > 0.80 → INT8, > 0.76 → Q4, > 0.72 → Q3, sinon Q2
        for eid, data in router.items():
            imp = data["importance"]
            if imp > 0.80:
                allocation[eid] = "INT8"
            elif imp > 0.77:
                allocation[eid] = "Q4"
            elif imp > 0.74:
                allocation[eid] = "Q3"
            else:
                allocation[eid] = "Q2"
    
    elif strategy == "three_tier":
        # INT8 × 4 experts, Q4 × 3 experts, Q2 × 1 expert → moyenne pondérée
        # Simule que le routeur active 8 experts: 4 importants, 3 moyens, 1 faible
        # [CORRIGÉ 25/08/2026] ⚠️ PRÉMISSE INVALIDE : suppose que les top-experts
        # restent stables. FAITS MESURÉS : entropie routeur = 0.998, 256/256
        # experts actifs, aucun expert froid → PAS de top-experts stables.
        # Résultats de cette stratégie NON FIABLES (conservée pour historique).
        t1 = int(n * 0.15)  # top 15% → INT8 (couvre les 4/8 experts actifs)
        t2 = int(n * 0.55)  # mid 40% → Q4 (couvre 3/8)
        for i, (eid, _) in enumerate(sorted_experts):
            if i < t1:
                allocation[eid] = "INT8"
            elif i < t2:
                allocation[eid] = "Q4"
            else:
                allocation[eid] = "Q3"
    
    return allocation


def compute_stats(allocation, router, strategy_name):
    """Calcule taille, coût mémoire, SNR pondéré pour une allocation."""
    # Distribution
    dist = Counter(allocation.values())
    
    # Taille des experts
    expert_bytes = 0
    for eid, fmt in allocation.items():
        bpe = BYTES_PER_FMT[fmt]
        expert_bytes += EXPERT_ELEMS * bpe * N_LAYERS
    
    expert_gb = expert_bytes / (1024**3)
    total_gb = expert_gb + NON_EXPERT_GB_Q8
    
    # SNR pondéré par importance (qualité effective)
    weighted_snr = 0
    total_weight = 0
    for eid, fmt in allocation.items():
        imp = router[eid]["importance"]
        snr = SNR_PER_FMT[fmt]
        weighted_snr += snr * imp
        total_weight += imp
    avg_weighted_snr = weighted_snr / total_weight if total_weight > 0 else 0
    
    # SNR par token (8 experts actifs, SNR du pire expert activé)
    # En pratique: un token active 8 experts, la qualité dépend du plus faible
    sorted_by_imp = sorted(router.items(), key=lambda x: x[1]["importance"], reverse=True)
    top8_formats = [allocation[eid] for eid, _ in sorted_by_imp[:N_ACTIVE]]
    top8_snr = [SNR_PER_FMT[f] for f in top8_formats]
    min_top8_snr = min(top8_snr)
    avg_top8_snr = sum(top8_snr) / len(top8_snr)
    
    # VRAM estimation
    # CPU+GPU: 33 Go RAM, 8 Go VRAM
    # Priorité: mettre les experts INT8 (plus lourds) en RAM, Q3/Q2 en VRAM
    
    # Taille par stratum
    int8_experts = sum(1 for f in allocation.values() if f == "INT8")
    q4_experts = sum(1 for f in allocation.values() if f == "Q4")
    q3_experts = sum(1 for f in allocation.values() if f == "Q3")
    q2_experts = sum(1 for f in allocation.values() if f == "Q2")
    
    int8_size_gb = int8_experts * EXPERT_ELEMS * BYTES_PER_FMT["INT8"] * N_LAYERS / (1024**3)
    q4_size_gb = q4_experts * EXPERT_ELEMS * BYTES_PER_FMT["Q4"] * N_LAYERS / (1024**3)
    q3_size_gb = q3_experts * EXPERT_ELEMS * BYTES_PER_FMT["Q3"] * N_LAYERS / (1024**3)
    q2_size_gb = q2_experts * EXPERT_ELEMS * BYTES_PER_FMT["Q2"] * N_LAYERS / (1024**3)
    
    # Per-layer breakdown
    per_layer_expert_gb = expert_gb / N_LAYERS  # total expert size for ONE expert per layer
    per_layer_total_gb = per_layer_expert_gb + NON_EXPERT_GB_Q8 / N_LAYERS
    
    # Max layers in 8 GB VRAM
    vram_budget = 8.0  # GiB
    kv_cache_ctx4k_gb = 80 / 1024  # ~0.08 GB
    runtime_gb = 0.5
    available_vram = vram_budget - kv_cache_ctx4k_gb - runtime_gb
    max_layers_vram = int(available_vram / per_layer_total_gb)
    
    # t/s estimation (CPU vs GPU split)
    # GPU layers: compute at memory bandwidth speed (~180 GB/s for RTX 5070)
    # CPU layers: bottleneck = RAM bandwidth (~50 GB/s) + routing overhead
    gpu_ms_per_layer = 2  # ~2ms per layer on GPU with expert dispatch
    # [CORRIGÉ 25/08/2026] cpu_ms_per_layer=60ms était faux : MESURÉ = ~2.5 ms
    # par couche CPU (calibré ngl=0 → 40×2.5 ms ≈ 100 ms/token ≈ 10 t/s).
    cpu_ms_per_layer = 2.5
    
    n_gpu_layers = min(max_layers_vram, N_LAYERS)
    n_cpu_layers = N_LAYERS - n_gpu_layers
    est_tps = 1000 / (n_gpu_layers * gpu_ms_per_layer + n_cpu_layers * cpu_ms_per_layer)
    
    # Active weights per token (8 experts + shared)
    active_weight_per_layer_gb = (sum(
        EXPERT_ELEMS * BYTES_PER_FMT[allocation[eid]] for eid, _ in sorted_by_imp[:N_ACTIVE]
    ) / N_ACTIVE) / (1024**3)
    
    return {
        "strategy": strategy_name,
        "distribution": {k: v for k, v in dist.items()},
        "expert_gb": round(expert_gb, 1),
        "total_gb": round(total_gb, 1),
        "per_layer_gb": round(per_layer_total_gb, 2),
        "int8_gb": round(int8_size_gb, 1),
        "q4_gb": round(q4_size_gb, 1),
        "q3_gb": round(q3_size_gb, 1),
        "q2_gb": round(q2_size_gb, 1),
        "avg_weighted_snr": round(avg_weighted_snr, 1),
        "avg_top8_snr": round(avg_top8_snr, 1),
        "min_top8_snr": round(min_top8_snr, 1),
        "max_layers_vram": max_layers_vram,
        "n_gpu_layers": n_gpu_layers,
        "n_cpu_layers": n_cpu_layers,
        "est_tps": round(est_tps, 2),
        "est_gpu_ms": round(n_gpu_layers * gpu_ms_per_layer, 1),
        "est_cpu_ms": round(n_cpu_layers * cpu_ms_per_layer, 1),
        "est_total_ms": round(n_gpu_layers * gpu_ms_per_layer + n_cpu_layers * cpu_ms_per_layer, 1),
    }


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    
    # Charger les données routeur
    router = load_router_importance()
    print(f"Routeur: {len(router)} experts chargés")
    print(f"Importance: min={min(e['importance'] for e in router.values()):.3f} "
          f"max={max(e['importance'] for e in router.values()):.3f} "
          f"spread={100*(max(e['importance'] for e in router.values())-min(e['importance'] for e in router.values()))/0.75:.1f}%")
    print()
    
    # Architecture
    print(f"Architecture:")
    print(f"  {N_LAYERS} couches, {N_EXPERTS} experts, {N_ACTIVE} actifs/token")
    print(f"  hidden={HIDDEN}, moe_interm={MOE_INTERM}")
    print(f"  expert elems: {EXPERT_ELEMS/1e6:.1f}M ({GATE_UP_ELEMS/1e6:.1f}M gate_up + {DOWN_ELEMS/1e6:.1f}M down)")
    print(f"  non-expert weight: {NON_EXPERT_GB_BF16} GB BF16 → {NON_EXPERT_GB_Q8:.1f} GB Q8")
    print()
    
    # SNR
    print(f"SNR par format (mesuré, uniforme sur tous les experts):")
    for fmt, snr in SNR_PER_FMT.items():
        bpe = BYTES_PER_FMT[fmt]
        print(f"  {fmt:6s}: SNR={snr:5.1f} dB, {bpe} oct/élem")
    print()
    
    # Stratégies
    strategies = ["uniform_int8", "uniform_q4", "balanced", "aggressive", "conservative", "weighted", "three_tier"]
    results = []
    
    print("=" * 110)
    print(f"  {'Stratégie':<18s} | {'Dist':<25s} | {'Poids':>6s} | {'SNR top8':>8s} | "
          f"{'VRAM':>5s} | {'GPU':>3s} | {'CPU':>3s} | {'t/s':>6s}")
    print("  " + "-" * 106)
    
    for strat in strategies:
        alloc = allocate_experts(router, strat)
        stats = compute_stats(alloc, router, strat)
        results.append(stats)
        
        dist_str = " ".join(f"{k}={v}" for k, v in sorted(stats["distribution"].items()))
        print(f"  {stats['strategy']:<18s} | {dist_str:<25s} | {stats['total_gb']:>5.1f}G | "
              f"{stats['avg_top8_snr']:>5.1f}/{stats['min_top8_snr']:>4.1f} | "
              f"{stats['max_layers_vram']:>3d}L | {stats['n_gpu_layers']:>3d} | {stats['n_cpu_layers']:>3d} | "
              f"{stats['est_tps']:>5.2f}")
    
    print("  " + "-" * 106)
    
    # Meilleure stratégie
    best = max(results, key=lambda r: r["est_tps"] * r["avg_top8_snr"])  # t/s × qualité
    print(f"\n  Meilleure stratégie (t/s × SNR): {best['strategy']}")
    print(f"    Poids total: {best['total_gb']} GB, t/s estimé: {best['est_tps']}")
    print(f"    SNR top8: {best['avg_top8_snr']:.1f} dB, layers GPU: {best['n_gpu_layers']}")
    print()
    
    # Sauvegarde
    output = {
        "architecture": {
            "n_layers": N_LAYERS, "n_experts": N_EXPERTS, "n_active": N_ACTIVE,
            "hidden": HIDDEN, "moe_interm": MOE_INTERM,
            "non_expert_gb_q8": NON_EXPERT_GB_Q8,
        },
        "snr_per_format": SNR_PER_FMT,
        "strategies": results,
        "best": best,
    }
    
    with open(OUT_ALLOC, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[+] Plan sauvegardé: {OUT_ALLOC}")
    
    # Analyse de sensibilité: quel SNR par stratum?
    print("\n" + "=" * 80)
    print("  SENSIBILITÉ — Quel SNR pour les experts les plus/moins importants?")
    print("=" * 80)
    sorted_exp = sorted(router.items(), key=lambda x: x[1]["importance"], reverse=True)
    
    for label, indices in [
        ("Top 8 (activés)", range(0, 8)),
        ("Top 25% (très importants)", range(0, 64)),
        ("Mid 50%", range(64, 192)),
        ("Bottom 25% (peu importants)", range(192, 256)),
    ]:
        imps = [sorted_exp[i][1]["importance"] for i in indices if i < len(sorted_exp)]
        print(f"  {label:<30s}: imp={min(imps):.3f}-{max(imps):.3f} → "
              f"INT8={SNR_PER_FMT['INT8']:.0f}/Q4={SNR_PER_FMT['Q4']:.0f}/"
              f"Q3={SNR_PER_FMT['Q3']:.0f}/Q2={SNR_PER_FMT['Q2']:.0f} dB")
    
    # Conclusion
    # [CORRIGÉ 25/08/2026] conclusion mise à jour : le « 27B seul viable » est
    # OBSOLÈTE. PRODUCTION ACTUELLE = Qwen3.6-35B-A3B-D2-MOE (17.5 GB,
    # gate_up=IQ4_NL + down=Q3_K, PPL 7.593) avec -ngl 15 sur la RTX 5070.
    print("\n" + "=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    print(f"  ⚠️ three_tier et stratégies par strates : NON FIABLES (entropie routeur")
    print(f"  0.998, 256/256 experts actifs — pas de top-experts stables).")
    print(f"  Le 35B D2-MOE EST viable en production sur RTX 5070 8 Go avec -ngl 15")
    print(f"  (15 couches GPU / 25 CPU) : c'est le modèle de production actuel.")
    print(f"  Débit réel mesuré 24/08 : ~20-25 t/s tg32 selon bridage GPU variable")
    print(f"  (~5 t/s en serveur) ; le 27.5 t/s historique n'est PAS réplicable.")
    print(f"  La qualité (PPL {7.593}) prime : D2-MOE 35B > 27B dense (8.21).")


if __name__ == "__main__":
    main()