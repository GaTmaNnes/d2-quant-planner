#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 MoE ALLOCATION PLAN — Optimisation per-expert pour 35B MoE.
================================================================
Base sur :
  - Routing reel (embedding proxy) : 256/256 experts actifs, 6.4% overlap
  - SNR uniforme (FP16=147, INT8=42, Q4=17, Q3=10, Q2=1.4)
  - PPL mesure : 27B=8.21, 35B IQ4_NL=7.58
  - ngl sweep : best=15 (27.5 t/s), 5 (20), 20 (16), 30 (10)
    [CORRIGÉ 25/08/2026] sweep HISTORIQUE non réplicable (~20-25 mesuré le 24/08)
  - Architecture : 40 layers, hidden=2048, moe_interm=512

Le cout MoE par couche :
  - GPU : ~1.8ms (mesure calibree)
  - CPU : ~2.5ms (mesure calibree)
  - Le routing change 93.6% entre tokens → pas d'effet cache L3 significatif
  - MAIS les poids experts sont deja tous en RAM (pas de chargement par token)
    → c'est le COMPUTE qui est selectif, pas la memoire
"""

import json
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))

# === ARCHITECTURE ===
ARCH = {
    "n_layers": 40,
    "n_experts": 256,
    "n_active": 8,
    "n_full_attn": 10,
    "n_linear_attn": 30,
    "hidden": 2048,
    "moe_interm": 512,
    "kv_heads": 2,
    "q_heads": 16,
    "head_dim": 256,
}

# Tailles exactes des tenseurs experts (depuis les safetensors)
GATE_UP_ELEMS = 2048 * 512 * 2   # [1024, 2048] → 1024*2048×2 pour gate+up (ils sont concat)
DOWN_ELEMS = 2048 * 512          # [2048, 512]
EXPERT_ELEMS = GATE_UP_ELEMS + DOWN_ELEMS

# SNR par format (mesure, uniforme)
SNR = {
    "FP16": 147.2, "INT8": 42.3, "FP8": 31.7,
    "IQ4_NL": 17.1, "Q4_K_M": 17.0, "Q4_K_S": 16.5,
    "IQ3_S": 10.0, "Q3_K_M": 9.8, "Q3_K_S": 9.5,
    "Q2_K": 2.0, "IQ2_S": 1.8,
}

# BPW reel (pas theorique — IQ4_NL = 4.5 bpw dans le GGUF)
BPW_REAL = {
    "FP16": 16.0, "INT8": 8.0, "FP8": 8.0,
    "IQ4_NL": 4.5, "Q4_K_M": 4.8, "Q4_K_S": 4.4,
    "IQ3_S": 3.5, "Q3_K_M": 3.7, "Q3_K_S": 3.4,
    "Q2_K": 2.7, "IQ2_S": 2.5,
}

# Benchmark reel : ngl → t/s (tg32, KV f16)
# [CORRIGÉ 25/08/2026] ÉTIQUETTE HISTORIQUE : ces valeurs viennent du build
# antérieur ; le best 15→27.5 t/s est NON réplicable (~20-25 t/s mesuré le
# 24/08 soir, bridage GPU variable 1845 MHz/19 W vs 3090 MHz/88 W).
NGL_TPS_HISTORIQUE = {0: 10.0, 3: 15.0, 5: 20.1, 8: 22.5, 10: 24.6,
                      12: 26.0, 15: 27.5, 18: 20.0, 20: 16.2, 25: 11.9,
                      30: 9.9, 35: 7.0, 40: 5.0}
NGL_TPS = NGL_TPS_HISTORIQUE  # alias conservé pour compatibilité

# Poids non-experts par couche (SSM, attention, norms, shared expert)
# Mesure depuis les safetensors : ~3.1 GB BF16 pour les 40 couches hors experts
# En Q8 : ~1.55 GB. Par couche : ~39 MB
NON_EXPERT_PER_LAYER_MB_Q8 = 39  # Q8_0 
NON_EXPERT_TOTAL_GB_Q8 = 1.55

# Poids expert BF16 total : ~64 GB (mesure safetensors)
# Chaque expert (gate_up + down) : (1024*2048*2 + 2048*512) * 2 bytes = ~10.5 MB BF16
EXPERT_BF16_MB = (GATE_UP_ELEMS + DOWN_ELEMS) * 2 / (1024*1024)

# VRAM budget
VRAM_TOTAL_MIB = 8150
VRAM_KV_CACHE_4K_MIB = 80
VRAM_RUNTIME_MIB = 500
VRAM_AVAILABLE_MIB = VRAM_TOTAL_MIB - VRAM_KV_CACHE_4K_MIB - VRAM_RUNTIME_MIB


def calc_strategy(strategy_name, fmt_ratio):
    """
    fmt_ratio: dict {format: n_experts}  → les 256 experts repartis par format
    
    Retourne : taille GGUF, ngl_max, t/s estime, SNR pondere
    """
    # Taille des experts
    expert_gb = 0
    for fmt, n in fmt_ratio.items():
        bpe = BPW_REAL.get(fmt, 4.5) / 8
        expert_gb += EXPERT_ELEMS * bpe * n * ARCH["n_layers"] / (1024**3)
    
    # Taille par couche (TOUS les experts d'une couche)
    per_layer_expert_mb = EXPERT_ELEMS * sum(
        BPW_REAL[f] / 8 * n for f, n in fmt_ratio.items()
    ) / (1024**2)
    per_layer_total_mb = per_layer_expert_mb + NON_EXPERT_PER_LAYER_MB_Q8
    
    # Total
    total_gb = expert_gb + NON_EXPERT_TOTAL_GB_Q8

    # Combien de couches tiennent en VRAM
    # Note: VRAM ne peut pas contenir tous les experts — on met les couches entieres
    vram_per_layer_mib = per_layer_total_mb
    max_layers = int(VRAM_AVAILABLE_MIB / vram_per_layer_mib)
    n_gpu = min(max_layers, ARCH["n_layers"])
    n_cpu = ARCH["n_layers"] - n_gpu

    # t/s estime via interpolation du ngl sweep
    # On calibre : t_gpu, t_cpu depuis les donnees reelles
    # ngl=0 → 40 * t_cpu = 1000/10 = 100ms → t_cpu = 2.5ms
    # ngl=15 → 15 * t_gpu + 25 * t_cpu = 1000/27.5 = 36.4ms → t_gpu = (36.4-62.5)/15 ≈ -1.7
    # Ajustement: le modele n'est pas lineaire a cause du parallelisme CPU/GPU
    
    # Utilisons une interpolation directe du sweep
    tps = None
    for ngl_test, tps_test in sorted(NGL_TPS.items()):
        if ngl_test == n_gpu:
            tps = tps_test
            break
    if tps is None:
        # Interpolation lineaire entre les deux ngl les plus proches
        ngls_sorted = sorted(NGL_TPS.keys())
        for i in range(len(ngls_sorted) - 1):
            if ngls_sorted[i] <= n_gpu <= ngls_sorted[i + 1]:
                lo, hi = ngls_sorted[i], ngls_sorted[i + 1]
                frac = (n_gpu - lo) / (hi - lo)
                tps = NGL_TPS[lo] + frac * (NGL_TPS[hi] - NGL_TPS[lo])
                break
        if tps is None:
            tps = NGL_TPS.get(n_gpu, NGL_TPS[max(NGL_TPS.keys())])

    # SNR pondere (moyenne sur les 256 experts)
    avg_snr = sum(SNR.get(f, 17) * n for f, n in fmt_ratio.items()) / 256
    
    # PPL estime : modele lineaire SNR→PPL calibre sur les 2 points connus
    # IQ4_NL : SNR=17.1, PPL=7.58
    # D2-ECO Q2/Q3 mix : SNR~8, PPL=8.21
    # Delta SNR = 17.1-8 = 9.1 → Delta PPL = 7.58-8.21 = -0.63
    # Donc ~0.07 PPL par dB de SNR
    ppl_est = 7.58 + (17.1 - avg_snr) * 0.069

    return {
        "strategy": strategy_name,
        "format_distribution": fmt_ratio,
        "expert_gb": round(expert_gb, 1),
        "total_gguf_gb": round(total_gb, 1),
        "per_layer_mb": round(per_layer_total_mb, 0),
        "max_layers_vram": max_layers,
        "n_gpu_layers": n_gpu,
        "n_cpu_layers": n_cpu,
        "vram_used_mib": round(n_gpu * per_layer_total_mb + VRAM_KV_CACHE_4K_MIB + VRAM_RUNTIME_MIB, 0),
        "est_tps": round(tps, 1),
        "avg_snr_db": round(avg_snr, 1),
        "est_ppl": round(ppl_est, 2),
    }


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 105)
    print("  D2 MoE ALLOCATION PLAN — Qwen3.6-35B-A3B")
    print("=" * 105)
    print("  Architecture: {} layers, {} experts, {} actifs, hidden={}".format(
        ARCH["n_layers"], ARCH["n_experts"], ARCH["n_active"], ARCH["hidden"]))
    print("  VRAM dispo: {} MiB (total {} - {} KV - {} runtime)".format(
        VRAM_AVAILABLE_MIB, VRAM_TOTAL_MIB, VRAM_KV_CACHE_4K_MIB, VRAM_RUNTIME_MIB))
    # [CORRIGÉ 25/08/2026] Avertissement CRITIQUE sur les stratégies par strates
    print()
    print("  ⚠️ AVERTISSEMENT CRITIQUE [CORRIGÉ 25/08/2026] :")
    print("  Les stratégies INT8/Q2 PAR STRATES D'IMPORTANCE (top X% / bottom Y%)")
    print("  supposent que les experts sont différemment compressibles selon leur")
    print("  'importance'. OR les FAITS MESURÉS contredisent cette prémisse :")
    print("    - 256/256 experts actifs, aucun expert froid")
    print("    - entropie du routeur = 0.998 (routing quasi uniforme)")
    print("    - le proxy statique (gate norm) n'est fiable qu'à ~25%")
    print("  → Les résultats de ces stratégies par strates NE SONT PAS FIABLES.")
    print()

    strategies = {
        "IQ4_NL baseline": {"IQ4_NL": 256},
        "Q4_K_M": {"Q4_K_M": 256},
        "Q3_K_M (uniform)": {"Q3_K_M": 256},
        "Q3+Q4 mix (50/50)": {"Q3_K_M": 128, "Q4_K_M": 128},
        "Q2+K mix (70/30)": {"Q2_K": 180, "Q3_K_M": 76},
        "Q2_K (uniform)": {"Q2_K": 256},
        "IQ2_S (uniform)": {"IQ2_S": 256},
        "INT8 top25% + Q4 mid50% + Q2 bot25%": {"INT8": 64, "Q4_K_M": 128, "Q2_K": 64},
        "INT8 top15% + Q3 mid50% + Q2 bot35%": {"INT8": 38, "Q3_K_M": 128, "Q2_K": 90},
        "Q4 top50% + Q2 bot50%": {"Q4_K_M": 128, "Q2_K": 128},
    }

    results = []
    
    print("  {:35s} | {:>14s} | {:>8s} | {:>4s} | {:>4s} | {:>6s} | {:>6s} | {:>5s}".format(
        "Strategie", "Taille GGUF", "VRAM", "GPU", "CPU", "t/s", "SNR", "PPL"))
    print("  " + "-" * 101)

    for name, fmt_ratio in strategies.items():
        r = calc_strategy(name, fmt_ratio)
        results.append(r)
        print("  {:35s} | {:>9.1f} GB | {:>4.0f} MiB | {:>3d}L | {:>3d}L | {:>5.1f} t/s | {:>5.1f} dB | {:>5.2f}".format(
            name, r["total_gguf_gb"], r["vram_used_mib"],
            r["n_gpu_layers"], r["n_cpu_layers"], r["est_tps"],
            r["avg_snr_db"], r["est_ppl"]))

    print("  " + "-" * 101)
    
    # Best by different criteria
    best_tps = max(results, key=lambda r: r["est_tps"])
    best_ppl = min(results, key=lambda r: r["est_ppl"])
    best_balanced = max(results, key=lambda r: r["est_tps"] * (20 - r["est_ppl"]))

    print()
    print("  Meilleur t/s : {} ({:.1f} t/s, PPL={:.2f})".format(
        best_tps["strategy"], best_tps["est_tps"], best_tps["est_ppl"]))
    print("  Meilleur PPL : {} (PPL={:.2f}, {:.1f} t/s)".format(
        best_ppl["strategy"], best_ppl["est_ppl"], best_ppl["est_tps"]))
    print("  Meilleur equilibre : {} ({:.1f} t/s, PPL={:.2f})".format(
        best_balanced["strategy"], best_balanced["est_tps"], best_balanced["est_ppl"]))

    # Comparison avec la production actuelle
    # [CORRIGÉ 25/08/2026] baseline remplacée : le 27B D2-ECO est abandonné ;
    # PRODUCTION = 35B D2-MOE (17.5 GB, PPL 7.593, -ngl 15, ~5 t/s serveur,
    # tg32 bench ~5.6 sur build actuel — le 27.5 historique n'est pas réplicable).
    print()
    print("  --- VS PRODUCTION 35B D2-MOE (17.5 GB, PPL=7.593, ngl=15) ---")
    print("  [!] t/s de référence : ~5 serveur / 5.6 tg32 bench (build actuel, bridage variable)")
    baseline_prod = {"est_tps": 5.6, "est_ppl": 7.593}
    for r in [best_tps, best_ppl, best_balanced]:
        print("  {}: t/s {:.1f}→{:.1f} ({:.1f}x), PPL {:.2f}→{:.2f} ({:+.2f})".format(
            r["strategy"][:30],
            baseline_prod["est_tps"], r["est_tps"], r["est_tps"]/baseline_prod["est_tps"],
            baseline_prod["est_ppl"], r["est_ppl"], r["est_ppl"]-baseline_prod["est_ppl"]))

    # Sauvegarde
    output = {
        "architecture": ARCH,
        "snr_per_format": SNR,
        "bpw_real": BPW_REAL,
        "routing_findings": {
            "experts_active": "256/256 (100%)",
            "consecutive_overlap": "6.4%",
            "method": "embedding_proxy",
            # [CORRIGÉ 25/08/2026] faits consolidés
            "router_entropy": 0.998,
            "cold_experts": 0,
            "static_proxy_reliability": "~25%",
            "note": "Routing hautement dynamique — tous les experts sont utilises"
        },
        # [CORRIGÉ 25/08/2026] baseline 27B remplacée par la production D2-MOE
        "baseline_prod": {"model": "35B D2-MOE", "ppl": 7.593, "tps": 5.6,
                          "size_gb": 17.5, "ngl": 15,
                          "note": "~5 t/s serveur ; 27.5 tg32 historique NON réplicable"},
        "baseline_27b": {"model": "D2-ECO (ABANDONNÉ, historique)", "ppl": 8.21,
                         "tps": 7.5, "size_gb": 12.0},
        "baseline_35b": {"model": "IQ4_NL", "ppl": 7.58, "tps_historique_non_replicable": 27.5, "size_gb": 18.4},
        "strategies": results,
        "best_tps": best_tps,
        "best_ppl": best_ppl,
        "best_balanced": best_balanced,
    }
    
    out_path = os.path.join(HERE, "d2_moe_allocation_plan.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n  [+] Plan: {}".format(out_path))


if __name__ == "__main__":
    main()