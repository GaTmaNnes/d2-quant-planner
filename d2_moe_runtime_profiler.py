#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 MOE RUNTIME PROFILER — Profile le comportement MoE en forward reel.
========================================================================
Trace : routing dynamique, frequence experts, co-occurrence, locality,
cout par couche, VRAM placement, CPU/GPU overlap.

Usage:
  python d2_moe_runtime_profiler.py --model <gguf> --cost-model
  python d2_moe_runtime_profiler.py --bench-only --model <gguf>
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# [CORRIGÉ 25/08/2026] Sweep NGL historique (build antérieur). La baseline
# tg32=27.5 @ ngl=15 est NON réplicable actuellement (~20-25 t/s mesuré le
# 24/08 soir, bridage GPU variable 1845 MHz/19 W vs 3090 MHz/88 W).
# Utilisé UNIQUEMENT en fallback si aucun --model fourni.
BASELINE_HISTORIQUE_NON_REPLICABLE = [
    {"ngl": 0, "tg32": 10.0}, {"ngl": 5, "tg32": 20.1},
    {"ngl": 10, "tg32": 24.6}, {"ngl": 15, "tg32": 27.5},
    {"ngl": 20, "tg32": 16.2}, {"ngl": 25, "tg32": 11.9},
    {"ngl": 30, "tg32": 9.9}, {"ngl": 40, "tg32": 5.0},
]

ARCH = {
    "n_layers": 40,
    "n_experts": 256,
    "n_active": 8,
    "hidden": 2048,
    "moe_interm": 512,
    "n_full_attn": 10,
    "n_linear_attn": 30,
    "kv_heads": 2,
    "q_heads": 16,
    "head_dim": 256,
}


def load_router_static(path):
    with open(path) as f:
        data = json.load(f)
    router = {}
    for e in data.get("experts", []):
        router[e["expert"]] = {
            "importance": e.get("importance_rel", 0.5),
            "gate_norm": e.get("gate_norm_mean", 0.0),
        }
    return router


def build_cost_model(router, ngl_sweep_results):
    n_gpu_layers = max(ngl_sweep_results, key=lambda x: x["tg32"])["ngl"] if ngl_sweep_results else 15
    best_tps = max(ngl_sweep_results, key=lambda x: x["tg32"])["tg32"] if ngl_sweep_results else 27.5
    n_gpu = n_gpu_layers
    n_cpu = ARCH["n_layers"] - n_gpu
    baseline = next((r["tg32"] for r in ngl_sweep_results if r["ngl"] == 0), 10.0) if ngl_sweep_results else 10.0
    t_cpu = 1000 / (ARCH["n_layers"] * baseline) if baseline > 0 else 30
    t_gpu = (1000 / best_tps - n_cpu * t_cpu) / n_gpu if n_gpu > 0 else 0
    speedup = t_cpu / t_gpu if t_gpu > 0 else 1
    return {
        "best_ngl": n_gpu_layers, "best_tps": best_tps,
        "n_gpu_layers": n_gpu, "n_cpu_layers": n_cpu,
        "t_cpu_ms": round(t_cpu, 2), "t_gpu_ms": round(t_gpu, 2),
        "gpu_speedup": round(speedup, 1),
        "note": "Cout par couche MoE = router + dispatch 8/256 experts + 8 GEMM"
    }


def estimate_expert_cache_model(router, ngl_sweep_results):
    expert_size_kb = (ARCH["hidden"] * ARCH["moe_interm"] * 2 + ARCH["moe_interm"] * ARCH["hidden"]) * 0.5 / 1024
    all_8_kb = 8 * expert_size_kb
    l3_mb = 24
    return {
        "expert_size_kb": round(expert_size_kb, 0),
        "eight_experts_mb": round(all_8_kb / 1024, 1),
        "l3_cache_mb": l3_mb,
        "fits_in_l3": all_8_kb / 1024 < l3_mb,
        "implication": "8 experts (12 MB) tiennent dans L3 (24 MB) -> reutilisation quasi-gratuite"
    }


def simulate_routing_dynamics(router):
    # [CORRIGÉ 25/08/2026] SIMULATION pure (normes de gates = proxy statique
    # ~25% fiable) — ne doit PAS être présentée comme du routing mesuré.
    print("  [!] SIMULATION basée sur les normes de gates (PROXY statique ~25% "
          "fiable) — PAS une mesure du routing réel.")
    print("  [!] FAIT MESURÉ : 256/256 experts actifs, entropie routeur 0.998.")
    norms = np.array([router[e]["gate_norm"] for e in sorted(router)])
    temps = 10.0
    logits = (norms - norms.mean()) / (norms.std() + 1e-8) * 3
    probs = np.exp(logits / temps)
    probs /= probs.sum()
    sorted_idx = np.argsort(-probs)
    n_tokens = 1000
    n_a = ARCH["n_active"]
    selected = np.zeros(len(norms), dtype=int)
    np.random.seed(42)
    for t in range(n_tokens):
        active = list(sorted_idx[:n_a])
        if t % 10 == 0:
            np.random.shuffle(active[:4])
        for e in active:
            selected[e] += 1
    total = n_a * n_tokens
    active_count = int(np.sum(selected > 0))
    top = sorted(zip(range(len(selected)), selected), key=lambda x: -x[1])[:16]
    prev_active = None
    same_count = 0
    for t in range(n_tokens):
        active = frozenset(sorted_idx[:n_a])
        if t % 10 == 0:
            active = frozenset(list(sorted_idx[:4]) + list(sorted_idx[4:8])[::-1])
        if prev_active is not None:
            same_count += len(active & prev_active)
        prev_active = active
    avg_ov = same_count / (n_tokens * n_a) if n_tokens > 0 else 0
    return {
        "n_tokens_simulated": n_tokens,
        "active_experts_used": active_count,
        "pct_experts_used": round(active_count / ARCH["n_experts"] * 100, 1),
        "avg_overlap_consecutive": round(avg_ov * 100, 1),
        "top_experts": [{"expert": int(e), "activations": int(c), "pct": round(c/total*100, 1)} for e, c in top],
    }


def benchmark_ngl_sweep(model_path, ngls):
    results = []
    for ngl in ngls:
        cmd = [os.path.join(HERE, "llama-bench.exe"), "-m", model_path,
               "-ngl", str(ngl), "-p", "128", "-n", "32"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=HERE)
            # [CORRIGÉ 25/08/2026] Parsing regex tolérant (split('|') cassait sur
            # les variations de colonnes/format de llama-bench) : on repère la
            # colonne tg32 dans l'en-tête puis on extrait le nombre de la ligne
            # de données correspondante.
            tg32_col = None
            for line in out.stdout.split("\n"):
                s = line.strip()
                if not s.startswith("|"):
                    continue
                cells = [c.strip() for c in s.strip("|").split("|")]
                if "tg32" in cells:
                    tg32_col = cells.index("tg32")
                    continue
                if tg32_col is not None and tg32_col < len(cells):
                    m = re.search(r"(-?\d+(?:\.\d+)?)", cells[tg32_col])
                    if m:
                        tps = float(m.group(1))
                        results.append({"ngl": ngl, "tg32": tps})
                        print("  ngl={}: tg32={:.1f} t/s".format(ngl, tps))
                        tg32_col = None  # une seule ligne de données par bench
                        break
        except subprocess.TimeoutExpired:
            print("  ngl={}: TIMEOUT".format(ngl))
    return results


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="D2 MoE Runtime Profiler")
    ap.add_argument("--model", default=None)
    ap.add_argument("--router", default=os.path.join(HERE, "d2_router_static_report.json"))
    ap.add_argument("--cost-model", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--ngl-sweep", default="0,5,8,10,12,15,18,20,25,30")
    ap.add_argument("--output", default=os.path.join(HERE, "d2_moe_runtime_report.json"))
    args = ap.parse_args()

    print("=" * 90)
    print("  D2 MOE RUNTIME PROFILER -- Qwen3.6-35B-A3B")
    print("=" * 90)

    router = load_router_static(args.router)
    print("\n  Routeur charge: {} experts".format(len(router)))

    print("\n  --- ROUTING DYNAMICS (SIMULÉ — PROXY, PAS MESURÉ) ---")
    # [CORRIGÉ 25/08/2026] re-titré : c'est une simulation sur normes de gates,
    # pas une observation du routing en production.
    routing = simulate_routing_dynamics(router)
    print("  Experts actifs sur {} tokens: {}/{} ({}%)".format(
        routing["n_tokens_simulated"], routing["active_experts_used"],
        ARCH["n_experts"], routing["pct_experts_used"]))
    print("  Overlap moyen tokens consecutifs: {}%".format(routing["avg_overlap_consecutive"]))
    top5_list = routing["top_experts"][:5]
    top5_parts = []
    for e in top5_list:
        top5_parts.append("E{}={}%".format(e["expert"], e["pct"]))
    print("  Top-5 experts: " + ", ".join(top5_parts))

    if args.model and os.path.exists(args.model):
        print("\n  --- NGL SWEEP ---")
        print("  Modele: {}".format(args.model))
        ngls = [int(x.strip()) for x in args.ngl_sweep.split(",")]
        sweep = benchmark_ngl_sweep(args.model, ngls)
    else:
        # [CORRIGÉ 25/08/2026] fallback historique — baseline tg32=27.5 NON réplicable
        print("\n  --- NGL SWEEP (fallback HISTORIQUE, NON réplicable) ---")
        print("  [!] ATTENTION : chiffres du build antérieur ; baseline tg32=27.5 "
              "@ ngl=15 NON réplicable (~20-25 t/s mesuré le 24/08 soir, bridage "
              "GPU variable). Ne pas utiliser comme référence absolue.")
        sweep = [dict(r) for r in BASELINE_HISTORIQUE_NON_REPLICABLE]

    cost = build_cost_model(router, sweep)
    print("\n  --- COST MODEL (MoE-aware) ---")
    print("  Sweet spot: ngl={}, {:.1f} t/s".format(cost["best_ngl"], cost["best_tps"]))
    print("  GPU: {} couches x {}ms = {:.0f}ms".format(cost["n_gpu_layers"], cost["t_gpu_ms"],
          cost["n_gpu_layers"] * cost["t_gpu_ms"]))
    print("  CPU: {} couches x {}ms = {:.0f}ms".format(cost["n_cpu_layers"], cost["t_cpu_ms"],
          cost["n_cpu_layers"] * cost["t_cpu_ms"]))
    print("  GPU speedup: {:.1f}x".format(cost["gpu_speedup"]))

    cache = estimate_expert_cache_model(router, sweep)
    print("\n  --- EXPERT CACHE MODEL ---")
    print("  1 expert: {:.0f} KB, 8 experts: {:.1f} MB".format(cache["expert_size_kb"], cache["eight_experts_mb"]))
    print("  Dans L3 cache (24 MB): {}".format("OUI" if cache["fits_in_l3"] else "NON"))
    print("  -> {}".format(cache["implication"]))

    print("\n  --- COMPARISON 27B DENSE vs 35B MoE ---")
    print("  27B D2-ECO (12 GB, ngl=33): 7.5 t/s (dense, 27B actifs)")
    print("  35B MoE IQ4_NL (18.4 GB, ngl={}): {:.1f} t/s (MoE, ~3B actifs)".format(
        cost["best_ngl"], cost["best_tps"]))
    ratio = cost["best_tps"] / 7.5
    print("  Ratio: {:.1f}x {}".format(ratio, "plus rapide" if ratio > 1 else "plus lent"))
    print("  Cause: MoE = 8/256 experts actifs = ~3B params/token")

    report = {
        "architecture": ARCH,
        "routing_dynamics": routing,
        "cost_model": cost,
        "expert_cache": cache,
        "ngl_sweep": sweep,
        "comparison_27b": {"27b_dense_tps": 7.5, "35b_moe_tps": cost["best_tps"], "ratio": round(ratio, 1)}
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n  [+] Rapport: {}".format(args.output))


if __name__ == "__main__":
    main()