#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 SLOW LAYER PROFILER — identifier les couches lentes du 27B
================================================================

Principe :
  1. Prendre le D2-ECO (12 GiB, 6.84 t/s à ngl=30)
  2. Pour chaque layer, essayer de remonter ses tensors en Q3/Q4/Q8
  3. Mesurer le delta tg32 avec llama-bench --override-tensor
  4. Classer les layers par EXCESS_TIME (ms perdues, pas %)
  5. Générer D2-ECO-v2 avec les slow layers remontés

Utilise :
  - llama-bench.exe -m <model> -ngl 30 --override-tensor <pattern>=<type> -n 32 -p 128
  - Les types supportés : F16, Q8_0, Q4_K, Q3_K, Q2_K, IQ4_NL, ...

Usage:
  python d2_slow_layer_profiler.py
  python d2_slow_layer_profiler.py --model models/Qwen3.8-27B-D2-ECO.gguf --ngl 30
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Config par défaut
DEFAULT_MODEL = os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO.gguf")
BENCH_EXE = os.path.join(HERE, "llama-bench.exe")
N_GPU_LAYERS = 30
N_TOKENS = 32
P_PROMPT = 128
N_RUNS = 3  # mesures par config pour la stabilité

# Types de quantification à tester par layer
# Q3_K et Q4_K sont les plus pertinents (entre Q2 actuel et Q8)
OVERRIDE_PRECISIONS = ["Q8_0", "Q4_K", "Q3_K"]

# Patterns d'override par type de tensor dans un layer
TENSOR_PATTERNS = {
    "ffn_down": "blk.{layer}.ffn_down.weight",
    "ffn_up":   "blk.{layer}.ffn_up.weight",
    "ffn_gate": "blk.{layer}.ffn_gate.weight",
    "attn_q":   "blk.{layer}.attn_q.weight",
    "attn_k":   "blk.{layer}.attn_k.weight",
    "attn_v":   "blk.{layer}.attn_v.weight",
    "attn_o":   "blk.{layer}.attn_o.weight",
}

# Layers à tester (tous les 2 pour gagner du temps, puis les candidats à pleine résolution)
LAYER_SCAN_STEP = 2  # tester tous les 2 layers d'abord
FULL_SCAN_LAYERS = []  # rempli après le scan initial


def run_bench(model, ngl, override=None, n=N_TOKENS, p=P_PROMPT, runs=N_RUNS):
    """Lance llama-bench et retourne le tg median sur 'runs' mesures."""
    cmd = [
        BENCH_EXE,
        "-m", model,
        "-ngl", str(ngl),
        "-n", str(n),
        "-p", str(p),
        "-t", "8",
    ]
    if override:
        cmd += ["--override-tensor", override]

    times = []
    for _ in range(runs):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                encoding="utf-8", errors="replace"
            )
            output = result.stdout + result.stderr
            # Extraire tg32 : | ... | tg32 | X.XX ± Y.YY |
            m = re.search(r"\|\s*tg\d+\s*\|\s*([\d.]+)\s*±", output)
            if m:
                times.append(float(m.group(1)))
        except (subprocess.TimeoutExpired, Exception):
            pass

    if not times:
        return None
    times.sort()
    return times[len(times) // 2]  # median


def get_model_layers(model_path):
    """Détecte le nombre de layers du modèle via les strings GGUF ou un bench quick."""
    # Estimation pour Qwen3.8-27B : 64 layers
    # On peut aussi le lire depuis le config si disponible
    cfg_path = os.path.join(HERE, "models", "Qwen3.8-27B-FP8", "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        n_layers = cfg.get("num_hidden_layers", 64)
        return n_layers
    return 64  # défaut Qwen3.8-27B


def profile_baseline(model, ngl):
    """Mesure la baseline tg32 sans override."""
    print(f"[+] Baseline : {model} ngl={ngl}")
    tg = run_bench(model, ngl)
    if tg:
        print(f"    baseline tg32 = {tg:.2f} t/s")
    else:
        print("    [!] baseline FAILED")
    return tg


def profile_layer_overrides(model, ngl, n_layers, baseline_tg, scan_step=4):
    """Pour chaque layer, override ses tensors principaux en Q8_0/Q4_K/Q3_K
    et mesure le delta de vitesse."""
    results = []

    for layer in range(0, n_layers, scan_step):
        layer_results = {"layer": layer, "overrides": {}}

        for prec in OVERRIDE_PRECISIONS:
            # Override tous les tensors du layer d'un coup
            pattern = f"blk.{layer}.*.weight"
            override_str = f"{pattern}={prec}"

            tg = run_bench(model, ngl, override=override_str)
            if tg and baseline_tg:
                delta_ms = (1000 / tg) - (1000 / baseline_tg)  # ms per token
                delta_pct = ((tg / baseline_tg) - 1) * 100
                layer_results["overrides"][prec] = {
                    "tg32": round(tg, 2),
                    "delta_pct": round(delta_pct, 1),
                    "excess_ms": round(delta_ms, 4),
                }
                status = "FASTER" if delta_pct > 0 else "SLOWER"
                print(f"    L{layer:02d} {prec:5s} -> {tg:.2f} t/s ({delta_pct:+.1f}%) {status}")
            else:
                layer_results["overrides"][prec] = {"tg32": None, "error": "FAILED"}

        results.append(layer_results)

    return results


def profile_slow_layers(model, ngl, slow_layers, baseline_tg):
    """Pour les slow layers identifiés, faire un profil tensor-par-tensor."""
    tensor_results = []

    for layer in slow_layers:
        print(f"\n[+] Profil tensor détaillé L{layer:02d}:")
        for tname, tpattern in TENSOR_PATTERNS.items():
            pattern = tpattern.format(layer=layer)
            for prec in OVERRIDE_PRECISIONS:
                override_str = f"{pattern}={prec}"
                tg = run_bench(model, ngl, override=override_str, runs=2)
                if tg and baseline_tg:
                    delta_pct = ((tg / baseline_tg) - 1) * 100
                    tensor_results.append({
                        "layer": layer, "tensor": tname,
                        "precision": prec, "tg32": round(tg, 2),
                        "delta_pct": round(delta_pct, 1),
                    })
                    print(f"      {tname:10s} {prec:5s} -> {tg:.2f} t/s ({delta_pct:+.1f}%)")

    return tensor_results


def generate_slow_layer_map(results, baseline_tg):
    """Identifie les slow layers : ceux où passer de Q2 à Q3/Q4 améliore la vitesse."""
    slow_layers = []
    for r in results:
        layer = r["layer"]
        base_improvement = r["overrides"].get("Q3_K", {}).get("delta_pct", 0)
        if base_improvement > 5:  # >5% de gain en passant à Q3
            slow_layers.append({
                "layer": layer,
                "gain_q3": base_improvement,
                "excess_ms": r["overrides"].get("Q3_K", {}).get("excess_ms", 0),
            })
    # Trier par gain décroissant
    slow_layers.sort(key=lambda x: -x["gain_q3"])
    return slow_layers


def generate_mixed_precision_map(results, slow_layers, baseline_tg):
    """Génère la carte de précision mixte pour le rebuild."""
    precision_map = {}
    slow_layer_ids = {s["layer"] for s in slow_layers}

    for r in results:
        layer = r["layer"]
        if layer not in slow_layer_ids:
            continue

        # Choisir la meilleure précision pour ce layer
        best_prec = "Q3_K"  # défaut pour slow layers
        best_gain = 0
        for prec, data in r["overrides"].items():
            if data.get("delta_pct", 0) > best_gain:
                best_gain = data["delta_pct"]
                best_prec = prec

        precision_map[f"blk.{layer}"] = {
            "recommended_precision": best_prec,
            "gain_pct": round(best_gain, 1),
            "reason": f"slow layer: +{best_gain:.1f}% with {best_prec}",
        }

    return precision_map


def generate_override_file(precision_map):
    """Génère un fichier d'override pour llama-quantize."""
    overrides = []
    for layer_key, info in precision_map.items():
        prec = info["recommended_precision"]
        # Override tous les .weight tensors de ce layer
        overrides.append(f"{layer_key}.*.weight={prec}")
    return overrides


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 SLOW LAYER PROFILER")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="GGUF à profiler")
    ap.add_argument("--ngl", type=int, default=N_GPU_LAYERS, help="n GPU layers")
    ap.add_argument("--step", type=int, default=4, help="pas de scan layer")
    ap.add_argument("--json", default=os.path.join(HERE, "d2_slow_layer_report.json"))
    args = ap.parse_args()

    scan_step = args.step
    n_layers = get_model_layers(args.model)
    print(f"[+] Modèle : {args.model}")
    print(f"[+] Layers : {n_layers}, ngl={args.ngl}, step={scan_step}")
    print(f"[+] Precisions testées : {OVERRIDE_PRECISIONS}")
    print()

    # 1. Baseline
    baseline_tg = profile_baseline(args.model, args.ngl)
    if not baseline_tg:
        print("[!] Impossible de mesurer la baseline. Abort.")
        sys.exit(1)

    # 2. Scan par layer
    print(f"\n[+] Scan layer (step={scan_step})...")
    layer_results = profile_layer_overrides(args.model, args.ngl, n_layers, baseline_tg, scan_step)

    # 3. Identifier les slow layers
    slow_layers = generate_slow_layer_map(layer_results, baseline_tg)
    print(f"\n[+] Slow layers identifiés : {len(slow_layers)}")
    for s in slow_layers[:10]:
        print(f"    L{s['layer']:02d} : gain Q3 = +{s['gain_q3']:.1f}% ({s['excess_ms']:.4f} ms)")

    # 4. Profil tensor détaillé des slow layers
    tensor_results = []
    if slow_layers:
        print(f"\n[+] Profil tensor détaillé top {min(5, len(slow_layers))} slow layers...")
        tensor_results = profile_slow_layers(
            args.model, args.ngl, [s["layer"] for s in slow_layers[:5]], baseline_tg
        )

    # 5. Générer la carte de précision mixte
    precision_map = generate_mixed_precision_map(layer_results, slow_layers, baseline_tg)
    overrides = generate_override_file(precision_map)

    # 6. Sauvegarder le rapport
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "ngl": args.ngl,
        "baseline_tg32": round(baseline_tg, 2),
        "n_layers": n_layers,
        "slow_layers": slow_layers,
        "layer_results": layer_results,
        "tensor_results": tensor_results,
        "precision_map": precision_map,
        "overrides_for_rebuild": overrides,
        "caveats": [
            "tg32 est un test court — à confirmer avec tg128/tg256",
            "override-tensor modifie UN tensor à la fois, pas l'interaction layer",
            "Les gains sont additifs estimés, pas mesurés conjointement",
            "Le vrai rebuild nécessite llama-quantize avec --override-tensor",
        ],
    }

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Rapport sauvegardé : {args.json}")
    print(f"[+] {len(overrides)} overrides générés pour le rebuild")

    if overrides:
        print("\n[+] Commande de rebuild :")
        override_str = " ".join(f"--override-tensor '{o}'" for o in overrides[:10])
        print(f"    llama-quantize {args.model} output.gguf Q4_K {override_str}")
        print(f"    ({len(overrides)} overrides au total)")


if __name__ == "__main__":
    main()
