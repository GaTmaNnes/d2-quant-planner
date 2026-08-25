#!/usr/bin/env python3
# [OBSOLETE 25/08/2026] Périmètre 27B abandonné (prod = Qwen3.6-35B-A3B-D2-MOE).
# Raisons : cible Qwen3.8-27B inexistante ; --override-tensor inexistant dans ce fork.
# Conservé pour historique — NE PAS EXÉCUTER.
# -*- coding: utf-8 -*-
"""
D2 REBUILD ECO-V2 — reconstruire le GGUF avec précision mixte
==============================================================

Lit le rapport du slow layer profiler et génère un GGUF D2-ECO-v2
où les slow layers sont remontés en précision supérieure.

Principe :
  1. Lire d2_slow_layer_report.json
  2. Pour chaque slow layer, appliquer un override de quantization
  3. Utiliser llama-quantize avec --override-tensor pour reconstruire
  4. Benchmark le résultat avec llama-bench

Usage:
  python d2_rebuild_eco_v2.py
  python d2_rebuild_eco_v2.py --report d2_slow_layer_report.json
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_REPORT = os.path.join(HERE, "d2_slow_layer_report.json")
QUANTIZE_EXE = os.path.join(HERE, "tools", "llama-quantize.exe")
BENCH_EXE = os.path.join(HERE, "llama-bench.exe")
DEFAULT_MODEL = os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO.gguf")
OUTPUT_MODEL = os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO-v2.gguf")


def load_report(report_path):
    with open(report_path, encoding="utf-8") as f:
        return json.load(f)


def build_override_args(precision_map, base_quant="Q4_K"):
    """Construit les arguments --override-tensor pour llama-quantize.
    
    Pour chaque layer dans precision_map, on override les .weight tensors
    vers la précision recommandée.
    """
    overrides = []
    for layer_key, info in precision_map.items():
        prec = info["recommended_precision"]
        # Pattern pour tous les .weight tensors de ce layer
        pattern = f"{layer_key}.*.weight"
        overrides.append(f"{pattern}={prec}")
    return overrides


def run_quantize(input_model, output_model, base_quant, overrides):
    """Lance llama-quantize avec les overrides."""
    cmd = [QUANTIZE_EXE, input_model, output_model, base_quant]
    
    for ov in overrides:
        cmd += ["--override-tensor", ov]
    
    print(f"[+] Quantization : {len(overrides)} overrides")
    print(f"    Commande : {' '.join(cmd[:5])} ... ({len(overrides)} overrides)")
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace"
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"[!] Erreur quantize : {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("[!] Timeout quantize (>10 min)")
        return False
    except FileNotFoundError:
        print(f"[!] {QUANTIZE_EXE} introuvable — outils non compilés")
        print(f"    Alternative : utiliser llama-quantize du PATH ou du build CUDA")
        return False


def run_benchmark(model_path, ngl=30, n=32, p=128):
    """Benchmark rapide du modèle reconstruit."""
    cmd = [
        BENCH_EXE,
        "-m", model_path,
        "-ngl", str(ngl),
        "-n", str(n),
        "-p", str(p),
        "-t", "8",
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace"
        )
        output = result.stdout + result.stderr
        import re
        m = re.search(r"\|\s*tg\d+\s*\|\s*([\d.]+)\s*±", output)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 REBUILD ECO-V2")
    ap.add_argument("--report", default=DEFAULT_REPORT, help="Rapport du slow layer profiler")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Modèle source")
    ap.add_argument("--output", default=OUTPUT_MODEL, help="Modèle de sortie")
    ap.add_argument("--base-quant", default="Q4_K", help="Quantification de base")
    ap.add_argument("--benchmark", action="store_true", help="Bench après rebuild")
    ap.add_argument("--ngl", type=int, default=30, help="ngl pour le benchmark")
    args = ap.parse_args()

    # 1. Charger le rapport
    if not os.path.exists(args.report):
        print(f"[!] Rapport introuvable : {args.report}")
        print("    Lancer d2_slow_layer_profiler.py d'abord")
        sys.exit(1)

    report = load_report(args.report)
    precision_map = report.get("precision_map", {})
    baseline_tg = report.get("baseline_tg32", 0)

    print(f"[+] Rapport : {args.report}")
    print(f"[+] Baseline : {baseline_tg:.2f} t/s")
    print(f"[+] Slow layers à corriger : {len(precision_map)}")

    if not precision_map:
        print("[!] Aucun slow layer identifié. Rien à faire.")
        sys.exit(0)

    # 2. Afficher le plan
    print("\n[+] Plan de rebuild :")
    for layer_key, info in sorted(precision_map.items()):
        print(f"    {layer_key:25s} -> {info['recommended_precision']:5s} (+{info['gain_pct']:.1f}%)")

    # 3. Construire les overrides
    overrides = build_override_args(precision_map, args.base_quant)
    print(f"\n[+] {len(overrides)} overrides générés")

    # 4. Quantize
    if os.path.exists(QUANTIZE_EXE):
        success = run_quantize(args.model, args.output, args.base_quant, overrides)
        if not success:
            print("[!] Échec de la quantization")
            sys.exit(1)
        
        # Taille du résultat
        if os.path.exists(args.output):
            size_gb = os.path.getsize(args.output) / (1024**3)
            print(f"\n[+] Modèle reconstruit : {args.output}")
            print(f"    Taille : {size_gb:.2f} GiB")
    else:
        print(f"\n[!] llama-quantize.exe introuvable à {QUANTIZE_EXE}")
        print("    Sauvegarde du plan d'override dans d2_eco_v2_overrides.json")
        override_plan = {
            "base_model": args.model,
            "output_model": args.output,
            "base_quant": args.base_quant,
            "overrides": overrides,
            "precision_map": precision_map,
            "manual_command": f"llama-quantize {args.model} {args.output} {args.base_quant} " + 
                            " ".join(f"--override-tensor '{o}'" for o in overrides),
        }
        with open(os.path.join(HERE, "d2_eco_v2_overrides.json"), "w", encoding="utf-8") as f:
            json.dump(override_plan, f, ensure_ascii=False, indent=2)
        print("    Plan sauvegardé.")

    # 5. Benchmark
    if args.benchmark and os.path.exists(args.output):
        print(f"\n[+] Benchmark D2-ECO-v2 (ngl={args.ngl})...")
        tg_new = run_benchmark(args.output, ngl=args.ngl)
        if tg_new and baseline_tg:
            delta = ((tg_new / baseline_tg) - 1) * 100
            print(f"    D2-ECO    : {baseline_tg:.2f} t/s")
            print(f"    D2-ECO-v2 : {tg_new:.2f} t/s ({delta:+.1f}%)")
            if delta > 0:
                print(f"    ✓ GAIN : +{delta:.1f}% de tokens/s")
            else:
                print(f"    ✗ PERTE : {delta:.1f}% — les slow layers n'étaient pas le goulot")
        elif tg_new:
            print(f"    D2-ECO-v2 : {tg_new:.2f} t/s (baseline non disponible)")
        else:
            print("    [!] Benchmark échoué")

    print("\n[+] Terminé.")


if __name__ == "__main__":
    main()
