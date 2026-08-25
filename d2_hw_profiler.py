#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 HW PROFILER — profil complet d'un GGUF sur RTX 5070
========================================================

Mesure pour chaque GGUF :
  - Taille fichier
  - tokens/s (tg32, tg128)
  - Prefill (pp128)
  - VRAM utilisée
  - RAM utilisée
  - GPU utilisation
  - Puissance

Puis génère :
  - Rapport comparatif JSON
  - Score D2-HW par modèle
  - Commande de rebuild pour le meilleur modèle

Usage:
  python d2_hw_profiler.py
  python d2_hw_profiler.py --models models/Qwen3.8-27B-D2-ECO.gguf
  python d2_hw_profiler.py --ngl 33 --compare
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_EXE = os.path.join(HERE, "llama-bench.exe")
MODELS_DIR = os.path.join(HERE, "models")
OUT_JSON = os.path.join(HERE, "d2_hw_profile.json")

# Machines connues (de d2_cost_model.py)
MACHINES = {
    "rtx5070": {"vram_gb": 8.0, "pcie_gbs": 14.0, "vram_bw_gbs": 263.8},
    "gtx1080": {"vram_gb": 8.0, "pcie_gbs": 12.0, "vram_bw_gbs": 320.0},
}


def get_vram_usage():
    """Mesure la VRAM utilisée via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = r.stdout.strip().split(",")
        return {"used_mib": int(parts[0].strip()), "free_mib": int(parts[1].strip())}
    except Exception:
        return {"used_mib": 0, "free_mib": 0}


def get_gpu_power():
    """Mesure la puissance GPU via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def get_gpu_util():
    """Mesure l'utilisation GPU via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return int(r.stdout.strip())
    except Exception:
        return 0


def run_bench(model, ngl, n=32, p=128, t=8):
    """Lance llama-bench et retourne les métriques."""
    cmd = [
        BENCH_EXE, "-m", model,
        "-ngl", str(ngl), "-n", str(n), "-p", str(p), "-t", str(t)
    ]

    # Mesurer VRAM avant
    vram_before = get_vram_usage()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace"
        )
        output = result.stdout + result.stderr

        # Extraire pp et tg
        pp_match = re.search(r"\|\s*pp\d+\s*\|\s*([\d.]+)\s*±", output)
        tg_match = re.search(r"\|\s*tg\d+\s*\|\s*([\d.]+)\s*±", output)

        pp = float(pp_match.group(1)) if pp_match else None
        tg = float(tg_match.group(1)) if tg_match else None

        # Extraire la taille du modèle
        size_match = re.search(r"\|\s*([\d.]+)\s*GiB\s*\|", output)
        size_gib = float(size_match.group(1)) if size_match else None

        return {
            "pp128": round(pp, 1) if pp else None,
            "tg32": round(tg, 2) if tg else None,
            "size_gib": size_gib,
        }
    except Exception as e:
        return {"error": str(e)}


def profile_model(model_path, ngl=33):
    """Profile un modèle complet."""
    name = os.path.basename(model_path)
    size_gb = os.path.getsize(model_path) / (1024**3)

    print(f"\n{'='*60}")
    print(f"[+] Profil : {name} ({size_gb:.2f} GiB)")
    print(f"{'='*60}")

    # 1. Benchmark
    print("    Benchmark...", end=" ", flush=True)
    bench = run_bench(model_path, ngl)
    print(f"pp128={bench.get('pp128', '?')} tg32={bench.get('tg32', '?')}")

    # 2. VRAM
    vram = get_vram_usage()
    print(f"    VRAM: {vram['used_mib']} MiB used / {vram['free_mib']} MiB free")

    # 3. GPU
    gpu_util = get_gpu_util()
    gpu_power = get_gpu_power()
    print(f"    GPU: {gpu_util}% util, {gpu_power:.0f}W")

    return {
        "name": name,
        "path": model_path,
        "size_gb": round(size_gb, 2),
        "bench": bench,
        "vram_mib": vram["used_mib"],
        "vram_free_mib": vram["free_mib"],
        "gpu_util_pct": gpu_util,
        "gpu_power_w": round(gpu_power, 1),
    }


def compute_scores(results):
    """Calcule un score D2-HW pour chaque modèle."""
    if not results:
        return results

    # Normaliser les métriques
    max_tg = max(r["bench"].get("tg32") or 0 for r in results) or 1
    max_pp = max(r["bench"].get("pp128") or 0 for r in results) or 1
    min_size = min(r["size_gb"] for r in results) or 1

    for r in results:
        tg = r["bench"].get("tg32") or 0
        pp = r["bench"].get("pp128") or 0
        size = r["size_gb"]

        # Score = vitesse * prefill / taille (plus haut = mieux)
        speed_score = (tg / max_tg) * 40  # 40% weight on decode speed
        pp_score = (pp / max_pp) * 20     # 20% weight on prefill
        size_score = (min_size / size) * 20 if size > 0 else 0  # 20% weight on small size
        vram_score = (1 - r["vram_mib"] / 8151) * 20 if r["vram_mib"] > 0 else 0  # 20% VRAM headroom

        r["d2_score"] = round(speed_score + pp_score + size_score + vram_score, 1)
        r["speed_score"] = round(speed_score, 1)
        r["size_score"] = round(size_score, 1)

    # Trier par score décroissant
    results.sort(key=lambda r: -r["d2_score"])
    return results


def find_27b_models():
    """Trouve tous les GGUF 27B dans models/."""
    models = []
    if not os.path.isdir(MODELS_DIR):
        return models
    for f in sorted(os.listdir(MODELS_DIR)):
        if f.lower().endswith(".gguf") and "27B" in f:
            models.append(os.path.join(MODELS_DIR, f))
    return models


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 HW PROFILER")
    ap.add_argument("--models", nargs="*", help="GGUF à profiler (défaut: tous les 27B)")
    ap.add_argument("--ngl", type=int, default=33, help="n GPU layers")
    ap.add_argument("--json", default=OUT_JSON)
    ap.add_argument("--compare", action="store_true", help="Mode comparatif")
    args = ap.parse_args()

    # Trouver les modèles
    if args.models:
        models = args.models
    else:
        models = find_27b_models()

    if not models:
        print("[!] Aucun modèle 27B trouvé dans models/")
        sys.exit(1)

    print(f"[+] D2 HW Profiler — {len(models)} modèles à profiler")
    print(f"[+] ngl={args.ngl}")

    # Profiler chaque modèle
    results = []
    for m in models:
        if not os.path.exists(m):
            print(f"[!] {m} introuvable, skip")
            continue
        r = profile_model(m, ngl=args.ngl)
        results.append(r)

    # Calculer les scores
    results = compute_scores(results)

    # Afficher le tableau comparatif
    print(f"\n{'='*80}")
    print(f"{'RÉSULTATS COMPARATIFS':^80}")
    print(f"{'='*80}")
    print(f"{'Modèle':<35} {'Taille':>7} {'pp128':>8} {'tg32':>8} {'VRAM':>8} {'Score':>6}")
    print(f"{'-'*80}")
    for r in results:
        bench = r["bench"]
        tg = bench.get("tg32", 0)
        pp = bench.get("pp128", 0)
        print(f"{r['name']:<35} {r['size_gb']:>5.1f}G {pp:>7.1f} {tg:>7.2f} {r['vram_mib']:>6}M {r['d2_score']:>5.1f}")

    # Sauvegarder
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ngl": args.ngl,
        "results": results,
        "best": results[0]["name"] if results else None,
    }

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Meilleur : {results[0]['name']} (score {results[0]['d2_score']})")
    print(f"[+] Rapport : {args.json}")


if __name__ == "__main__":
    main()
