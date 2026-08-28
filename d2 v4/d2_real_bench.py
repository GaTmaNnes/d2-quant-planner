#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 REAL BENCH — mesure RÉELLE des performances du PC (aucune simulation).
=========================================================================
Lance llama-bench.exe (binaire llama.cpp compilé pour ce GPU) sur un modèle
GGUF réel et lit la VRAM via nvidia-smi. Aucun chiffre n'est inventé :
  - tok/s prompt (pp) et génération (tg) = sortie réelle de llama-bench
    (CUDA events, warmup + répétitions, écart-type inclus)
  - VRAM chargée = nvidia-smi avant/pendant/après

Usage :
  python d2_real_bench.py --model <fichier.gguf> [--pp 512] [--tg 128] [--out rapport.json]
  python d2_real_bench.py --model <fichier.gguf> --all        # baseline + ctk/ctv variantes
  python d2_real_bench.py --detect                             # GPU + modèle dispo

Contrairement à d2_profiler.py (latences = m*n*bpe/1000, prédictions codées
en dur), d2_rtx_gguf_profiler.py --benchmark (GEMM isolés FP16 simulés) ou
d2_transport_profiler.py (octets/token déduits des shapes), ce script mesure
le modèle COMPLET en conditions réelles sur le GPU réel.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))   # .../d2 v3/d2 v3
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))  # racine du projet
LMSTUDIO_MODELS = os.path.expandvars(r"%USERPROFILE%\.lmstudio\models")


def find_exe():
    """Cherche llama-bench.exe : dossier racine, build D:, PATH."""
    candidates = [
        os.path.join(PROJECT_ROOT, "llama-bench.exe"),
        os.path.join(PROJECT_ROOT, "llama.cpp", "build", "bin", "Release", "llama-bench.exe"),
        r"D:\llama-build\bin\Release\llama-bench.exe",
        "llama-bench",
    ]
    for c in candidates:
        if shutil.which(c) if os.sep not in c else os.path.exists(c):
            return c
        if os.path.exists(c):
            return c
    return None


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        if not out:
            return None
        name, mem_total, mem_used, drv = [x.strip() for x in out[0].split(",")]
        return {"name": name, "vram_total_mb": int(float(mem_total)),
                "vram_used_mb": int(float(mem_used)), "driver": drv}
    except Exception:
        return None


def vram_used_mb():
    g = gpu_info()
    return g["vram_used_mb"] if g else None


def parse_bench_output(text):
    """Parse la table de llama-bench (lignes de données uniquement).
    Format réel :  | qwen35 9B Q4_K | 5.37 GiB | 9.20 B | CUDA | 999 | pp64 | 557.31 ± 14.26 |
    (avec -ctk/-ctv, deux colonnes type_k/type_v s'intercalent avant le test).
    """
    rows = []
    for line in text.splitlines():
        if "|" not in line or "----" in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]
        if len(cells) < 7 or cells[0].startswith("model"):
            continue
        # repère la colonne test (pp<n> ou tg<n>) et la valeur t/s qui la suit
        m = re.search(r"\|\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)", line)
        if not m:
            continue
        row = {"model": cells[0], "size": cells[1], "params": cells[2],
               "backend": cells[3], "ngl": cells[4]}
        # type_k/type_v éventuels entre ngl et test
        extra = [c for c in cells[5:-2] if not c.lower().startswith("pp") and not c.lower().startswith("tg")]
        if len(extra) >= 2 and extra[0] != row["ngl"]:
            row["type_k"] = extra[-2]
            row["type_v"] = extra[-1]
        row["test"] = m.group(1)
        row["tps"] = m.group(2)
        row["tps_std"] = m.group(3)
        rows.append(row)
    return rows


def run_llama_bench(exe, model, extra_args, pp, tg, timeout):
    vram0 = vram_used_mb()
    cmd = [exe, "-m", model, "-p", str(pp), "-n", str(tg), "-ngl", "999"] + extra_args
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout)
        text = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return {"error": f"timeout après {timeout}s", "cmd": cmd}
    wall = time.time() - t0
    vram1 = vram_used_mb()
    rows = parse_bench_output(text)
    return {"cmd": [os.path.basename(exe)] + [str(x) for x in cmd[1:]],
            "wall_s": round(wall, 1), "vram_before_mb": vram0,
            "vram_after_mb": vram1, "rows": rows, "raw": text[-4000:]}


def main():
    ap = argparse.ArgumentParser(description="D2 REAL BENCH — perf réelles du PC (llama-bench + nvidia-smi)")
    ap.add_argument("--model", help="chemin du GGUF à mesurer")
    ap.add_argument("--detect", action="store_true", help="GPU + modèle dispo, sans bench")
    ap.add_argument("--pp", type=int, default=512, help="tokens prompt (défaut 512)")
    ap.add_argument("--tg", type=int, default=128, help="tokens générés (défaut 128)")
    ap.add_argument("--all", action="store_true", help="baseline + KV q8_0 + KV f16 variantes")
    ap.add_argument("--out", default=os.path.join(HERE, "d2_real_bench_report.json"))
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    exe = find_exe()
    if exe is None:
        sys.exit("[!] llama-bench.exe introuvable — compiler d'abord (cmake --build . --target llama-bench)")
    gpu = gpu_info()

    if args.detect:
        print(f"GPU     : {gpu}")
        print(f"llama-bench : {exe}")
        seen = set()
        print(f"Modèles GGUF (projet {PROJECT_ROOT} + LM Studio) :")
        for base in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "models"),
                     os.path.join(PROJECT_ROOT, "d2 v3"), LMSTUDIO_MODELS):
            if not os.path.isdir(base):
                continue
            for dirpath, _dirs, files in os.walk(base):
                for f in sorted(files):
                    if f.lower().endswith(".gguf") and f not in seen:
                        p = os.path.join(dirpath, f)
                        seen.add(f)
                        print(f"  - {p}  ({os.path.getsize(p)/1e9:.2f} GB)")
        return

    if not args.model:
        # détection auto : premier .gguf valide (projet, puis LM Studio)
        for base in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "models"),
                     os.path.join(PROJECT_ROOT, "d2 v3"), LMSTUDIO_MODELS):
            if not os.path.isdir(base):
                continue
            for dirpath, _dirs, files in os.walk(base):
                for f in sorted(files):
                    if f.lower().endswith(".gguf"):
                        args.model = os.path.join(dirpath, f)
                        break
                if args.model:
                    break
            if args.model:
                break
    if not args.model or not os.path.exists(args.model):
        sys.exit(f"[!] Modèle introuvable : {args.model}")

    print("=" * 78)
    print("  D2 REAL BENCH — performances réelles du PC (aucune simulation)")
    print("=" * 78)
    print(f"GPU        : {gpu['name']} | {gpu['vram_total_mb']} MiB | driver {gpu['driver']}")
    print(f"llama-bench: {exe}")
    print(f"Modèle     : {args.model} ({os.path.getsize(args.model)/1e9:.2f} GB)")
    print(f"Tests      : pp{args.pp} (prompt) + tg{args.tg} (génération), ngl=999\n")

    configs = [("BASELINE (KV défaut f16)", [])]
    if args.all:
        configs += [
            ("KV q8_0", ["-ctk", "q8_0", "-ctv", "q8_0"]),
            ("KV f16 explicite", ["-ctk", "f16", "-ctv", "f16"]),
        ]

    results = []
    for label, extra in configs:
        print(f"--- {label} ---")
        r = run_llama_bench(exe, args.model, extra, args.pp, args.tg, args.timeout)
        r["label"] = label
        results.append(r)
        if "error" in r:
            print(f"  [!] {r['error']}")
        else:
            for row in r["rows"]:
                print(f"  {row['test']:>8} : {row['tps']:>10} t/s   "
                      f"(VRAM {r['vram_before_mb']} -> {r['vram_after_mb']} MiB, {r['wall_s']}s)")
        print()

    # Synthèse lisible
    print("=" * 78)
    print("  SYNTHÈSE")
    print("=" * 78)
    hdr = f"  {'Config':<24} {'pp t/s':>10} {'tg t/s':>10} {'VRAM après':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        if "error" in r:
            print(f"  {r['label']:<24} ÉCHEC")
            continue
        pp = tg = "-"
        for row in r["rows"]:
            if row["test"].startswith("pp"):
                pp = row["tps"]
            if row["test"].startswith("tg"):
                tg = row["tps"]
        print(f"  {r['label']:<24} {pp:>10} {tg:>10} {str(r['vram_after_mb'])+' MiB':>12}")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "llama-bench (mesure réelle CUDA) + nvidia-smi",
        "gpu": gpu,
        "model": args.model,
        "model_gb": round(os.path.getsize(args.model) / 1e9, 2),
        "pp": args.pp, "tg": args.tg,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Rapport : {args.out}")


if __name__ == "__main__":
    main()
