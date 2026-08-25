#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 PPL SWEEP — mesure la perplexite reelle de chaque variante GGUF d'un dossier.
==================================================================================
Comble l'axe "PPL par variante" identifie manquant : jusqu'ici chaque PPL etait
mesuree a la main, une commande llama-perplexity a la fois, sans tableau
comparatif automatique ni export JSON reutilisable par d2_report/d2_precision_optimizer.

Exemples :
  python d2_ppl_sweep.py --models-dir models
  python d2_ppl_sweep.py --model models/Qwen3.8-27B-Q4_K_M.gguf --model models/D2-ECO.gguf
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PPL_EXE = os.path.join(HERE, "llama-perplexity.exe")
CORPUS = os.path.join(HERE, "corpus", "wiki.test.raw")
OUT_JSON = os.path.join(HERE, "d2_ppl_sweep_report.json")

FINAL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")


def run_one(model, corpus, chunks, ctx, timeout):
    args = [PPL_EXE, "-m", model, "-f", corpus, "-c", str(ctx), "-fitt", "1024", "-fa", "1"]
    if chunks:
        args += ["--chunks", str(chunks)]
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"model": os.path.basename(model), "path": model, "error": "timeout"}
    out = p.stdout + p.stderr
    if p.returncode != 0:
        return {"model": os.path.basename(model), "path": model,
                "error": f"exit={p.returncode}", "tail": out[-1500:]}
    m = FINAL_RE.search(out)
    if not m:
        return {"model": os.path.basename(model), "path": model,
                "error": "sortie non parsée", "tail": out[-1500:]}
    return {
        "model": os.path.basename(model), "path": model,
        "size_gb": round(os.path.getsize(model) / (1024 ** 3), 2),
        "ppl": float(m.group(1)), "ppl_stderr": float(m.group(2)),
    }


def main():
    ap = argparse.ArgumentParser(description="D2 PPL Sweep — perplexite reelle sur plusieurs GGUF")
    ap.add_argument("--model", action="append", default=[], help="GGUF à tester (répétable)")
    ap.add_argument("--models-dir", default=None, help="teste tous les .gguf du dossier (hors imatrix)")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--chunks", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=1800, help="secondes max par modèle (défaut 30 min)")
    ap.add_argument("--json", default=OUT_JSON)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not os.path.isfile(PPL_EXE):
        sys.exit(f"[!] {PPL_EXE} introuvable.")
    if not os.path.isfile(args.corpus):
        sys.exit(f"[!] Corpus introuvable : {args.corpus}")

    models = list(args.model)
    if args.models_dir:
        for f in sorted(glob.glob(os.path.join(args.models_dir, "*.gguf"))):
            if "imatrix" not in f.lower():
                models.append(f)
    if not models:
        sys.exit("[!] Aucun modèle (--model ou --models-dir).")

    results = []
    for i, m in enumerate(models, 1):
        if not os.path.isfile(m):
            print(f"[{i}/{len(models)}] [!] introuvable, ignoré : {m}")
            continue
        print(f"[{i}/{len(models)}] {os.path.basename(m)}...")
        r = run_one(m, args.corpus, args.chunks, args.ctx, args.timeout)
        if "error" in r:
            print(f"    [!] {r['error']}")
        else:
            print(f"    PPL = {r['ppl']:.4f} ± {r['ppl_stderr']:.4f}  ({r['size_gb']} Go)")
        results.append(r)

    ok = [r for r in results if "ppl" in r]
    ok.sort(key=lambda r: r["ppl"])

    print("\n" + "=" * 90)
    print(f"  D2 PPL SWEEP — corpus : {os.path.basename(args.corpus)}, {args.chunks} chunks, ctx {args.ctx}")
    print("=" * 90)
    print(f"  {'Modèle':<45} {'Taille':>9} {'PPL':>12}")
    print("-" * 90)
    for r in ok:
        print(f"  {r['model']:<45} {r['size_gb']:>7.2f} Go {r['ppl']:>10.4f} ± {r['ppl_stderr']:.4f}")
    failed = [r for r in results if "ppl" not in r]
    if failed:
        print("-" * 90)
        for r in failed:
            print(f"  {r['model']:<45} ÉCHEC : {r.get('error')}")
    print("=" * 90)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({"corpus": args.corpus, "chunks": args.chunks, "ctx": args.ctx,
                    "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] -> {args.json}")


if __name__ == "__main__":
    main()
