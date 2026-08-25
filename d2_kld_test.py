#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 KLD TEST — divergence KL reelle par variante GGUF, contre une reference.
============================================================================
Comble l'axe "KLD tensor/layer reel" identifie manquant dans la couverture D2 :
jusqu'ici seule la PPL etait mesuree par variante, pas la vraie divergence de
distribution token par token contre une reference haute precision.

N'implemente AUCUN calcul KLD maison : utilise le support natif de
llama-perplexity (--kl-divergence-base / --kl-divergence), deja verifie
fonctionnel (auto-test reference-contre-elle-meme -> KLD ~0).

Principe :
  1. --reference genere les logits complets sur le corpus (--save-all-logits)
  2. chaque --candidate est ensuite compare a ces logits (--kl-divergence)
  3. parse la sortie native de llama-perplexity, agrege en JSON + tableau

Exemples :
  python d2_kld_test.py --reference models/Qwen3.8-27B-hybrid.gguf ^
                         --candidate models/Qwen3.8-27B-Q4_K_M.gguf ^
                         --candidate models/Qwen3.8-27B-D2-BALANCED-IM.gguf
  python d2_kld_test.py --reference models/ref.gguf --candidates-dir models
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PPL_EXE = os.path.join(HERE, "llama-perplexity.exe")
CORPUS = os.path.join(HERE, "corpus", "wiki.test.raw")
OUT_JSON = os.path.join(HERE, "d2_kld_report.json")

STAT_PATTERNS = {
    "ppl_q":        r"Mean PPL\(Q\)\s*:\s*([\-\d.]+)\s*±\s*([\-\d.]+)",
    "ppl_base":     r"Mean PPL\(base\)\s*:\s*([\-\d.]+)\s*±\s*([\-\d.]+)",
    "mean_kld":     r"Mean\s+KLD:\s*([\-\d.]+)\s*±\s*([\-\d.]+)",
    "max_kld":      r"Maximum KLD:\s*([\-\d.]+)",
    "median_kld":   r"Median\s+KLD:\s*([\-\d.]+)",
    "mean_dp_pct":  r"Mean\s+Δp:\s*([\-\d.]+)\s*±\s*([\-\d.]+)\s*%",
    "rms_dp_pct":   r"RMS Δp\s*:\s*([\-\d.]+)\s*±\s*([\-\d.]+)\s*%",
    "same_top_p":   r"Same top p:\s*([\-\d.]+)\s*±\s*([\-\d.]+)\s*%",
}


def run_ppl(args_list, timeout=1800):
    p = subprocess.run([PPL_EXE] + args_list, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def parse_stats(output):
    stats = {}
    for key, pat in STAT_PATTERNS.items():
        m = re.search(pat, output)
        if not m:
            continue
        groups = m.groups()
        if len(groups) == 2:
            stats[key] = {"value": float(groups[0]), "stderr": float(groups[1])}
        else:
            stats[key] = {"value": float(groups[0])}
    return stats


def gen_reference_logits(reference, corpus, chunks, ctx, logits_path):
    print(f"[1/2] Génération des logits de référence : {os.path.basename(reference)}")
    args = ["-m", reference, "-f", corpus, "-c", str(ctx), "-fitt", "1024", "-fa", "1",
            "--save-all-logits", logits_path]
    if chunks:
        args += ["--chunks", str(chunks)]
    rc, out = run_ppl(args)
    if rc != 0:
        print(out[-3000:])
        sys.exit(f"[!] Échec génération logits référence (exit={rc})")
    if not os.path.isfile(logits_path):
        sys.exit("[!] Fichier de logits non produit — voir sortie ci-dessus.")
    print(f"    -> {logits_path} ({os.path.getsize(logits_path)/1e6:.0f} Mo)")


def test_candidate(candidate, corpus, chunks, ctx, logits_path):
    name = os.path.basename(candidate)
    print(f"[2/2] KLD vs référence : {name}")
    args = ["-m", candidate, "-f", corpus, "-c", str(ctx), "-fitt", "1024", "-fa", "1",
            "--kl-divergence-base", logits_path, "--kl-divergence"]
    if chunks:
        args += ["--chunks", str(chunks)]
    rc, out = run_ppl(args)
    if rc != 0:
        print(f"    [!] échec (exit={rc})")
        return {"model": name, "error": out[-2000:]}
    stats = parse_stats(out)
    if not stats:
        print("    [!] sortie non parsée — format inattendu, voir d2_kld_report.json[raw]")
    return {"model": name, "path": candidate, "stats": stats, "raw": out[-4000:]}


def main():
    ap = argparse.ArgumentParser(description="D2 KLD Test — divergence KL reelle par variante GGUF")
    ap.add_argument("--reference", required=True, help="GGUF de référence (le plus proche du FP8 original)")
    ap.add_argument("--candidate", action="append", default=[], help="GGUF à tester (répétable)")
    ap.add_argument("--candidates-dir", default=None,
                     help="dossier : teste tous les .gguf trouvés (hors référence et imatrix)")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--chunks", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--json", default=OUT_JSON)
    ap.add_argument("--keep-logits", action="store_true",
                     help="ne pas supprimer le fichier de logits de référence après usage (volumineux)")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not os.path.isfile(PPL_EXE):
        sys.exit(f"[!] {PPL_EXE} introuvable.")
    if not os.path.isfile(args.reference):
        sys.exit(f"[!] Référence introuvable : {args.reference}")
    if not os.path.isfile(args.corpus):
        sys.exit(f"[!] Corpus introuvable : {args.corpus}")

    candidates = list(args.candidate)
    if args.candidates_dir:
        ref_abs = os.path.abspath(args.reference)
        for f in sorted(glob.glob(os.path.join(args.candidates_dir, "*.gguf"))):
            fl = f.lower()
            if "imatrix" in fl or os.path.abspath(f) == ref_abs:
                continue
            candidates.append(f)

    if not candidates:
        sys.exit("[!] Aucun candidat (--candidate ou --candidates-dir).")

    for c in candidates:
        if not os.path.isfile(c):
            sys.exit(f"[!] Candidat introuvable : {c}")

    logits_fd, logits_path = tempfile.mkstemp(suffix=".kld", prefix="d2_ref_")
    os.close(logits_fd)
    try:
        gen_reference_logits(args.reference, args.corpus, args.chunks, args.ctx, logits_path)

        results = []
        for c in candidates:
            results.append(test_candidate(c, args.corpus, args.chunks, args.ctx, logits_path))
    finally:
        if not args.keep_logits and os.path.exists(logits_path):
            os.remove(logits_path)

    print("\n" + "=" * 100)
    print(f"  D2 KLD TEST — référence : {os.path.basename(args.reference)}")
    print("=" * 100)
    print(f"  {'Modèle':<40} {'PPL':>10} {'Mean KLD':>12} {'Max KLD':>12} {'Same top-1 %':>14}")
    print("-" * 100)
    for r in results:
        s = r.get("stats", {})
        ppl = s.get("ppl_q", {}).get("value")
        mkld = s.get("mean_kld", {}).get("value")
        xkld = s.get("max_kld", {}).get("value")
        top1 = s.get("same_top_p", {}).get("value")
        row = (f"  {r['model']:<40} "
               f"{ppl if ppl is not None else '?':>10} "
               f"{mkld if mkld is not None else '?':>12} "
               f"{xkld if xkld is not None else '?':>12} "
               f"{top1 if top1 is not None else '?':>14}")
        print(row)
    print("=" * 100)

    payload = {
        "reference": args.reference,
        "corpus": args.corpus,
        "chunks": args.chunks,
        "ctx": args.ctx,
        "results": results,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[+] -> {args.json}")


if __name__ == "__main__":
    main()
