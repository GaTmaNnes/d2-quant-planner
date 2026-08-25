#!/usr/bin/env python3
# [OBSOLETE 25/08/2026] Périmètre 27B abandonné (prod = Qwen3.6-35B-A3B-D2-MOE).
# Raisons : cible Qwen3.8-27B inexistante ; --override-tensor inexistant dans ce fork.
# Pour ssmfix : table GGML_ID_TO_NAME contient un SWAP dangereux (7 vs 8) — NE PAS
# EXÉCUTER sans correction. (Table corrigée ci-dessous le 25/08/2026 pour
# neutraliser le danger : 7=Q5_1, 8=Q8_0 conformes ggml.h.)
# -*- coding: utf-8 -*-
"""
D2 BUILD ECO-SSMFIX — test #5 du bilan (22/08/2026)
====================================================
Le KLD "réel" (profil) montrait ssm_alpha/ssm_beta en Q2_K (SNR ~0, cosine ≈ 0,
tensors "détruits" : in_proj_a / in_proj_b, 245 760 éléments chacun, ~20 Mo au
total pour les 64 couches). L'hypothèse : les passer en q8_0 ne coûte presque
rien en VRAM (~+17 Mo) et devrait améliorer la qualité (PPL).

Méthode :
  1. Dériver le fichier tensor-types depuis le GGUF D2-ECO ACTUEL (tous les
     tensors avec leur type réel, y compris MTP blk.64, token_embd, output).
  2. Overrider ssm_alpha / ssm_beta -> q8_0.
  3. Re-quantifier depuis models/Qwen3.8-27B-Q4_K_M.gguf (seul GGUF de haute
     fidélité encore présent : ssm_alpha/beta y sont en Q4_K, valeurs saines).
     --allow-requantize requis (source déjà quantifiée).
  4. (optionnel) --control : reconstruit aussi D2-ECO-req (types exacts D2-ECO
     sans fix) pour isoler l'effet du fix dans l'A/B PPL.

Usage :
  python d2_build_eco_ssmfix.py --dry-run
  python d2_build_eco_ssmfix.py [--control]
"""
import argparse
import os
import subprocess
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D2ECO = os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO.gguf")
SOURCE = os.path.join(HERE, "models", "Qwen3.8-27B-Q4_K_M.gguf")
OUT_FIX = os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO-ssmfix.gguf")
OUT_CTRL = os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO-req.gguf")
QUANTIZE = os.path.join(HERE, "llama-quantize.exe")
TYPES_FIX = os.path.join(HERE, "d2_tensor_types_eco_ssmfix.txt")
TYPES_CTRL = os.path.join(HERE, "d2_tensor_types_eco_req.txt")

# ggml_type id -> nom llama.cpp
# [CORRIGÉ 25/08/2026] ancienne table contenait un SWAP dangereux (7="q8_0",
# 8="q5_0") — réel ggml.h : 7=Q5_1, 8=Q8_0. Table complète corrigée.
GGML_NAME = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1",
             6: "q5_0", 7: "q5_1", 8: "q8_0", 9: "q8_1",
             10: "q2_K", 11: "q3_K", 12: "q4_K", 13: "q5_K",
             14: "q6_K", 15: "q8_K", 20: "iq4_nl", 22: "iq2_s",
             23: "iq4_xs", 30: "bf16"}


def derive_types(gguf_path):
    """Lit le GGUF et retourne {nom_tensor: nom_type} pour TOUS les tensors."""
    from gguf import GGUFReader
    r = GGUFReader(gguf_path)
    out = {}
    for t in r.tensors:
        tid = t.tensor_type
        nm = GGML_NAME.get(tid)
        if nm is None:
            print(f"  [!] type inconnu {tid} pour {t.name} — ignoré")
            continue
        out[t.name] = nm
    return out


def write_types(path, types):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in sorted(types.items()):
            f.write(f"{k}={v}\n")
    c = Counter(types.values())
    print(f"  [+] {path} : {len(types)} tensors -> {dict(c)}")


def quantize(source, output, types_file, base="q4_K_M"):
    if not os.path.isfile(QUANTIZE):
        print(f"[!] {QUANTIZE} introuvable")
        sys.exit(1)
    cmd = [QUANTIZE, source, output, base, "--tensor-type-file", types_file,
           "--allow-requantize"]
    print(f"[+] CMD: {' '.join(cmd)}")
    return cmd


def main():
    ap = argparse.ArgumentParser(description="D2 build ECO-ssmfix (test #5)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--control", action="store_true",
                    help="reconstruit aussi D2-ECO-req (sans fix) pour A/B propre")
    ap.add_argument("--skip-quantize", action="store_true", help="génère les fichiers mais ne quantize pas")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not os.path.isfile(D2ECO):
        sys.exit(f"[!] {D2ECO} introuvable")
    if not os.path.isfile(SOURCE):
        sys.exit(f"[!] Source {SOURCE} introuvable")

    print(f"[1] Dérivation des types depuis {os.path.basename(D2ECO)} ...")
    types = derive_types(D2ECO)
    n_ssm = sum(1 for k in types if "ssm_alpha" in k or "ssm_beta" in k)
    print(f"    {len(types)} tensors dérivés, {n_ssm} ssm_alpha/ssm_beta en "
          f"{Counter(v for k, v in types.items() if 'ssm_alpha' in k or 'ssm_beta' in k)}")

    # Fix : ssm_alpha / ssm_beta -> q8_0
    types_fix = dict(types)
    for k in list(types_fix):
        if "ssm_alpha" in k or "ssm_beta" in k:
            types_fix[k] = "q8_0"
    c = Counter(types_fix.values())
    print(f"    après fix : {dict(c)}")

    write_types(TYPES_FIX, types_fix)
    if args.control:
        write_types(TYPES_CTRL, types)

    if args.skip_quantize or args.dry_run:
        print("\n[dry-run] quantize serait :")
        print("   ", " ".join(quantize(SOURCE, OUT_FIX, TYPES_FIX)))
        if args.control:
            print("   ", " ".join(quantize(SOURCE, OUT_CTRL, TYPES_CTRL)))
        return

    # Quantize (un à la fois — le GPU/CPU est utilisé par les sweeps)
    print("\n[2] Quantization D2-ECO-ssmfix ...")
    if not args.dry_run:
        cmd = quantize(SOURCE, OUT_FIX, TYPES_FIX)
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=3600)
        print(r.stdout[-400:] if r.stdout else "")
        if r.returncode != 0:
            print(f"[!] échec : {r.stderr[-600:]}")
            sys.exit(1)
        print(f"    -> {OUT_FIX} "
              f"({os.path.getsize(OUT_FIX)/(1024**3):.2f} GiB)")

    if args.control:
        print("\n[3] Quantization D2-ECO-req (contrôle) ...")
        cmd = quantize(SOURCE, OUT_CTRL, TYPES_CTRL)
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=3600)
        print(r.stdout[-400:] if r.stdout else "")
        if r.returncode != 0:
            print(f"[!] échec : {r.stderr[-600:]}")
            sys.exit(1)
        print(f"    -> {OUT_CTRL} "
              f"({os.path.getsize(OUT_CTRL)/(1024**3):.2f} GiB)")


if __name__ == "__main__":
    main()
