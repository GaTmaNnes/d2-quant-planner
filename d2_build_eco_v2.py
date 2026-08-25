#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 BUILD ECO-V2 — construire un GGUF mixed precision optimisé
==============================================================

Stratégie D2-ECO-v2 :
  - Modèle de base : FP8 safetensors (models/Qwen3.8-27B-FP8/)
  - Cible : ~12.5 GiB (légèrement plus gros que D2-ECO pour plus de qualité)
  - Attention layers (16) : Q5_K (qualité supérieure pour le routing)
  - Linear attention/GDN layers (48) : Q4_K (standard)
  - FFN down (les plus gros) : Q4_K
  - Norms/scales : F16 (pas de quantification)
  - token_embd/output : Q8_0

Usage:
  python d2_build_eco_v2.py
  python d2_build_eco_v2.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FP8_DIR = os.path.join(HERE, "models", "Qwen3.8-27B-FP8")
QUANTIZE_EXE = os.path.join(HERE, "tools", "llama-quantize.exe")
OUTPUT = os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO-v2.gguf")
TENSOR_TYPE_FILE = os.path.join(HERE, "d2_eco_v2_tensor_types.txt")


def load_config():
    """Charge la config du modèle pour connaître la structure des layers."""
    cfg_path = os.path.join(FP8_DIR, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", {})
    return {
        "n_layers": tc.get("num_hidden_layers", 64),
        "layer_types": tc.get("layer_types", []),
        "hidden": tc.get("hidden_size", 5120),
        "ffn": tc.get("intermediate_size", 17408),
        "n_heads": tc.get("num_attention_heads", 24),
        "kv_heads": tc.get("num_key_value_heads", 4),
        "head_dim": tc.get("head_dim", 256),
    }


def generate_tensor_types(config):
    """Génère la liste des tensors avec leur précision cible.
    
    Stratégie D2-ECO-v2 :
      - attention layers : Q5_K (routing = qualité cruciale)
      - linear attention (GDN) : Q4_K (standard)
      - FFN down/up/gate : Q4_K (les plus gros tensors)
      - norms : F16 (pas de quantification)
      - token_embd : Q8_0
      - output.weight : Q8_0
    """
    lines = []
    n_layers = config["n_layers"]
    layer_types = config["layer_types"]

    for i in range(n_layers):
        lt = layer_types[i] if i < len(layer_types) else "linear_attention"
        is_attn = lt == "full_attention"

        # Attention tensors
        # [CORRIGÉ 25/08/2026] format "nom=TYPE" (parse_tensor_type exige '=',
        # l'espace rendait les lignes invalides) + noms réels qwen3.5-next.
        for tname in ["attn_q", "attn_k", "attn_v", "attn_o"]:
            prec = "Q5_K" if is_attn else "Q4_K"
            lines.append(f"blk.{i}.{tname}.weight={prec}")

        # FFN tensors (toujours Q4_K)
        for tname in ["ffn_gate", "ffn_up", "ffn_down"]:
            lines.append(f"blk.{i}.{tname}.weight=Q4_K")

        # Norms (F16) — pas de .bias dans cette archi (anciens noms inventés)
        for tname in ["attn_norm", "ffn_norm"]:
            lines.append(f"blk.{i}.{tname}.weight=F16")

        # SSM/GDN : préfixes réels qwen3.5-next (linear_attn_qkv, linear_attn_ba,
        # linear_attn_out, ssm_conv1d, ssm_dt, ssm_norm). Les anciens noms
        # (ssm_alpha, ssm_beta, linear_attn_q/k/v/o, .bias) n'existent pas.
        # TENSOR_NOT_REQUIRED : entrée ignorée si le tenseur est absent du GGUF.
        for tname in ["ssm_conv1d", "ssm_dt", "ssm_norm",
                       "linear_attn_qkv", "linear_attn_ba", "linear_attn_out"]:
            lines.append(f"blk.{i}.{tname}=Q4_K")

    # Global tensors
    # [CORRIGÉ 25/08/2026] format "nom=TYPE"
    lines.append("token_embd.weight=Q8_0")
    lines.append("output.weight=Q8_0")

    return lines


def find_source_gguf():
    """Trouve le GGUF source FP8 pour la quantization."""
    # Chercher un GGUF FP8 dans le dossier
    for f in os.listdir(FP8_DIR):
        if f.endswith(".gguf"):
            return os.path.join(FP8_DIR, f)
    
    # Sinon, utiliser le premier GGUF trouvé comme source
    # (llama-quantize peut re-quantizer depuis un GGUF existant)
    for f in os.listdir(os.path.join(HERE, "models")):
        if f.endswith(".gguf") and "27B" in f:
            return os.path.join(HERE, "models", f)
    
    return None


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 BUILD ECO-V2")
    ap.add_argument("--source", help="GGUF source (défaut: auto-detect)")
    ap.add_argument("--output", default=OUTPUT)
    ap.add_argument("--dry-run", action="store_true", help="Générer le fichier sans quantizer")
    ap.add_argument("--ngl", type=int, default=33, help="ngl pour le benchmark post-rebuild")
    args = ap.parse_args()

    # 1. Config
    config = load_config()
    print(f"[+] Modèle : {config['n_layers']} layers ({sum(1 for t in config['layer_types'] if t=='full_attention')} attn, {sum(1 for t in config['layer_types'] if t=='linear_attention')} GDN)")

    # 2. Générer le fichier tensor-types
    tensor_types = generate_tensor_types(config)
    with open(TENSOR_TYPE_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(tensor_types))
    print(f"[+] Tensor types : {len(tensor_types)} règles -> {TENSOR_TYPE_FILE}")

    # 3. Afficher la répartition
    from collections import Counter
    counts = Counter(line.split()[-1] for line in tensor_types)
    print(f"    Répartition : {dict(counts)}")

    # 4. Trouver la source
    source = args.source or find_source_gguf()
    if not source:
        print("[!] Aucun GGUF source trouvé")
        sys.exit(1)
    print(f"[+] Source : {source}")

    if args.dry_run:
        print(f"\n[+] Dry run — commande serait :")
        print(f"    {QUANTIZE_EXE} {source} {args.output} Q4_K --tensor-type-file {TENSOR_TYPE_FILE}")
        return

    # 5. Quantize
    if not os.path.exists(QUANTIZE_EXE):
        print(f"[!] {QUANTIZE_EXE} introuvable")
        sys.exit(1)

    cmd = [
        QUANTIZE_EXE,
        source,
        args.output,
        "Q4_K",  # base quantization (overridden by tensor-type-file)
        "--tensor-type-file", TENSOR_TYPE_FILE,
    ]

    print(f"\n[+] Quantization en cours...")
    print(f"    Commande : {' '.join(cmd[:5])} ...")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,
            encoding="utf-8", errors="replace"
        )
        print(result.stdout[-500:] if result.stdout else "(pas de stdout)")
        if result.returncode != 0:
            print(f"[!] Erreur : {result.stderr[-500:] if result.stderr else ''}")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("[!] Timeout (>30 min)")
        sys.exit(1)

    # 6. Vérifier le résultat
    if os.path.exists(args.output):
        size_gb = os.path.getsize(args.output) / (1024**3)
        print(f"\n[+] D2-ECO-v2 créé : {args.output}")
        print(f"    Taille : {size_gb:.2f} GiB")

        # 7. Benchmark
        import re
        cmd_bench = [
            os.path.join(HERE, "llama-bench.exe"),
            "-m", args.output,
            "-ngl", str(args.ngl),
            "-n", "32", "-p", "128", "-t", "8"
        ]
        print(f"\n[+] Benchmark...")
        try:
            r = subprocess.run(cmd_bench, capture_output=True, text=True, timeout=120,
                             encoding="utf-8", errors="replace")
            m = re.search(r"\|\s*tg32\s*\|\s*([\d.]+)\s*±", r.stdout)
            if m:
                tg = float(m.group(1))
                print(f"    D2-ECO-v2 tg32 = {tg:.2f} t/s")
                print(f"    D2-ECO     tg32 = 7.83 t/s")
                delta = ((tg / 7.83) - 1) * 100
                print(f"    Delta : {delta:+.1f}%")
        except Exception as e:
            print(f"    Benchmark échoué : {e}")
    else:
        print("[!] Fichier de sortie non créé")


if __name__ == "__main__":
    main()
