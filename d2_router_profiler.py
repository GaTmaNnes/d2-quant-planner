#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 ROUTER PROFILER — profil comportemental du routeur MoE (Qwen3.5-35B-A3B).
=============================================================================
Répond à la question que le profilage des POIDS ne peut pas trancher :
« quels experts le modèle utilise-t-il réellement ? »

Deux modes :

  1. --forward  (référence, exige le modèle complet + un runtime qui peut le charger)
     Fait un forward du modèle sur un corpus de calibration, capture les logits
     du routeur (.mlp.gate, [B, S, 256]) par hook, puis agrège :
       - activation_count / activation_rate par expert
       - mean_router_prob (softmax quand sélectionné)
       - top1 / top8 counts
       - co-activation (top paires d'experts co-sélectionnés)

  2. --static  (immédiat, sans forward — lecture directe des gates safetensors)
     Calcule la norme L2 de chaque ligne de gate [256, 2048] sur les 40 couches.
     C'est un PROXY statique de l'importance d'un expert (pas la fréquence réelle),
     utile tant que le forward complet n'est pas possible.

Sorties :
  - d2_router_profiler_report.json  (couche/expert + stats + co-activation)
  - d2_router_static_report.json    (mode --static)

BLOCAGE CONNU (forward) :
  - le modèle complet (71,9 Go, 14 shards) doit être téléchargé ;
  - torch 2.13 CPU + 33,6 Go de RAM ne suffisent PAS pour charger 35B en BF16 (~72 Go)
    ni pour un forward raisonnable. Il faut soit un GPU avec assez de VRAM, soit
    >72 Go de RAM CPU, soit un runtime GGUF (llama.cpp) instrumenté pour exposer
    les logits du routeur (modification C++).

Usage :
  python d2_router_profiler.py --forward --model hf_weights_35b --corpus wiki.test.raw
    [CORRIGÉ 25/08/2026] corpus par défaut = wiki.test.raw (l'ancien
    corpus/test.txt n'existait pas → crash du mode --forward)
  python d2_router_profiler.py --static  --shard-dir hf_weights_35b
"""

import argparse
import json
import os
import struct
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD_DIR = os.path.join(HERE, "hf_weights_35b")
INDEX_PATH = os.path.join(SHARD_DIR, "model.safetensors.index.json")
OUT_FORWARD = os.path.join(HERE, "d2_router_profiler_report.json")
OUT_STATIC = os.path.join(HERE, "d2_router_static_report.json")

TOP_K = 8  # experts routés par token (config num_experts_per_tok = 8)


def _find_corpus():
    """[CORRIGÉ 25/08/2026] corpus/test.txt n'existe pas → wiki.test.raw.
    Cherche wiki.test.raw dans le dossier courant puis le dossier parent."""
    for cand in (os.path.join(HERE, "wiki.test.raw"),
                 os.path.join(os.path.dirname(HERE), "wiki.test.raw")):
        if os.path.exists(cand):
            return cand
    return os.path.join(HERE, "wiki.test.raw")


# ---------------------------------------------------------------------------
# MODE STATIC — norme des gates par expert (proxy d'importance, sans forward)
# ---------------------------------------------------------------------------
def _read_gate_tensor(path, hlen, info):
    s, e = info["data_offsets"]
    dt = info.get("dtype", "BF16")
    bpe = 4 if dt == "F32" else 2
    with open(path, "rb") as fh:
        fh.seek(8 + hlen + s)
        raw = fh.read((e - s))
    if dt == "F32":
        W = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        u = np.frombuffer(raw, dtype="<u2")
        W = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float32)
    return W.reshape(info["shape"])


def static_mode(shard_dir, index_path, out_path):
    print("=" * 100)
    print("  D2 ROUTER PROFILER — mode STATIC (norme des gates, proxy d'importance)")
    print("=" * 100)

    # liste des shards contenant un .mlp.gate.weight (via l'index si présent)
    gate_shards = set()
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as fh:
            wm = json.load(fh).get("weight_map", {})
        gate_shards = {v for k, v in wm.items() if ".mlp.gate.weight" in k}

    # on lit tous les shards présents qui peuvent contenir des gates
    acc = defaultdict(list)   # expert -> [norme L2 par couche]
    layer_seen = set()
    for sh in sorted(os.listdir(shard_dir)):
        if not sh.endswith(".safetensors"):
            continue
        if gate_shards and sh not in gate_shards:
            continue  # ce shard ne contient pas de router gate
        path = os.path.join(shard_dir, sh)
        with open(path, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            if ".mlp.gate.weight" not in name:
                continue
            if ".layers." not in name:
                continue
            L = int(name.split(".layers.")[1].split(".")[0])
            W = _read_gate_tensor(path, hlen, info)   # [256, 2048]
            norms = np.linalg.norm(W, axis=1)         # [256]
            for e in range(W.shape[0]):
                acc[e].append(float(norms[e]))
            layer_seen.add(L)
            del W

    if not acc:
        print("[!] aucun router gate trouvé dans les shards présents.")
        print("    (les gates sont dans model.safetensors-00014-of-00014.safetensors)")
        return 1

    print(f"  Couches de gate lues : {len(layer_seen)}")
    print(f"  Experts              : {len(acc)}")

    rows = []
    for e in sorted(acc):
        v = np.array(acc[e])
        rows.append({
            "expert": e,
            "layers": len(v),
            "gate_norm_mean": round(float(v.mean()), 4),
            "gate_norm_std": round(float(v.std()), 4),
        })
    rows.sort(key=lambda r: -r["gate_norm_mean"])

    # normalise pour un ranking lisible (importance relative)
    mx = max(r["gate_norm_mean"] for r in rows)
    mn = min(r["gate_norm_mean"] for r in rows)
    for r in rows:
        r["importance_rel"] = round((r["gate_norm_mean"] - mn) / max(mx - mn, 1e-9), 4)

    print(f"\n  {'Expert':>7} | {'norme L2 moy':>13} | {'std':>8} | {'importance':>10}")
    print("  " + "-" * 50)
    for r in rows[:15]:
        print(f"  E{r['expert']:>6} | {r['gate_norm_mean']:>13.3f} | {r['gate_norm_std']:>8.3f} | {r['importance_rel']:>10.3f}")
    print("  ...")
    print(f"\n  Top 5 experts (norme la plus élevée) : {[r['expert'] for r in rows[:5]]}")
    print(f"  Bottom 5 experts (norme la plus basse) : {[r['expert'] for r in rows[-5:]]}")

    # spread
    all_n = np.array([r["gate_norm_mean"] for r in rows])
    print(f"\n  Norme L2 des gates : min={all_n.min():.3f} max={all_n.max():.3f} "
          f"spread={(all_n.max()-all_n.min())/all_n.mean()*100:.1f}% de la moyenne")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"experts": rows, "layers_seen": sorted(layer_seen)}, fh, ensure_ascii=False, indent=2)
    print(f"\n  [+] Rapport : {out_path}")
    print("  NOTE : norme de gate = proxy statique. La fréquence réelle exige un forward.")
    print("=" * 100)
    return 0


# ---------------------------------------------------------------------------
# MODE FORWARD — logits du routeur via transformers (exige modèle complet + runtime)
# ---------------------------------------------------------------------------
def forward_mode(model_path, corpus_path, out_path, max_tokens=512):
    print("=" * 100)
    print("  D2 ROUTER PROFILER — mode FORWARD (logits du routeur)")
    print("=" * 100)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print(f"[!] transformers/torch indisponibles : {e}")
        return 1

    # Vérifie la présence du modèle complet
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"[!] config.json absent dans {model_path} — modèle incomplet.")
        return 1

    print(f"[*] Chargement du modèle depuis {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    # Capture des logits du routeur : hook sur chaque module .mlp.gate (Linear)
    captures = []
    hooks = []
    for name, mod in model.named_modules():
        if name.endswith(".mlp.gate") and isinstance(mod, torch.nn.Linear):
            def _hook(m, inp, out, _name=name):
                captures.append((_name, out.detach().float()))
            hooks.append(mod.register_forward_hook(_hook))
    print(f"[*] {len(hooks)} modules routeur hookés")

    # Corpus
    text = open(corpus_path, encoding="utf-8").read()[:max_tokens * 4]
    toks = tokenizer(text, return_tensors="pt")
    seq_len = min(toks.input_ids.shape[1], max_tokens)
    toks = {k: v[:, :seq_len] for k, v in toks.items()}

    print(f"[*] Forward sur {seq_len} tokens ...")
    with torch.no_grad():
        model(**toks)

    for h in hooks:
        h.remove()

    # Agrégation
    act_count = defaultdict(int)
    prob_sum = defaultdict(float)
    rank_count = defaultdict(lambda: defaultdict(int))
    co_act = defaultdict(int)

    total_selections = 0
    for name, logits in captures:
        layer = int(name.split(".layers.")[1].split(".")[0]) if ".layers." in name else -1
        B, S, E = logits.shape
        probs = torch.softmax(logits, dim=-1)                       # [B,S,E]
        topv, topi = torch.topk(logits, k=min(TOP_K, E), dim=-1)    # [B,S,K]
        for b in range(B):
            for s in range(S):
                sel = topi[b, s].tolist()
                sel.sort()
                co_act[tuple(sel)] += 1
                for r, e in enumerate(sel):
                    act_count[(layer, e)] += 1
                    prob_sum[(layer, e)] += float(probs[b, s, e])
                    rank_count[(layer, e)][r + 1] += 1
                    total_selections += 1

    experts = sorted({e for (_, e) in act_count})
    rows = []
    for e in experts:
        cnt = sum(act_count[(L, e)] for L in range(40))
        pr = sum(prob_sum[(L, e)] for L in range(40))
        top1 = sum(rank_count[(L, e)].get(1, 0) for L in range(40))
        rows.append({
            "expert": e,
            "activation_count": cnt,
            "activation_rate": round(cnt / max(total_selections / len(experts), 1), 4),
            "mean_router_prob": round(pr / max(cnt, 1), 5),
            "top1_count": top1,
        })
    rows.sort(key=lambda r: -r["activation_count"])

    print(f"\n  {'Expert':>7} | {'activations':>12} | {'rate':>7} | {'prob moy':>9} | {'top1':>6}")
    print("  " + "-" * 52)
    for r in rows[:15]:
        print(f"  E{r['expert']:>6} | {r['activation_count']:>12} | {r['activation_rate']:>7.4f} | "
              f"{r['mean_router_prob']:>9.5f} | {r['top1_count']:>6}")
    print("  ...")

    co_top = sorted(co_act.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top 10 co-activations (groupes de {TOP_K} experts) :")
    for grp, c in co_top:
        print(f"    {list(grp)} : {c} tokens")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"experts": rows, "coactivation_top": [{"group": list(g), "count": c} for g, c in co_top]},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n  [+] Rapport : {out_path}")
    print("=" * 100)
    return 0


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 Router Profiler (Qwen3.5-35B-A3B)")
    ap.add_argument("--forward", action="store_true", help="forward via transformers (exige modèle complet + runtime)")
    ap.add_argument("--static", action="store_true", help="analyse statique des gates (sans forward)")
    ap.add_argument("--model", default=SHARD_DIR)
    ap.add_argument("--shard-dir", default=SHARD_DIR)
    ap.add_argument("--corpus", default=_find_corpus())
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--report", default=OUT_FORWARD)
    ap.add_argument("--static-report", default=OUT_STATIC)
    args = ap.parse_args()

    if args.forward:
        return forward_mode(args.model, args.corpus, args.report, args.max_tokens)
    if args.static:
        return static_mode(args.shard_dir, INDEX_PATH, args.static_report)

    print("[!] Choisis un mode : --forward ou --static")
    print("    (--static est disponible sans télécharger le modèle complet)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
