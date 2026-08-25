#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 LAYER BEHAVIOR PROFILER — Qwen3.5-35B-A3B / Qwen3.8-27B (STATIQUE)
=====================================================================
Version STATIQUE du profilage comportemental par couche (protocole D2 :
layer -> router -> expert -> tensor -> allocation). Sans forward : le
runtime GGUF instrumenté n'expose pas encore les activations, donc on
travaille sur les signaux statiques disponibles :

  1. Norme L2 du router gate par expert et par COUCHE (proxy de load/importance,
     validé par la littérature ICLR 2026 citée dans le protocole).
  2. Sensibilité par expert (SNR Q2/Q3/Q4/INT8/FP16, d2_expert_profiler).
  3. Croisement load x sensibilité -> allocation par layer/expert.

Métriques par couche L (proxy statique, à remplacer par le forward quand
beellama.cpp sera instrumenté) :
  - router_entropy   : entropie normalisée de la distribution des normes de gate
                       (0 = routing concentré sur qq experts, 1 = uniforme)
  - expert_load_std  : dispersion des normes (écart-type / moyenne)
  - top8_concentration : masse de norme portée par les 8 experts dominants
  - active_experts   : nb d'experts dont la norme dépasse la moyenne (proxy)
  - sensitivity      : SNR par format pondéré par la charge expert (proxy de
                       sensibilité "utile" = sensibilité x fréquence statique)
  - recommended      : plus petit palier dont le SNR pondéré atteint min_snr

Champs forward (input_rms, output_rms, residual_ratio, layer_time, ...) sont
laissés à null : ils exigent le hook C++ (d2_layer_behavior_profiler --forward
une fois beellama.cpp instrumenté).

Sorties :
  - d2_layer_behavior_report.json  (couches + matrices layer x expert)
  - d2_layer_behavior_matrix.csv   (matrice load %, exploitable)

Usage :
  python d2_layer_behavior_profiler.py
  python d2_layer_behavior_profiler.py --min-snr 20.0 --json out.json
"""

import argparse
import csv
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD_DIR = os.path.join(HERE, "hf_weights_35b")
INDEX_PATH = os.path.join(SHARD_DIR, "model.safetensors.index.json")
CONFIG_PATH = os.path.join(SHARD_DIR, "config.json")
EXPERT_REPORT = os.path.join(HERE, "d2_expert_profiler_report.json")
OUT_JSON = os.path.join(HERE, "d2_layer_behavior_report.json")
OUT_CSV = os.path.join(HERE, "d2_layer_behavior_matrix.csv")

TOP_K = 8  # num_experts_per_tok = 8

# Palier de taille -> formats (du plus petit au plus grand). Reflète BYTES_PER_FMT
# de d2_expert_profiler (Q2=0.25, Q3=0.375, Q4/NVFP4=0.5, INT8/FP8=1.0, FP16=2.0).
TIERS = [
    (0.25, ["Q2"]),
    (0.375, ["Q3"]),
    (0.5, ["Q4", "NVFP4"]),
    (1.0, ["INT8", "FP8"]),
    (2.0, ["FP16"]),
]
BYTES_PER_FMT = {"FP16": 2.0, "FP8": 1.0, "INT8": 1.0, "NVFP4": 0.5,
                 "Q4": 0.5, "Q3": 0.375, "Q2": 0.25}


def read_header(path):
    with open(path, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hlen))
    return hlen, hdr


def read_gate_tensor(path, hlen, info):
    """Lit un tensor gate [256, 2048] en float32."""
    s, e = info["data_offsets"]
    dt = info.get("dtype", "BF16")
    bpe = 4 if dt == "F32" else 2
    with open(path, "rb") as fh:
        fh.seek(8 + hlen + s)
        raw = fh.read(e - s)
    if dt == "F32":
        W = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        u = np.frombuffer(raw, dtype="<u2")
        W = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float32)
    return W.reshape(info["shape"])


def layer_types():
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = json.load(fh)
    tc = cfg.get("text_config", {})
    return tc.get("layer_types", []), tc


def gate_shard_map():
    with open(INDEX_PATH, encoding="utf-8") as fh:
        wm = json.load(fh).get("weight_map", {})
    return {k: v for k, v in wm.items() if ".mlp.gate.weight" in k}


def entropy_norm(p):
    """Entropie de Shannon normalisée par log2(n). p = distribution >= 0."""
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / np.log2(max(len(p) + 1, 2)))


def load_expert_sensitivity(report_path):
    """expert -> {fmt: snr} depuis d2_expert_profiler_report.json."""
    if not os.path.exists(report_path):
        return {}
    with open(report_path, encoding="utf-8") as fh:
        d = json.load(fh)
    out = {}
    for r in d.get("experts", []):
        e = r["expert"]
        out[e] = {f: r.get(f"{f}_snr") for f in ("Q2", "Q3", "Q4", "NVFP4", "FP8", "INT8", "FP16")}
    return out


def recommend(snr_weighted, min_snr):
    """Plus petit palier dont le meilleur format atteint min_snr sur le SNR pondéré."""
    for _size, fmts in TIERS:
        vals = [(f, snr_weighted.get(f)) for f in fmts]
        vals = [(f, v) for f, v in vals if v is not None]
        if not vals:
            continue
        best_fmt, best_snr = max(vals, key=lambda x: x[1])
        if best_snr >= min_snr:
            return best_fmt
    return "FP16"


def load_forward_jsonl(path):
    """Charge la sortie de llama-layer-profiler (C++) : layer -> {n_tokens,
    router_entropy, active_experts, expert_freq, expert_mean_prob}."""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["layer"]] = r
    return out


def build(min_snr=20.0, out_json=OUT_JSON, out_csv=OUT_CSV, forward_jsonl=None):
    ltypes, tc = layer_types()
    n_layers = len(ltypes)
    gates = gate_shard_map()
    sens = load_expert_sensitivity(EXPERT_REPORT)
    fwd = load_forward_jsonl(forward_jsonl)

    # regroupe les tensors gate par shard pour ne lire chaque shard qu'une fois
    by_shard = {}
    for name, sh in gates.items():
        by_shard.setdefault(sh, []).append(name)

    # layer_idx -> {norms[256]} (40 couches texte) ; MTP mis à part
    layer_norms = {}
    mtp_norms = None
    for sh, names in by_shard.items():
        path = os.path.join(SHARD_DIR, sh)
        if not os.path.exists(path):
            continue
        hlen, hdr = read_header(path)
        for name in names:
            info = hdr.get(name)
            if info is None:
                continue
            W = read_gate_tensor(path, hlen, info)   # [256, 2048]
            norms = np.linalg.norm(W, axis=1)
            if ".layers." in name:
                L = int(name.split(".layers.")[1].split(".")[0])
                layer_norms[L] = norms
            else:
                mtp_norms = norms

    rows = []
    matrices_load = []   # layer x expert -> charge normalisée (%)
    matrices_prec = []   # layer x expert -> precision recommandée (statique)

    for L in range(n_layers):
        norms = layer_norms.get(L)
        if norms is None:
            continue
        typ = ltypes[L] if L < len(ltypes) else "unknown"
        n = len(norms)
        mean = float(norms.mean())
        std = float(norms.std())
        order = np.argsort(-norms)
        top8 = norms[order[:TOP_K]]
        top8_conc = float(top8.sum() / max(norms.sum(), 1e-12))

        # charge expert : forward (fréquence réelle) si dispo, sinon proxy statique
        fL = fwd.get(L)
        if fL is not None:
            freq = fL.get("expert_freq", {})
            p = np.zeros(n, dtype=np.float64)
            for k, v in freq.items():
                p[int(k)] = v
            p = p / max(p.sum(), 1e-12)
            entropy = fL.get("router_entropy", entropy_norm(p))
            active = int(fL.get("active_experts", int((p > 0).sum())))
            load_std = float(p.std() / max(p.mean(), 1e-12))
            top_experts = [int(x) for x in np.argsort(-p)[:TOP_K]]
            top8_conc = float(p[np.argsort(-p)[:TOP_K]].sum())
        else:
            p = norms / max(norms.sum(), 1e-12)
            entropy = entropy_norm(p)
            active = int((norms > mean).sum())
            load_std = std / max(mean, 1e-9)
            top_experts = [int(x) for x in order[:TOP_K]]

        # sensibilité pondérée par la charge de chaque expert
        acc = {f: 0.0 for f in ("Q2", "Q3", "Q4", "NVFP4", "FP8", "INT8", "FP16")}
        per_expert_prec = {}
        for e in range(n):
            w = float(p[e])
            s = sens.get(e, {})
            for f in acc:
                v = s.get(f)
                if v is not None:
                    acc[f] += w * v
            # choix local par expert (SNR brut, pas pondéré)
            per_expert_prec[e] = recommend({f: s.get(f) for f in acc}, min_snr) if s else "FP16"

        # normalise la sensibilité pondérée si le total des poids < 1 (experts sans SNR)
        wsum = sum(float(p[e]) for e in range(n) if sens.get(e))
        if wsum > 0:
            for f in acc:
                acc[f] = acc[f] / wsum if acc[f] else None
        rec = recommend(acc, min_snr)

        rows.append({
            "layer": L,
            "type": typ,
            "router_entropy": round(entropy, 4),
            "expert_load_std": round(load_std, 4),
            "top8_concentration": round(top8_conc, 4),
            "active_experts": active,
            "top_experts": top_experts,
            "source": "forward" if fL is not None else "static",
            "sensitivity": {f: (round(v, 1) if v else None) for f, v in acc.items()},
            "recommended": rec,
            # forward (hook C++) — laissés null en statique
            "input_rms": None, "output_rms": None, "residual_ratio": None,
            "attention_ratio": None, "ffn_ratio": None, "layer_time_ms": None,
            "memory_read_gb": None, "memory_write_gb": None,
        })
        matrices_load.append({"layer": L, "load": {str(e): round(float(p[e]), 6) for e in range(n)}})
        matrices_prec.append({"layer": L, "precision": {str(e): per_expert_prec[e] for e in range(n)}})

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen3.5-35B-A3B (27B actifs)",
        "mode": ("FORWARD (fréquence routeur réelle x expert SNR)" if fwd
                 else "STATIC (router gate L2 norm x expert SNR)"),
        "n_layers": n_layers,
        "layer_types_count": {t: ltypes.count(t) for t in set(ltypes)},
        "num_experts": tc.get("num_experts"),
        "num_experts_per_tok": tc.get("num_experts_per_tok"),
        "min_snr": min_snr,
        "note": ("forward metrics null tant que beellama.cpp n'expose pas les "
                 "activations/layer ; la fréquence réelle exige un forward (GEMQ : "
                 "la quantification modifie le routing -> re-mesurer après quantification)."),
        "layers": rows,
        "matrix_load": matrices_load,
        "matrix_precision": matrices_prec,
        "mtp_gate_norms": [round(float(x), 4) for x in mtp_norms] if mtp_norms is not None else None,
    }

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # CSV de la matrice load (layer en ligne, expert en colonne)
    n_exp = tc.get("num_experts") or 256
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["layer", "type"] + [f"E{e}" for e in range(n_exp)])
        for r in rows:
            m = matrices_load[r["layer"]]["load"]
            wr.writerow([r["layer"], r["type"]] + [m.get(str(e), "") for e in range(n_exp)])

    return report


def main():
    ap = argparse.ArgumentParser(description="D2 LAYER BEHAVIOR PROFILER")
    ap.add_argument("--min-snr", type=float, default=20.0, help="SNR min pour le choix")
    ap.add_argument("--json", default=OUT_JSON)
    ap.add_argument("--csv", default=OUT_CSV)
    ap.add_argument("--forward-jsonl", default=None,
                    help="sortie JSONL de llama-layer-profiler (C++) pour utiliser la "
                         "fréquence routeur réelle au lieu du proxy statique")
    args = ap.parse_args()

    report = build(args.min_snr, args.json, args.csv, args.forward_jsonl)

    print("=" * 96)
    print("  D2 LAYER BEHAVIOR — " + ("forward" if args.forward_jsonl else "statique")
          + " (charge expert x sensibilité expert)")
    print("=" * 96)
    print(f"  {'L':>3} | {'type':<16} | {'entropy':>8} | {'load_std':>8} | {'top8':>6} | "
          f"{'act':>4} | {'Q2':>6} {'Q3':>6} {'Q4':>6} {'INT8':>6} | {'rec':<7}")
    print("  " + "-" * 90)
    for r in report["layers"]:
        s = r["sensitivity"]
        f = lambda k: (f"{s[k]:5.1f}" if s.get(k) is not None else "     ")
        print(f"  {r['layer']:>3} | {r['type']:<16} | {r['router_entropy']:>8.3f} | "
              f"{r['expert_load_std']:>8.3f} | {r['top8_concentration']:>6.3f} | "
              f"{r['active_experts']:>4} | {f('Q2')} {f('Q3')} {f('Q4')} {f('INT8')} | {r['recommended']:<7}")
    print("  " + "-" * 90)
    print(f"  [+] JSON : {args.json}")
    print(f"  [+] CSV  : {args.csv}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
