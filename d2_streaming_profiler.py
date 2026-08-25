#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 STREAMING PROFILER — profile les poids « au fur et à mesure ».
=================================================================
Lit les shards safetensors TENSOR PAR TENSOR (jamais le modèle entier),
mesure la sensibilité de chaque poids, décide une précision, écrit le
résultat puis libère la mémoire — exactement le pipeline streaming du D2 :

    shard -> tensor -> decode -> métriques -> décision -> écriture -> free -> suivant

Entrées (lecture metadata uniquement, pas de chargement global) :
  - hf_weights/model.safetensors.index.json  (weight_map + taille totale)
  - hf_weights/model.safetensors-*-of-*.safetensors  (headers lus une fois)

Sorties :
  - d2_stream_profiler.jsonl         (une ligne JSON par tensor, ajoutée au fil de l'eau,
                                      reprise possible après interruption)
  - d2_stream_profiler_report.json   (synthèse : couverture, décision, tailles, top sensibles)

Métriques par tensor :
  - shape / dtype / n_elems / octets (BF16=2, F32=4)
  - max_abs, rms, density (fraction d'outliers > 3 sigma), alpha_proxy (max_abs/rms)
  - SNR (dB) de re-quantification : FP16, FP8 (E4M3), INT8, NVFP4 (E2M1+E4M3)

Décision (compatible precision_map.json, mêmes passes que d2_generate_precision_map) :
  1. STRUCTURAL  : norm / A_log / dt_bias / conv1d -> FP16_REQUIRED (kernel F32 / critique)
  2. ECONOMIC    : tenseur < 1 Mo FP16 -> FP16_REQUIRED (gain nul)
  3. NUMERICAL   : score UTILITY = gain VRAM - erreur_effective/tolérance
                   -> FP16_REQUIRED | INT8_SAFE | NVFP4_SAFE
  (FP8 est mesuré et affiché mais pas encore branché dans precision_map.json)

Usage :
  python d2_streaming_profiler.py [--max-elems N] [--limit N] [--shard-dir DIR]
"""

import argparse
import json
import os
import struct
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD_DIR = os.path.join(HERE, "hf_weights")
INDEX_PATH = os.path.join(SHARD_DIR, "model.safetensors.index.json")
OUT_JSONL = os.path.join(HERE, "d2_stream_profiler.jsonl")
OUT_REPORT = os.path.join(HERE, "d2_stream_profiler_report.json")

from d2_noise_profiler import quant_int8, quant_e2m1_block, E4M3
from d2_autonomous_profiler import compute_utility, DEFAULT_SNR, spectral_factor

LABEL = {"FP16": "FP16_REQUIRED", "INT8": "INT8_SAFE", "NVFP4": "NVFP4_SAFE"}
BYTES_PER = {"FP16_REQUIRED": 2.0, "INT8_SAFE": 1.0, "NVFP4_SAFE": 0.5}
ECON_MIN_BYTES = 1.0e6          # tenseur < 1 Mo (FP16) => gain économique nul
SAMPLE_MAX_ELEMS = 4_000_000    # cap par défaut des éléments mesurés (≈16 Mo en F32)


# ---------------------------------------------------------------------------
# Quantification FP8 (E4M3) — mesurée, non encore branchée dans precision_map.json
# ---------------------------------------------------------------------------
def quant_fp8_e4m3(W):
    """FP8 E4M3 par ligne (scale = max|row|), valeurs normalisées dans [-1, 1].

    Utilise searchsorted (E4M3 trié) plutôt qu'une matrice de distance [.., 448]
    qui exploserait en mémoire sur les gros tensors.
    """
    m = np.abs(W).max(axis=-1, keepdims=True)
    m[m == 0] = 1.0
    norm = W / m
    sign = np.sign(norm)
    mag = np.abs(norm).astype(np.float32)
    idx = np.searchsorted(E4M3, mag)
    idx = np.clip(idx, 1, len(E4M3) - 1)
    lo = E4M3[idx - 1]
    hi = E4M3[idx]
    nearest = np.where(mag - lo <= hi - mag, lo, hi)
    return (sign * nearest * m).astype(np.float32)


def quant_symmetric(W, bits):
    """Quantification symétrique par ligne sur `bits` (proxy Q4/Q3/Q2).

    levels = 2^(bits-1)-1 : Q4=7, Q3=3, Q2=1. Scale = max|row|.
    C'est un proxy simple (pas la recette exacte Q4_K/Q3_K/Q2_K à blocs + mins),
    suffisant pour CLASSER la sensibilité relative des tensors/experts.
    """
    levels = (1 << (bits - 1)) - 1
    m = np.abs(W).max(axis=-1, keepdims=True)
    m[m == 0] = 1.0
    q = np.clip(np.round(W / m * levels), -levels, levels)
    return (q / levels * m).astype(np.float32)


def _snr(W, Wq):
    Wf = W.astype(np.float64)
    e = Wf - Wq.astype(np.float64)
    num = float(np.sum(Wf * Wf))
    den = float(np.sum(e * e))
    return 10.0 * np.log10(num / max(den, 1e-30))


def measure(W):
    """SNR FP16/FP8/INT8/NVFP4 d'un tenseur réel (2D, orientation (lignes, colonnes))."""
    W = W.astype(np.float32)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    elif W.ndim > 2:
        W = W.reshape(W.shape[0], -1)
    Wt = W.T if W.shape[0] < W.shape[1] else W
    out = {
        "FP16": _snr(Wt, Wt.astype(np.float16).astype(np.float32)),
        "FP8": _snr(Wt, quant_fp8_e4m3(Wt)),
        "INT8": _snr(Wt, quant_int8(Wt)),
        "NVFP4": _snr(Wt, quant_e2m1_block(Wt, scale_e4m3=True)),
        "Q4": _snr(Wt, quant_symmetric(Wt, 4)),
        "Q3": _snr(Wt, quant_symmetric(Wt, 3)),
        "Q2": _snr(Wt, quant_symmetric(Wt, 2)),
    }
    for k in out:
        if not np.isfinite(out[k]):
            out[k] = DEFAULT_SNR.get(k, 90.0)
    return out


def stats(W):
    """Statistiques légères de distribution (proxys spectraux, sans SVD)."""
    W = W.ravel().astype(np.float64)
    rms = float(np.sqrt(np.mean(W ** 2))) if W.size else 0.0
    maxa = float(np.abs(W).max()) if W.size else 0.0
    std = float(W.std()) if W.size else 0.0
    density = float(np.mean(np.abs(W) > 3.0 * std)) if std > 0 else 0.0
    alpha = (maxa / rms) if rms > 0 else 0.0
    return dict(rms=round(rms, 6), max_abs=round(maxa, 4),
                density=round(density, 6), alpha_proxy=round(alpha, 3))


def category(name):
    n = name.lower()
    if "visual" in n or "vision" in n:
        return "vision"
    if "mtp" in n:
        return "mtp"
    if "embed_tokens" in n or "lm_head" in n:
        return "embed/head"
    if "norm" in n or "layernorm" in n:
        return "norm"
    if "linear_attn" in n or "conv1d" in n or "in_proj" in n or "a_log" in n or "dt_bias" in n:
        return "ssm"
    if "self_attn" in n or "attn" in n:
        return "attention"
    if "mlp" in n or "ffn" in n:
        return "ffn"
    return "other"


def gguf_name(name):
    """Mapping HF -> nom GGUF (best-effort, réutilise le mapping du projet)."""
    n = name
    if "mlp.up_proj" in n:
        return "ffn_up.weight"
    if "mlp.gate_proj" in n:
        return "ffn_gate.weight"
    if "mlp.down_proj" in n:
        return "ffn_down.weight"
    if "linear_attn.conv1d" in n:
        return "ssm_conv1d.weight"
    if "linear_attn.A_log" in n:
        return "ssm_a"
    if "linear_attn.dt_bias" in n:
        return "ssm_dt.bias"
    if "linear_attn.in_proj_a" in n:
        return "ssm_alpha.weight"
    if "linear_attn.in_proj_b" in n:
        return "ssm_beta.weight"
    if "linear_attn.in_proj_qkv" in n:
        return "attn_qkv.weight"
    if "linear_attn.in_proj_z" in n:
        return "attn_gate.weight"
    if "linear_attn.out_proj" in n:
        return "ssm_out.weight"
    if "linear_attn.norm" in n:
        return "ssm_norm.weight"
    if "self_attn.q_proj" in n:
        return "attn_q.weight"
    if "self_attn.k_proj" in n:
        return "attn_k.weight"
    if "self_attn.v_proj" in n:
        return "attn_v.weight"
    if "self_attn.o_proj" in n:
        return "attn_output.weight"
    return None


def layer_num(name):
    for token in (".layers.", ".blocks."):
        if token in name:
            return name.split(token)[1].split(".")[0]
    return "?"


def decide(name, n_elems, snr, alpha, density):
    """(label_precision_map, raison) — mêmes passes que d2_generate_precision_map."""
    ln = name.lower()
    fp16_bytes = n_elems * 2.0

    # 1. STRUCTURAL PASS
    if "norm" in ln or "layernorm" in ln:
        return "FP16_REQUIRED", "STRUCTURAL: normalisation (critique)"
    if "a_log" in ln or "dt_bias" in ln or "conv1d" in ln:
        return "FP16_REQUIRED", "STRUCTURAL: SSM recurrent (kernel F32 / minuscule)"

    # 2. ECONOMIC PASS
    if fp16_bytes < ECON_MIN_BYTES:
        return "FP16_REQUIRED", f"ECONOMIC: {fp16_bytes/1e6:.3f} Mo FP16 (gain nul)"

    # 3. NUMERICAL PASS (score UTILITY)
    snr_map = {"FP16": snr["FP16"], "INT8": snr["INT8"], "NVFP4": snr["NVFP4"]}
    sf = spectral_factor(alpha, density, name)
    u = compute_utility(name, n_elems, snr_map, spectral_factor=sf)
    best = u["best"]
    reason = (f"UTILITY={u[best]['utility']:+.3f} "
              f"(SNR FP8={snr['FP8']:.1f} INT8={snr['INT8']:.1f} NVFP4={snr['NVFP4']:.1f} ; "
              f"SF={sf:.2f})")
    return LABEL[best], reason


# ---------------------------------------------------------------------------
# Inventaire (metadata uniquement)
# ---------------------------------------------------------------------------
def build_inventory(shard_dir, index_path):
    expected = set()
    total_size = None
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as fh:
            idx = json.load(fh)
        expected = set(idx.get("weight_map", {}).values())
        total_size = idx.get("metadata", {}).get("total_size")

    present = sorted(f for f in os.listdir(shard_dir) if f.endswith(".safetensors"))
    missing = sorted(expected - set(present))

    tensors = []  # (name, info, shard_path, header_len)
    for sh in present:
        path = os.path.join(shard_dir, sh)
        with open(path, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            tensors.append((name, info, path, hlen))
    return tensors, present, missing, total_size


def read_tensor(path, hlen, info, max_elems):
    """Lit UN tenseur (seek + read ciblé), le décode en float32. Ne garde rien d'autre."""
    s, e = info["data_offsets"]
    dt = info["dtype"]
    shape = info["shape"]
    nel = int(np.prod(shape))
    nread = min(nel, max_elems)
    bytes_per = 4 if dt == "F32" else 2
    with open(path, "rb") as fh:
        fh.seek(8 + hlen + s)          # data_offsets relatifs à la section data
        raw = fh.read(nread * bytes_per)
    if dt == "F32":
        arr = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:  # BF16 (et F16 traité comme BF16 au mieux)
        u = np.frombuffer(raw, dtype="<u2")
        arr = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float32)
    cols = shape[-1] if shape else 1
    if nread < nel:
        rows = nread // cols
        arr = arr[: rows * cols].reshape(rows, cols)
    else:
        arr = arr.reshape(shape)
    return arr, nel, nread


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 Streaming Profiler (poids au fil de l'eau)")
    ap.add_argument("--shard-dir", default=SHARD_DIR)
    ap.add_argument("--index", default=INDEX_PATH)
    ap.add_argument("--max-elems", type=int, default=SAMPLE_MAX_ELEMS,
                    help="éléments max mesurés par tensor (défaut 4M, ≈16 Mo F32)")
    ap.add_argument("--limit", type=int, default=0, help="ne traiter que N tensors (0 = tous)")
    ap.add_argument("--jsonl", default=OUT_JSONL)
    ap.add_argument("--report", default=OUT_REPORT)
    args = ap.parse_args()

    tensors, present, missing, total_size = build_inventory(args.shard_dir, args.index)

    print("=" * 100)
    print("  D2 STREAMING PROFILER — profil des poids tensor par tensor")
    print("=" * 100)
    print(f"  Shards présents : {len(present)}  |  manquants : {len(missing)}")
    for sh in present:
        print(f"    [OK]  {sh}")
    for sh in missing:
        print(f"    [--]  {sh}  (ABSENT)")
    if total_size:
        print(f"  Taille totale attendue (index) : {total_size/1e9:.2f} Go")
    print(f"  Tensors à profiler             : {len(tensors)}")
    print("=" * 100)

    # ré-écrit le JSONL à zéro (reprise = relancer, le fichier est rejouable)
    if os.path.exists(args.jsonl):
        os.remove(args.jsonl)

    rows = []
    t0 = time.time()
    n_done = 0
    hdr = (f"  {'#':>4} | {'cat':<10} | {'tensor':<52} | {'shape':<14} | "
           f"{'FP16':>6} | {'FP8':>6} | {'INT8':>6} | {'NVFP4':>6} | "
           f"{'Q4':>6} | {'Q3':>6} | {'Q2':>6} | {'décision':<14}")
    print("\n" + hdr)
    print("  " + "-" * 128)

    for i, (name, info, path, hlen) in enumerate(tensors):
        if args.limit and i >= args.limit:
            break
        try:
            W, nel, nread = read_tensor(path, hlen, info, args.max_elems)
            if not np.isfinite(W).all() or float(np.abs(W).max()) > 1e6:
                continue  # tenseur corrompu / anormal
            cat = category(name)
            gg = gguf_name(name)
            st = stats(W)
            snr = measure(W)
            label, reason = decide(name, nel, snr, st["alpha_proxy"], st["density"])
        except Exception as exc:
            print(f"  [!] échec {name}: {exc}")
            continue

        rec = {
            "name": name,
            "gguf": gg,
            "category": cat,
            "layer": layer_num(name),
            "dtype": info["dtype"],
            "shape": info["shape"],
            "n_elems": nel,
            "sampled": nread < nel,
            "bytes_fp16": nel * 2.0,
            "bytes_chosen": nel * BYTES_PER.get(label, 2.0),
            "max_abs": st["max_abs"],
            "rms": st["rms"],
            "density": st["density"],
            "alpha_proxy": st["alpha_proxy"],
            "snr": {k: round(v, 1) for k, v in snr.items()},
            "precision": label,
            "reason": reason,
        }
        rows.append(rec)

        with open(args.jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        n_done += 1
        shape_s = "x".join(str(d) for d in info["shape"])
        print(f"  {n_done:>4} | {cat:<10} | {name:<52} | {shape_s:<14} | "
              f"{snr['FP16']:>6.1f} | {snr['FP8']:>6.1f} | {snr['INT8']:>6.1f} | "
              f"{snr['NVFP4']:>6.1f} | {snr['Q4']:>6.1f} | {snr['Q3']:>6.1f} | "
              f"{snr['Q2']:>6.1f} | {label:<14}")

        del W  # libère explicitement avant le tensor suivant

    dt = time.time() - t0

    # --- synthèse ---
    c = Counter(r["precision"] for r in rows)
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], Counter())[r["precision"]] += 1

    bf16_total = sum(r["bytes_fp16"] for r in rows)
    chosen_total = sum(r["bytes_chosen"] for r in rows)

    print("\n" + "=" * 100)
    print("  SYNTHÈSE")
    print("=" * 100)
    print(f"  Tensors profilés       : {n_done}  (en {dt:.1f} s)")
    print(f"  Répartition précision  : FP16_REQUIRED {c.get('FP16_REQUIRED',0):>4} | "
          f"INT8_SAFE {c.get('INT8_SAFE',0):>4} | NVFP4_SAFE {c.get('NVFP4_SAFE',0):>4}")
    print(f"  Poids profilés (BF16)  : {bf16_total/1e9:,.2f} Go")
    print(f"  Poids après décision   : {chosen_total/1e9:,.2f} Go  "
          f"(-{(bf16_total - chosen_total)/1e9:,.2f} Go, {100*chosen_total/max(bf16_total,1):.0f}%)")
    if total_size:
        print(f"  Couverture vs modèle   : {bf16_total/1e9:.2f} Go / {total_size/1e9:.2f} Go "
              f"({100*bf16_total/total_size:.0f}%)  [shards manquants : {', '.join(missing) or 'aucun'}]")

    print("\n  Par catégorie (FP16 / INT8 / NVFP4) :")
    for cat in sorted(by_cat):
        cc = by_cat[cat]
        print(f"    {cat:<12} {cc.get('FP16_REQUIRED',0):>4} / {cc.get('INT8_SAFE',0):>4} / "
              f"{cc.get('NVFP4_SAFE',0):>4}")

    # top tensors les plus sensibles (SNR NVFP4 le plus faible parmi les non-normes)
    ranked = sorted((r for r in rows if r["category"] not in ("norm", "vision")),
                    key=lambda r: r["snr"]["NVFP4"])
    print("\n  Top tensors sensibles (SNR NVFP4 le plus faible → garder haute précision) :")
    for r in ranked[:12]:
        print(f"    {r['name']:<52} NVFP4={r['snr']['NVFP4']:>6.1f} dB  "
              f"FP8={r['snr']['FP8']:>6.1f} dB  -> {r['precision']}")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shards_present": present,
        "shards_missing": missing,
        "total_size_index": total_size,
        "tensors_profiled": n_done,
        "elapsed_s": round(dt, 2),
        "distribution": dict(c),
        "by_category": {k: dict(v) for k, v in by_cat.items()},
        "bytes_bf16": bf16_total,
        "bytes_chosen": chosen_total,
        "coverage_ratio": (bf16_total / total_size) if total_size else None,
        "rows": rows,
    }
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("=" * 100)
    print(f"  [+] JSONL  : {args.jsonl}")
    print(f"  [+] Report : {args.report}")
    print("  Note : FP8 (E4M3) est mesuré mais non branché dans precision_map.json (labels INT8/NVFP4/FP16).")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
