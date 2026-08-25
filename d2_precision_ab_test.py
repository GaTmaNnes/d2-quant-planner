#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 PRECISION A/B TEST — erreur de quantification sur les VRAIS poids (HuggingFace).
===================================================================================
Compare FP16 / INT8 / NVFP4(E2M1) sur les tenseurs réels de Qwen/Qwen3.5-9B.

Lecture safetensors MANUELLE (seek + lecture ciblée, sans mmap) -> évite l'erreur
Windows "fichier de pagination insuffisant" (1455).

Pour chaque tenseur : référence (BF16->F32), FP16, INT8 (par canal), FP4 E2M1 (bloc 16).
Métriques : SNR (dB), erreur relative, cosinus.
"""

import json
import os
import struct
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Grille E2M1 (1 signe, 2 exposants, 1 mantisse) — approximation NVFP4
FP4_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}


# ---------------------------------------------------------------------------
# Lecteur safetensors manuel (sans mmap)
# ---------------------------------------------------------------------------
def read_header(path):
    with open(path, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(hlen).decode("utf-8")), hlen


def _bf16_to_f32(raw):
    u16 = np.frombuffer(raw, dtype="<u2")
    out = np.zeros(u16.shape, dtype=np.float32)
    out.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return out


def load_tensor(path, hdr, name, hlen, max_elems=2_000_000):
    info = hdr[name]
    dtype = info["dtype"]
    shape = list(info["shape"])
    start, end = info["data_offsets"]
    nel = int(np.prod(shape)) if shape else 1
    dt = DTYPE_BYTES[dtype]
    cols = shape[-1] if shape else 1

    nread = min(nel, max_elems) if max_elems else nel
    with open(path, "rb") as fh:
        fh.seek(8 + hlen + start)   # data_offsets relatifs à la section data
        raw = fh.read(nread * dt)

    if dtype == "BF16":
        arr = _bf16_to_f32(raw)
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"dtype non supporté: {dtype}")

    nrows = nread // cols
    return arr[:nrows * cols].reshape(nrows, cols).astype(np.float32)


# ---------------------------------------------------------------------------
# Quantification
# ---------------------------------------------------------------------------
def quant_fp16(W):
    return W.astype(np.float16).astype(np.float32)


def quant_int8(W):
    m = np.abs(W).max(axis=-1, keepdims=True)
    m[m == 0] = 1.0
    q = np.round(W / m * 127.0)
    return np.clip(q, -127, 127) / 127.0 * m


def quant_fp4(W, block=16):
    out = np.empty_like(W, dtype=np.float32)
    for i in range(W.shape[0]):
        row = W[i]
        n = row.shape[0]
        pad = (block - n % block) % block
        wp = np.pad(row, (0, pad), mode="constant")
        wr = wp.reshape(-1, block)
        m = np.abs(wr).max(axis=1, keepdims=True)
        m[m == 0] = 1.0
        norm = wr / m
        sign = np.sign(norm)
        mag = np.abs(norm)
        dist = np.abs(mag[:, :, None] - FP4_GRID[None, None, :])
        out[i] = (sign * FP4_GRID[dist.argmin(axis=-1)] * m).reshape(-1)[:n]
    return out


def metrics(W, Wq):
    W = W.astype(np.float32)
    Wq = Wq.astype(np.float32)
    err = W - Wq
    num = float(np.sum(W * W, dtype=np.float64))
    den = float(np.sum(err * err, dtype=np.float64))
    snr = 10.0 * np.log10(num / max(den, 1e-30))
    rel = float(np.sqrt(max(den, 0.0) / max(num, 1e-30)))
    cos = float(np.sum(W * Wq, dtype=np.float64) /
                (np.sqrt(num) * np.sqrt(float(np.sum(Wq * Wq, dtype=np.float64))) + 1e-30))
    return snr, rel, cos


def quantize_all(W):
    if W.ndim == 1:
        W = W.reshape(1, -1)
    W = W.astype(np.float32)
    return {
        "fp16": metrics(W, quant_fp16(W)),
        "int8": metrics(W, quant_int8(W)),
        "fp4": metrics(W, quant_fp4(W)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    shard_dir = os.path.join(HERE, "hf_weights")
    shards = [
        os.path.join(shard_dir, "model.safetensors-00002-of-00004.safetensors"),
        os.path.join(shard_dir, "model.safetensors-00004-of-00004.safetensors"),
    ]

    targets = {
        "conv1d.weight": ("ssm_conv1d.weight", "SSM conv1d (FP16_REQUIRED)"),
        "A_log": ("ssm_a", "SSM A_log (ssm_a)"),
        "in_proj_a.weight": ("ssm_alpha.weight", "SSM in_proj_a (alpha)"),
        "in_proj_b.weight": ("ssm_beta.weight", "SSM in_proj_b (beta)"),
        "out_proj.weight": ("ssm_out.weight", "SSM out_proj"),
        "mlp.up_proj.weight": ("ffn_up.weight", "FFN up (NVFP4_SAFE)"),
        "mlp.gate_proj.weight": ("ffn_gate.weight", "FFN gate (NVFP4_SAFE)"),
        "mlp.down_proj.weight": ("ffn_down.weight", "FFN down (NVFP4_SAFE)"),
    }

    pm = {}
    pmpath = os.path.join(HERE, "precision_map.json")
    if os.path.exists(pmpath):
        with open(pmpath, encoding="utf-8") as fh:
            pm = json.load(fh)

    agg = {}
    details = []

    for shard in shards:
        if not os.path.exists(shard):
            print(f"[!] shard absent : {shard}", flush=True)
            continue
        print(f"[...] lecture {os.path.basename(shard)}", flush=True)
        hdr, hlen = read_header(shard)
        for key in hdr.keys():
            if key.startswith("__"):
                continue
            for suffix, (gguf_name, cat) in targets.items():
                if key.endswith(suffix):
                    layer = key.split(".layers.")[1].split(".")[0] if ".layers." in key else "?"
                    W = load_tensor(shard, hdr, key, hlen, max_elems=1_000_000)
                    res = quantize_all(W)
                    snr_fp16, rel_fp16, _ = res["fp16"]
                    snr_i8, rel_i8, _ = res["int8"]
                    snr_f4, rel_f4, _ = res["fp4"]

                    gguf_full = f"blk.{layer}.{gguf_name}"
                    prec = pm.get(gguf_full, {}).get("precision", "?")
                    density = pm.get(gguf_full, {}).get("density", None)

                    details.append({
                        "layer": layer, "cat": cat, "gguf": gguf_full,
                        "precision": prec, "density": density,
                        "fp16_snr": round(snr_fp16, 1), "fp16_rel": round(rel_fp16, 4),
                        "int8_snr": round(snr_i8, 1), "int8_rel": round(rel_i8, 4),
                        "fp4_snr": round(snr_f4, 1), "fp4_rel": round(rel_f4, 4),
                    })
                    agg.setdefault(cat, []).append({
                        "fp16_snr": snr_fp16, "int8_snr": snr_i8, "fp4_snr": snr_f4,
                        "int8_rel": rel_i8, "fp4_rel": rel_f4,
                    })
                    print(f"    {layer:>3} {cat:<26} INT8={snr_i8:5.1f}dB  FP4={snr_f4:5.1f}dB", flush=True)
                    break

    # --- Rapport texte ---
    L = ["=" * 100,
         "  D2 PRECISION A/B TEST — erreur de quantification (vrais poids HF)",
         "  SNR plus haut = mieux. rel = erreur relative (plus bas = mieux).",
         "=" * 100]
    L.append(f"\n{'Catégorie':<28} | {'FP16 SNR':>9} | {'INT8 SNR':>9} | {'FP4 SNR':>9} | {'INT8 rel':>8} | {'FP4 rel':>8}")
    L.append("-" * 100)
    for cat in sorted(agg):
        rows = agg[cat]
        L.append(f"{cat:<28} | {np.mean([r['fp16_snr'] for r in rows]):>8.1f} | "
                 f"{np.mean([r['int8_snr'] for r in rows]):>8.1f} | "
                 f"{np.mean([r['fp4_snr'] for r in rows]):>8.1f} | "
                 f"{np.mean([r['int8_rel'] for r in rows]):>7.3f} | "
                 f"{np.mean([r['fp4_rel'] for r in rows]):>7.3f}")
    L.append("-" * 100)
    L.append("  (moyennes sur toutes les couches de chaque catégorie)")
    L.append("\n[DÉTAIL par layer — corrélé à la densité d'outliers]")
    L.append(f"{'Layer':<6} | {'Cat':<26} | {'densité':>8} | {'INT8 SNR':>9} | {'FP4 SNR':>9} | précision_map")
    L.append("-" * 100)
    for d in details:
        dens = f"{d['density']:.2f}" if d["density"] is not None else "  —"
        L.append(f"{d['layer']:<6} | {d['cat']:<26} | {dens:>8} | {d['int8_snr']:>8.1f} | {d['fp4_snr']:>8.1f} | {d['precision']}")
    L.append("=" * 100)

    report = "\n".join(L)
    print(report)

    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "summary": {k: {"fp16_snr": round(float(np.mean([r['fp16_snr'] for r in v])), 1),
                           "int8_snr": round(float(np.mean([r['int8_snr'] for r in v])), 1),
                           "fp4_snr": round(float(np.mean([r['fp4_snr'] for r in v])), 1),
                           "int8_rel": round(float(np.mean([r['int8_rel'] for r in v])), 4),
                           "fp4_rel": round(float(np.mean([r['fp4_rel'] for r in v])), 4)}
                       for k, v in agg.items()},
           "details": details}
    with open(os.path.join(HERE, "d2_precision_ab_report.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\n[+] JSON exporté : d2_precision_ab_report.json")


if __name__ == "__main__":
    main()
