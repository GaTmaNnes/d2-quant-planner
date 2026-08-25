#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 TENSOR PROFILER — full-model, precision error per tensor
============================================================
Profile les 1606 tensors du checkpoint (lecture brute des shards, decode
BF16/F16/F32/F8_E4M3 correct) et calcule pour CHAQUE tensor:

  - stats: rms / max / outlier_rate / entropie approx / sparsite
  - erreur de quantification RELLE par precision (block-scaled, group 32,
    similaire Q4_K/Q3_K/Q2_K/Q1 ternaire) : SNR_dB, erreur relative,
    erreur ponderee par canal (proxy activation-aware sans runtime)
  - score = E[||W-Wq||^2] / E[||W||^2]  (SNR poids)
          + score canal = moyenne ponderee par RMS de ligne (proxy AWQ)
            -> HEURISTIC tant que le runtime forward n'existe pas

Sorties agregees par: layer x tensor_type (gate/up/down/qkv/o_proj/...),
puis par role. Verrou MLP decompose -> down_proj vs gate/up_proj mesure.

Usage:
  python d2_tensor_profiler.py [--json d2_tensor_profile.json] [--limit N]
"""

import argparse
import json
import math
import os
import struct
import sys
import time

from d2_schema import TensorRecord, QWEN38_27B
from d2_registry import D2Registry

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "Qwen3.8-27B-FP8")
OUT = os.path.join(HERE, "d2_tensor_profile.json")

BLOCK = 32  # group size type k-quant
PRECS = [("Q4", 4), ("Q3", 3), ("Q2", 2), ("Q1", 1)]


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def f8e4m3_to_f32(raw):
    """F8_E4M3FN (float8_e4m3fn torch) -> float32.
    signe 1 | exposant 4 | mantisse 3, bias 7.
    NB: exp=15 avec man<7 est FINI (256..448) ; seul exp=15 man=7 = NaN. PAS d'infini.
    Valide contre torch.float8_e4m3fn (0 desaccord sur 0x00..0xFF)."""
    import numpy as np
    sign = ((raw >> 7) & 1).astype(np.float32)
    exp = ((raw >> 3) & 0xF).astype(np.int32)
    man = (raw & 0x7).astype(np.float32)
    out = np.zeros(raw.size, dtype=np.float32)
    nan = (exp == 15) & (man == 7)
    sub = exp == 0
    out[sub] = (1.0 - 2.0 * sign[sub]) * (2.0 ** (1 - 7)) * (man[sub] / 8.0)
    norm = (exp > 0) & (~nan)
    out[norm] = (1.0 - 2.0 * sign[norm]) * (2.0 ** (exp[norm] - 7)) * (1.0 + man[norm] / 8.0)
    out[nan] = np.nan
    return out


def dequant_fp8(f, scale_inv, shape):
    """Applique le scaling par bloc 128x128 (HF fine-grained FP8).
    scale_inv: tableau brut BF16 de forme [ceil(r/128), ceil(c/128)].
    f: valeurs F8_E4M3 dequantifiees, forme (r, c)."""
    import numpy as np
    r, c = shape
    if scale_inv.dtype == np.uint8:
        u16 = scale_inv.view(np.uint16)
        s = (u16.astype(np.uint32) << 16).view(np.float32)
    else:
        s = scale_inv
    if s.ndim == 1:
        nr, nc = -(-r // 128), -(-c // 128)
        s = s.reshape(nr, nc)
    ri = (np.arange(r) // 128)
    ci = (np.arange(c) // 128)
    grid = s[ri][:, ci]
    return f * grid


def decode(raw, dtype):
    import numpy as np
    if dtype == "BF16":
        u16 = raw.view(np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32)
    if dtype == "F32":
        return raw.view(np.float32)
    if dtype == "F16":
        return raw.view(np.float16).astype(np.float32)
    if dtype == "F8_E4M3":
        return f8e4m3_to_f32(raw)
    raise ValueError(f"dtype inconnu: {dtype}")


def block_quant_error(f, bits):
    """Erreur de quant block-scaled (group 32, scale symetrique, sans zeropoint).
    Q1 = ternaire {-1,0,+1} (sparse ternaire, 2 bits logiques mais ~1.6 bits/poids)."""
    import numpy as np
    n = f.size
    if n == 0:
        return {"snr_db": 0.0, "rel_err": 0.0}
    if bits == 1:
        # ternaire: seuil = 0.5*max; valeurs +/-1 ou 0
        thr = 0.5 * np.max(np.abs(f))
        q = np.zeros(f.shape, dtype=np.float32)
        m = np.abs(f) > thr
        q[m] = np.sign(f[m])
        err = f - q
        e2 = float(np.sum(err * err))
        s2 = float(np.sum(f * f))
    else:
        levels = (1 << bits) - 1
        fb = f.reshape(-1, BLOCK)
        amax = np.max(np.abs(fb), axis=1).astype(np.float32)
        amax_safe = np.where(amax == 0, 1.0, amax)
        qb = np.clip(np.round(fb / amax_safe[:, None] * levels), -levels, levels) / levels
        q = (qb * amax_safe[:, None]).reshape(f.shape)
        err = f - q
        e2 = float(np.sum(err * err))
        s2 = float(np.sum(f * f))
    if s2 == 0:
        return {"snr_db": 0.0, "rel_err": 0.0}
    rel = e2 / s2
    snr = 10 * math.log10(s2 / e2) if e2 > 0 else 99.0
    return {"snr_db": round(snr, 2), "rel_err": round(rel, 6)}


def channel_score(f, bits):
    """Proxy activation-aware (AWQ-like) : erreur ponderee par RMS de LIGNE (canal de sortie).
    Sans runtime forward -> HEURISTIC : les canaux a forte norme dominent la sortie.
    Retourne le ratio d'erreur pondere vs non pondere (>1 = canaux importants plus erratiques)."""
    import numpy as np
    if f.size == 0 or f.ndim < 2:
        return None
    rms = np.sqrt(np.mean(f.astype(np.float64) ** 2, axis=1))
    rms_safe = np.where(rms == 0, 1.0, rms)
    w = rms / rms_safe.max()
    if bits == 1:
        thr = 0.5 * np.max(np.abs(f), axis=1, keepdims=True)
        q = np.where(np.abs(f) > thr, np.sign(f), 0.0)
    else:
        levels = (1 << bits) - 1
        amax = np.max(np.abs(f), axis=1, keepdims=True).astype(np.float32)
        amax_safe = np.where(amax == 0, 1.0, amax)
        q = np.clip(np.round(f / amax_safe * levels), -levels, levels) / levels * amax_safe
    err = f - q
    e2 = float(np.sum((err * err) * w[:, None]))
    s2 = float(np.sum((f * f) * w[:, None]))
    if s2 == 0:
        return None
    return round(e2 / s2, 6)


def tensor_type(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "mlp":
            rest = ".".join(parts[i + 1:])
            for t in ("gate_proj", "up_proj", "down_proj", "gate", "shared_expert_gate"):
                if rest.startswith(t):
                    return t
            return "mlp_other"
        if p == "linear_attn":
            rest = ".".join(parts[i + 1:])
            for t in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                      "conv1d", "A_log", "dt_bias", "norm"):
                if rest.startswith(t):
                    return "linear_attn_" + t
            return "linear_attn_other"
        if p == "self_attn":
            rest = ".".join(parts[i + 1:])
            for t in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                if rest.startswith(t):
                    return "self_attn_" + t
            return "self_attn_other"
    for t in ("embed_tokens", "lm_head", "norm"):
        if name.endswith(t + ".weight") or name.endswith(t):
            return t
    return "other"


def layer_idx(name):
    m = name.split(".layers.")
    if len(m) > 1:
        try:
            return int(m[1].split(".")[0])
        except ValueError:
            return None
    return None


def process_tensor(path, name, meta, data_start, scale_inv_meta=None, scale_inv_data_start=None):
    import numpy as np
    a, b = meta["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(data_start + a)
        raw = fh.read(b - a)
    t0 = time.time()
    f = decode(np.frombuffer(raw, dtype=np.uint8), meta["dtype"])
    shape = tuple(meta["shape"])
    if len(shape) >= 2 and f.size == int(np.prod(shape)):
        f = f.reshape(shape)
    # dequant FP8 fine-grained par bloc 128x128 si scale_inv dispo
    dequant = None
    if meta["dtype"] == "F8_E4M3" and scale_inv_meta is not None:
        sa, sb = scale_inv_meta["data_offsets"]
        with open(path, "rb") as fh:
            fh.seek(scale_inv_data_start + sa)
            sraw = np.frombuffer(fh.read(sb - sa), dtype=np.uint8)
        f = dequant_fp8(f, sraw, shape)
        dequant = "fp8_block128"
    n = f.size
    fm = np.abs(f)
    rms = float(np.sqrt(np.mean(f.astype(np.float64) ** 2)))
    mx = float(np.max(fm)) if n else 0.0
    outlier = int(np.sum(fm > 4 * rms)) if rms > 0 and n else 0
    sparse = int(np.sum(fm < 1e-6)) if n else 0
    # entropie approx (128 bins sur plage [-max,max])
    hist = np.histogram(f, bins=128, range=(-mx, mx))[0] if mx > 0 else np.zeros(1)
    p = hist[hist > 0] / hist.sum()
    entropy = float(-np.sum(p * np.log2(p)))
    stats = {
        "name": name, "dtype": meta["dtype"], "shape": meta["shape"],
        "n": n, "bytes": b - a,
        "rms": round(rms, 6), "max": round(mx, 4),
        "outlier_rate": round(outlier / n, 7) if n else 0.0,
        "sparsity": round(sparse / n, 7) if n else 0.0,
        "entropy_bits": round(entropy, 3),
        "decode_ms": round((time.time() - t0) * 1000, 1),
        "dequant": dequant,
    }
    if n < 16 or f.ndim < 2:
        stats["precision"] = {}
        return stats
    qerr = {name_: block_quant_error(f, bits) for name_, bits in PRECS}
    for name_, bits in PRECS:
        qerr[name_]["channel_weighted_rel"] = channel_score(f, bits)
    stats["precision"] = qerr
    return stats


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 TENSOR PROFILER — full model")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--json", default=OUT)
    ap.add_argument("--limit", type=int, default=0, help="limite tensors (test)")
    ap.add_argument("--layer", type=int, default=0, help="profil complet (0) ou une couche")
    args = ap.parse_args()

    import numpy as np  # noqa
    shards = sorted(f for f in os.listdir(args.model) if f.endswith(".safetensors"))
    tensors = []
    shard_scale = {}
    for sh in shards:
        hdr, data_start = read_header(os.path.join(args.model, sh))
        # index des weight_scale_inv dans ce shard (companion dequant)
        scale_idx = {}
        for name, meta in hdr.items():
            if name.endswith(".weight_scale_inv"):
                base = name[: -len("_scale_inv")]  # "...up_proj.weight"
                scale_idx[base] = (meta, data_start)
        shard_scale[sh] = scale_idx
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            if args.layer:
                li = layer_idx(name)
                if li != args.layer:
                    continue
            tensors.append((sh, name, meta, data_start))
            if args.limit and len(tensors) >= args.limit:
                break
        if args.limit and len(tensors) >= args.limit:
            break

    t_start = time.time()
    results = []
    for i, (sh, name, meta, data_start) in enumerate(tensors):
        try:
            if name.endswith("weight_scale_inv"):
                st = process_tensor(os.path.join(args.model, sh), name, meta, data_start)
            else:
                sc = shard_scale.get(sh, {}).get(name)
                st = process_tensor(os.path.join(args.model, sh), name, meta, data_start,
                                    scale_inv_meta=sc[0] if sc else None,
                                    scale_inv_data_start=sc[1] if sc else None)
            results.append(st)
        except Exception as e:
            results.append({"name": name, "error": str(e)})
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(tensors)}] {elapsed:.0f}s", flush=True)

    # aggregations
    from collections import defaultdict
    agg = defaultdict(lambda: {"count": 0, "bytes": 0,
                               "snr": {"Q4": [], "Q3": [], "Q2": []},
                               "rel": {"Q4": [], "Q3": [], "Q2": []},
                               "chan": {"Q4": [], "Q3": [], "Q2": []},
                               "outlier": []})
    for r in results:
        if "error" in r or "precision" not in r:
            continue
        tt = tensor_type(r["name"])
        li = layer_idx(r["name"])
        key = f"L{li:02d}/{tt}" if li is not None else tt
        a = agg[key]
        a["count"] += 1
        a["bytes"] += r["bytes"]
        for prec in ("Q4", "Q3", "Q2"):
            pe = r["precision"].get(prec, {})
            a["snr"][prec].append(pe.get("snr_db", 0))
            a["rel"][prec].append(pe.get("rel_err", 0))
            if pe.get("channel_weighted_rel") is not None:
                a["chan"][prec].append(pe["channel_weighted_rel"])
        a["outlier"].append(r["outlier_rate"])
    agg_out = {}
    for k, a in sorted(agg.items()):
        def avg(l):
            return round(sum(l) / len(l), 5) if l else None
        agg_out[k] = {
            "count": a["count"], "gb": round(a["bytes"] / 1e9, 3),
            "snr_db_Q4": avg(a["snr"]["Q4"]),
            "snr_db_Q3": avg(a["snr"]["Q3"]),
            "snr_db_Q2": avg(a["snr"]["Q2"]),
            "rel_err_Q4": avg(a["rel"]["Q4"]),
            "rel_err_Q2": avg(a["rel"]["Q2"]),
            "channel_ratio_Q4": avg(a["chan"]["Q4"]),
            "outlier_mean": avg(a["outlier"]),
        }

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen3.8-27B-FP8",
        "scope": f"{len(results)}/{len(tensors)} tensors (full model)" if not args.limit else f"limit={args.limit}",
        "elapsed_s": round(time.time() - t_start, 1),
        "tensor_type": {
            "embed_tokens": "vocab x hidden -> critique (freq acces elevee)",
            "lm_head": "vocab x hidden -> critique (logits finaux)",
            "self_attn_q/k/v/o_proj": "full-attention 16 couches",
            "linear_attn_*": "GDN/SSM 48 couches -> carte officielle BF16 protege",
            "mlp_gate/up/down_proj": "FFN 17.38 GB -> cible selective Q4/Q3/Q2",
        },
        "aggregated": agg_out,
        "tensors": results,
        "caveats": [
            "channel_weighted_rel = proxy HEURISTIC (RMS de ligne) SANS runtime forward",
            "remplacer par E[||X(W-Wq)||^2]/E[||XW||^2] via activations reelles (AWQ) quand le runtime existe",
            "block_quant_error = block-scaled group 32, pas un GGUF K-quant exact",
            "Q1 = ternaire {-1,0,+1} (pas 1 bit pur)",
        ],
    }
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # Also save to registry
    registry_path = os.path.join(HERE, "d2_ecosystem", "registry.json")
    reg = D2Registry(model=QWEN38_27B)
    if os.path.exists(registry_path):
        reg.load(registry_path)
    
    for r in results:
        if "error" in r or "precision" not in r:
            continue
        name = r.get("name", "")
        tensor_id = _hf_to_tensor_id(name)
        
        # Get best precision metrics
        prec = r.get("precision", {})
        best_snr = 0
        best_rel = 0
        for fmt in ("Q4", "Q3", "Q2"):
            m = prec.get(fmt, {})
            if m.get("snr_db", 0) > best_snr:
                best_snr = m["snr_db"]
            if m.get("rel_err", 0) > best_rel:
                best_rel = m["rel_err"]
        
        reg.add_tensor_noise(tensor_id, snr_db=best_snr, rel_err=best_rel)
        reg.add_tensor_latency(tensor_id, decode_ms=r.get("decode_ms", 0))
        reg.add_tensor_shape(tensor_id, shape=r.get("shape", []),
                            original_bytes=r.get("bytes", 0),
                            original_precision=r.get("dtype", "FP8"))
    
    reg.save(registry_path)
    print(f"[+] Registry updated: {registry_path}")
    print(f"[+] {len(results)} tensors profiles en {report['elapsed_s']}s -> {args.json}")
    print("    MLP decomposition (L=mean sur couches):")
    for k in sorted(agg_out):
        if "/mlp_" in k or k.startswith("mlp_") and k not in ("mlp_other",):
            a = agg_out[k]
            print(f"      {k:28s} gb={a['gb']:7.3f} snrQ4={a['snr_db_Q4']:7.2f} "
                  f"relQ2={a['rel_err_Q2']:.5f} chan={a['channel_ratio_Q4']} out={a['outlier_mean']:.5f}")


def _hf_to_tensor_id(hf_name: str) -> str:
    """Convert HuggingFace tensor name to canonical tensor_id."""
    parts = hf_name.split(".")
    if "layers" in parts:
        idx = parts.index("layers")
        if idx + 1 < len(parts):
            layer_num = parts[idx + 1]
            remaining = parts[idx + 2:]
            mapping = {
                "mlp.down_proj": "ffn_down",
                "mlp.gate_proj": "ffn_gate",
                "mlp.up_proj": "ffn_up",
                "self_attn.q_proj": "attn_q",
                "self_attn.k_proj": "attn_k",
                "self_attn.v_proj": "attn_v",
                "self_attn.o_proj": "attn_o",
                "self_attn.qkv_proj": "attn_qkv",
                "linear_attn.in_proj_a": "attn_qkv",
                "linear_attn.out_proj": "attn_o",
                "linear_attn.conv1d": "ssm_conv1d",
                "input_layernorm": "input_layernorm",
                "post_attention_layernorm": "post_attention_layernorm",
            }
            for hf_prefix, canon_name in mapping.items():
                rem_str = ".".join(remaining)
                if rem_str.startswith(hf_prefix):
                    suffix = rem_str[len(hf_prefix):]
                    suffix = suffix.lstrip(".")
                    return f"blk.{layer_num}.{canon_name}{'.' + suffix if suffix else ''}"
    return hf_name


if __name__ == "__main__":
    main()