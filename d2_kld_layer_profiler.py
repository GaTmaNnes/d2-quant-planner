#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 KLD LAYER PROFILER - FP8 reference vs quantized per tensor (v3)
===================================================================
Lightweight GGUF metadata reader + vectorized quantization simulation.

Usage:
  python d2_kld_layer_profiler.py --layers 0-3
  python d2_kld_layer_profiler.py --all
"""

import argparse
import json
import os
import struct
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FP8 = os.path.join(HERE, "models", "Qwen3.8-27B-FP8")
DEFAULT_GGUF = os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO.gguf")

GGML_TYPES = {
    0: ("F32", 32.0), 1: ("F16", 16.0),
    2: ("Q4_0", 4.5), 3: ("Q4_1", 5.5),
    6: ("Q5_0", 5.5), 7: ("Q5_1", 6.5),
    8: ("Q8_0", 8.5), 9: ("Q8_1", 10.5),
    10: ("Q2_K", 2.7), 11: ("Q3_K", 3.4),
    12: ("Q4_K", 4.5), 13: ("Q5_K", 5.5),
    14: ("Q6_K", 6.6), 15: ("IQ4_NL", 4.0),
}


def gguf_to_hf(name):
    if name.startswith("blk."):
        parts = name.split(".", 2)
        li, c = parts[1], parts[2]
        m = {
            "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
            "attn_gate.weight": "linear_attn.in_proj_b.weight",
            "attn_norm.weight": "linear_attn.norm.weight",
            "ffn_down.weight": "mlp.down_proj.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight",
            "ffn_up.weight": "mlp.up_proj.weight",
            "post_attention_norm.weight": "post_attention_layernorm.weight",
            "ssm_alpha.weight": "linear_attn.in_proj_z.weight",
            "ssm_beta.weight": "linear_attn.out_proj.weight",
            "ssm_conv1d.weight": "linear_attn.conv1d.weight",
            "ssm_dt.bias": "linear_attn.dt_bias",
            "ssm_A": "linear_attn.A_log",
            "attn_q.weight": "self_attn.q_proj.weight",
            "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight",
            "attn_o.weight": "self_attn.o_proj.weight",
        }
        hf = m.get(c)
        return f"model.language_model.layers.{li}.{hf}" if hf else None

    return {
        "token_embd.weight": "model.language_model.embed_tokens.weight",
        "output_norm.weight": "model.language_model.norm.weight",
        "output.weight": "lm_head.weight",
    }.get(name, name)


def read_gguf_names(path):
    """Read only tensor names and types from GGUF (no data)."""
    tensors = []
    with open(path, "rb") as f:
        f.seek(4 + 4)  # magic + version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        # Skip KV pairs with safe size tracking
        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            f.seek(klen, 1)  # skip key
            vtype = struct.unpack("<I", f.read(4))[0]

            type_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                          10: 8, 11: 8, 12: 8}

            if vtype == 8:  # STRING
                slen = struct.unpack("<Q", f.read(8))[0]
                f.seek(slen, 1)
            elif vtype == 9:  # ARRAY
                atype = struct.unpack("<I", f.read(4))[0]
                alen = struct.unpack("<Q", f.read(8))[0]
                if atype == 8:  # array of strings
                    for _ in range(alen):
                        slen = struct.unpack("<Q", f.read(8))[0]
                        f.seek(slen, 1)
                else:
                    f.seek(alen * type_sizes.get(atype, 0), 1)
            else:
                f.seek(type_sizes.get(vtype, 0), 1)

        # Read tensor names
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", f.read(8))[0]
            name = f.read(nlen).decode("utf-8").rstrip("\x00")
            ndims = struct.unpack("<I", f.read(4))[0]
            ne = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndims)]
            tid = struct.unpack("<I", f.read(4))[0]
            f.read(8)  # offset
            tensors.append({"name": name, "ne": ne, "type_id": tid})

    return tensors


def load_layer(fp8_dir, li):
    import torch
    from safetensors.torch import load_file
    fp = os.path.join(fp8_dir, f"layers-{li}.safetensors")
    if not os.path.exists(fp):
        return {}
    raw = load_file(fp)
    return {k: v.to(torch.float32).numpy() for k, v in raw.items()}


def sim_q(ref, qtype):
    flat = ref.flatten().astype(np.float32)
    n = len(flat)
    bs = 32
    pad = (bs - n % bs) % bs
    if pad:
        flat = np.pad(flat, (0, pad))
    blocks = flat.reshape(-1, bs)

    if qtype == "Q8_0":
        s = np.max(np.abs(blocks), axis=1, keepdims=True) / 127.0
        s = np.where(s == 0, 1.0, s)
        out = (np.clip(np.round(blocks / s), -128, 127) * s).flatten()[:n]
    elif qtype in ("Q4_0", "IQ4_NL"):
        s = np.max(np.abs(blocks), axis=1, keepdims=True) / 8.0
        s = np.where(s == 0, 1.0, s)
        out = ((np.clip(np.round(blocks / s) + 8, 0, 15) - 8) * s).flatten()[:n]
    elif qtype == "Q5_0":
        s = np.max(np.abs(blocks), axis=1, keepdims=True) / 16.0
        s = np.where(s == 0, 1.0, s)
        out = ((np.clip(np.round(blocks / s) + 16, 0, 31) - 16) * s).flatten()[:n]
    elif qtype == "Q2_K":
        s = np.max(np.abs(blocks), axis=1, keepdims=True) / 1.5
        s = np.where(s == 0, 1.0, s)
        out = ((np.clip(np.round(blocks / s) + 2, 0, 3) - 2) * s).flatten()[:n]
    elif qtype == "Q3_K":
        s = np.max(np.abs(blocks), axis=1, keepdims=True) / 3.5
        s = np.where(s == 0, 1.0, s)
        out = ((np.clip(np.round(blocks / s) + 4, 0, 7) - 4) * s).flatten()[:n]
    elif qtype == "Q4_K":
        s = np.max(np.abs(blocks), axis=1, keepdims=True) / 8.0
        s = np.where(s == 0, 1.0, s)
        out = ((np.clip(np.round(blocks / s) + 8, 0, 15) - 8) * s).flatten()[:n]
    elif qtype == "Q5_K":
        s = np.max(np.abs(blocks), axis=1, keepdims=True) / 16.0
        s = np.where(s == 0, 1.0, s)
        out = ((np.clip(np.round(blocks / s) + 16, 0, 31) - 16) * s).flatten()[:n]
    elif qtype == "Q6_K":
        s = np.max(np.abs(blocks), axis=1, keepdims=True) / 32.0
        s = np.where(s == 0, 1.0, s)
        out = ((np.clip(np.round(blocks / s) + 32, 0, 63) - 32) * s).flatten()[:n]
    elif qtype == "F16":
        return ref.astype(np.float16).astype(np.float32)
    else:
        return ref.copy()
    return out.reshape(ref.shape)


def calc(ref, quant):
    rf = ref.flatten().astype(np.float32)
    qf = quant.flatten().astype(np.float32)
    ml = min(len(rf), len(qf))
    rf, qf = rf[:ml], qf[:ml]
    d = rf - qf
    sp = float(np.sum(rf ** 2))
    noise = float(np.sum(d ** 2)) + 1e-10
    return {
        "rel_l2": round(float(np.sqrt(np.sum(d ** 2)) / (np.sqrt(sp) + 1e-10)), 6),
        "snr_db": round(float(10 * np.log10(sp / noise)), 2),
        "cosine": round(float(np.dot(rf, qf) / (np.linalg.norm(rf) * np.linalg.norm(qf) + 1e-10)), 6),
        "mse": float(np.mean(d ** 2)),
        "max_err": round(float(np.max(np.abs(d))), 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", default=DEFAULT_FP8)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--layers", default="0-3")
    ap.add_argument("--output", default=os.path.join(HERE, "d2_kld_layer_report.json"))
    args = ap.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  D2 KLD LAYER PROFILER v3")
    print("=" * 70)

    t0 = time.time()
    gguf_tensors = read_gguf_names(args.gguf)
    print(f"  GGUF metadata: {len(gguf_tensors)} tensors ({time.time()-t0:.1f}s)")

    if args.layers == "all":
        layers = list(range(64))
    elif "-" in args.layers:
        lo, hi = map(int, args.layers.split("-"))
        layers = list(range(lo, hi + 1))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    print(f"  Layers: {layers}")
    print()

    all_results = []

    for li in layers:
        layer_gguf = [gt for gt in gguf_tensors
                      if gt["name"].startswith(f"blk.{li}.")
                      and gt["name"].endswith(".weight")
                      and "scale_inv" not in gt["name"]
                      and gt["type_id"] != 0]

        if not layer_gguf:
            continue

        t0 = time.time()
        fp8 = load_layer(args.fp8, li)
        lt = time.time() - t0
        if not fp8:
            continue

        t1 = time.time()
        lr = []
        for gt in layer_gguf:
            hf = gguf_to_hf(gt["name"])
            if not hf or hf not in fp8:
                continue
            ref = fp8[hf]
            tid = gt["type_id"]
            tn = GGML_TYPES.get(tid, ("?", 0))[0]
            q = sim_q(ref, tn)
            m = calc(ref, q)
            bpp = GGML_TYPES.get(tid, ("?", 0))[1]
            lr.append({
                "layer": li, "gguf_name": gt["name"], "hf_name": hf,
                "quant_type": tn, "shape": list(ref.shape),
                "hf_mb": round(ref.nbytes / 1e6, 2),
                "quant_mb": round(ref.size * bpp / 8 / 1e6, 2),
                "compression": round(ref.nbytes / (ref.size * bpp / 8 + 1), 2),
                **m,
            })
        pt = time.time() - t1

        if lr:
            avg_snr = sum(r["snr_db"] for r in lr) / len(lr)
            avg_cos = sum(r["cosine"] for r in lr) / len(lr)
            total_q = sum(r["quant_mb"] for r in lr)
            types = set(r["quant_type"] for r in lr)
            flag = " ***LOW***" if avg_snr < 8 else ""
            print(f"  L{li:02d}: {len(lr)}t  load={lt:.1f}s  sim={pt:.1f}s  SNR={avg_snr:.1f}dB  cos={avg_cos:.4f}  Q={total_q:.1f}MB [{','.join(sorted(types))}]{flag}")
            all_results.extend(lr)

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if not all_results:
        print("  No results!")
        return

    by_type = defaultdict(list)
    for r in all_results:
        by_type[r["quant_type"]].append(r)

    print(f"\n  {'Type':8s} {'N':5s} {'HF MB':8s} {'Q MB':8s} {'SNR dB':8s} {'Cosine':8s}")
    print("  " + "-" * 50)
    for qt in sorted(by_type.keys()):
        ts = by_type[qt]
        thf = sum(t["hf_mb"] for t in ts)
        tq = sum(t["quant_mb"] for t in ts)
        asnr = sum(t["snr_db"] for t in ts) / len(ts)
        acos = sum(t["cosine"] for t in ts) / len(ts)
        print(f"  {qt:8s} {len(ts):5d} {thf:8.1f} {tq:8.1f} {asnr:8.1f} {acos:8.4f}")

    print(f"\n  LOWEST SNR (most sensitive):")
    for r in sorted(all_results, key=lambda x: x["snr_db"])[:15]:
        print(f"    L{r['layer']:02d} {r['gguf_name']:45s} {r['quant_type']:6s} SNR={r['snr_db']:6.1f}  cos={r['cosine']:.4f}")

    print(f"\n  HIGHEST RELATIVE ERROR:")
    for r in sorted(all_results, key=lambda x: x["rel_l2"], reverse=True)[:15]:
        print(f"    L{r['layer']:02d} {r['gguf_name']:45s} {r['quant_type']:6s} L2={r['rel_l2']:.4f}  SNR={r['snr_db']:6.1f}dB")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_tensors": len(all_results),
        "layers": layers,
        "summary": {
            qt: {"count": len(ts), "avg_snr_db": round(sum(t["snr_db"] for t in ts)/len(ts), 1),
                 "avg_cosine": round(sum(t["cosine"] for t in ts)/len(ts), 4)}
            for qt, ts in by_type.items()
        },
        "tensors": all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  [+] {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
