#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 KLD REAL PROFILER v2 — True FP8 vs GGUF dequantization
===========================================================

Uses exact llama.cpp dequantization algorithms from ggml-quants.c.

Usage:
  python d2_kld_real_profiler.py --layers 0-3
  python d2_kld_real_profiler.py --layers 0-63
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


def gguf_to_hf(name):
    if name.startswith("blk."):
        parts = name.split(".", 2)
        li, c = parts[1], parts[2]
        m = {
            # CORRIGÉ 22/08/2026 (mapping vérifié par taille d'éléments sur
            # models/Qwen3.8-27B-FP8/layers-0.safetensors vs D2-ECO.gguf) :
            #   ssm_alpha (5120,48) <-> in_proj_a (48,5120)
            #   ssm_beta  (5120,48) <-> in_proj_b (48,5120)
            #   attn_gate (5120,6144) <-> in_proj_z (6144,5120)
            #   ssm_out   (6144,5120) <-> out_proj (5120,6144)
            #   attn_qkv  (5120,10240) <-> in_proj_qkv (10240,5120)
            "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
            "attn_gate.weight": "linear_attn.in_proj_z.weight",
            "attn_norm.weight": "linear_attn.norm.weight",
            "ffn_down.weight": "mlp.down_proj.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight",
            "ffn_up.weight": "mlp.up_proj.weight",
            "post_attention_norm.weight": "post_attention_layernorm.weight",
            "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
            "ssm_beta.weight": "linear_attn.in_proj_b.weight",
            "ssm_out.weight": "linear_attn.out_proj.weight",
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


def load_fp8_dequanted(fp8_dir, li):
    """Load layer FP8 safetensors and dequantize to float32 numpy."""
    import torch
    from safetensors.torch import load_file

    fp = os.path.join(fp8_dir, f"layers-{li}.safetensors")
    if not os.path.exists(fp):
        return {}
    raw = load_file(fp)
    result = {}
    for k, v in raw.items():
        if v.dtype == torch.float8_e4m3fn:
            scale_key = k + "_scale_inv"
            if scale_key in raw:
                scale = raw[scale_key]
                fp8_f32 = v.to(torch.float32)
                scale_f32 = scale.to(torch.float32)
                if fp8_f32.dim() == 2 and scale_f32.dim() == 2 and scale_f32.shape[0] > 0 and scale_f32.shape[1] > 0:
                    block_r = fp8_f32.shape[0] // scale_f32.shape[0]
                    block_c = fp8_f32.shape[1] // scale_f32.shape[1]
                    if block_r > 0 and block_c > 0:
                        se = scale_f32.repeat_interleave(block_r, dim=0).repeat_interleave(block_c, dim=1)
                        result[k] = (fp8_f32 * se).numpy()
                    else:
                        result[k] = fp8_f32.numpy()
                else:
                    result[k] = fp8_f32.numpy()
            else:
                result[k] = v.to(torch.float32).numpy()
        else:
            result[k] = v.to(torch.float32).numpy()
    return result


# ─── Exact GGUF Dequantization (from ggml-quants.c) ─────────────────────────

QK_K = 256


def f16_to_f32(raw_bytes):
    """Convert 2-byte float16 to float32."""
    return np.frombuffer(raw_bytes[:2], dtype=np.float16).astype(np.float32).item()


def dequant_q2_K_batch(raw, n_blocks):
    """Vectorized batch dequantize_row_q2_K — processes all blocks at once with NumPy.

    Layout per block (84 bytes): scales[16](u8) + qs[64](u8) + d(f16) + dmin(f16)
    """
    if n_blocks == 0:
        return np.zeros(0, dtype=np.float32)
    raw = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 84)
    scales = raw[:, :16]           # (n_blocks, 16)
    qs = raw[:, 16:80]             # (n_blocks, 64)
    d_f16 = raw[:, 80:82].copy()   # (n_blocks, 2)
    dmin_f16 = raw[:, 82:84].copy()
    d = d_f16.view(np.float16).astype(np.float32).ravel()       # (n_blocks,)
    dmin = dmin_f16.view(np.float16).astype(np.float32).ravel()

    y = np.zeros((n_blocks, QK_K), dtype=np.float32)

    # For each block: 2 halves (n=0,128), 4 sub-blocks of 32 each
    for n in (0, 128):
        for j in range(4):
            si = j * 2       # 2 scales per sub-block (low nibble, high nibble)
            # First 16 elements — low nibble of scale[si]
            dl = d * (scales[:, si] & 0xF).astype(np.float32)
            ml = dmin * (scales[:, si] >> 4).astype(np.float32)
            qvals = (qs[:, (j % 2) * 32:(j % 2) * 32 + 16] >> (j // 2 * 2)) & 3
            qvals_i8 = np.where(qvals >= 2, qvals.astype(np.int32) - 4, qvals.astype(np.int32))
            y[:, n + j * 32:n + j * 32 + 16] = dl[:, None] * qvals_i8 - ml[:, None]

            # Next 16 elements — high nibble of scale[si+1]
            dl = d * (scales[:, si + 1] & 0xF).astype(np.float32)
            ml = dmin * (scales[:, si + 1] >> 4).astype(np.float32)
            qvals = (qs[:, (j % 2) * 32 + 16:(j % 2) * 32 + 32] >> (j // 2 * 2)) & 3
            qvals_i8 = np.where(qvals >= 2, qvals.astype(np.int32) - 4, qvals.astype(np.int32))
            y[:, n + j * 32 + 16:n + j * 32 + 32] = dl[:, None] * qvals_i8 - ml[:, None]

    return y.ravel()


def dequant_q2_K(block_bytes):
    """Exact dequantize_row_q2_K from ggml-quants.c line 983 (single-block, for compatibility)."""
    if len(block_bytes) < 84:
        return np.zeros(QK_K, dtype=np.float32)
    return dequant_q2_K_batch(block_bytes, 1)


def int8_t(x):
    """Simulate C int8_t cast (sur entiers Python, pas de wrap numpy)."""
    x = int(x) & 0xFF
    return x if x < 128 else x - 256


def dequant_q3_K(block_bytes):
    """Exact dequantize_row_q3_K de ggml-quants.c (fork beellama.cpp, ligne 1327).

    Layout réel du block_q3_K (110 octets, ggml-common.h) :
      hmask (32) + qs (64) + scales (12) + d (f16, 2)
    Le décodage des 16 scales int8 se fait via le shuffle uint32 (aux[4]).
    """
    if len(block_bytes) < 110:
        return np.zeros(QK_K, dtype=np.float32)

    d = f16_to_f32(block_bytes[108:110])
    hmask = np.frombuffer(block_bytes[0:32], dtype=np.uint8)
    qs = np.frombuffer(block_bytes[32:96], dtype=np.uint8)
    scales_raw = block_bytes[96:108]

    # Shuffle uint32 exact du C : aux[4] d'octets, vu comme 16 int8
    KMASK1 = 0x03030303
    KMASK2 = 0x0f0f0f0f
    b = [int(x) for x in scales_raw] + [0] * 4
    aux = [(b[0] | (b[4] << 8) | (b[8] << 16)),
           (b[1] | (b[5] << 8) | (b[9] << 16)),
           (b[2] | (b[6] << 8) | (b[10] << 16)),
           (b[3] | (b[7] << 8) | (b[11] << 16))]
    tmp = aux[2]
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4)
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4)
    aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4)
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4)

    scales = []
    for word in aux:
        for k in range(4):
            s = (word >> (8 * k)) & 0xFF
            scales.append(s if s < 128 else s - 256)

    y = np.zeros(QK_K, dtype=np.float32)
    is_ = 0
    m = 1
    q = qs
    for n in (0, 128):
        shift = 0
        for j in range(4):
            dl = d * (scales[is_] - 32); is_ += 1
            for l in range(16):
                val = int((q[l] >> shift) & 3)
                hi = int(hmask[l] & m)
                y[n + j * 32 + l] = dl * (val - (0 if hi else 4))
            dl = d * (scales[is_] - 32); is_ += 1
            for l in range(16):
                val = int((q[16 + l] >> shift) & 3)
                hi = int(hmask[16 + l] & m)
                y[n + j * 32 + 16 + l] = dl * (val - (0 if hi else 4))
            shift += 2
            m <<= 1
        q = qs[32:]

    return y


def get_scale_min_k4(j, scales_bytes):
    """Exact get_scale_min_k4 from ggml-quants.c line 902."""
    if j < 4:
        d = scales_bytes[j] & 63
        m = scales_bytes[j + 4] & 63
    else:
        d = (scales_bytes[j + 4] & 0xF) | ((scales_bytes[j - 4] >> 6) << 4)
        m = (scales_bytes[j + 4] >> 4) | ((scales_bytes[j] >> 6) << 4)
    return d, m


def dequant_q4_K_batch(raw, n_blocks):
    """Vectorized batch dequantize_row_q4_K — all blocks at once with NumPy.

    Layout per block (144 bytes): d(f16,2) + dmin(f16,2) + scales[12](u8) + qs[128](u8)
    """
    if n_blocks == 0:
        return np.zeros(0, dtype=np.float32)
    raw = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 144)
    d_f16 = raw[:, :2].copy()
    dmin_f16 = raw[:, 2:4].copy()
    d = d_f16.view(np.float16).astype(np.float32).ravel()
    dmin = dmin_f16.view(np.float16).astype(np.float32).ravel()
    scales = raw[:, 4:16]  # (n_blocks, 12)
    qs = raw[:, 16:144]    # (n_blocks, 128)

    y = np.zeros((n_blocks, QK_K), dtype=np.float32)

    for j in range(0, QK_K, 64):
        is_idx = j // 32  # 0,2,4,6 (2 scales per 64-element group)
        # get_scale_min_k4 logic — use int view for bitwise ops, then cast to float
        sc_raw = scales.astype(np.int32)  # (n_blocks, 12) as int for bitwise ops
        mask = is_idx < 4

        if mask:
            d1 = d * (sc_raw[:, is_idx] & 63).astype(np.float32)
            m1v = dmin * (sc_raw[:, is_idx + 4] & 63).astype(np.float32)
        else:
            d1 = d * (((sc_raw[:, is_idx + 4] & 0xF) | ((sc_raw[:, is_idx - 4] >> 6) & 0xF) * 16)).astype(np.float32)
            m1v = dmin * (((sc_raw[:, is_idx + 4] >> 4) | ((sc_raw[:, is_idx] >> 6) & 0xF) * 16)).astype(np.float32)
        lo_nib = (qs[:, j // 2:j // 2 + 32].astype(np.int32) & 0xF).astype(np.float32)
        y[:, j:j + 32] = d1[:, None] * lo_nib - m1v[:, None]

        # Second 32 of this group: sc/m from scales[is_idx+1]
        is2 = is_idx + 1
        mask2 = is2 < 4
        if mask2:
            d2 = d * (sc_raw[:, is2] & 63).astype(np.float32)
            m2v = dmin * (sc_raw[:, is2 + 4] & 63).astype(np.float32)
        else:
            d2 = d * (((sc_raw[:, is2 + 4] & 0xF) | ((sc_raw[:, is2 - 4] >> 6) & 0xF) * 16)).astype(np.float32)
            m2v = dmin * (((sc_raw[:, is2 + 4] >> 4) | ((sc_raw[:, is2] >> 6) & 0xF) * 16)).astype(np.float32)
        hi_nib = (qs[:, j // 2:j // 2 + 32].astype(np.int32) >> 4).astype(np.float32)
        y[:, j + 32:j + 64] = d2[:, None] * hi_nib - m2v[:, None]

    return y.ravel()


def dequant_q4_K(block_bytes):
    """Exact dequantize_row_q4_K from ggml-quants.c line 1551 (single-block, for compatibility)."""
    if len(block_bytes) < 144:
        return np.zeros(QK_K, dtype=np.float32)
    return dequant_q4_K_batch(block_bytes, 1)


def dequant_q6_K(block_bytes):
    """Approximate dequantize_row_q6_K from ggml-quants.c line 1961."""
    if len(block_bytes) < 210:
        return np.zeros(QK_K, dtype=np.float32)

    # Struct layout: ql[128] + qh[64] + scales[16] + d(f16)
    ql = np.frombuffer(block_bytes[0:128], dtype=np.uint8)
    qh = np.frombuffer(block_bytes[128:192], dtype=np.uint8)
    scales = np.frombuffer(block_bytes[192:208], dtype=np.int8)
    d = f16_to_f32(block_bytes[208:210])

    y = np.zeros(QK_K, dtype=np.float32)
    is_idx = 0

    for n in range(0, QK_K, 128):
        for j in range(4):
            dl = d * scales[is_idx]; is_idx += 1
            for l in range(16):
                lo = (ql[j*32 + l] & 0xF) | (((qh[j*32 + l] >> 0) & 3) << 4)
                y[n + j*32 + l] = dl * (int8_t(lo) - 32)

            dl = d * scales[is_idx]; is_idx += 1
            for l in range(16):
                lo = (ql[j*32 + 16 + l] & 0xF) | (((qh[j*32 + 16 + l] >> 2) & 3) << 4)
                y[n + j*32 + 16 + l] = dl * (int8_t(lo) - 32)

    return y


def dequant_gguf_tensor(tensor):
    """Dequantize a GGUF tensor to float32 numpy array.

    Utilise la SHAPE LOGIQUE (tensor.shape = ne). Les GGUF stockent les poids
    transposés par rapport à HF : on reshape en (ne1, ne0) pour comparaison directe.
    """
    raw_bytes = tensor.data.flatten().tobytes()
    tid = tensor.tensor_type
    ne0, ne1 = int(tensor.shape[0]), int(tensor.shape[1])
    total_elements = ne0 * ne1

    if tid == 0:  # F32
        return np.frombuffer(raw_bytes[:total_elements*4], dtype=np.float32).reshape(ne1, ne0)

    block_sizes = {10: 84, 11: 110, 12: 144, 14: 210}
    batch_fns = {10: dequant_q2_K_batch, 11: None, 12: dequant_q4_K_batch, 14: None}
    bs = block_sizes.get(tid)
    batch_fn = batch_fns.get(tid)

    if not bs:
        return np.zeros(total_elements, dtype=np.float32).reshape(ne1, ne0)

    n_blocks = len(raw_bytes) // bs
    if n_blocks == 0:
        return np.zeros(total_elements, dtype=np.float32).reshape(ne1, ne0)

    if batch_fn is not None:
        result = batch_fn(raw_bytes[:n_blocks * bs], n_blocks)[:total_elements]
    else:
        # Fallback: per-block (Q3_K, Q6_K — rares)
        fallback = {10: dequant_q2_K, 11: dequant_q3_K, 12: dequant_q4_K, 14: dequant_q6_K}.get(tid)
        vals = [fallback(raw_bytes[i:i+bs]) for i in range(0, n_blocks * bs, bs)]
        result = np.concatenate(vals)[:total_elements]

    return result.reshape(ne1, ne0)


def compute_metrics(ref, quant):
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
    ap.add_argument("--output", default=os.path.join(HERE, "d2_kld_real_report.json"))
    args = ap.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  D2 KLD REAL PROFILER v2 — True Dequantization")
    print("=" * 70)

    from gguf import GGUFReader
    print("  Loading GGUF...")
    t0 = time.time()
    r = GGUFReader(args.gguf)
    print(f"  GGUF: {len(r.tensors)} tensors ({time.time()-t0:.1f}s)")

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
    type_names = {10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}

    for li in layers:
        t0 = time.time()
        fp8 = load_fp8_dequanted(args.fp8, li)
        lt = time.time() - t0
        if not fp8:
            continue

        layer_tensors = [t for t in r.tensors
                         if t.name.startswith(f"blk.{li}.")
                         and t.name.endswith(".weight")
                         and "scale_inv" not in t.name
                         and t.tensor_type != 0]

        t1 = time.time()
        lr = []
        for gt in layer_tensors:
            hf = gguf_to_hf(gt.name)
            if not hf or hf not in fp8:
                continue

            ref = fp8[hf]

            try:
                gguf_d = dequant_gguf_tensor(gt)
            except Exception as e:
                continue

            # Match shapes (ref HF [out,in] ; gguf_d déjà en [out,in])
            ref_ne = ref.shape[0] * ref.shape[1] if ref.ndim >= 2 else ref.shape[0]
            gd_ne = gguf_d.shape[0] * gguf_d.shape[1] if gguf_d.ndim >= 2 else gguf_d.shape[0]
            if ref.shape != gguf_d.shape and ref_ne != gd_ne:
                print(f"    ⚠ SHAPE MISMATCH {gt.name}: ref={ref.shape} ({ref_ne}e) gguf={gguf_d.shape} ({gd_ne}e) — SKIPPED")
                continue
            if ref.shape != gguf_d.shape:
                # même nombre d'éléments mais layout différent (ex: transpose) — on aplatit
                ref = ref.flatten()
                gguf_d = gguf_d.flatten()

            m = compute_metrics(ref, gguf_d)
            lr.append({
                "layer": li, "gguf_name": gt.name, "hf_name": hf,
                "quant_type": type_names.get(gt.tensor_type, f"T{gt.tensor_type}"),
                "shape": list(ref.shape) if hasattr(ref, 'shape') else [len(ref)],
                "hf_mb": round(ref.nbytes / 1e6, 2) if hasattr(ref, 'nbytes') else 0,
                **m,
            })

        pt = time.time() - t1

        if lr:
            valid = [r for r in lr if not np.isnan(r["snr_db"])]
            if valid:
                avg_snr = sum(r["snr_db"] for r in valid) / len(valid)
                avg_cos = sum(r["cosine"] for r in valid) / len(valid)
                types = set(r["quant_type"] for r in valid)
                flag = " ***LOW***" if avg_snr < 8 else ""
                print(f"  L{li:02d}: {len(lr)}t  load={lt:.1f}s  dequant={pt:.1f}s  SNR={avg_snr:.1f}dB  cos={avg_cos:.4f} [{','.join(sorted(types))}]{flag}")
            else:
                print(f"  L{li:02d}: {len(lr)}t  ALL NaN")
            all_results.extend(lr)

    # Report
    print()
    print("=" * 70)
    print("  REAL KLD RESULTS")
    print("=" * 70)

    if not all_results:
        print("  No results!")
        return

    by_type = defaultdict(list)
    for r in all_results:
        if not np.isnan(r["snr_db"]):
            by_type[r["quant_type"]].append(r)

    print(f"\n  {'Type':8s} {'N':5s} {'SNR dB':8s} {'Cosine':8s} {'L2 Err':8s}")
    print("  " + "-" * 45)
    for qt in sorted(by_type.keys()):
        ts = by_type[qt]
        asnr = sum(t["snr_db"] for t in ts) / len(ts)
        acos = sum(t["cosine"] for t in ts) / len(ts)
        al2 = sum(t["rel_l2"] for t in ts) / len(ts)
        print(f"  {qt:8s} {len(ts):5d} {asnr:8.1f} {acos:8.4f} {al2:8.4f}")

    all_valid = [r for r in all_results if not np.isnan(r["snr_db"])]
    print(f"\n  LOWEST SNR (real dequantization):")
    for r in sorted(all_valid, key=lambda x: x["snr_db"])[:15]:
        print(f"    L{r['layer']:02d} {r['gguf_name']:45s} {r['quant_type']:6s} SNR={r['snr_db']:6.1f}  cos={r['cosine']:.4f}")

    print(f"\n  HIGHEST RELATIVE ERROR:")
    for r in sorted(all_valid, key=lambda x: x["rel_l2"], reverse=True)[:15]:
        print(f"    L{r['layer']:02d} {r['gguf_name']:45s} {r['quant_type']:6s} L2={r['rel_l2']:.4f}  SNR={r['snr_db']:6.1f}dB")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "real_dequant_llama_cpp",
        "n_tensors": len(all_results),
        "n_valid": len(all_valid),
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
