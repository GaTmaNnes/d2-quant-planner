#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic attn_qkv — SNR 2.7 dB suspect pour un Q4_K (test #6).

Compare la déquantification réelle GGUF vs la référence FP8 pour les tensors
de la couche 0 (GDN), en testant les DEUX orientations (as-is et transposée)
pour vérifier si l'anomalie est un artefact de layout ou une vraie perte.
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gguf import GGUFReader
from d2_kld_real_profiler import (
    dequant_gguf_tensor, gguf_to_hf, load_fp8_dequanted,
)

GGUF = "models/Qwen3.8-27B-D2-ECO.gguf"
FP8 = "models/Qwen3.8-27B-FP8"
LI = 0  # couche 0 = linear_attention (GDN)


def metrics(ref, quant):
    rf = ref.flatten().astype(np.float32)
    qf = quant.flatten().astype(np.float32)
    ml = min(len(rf), len(qf))
    rf, qf = rf[:ml], qf[:ml]
    d = rf - qf
    sp = float(np.sum(rf ** 2))
    noise = float(np.sum(d ** 2)) + 1e-10
    return {
        "rel_l2": float(np.sqrt(np.sum(d ** 2)) / (np.sqrt(sp) + 1e-10)),
        "snr_db": round(float(10 * np.log10(sp / noise)), 2),
        "cosine": round(float(np.dot(rf, qf) / (np.linalg.norm(rf) * np.linalg.norm(qf) + 1e-10)), 6),
    }


def main():
    print(f"GGUF : {GGUF}")
    print(f"FP8  : {FP8}")
    r = GGUFReader(GGUF)
    fp8 = load_fp8_dequanted(FP8, LI)
    print(f"tensors FP8 L{LI} : {sorted(fp8.keys())}\n")

    targets = ["blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
               "blk.0.attn_qkv.weight", "blk.0.ffn_down.weight",
               "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"]
    # contrôle : couche 3 (attention) attn_k
    gt3 = next(t for t in r.tensors if t.name == "blk.3.attn_k.weight")
    fp8_3 = load_fp8_dequanted(FP8, 3)

    for name in targets:
        gt = next(t for t in r.tensors if t.name == name)
        hf = gguf_to_hf(gt.name)
        ref = fp8.get(hf)
        if ref is None:
            print(f"{name:28s} pas de réf FP8 ({hf})")
            continue
        g = dequant_gguf_tensor(gt)
        print(f"{name:28s} shape GGUF={g.shape}  HF={ref.shape}  type={gt.tensor_type}")
        m0 = metrics(ref, g)
        mT = metrics(ref, g.T) if g.ndim == 2 else None
        print(f"    as-is   : SNR={m0['snr_db']:6.2f} dB  cos={m0['cosine']:.4f}")
        if mT:
            print(f"    transposé: SNR={mT['snr_db']:6.2f} dB  cos={mT['cosine']:.4f}")
        print()

    # contrôle attn_k (couche 3, attention classique)
    if gt3 is not None and fp8_3:
        hf = gguf_to_hf(gt3.name)
        ref = fp8_3.get(hf)
        g = dequant_gguf_tensor(gt3)
        print(f"{gt3.name:28s} shape GGUF={g.shape}  HF={ref.shape if ref is not None else '?'}")
        if ref is not None:
            m0 = metrics(ref, g)
            mT = metrics(ref, g.T) if g.ndim == 2 else None
            print(f"    as-is   : SNR={m0['snr_db']:6.2f} dB  cos={m0['cosine']:.4f}")
            if mT:
                print(f"    transposé: SNR={mT['snr_db']:6.2f} dB  cos={mT['cosine']:.4f}")


if __name__ == "__main__":
    main()
