#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 PRECISION OPTIMIZER — génère precision_map.json automatiquement.
====================================================================
Combine :
  - WEIGHT NOISE (SNR par précision, mesuré sur les vrais poids)
  - DYNAMIC SENSITIVITY (amplification récurrente pour les couches SSM)
  - MEMORY BENEFIT (octets économisés vs FP16)

Décision par layer :
  effective_err = rel_err(précision) * amplification(layer)
  récurrent (conv1d/A/in_proj) : garder FP16 sauf si effective_err < 0.01
  local (FFN/out_proj)         : NVFP4 si effective_err < 0.2, sinon INT8, sinon FP16
"""

import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(HERE, "beellama.cpp", "gguf-py"))
from gguf import GGUFReader

from d2_noise_profiler import (quant_e2m1_block, quant_int8, full_metrics,
                               load_gguf_conv1d, load_hf_ffn)

RHO = 0.99
AMP = 1.0 / (1.0 - RHO)  # amplification récurrente (pire cas)

# Couches récurrentes (erreur amplifiée dans l'état SSM) : seul conv1d est la porte d'entrée
RECURRENT_SUFFIXES = ("conv1d",)


def load_hf_generic(path, keep=("linear_attn.",), max_elems=1_000_000):
    """Charge les tenseurs BF16/F32 d'un shard safetensors (sans mmap), filtrés par `keep`."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    out = {}
    for name, info in hdr.items():
        if not any(k in name for k in keep):
            continue
        s, e = info["data_offsets"]
        dt = info["dtype"]
        nel = int(np.prod(info["shape"]))
        nread = min(nel, max_elems)
        with open(path, "rb") as f:
            f.seek(8 + hlen + s)   # data_offsets relatifs à la section data
            raw = f.read(nread * (4 if dt == "F32" else 2))
        if dt == "BF16":
            u = np.frombuffer(raw, dtype="<u2")
            arr = (u.astype(np.uint32) << 16).view(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype="<f4")
        else:
            continue
        cols = info["shape"][-1] if info["shape"] else 1
        arr = arr[: (nread // cols) * cols].reshape(nread // cols, cols).astype(np.float32)
        if np.isfinite(arr).all() and float(np.abs(arr).max()) < 1e6:
            out[name] = arr
    return out


def hf_to_gguf(name):
    """mapping HF -> nom GGUF (canonique)."""
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
    if "linear_attn.in_proj_a" in n:
        return "ssm_alpha.weight"
    if "linear_attn.in_proj_b" in n:
        return "ssm_beta.weight"
    if "linear_attn.out_proj" in n:
        return "ssm_out.weight"
    return None


def layer_num(name):
    if ".layers." in name:
        return name.split(".layers.")[1].split(".")[0]
    if "blk." in name:
        return name.split("blk.")[1].split(".")[0]
    return "?"


def recommend(snr_fp16, snr_int8, snr_fp4, recurrent, name=""):
    """Règle de décision -> ('NVFP4_SAFE'|'INT8_SAFE'|'FP16_REQUIRED', raison)."""
    ln = (name or "").lower()
    # Passe structurelle : mêmes verrous que d2_generate_precision_map.
    # ssm_conv1d / ssm_a / ssm_dt / norm sont critiques (récurrence/kernel F32) -> FP16.
    if ("ssm_conv1d" in ln or "norm" in ln
            or ln.endswith(".ssm_a") or ".ssm_dt" in ln):
        return "FP16_REQUIRED", "STRUCTURAL: SSM récurrent / normalisation (verrouillé FP16)"

    rel = lambda s: 10 ** (-s / 20.0)
    amp = AMP if recurrent else 1.0

    eff_fp4 = rel(snr_fp4) * amp
    eff_i8 = rel(snr_int8) * amp

    if recurrent:
        # la récurrence amplifie : seuil strict
        if eff_fp4 < 0.01:
            return "NVFP4_SAFE", f"eff_err={eff_fp4:.4f}"
        if eff_i8 < 0.01:
            return "INT8_SAFE", f"eff_err={eff_i8:.4f}"
        return "FP16_REQUIRED", f"amplif={amp:.0f}x -> eff_err INT8={eff_i8:.3f}"
    # local : seuil plus souple (10-11 dB NVFP4 ~ 0.3 erreur relative acceptée)
    if eff_fp4 < 0.5:
        return "NVFP4_SAFE", f"eff_err={eff_fp4:.4f}"
    if eff_i8 < 0.5:
        return "INT8_SAFE", f"eff_err={eff_i8:.4f}"
    return "FP16_REQUIRED", f"eff_err INT8={eff_i8:.3f}"


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    gguf = os.path.join(HERE, "models", "Qwen3.5-9B-Q4_K_S.gguf")
    shard2 = os.path.join(HERE, "hf_weights", "model.safetensors-00002-of-00004.safetensors")
    shard4 = os.path.join(HERE, "hf_weights", "model.safetensors-00004-of-00004.safetensors")

    measurements = {}   # gguf_name -> {snr fp16/int8/fp4, recurrent, bytes}

    # 1. conv1d (GGUF F32, fiable)
    for name, W in load_gguf_conv1d(gguf).items():
        Wt = W.T if (W.ndim > 1 and W.shape[0] < W.shape[1]) else W.reshape(1, -1)
        gg = "ssm_conv1d.weight"
        blk = layer_num(name)
        fp16 = full_metrics(Wt, Wt.astype(np.float16).astype(np.float32))
        i8 = full_metrics(Wt, quant_int8(Wt))
        f4 = full_metrics(Wt, quant_e2m1_block(Wt, scale_e4m3=True))
        measurements[f"blk.{blk}.{gg}"] = {
            "snr": {"FP16": fp16["snr"], "INT8": i8["snr"], "NVFP4": f4["snr"]},
            "recurrent": True, "bytes_fp16": Wt.size * 2,
        }

    # 2. FFN (HF shard 2)
    for name, W in load_hf_ffn(shard2).items():
        gg = hf_to_gguf(name)
        if not gg:
            continue
        blk = layer_num(name)
        i8 = full_metrics(W, quant_int8(W))
        f4 = full_metrics(W, quant_e2m1_block(W, scale_e4m3=True))
        measurements[f"blk.{blk}.{gg}"] = {
            "snr": {"FP16": 150.0, "INT8": i8["snr"], "NVFP4": f4["snr"]},
            "recurrent": False, "bytes_fp16": W.size * 2,
        }

    # 3. projections SSM (HF shard 4)
    if os.path.exists(shard4):
        for name, W in load_hf_generic(shard4, keep=("linear_attn.",)).items():
            gg = hf_to_gguf(name)
            if not gg:
                continue
            blk = layer_num(name)
            rec = any(s in name for s in RECURRENT_SUFFIXES)
            i8 = full_metrics(W, quant_int8(W))
            f4 = full_metrics(W, quant_e2m1_block(W, scale_e4m3=True))
            measurements[f"blk.{blk}.{gg}"] = {
                "snr": {"FP16": 150.0, "INT8": i8["snr"], "NVFP4": f4["snr"]},
                "recurrent": rec, "bytes_fp16": W.size * 2,
            }

    # --- Génère le precision_map recommandé ---
    new_map = {}
    table = []
    for name in sorted(measurements, key=lambda x: (int(x.split(".")[1]), x)):
        m = measurements[name]
        prec, reason = recommend(m["snr"]["FP16"], m["snr"]["INT8"], m["snr"]["NVFP4"], m["recurrent"], name)
        new_map[name] = {
            "precision": prec,
            "snr_fp16": m["snr"]["FP16"], "snr_int8": m["snr"]["INT8"], "snr_fp4": m["snr"]["NVFP4"],
            "recurrent": m["recurrent"],
        }
        table.append((name, m["snr"]["INT8"], m["snr"]["NVFP4"], prec, m["recurrent"], reason))

    # --- Compare à l'existant ---
    existing = {}
    pmpath = os.path.join(HERE, "precision_map.json")
    if os.path.exists(pmpath):
        with open(pmpath, encoding="utf-8") as fh:
            existing = json.load(fh)

    print("=" * 100)
    print("  D2 PRECISION OPTIMIZER — recommandation auto (weight noise + amplification)")
    print("=" * 100)
    print(f"  {'Layer':<26} {'INT8':>7} {'NVFP4':>7} {'Rec?':>5}  {'Préconisé':<14} raison")
    print("-" * 100)
    for name, i8, f4, prec, rec, reason in table:
        print(f"  {name:<26} {i8:>6.1f} {f4:>6.1f} {'oui' if rec else 'non':>5}  {prec:<14} {reason}")
    print("=" * 100)

    # Stats de concordance
    match = tot = 0
    for name, m in new_map.items():
        if name in existing:
            tot += 1
            if existing[name].get("precision") == m["precision"]:
                match += 1
    print(f"\n  Concordance avec precision_map.json existant : {match}/{tot} "
          f"({100*match/max(tot,1):.0f} %) sur les couches mesurées")

    # Écrit le nouveau plan
    out = {}
    for name, m in new_map.items():
        out[name] = {"precision": m["precision"],
                     "alpha": existing.get(name, {}).get("alpha", 1.0),
                     "density": existing.get(name, {}).get("density", 0.0),
                     "snr_int8": m["snr_int8"], "snr_fp4": m["snr_fp4"]}
    with open(os.path.join(HERE, "precision_map_auto.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n[+] Plan généré : precision_map_auto.json ({len(out)} couches)")
    print("=" * 100)


if __name__ == "__main__":
    main()
