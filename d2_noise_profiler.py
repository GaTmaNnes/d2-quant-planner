#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 NOISE PROFILER — sépare bruit de QUANTIFICATION (poids) et bruit de PROPAGATION (SSM).
=========================================================================================
⚠️⚠️⚠️ BANDEAU OBSOLETE [CORRIGÉ 25/08/2026] ⚠️⚠️⚠️
Ce script mesure sur la référence 9B LEGACY (Qwen3.5-9B-Q4_K_S.gguf +
hf_weights shard 9B). La PRODUCTION est désormais Qwen3.6-35B-A3B-D2-MOE
(17.5 GB, gate_up=IQ4_NL + down=Q3_K). Les chiffres SNR/conv1d ci-dessous ne
s'appliquent PAS au modèle de production 35B (architectures et quantizations
différentes). Script conservé FONCTIONNEL pour historique/comparaison 9B.

Corrections vs A/B précédent :
  - Vrai NVFP4 : E2M1 (data) + E4M3 (block scale, 16 élts) + FP32 (scale tenseur).
  - Comparé à un FP4 "naïf" (scale FP32 par bloc) pour montrer l'écart de recette.
  - Métriques complètes : SNR, MSE, erreur relative, cosinus, max error, NaN/Inf.
  - Amplification SSM : simulation de la récurrence h_t = rho*h_{t-1} + B*conv1d(u_t).
"""

import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Le paquet "gguf" n'est pas forcement installe via pip : se rabattre sur le
# gguf-py livre avec le fork beellama.cpp plutot que planter avec
# ModuleNotFoundError des qu'on lance ce script hors d'un env qui l'a (ex: AMD_Quark).
sys.path.insert(1, os.path.join(HERE, "beellama.cpp", "gguf-py"))
from gguf import GGUFReader

# --- Formats flottants réduits -------------------------------------------------
# E2M1 (data 4 bits) : 1 signe, 2 exposants (bias 1), 1 mantisse
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e4m3_grid():
    vals = [0.0]
    for e in range(1, 16):
        for m in range(8):
            vals.append(2.0 ** (e - 7) * (1.0 + m / 8.0))
    for m in range(1, 8):
        vals.append(2.0 ** -6 * (m / 8.0))
    return np.array(sorted(set(vals)), dtype=np.float32)


E4M3 = e4m3_grid()


def quant_e2m1_block(W, block=16, scale_e4m3=False):
    """FP4 E2M1 par blocs. scale_e4m3=True -> block scale quantifié en E4M3 (NVFP4)."""
    out = np.empty_like(W, dtype=np.float32)
    for i in range(W.shape[0]):
        row = W[i]
        n = row.size
        pad = (block - n % block) % block
        wp = np.pad(row, (0, pad), mode="constant")
        wr = wp.reshape(-1, block)
        m = np.abs(wr).max(axis=1, keepdims=True)
        m[m == 0] = 1.0
        if scale_e4m3:
            sc = (m / E2M1[-1]).reshape(-1)
            idx = np.abs(sc[:, None] - E4M3[None, :]).argmin(axis=1)
            m = (E4M3[idx] * E2M1[-1]).reshape(-1, 1)
            m[m == 0] = 1.0   # évite la division par zéro (scale E4M3 -> 0)
        norm = wr / m
        sign = np.sign(norm)
        mag = np.abs(norm)
        dist = np.abs(mag[:, :, None] - E2M1[None, None, :])
        out[i] = (sign * E2M1[dist.argmin(axis=-1)] * m).reshape(-1)[:n]
    return out


def quant_int8(W):
    m = np.abs(W).max(axis=-1, keepdims=True)
    m[m == 0] = 1.0
    return np.clip(np.round(W / m * 127.0), -127, 127) / 127.0 * m


def full_metrics(W, Wq):
    W = W.astype(np.float64)
    Wq = Wq.astype(np.float64)
    err = W - Wq
    num = float(np.sum(W * W))
    den = float(np.sum(err * err))
    snr = 10.0 * np.log10(num / max(den, 1e-30))
    mse = den / W.size
    rel = float(np.sqrt(max(den, 0.0) / max(num, 1e-30)))
    cos = float(np.sum(W * Wq) / (np.sqrt(num) * np.sqrt(float(np.sum(Wq * Wq))) + 1e-30))
    maxerr = float(np.abs(err).max())
    nan = int(np.isnan(Wq).sum())
    pinf = int(np.isposinf(Wq).sum())
    ninf = int(np.isneginf(Wq).sum())
    return dict(snr=round(snr, 1), mse=round(mse, 8), rel=round(rel, 4),
                cos=round(cos, 6), maxerr=round(maxerr, 4),
                nan=nan, posinf=pinf, neginf=ninf)


# --- Chargement ----------------------------------------------------------------
def load_gguf_conv1d(path):
    r = GGUFReader(path)
    out = {}
    for t in r.tensors:
        if "ssm_conv1d" in t.name:
            out[t.name] = np.asarray(t.data, dtype=np.float32)
    return out


def load_hf_ffn(path):
    """Lecture manuelle des FFN (mlp.*_proj) d'un shard safetensors, sans mmap."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    out = {}
    for name, info in hdr.items():
        if any(s in name for s in ("mlp.up_proj", "mlp.gate_proj", "mlp.down_proj")):
            s, e = info["data_offsets"]
            with open(path, "rb") as f:
                f.seek(8 + hlen + s)   # data_offsets relatifs à la section data (après header)
                raw = f.read(e - s)
            # BF16 -> F32 (toutes les FFN sont BF16)
            u = np.frombuffer(raw, dtype="<u2")
            arr = (u.astype(np.uint32) << 16).view(np.float32)
            arr = arr.reshape(info["shape"]).astype(np.float32)
            # sous-échantillonne pour la mémoire
            if arr.size > 1_000_000:
                arr = arr[:1_000_000 // arr.shape[-1]]
            if np.isfinite(arr).all() and float(np.abs(arr).max()) < 1e6:
                out[name] = arr
    return out


# --- Amplification SSM (simulation) -------------------------------------------
def ssm_amplification(rho, T=300, eps=1e-3):
    """Amplification d'une erreur injectée sur l'entrée x (sortie conv1d) à travers
    la récurrence h_t = rho*h_{t-1} + x_t.  Injection constante eps.
    Retourne ||h_pert - h_ref|| / ||x_pert - x_ref||  (~ 1/(1-rho) en régime établi)."""
    x_ref = np.ones(T)
    x_pert = np.ones(T) * (1.0 + eps)
    h_ref = np.zeros(T)
    h_pert = np.zeros(T)
    for t in range(1, T):
        h_ref[t] = rho * h_ref[t - 1] + x_ref[t]
        h_pert[t] = rho * h_pert[t - 1] + x_pert[t]
    err_x = np.linalg.norm(x_pert - x_ref)
    err_h = np.linalg.norm(h_pert - h_ref)
    return float(err_h / err_x)


# --- Main ----------------------------------------------------------------------
def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    gguf = os.path.join(HERE, "models", "Qwen3.5-9B-Q4_K_S.gguf")
    shard2 = os.path.join(HERE, "hf_weights", "model.safetensors-00002-of-00004.safetensors")

    print("=" * 96)
    # [CORRIGÉ 25/08/2026] warning runtime : référence 9B legacy, prod = 35B
    print("  ⚠️ OBSOLETE : mesures sur la référence 9B LEGACY — la production est")
    print("  D2-MOE 35B. Chiffres non transposables au modèle de production.")
    print("  D2 NOISE PROFILER — vrai NVFP4 (E2M1+E4M3) vs FP4 naïf vs INT8")
    print("=" * 96)

    # 1. conv1d (GGUF F32 = référence fiable)
    conv = load_gguf_conv1d(gguf)
    rows = []
    for name, W in conv.items():
        if W.ndim > 1 and W.shape[0] > 1:
            Wt = W.T if W.shape[0] < W.shape[1] else W  # (canaux, taps)
        else:
            Wt = W.reshape(1, -1)
        fp4n = full_metrics(Wt, quant_e2m1_block(Wt, scale_e4m3=False))
        nvfp4 = full_metrics(Wt, quant_e2m1_block(Wt, scale_e4m3=True))
        i8 = full_metrics(Wt, quant_int8(Wt))
        rows.append((name.split(".")[1], fp4n, nvfp4, i8))

    print("\n[SSM conv1d — SNR (dB) moyen, vraies matrices F32]")
    print(f"  {'':<6} {'FP4 naïf':>10} {'NVFP4(E4M3)':>12} {'INT8':>8}   {'NaN/Inf':>8}")
    for blk, fp4n, nv, i8 in rows[:6]:
        print(f"  blk.{blk:>2} {fp4n['snr']:>10.1f} {nv['snr']:>12.1f} {i8['snr']:>8.1f}   "
              f"{fp4n['nan'] + fp4n['posinf'] + fp4n['neginf']:>8}")
    print("  ...")

    m_fp4 = np.mean([r[1]["snr"] for r in rows])
    m_nv = np.mean([r[2]["snr"] for r in rows])
    m_i8 = np.mean([r[3]["snr"] for r in rows])
    print(f"\n  MOYENNE conv1d : FP4 naïf = {m_fp4:.1f} dB | NVFP4(E4M3) = {m_nv:.1f} dB | INT8 = {m_i8:.1f} dB")

    # 2. FFN (HF BF16)
    if os.path.exists(shard2):
        ffn = load_hf_ffn(shard2)
        cats = {"up": [], "gate": [], "down": []}
        for name, W in ffn.items():
            if not np.isfinite(W).all():
                continue  # skip tenseurs corrompus (ex. blk.0)
            for k in cats:
                if f".{k}_proj" in name:
                    cats[k].append(full_metrics(W, quant_e2m1_block(W, scale_e4m3=True))["snr"])
        print("\n[FFN — SNR NVFP4(E4M3) moyen]")
        for k, v in cats.items():
            print(f"  {k:>6} : {np.mean(v):.1f} dB ({len(v)} couches)")

    # 3. Amplification SSM
    print("\n[AMPLIFICATION SSM — erreur d'état / erreur injectée sur conv1d]")
    print(f"  {'rho (decroissance A)':<22} {'amplification (mesure)':>24}   {'1/(1-rho)':>10}   {'lecture':<22}")
    for rho in [0.5, 0.9, 0.99, 0.999]:
        amp = ssm_amplification(rho)
        theo = 1.0 / (1.0 - rho)
        interp = ("stable" if amp < 3 else "accumule" if amp < 20 else "DERIVE forte (contexte long)")
        print(f"  {rho:<22} {amp:>22.1f}x   {theo:>10.1f}   {interp:<22}")
    print("\n  (erreur constante sur conv1d -> amplifiee ~1/(1-rho) dans l'etat recurrent)")
    print("=" * 96)


if __name__ == "__main__":
    main()
