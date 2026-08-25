#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 AUTONOMOUS PROFILER
=======================
Profileur multi-niveaux autonome pour llama.cpp / GGUF.

Pour chaque layer du modèle, il mesure/estime :
  Layer -> operateurs -> kernels -> VRAM/cache -> GPU compute -> CPU/RAM
        -> precision -> memoire -> tokens/s -> VERDICT (bottleneck + recommandation)

Backends détectés automatiquement (dans l'ordre de priorité) :
  1. nvidia-smi        : télémétrie GPU réelle (util, VRAM, power, clocks, temp) -> TOUJOURS dispo
  2. llama-bench.exe   : tokens/s réels pp/tg -> dispo si le runtime CUDA (cudart/cublas) est présent
  3. Nsight (nsys/ncu) : profiling kernel profond -> dispo si installé

Sortie : rapport texte + JSON (`d2_autonomous_report.json`).

Exemples :
  python d2_autonomous_profiler.py
  python d2_autonomous_profiler.py --model models/model.gguf --gpu gtx1080
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. PROFILS MATERIELS (roofline) — valeurs estimées, à ajuster si besoin
# ---------------------------------------------------------------------------
GPU_PROFILES = {
    "rtx5070": {
        "label": "NVIDIA GeForce RTX 5070 Laptop GPU",
        "arch": "Blackwell", "sm": 120, "vram_gb": 8.0,
        "mem_bw_gbs": 448.0,          # GDDR7 128-bit @28 Gbps = 448 Go/s théorique (memset mesuré ~306 Go/s)
        "fp16_tflops": 60.0,          # estimation dense
        "fp32_tflops": 30.0,
        "int8_tops": 120.0,
        "tensor_cores": True, "fp8_hw": True, "nvfp4_hw": True,
    },
    "gtx1080": {
        "label": "NVIDIA GeForce GTX 1080",
        "arch": "Pascal", "sm": 61, "vram_gb": 8.0,
        "mem_bw_gbs": 320.0,          # GDDR5X 256-bit
        "fp16_tflops": 0.14,          # taux 1/64, pas de Tensor Core
        "fp32_tflops": 8.87,
        "int8_tops": 0.0,
        "tensor_cores": False, "fp8_hw": False, "nvfp4_hw": False,
    },
}

# Octets par élément selon la précision (stockage)
PRECISION_BYTES = {
    "FP32": 4.0, "FP16": 2.0, "BF16": 2.0, "FP8": 1.0,
    "INT8": 1.0, "Q8_0": 1.0, "Q6_K": 0.86, "Q5_K": 0.72,
    "Q4_K": 0.56, "INT4": 0.5, "NVFP4": 0.5, "Q4_0": 0.5,
}

# Mapping precision_map.json -> octets/élément (estimation stockage)
PREC_TO_BYTES = {
    "NVFP4_SAFE": 0.5, "INT8_SAFE": 1.0, "FP16_REQUIRED": 2.0,
}


# ---------------------------------------------------------------------------
# 2. HELPERS SUBPROCESS
# ---------------------------------------------------------------------------
def run(cmd, timeout=30):
    """Exécute une commande, renvoie (rc, stdout)."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return -1, str(e)


def which(prog):
    return shutil.which(prog)


# ---------------------------------------------------------------------------
# 3. DETECTION GPU (nvidia-smi)
# ---------------------------------------------------------------------------
def detect_gpu():
    rc, out = run('nvidia-smi --query-gpu=name,memory.total,driver_version '
                  '--format=csv,noheader,nounits', timeout=15)
    if rc != 0 or not out:
        return None, {"error": "nvidia-smi indisponible ou aucun GPU NVIDIA détecté."}

    # nvidia-smi renvoie une ligne par GPU : on prend la première (sinon
    # "too many values to unpack" quand plusieurs GPU sont présents).
    first_line = out.splitlines()[0]
    name, vram_mib, driver = [x.strip() for x in first_line.split(",")]
    vram_gb = round(int(vram_mib) / 1024.0, 2)

    # Mappe le nom -> profil matériel
    key = None
    nl = name.lower()
    if "5070" in nl or "blackwell" in nl:
        key = "rtx5070"
    elif "1080" in nl or "pascal" in nl:
        key = "gtx1080"

    info = {"name": name, "vram_gb": vram_gb, "driver": driver, "profile_key": key}
    if key and key in GPU_PROFILES:
        info.update(GPU_PROFILES[key])
    else:
        # profil générique inconnu
        info.update({
            "arch": "unknown", "sm": None, "mem_bw_gbs": 400.0,
            "fp16_tflops": 30.0, "fp32_tflops": 15.0, "int8_tops": 60.0,
            "tensor_cores": True, "fp8_hw": True, "nvfp4_hw": True,
        })
    return key, info


def sample_telemetry(n=4, interval=0.4):
    """Échantillonne nvidia-smi n fois ; renvoie listes de dicts."""
    q = ("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,"
         "memory.total,power.draw,clocks.sm,clocks.mem,temperature.gpu "
         "--format=csv,noheader,nounits")
    samples = []
    for _ in range(n):
        rc, out = run(q, timeout=15)
        if rc == 0 and out:
            parts = [x.strip() for x in out.split(",")]
            if len(parts) >= 8:
                samples.append({
                    "gpu_util_pct": _f(parts[0]), "mem_util_pct": _f(parts[1]),
                    "vram_used_mib": _f(parts[2]), "vram_total_mib": _f(parts[3]),
                    "power_w": _f(parts[4]), "sm_clock_mhz": _f(parts[5]),
                    "mem_clock_mhz": _f(parts[6]), "temp_c": _f(parts[7]),
                })
        time.sleep(interval)
    return samples


def _f(s):
    try:
        return float(s)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. BACKENDS DISPONIBLES
# ---------------------------------------------------------------------------
def detect_backends():
    be = {}
    be["llama_bench"] = which("llama-bench.exe") or (os.path.join(HERE, "llama-bench.exe")
                                                     if os.path.exists(os.path.join(HERE, "llama-bench.exe")) else None)
    be["llama_server"] = which("llama-server.exe") or (os.path.join(HERE, "llama-server.exe")
                                                       if os.path.exists(os.path.join(HERE, "llama-server.exe")) else None)

    # Runtime CUDA (cudart/cublas) requis par ggml-cuda.dll
    be["cuda_runtime"] = _find_cuda_runtime()

    be["nsys"] = which("nsys")
    be["ncu"] = which("ncu") or which("nv-nsight-cu-cli")
    return be


def _find_cuda_runtime():
    need = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
    found, missing = [], []
    dirs = [HERE, os.environ.get("SystemRoot", r"C:\Windows") + r"\System32"]
    for dll in need:
        hit = None
        for d in dirs:
            if os.path.exists(os.path.join(d, dll)):
                hit = os.path.join(d, dll)
                break
        if hit:
            found.append(hit)
        else:
            missing.append(dll)
    return {"found": found, "missing": missing}


# ---------------------------------------------------------------------------
# 5. LECTURE DU MODELE (GGUF metadata + precision_map.json)
# ---------------------------------------------------------------------------
def read_model_meta(model_path):
    meta = {"path": model_path, "exists": os.path.exists(model_path),
            "blocks": 32, "hidden": 4096, "ffn": 14336, "heads": 32}
    if not meta["exists"]:
        meta["note"] = "modèle absent — dimensions par défaut (Qwen3.5-9B)"
        return meta
    try:
        from gguf import GGUFReader
        r = GGUFReader(model_path)

        def _field_scalar(v):
            """Extrait la valeur scalaire d'un champ GGUF (ReaderField = tuple)."""
            # API récente : le dernier part est la valeur (memmap 0-D ou 1 élément)
            try:
                if hasattr(v, "parts") and v.parts:
                    p = v.parts[-1]
                    a = p[0] if hasattr(p, "__len__") and len(p) == 1 else p
                    return int(a)
            except Exception:
                pass
            # API ancienne
            try:
                if hasattr(v, "value"):
                    return int(v.value)
            except Exception:
                pass
            return None

        f = {k: _field_scalar(v) for k, v in r.fields.items()}

        def pick(*keys, default):
            for k in keys:
                if k in f and f[k] is not None:
                    return f[k]
            return default

        meta["blocks"] = pick("qwen35.block_count", "qwen3.block_count", "block_count", default=meta["blocks"])
        meta["hidden"] = pick("qwen35.embedding_length", "qwen3.embedding_length", "embedding_length", default=meta["hidden"])
        meta["ffn"] = pick("qwen35.feed_forward_length", "qwen3.feed_forward_length", "feed_forward_length", default=meta["ffn"])
        meta["heads"] = pick("qwen35.attention.head_count", "qwen3.attention.head_count", "head_count", default=meta["heads"])
        meta["tensor_count"] = len(getattr(r, "tensors", []) or [])
        meta["arch"] = str(f.get("general.architecture", "?"))
        # --- Dimensions RÉELLES des tenseurs (vrais poids) ---
        # weights_mb / n_elems du rapport doivent refléter les tenseurs réels du GGUF,
        # pas une géométrie approximée (voir load_real_shapes).
        try:
            real = load_real_shapes(model_path)
            if real:
                if "blk.0.ffn_down.weight" in real:
                    meta["ffn"] = real["blk.0.ffn_down.weight"][1][0]
                if "token_embd.weight" in real:
                    meta["vocab"] = real["token_embd.weight"][1][1]
                meta["real_shapes"] = len(real)
        except Exception:
            pass
    except Exception as e:
        meta["note"] = f"lecture GGUF impossible ({e}) — dimensions par défaut"
    return meta


def load_real_shapes(model_path):
    """Shapes RÉELS de tous les tenseurs GGUF : {nom: (n_elems, shape)}.

    Sert à remplacer la géométrie approximée (layer_geometrie) par les vrais
    nombres d'éléments des poids — c'est la donnée matérielle réelle.
    """
    real = {}
    try:
        from gguf import GGUFReader
        r = GGUFReader(model_path)
        for t in r.tensors:
            ne = 1
            for d in t.shape:
                ne *= int(d)
            real[t.name] = (ne, tuple(int(x) for x in t.shape))
    except Exception:
        pass
    return real


def read_precision_map(path):
    """Charge precision_map.json. Renvoie (layers, source)."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            layers = []
            for name, v in data.items():
                if not isinstance(v, dict):
                    continue
                layers.append({
                    "name": name,
                    "precision": v.get("precision", "NVFP4_SAFE"),
                    "alpha": float(v.get("alpha", 1.0)),
                    "density": float(v.get("density", 0.0)),
                })
            return layers, path
        except Exception as e:
            return [], f"erreur lecture {path}: {e}"
    return [], f"precision_map.json absent ({path})"


# ---------------------------------------------------------------------------
# 6. ROOFLINE / CLASSIFICATION DU BOTTLENECK
# ---------------------------------------------------------------------------
def layer_geometrie(name, hidden, ffn):
    """Retourne (M, N, K) approximé pour le GEMM dominant du layer."""
    n = name.lower()
    if "ssm_conv1d" in n or "ssm_alpha" in n or "ssm_beta" in n or "ssm_out" in n:
        # SSM : petit noyau convolutif, proche de l'activation
        return (1, hidden, hidden // 4)
    if "ffn_up" in n or "ffn_gate" in n:
        return (1, ffn, hidden)
    if "ffn_down" in n:
        return (1, hidden, ffn)
    if "attn_qkv" in n:
        return (1, hidden * 3, hidden)
    if "attn_q" in n or "attn_k" in n or "attn_v" in n or "attn_gate" in n:
        return (1, hidden, hidden)
    if "attn_output" in n:
        return (1, hidden, hidden)
    if "token_embd" in n or "output" in n:
        return (1, hidden, hidden)
    return (1, hidden, hidden)


def classify_layer(layer, gpu, hidden, ffn, real_elems=None):
    """Roofline par layer -> bottleneck + recommandation de précision.

    real_elems : nombre d'éléments du VRAI tenseur GGUF (si dispo).
    S'il est fourni, weights_mb/n_elems reflètent les vrais poids ;
    sinon on retombe sur la géométrie approximée (fallback).
    """
    m, n, k = layer_geometrie(layer["name"], hidden, ffn)
    bpe = PREC_TO_BYTES.get(layer["precision"], 0.56)

    if real_elems:
        n_elems = real_elems
        bytes_w = n_elems * bpe            # octets de poids relus (décode 1 token)
        flops = 2.0 * n_elems              # 2*N*K par token (M=1)
    else:
        n_elems = n * k
        bytes_w = n_elems * bpe
        flops = 2.0 * m * n * k            # 2*M*N*K
    ai = flops / max(bytes_w, 1e-9)  # intensité arithmétique (FLOP/octet)

    ridge = (gpu.get("fp16_tflops", 30.0) * 1e12) / (gpu.get("mem_bw_gbs", 400.0) * 1e9)
    bound = "COMPUTE" if ai > ridge else "MEMORY"

    # Recommandation (politique spectrale + roofline)
    name = layer["name"].lower()
    dens = layer["density"]
    alpha = layer["alpha"]

    if "ssm_conv1d" in name:
        if dens >= 0.31:
            rec = "FP16"
        elif dens >= 0.24:
            rec = "INT8"
        else:
            rec = "FP16" if bound == "COMPUTE" else ("NVFP4" if gpu.get("nvfp4_hw") else "Q4_K")
    elif alpha >= 1.4 and ("ssm_alpha" in name or "ssm_beta" in name):
        rec = "INT8"
    else:
        if bound == "COMPUTE":
            rec = "FP16" if gpu.get("fp8_hw") is False else "BF16"
        else:
            rec = "NVFP4" if gpu.get("nvfp4_hw") else ("INT8" if gpu.get("int8_tops", 0) > 0 else "Q4_K")

    # Bottleneck secondaire
    secondary = "SSM outlier density" if (dens >= 0.24 and "ssm" in name) else \
                ("kernel occupancy (Tensor)" if bound == "COMPUTE" else "L2/DRAM")

    return {
        "name": layer["name"],
        "precision": layer["precision"],
        "alpha": round(alpha, 3),
        "density": round(dens, 3),
        "weights_mb": round(bytes_w / 1e6, 2),
        "ai_flop_byte": round(ai, 1),
        "bound": bound,
        "secondary": secondary,
        "recommendation": rec,
        "n_elems": n_elems,
        "shape_source": "gguf" if real_elems else "approx",
    }


# ---------------------------------------------------------------------------
# 6bis. NOISE MODULE (weight noise + propagation) + SCORE UTILITY
# ---------------------------------------------------------------------------
RHO = 0.99
AMP = 1.0 / (1.0 - RHO)               # amplification récurrente (pire cas)
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# SNR par défaut (mesurés sur Qwen3.5-9B, vrais poids)
DEFAULT_SNR = {"FP16": 90.0, "INT8": 40.0, "NVFP4": 11.0}
BYTES_PER = {"FP16": 2.0, "INT8": 1.0, "NVFP4": 0.5}
RECURRENT_MARKER = ("ssm_conv1d",)
TOL_RECURRENT = 0.01   # erreur d'état SSM tolérée après amplification (1 %)
TOL_LOCAL = 0.5        # erreur locale tolérée (couches memory-bound non récurrentes)

# Seuils de la politique spectrale (precision_map.json)
DENSITY_FP16_THRESH = 0.31   # densité d'outliers -> verrouille haute précision
ALPHA_INT8_THRESH = 1.4       # santé spectrale alpha_w -> évite le 4 bits


def spectral_factor(alpha=1.0, density=0.0, name=""):
    """Sensibilité spectrale (alpha/density) -> facteur multiplicatif de la pénalité.

    Encode la politique spectrale documentée (precision_map.json) :
      - ssm_conv1d   : densité d'outliers (blocs 32, warps CUDA).
            density >= 0.31  -> x100  (outliers massifs -> verrouille haute précision)
            sinon            -> x1    (la récurrence AMP gère déjà le reste)
      - ssm_alpha/beta : santé spectrale alpha_w.
            alpha >= 1.4     -> x4    (couche sensible -> évite le NVFP4)
            sinon            -> x1
      - sinon -> x1

    `factor > 1` augmente la pénalité qualité -> pousse vers une précision PLUS HAUTE.
    `factor < 1` relâche la pénalité -> autorise une précision plus basse.
    """
    ln = (name or "").lower()
    if "ssm_conv1d" in ln:
        return 100.0 if density >= DENSITY_FP16_THRESH else 1.0
    if "ssm_alpha" in ln or "ssm_beta" in ln:
        return 4.0 if alpha >= ALPHA_INT8_THRESH else 1.0
    return 1.0


def _quant_int8(W):
    m = np.abs(W).max(axis=-1, keepdims=True)
    m[m == 0] = 1.0
    return np.clip(np.round(W / m * 127.0), -127, 127) / 127.0 * m


def _quant_e2m1(W):
    out = np.empty_like(W, dtype=np.float32)
    for i in range(W.shape[0]):
        row = W[i]
        n = row.size
        pad = (16 - n % 16) % 16
        wr = np.pad(row, (0, pad)).reshape(-1, 16)
        m = np.abs(wr).max(axis=1, keepdims=True)
        m[m == 0] = 1.0
        norm = wr / m
        sign = np.sign(norm)
        dist = np.abs(np.abs(norm)[:, :, None] - E2M1[None, None, :])
        out[i] = (sign * E2M1[dist.argmin(-1)] * m).reshape(-1)[:n]
    return out


def _snr(W, Wq):
    err = (W - Wq).astype(np.float64)
    num = float(np.sum(W.astype(np.float64) ** 2))
    den = float(np.sum(err ** 2))
    return 10.0 * np.log10(num / max(den, 1e-30))


def measure_snr(W):
    """SNR FP16 / INT8 / NVFP4 d'un tenseur réel (poids)."""
    W = W.astype(np.float32)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    Wt = W.T if W.shape[0] < W.shape[1] else W
    out = {
        "FP16": _snr(Wt, Wt.astype(np.float16).astype(np.float32)),
        "INT8": _snr(Wt, _quant_int8(Wt)),
        "NVFP4": _snr(Wt, _quant_e2m1(Wt)),
    }
    for k in out:
        if not np.isfinite(out[k]):
            out[k] = DEFAULT_SNR[k]
    return out


def load_measurement_weights(model_path):
    """Tenseurs mesurables : conv1d (GGUF F32) + FFN (HF shard 2 si présent)."""
    meas = {}
    try:
        from gguf import GGUFReader
        r = GGUFReader(model_path)
        for t in r.tensors:
            if "ssm_conv1d" in t.name:
                meas[t.name] = np.asarray(t.data, dtype=np.float32)
    except Exception:
        pass
    shard2 = os.path.join(HERE, "hf_weights", "model.safetensors-00002-of-00004.safetensors")
    shard4 = os.path.join(HERE, "hf_weights", "model.safetensors-00004-of-00004.safetensors")
    import struct as _st

    def _read_tensor(fh, hlen, info):
        """Lit un tenseur safetensors (BF16 ou F32) -> float32."""
        s, e = info["data_offsets"]
        fh.seek(8 + hlen + s)   # data_offsets relatifs à la section data
        raw = fh.read(e - s)
        dt = info.get("dtype", "BF16")
        if dt == "F32":
            W = np.frombuffer(raw, dtype="<f4")
        else:  # BF16 (et F16 traité au mieux)
            u = np.frombuffer(raw, dtype="<u2")
            W = (u.astype(np.uint32) << 16).view(np.float32)
        return W.reshape(info["shape"]).astype(np.float32)

    def _add_meas(name, W):
        """Ajoute si le tenseur est raisonnable (>512 éléments) et fini."""
        if W.size <= 512:
            return
        W = W.astype(np.float32)
        if W.size > 8_000_000:
            W = W[:8_000_000 // W.shape[-1]]
        if np.isfinite(W).all() and float(np.abs(W).max()) < 1e6:
            meas[name] = W

    if os.path.exists(shard2):
        try:
            with open(shard2, "rb") as fh:
                hlen = _st.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(hlen))
            for name, info in hdr.items():
                for k, gg in (("mlp.up_proj", "ffn_up"), ("mlp.gate_proj", "ffn_gate"),
                              ("mlp.down_proj", "ffn_down")):
                    if k in name:
                        blk = name.split(".layers.")[1].split(".")[0]
                        with open(shard2, "rb") as fh:
                            _add_meas(f"blk.{blk}.{gg}.weight", _read_tensor(fh, hlen, info))
                        break
        except Exception:
            pass

    # Shard 4 : attention classique (8 blocs) + SSM (24 blocs) — vrais poids BF16
    HF_TO_GGUF = [
        ("self_attn.q_proj", "attn_q.weight"),
        ("self_attn.k_proj", "attn_k.weight"),
        ("self_attn.v_proj", "attn_v.weight"),
        ("self_attn.o_proj", "attn_output.weight"),
        ("linear_attn.out_proj", "ssm_out.weight"),
        ("linear_attn.in_proj_a", "ssm_alpha.weight"),
        ("linear_attn.in_proj_b", "ssm_beta.weight"),
        ("linear_attn.A_log", "ssm_a"),
    ]
    if os.path.exists(shard4):
        try:
            with open(shard4, "rb") as fh:
                hlen = _st.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(hlen))
            for name, info in hdr.items():
                for hf_key, gg_suffix in HF_TO_GGUF:
                    if hf_key in name:
                        blk = name.split(".layers.")[1].split(".")[0]
                        with open(shard4, "rb") as fh:
                            W = _read_tensor(fh, hlen, info)
                        # transposition pour coller à l'orientation GGUF (K,N)->(N,K)
                        if hf_key == "linear_attn.in_proj_a" or hf_key == "linear_attn.in_proj_b":
                            W = W.T
                        _add_meas(f"blk.{blk}.{gg_suffix}", W)
                        break
        except Exception:
            pass
    return meas


def compute_utility(name, n_elems, snr_map, spectral_factor=1.0, tol=None):
    """UTILITY(p) = VRAM_gain_normalise - quality_penalty(p).

    quality_penalty(p) = erreur_effective(p) / tolerance
      erreur_effective = rel_err(p) * amplification * spectral_factor
    spectral_factor > 1 pousse vers une précision plus haute (sensibilité spectrale).
    Penalty >= 1 signifie 'erreur au-dela de la tolerance' (rejet).
    """
    recurrent = any(m in name.lower() for m in RECURRENT_MARKER)
    amp = AMP if recurrent else 1.0
    if tol is None:
        tol = TOL_RECURRENT if recurrent else TOL_LOCAL
    res = {}
    best = (None, -1e18)
    for p in ("FP16", "INT8", "NVFP4"):
        snr = float(snr_map.get(p, DEFAULT_SNR[p]))
        if not np.isfinite(snr):
            snr = DEFAULT_SNR[p]
        rel_err = 10.0 ** (-snr / 20.0)
        effective_err = rel_err * amp * spectral_factor
        quality_penalty = effective_err / tol
        vram_gain_mb = (BYTES_PER["FP16"] - BYTES_PER[p]) * n_elems / 1e6
        vram_gain_norm = (BYTES_PER["FP16"] - BYTES_PER[p]) / BYTES_PER["FP16"]
        utility = vram_gain_norm - quality_penalty
        res[p] = {"snr": round(snr, 1), "rel_err": round(rel_err, 5),
                  "effective_err": round(effective_err, 5),
                  "quality_penalty": round(quality_penalty, 4),
                  "vram_gain_mb": round(vram_gain_mb, 3),
                  "utility": round(utility, 4)}
        if utility > best[1]:
            best = (p, utility)
    res["best"] = best[0]
    res["recurrent"] = recurrent
    res["amplification"] = amp
    res["tolerance"] = tol
    res["spectral_factor"] = spectral_factor
    return res


# ---------------------------------------------------------------------------
# 7. BENCHMARK llama-bench (si runtime CUDA présent)
# ---------------------------------------------------------------------------
def run_llama_bench(backends, model_path, gpu_key):
    if not backends.get("llama_bench"):
        return {"status": "skip", "reason": "llama-bench.exe introuvable"}
    if backends.get("cuda_runtime", {}).get("missing"):
        return {"status": "skip",
                "reason": "runtime CUDA manquant: " + ", ".join(backends["cuda_runtime"]["missing"])}

    cache = {"rtx5070": "turbo4", "gtx1080": "turbo3"}.get(gpu_key, "f16")
    # n_gen=512 : assez long pour que le GPU monte en charge (tg128 est trop court,
    # le GPU reste à ~10 W et la BW mesurée est sous-estimée).
    cmd = (f'"{backends["llama_bench"]}" -m "{model_path}" -ngl 99 '
           f'--cache-type-k {cache} --cache-type-v {cache} -p 128 -n 512')
    rc, out = run(cmd, timeout=300)
    res = {"status": "ok" if rc == 0 else f"error({rc})", "cmd": cmd,
           "output": out[-4000:] if out else ""}
    # Bande passante RÉELLEMENT ATTEINTE : modèle (GiB) × tokens/s du décode.
    # En tg, chaque token relit tous les poids : BW = taille_modèle × tps.
    if res["status"] == "ok" and out:
        try:
            import re as _re
            for line in out.splitlines():
                # ligne de résultat de décode : ... | tg512 | 49.04 ± 5.25 |
                if _re.search(r"\|\s+tg\d+\s*\|", line):
                    cells = [c.strip() for c in line.split("|")]
                    size_gib = float(cells[2].split()[0])
                    tps = float(cells[-2].split()[0])
                    res["bw_achieved_gbs"] = round(size_gib * 1024**3 * tps / 1e9, 1)
                    res["model_gib"] = size_gib
                    res["tg_tps"] = tps
                    res["tg_n"] = cells[-3]
                    break
        except Exception:
            pass
    return res


# ---------------------------------------------------------------------------
# 8. RAPPORT
# ---------------------------------------------------------------------------
def render_report(gpu_key, gpu, model, layers_rows, telemetry, backends, bench, noise_rows=None):
    L = []
    L.append("=" * 92)
    L.append("  D2 AUTONOMOUS PROFILER — rapport par GPU + par layer")
    L.append("=" * 92)

    L.append("\n[GPU]")
    L.append(f"  Cible            : {gpu.get('label', gpu.get('name','?'))}")
    L.append(f"  Architecture     : {gpu.get('arch','?')} (sm_{gpu.get('sm','?')})")
    L.append(f"  VRAM             : {gpu.get('vram_gb','?')} GB")
    L.append(f"  Mem BW (est.)    : {gpu.get('mem_bw_gbs','?')} GB/s")
    L.append(f"  FP16 (est.)      : {gpu.get('fp16_tflops','?')} TFLOPS")
    L.append(f"  Tensor/FP8/NVFP4 : {gpu.get('tensor_cores')}/{gpu.get('fp8_hw')}/{gpu.get('nvfp4_hw')}")

    L.append("\n[MODÈLE]")
    L.append(f"  Fichier   : {model['path']}")
    L.append(f"  Présent   : {model['exists']}")
    L.append(f"  Blocs/Hid/FFN/Têtes : {model['blocks']}/{model['hidden']}/{model['ffn']}/{model['heads']}")
    if model.get("note"):
        L.append(f"  Note      : {model['note']}")

    L.append("\n[BACKENDS]")
    L.append(f"  nvidia-smi      : OK")
    L.append(f"  llama-bench.exe : {backends.get('llama_bench') or 'ABSENT'}")
    L.append(f"  runtime CUDA    : manquant={', '.join(backends.get('cuda_runtime',{}).get('missing',[])) or 'aucun'}")
    L.append(f"  nsys            : {backends.get('nsys') or 'ABSENT'}")
    L.append(f"  ncu             : {backends.get('ncu') or 'ABSENT'}")

    if telemetry:
        t = telemetry[-1]
        # pics sur toute la fenêtre (utile si échantillonnage pendant le bench)
        def _mx(k):
            vals = [x[k] for x in telemetry if x.get(k) is not None]
            return max(vals) if vals else None
        label = "PENDANT le bench (pics)" if len(telemetry) > 1 else "état courant"
        L.append(f"\n[TÉLÉMÉTRIE {label}]")
        L.append(f"  GPU util  : {t['gpu_util_pct']}% (pic {_mx('gpu_util_pct')}%)   Mem util: {t['mem_util_pct']}% (pic {_mx('mem_util_pct')}%)")
        L.append(f"  VRAM      : {t['vram_used_mib']} / {t['vram_total_mib']} MiB")
        L.append(f"  Power     : {t['power_w']} W (pic {_mx('power_w')} W)   SM: {t['sm_clock_mhz']} MHz (pic {_mx('sm_clock_mhz')})   Mem: {t['mem_clock_mhz']} MHz (pic {_mx('mem_clock_mhz')})   T: {t['temp_c']}°C")

    L.append("\n[BENCHMARK llama-bench]")
    if bench.get("status") == "ok":
        L.append(bench["output"])
        if bench.get("bw_achieved_gbs"):
            est = gpu.get("mem_bw_gbs", 0)
            eff = bench["bw_achieved_gbs"] / est * 100 if est else 0
            L.append(f"  -> Bande passante RÉELLE atteinte (tg) : {bench['bw_achieved_gbs']} GB/s "
                     f"({eff:.0f}% de l'estimation {est} GB/s)")
    else:
        L.append(f"  {bench.get('status')}: {bench.get('reason','')}")

    L.append("\n[TABLE LAYERS]")
    hdr = f"  {'Layer':<26} | {'Prec':<14} | {'alpha':>5} | {'dens':>5} | {'MB':>8} | {'AI':>7} | {'Bound':<8} | {'Roofline':<8} | {'UtilBest':<8}"
    L.append(hdr)
    L.append("  " + "-" * 100)
    for r in layers_rows:
        L.append(f"  {r['name']:<26} | {r['precision']:<14} | {r['alpha']:>5.2f} | "
                 f"{r['density']:>5.2f} | {r['weights_mb']:>8.1f} | {r['ai_flop_byte']:>7.0f} | "
                 f"{r['bound']:<8} | {r['recommendation']:<8} | {r['utility_best']:<8}")

    if noise_rows:
        L.append("")
        L.append("[NOISE + UTILITY (weight noise x amplification)]")
        L.append(f"  Amplification recurrente ssm_conv1d (rho={RHO}) : {AMP:.0f}x")
        L.append("  UTILITY(p) = VRAM_gain_normalise - (rel_err(p)*amp / tolerance)")
        L.append(f"  {'Layer':<26} | {'FP16':>6} | {'INT8':>6} | {'NVFP4':>6} | {'best':<6} | {'UTIL':>7} | {'SF':>5}")
        L.append("  " + "-" * 82)
        for r in noise_rows:
            s = r["snr"]
            u = r["utility"]
            bu = u["best"]
            L.append(f"  {r['name']:<26} | {s['FP16']:>6.1f} | {s['INT8']:>6.1f} | {s['NVFP4']:>6.1f} | "
                     f"{bu:<6} | {u[bu]['utility']:>7.3f} | {r['spectral_factor']:>5.1f}")

    # Bilan global
    nb_mem = sum(1 for r in layers_rows if r["bound"] == "MEMORY")
    nb_comp = sum(1 for r in layers_rows if r["bound"] == "COMPUTE")
    L.append("\n[VERDICT GLOBAL]")
    L.append(f"  Layers memory-bound : {nb_mem}")
    L.append(f"  Layers compute-bound: {nb_comp}")
    L.append(f"  PRIMARY bottleneck   : {'VRAM bandwidth' if nb_mem >= nb_comp else 'GPU compute'}")
    L.append("=" * 92)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------------------------
def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 Autonomous Profiler")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "Qwen3.5-9B-Q4_K_S.gguf"))
    ap.add_argument("--precision-map", default=os.path.join(HERE, "precision_map.json"))
    ap.add_argument("--gpu", default=None, help="force gtx1080 / rtx5070")
    ap.add_argument("--bench", action="store_true", help="lancer llama-bench si possible")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--json", default=os.path.join(HERE, "d2_autonomous_report.json"))
    args = ap.parse_args()

    # Sortie console robuste (Windows cp1252 -> UTF-8)
    # 1. GPU
    gpu_key, gpu = detect_gpu()
    if args.gpu:
        gpu_key = args.gpu
    if gpu_key and gpu_key in GPU_PROFILES:
        gpu = dict(GPU_PROFILES[gpu_key])
    if not gpu or "error" in gpu:
        print(json.dumps(gpu, ensure_ascii=False, indent=2))
        return 1

    # 2. Backends
    backends = detect_backends()

    # 3. Modèle + layers
    model = read_model_meta(args.model)
    layers, src = read_precision_map(args.precision_map)

    # 4. Roofline par layer + module noise (weight + propagation) + score UTILITY
    meas = load_measurement_weights(args.model)
    snr_by_name = {}
    for name, W in meas.items():
        try:
            snr_by_name[name] = measure_snr(W)
        except Exception:
            snr_by_name[name] = dict(DEFAULT_SNR)

    rows = []
    noise_rows = []
    real_shapes = load_real_shapes(args.model)
    if layers:
        for ly in layers:
            real_elems = real_shapes.get(ly["name"], (None,))[0] if ly["name"] in real_shapes else None
            row = classify_layer(ly, gpu, model["hidden"], model["ffn"], real_elems=real_elems)
            name = ly["name"]
            snr_map = snr_by_name.get(name, dict(DEFAULT_SNR))
            n_elems = int(meas[name].size) if name in meas else int(row["n_elems"])
            sf = spectral_factor(ly.get("alpha", 1.0), ly.get("density", 0.0), name)
            u = compute_utility(name, n_elems, snr_map, spectral_factor=sf)
            row["snr"] = {k: round(v, 1) for k, v in snr_map.items()}
            row["utility"] = u
            row["utility_best"] = u["best"]
            row["spectral_factor"] = sf
            rows.append(row)
            if name in meas:
                noise_rows.append({
                    "name": name,
                    "snr": {k: round(v, 1) for k, v in snr_map.items()},
                    "utility": u,
                    "recurrent": u["recurrent"],
                    "amplification": u["amplification"],
                    "spectral_factor": sf,
                })
    else:
        print(f"[!] precision_map non chargé : {src}")

    # 5. Télémétrie : si bench demandé, on échantillonne PENDANT le bench
    #    (état réel sous charge) ; sinon état courant (idle).
    telemetry = []
    bench = {"status": "skip", "reason": "non demandé (--bench)"}
    if args.bench:
        import threading as _th
        stop = {"go": True}
        def _sample_loop():
            while stop["go"]:
                for s in sample_telemetry(n=1, interval=0.0):
                    telemetry.append(s)
                time.sleep(0.4)
        th = _th.Thread(target=_sample_loop, daemon=True)
        th.start()
        bench = run_llama_bench(backends, args.model, gpu_key)
        stop["go"] = False
        th.join(timeout=2)
    else:
        telemetry = sample_telemetry(n=args.samples)
    if not telemetry:
        telemetry = sample_telemetry(n=args.samples)  # repli idle

    # 7. Rapport
    report = render_report(gpu_key, gpu, model, rows, telemetry, backends, bench, noise_rows)
    print(report)

    # 8. Export JSON
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": gpu, "model": model,
        "backends": {"llama_bench": backends.get("llama_bench"),
                     "cuda_runtime_missing": backends.get("cuda_runtime", {}).get("missing", []),
                     "nsys": backends.get("nsys"), "ncu": backends.get("ncu")},
        "telemetry": telemetry,
        "layers": rows,
        "bench": bench,
        "data_sources": {
            "real": ["nvidia-smi (GPU/VRAM/driver/télémétrie)",
                      "llama-bench (tokens/s réels CUDA)",
                      "GGUF metadata + shapes réels des tenseurs",
                      f"poids réels mesurés SNR ({len(snr_by_name)} tenseurs : conv1d GGUF + FFN shard2 + attn/SSM shard4)"],
            "estimated": ["mem_bw_gbs / fp16_tflops (spécifications constructeur, non mesurés)",
                           f"SNR par défaut 90/40/11 pour {sum(1 for l in rows if l.get('snr',{}).get('FP16')==90.0)} layers non mesurés"],
        },
    }
    try:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n[+] JSON exporté : {args.json}")
    except Exception as e:
        print(f"[!] export JSON impossible : {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
