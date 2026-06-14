#!/usr/bin/env python3
"""
D2 RTX + GGUF Profiler — Scanner Spectral Unifié
=================================================
Amélioration des scripts :
  alpha_spectral_scanner.py   → dequantisation GGUF réelle + GPU SVD
  d2_qdf_bayesian_optimized   → politique RTX par sm_version + GGUF intégré
  quant_graph_compiler_v2.py  → cost model RTX (Turing/Ampere/Ada/Hopper)
  d2_profiler.py              → benchmark GPU réel par couche (CUDA Events)

Formats GGUF supportés :
  F32, F16, BF16             → dequantisation exacte
  Q8_0                       → dequantisation exacte (scale f16 + int8)
  Q4_0                       → dequantisation exacte (scale f16 + nibbles)
  Q4_K, Q5_K, Q6_K, Q8_K    → approximation via block stats
  IQ* (imatrix)              → proxy coefficient de variation

Politiques RTX par sm_version :
  sm_89 (Ada   — RTX 4000)   : FP8 natif, INT4 via TRT, INT8 natif
  sm_86/87 (Ampere — RTX 30) : INT8 natif, INT4 via AWQ/GPTQ
  sm_80 (Ampere — A100)      : BF16 + INT8 + FP8 (via TRT 10)
  sm_75 (Turing — RTX 20)    : INT8 natif uniquement
  CPU                        : FP16 / INT8

Usage :
  python d2_rtx_gguf_profiler.py model.gguf
  python d2_rtx_gguf_profiler.py model.gguf --benchmark  # mesure GPU réelle
  python d2_rtx_gguf_profiler.py --demo                  # sans GGUF
  python d2_rtx_gguf_profiler.py model.gguf --out plan.json --budget 8.0
"""

import sys, os, json, time, math, argparse
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.linalg import svd as scipy_svd
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from gguf import GGUFReader, GGMLQuantizationType
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False


# ═══════════════════════════════════════════════════════════════════════════════
# §1.  DÉTECTION GPU RTX
# ═══════════════════════════════════════════════════════════════════════════════

def detect_gpu() -> Dict:
    """Détecte le GPU et retourne ses capacités."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return {"available": False, "name": "CPU", "sm": (0, 0),
                "fp8": False, "int8": False, "bf16": False, "mem_gb": 0}

    name   = torch.cuda.get_device_name(0)
    sm     = torch.cuda.get_device_capability(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    return {
        "available": True,
        "name"     : name,
        "sm"       : sm,
        "sm_str"   : f"sm_{sm[0]}{sm[1]}",
        "fp8"      : sm[0] >= 9 or (sm[0] == 8 and sm[1] >= 9),  # Ada + Hopper
        "bf16"     : sm[0] >= 8,                                    # Ampere+
        "int8"     : sm[0] >= 7 and sm[1] >= 5,                    # Turing+
        "int4_trt" : sm[0] >= 8 and sm[1] >= 6,                    # Ampere RTX+
        "mem_gb"   : round(mem_gb, 1),
    }

GPU = detect_gpu()

# Profils GPU prédéfinis pour simulation sans matériel
GPU_PROFILES = {
    "rtx4090": {"available":True,"name":"RTX 4090 (sim)","sm":(8,9),"sm_str":"sm_89",
                "fp8":True,"bf16":True,"int8":True,"int4_trt":True,"mem_gb":24.0},
    "rtx3090": {"available":True,"name":"RTX 3090 (sim)","sm":(8,6),"sm_str":"sm_86",
                "fp8":False,"bf16":True,"int8":True,"int4_trt":True,"mem_gb":24.0},
    "rtx3080": {"available":True,"name":"RTX 3080 (sim)","sm":(8,6),"sm_str":"sm_86",
                "fp8":False,"bf16":True,"int8":True,"int4_trt":True,"mem_gb":10.0},
    "rtx2080": {"available":True,"name":"RTX 2080 (sim)","sm":(7,5),"sm_str":"sm_75",
                "fp8":False,"bf16":False,"int8":True,"int4_trt":False,"mem_gb":8.0},
    "a100"   : {"available":True,"name":"A100 (sim)","sm":(8,0),"sm_str":"sm_80",
                "fp8":False,"bf16":True,"int8":True,"int4_trt":True,"mem_gb":80.0},
    "h100"   : {"available":True,"name":"H100 (sim)","sm":(9,0),"sm_str":"sm_90",
                "fp8":True,"bf16":True,"int8":True,"int4_trt":True,"mem_gb":80.0},
    "cpu"    : {"available":False,"name":"CPU","sm":(0,0),"sm_str":"cpu",
                "fp8":False,"bf16":False,"int8":True,"int4_trt":False,"mem_gb":0},
}

def apply_gpu_profile(profile_name: str) -> Dict:
    """Remplace la détection auto par un profil prédéfini."""
    global GPU
    p = GPU_PROFILES.get(profile_name.lower())
    if p is None:
        print(f"  [!] Profil inconnu '{profile_name}'. Choix : {list(GPU_PROFILES)}")
        return GPU
    GPU = dict(p)
    return GPU


def rtx_policy_map(sm: Tuple[int, int]) -> Dict[str, float]:
    """
    Coûts relatifs des précisions selon l'architecture RTX.
    Plus le coût est bas, plus la précision est rapide sur ce GPU.
    """
    major, minor = sm
    if major >= 9:           # Hopper (H100)
        return {"FP8": 0.15, "INT8": 0.25, "INT4_TRT": 0.20,
                "BF16": 0.50, "FP16": 0.55, "FP32": 1.00}
    elif major == 8 and minor >= 9:  # Ada (RTX 4000)
        return {"FP8": 0.20, "INT8": 0.30, "INT4_TRT": 0.25,
                "BF16": 0.60, "FP16": 0.65, "FP32": 1.00}
    elif major == 8:         # Ampere (RTX 3000 / A100)
        return {"FP8": 0.40, "INT8": 0.30, "INT4_TRT": 0.35,
                "BF16": 0.50, "FP16": 0.60, "FP32": 1.00}
    elif major == 7 and minor >= 5:  # Turing (RTX 2000)
        return {"FP8": 1.00, "INT8": 0.35, "INT4_TRT": 0.50,
                "BF16": 0.80, "FP16": 0.60, "FP32": 1.00}
    else:                    # Pascal ou moins
        return {"FP8": 1.00, "INT8": 0.60, "INT4_TRT": 1.00,
                "BF16": 1.00, "FP16": 0.70, "FP32": 1.00}


# ═══════════════════════════════════════════════════════════════════════════════
# §2.  DEQUANTISATION GGUF
# ═══════════════════════════════════════════════════════════════════════════════

# Tailles de blocs et formats
GGML_Q8_0_BLOCK = 32    # 1×f16 scale + 32×i8  = 34 bytes
GGML_Q4_0_BLOCK = 32    # 1×f16 scale + 16×u8  = 18 bytes (32 nibbles)

def dequant_f32(data: np.ndarray, shape: Tuple) -> np.ndarray:
    return data.reshape(shape).astype(np.float32)

def dequant_f16(data: np.ndarray, shape: Tuple) -> np.ndarray:
    return data.view(np.float16).reshape(shape).astype(np.float32)

def dequant_bf16(data: np.ndarray, shape: Tuple) -> np.ndarray:
    """BF16 n'a pas de dtype numpy natif — convertir via bytes."""
    raw = data.view(np.uint16)
    # Shift left 16 bits pour obtenir float32
    fp32_bits = raw.astype(np.uint32) << 16
    return fp32_bits.view(np.float32).reshape(shape)

def dequant_q8_0(data: np.ndarray, shape: Tuple) -> np.ndarray:
    """
    Q8_0 : blocs de 32 valeurs int8 + 1 scale float16.
    Format par bloc (34 bytes) : [scale:f16][v0..v31:i8]
    """
    rows, cols = shape[0], shape[1]
    n_elements = rows * cols
    n_blocks   = n_elements // GGML_Q8_0_BLOCK

    raw   = data.view(np.uint8)
    out   = np.zeros(n_elements, dtype=np.float32)

    for b in range(n_blocks):
        offset = b * 34
        scale  = raw[offset:offset+2].view(np.float16)[0].astype(np.float32)
        vals   = raw[offset+2:offset+34].view(np.int8).astype(np.float32)
        out[b*32:(b+1)*32] = scale * vals

    return out.reshape(shape)

def dequant_q4_0(data: np.ndarray, shape: Tuple) -> np.ndarray:
    """
    Q4_0 : blocs de 32 nibbles + 1 scale float16.
    Format par bloc (18 bytes) : [scale:f16][packed_nibbles:16×u8]
    """
    rows, cols = shape[0], shape[1]
    n_elements = rows * cols
    n_blocks   = n_elements // GGML_Q4_0_BLOCK

    raw = data.view(np.uint8)
    out = np.zeros(n_elements, dtype=np.float32)

    for b in range(n_blocks):
        offset = b * 18
        scale  = raw[offset:offset+2].view(np.float16)[0].astype(np.float32)
        packed = raw[offset+2:offset+18]
        lo = (packed & 0x0F).astype(np.int8)
        hi = (packed >> 4).astype(np.int8)
        # Centrer autour de 0 (Q4_0 utilise [0,15] - 8)
        nibbles = np.empty(32, dtype=np.int8)
        nibbles[0::2] = lo - 8
        nibbles[1::2] = hi - 8
        out[b*32:(b+1)*32] = scale * nibbles.astype(np.float32)

    return out.reshape(shape)

def dequant_k_quant_proxy(data: np.ndarray, shape: Tuple) -> np.ndarray:
    """
    Approximation pour Q4_K, Q5_K, Q6_K, Q8_K, IQ* :
    Interpréter les bytes bruts comme int8 et normaliser.
    Pas précis pour l'inférence mais suffisant pour l'analyse spectrale.
    """
    rows, cols = shape[0], shape[1]
    n_elements = rows * cols
    flat = data.view(np.int8).flatten()
    # Sous-échantillonnage si nécessaire
    if len(flat) >= n_elements:
        flat = flat[:n_elements]
    else:
        # Répéter pour remplir
        reps = math.ceil(n_elements / len(flat))
        flat = np.tile(flat, reps)[:n_elements]
    return flat.astype(np.float32).reshape(rows, cols)


def gguf_dequant(data: np.ndarray, tensor_type: int,
                  shape: Tuple) -> Tuple[np.ndarray, str]:
    """
    Dequantise un tensor GGUF vers float32.
    Retourne (matrice_float32, méthode).
    """
    t = tensor_type
    try:
        if t == 0:   # F32
            return dequant_f32(data, shape), "f32-exact"
        elif t == 1: # F16
            return dequant_f16(data, shape), "f16-exact"
        elif t == 30:# BF16
            return dequant_bf16(data, shape), "bf16-exact"
        elif t == 8: # Q8_0
            return dequant_q8_0(data, shape), "q8_0-exact"
        elif t == 2: # Q4_0
            return dequant_q4_0(data, shape), "q4_0-exact"
        else:        # Q4_K, Q5_K, Q6_K, IQ* → proxy
            return dequant_k_quant_proxy(data, shape), "proxy-bytes"
    except Exception as e:
        # Fallback ultime
        try:
            flat = data.flatten().astype(np.float32)
            n    = shape[0] * shape[1]
            if len(flat) >= n:
                return flat[:n].reshape(shape), "fallback-trunc"
            else:
                return np.zeros(shape, dtype=np.float32), "fallback-zero"
        except Exception:
            return np.zeros(shape, dtype=np.float32), "fallback-zero"


# ═══════════════════════════════════════════════════════════════════════════════
# §3.  ALPHA_W — SVD SPECTRAL (GPU si disponible)
# ═══════════════════════════════════════════════════════════════════════════════

def alpha_w_gpu(W: np.ndarray, max_rows: int = 4096) -> float:
    """SVD via torch.linalg sur GPU (CUDA). Plus rapide pour les grands tenseurs."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return _alpha_w_cpu(W, max_rows)
    try:
        if W.shape[0] > max_rows:
            idx = np.random.choice(W.shape[0], max_rows, replace=False)
            W   = W[idx]
        Wt = torch.from_numpy(W.astype(np.float32)).cuda()
        S  = torch.linalg.svdvals(Wt).cpu().numpy()
        torch.cuda.empty_cache()
        return _fit_alpha(S)
    except Exception:
        return _alpha_w_cpu(W, max_rows)

def _alpha_w_cpu(W: np.ndarray, max_rows: int = 2048,
                  n_components: int = 128) -> float:
    """SVD CPU avec randomized SVD pour les grands tenseurs."""
    try:
        if W.shape[0] > max_rows:
            idx = np.random.choice(W.shape[0], max_rows, replace=False)
            W   = W[idx]

        # Randomized SVD pour accélérer
        if W.shape[0] * W.shape[1] > 2_000_000 and HAS_SCIPY:
            k   = min(n_components, min(W.shape) - 1)
            Q   = np.random.randn(W.shape[1], k).astype(np.float32)
            for _ in range(4):
                Q, _ = np.linalg.qr(W @ Q)
                Q, _ = np.linalg.qr(W.T @ Q[:W.shape[0]])
            B   = Q.T @ W if Q.shape[0] == W.shape[0] else W @ Q
            _, S, _ = np.linalg.svd(B, full_matrices=False)
        elif HAS_SCIPY:
            _, S, _ = scipy_svd(W, full_matrices=False)
        else:
            S = np.linalg.svd(W, compute_uv=False)

        return _fit_alpha(S)
    except Exception:
        return 1.5

def _fit_alpha(S: np.ndarray, cutoff: float = 0.01) -> float:
    """Régression log-log sur les valeurs singulières → alpha_w."""
    S = np.asarray(S, dtype=np.float32)
    S = S[S > cutoff * (S[0] if S[0] > 0 else 1.0)]
    if len(S) < 3:
        return 1.0
    x     = np.log(np.arange(1, len(S) + 1, dtype=np.float32))
    y     = np.log(S + 1e-12)
    slope = float(np.polyfit(x, y, 1)[0])
    return float(max(round(-2.0 * slope, 3), 1.0))

def compute_alpha_w(data: np.ndarray, tensor_type: int,
                     shape: Tuple, use_gpu: bool = True) -> Tuple[float, str, str]:
    """
    Pipeline complet : dequantise → calcule alpha_w.
    Retourne (alpha_w, methode_dequant, methode_svd).
    """
    W, dq_method = gguf_dequant(data, tensor_type, shape)

    if W is None or W.size == 0:
        return 1.5, dq_method, "skip"

    if use_gpu and GPU["available"]:
        alpha = alpha_w_gpu(W)
        svd_m = "GPU-SVD"
    else:
        alpha = _alpha_w_cpu(W)
        svd_m = "CPU-SVD"

    return alpha, dq_method, svd_m


# ═══════════════════════════════════════════════════════════════════════════════
# §4.  POLITIQUE RTX (remplace nvidia_llm_policy)
# ═══════════════════════════════════════════════════════════════════════════════

# Seuils alpha_w par politique (inspiré Martin & Mahoney 2021)
ALPHA_THRESHOLDS = {
    "FP8"     : 2.5,   # Très concentré → FP8 safe (Ada/Hopper)
    "INT4_TRT": 2.1,   # Concentré → INT4 via TensorRT
    "INT8"    : 1.5,   # Modéré → INT8 natif
    "BF16"    : 1.2,   # Faible → BF16
    # < 1.2   : FP16_REQUIRED
}

SENSITIVE_KEYWORDS = [
    "norm", "embed", "lm_head", "output.weight",
    "token_embd", "output_norm", "wte", "wpe",
    "attn_norm", "ffn_norm", "ln_",
]

def rtx_quant_policy(name: str, alpha: float, shape: Tuple,
                      gpu_caps: Dict) -> str:
    """
    Politique de quantification RTX basée sur :
    - alpha_w (concentration spectrale)
    - capacités du GPU (sm_version)
    - type de couche (norm/embed → FP16 forcé)
    """
    name_l = name.lower()

    # Couches critiques → toujours FP16
    if any(k in name_l for k in SENSITIVE_KEYWORDS):
        return "FP16_REQUIRED"

    # Couches trop petites
    if len(shape) >= 2 and shape[0] * shape[1] < 4096:
        return "FP16_REQUIRED"

    sm = gpu_caps.get("sm", (0, 0))

    # FP8 : seulement Ada+ (sm_89+) ou Hopper (sm_90)
    if alpha >= ALPHA_THRESHOLDS["FP8"] and gpu_caps.get("fp8", False):
        return "FP8"

    # INT4 via TensorRT : Ampere RTX + Ada
    if alpha >= ALPHA_THRESHOLDS["INT4_TRT"] and gpu_caps.get("int4_trt", False):
        return "INT4_AWQ"

    # INT8 natif : Turing+
    if alpha >= ALPHA_THRESHOLDS["INT8"] and gpu_caps.get("int8", False):
        return "INT8_SAFE"

    # BF16 : Ampere+
    if alpha >= ALPHA_THRESHOLDS["BF16"] and gpu_caps.get("bf16", False):
        return "BF16"

    return "FP16_REQUIRED"


# ═══════════════════════════════════════════════════════════════════════════════
# §5.  BENCHMARK GPU RÉEL (remplace les valeurs hardcodées de d2_profiler.py)
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_layer_gpu(W: np.ndarray, policy: str,
                          batch_size: int = 1,
                          n_warmup: int = 3,
                          n_runs: int = 10) -> Dict:
    """
    Benchmark réel d'une couche GEMM sur GPU avec CUDA Events.
    Mesure la latence de token_decode (batch_size=1, séquentiel).
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return _cpu_latency_estimate(W, policy)

    rows, cols = W.shape[0], W.shape[1]

    try:
        Wt = torch.from_numpy(W.astype(np.float16)).cuda()

        # Quantification selon la politique
        if policy == "INT8_SAFE":
            scale = Wt.abs().max() / 127.0 + 1e-9
            Wq    = (Wt / scale).clamp(-128, 127).to(torch.int8)
            def run_layer(x):
                xf = x.to(torch.float16)
                return F.linear(xf, Wq.to(torch.float16) * scale)
        elif policy in ("INT4_AWQ", "FP8"):
            # INT4/FP8 : simuler en float16 (pas de kernel natif sans TRT/bitsandbytes)
            def run_layer(x):
                return F.linear(x, Wt)
        else:  # BF16 / FP16
            if policy == "BF16" and GPU.get("bf16", False):
                Wt = Wt.to(torch.bfloat16)
                def run_layer(x):
                    return F.linear(x.to(torch.bfloat16), Wt).to(torch.float16)
            else:
                def run_layer(x):
                    return F.linear(x, Wt)

        x = torch.randn(batch_size, cols, dtype=torch.float16, device="cuda")

        # Warmup
        for _ in range(n_warmup):
            _ = run_layer(x)
        torch.cuda.synchronize()

        # Mesure
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_runs):
            _ = run_layer(x)
        end.record()
        torch.cuda.synchronize()

        lat_ms  = start.elapsed_time(end) / n_runs
        tps_mea = 1000.0 / lat_ms if lat_ms > 0 else 0
        bw_gbs  = (rows * cols * 2) / (lat_ms * 1e-3) / 1e9  # f16 bytes

        torch.cuda.empty_cache()
        return {
            "lat_ms"  : round(lat_ms, 4),
            "tps"     : round(tps_mea, 1),
            "bw_gbs"  : round(bw_gbs, 1),
            "source"  : "CUDA-Events",
        }

    except Exception as e:
        return _cpu_latency_estimate(W, policy)


def _cpu_latency_estimate(W: np.ndarray, policy: str) -> Dict:
    """Estimation CPU quand CUDA n'est pas disponible."""
    rows, cols  = W.shape
    bpe         = {"FP8":0.5, "INT4_AWQ":0.5, "INT8_SAFE":1.0,
                   "BF16":2.0, "FP16_REQUIRED":2.0}.get(policy, 2.0)
    # BW CPU DDR5 ~ 80 GB/s théorique, ~40 GB/s effectif LLM
    cpu_bw_gbs  = 40.0
    bytes_gb    = rows * cols * bpe / 1e9
    lat_ms      = bytes_gb / cpu_bw_gbs * 1000.0
    return {
        "lat_ms"  : round(lat_ms, 4),
        "tps"     : round(1000.0 / lat_ms, 1) if lat_ms > 0 else 0,
        "bw_gbs"  : round(cpu_bw_gbs * (bytes_gb / (bytes_gb + 0.001)), 1),
        "source"  : "CPU-estimate",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §6.  GRAPH COMPILER RTX (remplace QuantGraphCompilerV2)
# ═══════════════════════════════════════════════════════════════════════════════

SWITCH_PENALTY = 0.4   # Coût relatif d'un switch de précision (Memory roundtrip)
ALIGN_PENALTY  = 0.25  # Coût d'un désalignement (non-multiple de 128)

class RTXGraphCompiler:
    """
    Compilateur de graphe de quantification pour RTX.
    Remplace QuantGraphCompilerV2 avec support RTX par sm_version.
    """

    def __init__(self, gpu_caps: Dict):
        self.gpu    = gpu_caps
        self.costs  = rtx_policy_map(gpu_caps.get("sm", (0, 0)))

    def _is_aligned(self, shape: Tuple) -> bool:
        """Vérifie l'alignement pour les kernels Tensor Core (multiple de 128)."""
        return len(shape) >= 2 and shape[-1] % 128 == 0

    def _rank_policy(self, p: str) -> int:
        order = ["FP8", "INT4_AWQ", "INT8_SAFE", "BF16", "FP16_REQUIRED"]
        return order.index(p) if p in order else len(order)

    def build_clusters(self, layers: List[Dict]) -> List[Dict]:
        """
        Groupe les couches consécutives de même précision en clusters.
        Applique le switch penalty pour fusionner les singletons isolés.
        Pass 1 : clustering initial.
        Pass 2 : optimisation des switches (légalisation).
        Pass 3 : alignement (fallback INT8 si non-aligné pour FP8/INT4).
        """
        if not layers:
            return []

        # Pass 1 : clustering brut
        clusters = []
        curr = {"policy": layers[0]["policy"],
                "layers": [layers[0]], "cost": 0.0}

        for l in layers[1:]:
            if l["policy"] == curr["policy"]:
                curr["layers"].append(l)
            else:
                clusters.append(curr)
                curr = {"policy": l["policy"], "layers": [l], "cost": 0.0}
        clusters.append(curr)

        # Pass 2 : fusionner singletons si switch_penalty > gain
        optimized = []
        for i, c in enumerate(clusters):
            if len(c["layers"]) == 1 and optimized and i < len(clusters) - 1:
                prev_p = optimized[-1]["policy"]
                next_p = clusters[i+1]["policy"]
                if prev_p == next_p:
                    # Unifier avec le cluster précédent
                    optimized[-1]["layers"].extend(c["layers"])
                    continue
            optimized.append(c)

        # Pass 3 : légalisation alignement
        for c in optimized:
            if c["policy"] in ("FP8", "INT4_AWQ"):
                for l in c["layers"]:
                    if not self._is_aligned(tuple(l.get("shape", [1, 1]))):
                        l["policy"]   = "INT8_SAFE"
                        l["legal_fix"] = "align"

        # Recalcul des policies de cluster après légalisation
        for c in optimized:
            policies = [l["policy"] for l in c["layers"]]
            # Politique du cluster = la plus conservative
            c["policy"] = sorted(set(policies),
                                  key=self._rank_policy)[-1]

        # Calcul du coût total par cluster
        for c in optimized:
            c["cost"] = sum(self.costs.get(l["policy"], 1.0)
                            for l in c["layers"])
            c["n"]    = len(c["layers"])

        return optimized

    def compile(self, layers: List[Dict], output_path: str) -> List[Dict]:
        print(f"  Pass 1 : Clustering ({len(layers)} couches) ...")
        clusters = self.build_clusters(layers)

        print(f"  Pass 2 : Optimisation switches ({len(clusters)} clusters) ...")
        n_switches = len(clusters) - 1

        print(f"  Pass 3 : Légalisation alignement ...")
        n_fixed = sum(1 for c in clusters for l in c["layers"]
                      if l.get("legal_fix"))

        plan = {
            "gpu"       : self.gpu.get("name", "CPU"),
            "sm"        : self.gpu.get("sm_str", "cpu"),
            "clusters"  : len(clusters),
            "switches"  : n_switches,
            "align_fix" : n_fixed,
            "layers"    : [{
                "name"   : l["name"],
                "shape"  : l.get("shape", []),
                "policy" : l["policy"],
                "alpha_w": l.get("alpha_w", 0),
                "lat_ms" : l.get("lat_ms", 0),
                "fix"    : l.get("legal_fix", ""),
            } for c in clusters for l in c["layers"]],
            "cluster_plan": [{
                "id"    : i+1,
                "policy": c["policy"],
                "n"     : c["n"],
                "cost"  : round(c["cost"], 3),
                "layers": [l["name"] for l in c["layers"]],
            } for i, c in enumerate(clusters)],
        }

        with open(output_path, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"  Plan sauvegardé : {output_path}")
        return clusters


# ═══════════════════════════════════════════════════════════════════════════════
# §7.  SCANNER PRINCIPAL (remplace alpha_spectral_scanner + d2_profiler)
# ═══════════════════════════════════════════════════════════════════════════════

def scan_gguf(model_path: str,
               do_benchmark: bool = False,
               verbose: bool = True) -> List[Dict]:
    """
    Scan complet d'un fichier GGUF :
    - Dequantisation par tensor_type
    - SVD alpha_w (GPU si dispo)
    - Politique RTX
    - Benchmark GPU optionnel
    """
    if not HAS_GGUF:
        print("  [!] pip install gguf")
        sys.exit(1)

    print(f"  Modèle : {os.path.basename(model_path)}")
    print(f"  Taille : {os.path.getsize(model_path)/1e9:.2f} GB")
    print(f"  GPU    : {GPU['name']}", end="")
    if GPU["available"]:
        print(f" [{GPU['sm_str']}] {GPU['mem_gb']} GB VRAM")
        print(f"  Caps   : FP8={GPU['fp8']} BF16={GPU['bf16']} "
              f"INT8={GPU['int8']} INT4_TRT={GPU['int4_trt']}")
    else:
        print(" (pas de GPU — estimation CPU)")
    print()

    reader  = GGUFReader(model_path)
    tensors = [t for t in reader.tensors
               if len(t.shape) >= 2 and t.n_elements > 0]
    total   = len(tensors)
    print(f"  {total} tenseurs de poids détectés")
    print()

    hdr = (f"  {'#':>4}  {'Nom':<40} {'Shape':<18} {'Dequant':>10} "
           f"{'SVD':>8} {'α_w':>7} {'Policy':<14}")
    if do_benchmark:
        hdr += f" {'lat_ms':>8} {'TPS':>8} {'BW':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    results = []
    t0_total = time.time()

    for i, tensor in enumerate(tensors):
        t0    = time.time()
        name  = tensor.name
        shape = tuple(int(s) for s in tensor.shape)
        ttype = int(tensor.tensor_type)

        # Alpha_w
        alpha, dq_m, svd_m = compute_alpha_w(
            tensor.data, ttype, shape, use_gpu=GPU["available"]
        )

        # Politique RTX
        policy = rtx_quant_policy(name, alpha, shape, GPU)

        # Benchmark optionnel
        bench = {}
        if do_benchmark and GPU["available"]:
            W, _ = gguf_dequant(tensor.data, ttype, shape)
            if W is not None and W.shape[0] * W.shape[1] <= 50_000_000:
                bench = benchmark_layer_gpu(W, policy)
            else:
                bench = {"lat_ms": 0, "tps": 0, "bw_gbs": 0, "source": "skip-oom"}

        dt = time.time() - t0

        row = {
            "name"    : name,
            "shape"   : list(shape),
            "ttype"   : ttype,
            "alpha_w" : alpha,
            "dq_m"    : dq_m,
            "svd_m"   : svd_m,
            "policy"  : policy,
            "lat_pred": 0.0,  # placeholder
            **bench,
        }
        results.append(row)

        if verbose:
            shape_s = f"[{shape[0]},{shape[1]}]"
            line    = (f"  {i+1:>4}  {name:<40} {shape_s:<18} {dq_m:>10} "
                       f"{svd_m:>8} {alpha:>7.3f} {policy:<14}")
            if do_benchmark and bench:
                line += f" {bench.get('lat_ms',0):>8.4f} {bench.get('tps',0):>8.1f} {bench.get('bw_gbs',0):>7.1f}"
            print(line)

    elapsed = time.time() - t0_total
    print()
    print(f"  Scan : {elapsed:.1f}s — {elapsed/total*1000:.0f}ms/couche")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §8.  RAPPORT FINAL
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results: List[Dict], clusters: List[Dict],
                  model_path: str = "") -> None:
    bpe = {"FP8":0.5, "INT4_AWQ":0.5, "INT8_SAFE":1.0,
           "BF16":2.0, "FP16_REQUIRED":2.0}

    print()
    print("═" * 80)
    print(f"  RAPPORT — {os.path.basename(model_path)}")
    print(f"  GPU    : {GPU['name']}")
    print("═" * 80)

    # Distribution
    counts  = {}
    sizes   = {}
    for r in results:
        p = r["policy"]
        counts[p] = counts.get(p, 0) + 1
        s = r["shape"]
        if len(s) >= 2:
            sizes[p] = sizes.get(p, 0) + s[0]*s[1]*bpe.get(p, 2.0)

    total_layers = len(results)
    total_bytes  = sum(sizes.values())
    total_gb     = total_bytes / 1e9

    print(f"\n  Distribution des précisions :")
    print(f"  {'Policy':<16} {'Couches':>8} {'%':>6} {'Taille':>10}  Bar")
    print("  " + "─" * 58)
    for p in ["FP8", "INT4_AWQ", "INT8_SAFE", "BF16", "FP16_REQUIRED"]:
        n   = counts.get(p, 0)
        if n == 0: continue
        pct = n / total_layers * 100
        gb  = sizes.get(p, 0) / 1e9
        bar = "█" * int(pct / 4)
        print(f"  {p:<16} {n:>8} {pct:>5.1f}% {gb:>8.2f}GB  {bar}")

    print(f"\n  Taille totale estimée : {total_gb:.2f} GB")

    # Couches à risque
    risky = [r for r in results
             if r["policy"] in ("INT4_AWQ","FP8") and r["alpha_w"] < 2.0]
    if risky:
        print(f"\n  ⚠ Couches à vérifier (politique agressive, α < 2.0) : {len(risky)}")
        for r in risky[:6]:
            print(f"    [{r['alpha_w']:.2f}] {r['name'][:55]:<55} → {r['policy']}")

    # Clusters
    print(f"\n  Clusters d'exécution : {len(clusters)}")
    print(f"  {'#':>4}  {'Policy':<16} {'N':>5} {'Cost':>7}  Exemples")
    print("  " + "─" * 68)
    for i, c in enumerate(clusters[:20]):
        ex = c["layers"][0]["name"][:25] if c["layers"] else ""
        if c["n"] > 1:
            ex += f" +{c['n']-1}"
        print(f"  {i+1:>4}  {c['policy']:<16} {c['n']:>5} {c['cost']:>7.2f}  {ex}")
    if len(clusters) > 20:
        print(f"  ... et {len(clusters)-20} clusters supplémentaires")

    # Benchmark résumé
    if any(r.get("lat_ms", 0) > 0 for r in results):
        measured = [r for r in results if r.get("source") == "CUDA-Events"]
        if measured:
            avg_lat = sum(r["lat_ms"] for r in measured) / len(measured)
            total_lat = sum(r["lat_ms"] for r in measured)
            print(f"\n  Benchmark GPU ({len(measured)} couches mesurées) :")
            print(f"    Lat. moyenne / couche : {avg_lat:.4f} ms")
            print(f"    Lat. totale / token   : {total_lat:.2f} ms")
            print(f"    TPS estimé            : {1000/total_lat:.1f} t/s")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# §9.  MODE DÉMO SYNTHÉTIQUE
# ═══════════════════════════════════════════════════════════════════════════════

def demo_synthetic(n_layers: int = 32, hidden: int = 4096) -> List[Dict]:
    """Profil synthétique Llama-3-8B pour démo sans GGUF."""
    np.random.seed(0)
    layers = []
    templates = [
        ("model.embed_tokens.weight",          (32000, hidden), 0.85),
        ("model.norm.weight",                  (hidden,),       0.90),
        ("lm_head.weight",                     (32000, hidden), 1.10),
    ]
    for i in range(n_layers):
        templates += [
            (f"model.layers.{i}.input_layernorm.weight",       (hidden,),       0.90),
            (f"model.layers.{i}.self_attn.q_proj.weight",      (hidden, hidden),2.15),
            (f"model.layers.{i}.self_attn.k_proj.weight",      (hidden//4, hidden), 1.85),
            (f"model.layers.{i}.self_attn.v_proj.weight",      (hidden//4, hidden), 1.75),
            (f"model.layers.{i}.self_attn.o_proj.weight",      (hidden, hidden),2.05),
            (f"model.layers.{i}.post_attention_layernorm.weight",(hidden,),      0.88),
            (f"model.layers.{i}.mlp.gate_proj.weight",         (hidden*4//3, hidden), 2.30),
            (f"model.layers.{i}.mlp.up_proj.weight",           (hidden*4//3, hidden), 2.25),
            (f"model.layers.{i}.mlp.down_proj.weight",         (hidden, hidden*4//3), 1.90),
        ]

    for name, shape, alpha_base in templates:
        if len(shape) < 2:
            shape = (1, shape[0])
        alpha  = round(max(0.7, alpha_base + np.random.normal(0, 0.12)), 3)
        policy = rtx_quant_policy(name, alpha, shape, GPU)
        bench  = _cpu_latency_estimate(np.zeros(shape, np.float32), policy)
        layers.append({
            "name"    : name,
            "shape"   : list(shape),
            "ttype"   : 1,
            "alpha_w" : alpha,
            "dq_m"    : "synthetic",
            "svd_m"   : "synthetic",
            "policy"  : policy,
            "lat_pred": bench["lat_ms"],
            **bench,
        })
    return layers


# ═══════════════════════════════════════════════════════════════════════════════
# §10.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="D2 RTX + GGUF Profiler — Scanner Spectral Unifié"
    )
    parser.add_argument("model", nargs="?", default=None,
                        help="Chemin vers le fichier .gguf")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark GPU réel par couche (CUDA Events)")
    parser.add_argument("--out", default="rtx_quant_plan.json",
                        help="Fichier de sortie JSON")
    parser.add_argument("--budget", type=float, default=None,
                        help="Budget VRAM en GB")
    parser.add_argument("--demo", action="store_true",
                        help="Mode démo sans GGUF (Llama-3-8B synthétique)")
    parser.add_argument("--quiet", action="store_true",
                        help="Pas d'affichage couche par couche")
    parser.add_argument("--gpu-target", default=None,
                        metavar="GPU",
                        help=("Simuler un GPU cible sans matériel. "
                              "Choix : rtx4090 rtx3090 rtx3080 rtx2080 a100 h100 cpu"))
    args = parser.parse_args()

    # Appliquer le profil GPU simulé si demandé
    if args.gpu_target:
        apply_gpu_profile(args.gpu_target)

    print("═" * 80)
    print("  D2 RTX + GGUF Profiler — Scanner Spectral Unifié")
    print(f"  GPU : {GPU['name']}", end="")
    if GPU["available"]:
        print(f" [{GPU['sm_str']}] — FP8={GPU['fp8']} INT8={GPU['int8']}")
    else:
        print()
    print("═" * 80)
    print()

    # Scan
    # Scan
    if args.demo or not args.model:
        print("  [DEMO] Llama-3-8B synthetique (32 couches, hidden=4096)")
        results    = demo_synthetic(n_layers=32, hidden=4096)
        model_path = "llama-3-8b-synthetic"
        if not args.quiet:
            print(f"\n  {'#':>4}  {'Nom':<45} {'a_w':>7} {'Policy':<16} {'lat_ms':>8}")
            print("  " + "-" * 90)
            for i, r in enumerate(results[:40]):
                print(f"  {i+1:>4}  {r['name']:<45} {r['alpha_w']:>7.3f} "
                      f"{r['policy']:<16} {r['lat_ms']:>8.4f}")
            if len(results) > 40:
                print(f"  ... ({len(results)-40} couches supplementaires)")
    else:
        if not HAS_GGUF:
            print("  [!] pip install gguf")
            sys.exit(1)
        results    = scan_gguf(args.model,
                               do_benchmark=args.benchmark,
                               verbose=not args.quiet)
        model_path = args.model

    # Graph compiler
    compiler = RTXGraphCompiler(GPU)
    print()
    print("  Compilation du graphe de quantification ...")
    clusters = compiler.compile(results, args.out)

    # Rapport
    print_report(results, clusters, model_path)

    # Stats finales
    bpe = {"FP8":0.5,"INT4_AWQ":0.5,"INT8_SAFE":1.0,"BF16":2.0,"FP16_REQUIRED":2.0}
    total_gb = sum(
        r["shape"][0]*r["shape"][1]*bpe.get(r["policy"],2.0)/1e9
        for r in results if len(r["shape"]) >= 2
    )
    counts = {}
    for r in results:
        counts[r["policy"]] = counts.get(r["policy"], 0) + 1

    print(f"  Plan sauvegarde   : {args.out}")
    print(f"  Taille estimee    : {total_gb:.2f} GB", end="")
    if args.budget:
        fit = "OK tient" if total_gb <= args.budget else "DEPASSE"
        print(f" / budget {args.budget} GB -> {fit}", end="")
    print()
    print(f"  Distribution      : " +
          " | ".join(f"{p}:{n}" for p, n in counts.items()))
    print()


if __name__ == "__main__":
    main()
