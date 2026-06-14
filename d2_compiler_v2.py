#!/usr/bin/env python3
"""
D2 Compiler v2 — production-grade quantization planner
=======================================================
Corrige les 7 bugs du Compiler v1 :

  [F1] ablation mismatch          → opt_layers dans zip (pas all_layers)
  [F2] double fragmentation       → single source : post-solve 5% flat
  [F3] forced layers en runtime   → hard constraint x[i,'FP16']=1 dans ILP
  [F4] λ-invariance               → normalisation G'=G/μ_G  R'=R/σ_R
  [F5] roofline decode-only       → mode='decode'|'prefill'
  [F6] KV cache FP16 fixe         → kv_dtype_factor param
  [F7] features simulées          → from_huggingface() via KPEv14 réel

Architecture (TVM / TensorRT style) :
  FeatureExtract → CostModel → ILPSolver → ExecutionPlan → Runtime
                                 ↑
                         forced layers : x[i,FP16]=1  (pas de post-injection)

Formulation corrigée :
  min Σ_{i,q}  x_{i,q} · [ -G'_i(q) + λH·H'_i(q) + λR·R'_i(q) ]
  s.t.  Σ_q x_{i,q} = 1   ∀ i
        Σ_{i,q} VRAM_{i,q}·x_{i,q} ≤ B_eff  (budget après KV+Act, 1 seule fois)
        x_{lm_head, FP16} = 1                (hard constraint)
        x_{i,q} ∈ {0,1}
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# ─── Constantes ─────────────────────────────────────────────────────────────

PRECISIONS = ['FP16', 'INT8', 'INT4']
BPE        = {'FP16': 2.0, 'INT8': 1.0, 'INT4': 0.5}
QUANT_SENS = {'FP16': 0.0, 'INT8': 0.35, 'INT4': 1.0}

FORCED_FP16 = ('lm_head', 'embed_tokens', 'tok_embeddings',
               'wte', 'wpe')   # GPT-2 + LLaMA patterns

KV_DTYPE_FACTOR = {'FP16': 1.0, 'INT8': 0.5, 'INT4': 0.25}


# ─── [F7] Feature extractor KPEv14 (Power Iteration SVD) ───────────────────

class KPEv14:
    """
    Extrait alpha_w (power-law decay), spectral_gap, entropy_norm
    depuis un tensor de poids réel via Power Iteration.
    Compatible avec tout tensor 2D ≥ (5, 5).
    """
    def __init__(self, sketch_rank: int = 64):
        self.r = sketch_rank

    def extract(self, W: np.ndarray) -> dict:
        """
        W : np.ndarray shape (m, n)
        Retourne : {alpha_w, spectral_gap, entropy_norm, hw_stab_proxy}
        """
        if W.ndim != 2:
            W = W.reshape(W.shape[0], -1)
        m, n = W.shape
        r = min(self.r, m, n, 128)

        if r < 5 or min(m, n) < 5:
            return {'alpha_w': 1.5, 'spectral_gap': 0.3,
                    'entropy_norm': 0.7, 'hw_stab': 0.7}

        # Power iteration (2 passes, stable)
        rng   = np.random.default_rng(seed=0)
        Omega = rng.standard_normal((n, r)).astype(np.float32)
        Y     = W.astype(np.float32) @ Omega
        Y     = W.astype(np.float32) @ (W.astype(np.float32).T @ Y)
        Q, _  = np.linalg.qr(Y)
        B     = Q.T @ W.astype(np.float32)
        _, S, _ = np.linalg.svd(B, full_matrices=False)

        S = S[S > 1e-8]
        if len(S) < 5:
            return {'alpha_w': 1.5, 'spectral_gap': 0.3,
                    'entropy_norm': 0.7, 'hw_stab': 0.7}

        # Alpha_w : pente log-log spectre
        log_r = np.log(np.arange(1, len(S)+1, dtype=np.float32))
        log_s = np.log(S + 1e-10)
        alpha_w = float(-np.polyfit(log_r, log_s, 1)[0])

        # Spectral gap : (σ0 - σ1) / σ0
        spectral_gap = float((S[0] - S[1]) / (S[0] + 1e-9))

        # Entropie normalisée
        p = S**2 / (np.sum(S**2) + 1e-10)
        ent = float(-np.sum(p * np.log(p + 1e-10)))
        entropy_norm = ent / max(1.0, math.log(len(S)))

        # Proxy hw_stab : inverse de la concentration spectrale
        var_conc = float(np.max(p))
        hw_stab  = float(np.clip(1.0 - var_conc, 0.4, 0.95))

        return {
            'alpha_w'     : round(alpha_w, 4),
            'spectral_gap': round(spectral_gap, 4),
            'entropy_norm': round(entropy_norm, 4),
            'hw_stab'     : round(hw_stab, 4),
        }


# ─── [F7] HuggingFace loader ────────────────────────────────────────────────

def from_huggingface(model_id: str,
                     mode: str = 'decode',
                     kv_dtype: str = 'FP16',
                     cache_dir: str = '/tmp/hf_cache',
                     max_layers: Optional[int] = None,
                     verbose: bool = True) -> dict:
    """
    Charge un modèle HuggingFace, extrait les features spectrales via KPEv14.

    model_id  : "gpt2" | "TinyLlama/TinyLlama-1.1B-Chat-v1.0" | local path
    mode      : 'decode' (memory-bound) | 'prefill' (compute-bound)
    kv_dtype  : 'FP16' | 'INT8' (quantification KV cache)

    Retourne : dict compatible D2Compiler.compile()
    """
    try:
        from huggingface_hub import hf_hub_download
        import safetensors.torch as st
    except ImportError as e:
        raise ImportError(f"pip install huggingface_hub safetensors : {e}")

    # ── 1. Config ────────────────────────────────────────────────────────────
    cfg_path = hf_hub_download(model_id, 'config.json', cache_dir=cache_dir)
    with open(cfg_path) as f:
        cfg = json.load(f)

    arch = cfg.get('architectures', ['Unknown'])[0]
    n_layers    = cfg.get('num_hidden_layers', cfg.get('n_layer', 12))
    hidden_size = cfg.get('hidden_size', cfg.get('n_embd', 768))
    n_heads     = cfg.get('num_attention_heads', cfg.get('n_head', 12))
    n_kv_heads  = cfg.get('num_key_value_heads', n_heads)
    vocab_size  = cfg.get('vocab_size', 50257)
    d_head      = hidden_size // n_heads

    if verbose:
        print(f"\n  [HF] {model_id}  arch={arch}")
        print(f"  [HF] n_layers={n_layers}  hidden={hidden_size}"
              f"  heads={n_heads}  kv_heads={n_kv_heads}  vocab={vocab_size}")

    # ── 2. Télécharger poids (safetensors) ────────────────────────────────
    # Chercher model.safetensors ou sharded
    from huggingface_hub import list_repo_files
    repo_files = list(list_repo_files(model_id))
    st_files   = [f for f in repo_files if f.endswith('.safetensors')
                  and 'onnx' not in f and 'gguf' not in f]
    if not st_files:
        raise FileNotFoundError(f"Pas de .safetensors pour {model_id}")

    if verbose:
        print(f"  [HF] Downloading {st_files} ...")

    kpe    = KPEv14(sketch_rank=64)
    layers = []
    count  = 0

    for st_file in st_files:
        local = hf_hub_download(model_id, st_file, cache_dir=cache_dir)
        tensors = st.load_file(local)

        for name, tensor in tensors.items():
            if max_layers and count >= max_layers:
                break

            # Filtrer : seulement matrices de poids (2D, assez grandes)
            if tensor.ndim < 2:
                continue
            t = tensor.float().numpy()
            if t.ndim > 2:
                t = t.reshape(t.shape[0], -1)
            m, n = t.shape
            if m < 8 or n < 8:
                continue

            # Feature extraction
            feats = kpe.extract(t)

            layers.append({
                'name'        : name,
                'shape'       : [m, n],
                'alpha_w'     : feats['alpha_w'],
                'spectral_gap': feats['spectral_gap'],
                'entropy_norm': feats['entropy_norm'],
                'hw_stab'     : feats['hw_stab'],
                'l2_stall'    : float(np.clip((m*n*2)/(48*1e6), 0.05, 0.90)),
            })
            count += 1

            if verbose and count % 20 == 0:
                print(f"  [KPE] {count} layers extracted ...")

    if verbose:
        print(f"  [HF] Done — {len(layers)} weight matrices extracted")

    return {
        'name'       : model_id.split('/')[-1],
        'layers'     : layers,
        'n_layers'   : n_layers,
        'n_kv_heads' : n_kv_heads,
        'd_head'     : d_head,
        'arch'       : arch,
        'mode'       : mode,
        'kv_dtype'   : kv_dtype,
    }


# ─── [F5] CostModel : roofline decode|prefill + normalisation ───────────────

class CostModel:
    """
    Roofline unifié : decode (S=1, memory-bound) ou prefill (S≥512, compute-bound).

    [F4] Normalisation G' et R' pour éviter λ-invariance :
      G'_i(q) = G_i(q) / μ_G    (μ_G = mean gain over all non-FP16)
      R'_i(q) = R_i(q) / σ_R    (σ_R = std risk over all)
    Les λ s'appliquent sur des grandeurs O(1).
    """

    def __init__(self, hw_params: dict, mode: str = 'decode'):
        self.hw   = hw_params
        self.mode = mode

        bw_gbs = hw_params['bw_gbs']
        self.bw_eff = (bw_gbs * 1e9
                       * hw_params.get('l2_hit_ratio', 0.65)
                       * hw_params.get('warp_occupancy', 0.80))
        self.peak_flops = hw_params['peak_flops_tf'] * 1e12

    # [F5] latence roofline selon le mode
    def _latency(self, shape: Tuple[int,int], precision: str,
                 batch: int = 1) -> float:
        m, n   = shape
        bpe    = BPE[precision]
        bytes_ = m * n * bpe
        flops  = 2 * m * n * batch

        mem_bound     = bytes_   / self.bw_eff
        compute_bound = flops    / self.peak_flops

        if self.mode == 'decode':
            return mem_bound                       # S=1, toujours mémoire
        else:                                      # prefill S≥512
            return max(mem_bound, compute_bound)   # roofline correct

    def gain(self, shape: Tuple[int,int], precision: str) -> float:
        """G_i(q) = log(T_FP16 / T_q) ≥ 0."""
        t_fp16 = self._latency(shape, 'FP16')
        t_q    = self._latency(shape, precision)
        return float(math.log(max(t_fp16, 1e-30) / max(t_q, 1e-30)))

    def hw_risk(self, hw_stab: float, precision: str) -> float:
        return (1.0 - hw_stab) * QUANT_SENS[precision]

    def spec_risk(self, alpha_w: float, gap: float, precision: str) -> float:
        pressure = float(np.clip(math.tanh(max(0.0, alpha_w - 2.0)), 0, 1))
        gap_sens = float(np.clip(1.0 - gap, 0, 1))
        return pressure * gap_sens * QUANT_SENS[precision]

    def build_matrices(self, layers: List[dict],
                       lam: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        [F4] Construit cost_matrix normalisé et vram_matrix.

        cost[i,q] = -G'_i(q) + λH·H'_i(q) + λR·R'_i(q)

        G' = G / (μ_G + ε)   — gains comparables entre layers
        H' = H / (σ_H + ε)   — risques HW sur même échelle
        R' = R / (σ_R + ε)   — risques spectraux sur même échelle
        """
        n, P = len(layers), len(PRECISIONS)
        G = np.zeros((n, P))
        H = np.zeros((n, P))
        R = np.zeros((n, P))
        V = np.zeros((n, P))

        for i, layer in enumerate(layers):
            shape    = tuple(layer['shape'])
            alpha_w  = layer.get('alpha_w', 1.5)
            gap      = layer.get('spectral_gap', 0.3)
            hw_stab  = layer.get('hw_stab', 0.7)

            for qi, prec in enumerate(PRECISIONS):
                G[i, qi] = self.gain(shape, prec)
                H[i, qi] = self.hw_risk(hw_stab, prec)
                R[i, qi] = self.spec_risk(alpha_w, gap, prec)
                V[i, qi] = shape[0] * shape[1] * BPE[prec]  # bytes

        # [F4] Normalisation (évite λ-invariance)
        # μ_G : gain moyen sur précisions non-FP16 (qi>0)
        mu_G  = np.mean(G[:, 1:]) + 1e-8
        sig_H = np.std(H)         + 1e-8
        sig_R = np.std(R)         + 1e-8

        G_norm = G / mu_G
        H_norm = H / sig_H
        R_norm = R / sig_R

        cost = -G_norm + lam['H'] * H_norm + lam['R'] * R_norm
        return cost, V


# ─── [F2][F6] MemoryModel : source unique ───────────────────────────────────

class MemoryModel:
    """
    VRAM = W(Q) + KV(seq, kv_dtype) + Act_peak
    Fragmentation : post-solve seulement (+5% flat).  [F2]
    KV dtype configurable.  [F6]
    """

    @staticmethod
    def kv_bytes(seq_len: int, n_layers: int,
                 n_kv_heads: int, d_head: int,
                 kv_dtype: str = 'FP16') -> float:
        factor = KV_DTYPE_FACTOR.get(kv_dtype, 1.0)
        return 2.0 * n_layers * seq_len * n_kv_heads * d_head * 2.0 * factor

    @staticmethod
    def act_peak(max_shape: Tuple[int,int]) -> float:
        return 2.0 * max(max_shape) * 2.0   # 1 matrice FP16

    @staticmethod
    def total(layer_shapes: List[Tuple],
              precisions: List[str],
              seq_len: int, n_layers: int,
              n_kv_heads: int, d_head: int,
              kv_dtype: str = 'FP16') -> dict:
        W   = sum(s[0]*s[1]*BPE[p] for s, p in zip(layer_shapes, precisions))
        KV  = MemoryModel.kv_bytes(seq_len, n_layers, n_kv_heads, d_head, kv_dtype)
        Act = MemoryModel.act_peak(max(layer_shapes, key=lambda x: x[0]*x[1]))
        Frag = (W + KV + Act) * 0.05   # [F2] 5% post-solve, une seule fois
        return {'W': W, 'KV': KV, 'Act': Act, 'Frag': Frag,
                'total': W + KV + Act + Frag}


# ─── LambdaScheduler (log-calibré) ──────────────────────────────────────────

class LambdaScheduler:
    """
    λ calibrés pour G' ∈ O(1) après normalisation.
    Seuil d'indifférence : λH ≈ 1 (G'=1 ↔ H'=1 à seuil).
    """
    def schedule(self, vram_ratio: float, stability_mean: float) -> dict:
        if vram_ratio < 0.50:
            lam = {'H': 0.3, 'R': 0.1, 'S': 0.05}   # compute-first
        elif vram_ratio < 0.85:
            lam = {'H': 1.0, 'R': 0.5, 'S': 0.3}    # balanced
        else:
            lam = {'H': 3.0, 'R': 1.5, 'S': 1.0}    # risk-averse

        if stability_mean < 0.40:
            lam['H'] *= 1.5
            lam['R'] *= 1.5
        return lam


# ─── [F3] ILPSolver : forced layers comme hard constraints ──────────────────

class ILPSolver:
    """
    ILP avec scipy.optimize.milp.
    [F3] Les couches forcées FP16 entrent dans le solver comme
         contraintes d'égalité (x[i,'FP16']=1), pas en post-injection.
         → contrainte VRAM cohérente avec l'exécution réelle.
    """

    def solve(self, cost_matrix: np.ndarray,
              vram_matrix:  np.ndarray,
              vram_budget:  float,
              forced_mask:  List[bool]) -> np.ndarray:
        """
        cost_matrix : (n_layers, 3)
        vram_matrix : (n_layers, 3)
        forced_mask : (n_layers,)  True → x[i, FP16] = 1
        """
        n, P    = cost_matrix.shape
        n_vars  = n * P
        c       = cost_matrix.flatten().astype(float)

        rows_data = []

        # C1 : Σ_q x[i,q] = 1  pour tout i
        A_one = np.zeros((n, n_vars))
        for i in range(n):
            A_one[i, i*P:(i+1)*P] = 1.0

        lb_one = np.ones(n)
        ub_one = np.ones(n)

        # [F3] C2 : forced layers → x[i, 0] = 1  (FP16 = index 0)
        forced_idx = [i for i, f in enumerate(forced_mask) if f]
        A_forced   = np.zeros((len(forced_idx), n_vars))
        for row, i in enumerate(forced_idx):
            A_forced[row, i*P + 0] = 1.0   # x[i, FP16] = 1

        lb_forced = np.ones(len(forced_idx))
        ub_forced = np.ones(len(forced_idx))

        # C3 : VRAM ≤ budget
        A_vram = vram_matrix.flatten().reshape(1, -1).astype(float)
        lb_vram = np.array([-np.inf])
        ub_vram = np.array([vram_budget])

        # Stack all constraints
        A   = np.vstack([A_one, A_forced, A_vram]) if forced_idx else \
              np.vstack([A_one, A_vram])
        lb  = np.concatenate([lb_one, lb_forced, lb_vram]) if forced_idx else \
              np.concatenate([lb_one, lb_vram])
        ub  = np.concatenate([ub_one, ub_forced, ub_vram]) if forced_idx else \
              np.concatenate([ub_one, ub_vram])

        constraints = LinearConstraint(A, lb, ub)
        bounds      = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)

        res = milp(c, constraints=constraints,
                   integrality=integrality, bounds=bounds)

        if res.status == 0:
            x = res.x.reshape(n, P)
            return np.argmax(x, axis=1).astype(int)
        else:
            return self._greedy(cost_matrix, vram_matrix, vram_budget, forced_mask)

    def _greedy(self, cost_matrix, vram_matrix, budget, forced_mask):
        n, P   = cost_matrix.shape
        asgn   = np.full(n, P-1, dtype=int)   # INT4 par défaut

        # Respecter forced
        for i, f in enumerate(forced_mask):
            if f:
                asgn[i] = 0  # FP16

        total = sum(vram_matrix[i, asgn[i]] for i in range(n))
        if total <= budget:
            return asgn

        # Relâcher vers précision plus haute si dépassement
        candidates = sorted(
            [(cost_matrix[i,q] - cost_matrix[i,P-1],
              vram_matrix[i,q] - vram_matrix[i,P-1], i, q)
             for i in range(n) if not forced_mask[i]
             for q in range(P-1)
             if vram_matrix[i,q] > vram_matrix[i,P-1]],
            key=lambda x: x[0] / (x[1]+1)
        )
        for _, dv, i, q in candidates:
            if total <= budget:
                break
            total += vram_matrix[i,q] - vram_matrix[i,asgn[i]]
            asgn[i] = q
        return asgn


# ─── DP switching cost (time domain) ────────────────────────────────────────

def dp_switching(assignment: np.ndarray,
                 cost_matrix: np.ndarray,
                 vram_matrix: np.ndarray,
                 vram_budget: float,
                 forced_mask: List[bool],
                 gamma_s: float) -> np.ndarray:
    """
    [Bug 4 fix] gamma_s en time domain normalisé (pas log-space).
    Seulement sur les couches non-forcées.
    """
    opt_idx = [i for i, f in enumerate(forced_mask) if not f]
    if not opt_idx:
        return assignment

    sub_cost = cost_matrix[opt_idx]
    sub_vram = vram_matrix[opt_idx]
    m, P     = sub_cost.shape
    INF      = 1e18

    dp   = np.full((m, P), INF)
    back = np.zeros((m, P), dtype=int)
    dp[0, :] = sub_cost[0, :]

    for i in range(1, m):
        for q in range(P):
            for qp in range(P):
                sw  = gamma_s if q != qp else 0.0
                val = dp[i-1, qp] + sub_cost[i, q] + sw
                if val < dp[i, q]:
                    dp[i, q]   = val
                    back[i, q] = qp

    # Backtrack
    sub_asgn = np.zeros(m, dtype=int)
    sub_asgn[-1] = int(np.argmin(dp[-1]))
    for i in range(m-2, -1, -1):
        sub_asgn[i] = back[i+1, sub_asgn[i+1]]

    # Vérifier VRAM (sans les couches forcées)
    forced_vram = sum(vram_matrix[i, assignment[i]]
                      for i, f in enumerate(forced_mask) if f)
    opt_vram    = sum(sub_vram[j, sub_asgn[j]] for j in range(m))
    if forced_vram + opt_vram > vram_budget:
        return assignment   # fallback ILP

    result = assignment.copy()
    for j, i in enumerate(opt_idx):
        result[i] = sub_asgn[j]
    return result


# ─── Compilateur principal ───────────────────────────────────────────────────

class D2Compiler:
    """
    D2 Compiler v2 — pipeline 4 couches sans injection post-ILP.

    Usage :
        spec   = from_huggingface('gpt2')         # features réelles
        result = D2Compiler(hw, mp).compile(spec, 6.0)
        print(result['summary'])
        print(result['llama_cpp'])
    """

    def __init__(self, hw_params: dict, memory_params: dict):
        mode = memory_params.get('mode', 'decode')
        self.cost_model = CostModel(hw_params, mode=mode)
        self.lam_sched  = LambdaScheduler()
        self.ilp        = ILPSolver()
        self.mp         = memory_params

    def compile(self, model_spec: dict, vram_budget_gb: float) -> dict:
        layers      = model_spec['layers']
        n_layers_a  = model_spec.get('n_layers', len(layers))
        n_kv_heads  = model_spec.get('n_kv_heads', self.mp.get('n_kv_heads', 8))
        d_head      = model_spec.get('d_head',     self.mp.get('d_head', 128))
        seq_len     = self.mp.get('seq_len', 4096)
        kv_dtype    = model_spec.get('kv_dtype', self.mp.get('kv_dtype', 'FP16'))
        budget_b    = vram_budget_gb * 1024**3

        # [F3] Identifier les couches forcées AVANT ILP
        forced_mask = [any(p in l['name'] for p in FORCED_FP16)
                       for l in layers]

        # [F2] Budget net = total - KV - Act - forced_W  (une seule fois)
        kv_b   = MemoryModel.kv_bytes(seq_len, n_layers_a, n_kv_heads,
                                       d_head, kv_dtype)
        max_shape = max((l["shape"] for l in layers),
                        key=lambda s: s[0]*s[1])
        act_b  = MemoryModel.act_peak(tuple(max_shape))
        frc_b  = sum(l['shape'][0]*l['shape'][1]*BPE['FP16']
                     for l, f in zip(layers, forced_mask) if f)
        # [F2] Fragmentation unique post-solve : intégrée dans budget_eff
        budget_eff = budget_b - kv_b - act_b - frc_b * 1.05

        # VRAM FP16 des couches optimisables → ratio pour λ scheduling
        w_fp16 = sum(l['shape'][0]*l['shape'][1]*2.0
                     for l, f in zip(layers, forced_mask) if not f)
        vram_ratio     = (w_fp16 + kv_b) / budget_b
        stability_mean = float(np.mean([l.get('hw_stab', 0.7) for l in layers]))

        # λ scheduling
        lam = self.lam_sched.schedule(vram_ratio, stability_mean)

        # [F4] cost_matrix normalisé (G'/R')
        cost_matrix, vram_matrix = self.cost_model.build_matrices(layers, lam)

        # [F3] ILP avec forced layers comme hard constraints
        assignment = self.ilp.solve(cost_matrix, vram_matrix,
                                    budget_eff, forced_mask)

        # DP switching cost en time domain
        gamma_s = 0.05   # [Bug4 fix] 5% latence absolue normalisée
        assignment = dp_switching(assignment, cost_matrix, vram_matrix,
                                  budget_eff, forced_mask, gamma_s)

        # ── Plan final (NO modification post-ILP) ─────────────────────────
        plan = []
        for layer, qi, is_forced in zip(layers, assignment, forced_mask):
            prec   = PRECISIONS[qi]
            reason = ('forced_fp16' if is_forced else 'ilp')
            g_val  = self.cost_model.gain(tuple(layer['shape']), prec)
            h_val  = self.cost_model.hw_risk(layer.get('hw_stab',0.7), prec)
            r_val  = self.cost_model.spec_risk(
                        layer.get('alpha_w',1.5),
                        layer.get('spectral_gap',0.3), prec)
            plan.append({
                'layer'    : layer['name'],
                'shape'    : layer['shape'],
                'precision': prec,
                'reason'   : reason,
                'gain_log' : round(g_val, 4),
                'h_risk'   : round(h_val, 4),
                'r_spec'   : round(r_val, 4),
                'vram_gb'  : round(layer['shape'][0]*layer['shape'][1]*BPE[prec]/1e9, 4),
            })

        # Métriques
        prec_list   = [e['precision'] for e in plan]
        p_counts    = {p: prec_list.count(p) for p in PRECISIONS}
        n_sw        = sum(1 for a,b in zip(prec_list,prec_list[1:]) if a!=b)
        total_gain  = sum(e['gain_log'] for e in plan)
        total_h     = sum(e['h_risk']   for e in plan)
        total_r     = sum(e['r_spec']   for e in plan)

        vram_d = MemoryModel.total(
            [l['shape'] for l in layers], prec_list,
            seq_len, n_layers_a, n_kv_heads, d_head, kv_dtype)
        vram_gb = vram_d['total'] / 1024**3

        regime = ('compute' if vram_ratio < 0.50 else
                  'balanced' if vram_ratio < 0.85 else 'risk')

        # llama.cpp output
        gguf_map = {'FP16': 'f16', 'INT8': 'q8_0', 'INT4': 'q4_K_M'}
        llama_cpp = ' \\\n  '.join(
            f"--override-tensor {e['layer']}={gguf_map[e['precision']]}"
            for e in plan)

        summary = '\n'.join([
            f"\n{'='*62}",
            f"  D2 Compiler v2 — {model_spec['name']}",
            f"{'='*62}",
            f"  VRAM budget    : {vram_budget_gb:.1f} GB",
            f"  VRAM utilisé   : {vram_gb:.3f} GB  ({vram_gb/vram_budget_gb*100:.1f}%)",
            f"    W(Q)         : {vram_d['W']/1e9:.3f} GB",
            f"    KV ({kv_dtype:<4})   : {vram_d['KV']/1e9:.3f} GB  seq={seq_len}",
            f"    Act_peak     : {vram_d['Act']/1e9:.3f} GB",
            f"    Frag (5%)    : {vram_d['Frag']/1e9:.3f} GB",
            f"",
            f"  λ-régime       : {regime}",
            f"  λH={lam['H']:.2f}  λR={lam['R']:.2f}  λS={lam['S']:.2f}",
            f"  vram_ratio     : {vram_ratio:.3f}  stab_mean={stability_mean:.3f}",
            f"",
            f"  Plan           : INT4={p_counts['INT4']}  INT8={p_counts['INT8']}"
                f"  FP16={p_counts['FP16']}  (total={len(plan)})",
            f"  Transitions    : {n_sw} switches  γ_s={gamma_s}",
            f"  Gain log Σ     : {total_gain:.4f}",
            f"  Risque HW Σ    : {total_h:.4f}",
            f"  Risque Spec Σ  : {total_r:.4f}",
        ])

        return {
            'plan'      : plan,
            'vram_detail': vram_d,
            'lambdas'   : lam,
            'n_switches': n_sw,
            'total_gain': total_gain,
            'vram_gb'   : vram_gb,
            'summary'   : summary,
            'llama_cpp' : llama_cpp,
            'tensorrt'  : {
                'precision_map': {e['layer']: e['precision'] for e in plan},
                'forced_fp16'  : [e['layer'] for e in plan if e['reason']=='forced_fp16'],
            },
        }


# ─── [F1] Ablation λ (fixed : opt_layers dans zip) ──────────────────────────

def ablation_lambda(compiler: D2Compiler, model_spec: dict,
                    vram_budget_gb: float) -> dict:
    """
    [F1] Fix : utilise opt_layers pour zip (pas all layers).
    Compare 3 régimes λ sur l'espace décisionnel identique.
    """
    layers      = model_spec['layers']
    n_layers_a  = model_spec.get('n_layers', len(layers))
    n_kv_heads  = model_spec.get('n_kv_heads', 8)
    d_head      = model_spec.get('d_head', 128)
    seq_len     = compiler.mp.get('seq_len', 4096)
    kv_dtype    = model_spec.get('kv_dtype', 'FP16')
    budget_b    = vram_budget_gb * 1024**3

    forced_mask = [any(p in l['name'] for p in FORCED_FP16) for l in layers]
    # [F1] opt_layers : seulement les couches non-forcées
    opt_layers  = [l for l, f in zip(layers, forced_mask) if not f]

    kv_b  = MemoryModel.kv_bytes(seq_len, n_layers_a, n_kv_heads, d_head, kv_dtype)
    frc_b = sum(l['shape'][0]*l['shape'][1]*BPE['FP16']
                for l, f in zip(layers, forced_mask) if f)
    max_s = max((l["shape"] for l in layers), key=lambda s: s[0]*s[1])
    act_b = MemoryModel.act_peak(tuple(max_s))
    budget_eff = budget_b - kv_b - act_b - frc_b * 1.05

    results = {}
    for regime, lam in [
        ('compute',  {'H': 0.3, 'R': 0.1, 'S': 0.05}),
        ('balanced', {'H': 1.0, 'R': 0.5, 'S': 0.3}),
        ('risk',     {'H': 3.0, 'R': 1.5, 'S': 1.0}),
    ]:
        cost_m, vram_m = compiler.cost_model.build_matrices(opt_layers, lam)
        forced_opt = [False] * len(opt_layers)
        asgn = compiler.ilp.solve(cost_m, vram_m, budget_eff, forced_opt)
        asgn = dp_switching(asgn, cost_m, vram_m, budget_eff, forced_opt, 0.05)

        prec_list = [PRECISIONS[q] for q in asgn]
        counts    = {p: prec_list.count(p) for p in PRECISIONS}
        n_sw      = sum(1 for a,b in zip(prec_list,prec_list[1:]) if a!=b)

        # [F1] gain total sur opt_layers uniquement (pas all layers)
        total_gain = float(sum(
            compiler.cost_model.gain(tuple(opt_layers[j]['shape']), PRECISIONS[asgn[j]])
            for j in range(len(opt_layers))))

        results[regime] = {**counts, 'switches': n_sw,
                            'gain_log': round(total_gain, 4)}
    return results

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        from tabulate import tabulate
        TAB = True
    except ImportError:
        TAB = False

    hw_params = {
        'peak_flops_tf' : 82.6,    # A100 FP16 TFLOPs
        'bw_gbs'        : 2000.0,  # HBM2e GB/s
        'l2_hit_ratio'  : 0.65,
        'warp_occupancy': 0.80,
    }
    memory_params = {
        'seq_len'   : 4096,
        'n_kv_heads': 8,
        'd_head'    : 128,
        'mode'      : 'decode',
        'kv_dtype'  : 'FP16',
    }

    # ── Test 1 : GPT-2 real weights (HuggingFace) ────────────────────────
    print("\n" + "#"*62)
    print("  TEST 1 : GPT-2 (real weights from HuggingFace)")
    print("#"*62)

    spec_gpt2 = from_huggingface(
        'gpt2',
        mode='decode',
        kv_dtype='FP16',
        cache_dir='/tmp/hf_cache',
        verbose=True,
    )

    # GPT-2 n'a pas GQA ; d_head = 64 (12 heads, 768 hidden)
    spec_gpt2['n_kv_heads'] = 12
    spec_gpt2['d_head']     = 64

    # Afficher les features spectrales réelles
    print("\n  === Features KPEv14 (real weights, sample 10 layers) ===")
    sample = spec_gpt2['layers'][:10]
    rows_feat = [[l['name'][:45], l['shape'],
                  f"{l['alpha_w']:.3f}", f"{l['spectral_gap']:.3f}",
                  f"{l['hw_stab']:.3f}"]
                 for l in sample]
    hdrs = ['Name', 'Shape', 'alpha_w', 'spec_gap', 'hw_stab']
    if TAB:
        print(tabulate(rows_feat, headers=hdrs, tablefmt='psql'))
    else:
        print(f"  {'Name':<45} {'Shape':<16} alpha_w  spec_gap  hw_stab")
        for r in rows_feat:
            print(f"  {str(r[0]):<45} {str(r[1]):<16} {r[2]:>7}  {r[3]:>8}  {r[4]:>7}")

    # Compiler GPT-2 : budget 4 GB (petit GPU pour forcer INT4)
    compiler_gpt2 = D2Compiler(hw_params, memory_params)
    result_gpt2   = compiler_gpt2.compile(spec_gpt2, vram_budget_gb=4.0)

    print(result_gpt2['summary'])

    # Top 12 layers
    plan = result_gpt2['plan']
    rows_plan = [[e['layer'][:40], e['precision'], e['reason'],
                  f"{e['gain_log']:+.4f}", f"{e['h_risk']:.3f}",
                  f"{e['r_spec']:.3f}", f"{e['vram_gb']:.4f}"]
                 for e in plan[:12]]
    hdrs_p = ['Layer', 'Prec', 'Reason', 'Gain', 'H_risk', 'R_spec', 'VRAM_GB']
    print()
    if TAB:
        print(tabulate(rows_plan, headers=hdrs_p, tablefmt='psql'))
    else:
        print(f"  {'Layer':<40} {'Prec':<6} {'Reason':<14} Gain  H_risk  R_spec VRAM_GB")
        for r in rows_plan:
            print(f"  {r[0]:<40} {r[1]:<6} {r[2]:<14} {r[3]:>6} {r[4]:>7} {r[5]:>7} {r[6]:>7}")

    # Ablation λ [F1 fix]
    print("\n  === ABLATION lambda-REGIME (F1-fix: opt_layers correct) ===")
    abl = ablation_lambda(compiler_gpt2, spec_gpt2, vram_budget_gb=4.0)
    rows_abl = [[rg, d.get('FP16',0), d.get('INT8',0), d.get('INT4',0),
                 d['switches'], f"{d['gain_log']:.4f}"]
                for rg, d in abl.items()]
    if TAB:
        print(tabulate(rows_abl,
              headers=['Regime','FP16','INT8','INT4','Switches','Gain_log'],
              tablefmt='psql'))
    else:
        print(f"  {'Regime':<12} FP16  INT8  INT4  Switches  Gain_log")
        for r in rows_abl:
            print(f"  {r[0]:<12} {r[1]:>4}  {r[2]:>4}  {r[3]:>4}  {r[4]:>8}  {r[5]:>8}")

    # Extrait llama.cpp
    llama_lines = result_gpt2['llama_cpp'].split('\n')
    print(f"\n  === llama.cpp export (extrait) ===")
    for l in llama_lines[:8]:
        print(f"  {l}")
    print(f"  ... ({len(llama_lines)} tensors total)")

    # ── Test 2 : decode vs prefill [F5] ──────────────────────────────────
    print("\n" + "#"*62)
    print("  TEST 2 : decode vs prefill roofline [F5]")
    print("#"*62)

    for mode in ['decode', 'prefill']:
        mp = {**memory_params, 'mode': mode}
        c  = D2Compiler(hw_params, mp)
        r  = c.compile(spec_gpt2, 4.0)
        pc = {p: r['plan'].count({'precision':p}) for p in PRECISIONS}
        pcounts = {p: [e['precision'] for e in r['plan']].count(p) for p in PRECISIONS}
        print(f"  mode={mode:<8} INT4={pcounts['INT4']:>3} INT8={pcounts['INT8']:>3}"
              f" FP16={pcounts['FP16']:>3}  gain={r['total_gain']:.3f}"
              f"  VRAM={r['vram_gb']:.3f} GB")

    # ── Test 3 : KV dtype [F6] ───────────────────────────────────────────
    print("\n" + "#"*62)
    print("  TEST 3 : KV cache dtype [F6]")
    print("#"*62)

    for kv_dt in ['FP16', 'INT8']:
        s = {**spec_gpt2, 'kv_dtype': kv_dt}
        c = D2Compiler(hw_params, {**memory_params, 'kv_dtype': kv_dt})
        r = c.compile(s, 4.0)
        vd = r['vram_detail']
        print(f"  kv_dtype={kv_dt:<5}  KV={vd['KV']/1e9:.3f} GB"
              f"  total={r['vram_gb']:.3f} GB"
              f"  W(Q)={vd['W']/1e9:.3f} GB")

    print(f"\n{'='*62}")
    print("  D2 Compiler v2 -- done")
    print(f"{'='*62}\n")
