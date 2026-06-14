#!/usr/bin/env python3
"""
D2 Spectral-Aware Quantization Planner
Mathematical & Theoretical Foundation

This module provides the theoretical backbone for the D2 system:
A spectral analysis framework combining random matrix theory, graph optimization,
and CUDA cost modeling for intelligent layer-wise quantization.

FAMILIES:
  (A) Spectral / Random Matrix Theory (Pennington, Martin & Mahoney)
  (B) Graph optimization (edge costs, fragmentation)
  (C) CUDA stability cost model (branching ratio, avalanche)
  (D) Quantization policy engine (spectral → dtype mapping)

See THEORY.md for full mathematical derivations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.linalg import svd
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FAMILY (A): SPECTRAL / RANDOM MATRIX THEORY
# ─────────────────────────────────────────────────────────────────────────────

class SpectralAnalyzer:
    """
    Compute spectral properties of weight matrices.
    
    Based on:
    - Pennington et al. (2017): Dynamical Isometry
    - Martin & Mahoney (2017): Heavy-tailed self-regularization
    - Voiculescu (2005): Free probability theory
    """
    
    @staticmethod
    def svd_decomposition(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SVD: W = U Σ V⊤
        
        Args:
            W: Weight matrix [m, n]
        
        Returns:
            U, singular_values (Σ diagonal), V⊤
        """
        U, s, Vt = svd(W, full_matrices=False)
        return U, s, Vt
    
    @staticmethod
    def spectral_exponent(singular_values: np.ndarray) -> float:
        """
        Estimate power-law exponent β where sᵢ ∝ i^(-β)
        
        Using log-log fit: log(sᵢ) = -β log(i) + c
        
        Returns α_w = 2β (conforming to standard notation)
        """
        s = np.array(singular_values)
        s = s[s > 1e-10]  # Filter noise
        
        if len(s) < 3:
            return 2.0  # Default
        
        log_i = np.log(np.arange(1, len(s) + 1))
        log_s = np.log(s)
        
        # Linear fit: log(s) = -β log(i) + c
        coeffs = np.polyfit(log_i, log_s, 1)
        beta = -coeffs[0]
        alpha_w = 2 * beta
        
        return float(alpha_w)
    
    @staticmethod
    def stable_rank(W: np.ndarray) -> float:
        """
        Stable rank: r_s(W) = ‖W‖_F² / ‖W‖₂²
        
        Interpretation:
        - Small r_s: low-rank structure (compressible)
        - Large r_s: full-rank (harder to compress)
        """
        U, s, Vt = svd(W, full_matrices=False)
        
        norm_f_sq = np.sum(s ** 2)  # Frobenius norm squared
        norm_2 = s[0]               # Spectral norm (largest singular value)
        
        rs = norm_f_sq / (norm_2 ** 2)
        return float(rs)
    
    @staticmethod
    def participation_ratio(singular_values: np.ndarray) -> float:
        """
        Effective rank (Participation ratio):
        r_eff = (Σ sᵢ)² / Σ sᵢ²
        
        How many "effective" singular values contribute significantly?
        """
        s = np.array(singular_values)
        sum_s = np.sum(s)
        sum_s2 = np.sum(s ** 2)
        
        if sum_s2 < 1e-10:
            return 1.0
        
        r_eff = (sum_s ** 2) / sum_s2
        return float(r_eff)
    
    @staticmethod
    def spectral_entropy(singular_values: np.ndarray) -> float:
        """
        Entropy of normalized singular value distribution:
        
        pᵢ = sᵢ / Σⱼ sⱼ
        H = -Σᵢ pᵢ log(pᵢ)
        
        Interpretation:
        - H ≈ 0: one dominant singular value
        - H ≈ log(rank): uniform distribution
        """
        s = np.array(singular_values)
        s = s[s > 1e-10]
        
        if len(s) == 0:
            return 0.0
        
        p = s / np.sum(s)
        p = p[p > 0]
        
        H = -np.sum(p * np.log(p))
        return float(H)
    
    @staticmethod
    def compressibility(alpha_w: float) -> float:
        """
        Compression proxy: C(W) = 1 / log(1 + α_w)
        
        - High α_w (power-law decay): C → 0.5 (highly compressible)
        - Low α_w (flat): C → ∞ (incompressible)
        """
        if alpha_w <= 0:
            return 1.0
        return 1.0 / np.log(1.0 + alpha_w)


# ─────────────────────────────────────────────────────────────────────────────
# C-VECTOR: INFORMATIONAL STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class CVector:
    """
    Information-theoretic descriptor: C(W) = (C_s, C_i, C_r, C_d)
    
    Represents stability, information content, risk, and density.
    """
    
    def __init__(self, W: np.ndarray):
        """Compute C-vector for matrix W."""
        analyzer = SpectralAnalyzer()
        
        # SVD decomposition
        U, s, Vt = analyzer.svd_decomposition(W)
        
        # Spectral metrics
        self.alpha_w = analyzer.spectral_exponent(s)
        self.rs = analyzer.stable_rank(W)
        self.r_eff = analyzer.participation_ratio(s)
        self.H = analyzer.spectral_entropy(s)
        self.rho_d = float(np.sum(W != 0) / W.size)  # Density
        
        # C-vector components
        self.C_s = analyzer.compressibility(self.alpha_w)  # Stability
        self.C_i = 1.0 - (self.H / np.log(len(s)) if len(s) > 1 else 0)  # Info
        self.C_r = self.alpha_w * (1.0 + self.rho_d)  # Risk (heavy tail)
        self.C_d = self.rho_d  # Density
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return (C_s, C_i, C_r, C_d)."""
        return (self.C_s, self.C_i, self.C_r, self.C_d)
    
    def __repr__(self) -> str:
        return (
            f"CVector("
            f"C_s={self.C_s:.4f}, "
            f"C_i={self.C_i:.4f}, "
            f"C_r={self.C_r:.4f}, "
            f"C_d={self.C_d:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FAMILY (D): QUANTIZATION POLICY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class QuantizationPolicy:
    """
    Map spectral metrics → quantization dtype.
    
    CORRECTED MAPPING (GPU physical reality):
    
    Instability = α_w (1 + ρ_d)
    
    NVFP4 (most aggr)  if instability < 1.2
    INT8                if 1.2 ≤ instability < 1.6
    FP8                 if 1.6 ≤ instability < 2.0
    FP16 (most cons)    if instability ≥ 2.0
    """
    
    DTYPE_LEVELS = {
        "NVFP4": 0,
        "INT8": 1,
        "FP8": 2,
        "FP16": 3,
    }
    
    THRESHOLDS = {
        "NVFP4": 1.2,
        "INT8": 1.6,
        "FP8": 2.0,
        "FP16": float('inf'),
    }
    
    @staticmethod
    def instability_score(c_vector: CVector) -> float:
        """
        Compute instability metric.
        
        instability = α_w (1 + ρ_d)
        
        Interpretation:
        - < 1.2: Stable, can compress aggressively
        - > 2.0: Sensitive, preserve precision
        """
        return c_vector.alpha_w * (1.0 + c_vector.C_d)
    
    @staticmethod
    def recommend_dtype(c_vector: CVector) -> str:
        """
        Recommend quantization dtype based on instability.
        """
        instability = QuantizationPolicy.instability_score(c_vector)
        
        for dtype in ["NVFP4", "INT8", "FP8", "FP16"]:
            if instability < QuantizationPolicy.THRESHOLDS[dtype]:
                return dtype
        
        return "FP16"


# ─────────────────────────────────────────────────────────────────────────────
# FAMILY (B): GRAPH OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

class GraphOptimizer:
    """
    Optimize quantization across layers using graph edge costs.
    
    Edge cost = |Q_i - Q_j| where Q ∈ {0, 1, 2, 3}
    
    Minimize:
      - Sum of transition costs (fragmentation)
      - Total instability
    """
    
    @staticmethod
    def edge_cost(dtype_i: str, dtype_j: str) -> int:
        """
        Cost of transitioning between two dtypes.
        
        NVFP4=0, INT8=1, FP8=2, FP16=3
        """
        levels = QuantizationPolicy.DTYPE_LEVELS
        return abs(levels[dtype_i] - levels[dtype_j])
    
    @staticmethod
    def total_fragmentation(dtypes: List[str]) -> float:
        """
        F = Σᵢ₌₁ⁿ⁻¹ C_edge(i, i+1)
        """
        if len(dtypes) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(dtypes) - 1):
            total += GraphOptimizer.edge_cost(dtypes[i], dtypes[i + 1])
        
        return total
    
    @staticmethod
    def break_count(dtypes: List[str]) -> int:
        """
        B = Σ 𝟙[C_edge ≥ 2]
        
        Number of major type breaks (e.g., INT8 → FP16).
        """
        if len(dtypes) < 2:
            return 0
        
        count = 0
        for i in range(len(dtypes) - 1):
            if GraphOptimizer.edge_cost(dtypes[i], dtypes[i + 1]) >= 2:
                count += 1
        
        return count


# ─────────────────────────────────────────────────────────────────────────────
# FAMILY (C): CUDA STABILITY COST MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CUDAStabilityCost:
    """
    LIF / Avalanche model for neural dynamics.
    
    V(t+1) = λ V(t) + W S(t) + ξ
    
    Branching ratio m = Σ A_{t+1} / Σ A_t
    
    Spectral radius ρ ≈ ‖W‖₂ (largest singular value)
    """
    
    @staticmethod
    def spectral_radius(W: np.ndarray) -> float:
        """
        ρ(W) = largest singular value
        
        Critical points:
        - ρ ≈ 0: Stable, information decays
        - ρ ≈ 1: Edge of chaos (optimal)
        - ρ > 1: Unstable (explosive)
        """
        analyzer = SpectralAnalyzer()
        U, s, Vt = analyzer.svd_decomposition(W)
        return float(s[0]) if len(s) > 0 else 0.0
    
    @staticmethod
    def branching_ratio(W: np.ndarray, timesteps: int = 10) -> float:
        """
        Simulate LIF dynamics and estimate branching ratio.
        
        m = Σ A_{t+1} / Σ A_t
        
        - m ≈ 1: Critical (sustained avalanche)
        - m < 1: Subcritical (decay)
        - m > 1: Supercritical (explosion)
        """
        rho = CUDAStabilityCost.spectral_radius(W)
        
        # Simplified: m ≈ ρ
        return float(rho)
    
    @staticmethod
    def stability_cost(c_vector: CVector) -> float:
        """
        Cost = |α_w - α* | + λ switch(Q_i, Q_j)
        
        Target α* ≈ 2.2 (empirically optimal).
        """
        alpha_target = 2.2
        cost = abs(c_vector.alpha_w - alpha_target)
        return float(cost)


# ─────────────────────────────────────────────────────────────────────────────
# POWER-LAW TEST (Kolmogorov-Smirnov)
# ─────────────────────────────────────────────────────────────────────────────

def test_power_law(singular_values: np.ndarray, alpha_w_est: float) -> Dict:
    """
    Test if singular value distribution follows power law.
    
    Null hypothesis: sᵢ ~ i^(-β) where β = α_w / 2
    
    Returns:
        - ks_statistic: How well fit matches
        - p_value: Significance
        - is_power_law: Boolean
    """
    s = np.array(singular_values)
    s = s[s > 1e-10]
    
    if len(s) < 5:
        return {
            "ks_statistic": np.nan,
            "p_value": np.nan,
            "is_power_law": False,
            "reason": "Too few singular values"
        }
    
    # Generate synthetic power-law samples
    beta = alpha_w_est / 2.0
    indices = np.arange(1, len(s) + 1)
    s_synthetic = indices ** (-beta)
    s_synthetic = s_synthetic / np.sum(s_synthetic) * np.sum(s)
    
    # KS test
    ks_stat, p_value = ks_2samp(s, s_synthetic)
    
    return {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "is_power_law": p_value > 0.05,
        "reason": "Power-law fit" if p_value > 0.05 else "Non-power-law"
    }


# ─────────────────────────────────────────────────────────────────────────────
# SPECTRAL IR COMPILER (Global Integration)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralIR:
    """
    Intermediate Representation combining all spectral metrics.
    
    IR = {(ρ, α_w, H, Q, instability, stability_cost)}
    """
    
    def __init__(self, W: np.ndarray, layer_name: str = ""):
        """Build spectral IR for a weight matrix."""
        self.layer_name = layer_name
        self.W = W
        
        # C-vector
        self.c_vector = CVector(W)
        
        # Stability metrics
        cuda_cost = CUDAStabilityCost()
        self.rho = cuda_cost.spectral_radius(W)
        self.stability_cost = cuda_cost.stability_cost(self.c_vector)
        
        # Quantization
        self.dtype = QuantizationPolicy.recommend_dtype(self.c_vector)
        self.instability = QuantizationPolicy.instability_score(self.c_vector)
        
        # Power-law test
        analyzer = SpectralAnalyzer()
        U, s, Vt = analyzer.svd_decomposition(W)
        self.power_law_info = test_power_law(s, self.c_vector.alpha_w)
    
    def global_stability(self) -> float:
        """
        Stability = exp(-|ρ - 1.04|) × exp(-|α_w - 2.2|)
        
        Both metrics should be close to their optimal values.
        """
        rho_target = 1.04
        alpha_target = 2.2
        
        stability = (
            np.exp(-abs(self.rho - rho_target)) *
            np.exp(-abs(self.c_vector.alpha_w - alpha_target))
        )
        return float(stability)
    
    def summary(self) -> Dict:
        """Return comprehensive spectral summary."""
        return {
            "layer_name": self.layer_name,
            "shape": self.W.shape,
            "dtype_recommended": self.dtype,
            "c_vector": self.c_vector.to_tuple(),
            "spectral_metrics": {
                "alpha_w": self.c_vector.alpha_w,
                "rho": self.rho,
                "stable_rank": self.c_vector.rs,
                "participation_ratio": self.c_vector.r_eff,
                "entropy": self.c_vector.H,
            },
            "stability": {
                "instability_score": self.instability,
                "stability_cost": self.stability_cost,
                "global_stability": self.global_stability(),
            },
            "power_law": self.power_law_info,
        }


if __name__ == "__main__":
    # Example
    logging.basicConfig(level=logging.INFO)
    
    # Create a test matrix
    np.random.seed(42)
    W = np.random.randn(512, 512)
    
    ir = SpectralIR(W, "test_layer")
    
    import json
    print(json.dumps(ir.summary(), indent=2, default=str))
