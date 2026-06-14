import json
import numpy as np
import torch
from scipy.linalg import svd
from scipy.special import softmax, expit
import os

# =========================================================
# 1. OPTIMIZED SVD & QDF ENGINE
# =========================================================
class OptimizedSVD:
    @staticmethod
    def randomized_svd(W: np.ndarray, n_components=128, n_iter=4):
        if W.shape[0] * W.shape[1] < 3_000_000:
            _, S, _ = np.linalg.svd(W, full_matrices=False)
            return S
        m, n = W.shape
        # Q doit être (n, n_components) si on projette W @ Q
        # Si m > n (tall matrix), Q est (n, k). W @ Q -> (m, k)
        Q = np.random.randn(n, n_components)
        for _ in range(n_iter):
            Q, _ = np.linalg.qr(W @ Q)
            Q, _ = np.linalg.qr(W.T @ Q)
        _, S, _ = np.linalg.svd(W @ Q, full_matrices=False)
        return S

class QDFV5Engine:
    def __init__(self):
        self.w = np.array([0.35, 0.30, 0.20, 0.15])
        self.policy_params = {
            "FP32": np.array([-2.0, -2.0, -1.0, -1.0]),
            "FP16/INT8": np.array([0.5, 0.5, 0.5, 0.5]),
            "INT4": np.array([2.0, 1.5, 2.0, 1.0]),
            "NVFP4_SAFE": np.array([2.2, 1.8, 2.1, 1.2])
        }

    def bayesian_coherence(self, W: np.ndarray) -> float:
        if W.size > 10_000_000:
            S = OptimizedSVD.randomized_svd(W, n_components=64)
            S_norm = S / (np.linalg.norm(S) + 1e-9)
            return float(np.sum(S_norm[:int(len(S_norm)*0.25)] ** 2))
        logits = W.mean(axis=0)
        p = np.clip(expit(logits), 1e-8, 1 - 1e-8)
        entropy = -np.sum(p * np.log(p))
        return float(np.clip(1.0 - (entropy / (W.shape[1] * np.log(2) + 1e-9)), 0.0, 1.0))

    def analyze_layer(self, W: np.ndarray) -> dict:
        S = OptimizedSVD.randomized_svd(W)
        S_norm = S / (np.linalg.norm(S) + 1e-9)
        r_eff = float((np.sum(S)**2) / (np.sum(S**2) + 1e-12))
        
        Cs = 1.0 / np.log(1 + r_eff + 1e-9)
        H = -np.sum((S_norm**2) * np.log(S_norm**2 + 1e-12))
        Ci = np.clip(1.0 - (H / np.log(r_eff + 1.1)), 0, 1)
        
        cum_energy = np.cumsum(S_norm**2)
        Cr = float(np.sum(S_norm[cum_energy > 0.8]**2) / (np.sum(S_norm[(cum_energy >= 0.3) & (cum_energy <= 0.8)]**2) + 1e-9))
        Cd = float(np.mean(np.abs(W) < 1e-6))
        
        C = np.array([Cs, Ci, Cr, Cd])
        bc = self.bayesian_coherence(W)
        C_global = 0.85 * float(np.dot(self.w, C)) + 0.15 * bc
        
        logits = [np.dot(self.policy_params.get(cls, self.policy_params["FP16/INT8"]), C) 
                  for cls in ["FP32", "FP16/INT8", "INT4", "NVFP4_SAFE"]]
        probs = softmax(logits)
        
        return {"C_global": C_global, "bayesian_coherence": bc, "probs": probs.tolist()}

# =========================================================
# COMPILER
# =========================================================
class D2Compiler:
    def __init__(self):
        self.qdf = QDFV5Engine()

    def ssm_policy_override(self, name, alpha, current_policy):
        if "ssm_conv1d" in name:
            return "INT8_SAFE" if alpha < 2.5 else "FP16_REQUIRED"
        elif "ssm_alpha" in name or "ssm_beta" in name:
            return "NVFP4_SAFE"
        return current_policy

    def compile(self):
        print("🏗️ Running D2 Unified Compilation (Fast)...")
        with open("precision_map.json") as f: prec = json.load(f)
        with open("alpha_w_report.json") as f: alpha_rep = json.load(f)
        with open("final_sm120_graph_plan.json") as f: plan = json.load(f)
        
        unified_plan = {"nodes": {}, "global_stats": []}
        
        for name in prec.keys():
            alpha = alpha_rep.get(name, {}).get("alpha_w", 1.0)
            # Override SSM
            precision = self.ssm_policy_override(name, alpha, prec[name]["precision"])
            
            # Utiliser alpha pour estimer la cohérence plutôt que recalculer SVD
            coherence = np.clip(1.0 / alpha, 0.5, 0.9)
            
            unified_plan["nodes"][name] = {
                "precision": precision,
                "kernel": "NVFP4" if alpha > 1.5 else "INT8",
                "coherence": round(float(coherence), 3)
            }
            
        with open("d2_unified_plan.json", "w") as f:
            json.dump(unified_plan, f, indent=2)
        print("✅ D2 Plan generated: d2_unified_plan.json")

if __name__ == "__main__":
    D2Compiler().compile()
