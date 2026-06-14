import json
import numpy as np
from scipy.linalg import svd
from scipy.special import softmax, expit
from dataclasses import dataclass
from typing import Dict, Optional

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
        if W.size > 30_000_000:
            _, S, _ = svd(W, full_matrices=False, compute_uv=False)
            S_norm = S / (np.linalg.norm(S) + 1e-9)
            return float(np.sum(S_norm[:int(len(S_norm)*0.25)] ** 2))
        
        logits = W.mean(axis=0)
        sigmoid_out = expit(logits)
        p = np.clip(sigmoid_out, 1e-8, 1 - 1e-8)
        entropy = -np.sum(p * np.log(p))
        coherence = 1.0 - (entropy / (W.shape[1] * np.log(2) + 1e-9))
        return float(np.clip(coherence, 0.0, 1.0))

    def analyze_layer(self, W: np.ndarray, prev_C=None) -> Dict:
        # Échantillonnage sécurisé
        if W.shape[0] * W.shape[1] > 12_000_000:
            row_idx = np.random.choice(W.shape[0], min(2048, W.shape[0]), replace=False)
            sample = W[row_idx, :]
        else:
            sample = W

        _, S, _ = svd(sample, full_matrices=False)
        S_norm = S / (np.linalg.norm(S) + 1e-9)
        r_eff = float((np.sum(S)**2) / (np.sum(S**2) + 1e-12))

        Cs = 1.0 / np.log(1 + r_eff + 1e-9)
        p = S_norm**2 / (np.sum(S_norm**2) + 1e-12)
        H = -np.sum(p * np.log(p + 1e-12))
        Ci = np.clip(1.0 - (H / np.log(r_eff + 1.1)), 0, 1)

        cum_energy = np.cumsum(S_norm**2)
        bulk = (cum_energy >= 0.3) & (cum_energy <= 0.8)
        Cr = float(np.sum(S_norm[~bulk]**2) / (np.sum(S_norm[bulk]**2) + 1e-9))
        Cd = float(np.mean(np.abs(W) < 1e-6))

        C = np.array([Cs, Ci, Cr, Cd])
        if prev_C is not None:
            C = 0.7 * C + 0.3 * np.array(prev_C)

        C_global = float(np.dot(self.w, C))
        bc_score = self.bayesian_coherence(W)
        C_global = 0.85 * C_global + 0.15 * bc_score

        logits = [np.dot(self.policy_params.get(cls, self.policy_params["FP16/INT8"]), C) 
                 for cls in ["FP32", "FP16/INT8", "INT4", "NVFP4_SAFE"]]
        probs = softmax(logits)

        return {
            "C_vector": C.tolist(),
            "C_global": C_global,
            "bayesian_coherence": bc_score,
            "probs": {
                "FP32": float(probs[0]),
                "FP16/INT8": float(probs[1]),
                "INT4": float(probs[2]),
                "NVFP4_SAFE": float(probs[3])
            },
            "metrics": {"r_eff": r_eff, "entropy": float(H)}
        }


@dataclass
class TensorNode:
    name: str
    op_type: str
    shape: list
    alpha_w: float
    qdf: Optional[Dict] = None
    recommended_precision: str = "FP16_REQUIRED"
    op_class: str = "OTHER"


class D2Compiler:
    def __init__(self):
        self.qdf_engine = QDFV5Engine()

    def load_reports(self):
        base = "/home/workdir/attachments"
        with open(f"{base}/precision_map.json") as f:
            self.precision_map = json.load(f)
        with open(f"{base}/alpha_w_report.json") as f:
            self.alpha_report = json.load(f)
        with open(f"{base}/final_sm120_graph_plan.json") as f:
            self.graph_plan = json.load(f)

    def build_nodes(self):
        nodes = {}
        prev_C = None
        for name, data in self.alpha_report.items():
            node = TensorNode(
                name=name,
                op_type=data.get("layer_type", "UNKNOWN"),
                shape=self.graph_plan.get(name, {}).get("shape", [0, 0]),
                alpha_w=data.get("alpha_w", 1.0),
                recommended_precision=self.precision_map.get(name, {}).get("precision", "FP16_REQUIRED"),
                op_class=self.graph_plan.get(name, {}).get("op_class", "OTHER")
            )
            
            dummy_W = np.random.randn(*node.shape).astype(np.float32) * 0.01
            node.qdf = self.qdf_engine.analyze_layer(dummy_W, prev_C=prev_C)
            prev_C = node.qdf["C_vector"]
            
            nodes[name] = node
        return nodes

    def kernel_mapping(self, node: TensorNode):
        bc = node.qdf.get("bayesian_coherence", 0.0)
        if bc > 0.65 and "attn" in node.name:
            return "ATTENTION"
        elif "ssm_conv" in node.name:
            return "SSM_CONV"
        return "GEMM"

    def generate_unified_plan(self):
        nodes = self.build_nodes()
        plan = {"nodes": {}, "global_stats": {}}
        
        bc_scores = []
        for name, node in nodes.items():
            kernel = self.kernel_mapping(node)
            plan["nodes"][name] = {
                "precision": node.recommended_precision,
                "alpha_w": round(node.alpha_w, 3),
                "bayesian_coherence": round(node.qdf["bayesian_coherence"], 3),
                "C_global": round(node.qdf["C_global"], 3),
                "recommended_kernel": kernel,
                "op_class": node.op_class
            }
            bc_scores.append(node.qdf["bayesian_coherence"])
        
        plan["global_stats"] = {
            "mean_bayesian_coherence": round(float(np.mean(bc_scores)), 3),
            "mean_alpha_w": round(float(np.mean([n.alpha_w for n in nodes.values()])), 3),
            "nvfp4_ratio": round(sum(1 for n in plan["nodes"].values() if "NVFP4" in n["precision"]) / len(plan["nodes"]), 3)
        }
        return plan


if __name__ == "__main__":
    print("🚀 D2 + QDF V5 + Bayesian Compiler (version mémoire corrigée)")
    compiler = D2Compiler()
    compiler.load_reports()
    unified_plan = compiler.generate_unified_plan()
    
    print("\n=== STATISTIQUES GLOBALES ===")
    print(json.dumps(unified_plan["global_stats"], indent=2))
    
    with open("/home/workdir/d2_unified_plan.json", "w") as f:
        json.dump(unified_plan, f, indent=2)
    
    print(f"\n✅ Plan sauvegardé dans /home/workdir/d2_unified_plan.json")