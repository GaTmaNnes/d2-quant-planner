import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
import time
from scipy.linalg import svd

# Ajouter le chemin vers la librairie gguf-py
sys.path.append(str(Path("beellama.cpp-main/gguf-py").absolute()))
from gguf import GGUFReader

class AlphaScannerSM120:
    """
    Scanner Spectral Alpha_w (NVIDIA Optimized).
    Détecte la concentration spectrale pour guider la quantification Blackwell.
    """

    @staticmethod
    def alpha_w(W, cutoff=0.01):
        """Estimateur Alpha_w pour les couches de Transformer."""
        try:
            if isinstance(W, torch.Tensor):
                W = W.float().cpu().numpy()
            
            # Échantillonnage pour les gros tenseurs (vitesse)
            if W.shape[0] * W.shape[1] > 10**7:
                indices = np.random.choice(W.shape[0], min(W.shape[0], 2048), replace=False)
                W = W[indices, :]

            _, S, _ = svd(W, full_matrices=False)
            S = S[S > cutoff * S[0]]
            
            if len(S) < 2:
                return 1.0

            x = np.log(np.arange(1, len(S) + 1))
            y = np.log(S)
            slope = np.polyfit(x, y, 1)[0]
            return max(-2.0 * slope, 1.0)
        except:
            return 1.5

    @staticmethod
    def nvidia_llm_policy(alpha, layer_type="linear"):
        """Politique de décision Blackwell sm_120."""
        # Biais de sensibilité pour les couches d'attention (Critical logic)
        sensitivity_bias = 0.15 if "attn" in layer_type else 0.0
        score = alpha - sensitivity_bias

        if score >= 2.1:
            return "NVFP4_SAFE" # Blackwell Native
        elif score >= 1.8:
            return "NVFP4_SAFE" # Aggressive but safe
        elif score >= 1.5:
            return "INT8_SAFE"
        elif score >= 1.2:
            return "INT8_SAFE"
        else:
            return "FP16_REQUIRED"

def run_alpha_scan(model_path, output_json="alpha_w_report.json"):
    print(f"🔍 Scan Alpha_w SM120 sur {model_path}")
    reader = GGUFReader(model_path)
    results = {}
    
    tensors = [t for t in reader.tensors if "weight" in t.name.lower() and "norm" not in t.name.lower()]
    total = len(tensors)

    for i, tensor in enumerate(tensors):
        t0 = time.time()
        alpha = AlphaScannerSM120.alpha_w(tensor.data)
        policy = AlphaScannerSM120.nvidia_llm_policy(alpha, tensor.name.lower())
        dt = time.time() - t0
        
        results[tensor.name] = {
            "alpha_w": round(float(alpha), 3),
            "precision": policy,
            "layer_type": "attention" if "attn" in tensor.name.lower() else "mlp"
        }
        print(f"[{i+1}/{total}] {tensor.name:40} | α={alpha:.2f} | {policy} | {dt:.1f}s")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Scan Alpha terminé. Rapport sauvegardé dans {output_json}")

if __name__ == "__main__":
    run_alpha_scan("models/model.gguf")
