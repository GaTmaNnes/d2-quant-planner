#!/usr/bin/env python3
"""
D2 Quantization Planner — Real Model Testing Suite

Test D2 on actual HuggingFace models:
- GPT-2 (small, 124M parameters)
- TinyLlama (1.1B)
- Llama 2 7B (if available)
- Qwen models

Measures:
- Spectral exponent (α_w) per layer
- C-vector components
- Recommended quantization dtype
- VRAM usage estimation
- Fragmentation metrics
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from typing import Dict, List, Tuple
from scipy.linalg import svd

# ─────────────────────────────────────────────────────────────────────────────
# SPECTRAL ANALYSIS (from spectral_theory.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_alpha_w(W: np.ndarray) -> float:
    """Compute spectral exponent α_w from weight matrix."""
    try:
        U, s, Vt = svd(W, full_matrices=False)
    except:
        return 2.0  # Default on error
    
    s = s[s > 0.01 * s[0]]
    if len(s) < 2:
        return 2.0
    
    log_i = np.log(np.arange(1, len(s) + 1))
    log_s = np.log(s)
    
    try:
        coeffs = np.polyfit(log_i, log_s, 1)
        return float(max(-2.0 * coeffs[0], 1.0))
    except:
        return 2.0


def compute_stable_rank(W: np.ndarray) -> float:
    """Compute stable rank r_s = ||W||_F² / ||W||_2²."""
    try:
        U, s, Vt = svd(W, full_matrices=False)
        return float(np.sum(s**2) / (s[0]**2 + 1e-10))
    except:
        return 1.0


def compute_spectral_entropy(W: np.ndarray) -> float:
    """Compute spectral entropy H = -Σ p_i log(p_i)."""
    try:
        U, s, Vt = svd(W, full_matrices=False)
        s = s[s > 0.01 * s[0]]
        p = s**2 / np.sum(s**2)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))
    except:
        return 1.0


def compute_spectral_radius(W: np.ndarray) -> float:
    """Compute spectral radius ρ = largest singular value."""
    try:
        U, s, Vt = svd(W, full_matrices=False)
        return float(s[0] if len(s) > 0 else 1.0)
    except:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# QUANTIZATION POLICY (Corrected)
# ─────────────────────────────────────────────────────────────────────────────

def quantization_policy(alpha_w: float, density: float) -> str:
    """
    Corrected quantization mapping.
    
    instability = α_w (1 + ρ_d)
    
    LOW instability → NVFP4 (compressible)
    HIGH instability → FP16 (sensitive)
    """
    instability = alpha_w * (1.0 + density)
    
    if instability < 1.2:
        return "NVFP4"
    elif instability < 1.6:
        return "INT8"
    elif instability < 2.0:
        return "FP8"
    else:
        return "FP16"


# ─────────────────────────────────────────────────────────────────────────────
# DTYPE COSTS (VRAM + Time)
# ─────────────────────────────────────────────────────────────────────────────

DTYPE_INFO = {
    "FP16": {"bytes": 2.0, "tps_gain": 1.0},
    "FP8": {"bytes": 1.0, "tps_gain": 1.5},
    "INT8": {"bytes": 1.0, "tps_gain": 1.35},
    "NVFP4": {"bytes": 0.5, "tps_gain": 1.75},
}


def estimate_layer_vram(shape: Tuple, dtype: str) -> float:
    """Estimate VRAM usage in GB."""
    numel = np.prod(shape)
    bytes_per_param = DTYPE_INFO[dtype]["bytes"]
    return (numel * bytes_per_param) / (1024**3)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TESTING FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

class D2ModelAnalyzer:
    """Analyze a model for quantization planning."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize analyzer.
        
        Args:
            model_name: HuggingFace model ID
            device: CPU or CUDA
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.layers_analysis = []
    
    def load_model(self):
        """Load model from HuggingFace."""
        print(f"📥 Loading {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            print(f"✓ Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params")
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            raise
    
    def analyze_layers(self) -> List[Dict]:
        """
        Analyze all weight layers.
        
        Returns:
            List of layer analysis dictionaries
        """
        print("🔍 Analyzing layers...")
        self.layers_analysis = []
        
        total_params = 0
        layer_count = 0
        
        for name, param in self.model.named_parameters():
            # Skip biases and normalization
            if "bias" in name or "norm" in name or "ln_" in name:
                continue
            
            # Only analyze weight matrices
            if param.dim() < 2:
                continue
            
            layer_count += 1
            W = param.data.detach().cpu().numpy()
            
            # Reshape if needed
            if W.ndim > 2:
                W = W.reshape(W.shape[0], -1)
            
            # Skip tiny layers
            if W.shape[0] < 8 or W.shape[1] < 8:
                continue
            
            numel = np.prod(W.shape)
            total_params += numel
            
            # Compute spectral properties
            alpha_w = compute_alpha_w(W)
            stable_rank = compute_stable_rank(W)
            entropy = compute_spectral_entropy(W)
            rho = compute_spectral_radius(W)
            
            # Density (sparsity)
            density = float(np.count_nonzero(W) / W.size) if W.size > 0 else 0.0
            
            # Quantization decision
            dtype = quantization_policy(alpha_w, density)
            
            # VRAM estimation
            vram_fp16 = estimate_layer_vram(W.shape, "FP16")
            vram_dtype = estimate_layer_vram(W.shape, dtype)
            
            layer_info = {
                "name": name,
                "shape": W.shape,
                "params": int(numel),
                "alpha_w": float(alpha_w),
                "stable_rank": float(stable_rank),
                "entropy": float(entropy),
                "rho": float(rho),
                "density": float(density),
                "dtype_recommended": dtype,
                "vram_fp16_mb": float(vram_fp16 * 1024),
                "vram_dtype_mb": float(vram_dtype * 1024),
                "vram_saved_pct": float(100 * (1 - vram_dtype / vram_fp16)) if vram_fp16 > 0 else 0.0,
            }
            
            self.layers_analysis.append(layer_info)
        
        print(f"✓ Analyzed {layer_count} layers, {total_params / 1e6:.1f}M total parameters")
        return self.layers_analysis
    
    def compute_statistics(self) -> Dict:
        """Compute aggregate statistics."""
        if not self.layers_analysis:
            return {}
        
        alpha_ws = [l["alpha_w"] for l in self.layers_analysis]
        entropies = [l["entropy"] for l in self.layers_analysis]
        vrays = [l["vram_dtype_mb"] for l in self.layers_analysis]
        
        # Dtype distribution
        dtype_dist = {}
        for layer in self.layers_analysis:
            dtype = layer["dtype_recommended"]
            dtype_dist[dtype] = dtype_dist.get(dtype, 0) + 1
        
        # Fragmentation (consecutive layers with different dtypes)
        fragmentation = 0
        for i in range(len(self.layers_analysis) - 1):
            if self.layers_analysis[i]["dtype_recommended"] != self.layers_analysis[i+1]["dtype_recommended"]:
                fragmentation += 1
        
        return {
            "num_layers": len(self.layers_analysis),
            "alpha_w_mean": float(np.mean(alpha_ws)),
            "alpha_w_std": float(np.std(alpha_ws)),
            "alpha_w_min": float(np.min(alpha_ws)),
            "alpha_w_max": float(np.max(alpha_ws)),
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
            "dtype_distribution": dtype_dist,
            "total_vram_mb": float(np.sum(vrays)),
            "fragmentation_score": float(fragmentation / max(1, len(self.layers_analysis) - 1)),
            "fragmentation_events": int(fragmentation),
        }
    
    def generate_report(self, output_dir: str = "results"):
        """Generate analysis report."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Full analysis
        report = {
            "model": self.model_name,
            "layers": self.layers_analysis,
            "statistics": self.compute_statistics(),
        }
        
        output_file = Path(output_dir) / f"{self.model_name.replace('/', '_')}_analysis.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report saved: {output_file}")
        return report
    
    def print_summary(self):
        """Print human-readable summary."""
        stats = self.compute_statistics()
        
        print("\n" + "=" * 70)
        print(f"D2 ANALYSIS SUMMARY — {self.model_name}")
        print("=" * 70)
        print(f"Layers analyzed: {stats['num_layers']}")
        print(f"α_w range: {stats['alpha_w_min']:.2f} — {stats['alpha_w_max']:.2f} (mean: {stats['alpha_w_mean']:.2f})")
        print(f"Entropy: {stats['entropy_mean']:.2f} ± {stats['entropy_std']:.2f}")
        print(f"Total VRAM (quantized): {stats['total_vram_mb'] / 1024:.2f} GB")
        print(f"Fragmentation: {stats['fragmentation_score']:.2%} ({stats['fragmentation_events']} events)")
        print("\nQuantization distribution:")
        for dtype, count in sorted(stats['dtype_distribution'].items()):
            pct = 100 * count / stats['num_layers']
            print(f"  {dtype:6} : {count:3d} layers ({pct:5.1f}%)")
        print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────

MODELS_TO_TEST = [
    "gpt2",  # 124M - Fast test
    "distilgpt2",  # 82M - Very fast
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B - Good balance
    # "meta-llama/Llama-2-7b-hf",  # 7B - Large (requires auth)
    # "Qwen/Qwen-1.8B",  # 1.8B - Chinese LLM
]


def run_test_suite(models: List[str] = None, device: str = "cpu"):
    """Run analysis on multiple models."""
    if models is None:
        models = MODELS_TO_TEST
    
    results = {}
    
    for model_name in models:
        print(f"\n\n{'🔬 ' * 20}")
        print(f"Testing: {model_name}")
        print(f"{'🔬 ' * 20}\n")
        
        try:
            analyzer = D2ModelAnalyzer(model_name, device=device)
            analyzer.load_model()
            analyzer.analyze_layers()
            analyzer.print_summary()
            results[model_name] = analyzer.generate_report()
        
        except Exception as e:
            print(f"❌ Error analyzing {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    # Comparative summary
    print("\n\n" + "=" * 70)
    print("COMPARATIVE SUMMARY")
    print("=" * 70)
    
    for model_name, report in results.items():
        if "error" in report:
            print(f"{model_name}: ❌ FAILED")
        else:
            stats = report["statistics"]
            print(f"{model_name}:")
            print(f"  α_w: {stats['alpha_w_mean']:.2f} ± {stats['alpha_w_std']:.2f}")
            print(f"  VRAM: {stats['total_vram_mb'] / 1024:.2f} GB")
            print(f"  Fragmentation: {stats['fragmentation_score']:.2%}")
    
    print("=" * 70)
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="D2 Model Testing Suite")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--all", action="store_true", help="Test all default models")
    
    args = parser.parse_args()
    
    if args.model:
        models = [args.model]
    elif args.all:
        models = MODELS_TO_TEST
    else:
        # Default: test gpt2 only (fast)
        models = ["gpt2", "distilgpt2"]
    
    run_test_suite(models, device=args.device)
