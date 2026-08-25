#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 REGISTRY — unified tensor registry for the D2 ecosystem
=============================================================

The registry is the central database that connects all D2 profilers.
Each profiler writes its measurements to the registry, and the
precision optimizer reads from it to make decisions.

Architecture:

    d2_tensor_profiler  ──► registry.add_tensor_noise(...)
    d2_slow_layer_prof  ──► registry.add_tensor_latency(...)
    d2_hw_probe         ──► registry.set_hardware(...)
    d2_forward_test     ──► registry.add_tensor_quality(...)
    d2_cost_model       ──► registry.add_tensor_cost(...)
    d2_tensor_optimizer ◄── registry.get_tensors()
    d2_build_eco_v2     ◄── registry.get_recommendations()

Usage:
    from d2_registry import D2Registry
    
    reg = D2Registry()
    reg.load("d2_ecosystem/registry.json")
    reg.add_tensor_noise("blk.23.ffn_down.weight", snr_db=45.2, rel_err=0.003)
    reg.save("d2_ecosystem/registry.json")
    
    # Query
    slow = reg.get_slow_tensors(threshold_ms=1.0)
    sensitive = reg.get_sensitive_tensors(max_snr=30)
    summary = reg.layer_summary()
"""

import json
import os
import time
from typing import List, Dict, Optional, Tuple

from d2_schema import (
    TensorRecord, HardwareRecord, RunRecord, LayerRecord,
    ModelManifest, QWEN38_27B, generate_run_id, save_json, load_json
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "d2_ecosystem", "registry.json")


class D2Registry:
    """Unified tensor registry for the D2 ecosystem."""
    
    def __init__(self, model: ModelManifest = None):
        self.model = model or QWEN38_27B
        self.hardware: Optional[HardwareRecord] = None
        self.runs: List[RunRecord] = []
        self.tensors: Dict[str, TensorRecord] = {}
        self.layers: Dict[int, LayerRecord] = {}
        self.run_id = generate_run_id()
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    
    # =========================================================================
    # TENSOR OPERATIONS
    # =========================================================================
    
    def get_tensor(self, tensor_id: str) -> TensorRecord:
        """Get or create a tensor record."""
        if tensor_id not in self.tensors:
            # Parse tensor_id to extract layer and component
            layer_id, component = self._parse_tensor_id(tensor_id)
            self.tensors[tensor_id] = TensorRecord(
                tensor_id=tensor_id,
                layer_id=layer_id,
                component=component,
                original_name=tensor_id,
            )
        return self.tensors[tensor_id]
    
    def add_tensor_noise(self, tensor_id: str, snr_db: float = 0, 
                         rel_err: float = 0, cosine_sim: float = 0,
                         channel_err: float = 0):
        """Add noise/error metrics from d2_tensor_profiler.py."""
        t = self.get_tensor(tensor_id)
        t.snr_db = snr_db
        t.rel_err = rel_err
        t.cosine_sim = cosine_sim
        t.channel_err = channel_err
    
    def add_tensor_latency(self, tensor_id: str, latency_us: float = 0,
                           decode_ms: float = 0, bandwidth_gbs: float = 0):
        """Add latency metrics from d2_slow_layer_profiler.py."""
        t = self.get_tensor(tensor_id)
        t.latency_us = latency_us
        t.decode_ms = decode_ms
        t.bandwidth_gbs = bandwidth_gbs
    
    def add_tensor_shape(self, tensor_id: str, shape: list,
                         original_bytes: int = 0, quantized_bytes: int = 0,
                         original_precision: str = "FP8",
                         quant_precision: str = "Q2_K"):
        """Add shape and size info from d2_static_profile.py."""
        t = self.get_tensor(tensor_id)
        t.shape = shape
        t.n_elements = 1
        for s in shape:
            t.n_elements *= s
        t.original_bytes = original_bytes
        t.quantized_bytes = quantized_bytes
        t.original_precision = original_precision
        t.quant_precision = quant_precision
        if original_bytes > 0:
            t.bpw = (quantized_bytes * 8) / t.n_elements
    
    def add_tensor_quality(self, tensor_id: str, ppl_delta: float = 0,
                           importance: float = 0):
        """Add quality metrics from d2_forward_test.py."""
        t = self.get_tensor(tensor_id)
        t.importance = importance
    
    def add_tensor_memory(self, tensor_id: str, vram_bytes: int = 0,
                          offload: str = "CUDA0"):
        """Add memory info from d2_memory_model.py."""
        t = self.get_tensor(tensor_id)
        t.vram_bytes = vram_bytes
        t.offload = offload
    
    # =========================================================================
    # LAYER OPERATIONS
    # =========================================================================
    
    def layer_summary(self) -> List[LayerRecord]:
        """Aggregate tensor records into layer records."""
        layers = {}
        
        for t in self.tensors.values():
            lid = t.layer_id
            if lid not in layers:
                layer_type = "attention" if t.component.startswith("attn") else "gdn"
                if t.component.startswith("ssm"):
                    layer_type = "gdn"
                layers[lid] = LayerRecord(layer_id=lid, layer_type=layer_type)
            
            l = layers[lid]
            l.n_tensors += 1
            l.total_bytes += t.original_bytes
            l.quantized_bytes += t.quantized_bytes
            l.vram_bytes += t.vram_bytes
            l.decode_ms += t.decode_ms
            l.total_latency_us += t.latency_us
            
            if t.snr_db > 0:
                l.mean_snr_db += t.snr_db
            if t.rel_err > l.max_rel_err:
                l.max_rel_err = t.rel_err
            l.mean_rel_err += t.rel_err
        
        # Normalize averages
        for l in layers.values():
            if l.n_tensors > 0:
                l.mean_snr_db /= l.n_tensors
                l.mean_rel_err /= l.n_tensors
        
        self.layers = layers
        return [layers[k] for k in sorted(layers.keys())]
    
    def get_slow_tensors(self, threshold_ms: float = 1.0) -> List[TensorRecord]:
        """Get tensors with decode time above threshold."""
        return sorted(
            [t for t in self.tensors.values() if t.decode_ms > threshold_ms],
            key=lambda t: t.decode_ms, reverse=True
        )
    
    def get_sensitive_tensors(self, max_snr: float = 30.0) -> List[TensorRecord]:
        """Get tensors with low SNR (sensitive to quantization)."""
        return sorted(
            [t for t in self.tensors.values() if 0 < t.snr_db < max_snr],
            key=lambda t: t.snr_db
        )
    
    def get_large_tensors(self, min_bytes: int = 10_000_000) -> List[TensorRecord]:
        """Get tensors above minimum size."""
        return sorted(
            [t for t in self.tensors.values() if t.original_bytes >= min_bytes],
            key=lambda t: t.original_bytes, reverse=True
        )
    
    # =========================================================================
    # HARDWARE OPERATIONS
    # =========================================================================
    
    def set_hardware(self, hw: HardwareRecord):
        """Set hardware record from d2_hw_probe.py."""
        self.hardware = hw
    
    def load_hardware_from_json(self, path: str):
        """Load hardware fingerprint from JSON."""
        if os.path.exists(path):
            data = load_json(path)
            if isinstance(data, dict):
                self.hardware = HardwareRecord.from_dict(data)
    
    # =========================================================================
    # RUN OPERATIONS
    # =========================================================================
    
    def add_run(self, run: RunRecord):
        """Add a benchmark run record."""
        self.runs.append(run)
    
    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    
    def get_recommendations(self) -> Dict[str, str]:
        """Generate tensor-type recommendations based on registry data."""
        recs = {}
        
        for t in self.tensors.values():
            if t.quant_precision == "F32":
                continue  # Don't recommend changing F32 tensors
            
            # Default: keep current
            rec = t.quant_precision
            
            # Rule 1: Low SNR → upgrade precision
            if 0 < t.snr_db < 25:
                if t.quant_precision in ("Q2_K",):
                    rec = "Q3_K"
            
            # Rule 2: High latency → consider downgrade (but only if SNR is good)
            if t.decode_ms > 2.0 and t.snr_db > 40:
                if t.quant_precision in ("Q8_0", "Q5_K", "Q6_K"):
                    rec = "Q4_K"
            
            # Rule 3: Very large + memory-bound → keep small
            if t.original_bytes > 50_000_000 and t.latency_us > 0:
                # Large tensor: don't upgrade unless critical
                pass
            
            if rec != t.quant_precision:
                recs[t.tensor_id] = rec
        
        return recs
    
    # =========================================================================
    # IMPORT FROM EXISTING SCRIPTS
    # =========================================================================
    
    def import_tensor_profile(self, path: str):
        """Import from d2_tensor_profile.json / d2_tensor_profile_full.json."""
        if not os.path.exists(path):
            return
        
        data = load_json(path)
        tensors = data.get("tensors", data) if isinstance(data, dict) else data
        
        for t in tensors:
            name = t.get("name", "")
            if not name:
                continue
            
            # Convert HF name to tensor_id
            tensor_id = self._hf_to_tensor_id(name)
            
            rec = self.get_tensor(tensor_id)
            rec.original_name = name
            rec.shape = t.get("shape", [])
            rec.decode_ms = t.get("decode_ms", 0)
            rec.rms = t.get("rms", 0)
            rec.max_abs = t.get("max", 0)
            rec.outlier_rate = t.get("outlier_rate", 0)
            rec.sparsity = t.get("sparsity", 0)
            rec.entropy_bits = t.get("entropy_bits", 0)
            
            # Precision info
            prec = t.get("precision", {})
            if isinstance(prec, dict):
                for fmt, metrics in prec.items():
                    if isinstance(metrics, dict):
                        rec.snr_db = metrics.get("snr_db", rec.snr_db)
                        rec.rel_err = metrics.get("rel_err", rec.rel_err)
    
    def import_layer_behavior(self, path: str):
        """Import from d2_layer_behavior_report.json."""
        if not os.path.exists(path):
            return
        
        data = load_json(path)
        layers = data.get("layers", [])
        
        for l in layers:
            lid = l.get("layer_id", l.get("layer", -1))
            if lid < 0:
                continue
            
            if lid not in self.layers:
                self.layers[lid] = LayerRecord(layer_id=lid)
            
            rec = self.layers[lid]
            rec.mean_snr_db = l.get("snr", rec.mean_snr_db)
            rec.decode_ms = l.get("layer_time_ms", rec.decode_ms)
    
    def import_hw_fingerprint(self, path: str):
        """Import from hardware_fingerprint.json."""
        self.load_hardware_from_json(path)
    
    # =========================================================================
    # SAVE / LOAD
    # =========================================================================
    
    def save(self, path: str = None):
        """Save registry to JSON."""
        path = path or DEFAULT_DB
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            "model": self.model.to_dict() if self.model else None,
            "hardware": self.hardware.to_dict() if self.hardware else None,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "n_tensors": len(self.tensors),
            "n_layers": len(self.layers),
            "n_runs": len(self.runs),
            "tensors": {k: v.to_dict() for k, v in self.tensors.items()},
            "layers": {str(k): v.to_dict() for k, v in self.layers.items()},
            "runs": [r.to_dict() for r in self.runs],
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str = None):
        """Load registry from JSON."""
        path = path or DEFAULT_DB
        if not os.path.exists(path):
            return
        
        data = load_json(path)
        
        if data.get("model"):
            self.model = ModelManifest.from_dict(data["model"])
        if data.get("hardware"):
            self.hardware = HardwareRecord.from_dict(data["hardware"])
        
        self.run_id = data.get("run_id", self.run_id)
        self.timestamp = data.get("timestamp", self.timestamp)
        
        for k, v in data.get("tensors", {}).items():
            self.tensors[k] = TensorRecord.from_dict(v)
        
        for k, v in data.get("layers", {}).items():
            self.layers[int(k)] = LayerRecord.from_dict(v)
        
        for r in data.get("runs", []):
            self.runs.append(RunRecord.from_dict(r))
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _parse_tensor_id(self, tensor_id: str) -> Tuple[int, str]:
        """Parse tensor_id to extract layer_id and component."""
        parts = tensor_id.split(".")
        if len(parts) >= 3 and parts[0] == "blk":
            try:
                layer_id = int(parts[1])
            except ValueError:
                layer_id = -1
            component = ".".join(parts[2:])
            return layer_id, component
        return -1, tensor_id
    
    def _hf_to_tensor_id(self, hf_name: str) -> str:
        """Convert HuggingFace tensor name to canonical tensor_id."""
        # model.language_model.layers.23.mlp.down_proj.weight
        # → blk.23.ffn_down.weight
        parts = hf_name.split(".")
        
        if "layers" in parts:
            idx = parts.index("layers")
            if idx + 1 < len(parts):
                layer_num = parts[idx + 1]
                # Find the component
                remaining = parts[idx + 2:]
                
                # Map HF names to canonical names
                mapping = {
                    "mlp.down_proj": "ffn_down",
                    "mlp.gate_proj": "ffn_gate",
                    "mlp.up_proj": "ffn_up",
                    "self_attn.q_proj": "attn_q",
                    "self_attn.k_proj": "attn_k",
                    "self_attn.v_proj": "attn_v",
                    "self_attn.o_proj": "attn_o",
                    "self_attn.qkv_proj": "attn_qkv",
                    "self_attn.gate": "attn_gate",
                    "linear_attn.in_proj_a": "attn_qkv",
                    "linear_attn.ssm_out": "ssm_out",
                    "input_layernorm": "input_layernorm",
                    "post_attention_layernorm": "post_attention_layernorm",
                }
                
                # Try to match
                for hf_prefix, canon_name in mapping.items():
                    if ".".join(remaining[:len(hf_prefix.split("."))]) == hf_prefix:
                        suffix = remaining[len(hf_prefix.split(".")):]
                        return f"blk.{layer_num}.{canon_name}{'.' + '.'.join(suffix) if suffix else ''}"
        
        return hf_name
    
    def summary(self) -> str:
        """Print a summary of the registry."""
        lines = [
            f"D2 Registry Summary",
            f"  Model: {self.model.model_id if self.model else '?'}",
            f"  Hardware: {self.hardware.gpu_name if self.hardware else '?'}",
            f"  Tensors: {len(self.tensors)}",
            f"  Layers: {len(self.layers)}",
            f"  Runs: {len(self.runs)}",
            f"  Run ID: {self.run_id}",
        ]
        
        if self.tensors:
            total_orig = sum(t.original_bytes for t in self.tensors.values())
            total_quant = sum(t.quantized_bytes for t in self.tensors.values())
            lines.append(f"  Total original: {total_orig / 1e9:.2f} GB")
            lines.append(f"  Total quantized: {total_quant / 1e9:.2f} GB")
        
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo: load existing D2 data into registry."""
    import sys
    
    reg = D2Registry()
    
    # Try to import existing data
    print("Importing existing D2 data...")
    
    reg.import_tensor_profile(os.path.join(HERE, "d2_tensor_profile_full.json"))
    print(f"  Tensors: {len(reg.tensors)}")
    
    reg.import_layer_behavior(os.path.join(HERE, "d2_layer_behavior_report.json"))
    print(f"  Layers: {len(reg.layers)}")
    
    reg.import_hw_fingerprint(os.path.join(HERE, "hardware_fingerprint.json"))
    print(f"  Hardware: {reg.hardware.gpu_name if reg.hardware else 'not loaded'}")
    
    # Generate summary
    print()
    print(reg.summary())
    
    # Generate recommendations
    recs = reg.get_recommendations()
    if recs:
        print(f"\nRecommendations: {len(recs)} tensors to change")
        for tid, rec in list(recs.items())[:10]:
            t = reg.tensors[tid]
            print(f"  {tid}: {t.quant_precision} -> {rec} (SNR={t.snr_db:.1f}, latency={t.decode_ms:.2f}ms)")
    
    # Save
    reg.save()
    print(f"\nSaved to {DEFAULT_DB}")


if __name__ == "__main__":
    main()
