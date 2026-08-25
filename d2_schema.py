#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 SCHEMA — canonical data classes for the D2 ecosystem
=========================================================

Defines the single source of truth for all D2 data structures.
Every profiler, optimizer, and reporter should import from here.

Usage:
    from d2_schema import TensorRecord, HardwareRecord, RunRecord, LayerRecord
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# TENSOR RECORD — per-tensor data
# =============================================================================

@dataclass
class TensorRecord:
    """Canonical record for a single tensor in the model."""
    
    # Identity
    tensor_id: str              # e.g., "blk.23.ffn_down.weight"
    layer_id: int               # e.g., 23
    component: str              # e.g., "ffn_down"
    original_name: str          # e.g., "model.language_model.layers.23.mlp.down_proj.weight"
    
    # Shape and size
    shape: List[int] = field(default_factory=list)
    n_elements: int = 0
    original_bytes: int = 0     # bytes in FP8/F16 original
    
    # Quantization
    original_precision: str = "FP8"   # FP8, F16, F32
    quant_precision: str = "Q2_K"     # Q2_K, Q3_K, Q4_K, Q8_0, F16, etc.
    quantized_bytes: int = 0          # bytes after quantization
    bpw: float = 0.0                  # bits per weight
    
    # Quality metrics (from d2_tensor_profiler.py)
    rms: float = 0.0
    max_abs: float = 0.0
    outlier_rate: float = 0.0
    sparsity: float = 0.0
    entropy_bits: float = 0.0
    
    # Error metrics
    snr_db: float = 0.0          # signal-to-noise ratio
    rel_err: float = 0.0         # relative L2 error
    cosine_sim: float = 0.0      # cosine similarity
    channel_err: float = 0.0     # AWQ-like channel-weighted error
    
    # Latency (from d2_slow_layer_profiler.py)
    latency_us: float = 0.0      # microseconds
    decode_ms: float = 0.0       # milliseconds (from tensor profiler)
    bandwidth_gbs: float = 0.0   # GB/s achieved
    
    # Memory
    vram_bytes: int = 0          # bytes in VRAM
    offload: str = "CUDA0"       # CUDA0, CPU, CUDA_Host
    
    # D2 score
    importance: float = 0.0      # importance score (from d2_tensor_optimizer.py)
    d2_score: float = 0.0        # combined D2 score
    
    # Metadata
    run_id: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "TensorRecord":
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# =============================================================================
# HARDWARE RECORD — GPU/CPU/RAM specs
# =============================================================================

@dataclass
class HardwareRecord:
    """Canonical record for hardware configuration."""
    
    # GPU
    gpu_name: str = ""           # e.g., "NVIDIA GeForce RTX 5070 Laptop GPU"
    gpu_vram_total_mib: int = 0
    gpu_vram_free_mib: int = 0
    gpu_compute_cap: str = ""    # e.g., "12.0"
    gpu_arch: str = ""           # e.g., "Blackwell"
    gpu_sm: int = 0              # e.g., 1200
    gpu_driver: str = ""
    gpu_cuda_version: str = ""
    
    # Memory bandwidth
    vram_bandwidth_gbs: float = 0.0    # measured copy bandwidth
    pcie_bandwidth_gbs: float = 0.0    # PCIe bandwidth
    
    # CPU
    cpu_name: str = ""
    cpu_threads: int = 0
    cpu_ram_total_mib: int = 0
    
    # Software
    llama_cpp_version: str = ""
    llama_cpp_commit: str = ""
    ggml_cuda_archs: str = ""    # e.g., "610,1000" or "1200"
    
    # Metadata
    run_id: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "HardwareRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# =============================================================================
# RUN RECORD — benchmark run metadata
# =============================================================================@dataclass
class RunRecord:
    """Canonical record for a benchmark run."""

    # Run identity
    run_id: str = ""
    timestamp: str = ""

    # Model
    model_path: str = ""
    model_name: str = ""         # e.g., "Qwen3.8-27B-D2-ECO"
    model_size_gib: float = 0.0
    model_params_b: float = 0.0

    # Quantization
    quant_type: str = ""         # e.g., "Q2_K - Medium"
    bpw: float = 0.0

    # Configuration
    ngl: int = 33
    ctx_size: int = 4096
    n_predict: int = 32
    n_prompt: int = 128
    threads: int = 8
    parallel: int = 1
    flash_attn: bool = True

    # KV cache
    kv_type_k: str = "q4_0"
    kv_type_v: str = "q4_0"

    # Speculative decoding - MTP
    spec_type: str = ""          # draft-mtp, dflash, ngram-simple, none
    draft_model: str = ""        # path to draft model GGUF
    n_draft: int = 0             # max tokens drafted per cycle
    acceptance_rate: float = 0.0  # tokens accepted / tokens drafted
    draft_tokens: int = 0        # total tokens drafted
    accepted_tokens: int = 0     # total tokens accepted
    tokens_per_verification: float = 0.0  # avg tokens per verification cycle
    mtp_head_count: int = 0      # number of MTP heads
    mtp_context_memory_mib: float = 0.0  # MTP context memory overhead

    # Results
    pp_tokens_per_sec: float = 0.0    # prompt processing
    tg_tokens_per_sec: float = 0.0    # text generation (raw, no spec)
    effective_tokens_per_sec: float = 0.0  # with speculative decoding
    pp_ms: float = 0.0
    tg_ms: float = 0.0
    total_tokens: int = 0        # total tokens generated
    total_predict_time_ms: float = 0.0

    # VRAM
    vram_used_mib: int = 0
    vram_free_mib: int = 0

    # Quality
    ppl: float = 0.0             # perplexity
    kld_mean: float = 0.0        # KL divergence vs reference
    kld_max: float = 0.0

    # Hardware reference
    hardware: Optional[HardwareRecord] = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        if self.hardware:
            d["hardware"] = self.hardware.to_dict()
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        hw = d.pop("hardware", None)
        rec = cls(**{k: v for k, v in d.items() if k in known})
        if hw:
            rec.hardware = HardwareRecord.from_dict(hw)
        return rec


# =============================================================================
# LAYER RECORD — aggregated per-layer data
# =============================================================================

@dataclass
class LayerRecord:
    """Canonical record for a layer (aggregated from tensors)."""
    
    # Identity
    layer_id: int = 0
    layer_type: str = ""         # "attention" or "gdn"
    
    # Tensors
    n_tensors: int = 0
    total_bytes: int = 0
    quantized_bytes: int = 0
    
    # Aggregated quality
    mean_snr_db: float = 0.0
    max_rel_err: float = 0.0
    mean_rel_err: float = 0.0
    
    # Aggregated latency
    total_latency_us: float = 0.0
    decode_ms: float = 0.0
    
    # Memory
    vram_bytes: int = 0
    
    # D2 decision
    recommendation: str = ""     # "KEEP", "UPGRADE", "DOWNGRADE"
    target_precision: str = ""   # recommended precision
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "LayerRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# =============================================================================
# MODEL MANIFEST — model metadata
# =============================================================================

@dataclass
class ModelManifest:
    """Canonical record for model metadata."""
    
    model_id: str = ""           # e.g., "Qwen3.8-27B"
    architecture: str = ""       # e.g., "qwen35"
    n_layers: int = 64
    n_heads: int = 24
    n_kv_heads: int = 4
    head_dim: int = 256
    n_embd: int = 5120
    n_ff: int = 17408
    n_full_attn: int = 16        # attention layers (every 4th)
    n_gdn: int = 48              # GDN layers
    kv_heads: int = 4
    ssm_state_size: int = 128
    ssm_conv_kernel: int = 4
    
    # Source
    fp8_safetensors_dir: str = ""
    gguf_path: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "ModelManifest":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run_{int(time.time())}_{os.getpid()}"


def save_json(data: Any, path: str):
    """Save data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, list):
            json.dump([d.to_dict() if hasattr(d, "to_dict") else d for d in data], 
                     f, indent=2, ensure_ascii=False)
        elif hasattr(data, "to_dict"):
            json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str, cls=None):
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if cls is None:
        return data
    
    if isinstance(data, list):
        return [cls.from_dict(d) for d in data]
    else:
        return cls.from_dict(data)


# Qwen3.8-27B manifest (hardcoded for this project)
QWEN38_27B = ModelManifest(
    model_id="Qwen3.8-27B",
    architecture="qwen35",
    n_layers=64,
    n_heads=24,
    n_kv_heads=4,
    head_dim=256,
    n_embd=5120,
    n_ff=17408,
    n_full_attn=16,
    n_gdn=48,
    kv_heads=4,
    ssm_state_size=128,
    ssm_conv_kernel=4,
)
