#!/usr/bin/env python3
"""
D2 Quantization Compiler — Unified Framework

Combines:
- Spectral analysis (α_w)
- C-Vector IR (information-theoretic)
- CUDA cost modeling
- Graph rewriting / fragmentation fixing
- Corrected quantization policy (stability → precision mapping)

This is the complete compiler for layer-wise quantization planning.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


# ─────────────────────────────────────────────────────────────────────────────
# 1. SPECTRAL IR NODE (LLVM-STYLE)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpectralIRNode:
    """LLVM-style IR node with corrected spectrum logic."""
    
    name: str
    alpha: float          # Spectral exponent
    density: float        # Sparsity (0-1)
    
    def __post_init__(self):
        """Compute derived IR fields."""
        # C-vector components (information-theoretic descriptor)
        self.cs = 1.0 / np.log1p(self.alpha + 1e-6)      # Spectral stability
        self.ci = 1.0 - min(1.0, self.density)            # Information entropy proxy
        self.cr = self.alpha * (1.0 + self.density)       # Risk (heavy tail)
        self.cd = self.density                             # Density
        
        # Infer quantization type based on CORRECTED logic
        # HIGH α_w + HIGH density = UNSTABLE = needs FP16
        # LOW α_w + LOW density = STABLE = can use NVFP4
        self.type = self._infer_type()
    
    def _infer_type(self) -> str:
        """
        CORRECTED: Risk score → Precision mapping
        
        Instability = α_w (1 + ρ_d)
        
        HIGH instability → FP16 (conservative)
        LOW instability → NVFP4 (aggressive compression)
        """
        risk_score = 0.6 * self.alpha + 0.4 * self.cr
        
        # Thresholds for stability (INVERTED from naive approach)
        if risk_score > 1.8 or "ssm" in self.name.lower():
            return "FP16"
        elif risk_score > 1.4:
            return "INT8"
        elif risk_score > 1.1:
            return "FP8"
        else:
            return "NVFP4"
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            "name": self.name,
            "alpha": float(self.alpha),
            "density": float(self.density),
            "c_vector": [float(self.cs), float(self.ci), float(self.cr), float(self.cd)],
            "type": self.type,
        }


def build_ir_from_json(json_path: str) -> List[SpectralIRNode]:
    """
    Build LLVM-style Spectral IR from JSON precision map.
    
    Args:
        json_path: Path to precision_map.json
        
    Returns:
        List of SpectralIRNode objects (topologically sorted)
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Topological sort for GPU execution order
    sorted_keys = sorted(
        data.keys(),
        key=lambda x: (
            int(x.split('.')[1]) if "blk" in x else -1,
            x.split('.')[-2] if "blk" in x else x
        )
    )
    
    nodes = [
        SpectralIRNode(
            name=k,
            alpha=data[k].get("alpha", 1.0),
            density=data[k].get("density", 0.0)
        )
        for k in sorted_keys
    ]
    
    return nodes


# ─────────────────────────────────────────────────────────────────────────────
# 2. UNIFIED OPTIMIZER (SPECTRAL + CUDA COST)
# ─────────────────────────────────────────────────────────────────────────────

CUDA_TRANSITION_MATRIX = {
    # (from_type, to_type) → cost (hardware desync penalty)
    ("NVFP4", "FP16"): 2.5,  # Catastrophic tensor core mismatch
    ("NVFP4", "INT8"): 1.5,
    ("NVFP4", "FP8"): 1.2,
    ("INT8", "FP16"): 1.0,
    ("INT8", "FP8"): 0.7,
    ("FP8", "FP16"): 0.8,
    ("FP16", "NVFP4"): 2.5,  # Symmetrical
    ("FP16", "INT8"): 1.0,
    ("FP16", "FP8"): 0.8,
}


@dataclass
class OptimizedLayer:
    """Result of optimization pass."""
    name: str
    type: str                 # Quantization dtype
    cost: float               # Total cost
    isolation: bool           # Force isolation (no fusion)
    spectral_cost: float      # Intrinsic spectral drift
    switch_cost: float        # Hardware transition penalty


def unified_optimizer(ir_nodes: List[SpectralIRNode]) -> List[OptimizedLayer]:
    """
    Optimize layer-wise quantization considering:
    - Spectral stability (α_w)
    - CUDA hardware transition costs
    - SSM/attention critical paths
    
    Args:
        ir_nodes: List of SpectralIRNode from IR build
        
    Returns:
        List of OptimizedLayer with final decisions
    """
    optimized = []
    prev = None
    
    for node in ir_nodes:
        # 1. Intrinsic spectral cost (deviation from optimal α* ≈ 1.0)
        spectral_cost = abs(node.alpha - 1.0)
        
        # 2. Hardware transition cost (GPU kernel desync)
        switch_cost = 0.0
        if prev is not None and prev.type != node.type:
            pair = tuple(sorted([prev.type, node.type]))
            switch_cost = CUDA_TRANSITION_MATRIX.get(pair, 2.0)
        
        # 3. Critical path isolation (SSM, attention heads)
        # These layers have high sensitivity to quantization
        isolation = False
        if "ssm" in node.name.lower() and (node.alpha > 1.3 or node.density > 0.25):
            node.type = "FP16"
            isolation = True
        
        total_cost = spectral_cost + switch_cost
        
        optimized.append(OptimizedLayer(
            name=node.name,
            type=node.type,
            cost=total_cost,
            isolation=isolation,
            spectral_cost=spectral_cost,
            switch_cost=switch_cost
        ))
        prev = node
    
    return optimized


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH REWRITER (FRAGMENTATION FIXING)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphEvent:
    """Single node in fixed computation graph."""
    layer: str
    type: str
    fragmented: bool
    action: str


def simulate_and_fix_fragmentation(optimized: List[OptimizedLayer]) -> Dict:
    """
    Simulate CUDA execution and apply bidirectional boundary smoothing.
    
    Key insight: Don't just detect fragmentation, FIX it by propagating
    high-precision requirements forward/backward.
    
    Args:
        optimized: List of OptimizedLayer
        
    Returns:
        Dictionary with fixed graph and metrics
    """
    fragmentation_events = 0
    fixed_graph = []
    
    # Working copy for in-place graph rewriting
    working_graph = [
        {"name": opt.name, "type": opt.type}
        for opt in optimized
    ]
    
    graph_len = len(working_graph)
    
    # ─── PASS 1: Forward fragmentation detection ────────────────────────────
    for i in range(graph_len):
        event = GraphEvent(
            layer=working_graph[i]["name"],
            type=working_graph[i]["type"],
            fragmented=False,
            action="OK"
        )
        
        if i > 0:
            prev_layer = working_graph[i-1]
            curr_layer = working_graph[i]
            
            # Fragmentation = dtype mismatch between consecutive layers
            if prev_layer["type"] != curr_layer["type"]:
                fragmentation_events += 1
                event.fragmented = True
                
                # ─── AUTO-FIX STRATEGY (Bidirectional Propagation) ────────
                
                # Rule 1: FP16 "pulls down" lower precisions
                if prev_layer["type"] == "FP16" and curr_layer["type"] in ["INT8", "FP8"]:
                    curr_layer["type"] = "FP16"
                    event.action = "FORCED_FP16_FORWARD_BRIDGE"
                
                # Rule 2: FP16 "pulls up" lower precisions (backward)
                elif curr_layer["type"] == "FP16" and prev_layer["type"] in ["INT8", "FP8"]:
                    prev_layer["type"] = "FP16"
                    # Fix already-recorded event if needed
                    if len(fixed_graph) > 0:
                        fixed_graph[-1].type = "FP16"
                        fixed_graph[-1].action = "FORCED_FP16_BACKWARD_BRIDGE"
                    event.action = "ALIGN_WITH_PREVIOUS_BACKFIX"
                
                # Rule 3: NVFP4 isolation with INT8 buffer
                elif "NVFP4" in [prev_layer["type"], curr_layer["type"]]:
                    if prev_layer["type"] == "NVFP4":
                        curr_layer["type"] = "INT8"
                        event.action = "SMOOTH_NVFP4_EXIT_INT8"
                    else:
                        prev_layer["type"] = "INT8"
                        if len(fixed_graph) > 0:
                            fixed_graph[-1].type = "INT8"
                        event.action = "SMOOTH_INT8_ENTRY_FROM_NVFP4"
        
        # Update event with final type
        event.type = working_graph[i]["type"]
        fixed_graph.append(event)
    
    return {
        "initial_fragmentation_score": fragmentation_events / max(1, graph_len),
        "fragmentation_events": fragmentation_events,
        "graph": [asdict(e) for e in fixed_graph],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN COMPILER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class D2QuantizationCompiler:
    """End-to-end LLVM-style quantization compiler."""
    
    def __init__(self, json_path: str, output_prefix: str = "d2_compiled"):
        """
        Initialize compiler.
        
        Args:
            json_path: Input precision_map.json
            output_prefix: Output file prefix
        """
        self.json_path = json_path
        self.output_prefix = output_prefix
        self.ir_nodes = []
        self.optimized = []
        self.fixed_graph = {}
    
    def compile(self) -> Dict:
        """
        Run full compilation pipeline.
        
        Returns:
            Dictionary with all compilation results
        """
        print("🚀 D2 Quantization Compiler — Starting pipeline...\n")
        
        # ─── Pass 1: Build LLVM Spectral IR ────────────────────────────────
        print("[Pass 1] Building LLVM Spectral IR...")
        self.ir_nodes = build_ir_from_json(self.json_path)
        print(f"  ✓ {len(self.ir_nodes)} nodes typés")
        
        # ─── Pass 2: Unified Optimization ──────────────────────────────────
        print("[Pass 2] Running unified optimizer...")
        self.optimized = unified_optimizer(self.ir_nodes)
        
        total_cost = sum(opt.cost for opt in self.optimized)
        avg_switch_cost = np.mean([opt.switch_cost for opt in self.optimized])
        print(f"  ✓ Total cost: {total_cost:.3f}")
        print(f"  ✓ Avg switch cost: {avg_switch_cost:.3f}")
        
        # ─── Pass 3: Graph Fragmentation Fixing ────────────────────────────
        print("[Pass 3] Fixing CUDA graph fragmentation...")
        self.fixed_graph = simulate_and_fix_fragmentation(self.optimized)
        frag_score = self.fixed_graph["initial_fragmentation_score"]
        print(f"  ✓ Fragmentation score: {frag_score:.3f}")
        
        # ─── Construct final output ────────────────────────────────────────
        result = self._build_output()
        
        # ─── Save artifacts ───────────────────────────────────────────────
        self._save_artifacts(result)
        
        print("\n✅ Compilation complete!")
        return result
    
    def _build_output(self) -> Dict:
        """Build final compilation output."""
        return {
            "metadata": {
                "compiler": "D2QuantizationCompiler",
                "version": "2.0",
                "passes": 3,
            },
            "ir": {
                "nodes": [node.to_dict() for node in self.ir_nodes],
                "count": len(self.ir_nodes),
            },
            "optimization": {
                "layers": [asdict(opt) for opt in self.optimized],
                "total_cost": float(sum(opt.cost for opt in self.optimized)),
                "avg_spectral_cost": float(np.mean([opt.spectral_cost for opt in self.optimized])),
                "avg_switch_cost": float(np.mean([opt.switch_cost for opt in self.optimized])),
            },
            "graph_fixing": self.fixed_graph,
            "quantization_plan": self._generate_quantization_plan(),
        }
    
    def _generate_quantization_plan(self) -> List[Dict]:
        """Generate final quantization plan for llama.cpp."""
        plan = []
        for opt in self.optimized:
            plan.append({
                "layer": opt.name,
                "dtype": opt.type,
                "isolation": opt.isolation,
                "cost": opt.cost,
            })
        return plan
    
    def _save_artifacts(self, result: Dict):
        """Save compilation artifacts to files."""
        # Full result
        with open(f"{self.output_prefix}.json", "w") as f:
            json.dump(result, f, indent=2)
        
        # Quantization plan only (for llama.cpp)
        plan_only = result["quantization_plan"]
        with open(f"{self.output_prefix}_plan.json", "w") as f:
            json.dump(plan_only, f, indent=2)
        
        # Graph visualization (CSV for easy inspection)
        import csv
        with open(f"{self.output_prefix}_graph.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=["layer", "type", "fragmented", "action"])
            writer.writeheader()
            writer.writerows(result["graph_fixing"]["graph"])
        
        print(f"  📁 Saved: {self.output_prefix}.json")
        print(f"  📁 Saved: {self.output_prefix}_plan.json")
        print(f"  📁 Saved: {self.output_prefix}_graph.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python d2_compiler.py <precision_map.json> [output_prefix]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "d2_compiled"
    
    compiler = D2QuantizationCompiler(json_path, output_prefix)
    result = compiler.compile()
    
    # Print summary
    print("\n" + "=" * 65)
    print("COMPILATION SUMMARY")
    print("=" * 65)
    print(f"Input: {json_path}")
    print(f"Layers: {result['ir']['count']}")
    print(f"Total optimization cost: {result['optimization']['total_cost']:.3f}")
    print(f"Fragmentation events: {result['graph_fixing']['fragmentation_events']}")
    print(f"Output files: {output_prefix}.*")
    print("=" * 65)
