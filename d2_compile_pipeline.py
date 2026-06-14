"""
D2 Compile Pipeline — Unified Quantization Planner
===================================================
Pipeline complet : BW Sim → Spectral → ILP → Router → Export

Usage :
  python d2_compile_pipeline.py --demo --budget 8.0
  python d2_compile_pipeline.py --demo --budget 8.0 --target tensorrt
  python d2_compile_pipeline.py model.gguf --budget 8.0 --target llamacpp

Exports :
  llamacpp  → commande shell + JSON tensor-type
  tensorrt  → JSON plan TensorRT INT8/FP16
"""

import json, math, sys, argparse
from collections import Counter
from typing import Dict, List, Optional, Tuple

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import gguf
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — BW MODEL (ROCmFP4 Sim)
# ══════════════════════════════════════════════════════════════════════════════

FORMAT_SPECS = {
    "FP16":     {"bpw": 2.000, "jit": 0.00, "kernel": "CUBLASLT"},
    "BF16":     {"bpw": 2.000, "jit": 0.00, "kernel": "CUBLASLT"},
    "INT8":     {"bpw": 1.000, "jit": 0.00, "kernel": "CUTLASS"},
    "INT4_AWQ": {"bpw": 0.500, "jit": 0.00, "kernel": "MARLIN"},
    "NVFP4":    {"bpw": 0.531, "jit": 0.00, "kernel": "MARLIN"},
    "Q4NX_JIT": {"bpw": 0.563, "jit": 0.06, "kernel": "CUTLASS"},
}

def _tile_util(cols: int) -> float:
    per_col = max(cols // 8, 1)
    aligned = (per_col // 256) * 256
    if aligned == 0:
        return 0.5
    return aligned / per_col

def tps_bw(hidden: int, file_gb: float, fmt: str, bw_eff: float = 60.0) -> float:
    """Roofline TPS pour un format donné."""
    spec = FORMAT_SPECS.get(fmt, FORMAT_SPECS["FP16"])
    u = _tile_util(hidden)
    base = (bw_eff / max(file_gb, 1e-9)) / max(u ** 0.3, 0.1)
    return base * (1.0 - spec["jit"])

def bw_for_shape(shape: List[int], fmt: str) -> float:
    """GB occupés par une couche dans un format donné."""
    params = shape[0] * shape[1] if len(shape) >= 2 else shape[0]
    return params * FORMAT_SPECS.get(fmt, FORMAT_SPECS["FP16"])["bpw"] / 1e9

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — SPECTRAL ANALYZER (alpha_w)
# ══════════════════════════════════════════════════════════════════════════════

LAYER_WEIGHT_CLS = {
    "embed": 99.0, "head": 99.0, "norm": 99.0,
    "kv": 2.0, "attn": 1.6, "ffn": 0.7, "other": 1.0,
}

def classify(name: str) -> str:
    n = name.lower()
    if any(p in n for p in ("embed", "wte", "wpe")):             return "embed"
    if any(p in n for p in ("lm_head", "output.weight")):         return "head"
    if any(p in n for p in ("norm", "ln_", "rms_norm")):          return "norm"
    if any(p in n for p in ("k_proj", "v_proj", "wk", "wv")):    return "kv"
    if any(p in n for p in ("q_proj", "attn", "self_attn")):      return "attn"
    if any(p in n for p in ("ffn", "mlp", "gate", "up", "down")): return "ffn"
    return "other"

def alpha_w_synthetic(layer_idx: int, cls: str, n_layers: int) -> float:
    """Alpha spectral synthétique calibré sur distributions réelles LLM."""
    import random
    rng = random.Random(layer_idx * 31 + hash(cls) % 97)
    depth = layer_idx / max(n_layers - 1, 1)  # 0..1

    base = {
        "embed": 1.05, "head": 1.15, "norm": 0.90,
        "kv": 1.75, "attn": 2.10, "ffn": 2.20, "other": 1.50,
    }.get(cls, 1.5)

    # alpha augmente légèrement vers les couches profondes
    trend = 0.15 * depth
    noise = rng.gauss(0, 0.12)
    return max(0.5, base + trend + noise)

def alpha_w_from_bytes(data: bytes, shape: List[int], ttype: int) -> float:
    """Proxy alpha_w depuis bytes bruts GGUF (sans SVD)."""
    if not HAS_NP:
        return 1.8
    raw = np.frombuffer(data[:min(len(data), 8192)], dtype=np.uint8).astype(np.float32)
    if raw.size < 4:
        return 1.5
    cv = raw.std() / (raw.mean() + 1e-6)
    # calibration empirique: CV élevé → spectre concentré → alpha élevé
    return float(np.clip(0.8 + cv * 1.5, 0.5, 4.0))

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — ILP OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

# Seuils alpha_w (calibrés XDNA2 + RTX)
ALPHA_INT4  = 2.10   # alpha >= seuil → INT4 safe
ALPHA_INT8  = 1.50   # alpha >= seuil → INT8 safe
ALPHA_BF16  = 1.20   # alpha >= seuil → BF16 safe

# BPW par dtype
BPW = {"FP16": 2.0, "BF16": 2.0, "INT8": 1.0, "INT4": 0.5, "NVFP4": 0.531}

# TPS relatif (FP16 = 1.0)
TPS_REL = {"FP16": 1.0, "BF16": 1.05, "INT8": 1.8, "INT4": 2.65, "NVFP4": 2.65}

def _policy_from_alpha(name: str, alpha: float, cls: str) -> str:
    """Politique de quantization par alpha_w et classe de couche."""
    # couches critiques forcées FP16
    if cls in ("embed", "head", "norm"):
        return "FP16"
    if alpha >= ALPHA_INT4:
        return "INT4"
    if alpha >= ALPHA_INT8:
        return "INT8"
    if alpha >= ALPHA_BF16:
        return "BF16"
    return "FP16"

def ilp_optimize(layers: List[Dict], budget_gb: float) -> List[Dict]:
    """
    Optimisation ILP sous contrainte VRAM.
    Maximise TPS * qualité en respectant budget_gb.
    Fallback greedy si scipy absent.
    """
    n = len(layers)
    dtypes_per_layer = []
    for lay in layers:
        alpha = lay["alpha_w"]
        cls   = lay["cls"]
        base  = _policy_from_alpha(lay["name"], alpha, cls)

        # options disponibles (base + moins agressifs)
        order = ["INT4", "INT8", "BF16", "FP16"]
        idx   = order.index(base) if base in order else 3
        options = order[idx:]   # de la plus agressive à FP16
        dtypes_per_layer.append(options)

    if HAS_SCIPY and HAS_NP:
        return _ilp_scipy(layers, dtypes_per_layer, budget_gb)
    else:
        return _greedy(layers, dtypes_per_layer, budget_gb)

def _greedy(layers, dtypes_per_layer, budget_gb):
    """Greedy : part du plan le plus agressif, remonte vers FP16 si dépasse budget."""
    plan = []
    total = 0.0
    choices = []

    # Phase 1 : tout au plus agressif
    for lay, opts in zip(layers, dtypes_per_layer):
        dtype = opts[0]
        gb    = bw_for_shape(lay["shape"], dtype)
        total += gb
        choices.append({"idx": len(plan), "dtype": dtype, "gb": gb, "opts": opts})
        plan.append({**lay, "dtype": dtype, "gb": gb})

    # Phase 2 : upgrade vers FP16 les couches les moins critiques jusqu'à budget
    if total > budget_gb:
        # trier par poids sensibilité croissant (moins sensible = peut rester agressif)
        n = len(layers)
        upgrades = sorted(
            [(i, LAYER_WEIGHT_CLS.get(layers[i]["cls"], 1.0)) for i in range(n)],
            key=lambda x: x[1], reverse=True
        )
        for i, _ in upgrades:
            if total <= budget_gb:
                break
            opts = dtypes_per_layer[i]
            cur_idx = opts.index(plan[i]["dtype"]) if plan[i]["dtype"] in opts else 0
            if cur_idx + 1 < len(opts):
                old_gb = plan[i]["gb"]
                new_dt = opts[cur_idx + 1]
                new_gb = bw_for_shape(layers[i]["shape"], new_dt)
                total += new_gb - old_gb
                plan[i]["dtype"] = new_dt
                plan[i]["gb"]    = new_gb

    return plan

def _ilp_scipy(layers, dtypes_per_layer, budget_gb):
    """
    ILP relaxé via linprog.
    Variable x[i][j] = 1 si couche i utilise dtype j.
    Minimise -TPS, contrainte RAM <= budget.
    """
    n = len(layers)
    # aplatir variables
    vars_map = []  # (layer_idx, dtype_idx, dtype_name)
    for i, opts in enumerate(dtypes_per_layer):
        for j, dt in enumerate(opts):
            vars_map.append((i, j, dt))
    nv = len(vars_map)

    # objectif : max sum(TPS_REL[dt] * w_cls)
    c_obj = []
    for i, j, dt in vars_map:
        tps = TPS_REL.get(dt, 1.0)
        w   = LAYER_WEIGHT_CLS.get(layers[i]["cls"], 1.0)
        c_obj.append(-tps / max(w, 0.1))  # négatif car linprog minimise

    # contrainte RAM
    c_ram = [bw_for_shape(layers[i]["shape"], dt) for i, j, dt in vars_map]
    A_ub = [c_ram]
    b_ub = [budget_gb]

    # contrainte : une dtype par couche
    A_eq = []
    b_eq = []
    for li in range(n):
        row = [1.0 if i == li else 0.0 for i, j, dt in vars_map]
        A_eq.append(row)
        b_eq.append(1.0)

    bounds = [(0, 1)] * nv
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    plan = [{**lay, "dtype": dtypes_per_layer[i][0],
             "gb": bw_for_shape(lay["shape"], dtypes_per_layer[i][0])}
            for i, lay in enumerate(layers)]

    if res.success:
        x = res.x
        for vi, (i, j, dt) in enumerate(vars_map):
            if x[vi] > 0.5:
                plan[i]["dtype"] = dt
                plan[i]["gb"]    = bw_for_shape(layers[i]["shape"], dt)

    return plan

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — LATENCY MONITOR
# ══════════════════════════════════════════════════════════════════════════════

def bottleneck_analysis(plan: List[Dict], bw_eff: float = 60.0) -> Dict:
    """
    Identifie les bottlenecks compute vs memory bound par couche.
    Roofline : si lat_compute > lat_bw → compute bound.
    """
    results = []
    for lay in plan:
        shape  = lay["shape"]
        dtype  = lay["dtype"]
        if len(shape) < 2:
            continue
        rows, cols = shape[0], shape[1]
        params     = rows * cols
        gb         = params * BPW.get(dtype, 2.0) / 1e9

        # latence mémoire (decode batch=1)
        lat_bw_ms = gb / bw_eff * 1000.0

        # latence compute (FLOPS = 2*rows*cols, peak Watt ~ 200 TFLOPS pour RTX)
        flops      = 2 * rows * cols
        peak_tflops = 200e12  # RTX 4090 FP16
        lat_cp_ms  = flops / peak_tflops * 1000.0

        bound = "MEM" if lat_bw_ms > lat_cp_ms else "COMPUTE"
        ratio = lat_bw_ms / max(lat_cp_ms, 1e-12)

        results.append({
            "name":      lay["name"],
            "dtype":     dtype,
            "bound":     bound,
            "ratio_mem_cp": ratio,
            "lat_bw_ms": lat_bw_ms,
            "lat_cp_ms": lat_cp_ms,
            "gb":        gb,
        })

    n_mem  = sum(1 for r in results if r["bound"] == "MEM")
    n_cp   = len(results) - n_mem
    total_lat = sum(r["lat_bw_ms"] for r in results)

    return {
        "layers":    results,
        "n_mem":     n_mem,
        "n_compute": n_cp,
        "total_lat_ms": total_lat,
        "bottleneck": "MEMORY" if n_mem > n_cp else "COMPUTE",
    }

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — DYNAMIC ROUTER V13
# ══════════════════════════════════════════════════════════════════════════════

SINK_THRESHOLD = 128
STAB_EMA       = 0.15
HYSTERESIS_MIN = 0.60
ATTN_OPS       = {"attn", "kv"}

class D2DynamicRouter:
    """Routeur hystérésis V13 appliqué sur le plan ILP."""

    def __init__(self, plan: List[Dict]):
        self.plan   = {lay["name"]: lay for lay in plan}
        self.state  = {lay["name"]: lay["dtype"] for lay in plan}
        self.stab   = {lay["name"]: 1.0 for lay in plan}

    def route(self, complexity: float, token_age: int) -> Dict[str, str]:
        out = {}
        for name, lay in self.plan.items():
            cls     = lay["cls"]
            is_attn = cls in ATTN_OPS
            base_dt = lay["dtype"]

            # règles V13
            if token_age < SINK_THRESHOLD and is_attn:
                target = "FP16"
            elif complexity > 0.8 and is_attn:
                target = "FP16"
            else:
                target = base_dt

            # hystérésis
            current    = self.state[name]
            switching  = (current != target)
            if switching and self.stab[name] < HYSTERESIS_MIN:
                target = current
            elif switching:
                self.state[name] = target

            sig = 0.0 if switching else 1.0
            self.stab[name] = (1 - STAB_EMA) * self.stab[name] + STAB_EMA * sig
            out[name] = target

        return out

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — EXPORTERS
# ══════════════════════════════════════════════════════════════════════════════

# Correspondance dtype D2 → quantization llama.cpp
LLAMA_MAP = {
    "FP16": "f16",
    "BF16": "bf16",
    "INT8": "q8_0",
    "INT4": "q4_k",
    "NVFP4": "q4_k",
}

# Correspondance dtype D2 → TensorRT
TRT_MAP = {
    "FP16":  "fp16",
    "BF16":  "fp16",    # TRT ne supporte pas BF16 nativement en toutes versions
    "INT8":  "int8",
    "INT4":  "int4",
    "NVFP4": "int4",
}

def export_llamacpp(plan: List[Dict], model_path: str = "model.gguf",
                    out_path: str = "llama_quant_plan.json") -> str:
    """
    Génère la commande llama.cpp avec --tensor-type par couche.
    Retourne la commande shell.
    """
    tensor_args = []
    plan_json = {}

    for lay in plan:
        qt = LLAMA_MAP.get(lay["dtype"], "q8_0")
        tensor_args.append("{}={}".format(lay["name"], qt))
        plan_json[lay["name"]] = qt

    with open(out_path, "w") as f:
        json.dump({"model": model_path, "tensor_types": plan_json}, f, indent=2)

    # commande llama-quantize
    tt_args = " ".join('--tensor-type "{}"'.format(a) for a in tensor_args[:8])
    cmd = (
        "# Plan complet dans {}\n"
        "llama-quantize \\\n"
        "  {} \\\n"
        "  --allow-requantize \\\n"
        "  {} output.gguf\n"
        "\n"
        "# Ou avec llama.cpp server :\n"
        "llama-server -m {} \\\n"
        "  --n-gpu-layers 99 \\\n"
        "  --type-k q8_0 --type-v q8_0"
    ).format(out_path, model_path, tt_args, model_path)

    return cmd

def export_tensorrt(plan: List[Dict], out_path: str = "tensorrt_plan.json") -> Dict:
    """
    Génère le plan JSON pour TensorRT-LLM.
    Format : {"quantization": {"layers": {name: {dtype, calibration}}}}
    """
    layers_trt = {}
    for lay in plan:
        trt_dt = TRT_MAP.get(lay["dtype"], "fp16")
        layers_trt[lay["name"]] = {
            "dtype":       trt_dt,
            "calibration": "entropy" if trt_dt == "int8" else "none",
            "alpha_w":     round(lay.get("alpha_w", 0.0), 3),
        }

    trt_plan = {
        "quantization": {
            "mode":   "mixed",
            "layers": layers_trt,
        },
        "build_config": {
            "max_batch_size": 1,
            "max_seq_len":    4096,
            "strongly_typed": True,
        },
    }
    with open(out_path, "w") as f:
        json.dump(trt_plan, f, indent=2)

    return trt_plan

# ══════════════════════════════════════════════════════════════════════════════
# DEMO MODEL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def make_demo_layers(model_name: str = "Llama-3-8B", n_layers: int = 32) -> List[Dict]:
    hidden = 4096
    intermediate = 14336
    layers = []

    def add(name, shape, idx, cls):
        alpha = alpha_w_synthetic(idx, cls, n_layers)
        layers.append({"name": name, "shape": shape, "cls": cls,
                        "alpha_w": alpha, "layer_idx": idx})

    # embed + head
    add("model.embed_tokens.weight",   [32000, hidden],       0,  "embed")
    add("model.norm.weight",           [hidden],              0,  "norm")
    add("lm_head.weight",              [32000, hidden],       0,  "head")

    # transformer blocks
    for b in range(n_layers):
        add("model.layers.{}.input_layernorm.weight".format(b),
            [hidden], b, "norm")
        add("model.layers.{}.self_attn.q_proj.weight".format(b),
            [hidden, hidden], b, "attn")
        add("model.layers.{}.self_attn.k_proj.weight".format(b),
            [hidden // 8, hidden], b, "kv")
        add("model.layers.{}.self_attn.v_proj.weight".format(b),
            [hidden // 8, hidden], b, "kv")
        add("model.layers.{}.self_attn.o_proj.weight".format(b),
            [hidden, hidden], b, "attn")
        add("model.layers.{}.post_attention_layernorm.weight".format(b),
            [hidden], b, "norm")
        add("model.layers.{}.mlp.gate_proj.weight".format(b),
            [intermediate, hidden], b, "ffn")
        add("model.layers.{}.mlp.up_proj.weight".format(b),
            [intermediate, hidden], b, "ffn")
        add("model.layers.{}.mlp.down_proj.weight".format(b),
            [hidden, intermediate], b, "ffn")

    return layers

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(layers: List[Dict], budget_gb: float, model_path: str,
                 target: str, bw_eff: float = 60.0, quiet: bool = False) -> Dict:

    # ── Étape 1 : ILP Optimizer ─────────────────────────────────────────────
    if not quiet:
        print("  [1/4] ILP Optimizer (budget={:.1f} GB, scipy={})".format(
              budget_gb, HAS_SCIPY))
    plan = ilp_optimize(layers, budget_gb)
    total_gb = sum(lay["gb"] for lay in plan)

    # ── Étape 2 : BW Simulation ─────────────────────────────────────────────
    if not quiet:
        print("  [2/4] Simulation BW (bw_eff={} GB/s)".format(bw_eff))
    hidden  = next((l["shape"][1] for l in plan if len(l["shape"]) >= 2
                    and l["shape"][1] > 512), 4096)
    tps_map = {}
    for lay in plan:
        if len(lay["shape"]) >= 2:
            gb  = lay["gb"]
            fmt = lay["dtype"]
            tps_map[lay["name"]] = tps_bw(hidden, gb, fmt, bw_eff)

    # ── Étape 3 : Bottleneck Analysis ───────────────────────────────────────
    if not quiet:
        print("  [3/4] Bottleneck Analysis")
    bn = bottleneck_analysis(plan, bw_eff)

    # ── Étape 4 : Dynamic Router V13 (scénario génération) ─────────────────
    if not quiet:
        print("  [4/4] Dynamic Router V13 (génération batch=1)")
    router = D2DynamicRouter(plan)
    routed_gen  = router.route(complexity=0.2, token_age=512)
    routed_rag  = router.route(complexity=0.9, token_age=10)

    # ── Export ───────────────────────────────────────────────────────────────
    if target == "llamacpp":
        out_path = "llama_quant_plan.json"
        cmd      = export_llamacpp(plan, model_path, out_path)
        export_info = {"file": out_path, "cmd": cmd}
    else:
        out_path = "tensorrt_plan.json"
        export_tensorrt(plan, out_path)
        export_info = {"file": out_path}

    return {
        "plan":        plan,
        "total_gb":    total_gb,
        "bottleneck":  bn,
        "routed_gen":  routed_gen,
        "routed_rag":  routed_rag,
        "export":      export_info,
    }

# ══════════════════════════════════════════════════════════════════════════════
# RAPPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(result: Dict, budget_gb: float, target: str):
    plan   = result["plan"]
    bn     = result["bottleneck"]
    export = result["export"]

    dist = Counter(lay["dtype"] for lay in plan)
    total_gb = result["total_gb"]

    print()
    print("=" * 65)
    print("  RAPPORT D2 COMPILE PIPELINE")
    print("=" * 65)
    print("  Couches    : {}".format(len(plan)))
    print("  VRAM       : {:.2f} GB / {:.1f} GB ({:.0f}%)".format(
          total_gb, budget_gb, total_gb / budget_gb * 100))
    print()

    print("  Distribution :")
    for dt, n in sorted(dist.items(), key=lambda x: BPW.get(x[0], 2.0)):
        bar = "█" * (n * 30 // len(plan))
        print("    {:<8} {:>4} couches  {}".format(dt, n, bar))

    print()
    print("  Bottleneck : {} ({} couches MEM / {} COMPUTE)".format(
          bn["bottleneck"], bn["n_mem"], bn["n_compute"]))
    print("  Lat totale : {:.2f} ms (decode batch=1)".format(bn["total_lat_ms"]))

    print()
    print("  Router V13 — Génération (complexity=0.2, age=512) :")
    gen_dist = Counter(result["routed_gen"].values())
    for dt, n in gen_dist.items():
        print("    {:<8} {}".format(dt, n))

    print()
    print("  Router V13 — RAG (complexity=0.9, age=10) :")
    rag_dist = Counter(result["routed_rag"].values())
    for dt, n in rag_dist.items():
        print("    {:<8} {}".format(dt, n))

    print()
    print("  Export ({}) : {}".format(target, export["file"]))
    if "cmd" in export:
        print()
        print("  ─── Commande llama.cpp ───────────────────────────────")
        print(export["cmd"])

    print()
    print("  Échelle gains vs FP16 statique :")
    avg_tps = sum(TPS_REL.get(lay["dtype"], 1.0) for lay in plan) / len(plan)
    avg_mem = sum(BPW.get(lay["dtype"], 2.0) for lay in plan) / len(plan) / 2.0
    print("    TPS moyen  : {:.2f}x  (FP16=1.0)".format(avg_tps))
    print("    Mémoire    : {:.0f}%  vs FP16".format(avg_mem * 100))
    print()

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="D2 Compile Pipeline — BW Sim + ILP + Router → llama.cpp / TensorRT")
    parser.add_argument("model",    nargs="?", default=None,
                        help="Fichier GGUF (omis = mode demo)")
    parser.add_argument("--demo",   action="store_true",
                        help="Mode demo sans fichier modèle")
    parser.add_argument("--budget", type=float, default=8.0,
                        help="Budget VRAM en GB (défaut: 8.0)")
    parser.add_argument("--target", choices=["llamacpp", "tensorrt"],
                        default="llamacpp",
                        help="Format d'export (défaut: llamacpp)")
    parser.add_argument("--bw-eff", type=float, default=60.0,
                        help="Bande passante effective GB/s (défaut: 60)")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    demo = args.demo or not args.model
    model_path = args.model or "model.gguf"

    print("=" * 65)
    print("  D2 Compile Pipeline — {} → {}".format(
          "démo" if demo else model_path, args.target.upper()))
    print("  Budget: {:.1f} GB  |  BW: {} GB/s  |  scipy={}  |  gguf={}".format(
          args.budget, args.bw_eff, HAS_SCIPY, HAS_GGUF))
    print("=" * 65)
    print()

    # ── Génération des couches ──────────────────────────────────────────────
    if demo:
        print("  Génération modèle synthétique Llama-3-8B (32 blocs)...")
        layers = make_demo_layers("Llama-3-8B", n_layers=32)
    elif HAS_GGUF and model_path.endswith(".gguf"):
        print("  Lecture GGUF : {}".format(model_path))
        # scan GGUF simplifié (utilise d2_rtx_gguf_profiler si dispo)
        try:
            from d2_rtx_gguf_profiler import scan_gguf
            raw = scan_gguf(model_path, verbose=False)
            layers = [{"name": r["name"], "shape": r["shape"],
                       "cls":  classify(r["name"]),
                       "alpha_w": r["alpha_w"], "layer_idx": i}
                      for i, r in enumerate(raw)]
        except ImportError:
            print("  [!] d2_rtx_gguf_profiler non trouvé — mode démo activé")
            layers = make_demo_layers(n_layers=32)
    else:
        if not demo:
            print("  [!] gguf non installé — mode démo activé")
        layers = make_demo_layers(n_layers=32)

    print("  {} couches chargées.".format(len(layers)))
    print()

    # ── Pipeline ────────────────────────────────────────────────────────────
    result = run_pipeline(layers, args.budget, model_path,
                          args.target, args.bw_eff, args.quiet)

    if not args.quiet:
        print_report(result, args.budget, args.target)

    return 0


if __name__ == "__main__":
    sys.exit(main())
