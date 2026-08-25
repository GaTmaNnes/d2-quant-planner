#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 COST MODEL — prédire le meilleur quant (modèle, matériel, workload) AVANT le benchmark.
==========================================================================================
Consomme :
  - hardware_fingerprint.json  (d2_hw_probe.py : valeurs MESUREES, pas theoriques)
  - d2_quant_reference.json    (dataset qualite/compression : publisher-reported pour 27B,
                                tailles fichiers locales pour Qwen3.5-9B)
  - models/*.gguf              (tailles reelles detectees)

Modèle mémoire (spec D2) :
  VRAM = weights(precision) + KV(ctx, precision) + SSM_state(precision)
       + workspace(runtime) + activations + allocator

Modèle de temps (critical path, PAS de somme naive) :
  T_total = T_compute + T_dequant + T_DMA_setup + T_transfer + T_sync
  - decode : T_mem = weights_bytes / (vram_copy_bw * kernel_factor)
             T_cmp = flops_token / gpu_tflops
             T = max(T_mem, T_cmp)            # overlap mem/compute
  - offload : T_transport = offload_bytes / pcie_measured_bw  (H2D, chemin PCIe UNE fois,
              DMA pas re-additionne -> voir topology.double_count_risk)

Principes :
  - bande passante d'un kernel != bande passante memcpy. Recalibre le 21/08/2026 sur
    llama-bench (build 85e22ea) : decode Q4_K_S 55.34 t/s x 5.39 Go = 298 Go/s et
    NVFP4 54.89 t/s x 5.66 Go = 311 Go/s, soit ~305 Go/s moyen vs memcpy ~176 Go/s
    => kernel_factor ~1.7 (coherent avec memset ~306 Go/s). d2_runtime_probe.py
    remplira le facteur reel par quant.
  - la quantite utile = delta_quality / delta_GB et delta_quality / delta_ms_token.

Exemples :
  python d2_cost_model.py                          # machine auto-detectee
  python d2_cost_model.py --machine gtx1080 --ctx 16384
  python d2_cost_model.py --ctx 65536              # test 64k (deborde les 8 Go)
"""

import argparse
import json
import os
import re
import sys

from d2_schema import RunRecord, HardwareRecord, QWEN38_27B
from d2_registry import D2Registry

HERE = os.path.dirname(os.path.abspath(__file__))
FP_PATH = os.path.join(HERE, "hardware_fingerprint.json")
REF_PATH = os.path.join(HERE, "d2_quant_reference.json")
MODELS_DIR = os.path.join(HERE, "models")
OUT_PATH = os.path.join(HERE, "d2_cost_decision.json")

BYTES_PER_GB = 1e9

# Modèle cible (Qwen3.5-9B, AGENTS.md) — surcharge par le reference local si présent.
MODEL = {
    "name": "Qwen3.5-9B (hybride SSM+attention)",
    "params_b": 9.0,
    "layers": 32,
    "attention_layers": 8,
    "ssm_layers": 24,
    "hidden": 4096,
    "ffn": 12288,
    "head_count_kv": 4,
    "head_dim": 256,
}

# Config machines connues. capacity = VRAM utile. kernel_factor = bande passante de
# LECTURE effective du kernel de décodage / bande passante memcpy (copie D2D). Le decode
# relit les poids en lecture seule, donc dépasse la copie. Recalibre le 21/08/2026 sur
# llama-bench (85e22ea) : ~305 Go/s de lecture (Q4_K_S + NVFP4) vs memcpy ~176-184 Go/s
# => 1.7 pour la RTX 5070. 4090/5090 = heuristique (pas de mesure reelle).
MACHINES = {
    "rtx5070": {"name": "RTX 5070 Laptop (Blackwell sm_120)", "vram_gb": 8.0,
                "workspace_gb": 1.0, "ctx_def": 32768, "kernel_factor": 1.7},
    "gtx1080": {"name": "GTX 1080 (Pascal sm_61)", "vram_gb": 8.0,
                "workspace_gb": 1.0, "ctx_def": 16384, "kernel_factor": 1.4},
    "rtx4090": {"name": "RTX 4090 (Ada sm_89)", "vram_gb": 24.0,
                "workspace_gb": 2.0, "ctx_def": 32768, "kernel_factor": 1.8},
    "rtx5090": {"name": "RTX 5090 (Blackwell sm_120)", "vram_gb": 32.0,
                "workspace_gb": 2.0, "ctx_def": 32768, "kernel_factor": 1.8},
    "cpu": {"name": "CPU-only (AMD Ryzen AI 9 365, 33.6 Go RAM)", "vram_gb": 32.0,
            "workspace_gb": 0.5, "ctx_def": 8192, "kernel_factor": 1.0},
}

MODEL_27B = {
    "name": "Qwen3.8-27B",
    "params_b": 27.0,
    "layers": 64,
    "attention_layers": 16,
    "ssm_layers": 0,
    "hidden": 5120,
    "ffn": 15360,
    "head_count_kv": 4,
    "head_dim": 256,
}

BYTES_PER_ELEM = {"fp16": 2, "q8": 1, "q4": 0.5, "fp32": 4}


def kv_bytes_per_token(attn_layers, kv_heads, head_dim, precision="fp16"):
    """KV = 2 (K,V) x kv_heads x head_dim x attention_layers x octets/element."""
    return 2 * kv_heads * head_dim * attn_layers * BYTES_PER_ELEM[precision]


def load_fingerprint():
    if not os.path.exists(FP_PATH):
        sys.exit(f"[!] {FP_PATH} absent — lancer d2_hw_probe.py d'abord")
    with open(FP_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_reference():
    if not os.path.exists(REF_PATH):
        return {"models": []}
    with open(REF_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def scan_local_models():
    quants = []
    if not os.path.isdir(MODELS_DIR):
        return quants
    for f in sorted(os.listdir(MODELS_DIR)):
        if f.lower().endswith(".gguf"):
            p = os.path.join(MODELS_DIR, f)
            try:
                size_gb = round(os.path.getsize(p) / BYTES_PER_GB, 2)
            except OSError:
                size_gb = None
            quant = f.replace("Qwen3.5-", "").replace(".gguf", "") or f
            quants.append({"quant": quant, "size_gb": size_gb, "file": f})
    return quants


def quant_family(name):
    """Famille de quantification : 'Q4_K_S'->'Q4', '9B-Q4_K_S'->'Q4', 'UD-Q5_K_XL'->'Q5',
    'IQ2_S'->'IQ2', 'Q8_0'->'Q8'."""
    n = name or ""
    iq = re.search(r"IQ(\d+)", n)
    if iq:
        return "IQ" + iq.group(1)
    q = re.search(r"(?<![A-Za-z])Q(\d+)", n)
    if q:
        return "Q" + q.group(1)
    return None


def reference_quants(ref, model_name):
    for m in ref.get("models", []):
        if m.get("model", "").lower() == model_name.lower():
            return m.get("quants", [])
    return []


def reference_quality(ref, model_name, size_gb, family):
    """Qualite rapprochee : plus proche voisin en TAILLE au sein de la MEME famille de quant
    dans le reference (Q4_K_S ~= Q4, PAS par taille brute : 5.4GB != 11GB)."""
    quants = [q for q in reference_quants(ref, model_name) if q.get("top1") is not None]
    if family:
        fam = [q for q in quants if quant_family(q.get("quant", "")) == family]
        if fam:
            quants = fam
    best, bd = None, None
    for q in quants:
        d = abs(q["size_gb"] - size_gb) if size_gb else 0
        if bd is None or d < bd:
            best, bd = q, d
    if best is None:
        return None, None
    return best["top1"], best["quant"]


def eval_quant(q, hw, machine, model, ctx):
    cfg = MACHINES[machine]
    weights_gb = q.get("size_gb")
    if not weights_gb:
        return None

    # --- footprint mémoire ---
    kv_gb = kv_bytes_per_token(model["attention_layers"], model["head_count_kv"],
                               model["head_dim"], "fp16") * ctx / BYTES_PER_GB
    ssm_gb = (model["ssm_layers"] * model["hidden"] * 2 * 2) / BYTES_PER_GB  # etat SSM ~ negligeable
    workspace_gb = cfg["workspace_gb"]
    total_gb = weights_gb + kv_gb + ssm_gb + workspace_gb
    fit = total_gb <= cfg["vram_gb"]

    # --- temps par token (decode), critical path ---
    vram_bw = (hw.get("vram") or {}).get("measured_copy_gbs")
    pcie_bw = (hw.get("pcie") or {}).get("measured_h2d_gbs")
    gpu_tf = (hw.get("compute") or {}).get("gpu") or {}

    # vram_bw (memcpy) x kernel_factor = bande passante de lecture effective du kernel.
    t_mem = weights_gb / (vram_bw * cfg["kernel_factor"]) * 1000 if vram_bw else None  # ms
    flops_token = 2 * model["params_b"] * 1e9
    tf = gpu_tf.get("fp16_tflops") or gpu_tf.get("fp32_tflops")
    t_cmp = flops_token / (tf * 1e12) * 1000 if tf else None  # ms

    parts = []
    if t_mem is not None:
        parts.append(("memory", t_mem))
    if t_cmp is not None:
        parts.append(("compute", t_cmp))
    t_core = max(t for _, t in parts) if parts else None  # overlap mem/compute

    # --- offload : transport si pas fit ---
    t_transport = None
    if not fit and pcie_bw:
        offload_gb = total_gb - cfg["vram_gb"]
        t_transport = offload_gb * BYTES_PER_GB / (pcie_bw * 1e9) * 1000  # ms (chemin PCIe, DMA inclus)
    t_total = (t_core + t_transport) if t_transport else t_core
    tps = 1000 / t_total if t_total else None

    # --- qualité : locale inconnue -> rapprochee par famille dans le reference 27B ---
    top1 = q.get("top1")
    est_from = None
    if top1 is None:
        top1, est_from = reference_quality(ref_global, "Qwen3.8-27B", weights_gb,
                                           quant_family(q["quant"]))

    # --- classification du goulot ---
    bottleneck, conf = None, None
    if t_core and t_transport:
        dom = max(t_core, t_transport)
        total = t_core + t_transport
        bottleneck = "transport" if t_transport == dom else "kernel(memory/compute)"
        conf = round(dom / total, 2)
    elif parts:
        parts.sort(key=lambda x: x[1], reverse=True)
        dom, second = parts[0], parts[1]
        bottleneck = dom[0]
        conf = round(dom[1] / max(dom[1] + second[1], 1e-9), 2)

    return {
        "quant": q["quant"], "size_gb": weights_gb,
        "kv_gb": round(kv_gb, 3), "ssm_gb": round(ssm_gb, 4),
        "workspace_gb": workspace_gb, "total_vram_gb": round(total_gb, 2),
        "fit": fit,
        "t_memory_ms": round(t_mem, 2) if t_mem else None,
        "t_compute_ms": round(t_cmp, 2) if t_cmp else None,
        "t_transport_ms": round(t_transport, 2) if t_transport else None,
        "t_total_ms": round(t_total, 2) if t_total else None,
        "tps_est": round(tps, 1) if tps else None,
        "top1_est": top1,
        "top1_from": est_from or q["quant"],
        "bottleneck": bottleneck,
        "confidence": conf,
        "delta_gb": None, "delta_top1": None,
    }


def pick(qs, key):
    cands = [q for q in qs if q and q.get(key) is not None]
    return max(cands, key=lambda q: q[key]) if cands else None


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 COST MODEL — prédire le meilleur quant avant benchmark")
    ap.add_argument("--machine", choices=sorted(MACHINES), default=None,
                    help="auto-detecte depuis le fingerprint sinon")
    ap.add_argument("--ctx", type=int, default=None, help="contexte de workload (defaut selon machine)")
    ap.add_argument("--kernel-factor", type=float, default=None,
                    help="facteur bande passante kernel vs memcpy (defaut selon machine)")
    ap.add_argument("--reference-only", action="store_true",
                    help="utiliser les quants du reference 27B comme candidats (exercice de validation multi-machine)")
    ap.add_argument("--json", default=OUT_PATH)
    args = ap.parse_args()

    hw = load_fingerprint()
    global ref_global
    ref_global = load_reference()

    # machine auto-detectee : RTX 5070 present -> rtx5070, sinon gtx1080
    machine = args.machine
    if machine is None:
        gname = (hw.get("gpu") or {}).get("name", "").lower()
        machine = "rtx5070" if "5070" in gname else ("gtx1080" if "1080" in gname else "cpu")
    cfg = MACHINES[machine]
    ctx = args.ctx or cfg["ctx_def"]
    if args.kernel_factor:
        cfg = dict(cfg, kernel_factor=args.kernel_factor)

    # quants candidats : locaux (models/) preferes pour la decision 9B ;
    # reference 27B seulement pour la qualite OU l'exercice de validation (--reference-only)
    local = scan_local_models()
    model = MODEL
    if local and not args.reference_only:
        quants = []
        for q in local:
            d = dict(q)
            d["top1"] = None
            quants.append(d)
    else:
        model = MODEL_27B
        quants = list(reference_quants(ref_global, "Qwen3.8-27B"))

    evals = [r for r in (eval_quant(q, hw, machine, model, ctx) for q in quants) if r]

    # delta_quality/delta_GB (reference par taille croissante)
    evals.sort(key=lambda e: e["size_gb"])
    for i, e in enumerate(evals):
        if i == 0:
            e["delta_gb"], e["delta_top1"] = 0.0, 0.0
            continue
        p = evals[i - 1]
        e["delta_gb"] = round(e["size_gb"] - p["size_gb"], 2)
        if e["top1_est"] is not None and p["top1_est"] is not None:
            e["delta_top1"] = round(e["top1_est"] - p["top1_est"], 2)

    fits = [e for e in evals if e["fit"]]
    best_quality = pick(fits, "top1_est")
    fastest = pick(fits, "tps_est")
    # recommandation : meilleur ratio (top1 - penalite vitesse relative)
    if fits:
        ref_top = best_quality["top1_est"] or 0
        ref_tps = fastest["tps_est"] or 1
        for e in fits:
            e["score"] = round((e["top1_est"] or 0) - 0.05 * (1 - (e["tps_est"] or 0) / ref_tps) * 100, 2)
        recommended = max(fits, key=lambda e: e["score"])
    else:
        recommended = max(evals, key=lambda e: (e["top1_est"] or 0))

    decision = {
        "generated_at": hw.get("generated_at"),
        "model": model["name"],
        "machine": cfg["name"],
        "ctx": ctx,
        "kernel_factor": cfg["kernel_factor"],
        "hardware_measured": {
            "vram_copy_gbs": (hw.get("vram") or {}).get("measured_copy_gbs"),
            "pcie_h2d_gbs": (hw.get("pcie") or {}).get("measured_h2d_gbs"),
            "gpu_fp16_tflops": (hw.get("compute") or {}).get("gpu", {}).get("fp16_tflops"),
        },
        "recommendation": {
            "best_quality": best_quality["quant"] if best_quality else None,
            "best_quality_top1": best_quality["top1_est"] if best_quality else None,
            "fastest": fastest["quant"] if fastest else None,
            "fastest_tps": fastest["tps_est"] if fastest else None,
            "recommended": recommended["quant"] if recommended else None,
            "recommended_score": recommended.get("score") if recommended else None,
            "recommended_top1": recommended["top1_est"] if recommended else None,
            "recommended_tps": recommended["tps_est"] if recommended else None,
        },
        "candidates": evals,
        "topology_rule": "transport H2D compte une seule fois (chemin PCIe), DMA non re-additionne",
    }

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(decision, fh, ensure_ascii=False, indent=2)

    print(f"Machine : {cfg['name']}  | ctx {ctx}  | kernel_factor {cfg['kernel_factor']}")
    print(f"{'quant':<16}{'GB':>6}{'KV@ctx':>9}{'fit':>5}{'t_ms':>7}{'tps':>7}{'top1':>7}  goulot")
    for e in evals:
        print(f"{e['quant']:<16}{e['size_gb']:>6}{e['kv_gb']:>9.1f}{'Y' if e['fit'] else 'N':>5}"
              f"{e['t_total_ms'] if e['t_total_ms'] is not None else '-':>7}"
              f"{e['tps_est'] if e['tps_est'] else '-':>7}"
              f"{e['top1_est'] if e['top1_est'] is not None else '-':>7}  "
              f"{e['bottleneck']} (conf {e['confidence']})")
    r = decision["recommendation"]
    print(f"\n[+] recommande : {r['recommended']} (top1 {r['recommended_top1']}, "
          f"{r['recommended_tps']} t/s est) -> {args.json}")
    
    # Also save to registry
    registry_path = os.path.join(HERE, "d2_ecosystem", "registry.json")
    reg = D2Registry(model=QWEN38_27B)
    if os.path.exists(registry_path):
        reg.load(registry_path)
    
    # Add hardware record
    hw_data = decision.get("hardware_measured", {})
    hw = HardwareRecord(
        gpu_name=cfg.get("name", ""),
        gpu_vram_total_mib=int(cfg.get("vram_gb", 0) * 1024),
        vram_bandwidth_gbs=hw_data.get("vram_copy_gbs", 0),
        pcie_bandwidth_gbs=hw_data.get("pcie_h2d_gbs", 0),
    )
    reg.set_hardware(hw)
    
    # Add run record with recommendation
    rec_data = decision.get("recommendation", {})
    run = RunRecord(
        model_name=model.get("name", ""),
        model_params_b=model.get("params_b", 0),
        ctx_size=ctx,
        kv_type_k=args.kv_type if hasattr(args, 'kv_type') else "f16",
        kv_type_v=args.kv_type if hasattr(args, 'kv_type') else "f16",
        tg_tokens_per_sec=rec_data.get("recommended_tps", 0),
        hardware=hw,
    )
    reg.add_run(run)
    reg.save(registry_path)
    print(f"[+] Registry updated: {registry_path}")


if __name__ == "__main__":
    main()