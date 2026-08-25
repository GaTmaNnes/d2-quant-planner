#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 REPORT — generate comprehensive report from registry
=========================================================

Reads d2_ecosystem/registry.json and generates:
  - Console summary
  - Markdown report
  - Recommendations for ECO-v2 rebuild

Usage:
  python d2_report.py
  python d2_report.py --markdown d2_report.md
  python d2_report.py --json d2_report.json
"""

import argparse
import json
import os
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "d2_ecosystem", "registry.json")


def load_registry():
    """Load registry from JSON."""
    if not os.path.exists(REGISTRY):
        print(f"[!] Registry not found: {REGISTRY}")
        return None
    with open(REGISTRY, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_tensors(data):
    """Analyze tensor records from registry."""
    tensors = data.get("tensors", {})
    
    # Group by component
    by_component = defaultdict(list)
    for tid, t in tensors.items():
        comp = t.get("component", "unknown")
        by_component[comp].append(t)
    
    # Compute stats per component
    stats = {}
    for comp, ts in by_component.items():
        n = len(ts)
        total_bytes = sum(t.get("original_bytes", 0) for t in ts)
        snr_values = [t.get("snr_db", 0) for t in ts if t.get("snr_db", 0) > 0]
        rel_err_values = [t.get("rel_err", 0) for t in ts if t.get("rel_err", 0) > 0]
        latency_values = [t.get("decode_ms", 0) for t in ts if t.get("decode_ms", 0) > 0]
        
        stats[comp] = {
            "count": n,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / 1e6, 1),
            "mean_snr_db": round(sum(snr_values) / len(snr_values), 2) if snr_values else 0,
            "min_snr_db": round(min(snr_values), 2) if snr_values else 0,
            "mean_rel_err": round(sum(rel_err_values) / len(rel_err_values), 6) if rel_err_values else 0,
            "max_rel_err": round(max(rel_err_values), 6) if rel_err_values else 0,
            "mean_latency_ms": round(sum(latency_values) / len(latency_values), 2) if latency_values else 0,
            "max_latency_ms": round(max(latency_values), 2) if latency_values else 0,
        }
    
    return stats


def analyze_layers(data):
    """Analyze layer records from registry."""
    layers = data.get("layers", {})
    
    stats = {
        "total": len(layers),
        "attention": 0,
        "gdn": 0,
        "mean_snr": 0,
        "max_latency_ms": 0,
    }
    
    snr_values = []
    latency_values = []
    
    for lid, l in layers.items():
        if l.get("layer_type") == "attention":
            stats["attention"] += 1
        else:
            stats["gdn"] += 1
        
        snr = l.get("mean_snr_db", 0) or 0
        if snr > 0:
            snr_values.append(snr)
        
        lat = l.get("decode_ms", 0) or 0
        if lat > 0:
            latency_values.append(lat)
    
    if snr_values:
        stats["mean_snr"] = round(sum(snr_values) / len(snr_values), 2)
    if latency_values:
        stats["max_latency_ms"] = round(max(latency_values), 2)
    
    return stats


def analyze_runs(data):
    """Analyze run records from registry."""
    runs = data.get("runs", [])
    
    if not runs:
        return {"count": 0}
    
    stats = {
        "count": len(runs),
        "configs": [],
    }
    
    for r in runs:
        config = {
            "model": r.get("model_name", "?"),
            "ngl": r.get("ngl", 0),
            "ctx": r.get("ctx_size", 0),
            "kv_k": r.get("kv_type_k", "?"),
            "kv_v": r.get("kv_type_v", "?"),
            "spec": r.get("spec_type", "none"),
            "tg_tps": r.get("tg_tokens_per_sec", 0),
            "effective_tps": r.get("effective_tokens_per_sec", 0),
            "acceptance": r.get("acceptance_rate", 0),
            "vram_mib": r.get("vram_used_mib", 0),
        }
        stats["configs"].append(config)
    
    return stats


def generate_markdown(data):
    """Generate markdown report from registry data."""
    lines = []
    lines.append("# D2 REPORT — Registry Analysis")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Model info
    model = data.get("model", {})
    lines.append("## Model")
    lines.append(f"- **ID**: {model.get('model_id', '?')}")
    lines.append(f"- **Architecture**: {model.get('architecture', '?')}")
    lines.append(f"- **Layers**: {model.get('n_layers', '?')} ({model.get('n_full_attn', '?')} attention, {model.get('n_gdn', '?')} GDN)")
    lines.append(f"- **Hidden**: {model.get('n_embd', '?')}, FFN: {model.get('n_ff', '?')}")
    lines.append("")
    
    # Hardware info
    hw = data.get("hardware", {})
    if hw:
        lines.append("## Hardware")
        lines.append(f"- **GPU**: {hw.get('gpu_name', '?')}")
        lines.append(f"- **VRAM**: {hw.get('gpu_vram_total_mib', '?')} MiB")
        lines.append(f"- **Bandwidth**: {hw.get('vram_bandwidth_gbs', '?')} GB/s")
        lines.append("")
    
    # Tensor analysis
    lines.append("## Tensor Analysis")
    tensor_stats = analyze_tensors(data)
    
    lines.append("| Component | Count | Size (MB) | Mean SNR | Max Latency |")
    lines.append("|-----------|-------|-----------|----------|-------------|")
    for comp in sorted(tensor_stats.keys()):
        s = tensor_stats[comp]
        lines.append(f"| {comp} | {s['count']} | {s['total_mb']} | {s['mean_snr_db']} dB | {s['max_latency_ms']} ms |")
    lines.append("")
    
    # Sensitive tensors
    lines.append("### Sensitive Tensors (SNR < 30 dB)")
    tensors = data.get("tensors", {})
    sensitive = [(tid, t) for tid, t in tensors.items() 
                 if 0 < t.get("snr_db", 0) < 30]
    sensitive.sort(key=lambda x: x[1].get("snr_db", 0))
    
    if sensitive:
        lines.append("| Tensor | SNR (dB) | Rel Err | Latency (ms) |")
        lines.append("|--------|----------|---------|--------------|")
        for tid, t in sensitive[:10]:
            lines.append(f"| {tid} | {t.get('snr_db', 0):.2f} | {t.get('rel_err', 0):.6f} | {t.get('decode_ms', 0):.2f} |")
    else:
        lines.append("No sensitive tensors found.")
    lines.append("")
    
    # Slow tensors
    lines.append("### Slow Tensors (latency > 1 ms)")
    slow = [(tid, t) for tid, t in tensors.items() 
            if t.get("decode_ms", 0) > 1.0]
    slow.sort(key=lambda x: x[1].get("decode_ms", 0), reverse=True)
    
    if slow:
        lines.append("| Tensor | Latency (ms) | SNR (dB) | Size (MB) |")
        lines.append("|--------|--------------|----------|-----------|")
        for tid, t in slow[:10]:
            lines.append(f"| {tid} | {t.get('decode_ms', 0):.2f} | {t.get('snr_db', 0):.2f} | {t.get('original_bytes', 0) / 1e6:.1f} |")
    else:
        lines.append("No slow tensors found.")
    lines.append("")
    
    # Layer analysis
    lines.append("## Layer Analysis")
    layer_stats = analyze_layers(data)
    lines.append(f"- **Total layers**: {layer_stats['total']}")
    lines.append(f"- **Attention layers**: {layer_stats['attention']}")
    lines.append(f"- **GDN layers**: {layer_stats['gdn']}")
    lines.append(f"- **Mean SNR**: {layer_stats['mean_snr']} dB")
    lines.append(f"- **Max latency**: {layer_stats['max_latency_ms']} ms")
    lines.append("")
    
    # Run analysis
    lines.append("## Benchmark Runs")
    run_stats = analyze_runs(data)
    
    if run_stats["count"] > 0:
        lines.append(f"**{run_stats['count']} runs recorded**")
        lines.append("")
        lines.append("| Model | ngl | ctx | KV | Spec | TG (t/s) | VRAM (MiB) |")
        lines.append("|-------|-----|-----|-----|------|----------|------------|")
        for c in run_stats["configs"]:
            lines.append(f"| {c['model'][:30]} | {c['ngl']} | {c['ctx']} | {c['kv_k']}/{c['kv_v']} | {c['spec']} | {c['tg_tps']:.2f} | {c['vram_mib']} |")
    else:
        lines.append("No runs recorded.")
    lines.append("")
    
    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    lines.append("### Priority Actions")
    lines.append("1. **q4_0 KV cache** — +28% vs f16, compatible with partial offload")
    lines.append("2. **ngl=33** — sweet spot for RTX 5070")
    lines.append("3. **parallel=1** — eliminate unused KV slots")
    lines.append("4. **ctx=32768** — maximum context with q4_0 KV")
    lines.append("")
    lines.append("### Tensor Upgrades (if needed)")
    
    # Find tensors that could benefit from upgrade
    upgrades = []
    for tid, t in tensors.items():
        snr = t.get("snr_db", 0)
        lat = t.get("decode_ms", 0)
        quant = t.get("quant_precision", "")
        
        if snr > 0 and snr < 25 and quant in ("Q2_K",):
            upgrades.append((tid, "Q3_K", f"low SNR ({snr:.1f} dB)"))
        elif lat > 2.0 and snr > 40 and quant in ("Q8_0", "Q5_K"):
            upgrades.append((tid, "Q4_K", f"high latency ({lat:.1f} ms)"))
    
    if upgrades:
        lines.append("| Tensor | Current | Target | Reason |")
        lines.append("|--------|---------|--------|--------|")
        for tid, target, reason in upgrades[:10]:
            t = tensors[tid]
            lines.append(f"| {tid} | {t.get('quant_precision', '?')} | {target} | {reason} |")
    else:
        lines.append("No tensor upgrades recommended.")
    lines.append("")
    
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="D2 Report Generator")
    ap.add_argument("--markdown", default=os.path.join(HERE, "d2_report.md"))
    ap.add_argument("--json", default=os.path.join(HERE, "d2_report.json"))
    args = ap.parse_args()
    
    data = load_registry()
    if not data:
        return
    
    # Console summary
    print("=" * 70)
    print("  D2 REPORT — Registry Analysis")
    print("=" * 70)
    
    model = data.get("model", {})
    print(f"  Model: {model.get('model_id', '?')}")
    
    hw = data.get("hardware", {})
    if hw:
        print(f"  GPU: {hw.get('gpu_name', '?')}")
    
    tensors = data.get("tensors", {})
    layers = data.get("layers", {})
    runs = data.get("runs", {})
    
    print(f"  Tensors: {len(tensors)}")
    print(f"  Layers: {len(layers)}")
    print(f"  Runs: {len(runs) if isinstance(runs, (list, dict)) else 0}")
    print()
    
    # Tensor summary
    tensor_stats = analyze_tensors(data)
    print("  Tensor Components:")
    for comp in sorted(tensor_stats.keys()):
        s = tensor_stats[comp]
        print(f"    {comp:25s} count={s['count']:4d}  snr={s['mean_snr_db']:6.2f} dB  lat={s['max_latency_ms']:8.2f} ms")
    print()
    
    # Sensitive tensors
    sensitive = [(tid, t) for tid, t in tensors.items() 
                 if 0 < t.get("snr_db", 0) < 30]
    if sensitive:
        print(f"  Sensitive tensors (SNR < 30 dB): {len(sensitive)}")
        for tid, t in sorted(sensitive, key=lambda x: x[1].get("snr_db", 0))[:5]:
            print(f"    {tid}: SNR={t.get('snr_db', 0):.2f} dB, lat={t.get('decode_ms', 0):.2f} ms")
    print()
    
    # Run summary
    run_stats = analyze_runs(data)
    if run_stats["count"] > 0:
        print(f"  Benchmark Runs: {run_stats['count']}")
        for c in run_stats["configs"]:
            print(f"    {c['model'][:30]:30s} tps={c['tg_tps']:6.2f}  vram={c['vram_mib']} MiB")
    print()
    
    # Generate markdown
    md = generate_markdown(data)
    with open(args.markdown, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  [+] Markdown: {args.markdown}")
    
    # Generate JSON summary
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "hardware": hw,
        "tensor_stats": tensor_stats,
        "layer_stats": analyze_layers(data),
        "run_stats": run_stats,
        "n_sensitive_tensors": len(sensitive) if sensitive else 0,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [+] JSON: {args.json}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
