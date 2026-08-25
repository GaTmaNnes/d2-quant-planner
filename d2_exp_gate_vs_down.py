#!/usr/bin/env python3
"""
D2 EXPERIMENT #1: gate_up vs down sensitivity
Create 3 variants of 35B MoE and measure PPL + t/s for each:

Variant A (baseline): all IQ4_NL — the HauhauCS model as-is
Variant B (down-Q3):  ffn_down_exps → Q3_K, rest stays IQ4_NL
Variant C (gate_Q3):  ffn_gate_exps + ffn_up_exps → Q3_K, rest stays IQ4_NL

This isolates whether gate_up or down matters more for quality.
"""
import os, sys, io, json, subprocess, time, re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = "C:/Users/videl/Desktop/lama 1080-5070"
SOURCE_GGUF = "C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
QUANTIZE = os.path.join(BASE, "_backup_cuda12_prebuilt", "llama-quantize.exe")
PERPLEXITY = os.path.join(BASE, "llama-perplexity.exe")
BENCH = os.path.join(BASE, "llama-bench.exe")
PPL_FILE = os.path.join(BASE, "ppl_test_small.txt")
OUTPUT_DIR = os.path.join(BASE, "models", "35b_exp")

# MoE layer pattern
LAYERS = list(range(40))


def gen_tensor_types(variant, output_path):
    """Generate tensor-type file for a variant.
    
    Baseline = IQ4_NL for everything (as in the HauhauCS model).
    We override only the tensors we want to change.
    """
    lines = []
    
    for layer in LAYERS:
        prefix = f"blk.{layer}"

        if variant == "A_baseline":
            # Everything stays IQ4_NL — no overrides needed
            pass
        elif variant == "B_down_Q3":
            # Only ffn_down_exps → Q3_K, rest stays IQ4_NL
            # [CORRIGÉ 25/08/2026] séparateur '=' (l'espace n'est pas parsé par
            # llama-quantize --tensor-type-file)
            lines.append(f"{prefix}.ffn_down_exps.weight=Q3_K")
        elif variant == "C_gate_Q3":
            # Only ffn_gate_exps + ffn_up_exps → Q3_K, rest stays IQ4_NL
            lines.append(f"{prefix}.ffn_gate_exps.weight=Q3_K")
            lines.append(f"{prefix}.ffn_up_exps.weight=Q3_K")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    
    return len(lines)


def quantize_variant(source, types_file, output, base_quant="Q4_K"):
    """Run llama-quantize with tensor-type-file.
    [CORRIGÉ 25/08/2026] --allow-requantize requis : la source est DÉJÀ
    quantifiée (IQ4_NL), pas du F16 — sans ce flag llama-quantize refuse."""
    cmd = [QUANTIZE, source, output, base_quant,
           "--allow-requantize", "--tensor-type-file", types_file]
    print(f"  Quantizing: {os.path.basename(output)} ...")
    print(f"  Cmd: {' '.join(cmd[-3:])}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    size_gb = os.path.getsize(output) / 1e9 if os.path.exists(output) else 0
    print(f"  Done: {size_gb:.2f} GB in {elapsed:.0f}s (exit={result.returncode})")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
    return result.returncode == 0, size_gb, elapsed


def run_ppl(model_path, label):
    """Run perplexity measurement."""
    cmd = [PERPLEXITY, "-m", model_path, "-f", PPL_FILE, "-c", "512",
           "--chunks", "14", "-ngl", "15"]
    print(f"  PPL: {label} ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    
    ppl = None
    for line in result.stdout.split('\n'):
        if 'Final estimate' in line or 'ppl' in line.lower():
            print(f"    {line.strip()}")
        # [CORRIGÉ 25/08/2026] regex tolérante au lieu de split('PPL =') fragile
        m = re.search(r"PPL\s*=\s*([\d.]+)", line)
        if m:
            try:
                ppl = float(m.group(1))
            except ValueError:
                pass
    
    print(f"  PPL done in {elapsed:.0f}s")
    return ppl


def run_bench(model_path, label, ngl=15):
    """Run llama-bench pp128+tg32."""
    cmd = [BENCH, "-m", model_path, "-ngl", str(ngl), "-p", "128", "-n", "32"]
    print(f"  Bench: {label} ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    
    pp = tg = None
    tg_col = pp_col = None
    # [CORRIGÉ 25/08/2026] parsing par colonnes d'en-tête + regex, au lieu de
    # line.split()[1] fragile (dépendait de la position exacte des cellules).
    for line in result.stdout.split('\n'):
        s = line.strip()
        if not s.startswith('|'):
            continue
        cells = [c.strip() for c in s.strip('|').split('|')]
        if "pp128" in cells:
            pp_col = cells.index("pp128")
            continue
        if "tg32" in cells:
            tg_col = cells.index("tg32")
            continue
        if pp_col is not None and pp_col < len(cells):
            m = re.search(r"-?\d+(?:\.\d+)?", cells[pp_col])
            if m:
                try:
                    pp = float(m.group(0))
                except ValueError:
                    pass
                pp_col = None
        if tg_col is not None and tg_col < len(cells):
            m = re.search(r"-?\d+(?:\.\d+)?", cells[tg_col])
            if m:
                try:
                    tg = float(m.group(0))
                except ValueError:
                    pass
                tg_col = None
    
    print(f"  pp128={pp} tg32={tg} ({elapsed:.0f}s)")
    return pp, tg


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    variants = {
        "A_baseline": {"desc": "IQ4_NL full (19.78 GB)", "types_file": None},
        "B_down_Q3":  {"desc": "down_exps=Q3_K, rest=IQ4_NL", "types_file": None},
        "C_gate_Q3":  {"desc": "gate_up=Q3_K, rest=IQ4_NL", "types_file": None},
    }
    
    results = {}
    
    for name, info in variants.items():
        print(f"\n{'='*70}")
        print(f"  VARIANT {name}: {info['desc']}")
        print(f"{'='*70}")
        
        # Generate tensor-type file
        types_path = os.path.join(OUTPUT_DIR, f"types_{name}.txt")
        n = gen_tensor_types(name, types_path)
        print(f"  Tensor-type file: {types_path} ({n} overrides)")
        
        # Quantize
        output_gguf = os.path.join(OUTPUT_DIR, f"35B_{name}.gguf")
        
        if name == "A_baseline":
            # Baseline is the original — just copy/use directly
            output_gguf = SOURCE_GGUF
            print(f"  Using original IQ4_NL as baseline")
        else:
            ok, size, qtime = quantize_variant(SOURCE_GGUF, types_path, output_gguf)
            if not ok:
                print(f"  FAILED — skipping")
                continue
        
        size_gb = os.path.getsize(output_gguf) / 1e9
        
        # Measure PPL
        ppl = run_ppl(output_gguf, name)
        
        # Measure t/s
        pp, tg = run_bench(output_gguf, name)
        
        results[name] = {
            "desc": info['desc'],
            "size_gb": round(size_gb, 2),
            "ppl": ppl,
            "pp128": pp,
            "tg32": tg,
            "overrides": n,
        }
        
        # Save intermediate results
        with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
    
    # Final comparison
    print(f"\n{'='*70}")
    print(f"  RESULTS: gate_up vs down sensitivity")
    print(f"{'='*70}")
    print(f"  {'Variant':<15} {'Size':>8} {'PPL':>8} {'pp128':>8} {'tg32':>8}")
    print(f"  {'-'*47}")
    
    baseline_ppl = results.get("A_baseline", {}).get("ppl", 0)
    
    for name, r in results.items():
        ppl_str = f"{r['ppl']:.2f}" if r['ppl'] else "N/A"
        pp_str = f"{r['pp128']:.1f}" if r['pp128'] else "N/A"
        tg_str = f"{r['tg32']:.1f}" if r['tg32'] else "N/A"
        delta = ""
        if r['ppl'] and baseline_ppl:
            d = r['ppl'] - baseline_ppl
            delta = f" ({d:+.2f})"
        print(f"  {name:<15} {r['size_gb']:>6.1f}GB {ppl_str:>8}{delta:>8} {pp_str:>8} {tg_str:>8}")
    
    print(f"\n  CONCLUSION:")
    if "B_down_Q3" in results and "C_gate_Q3" in results:
        ppl_b = results["B_down_Q3"].get("ppl", 99)
        ppl_c = results["C_gate_Q3"].get("ppl", 99)
        if ppl_b and ppl_c:
            if ppl_b < ppl_c:
                print(f"  down_Q3 (PPL={ppl_b:.2f}) is BETTER than gate_Q3 (PPL={ppl_c:.2f})")
                print(f"  => gate_up is MORE SENSITIVE than down for quality")
                print(f"  => STRATEGY: keep gate_up in IQ4_NL, quantize down in Q3")
            else:
                print(f"  gate_Q3 (PPL={ppl_c:.2f}) is BETTER than down_Q3 (PPL={ppl_b:.2f})")
                print(f"  => down is MORE SENSITIVE than gate_up for quality")
                print(f"  => STRATEGY: keep down in IQ4_NL, quantize gate_up in Q3")
    
    # Save final report
    report_path = os.path.join(OUTPUT_DIR, "gate_vs_down_report.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report saved: {report_path}")
