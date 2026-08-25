#!/usr/bin/env python3
"""
D2 Axis #8: ngl vs PPL sweep for 35B MoE.
Measure PPL at different ngl values to find the sweet spot.
"""
import subprocess, os, sys, io, json, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = "C:/Users/videl/Desktop/lama 1080-5070"
IQ4 = "C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
PERPLEXITY = os.path.join(BASE, "llama-perplexity.exe")
BENCH = os.path.join(BASE, "llama-bench.exe")
PPL_FILE = os.path.join(BASE, "ppl_test_small.txt")
RESULTS = os.path.join(BASE, "models", "35b_exp", "ngl_ppl_sweep.json")

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)


def measure_ppl(model, ngl):
    cmd = [PERPLEXITY, "-m", model, "-f", PPL_FILE, "-c", "512",
           "--chunks", "14", "-ngl", str(ngl)]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    ppl = None
    for line in result.stdout.split('\n'):
        if 'Final estimate' in line:
            try:
                ppl = float(line.split('PPL =')[1].split()[0])
            except:
                pass
    return ppl, elapsed


def measure_bench(model, ngl):
    cmd = [BENCH, "-m", model, "-ngl", str(ngl), "-p", "128", "-n", "32"]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - t0
    pp = tg = None
    for line in result.stdout.split('\n'):
        if 'pp' in line and 't/s' in line:
            try: pp = float(line.split()[1])
            except: pass
        if 'tg' in line and 't/s' in line:
            try: tg = float(line.split()[1])
            except: pass
    return pp, tg, elapsed


if __name__ == '__main__':
    ngl_values = [5, 10, 15, 18, 20, 25, 30, 35, 40]
    results = []
    
    for ngl in ngl_values:
        print(f"\n--- ngl={ngl} ---")
        ppl, ppl_time = measure_ppl(IQ4, ngl)
        pp, tg, bench_time = measure_bench(IQ4, ngl)
        
        entry = {"ngl": ngl, "ppl": ppl, "pp128": pp, "tg32": tg}
        results.append(entry)
        ppl_s = f"{ppl:.4f}" if ppl else "FAIL"
        pp_s = f"{pp:.1f}" if pp else "FAIL"
        tg_s = f"{tg:.1f}" if tg else "FAIL"
        print(f"  PPL={ppl_s} ({ppl_time:.0f}s) pp128={pp_s} tg32={tg_s} ({bench_time:.0f}s)")
        
        with open(RESULTS, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  NGL SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"  {'ngl':>4} {'PPL':>8} {'pp128':>8} {'tg32':>8}")
    print(f"  {'-'*28}")
    for r in results:
        print(f"  {r['ngl']:>4} {r['ppl']:>8.4f} {r['pp128']:>8.1f} {r['tg32']:>8.1f}")
