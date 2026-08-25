#!/usr/bin/env python3
"""Quick ngl sweep: 5 key points for 35B MoE IQ4_NL."""
import subprocess, os, sys, io, json, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = "C:/Users/videl/Desktop/lama 1080-5070"
IQ4 = "C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
PPL_EXE = os.path.join(BASE, "llama-perplexity.exe")
BENCH = os.path.join(BASE, "llama-bench.exe")
PPL_FILE = os.path.join(BASE, "ppl_test_small.txt")

results = []

for ngl in [5, 10, 15, 18, 20]:
    print(f"\n--- ngl={ngl} ---")
    
    # PPL
    cmd_ppl = [PPL_EXE, "-m", IQ4, "-f", PPL_FILE, "-c", "512", "--chunks", "5", "-ngl", str(ngl)]
    t0 = time.time()
    r = subprocess.run(cmd_ppl, capture_output=True, text=True, timeout=300)
    ppl_t = time.time() - t0
    # Combine stdout+stderr
    out = r.stdout + r.stderr
    ppl = None
    for line in out.split('\n'):
        if 'Final estimate' in line:
            try: ppl = float(line.split('PPL =')[1].split()[0])
            except: pass
    
    # Bench
    cmd_bench = [BENCH, "-m", IQ4, "-ngl", str(ngl), "-p", "128", "-n", "32"]
    t0 = time.time()
    r = subprocess.run(cmd_bench, capture_output=True, text=True, timeout=120)
    bench_t = time.time() - t0
    pp = tg = None
    for line in (r.stdout + r.stderr).split('\n'):
        if 'pp128' in line:
            try: pp = float(line.split('|')[4].strip())
            except: pass
        if 'tg32' in line:
            try: tg = float(line.split('|')[4].strip())
            except: pass
    
    entry = {"ngl": ngl, "ppl": ppl, "pp128": pp, "tg32": tg,
             "ppl_time": round(ppl_t), "bench_time": round(bench_t)}
    results.append(entry)
    
    ppl_s = f"{ppl:.4f}" if ppl else "FAIL"
    pp_s = f"{pp:.1f}" if pp else "FAIL"
    tg_s = f"{tg:.1f}" if tg else "FAIL"
    print(f"  PPL={ppl_s} ({ppl_t:.0f}s) pp128={pp_s} tg32={tg_s} ({bench_t:.0f}s)")

print(f"\n{'='*60}")
print(f"  NGL SWEEP: 35B MoE IQ4_NL")
print(f"{'='*60}")
print(f"  {'ngl':>4} {'PPL':>8} {'pp128':>8} {'tg32':>8} {'PPL_s':>6} {'Bench_s':>7}")
print(f"  {'-'*41}")
for r in results:
    ppl_s = f"{r['ppl']:.4f}" if r['ppl'] else "FAIL"
    pp_s = f"{r['pp128']:.1f}" if r['pp128'] else "FAIL"
    tg_s = f"{r['tg32']:.1f}" if r['tg32'] else "FAIL"
    print(f"  {r['ngl']:>4} {ppl_s:>8} {pp_s:>8} {tg_s:>8} {r['ppl_time']:>5}s {r['bench_time']:>6}s")

with open(os.path.join(BASE, "models", "35b_exp", "ngl_ppl_sweep.json"), 'w') as f:
    json.dump(results, f, indent=2)
