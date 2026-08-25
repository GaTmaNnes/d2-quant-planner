#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPL sweep: D2-ECO sur ngl 28/30/32/35 × f16/q4_0, 50 chunks, ctx 1024.
~8 configs × ~3.5 min = ~28 min. Sauvegarde incrémentale."""
import json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO.gguf")
PPL_EXE = os.path.join(HERE, "llama-perplexity.exe")
CORPUS = os.path.join(HERE, "corpus", "wiki.test.raw")
OUT = os.path.join(HERE, "d2_ppl_ngl_sweep.json")
CHUNKS = 50; CTX = 1024; TIMEOUT = 900
NGL = [28, 30, 32, 35]
KV = [("f16", None), ("q4_0", "q4_0")]

FINAL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")

def run_one(ngl, kvt):
    cmd = [PPL_EXE, "-m", MODEL, "-f", CORPUS, "-c", str(CTX),
           "-ngl", str(ngl), "-fa", "on", "--chunks", str(CHUNKS)]
    if kvt: cmd += ["-ctk", kvt, "-ctv", kvt]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    wall = time.time() - t0
    out = p.stdout + p.stderr
    if p.returncode != 0:
        return {"error": f"exit={p.returncode}", "tail": out[-800:]}
    m = FINAL_RE.search(out)
    if not m:
        return {"error": "parse", "tail": out[-800:]}
    return {"ppl": float(m.group(1)), "ppl_stderr": float(m.group(2)), "wall_s": round(wall, 0)}

# charger existant (reprise)
results = {}
if os.path.isfile(OUT):
    with open(OUT, encoding="utf-8") as f:
        results = json.load(f)

total = len(NGL) * len(KV); done = 0
for ngl in NGL:
    for kvl, kvt in KV:
        done += 1
        key = f"{ngl}|{kvl}"
        if key in results:
            print(f"[{done}/{total}] SKIP ngl={ngl} kv={kvl}"); continue
        print(f"[{done}/{total}] ngl={ngl} kv={kvl} ...", flush=True)
        r = run_one(ngl, kvt)
        r["ngl"] = ngl; r["kv"] = kvl
        results[key] = r
        if "error" in r:
            print(f"  [!] {r['error']}")
        else:
            print(f"  PPL={r['ppl']:.4f} ± {r['ppl_stderr']:.4f} ({r['wall_s']:.0f}s)")
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n[+] {OUT}")
for k, v in sorted(results.items()):
    if "error" in v: print(f"  {k:>12}  ERR: {v['error']}")
    else: print(f"  {k:>12}  PPL={v['ppl']:.4f} ± {v['ppl_stderr']:.4f}  ({v['wall_s']:.0f}s)")