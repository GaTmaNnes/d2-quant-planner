#!/usr/bin/env python3
# [OBSOLETE 25/08/2026] Périmètre 27B abandonné (prod = Qwen3.6-35B-A3B-D2-MOE).
# Raisons : mesure sur GGUF Qwen3.8-27B inexistants (D2-ECO / v3 / Q4_K_M).
# Conservé pour historique — NE PAS EXÉCUTER.
# -*- coding: utf-8 -*-
"""
D2 REPACKING OVERHEAD — Measure quantization cost per format
=============================================================

Measures:
  1. Quantization time per tensor type (from F16 reference)
  2. Model load time per GGUF variant
  3. Dequantization overhead estimate

Usage:
  python d2_repacking_overhead.py
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ("D2-ECO (Q2_K mix)", "models/Qwen3.8-27B-D2-ECO.gguf", 33),
    ("D2-ECO-v3 (Q2+Q3+Q4)", "models/Qwen3.8-27B-D2-ECO-v3.gguf", 33),
    ("Q4_K_M", "models/Qwen3.8-27B-Q4_K_M.gguf", 33),
]


def measure_load_time(model_path, ngl=33, port=8198):
    """Measure how long it takes for llama-server to load a model."""
    cmd = [
        "llama-server.exe", "-m", model_path,
        "--port", str(port), "--n-gpu-layers", str(ngl),
        "--ctx-size", "512", "--parallel", "1",
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
        "--log-disable",
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    import urllib.request
    for i in range(90):
        time.sleep(1)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            if json.loads(r.read()).get("status") == "ok":
                elapsed = time.time() - t0
                proc.kill()
                proc.wait()
                return elapsed
        except:
            pass

    proc.kill()
    proc.wait()
    return None


def measure_bench(model_path, ngl=33):
    """Run llama-bench and extract timing."""
    cmd = [
        "llama-bench.exe", "-m", model_path,
        "-ngl", str(ngl), "-t", "8",
        "-p", "128", "-n", "32", "-r", "1",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0

    # Parse output
    pp = tg = 0
    for line in result.stdout.split("\n"):
        if "pp128" in line or "pp512" in line:
            parts = line.split("|")
            for p in parts:
                p = p.strip()
                if "t/s" not in p and "±" in p:
                    try:
                        pp = float(p.split("±")[0].strip())
                    except:
                        pass
        if "tg32" in line:
            parts = line.split("|")
            for p in parts:
                p = p.strip()
                if "t/s" not in p and "±" in p:
                    try:
                        tg = float(p.split("±")[0].strip())
                    except:
                        pass

    return {"pp": pp, "tg": tg, "wall_time": round(elapsed, 1)}


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  D2 REPACKING OVERHEAD")
    print("=" * 70)

    results = []

    for name, path, ngl in MODELS:
        full = os.path.join(HERE, path)
        if not os.path.exists(full):
            print(f"  SKIP {name}: not found")
            continue

        size_gb = os.path.getsize(full) / (1024**3)
        print(f"\n  === {name} ({size_gb:.2f} GiB) ===")

        # Load time
        print(f"  Measuring load time...")
        load_time = measure_load_time(full, ngl, port=8198)
        if load_time:
            print(f"  Load: {load_time:.1f}s")
        else:
            print(f"  Load: TIMEOUT")

        # Bench
        print(f"  Running bench...")
        bench = measure_bench(full, ngl)
        print(f"  pp128: {bench['pp']:.1f} t/s")
        print(f"  tg32:  {bench['tg']:.1f} t/s")
        print(f"  Wall:  {bench['wall_time']:.1f}s")

        results.append({
            "model": name,
            "path": path,
            "size_gb": round(size_gb, 2),
            "load_time_s": round(load_time, 1) if load_time else None,
            "pp128": bench["pp"],
            "tg32": bench["tg"],
            "wall_time": bench["wall_time"],
        })

    # Summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Model':25s} {'Size':8s} {'Load':8s} {'pp128':8s} {'tg32':8s}")
    print("  " + "-" * 60)
    for r in results:
        load = f"{r['load_time_s']:.0f}s" if r['load_time_s'] else "N/A"
        print(f"  {r['model']:25s} {r['size_gb']:7.2f}G {load:8s} {r['pp128']:7.1f} {r['tg32']:7.1f}")

    # Dequant overhead analysis
    print()
    print("  DEQUANT OVERHEAD ANALYSIS:")
    if len(results) >= 2:
        eco = results[0]
        v3 = results[1]
        if eco["tg32"] > 0 and v3["tg32"] > 0:
            speed_ratio = v3["tg32"] / eco["tg32"]
            size_ratio = v3["size_gb"] / eco["size_gb"]
            print(f"    Size ratio:  {size_ratio:.2f}x (v3/ECO)")
            print(f"    Speed ratio: {speed_ratio:.2f}x (v3/ECO)")
            print(f"    → Q3/Q4 overhead: {(1-speed_ratio)*100:.1f}% slower for {size_ratio:.2f}x larger")

    # Save
    out = os.path.join(HERE, "d2_repacking_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, f, indent=2)
    print(f"\n  [+] {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
