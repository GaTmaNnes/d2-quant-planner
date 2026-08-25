#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 PPL PROXY — Compare quality D2-ECO vs Q4_K_M
=================================================

Usage:
  python d2_ppl_proxy.py
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPTS = [
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Write a Python function that sorts a list:",
    "What is the meaning of life? According to philosophers",
    "The speed of light is approximately",
]

MODELS = [
    ("D2-ECO (Q2_K)", "models/Qwen3.8-27B-D2-ECO.gguf", 33),
    ("Q4_K_M", "models/Qwen3.8-27B-Q4_K_M.gguf", 33),
]


def start_server(model_path, port, ngl, ctx=512):
    cmd = [
        "llama-server.exe", "-m", model_path,
        "--port", str(port), "--n-gpu-layers", str(ngl),
        "--ctx-size", str(ctx), "--parallel", "1",
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
        "--log-disable",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(60):
        time.sleep(1)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            if json.loads(r.read()).get("status") == "ok":
                return proc
        except:
            pass
    proc.kill()
    return None


def generate(prompt, port, max_tokens=100):
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        return data.get("content", "")
    except Exception as e:
        return f"ERROR: {e}"


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  D2 PPL PROXY — Quality Comparison")
    print("=" * 70)

    all_results = []

    for name, path, ngl in MODELS:
        full = os.path.join(HERE, path)
        if not os.path.exists(full):
            print(f"  SKIP {name}: not found")
            continue

        port = 8199
        print(f"\n  === {name} (ngl={ngl}) ===")
        print(f"  Starting server...")
        proc = start_server(full, port, ngl)
        if not proc:
            print(f"  FAILED")
            continue

        try:
            model_results = []
            for pi, prompt in enumerate(PROMPTS):
                t0 = time.time()
                output = generate(prompt, port, max_tokens=100)
                elapsed = time.time() - t0
                print(f"  [{pi+1}/{len(PROMPTS)}] {prompt[:40]:40s} -> {output[:60].replace(chr(10), ' ')}... ({elapsed:.1f}s)")
                model_results.append({
                    "prompt": prompt,
                    "output": output,
                    "time_s": round(elapsed, 2),
                })

            all_results.append({
                "model": name,
                "path": path,
                "ngl": ngl,
                "results": model_results,
            })
        finally:
            proc.kill()
            proc.wait()
            time.sleep(2)

    # Side-by-side
    print()
    print("=" * 70)
    print("  SIDE-BY-SIDE")
    print("=" * 70)

    if len(all_results) >= 2:
        for pi, prompt in enumerate(PROMPTS):
            print(f"\n  Prompt: {prompt}")
            for r in all_results:
                if pi < len(r["results"]):
                    out = r["results"][pi]["output"][:200].replace("\n", " ")
                    t = r["results"][pi]["time_s"]
                    print(f"    {r['model']:20s} ({t:.1f}s): {out}...")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }
    out_path = os.path.join(HERE, "d2_ppl_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  [+] {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
