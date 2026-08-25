#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 ACCEPTANCE PROFILER — MTP + DFlash2 speculative decoding
=============================================================
Mesure pour chaque configuration speculative :
  - acceptance rate (tokens drafted / tokens accepted)
  - effective tokens/s (tokens acceptés / temps total)
  - raw tokens/s (sans speculative)
  - gain relatif
  - VRAM consommée (target + draft + KV)

Usage:
  python d2_acceptance_profiler.py --model models/Qwen3.8-27B-D2-ECO.gguf
  python d2_acceptance_profiler.py --model models/Qwen3.8-27B-Q4_K_M.gguf --draft models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

from d2_schema import RunRecord, HardwareRecord, QWEN38_27B
from d2_registry import D2Registry

HERE = os.path.dirname(os.path.abspath(__file__))

# The root binary (beellama build) supports MTP
ROOT_SERVER = os.path.join(HERE, "llama-server.exe")
# The backup binary (CUDA12 prebuilt) supports DFlash
BACKUP_SERVER = os.path.join(HERE, "_backup_cuda12_prebuilt", "llama-server.exe")

PROMPTS = [
    "Explain the difference between attention and state space models in one paragraph.",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr",
    "La mécanique quantique décrit le comportement des particules à l'échelle",
    "Write a Python function that implements a binary search tree.",
    "In the context of neural network quantization, the key insight is that",
]


def get_vram():
    """Get current VRAM usage via nvidia-smi"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        used, free = r.stdout.strip().split(", ")
        return int(used), int(free)
    except Exception:
        return -1, -1


def start_server(model, server_bin, extra_args, port=8099):
    """Start llama-server and wait for it to be ready"""
    cmd = [
        server_bin,
        "-m", model,
        "-ngl", "33",
        "--flash-attn", "on",
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--ctx-size", "4096",
        "--parallel", "1",
        "--port", str(port),
    ] + extra_args

    print(f"  Starting server: {' '.join(cmd[-10:])}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for server to be ready (max 60s)
    for i in range(60):
        time.sleep(1)
        try:
            req = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            if req.status == 200:
                print(f"  Server ready after {i+1}s")
                return proc
        except Exception:
            pass
    print("  WARNING: Server not ready after 60s")
    return proc


def send_completion(port, prompt, n_predict=128, temperature=0):
    """Send completion request and return stats"""
    url = f"http://127.0.0.1:{port}/completion"
    data = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "stream": False,
    }).encode()

    start = time.time()
    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read())
        elapsed = time.time() - start
        return result, elapsed
    except Exception as e:
        print(f"  Error: {e}")
        return None, time.time() - start


def get_metrics(port):
    """Get server metrics from /metrics"""
    try:
        req = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)
        text = req.read().decode()
        metrics = {}
        for line in text.split("\n"):
            if "spec_dec" in line or "draft" in line or "accept" in line:
                metrics[line.split("{")[0]] = line.split()[-1] if line.split() else ""
        return metrics
    except Exception:
        return {}


def profile_config(name, model, server_bin, extra_args, port=8099, n_predict=128):
    """Profile a single configuration"""
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")

    # Get baseline VRAM
    vram_before = get_vram()

    # Start server
    proc = start_server(model, server_bin, extra_args, port)
    time.sleep(3)  # Let it stabilize

    # Get VRAM with model loaded
    vram_loaded = get_vram()

    # Run completions
    results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"  Prompt {i+1}/{len(PROMPTS)}...")
        result, elapsed = send_completion(port, prompt, n_predict=n_predict)
        if result:
            results.append(result)

    # Collect metrics
    metrics = get_metrics(port)

    # Stop server
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    time.sleep(2)

    # Get VRAM after unload
    vram_after = get_vram()

    # Aggregate results
    if not results:
        print(f"  No results!")
        return {"name": name, "error": "no results"}

    # Parse predict timing from results
    total_tokens = 0
    total_predict_time = 0
    total_draft_tokens = 0
    total_accepted = 0

    for r in results:
        # tokens_eval_count = prompt tokens, tokens_predicted = generated tokens
        predicted = r.get("tokens_predicted", 0)
        total_tokens += predicted

        # Timing
        timings = r.get("timings", {})
        predict_ms = timings.get("predicted_ms", 0)
        total_predict_time += predict_ms

        # Speculative stats (if available)
        if "spec" in r:
            spec = r["spec"]
            total_draft_tokens += spec.get("drafts", 0)
            total_accepted += spec.get("accepted", 0)

    effective_tps = total_tokens / (total_predict_time / 1000) if total_predict_time > 0 else 0
    acceptance_rate = total_accepted / total_draft_tokens if total_draft_tokens > 0 else None

    # VRAM delta
    vram_model = vram_loaded[0] - vram_before[0] if vram_loaded[0] > 0 else -1

    summary = {
        "name": name,
        "total_tokens": total_tokens,
        "total_predict_time_ms": round(total_predict_time, 1),
        "effective_tps": round(effective_tps, 2),
        "draft_tokens": total_draft_tokens,
        "accepted_tokens": total_accepted,
        "acceptance_rate": round(acceptance_rate, 4) if acceptance_rate is not None else None,
        "vram_model_mib": vram_model,
        "vram_used_mib": vram_loaded[0],
        "vram_free_mib": vram_loaded[1],
        "metrics": metrics,
        "n_prompts": len(results),
    }

    print(f"\n  Results:")
    print(f"    tokens/s (effective): {summary['effective_tps']}")
    print(f"    acceptance rate:      {summary['acceptance_rate']}")
    print(f"    total tokens:         {summary['total_tokens']}")
    print(f"    VRAM used:            {summary['vram_used_mib']} MiB")
    print(f"    VRAM free:            {summary['vram_free_mib']} MiB")

    return summary


def main():
    ap = argparse.ArgumentParser(description="D2 Acceptance Profiler")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO.gguf"))
    ap.add_argument("--draft", default=None, help="Draft model for DFlash2")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--output", default=os.path.join(HERE, "d2_acceptance_report.json"))
    args = ap.parse_args()

    configs = []

    # Config 1: Baseline (no speculative)
    configs.append(("baseline", args.model, ROOT_SERVER, []))

    # Config 2: MTP (draft-mtp)
    configs.append(("MTP", args.model, ROOT_SERVER, ["--spec-type", "draft-mtp"]))

    # Config 3: DFlash2 (if draft model provided)
    if args.draft:
        configs.append(("DFlash2_Q4KM", args.model, BACKUP_SERVER, [
            "--model-draft", args.draft,
            "--spec-type", "dflash",
            "--spec-draft-ngl", "99",
            "--spec-draft-n-max", "5",
            "--spec-draft-ctx-size", "512",
        ]))

    # Config 4: N-gram (no draft model needed)
    configs.append(("ngram", args.model, ROOT_SERVER, ["--spec-type", "ngram-simple"]))

    all_results = []
    for name, model, server, extra in configs:
        result = profile_config(name, model, server, extra, port=args.port, n_predict=args.n_predict)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    baseline_tps = None
    for r in all_results:
        if r["name"] == "baseline":
            baseline_tps = r.get("effective_tps", 0)

    for r in all_results:
        tps = r.get("effective_tps", 0)
        gain = ((tps / baseline_tps - 1) * 100) if baseline_tps and baseline_tps > 0 else 0
        ar = r.get("acceptance_rate")
        ar_str = f"{ar*100:.1f}%" if ar is not None else "N/A"
        print(f"  {r['name']:20s}  {tps:8.2f} t/s  gain={gain:+6.1f}%  acceptance={ar_str}  VRAM={r.get('vram_used_mib', '?')} MiB")

    # Save report
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to {args.output}")
    
    # Also save to registry
    registry_path = os.path.join(HERE, "d2_ecosystem", "registry.json")
    reg = D2Registry(model=QWEN38_27B)
    if os.path.exists(registry_path):
        reg.load(registry_path)
    
    for r in all_results:
        if "error" in r:
            continue
        
        # Parse spec_type from config name
        spec = "none"
        if "MTP" in r["name"]:
            spec = "draft-mtp"
        elif "DFlash" in r["name"]:
            spec = "dflash"
        elif "ngram" in r["name"]:
            spec = "ngram-simple"
        
        run = RunRecord(
            model_name=os.path.basename(args.model),
            ngl=33,
            ctx_size=4096,
            parallel=1,
            kv_type_k="q4_0",
            kv_type_v="q4_0",
            spec_type=spec,
            draft_model=args.draft or "",
            n_draft=r.get("draft_tokens", 0),
            acceptance_rate=r.get("acceptance_rate", 0),
            draft_tokens=r.get("draft_tokens", 0),
            accepted_tokens=r.get("accepted_tokens", 0),
            tg_tokens_per_sec=r.get("effective_tps", 0),
            effective_tokens_per_sec=r.get("effective_tps", 0),
            total_tokens=r.get("total_tokens", 0),
            total_predict_time_ms=r.get("total_predict_time_ms", 0),
            vram_used_mib=r.get("vram_used_mib", 0),
            vram_free_mib=r.get("vram_free_mib", 0),
        )
        reg.add_run(run)
    
    reg.save(registry_path)
    print(f"[+] Registry updated: {registry_path}")


if __name__ == "__main__":
    main()
