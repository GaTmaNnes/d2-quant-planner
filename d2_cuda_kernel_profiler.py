#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 CUDA KERNEL PROFILER — mesurer le temps réel par kernel via CUDA events
===========================================================================

Approche : charger le modèle via ggml/llama.cpp binding (si dispo) ou
mesurer le temps total par token et estimer la contribution par layer
via des overrides sélectifs.

Alternative pragmatic : utiliser llama-server avec --override-tensor et
mesurer le temps de complétion via l'API HTTP.

Usage:
  python d2_cuda_kernel_profiler.py
  python d2_cuda_kernel_profiler.py --method server
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO.gguf")
SERVER_EXE = os.path.join(HERE, "llama-server.exe")
PORT = 8099
NGL = 30
CTX = 2048


def start_server(model, ngl=30, overrides=None, port=PORT):
    """Démarre llama-server avec des overrides optionnels."""
    cmd = [
        SERVER_EXE,
        "-m", model,
        "-ngl", str(ngl),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(CTX),
        "--parallel", "1",
    ]
    if overrides:
        for ov in overrides:
            cmd += ["-ot", ov]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8", errors="replace"
    )

    # Attendre que le serveur soit prêt
    for _ in range(30):
        time.sleep(1)
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            if resp.status == 200:
                return proc
        except Exception:
            pass

    proc.kill()
    return None


def stop_server(proc):
    """Arrête le serveur."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def measure_completion(prompt, n_predict=32, port=PORT):
    """Mesure le temps de complétion via l'API HTTP."""
    url = f"http://127.0.0.1:{port}/completion"
    data = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "stream": False,
    }).encode("utf-8")

    start = time.time()
    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - start

        tokens = result.get("tokens_evaluated", 0)
        speed = result.get("timings", {}).get("predicted_per_second", 0)
        prompt_ns = result.get("timings", {}).get("prompt_n", 0)
        predict_ns = result.get("timings", {}).get("predicted_n", 0)
        prompt_per_s = result.get("timings", {}).get("prompt_per_second", 0)

        return {
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": prompt_ns,
            "predict_tokens": predict_ns,
            "prompt_tps": round(prompt_per_s, 1) if prompt_per_s else None,
            "predict_tps": round(speed, 1) if speed else None,
        }
    except Exception as e:
        return {"error": str(e)}


def profile_baseline(model, ngl):
    """Mesure la baseline sans override."""
    print(f"[+] Baseline : {model} ngl={ngl}")
    proc = start_server(model, ngl)
    if not proc:
        print("    [!] Serveur impossible à démarrer")
        return None

    result = measure_completion("Explain quantum computing in 3 sentences.")
    stop_server(proc)
    time.sleep(2)

    if "error" in result:
        print(f"    [!] Erreur : {result['error']}")
        return None

    tps = result.get("predict_tps")
    print(f"    baseline predict_tps = {tps} t/s")
    return result


def profile_tensor_overrides(model, ngl, baseline, n_layers=64, step=4):
    """Override les tensors layer par layer et mesure l'impact."""
    results = []

    # Patterns de tensor par type
    tensor_groups = {
        "ffn": ["ffn_down", "ffn_up", "ffn_gate"],
        "attn": ["attn_q", "attn_k", "attn_v", "attn_o"],
        "all": ["ffn_down", "ffn_up", "ffn_gate", "attn_q", "attn_k", "attn_v", "attn_o"],
    }

    base_tps = baseline.get("predict_tps")
    if not base_tps:
        print("[!] Baseline TPS non disponible")
        return results

    for layer in range(0, n_layers, step):
        layer_result = {"layer": layer, "overrides": {}}

        for group_name, tensors in tensor_groups.items():
            # Construire les overrides pour tous les tensors de ce groupe dans ce layer
            overrides = []
            for tname in tensors:
                pattern = f"blk.{layer}.{tname}.weight"
                overrides.append(f"{pattern}=F16")

            print(f"    L{layer:02d} {group_name:4s} -> F16 ...", end=" ", flush=True)

            proc = start_server(model, ngl, overrides=overrides)
            if not proc:
                print("FAILED")
                layer_result["overrides"][group_name] = {"error": "server_start_failed"}
                continue

            result = measure_completion("Explain quantum computing in 3 sentences.")
            stop_server(proc)
            time.sleep(2)

            if "error" in result:
                print(f"ERROR: {result['error'][:50]}")
                layer_result["overrides"][group_name] = {"error": result["error"][:100]}
            else:
                new_tps = result.get("predict_tps")
                if new_tps and base_tps:
                    delta_pct = ((new_tps / base_tps) - 1) * 100
                    print(f"{new_tps:.1f} t/s ({delta_pct:+.1f}%)")
                    layer_result["overrides"][group_name] = {
                        "predict_tps": new_tps,
                        "delta_pct": round(delta_pct, 1),
                    }
                else:
                    print(f"{new_tps} t/s")
                    layer_result["overrides"][group_name] = {"predict_tps": new_tps}

        results.append(layer_result)

    return results


def analyze_slow_layers(results, baseline_tps):
    """Identifie les slow layers basés sur le gain de vitesse en passant à F16."""
    slow_layers = []

    for r in results:
        layer = r["layer"]
        for group, data in r.get("overrides", {}).items():
            if "error" in data or "delta_pct" not in data:
                continue
            gain = data["delta_pct"]
            if gain > 2:  # >2% de gain en F16 = ce layer était limité par la quant
                slow_layers.append({
                    "layer": layer,
                    "tensor_group": group,
                    "gain_f16_pct": gain,
                    "baseline_tps": baseline_tps,
                    "f16_tps": data.get("predict_tps"),
                })

    slow_layers.sort(key=lambda x: -x["gain_f16_pct"])
    return slow_layers


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 CUDA KERNEL PROFILER")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ngl", type=int, default=NGL)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--json", default=os.path.join(HERE, "d2_kernel_profile.json"))
    args = ap.parse_args()

    print(f"[+] D2 CUDA Kernel Profiler")
    print(f"[+] Modèle : {args.model}")
    print(f"[+] ngl={args.ngl}, step={args.step}")
    print()

    # 1. Baseline
    baseline = profile_baseline(args.model, args.ngl)
    if not baseline:
        print("[!] Baseline échouée. Abort.")
        sys.exit(1)

    # 2. Scan des overrides
    print(f"\n[+] Scan tensor overrides (step={args.step})...")
    results = profile_tensor_overrides(args.model, args.ngl, baseline, step=args.step)

    # 3. Analyser les slow layers
    slow_layers = analyze_slow_layers(results, baseline.get("predict_tps"))
    print(f"\n[+] Slow layers identifiés : {len(slow_layers)}")
    for s in slow_layers[:10]:
        print(f"    L{s['layer']:02d} {s['tensor_group']:5s} : +{s['gain_f16_pct']:.1f}% avec F16")

    # 4. Sauvegarder
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "ngl": args.ngl,
        "baseline": baseline,
        "layer_results": results,
        "slow_layers": slow_layers,
    }

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Rapport : {args.json}")


if __name__ == "__main__":
    main()
