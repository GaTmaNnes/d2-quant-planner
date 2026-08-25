#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 KV KLD TEST — mesurer la divergence KL entre f16 KV et q4_0 KV
================================================================

Principe :
  1. Lance llama-server avec KV f16, génère des tokens avec logprobs
  2. Lance llama-server avec KV q4_0, génère les mêmes tokens avec logprobs
  3. Calcule la KLD entre les deux distributions

Utilise l'API /completion avec n_probs pour obtenir les logprobs par token.

Usage:
  python d2_kv_kld_test.py --model models/Qwen3.8-27B-D2-ECO.gguf
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_SERVER = os.path.join(HERE, "llama-server.exe")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "La mécanique quantique décrit",
    "In machine learning, the attention mechanism",
    "Le cache KV d'un modèle transformer",
    "Write a Python function to sort a list",
    "The difference between Q2_K and Q4_K quantization",
    "Expliquer en une phrase ce qu'est un state space model",
    "A binary search tree has the property that",
    "Pour optimiser la bande passante mémoire d'un GPU",
]


def start_server(model, kv_type_k, kv_type_v, port, ngl=33, ctx=2048):
    """Start llama-server with specified KV cache type"""
    cmd = [
        ROOT_SERVER,
        "-m", model,
        "-ngl", str(ngl),
        "--flash-attn", "on",
        "--cache-type-k", kv_type_k,
        "--cache-type-v", kv_type_v,
        "--ctx-size", str(ctx),
        "--parallel", "1",
        "--port", str(port),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for ready
    for i in range(60):
        time.sleep(1)
        try:
            req = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            if req.status == 200:
                return proc
        except Exception:
            pass
    print(f"  [!] Server not ready after 60s on port {port}")
    proc.kill()
    return None


def get_completion_with_logprobs(port, prompt, n_predict=32, n_probs=10):
    """Get completion with logprobs"""
    url = f"http://127.0.0.1:{port}/completion"
    data = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "stream": False,
        "n_probs": n_probs,
    }).encode()
    
    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=60)
        return json.loads(resp.read())
    except Exception as e:
        print(f"  Error: {e}")
        return None


def softmax(logits):
    """Numerically stable softmax"""
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def compute_kld(p_dist, q_dist):
    """Compute KLD(P || Q) = sum(p * log(p/q))"""
    kld = 0.0
    for p, q in zip(p_dist, q_dist):
        if p > 1e-10 and q > 1e-10:
            kld += p * math.log(p / q)
    return kld


def compute_token_kld(ref_logprobs, cand_logprobs):
    """Compute KLD for a single token given logprobs of top candidates"""
    # ref_logprobs and cand_logprobs are lists of {token, logprob}
    # Build probability distributions over the union of tokens
    ref_dict = {}
    cand_dict = {}
    
    for item in ref_logprobs:
        token = item.get("token_str", item.get("token", ""))
        logprob = item.get("logprob", -100)
        ref_dict[token] = math.exp(logprob)
    
    for item in cand_logprobs:
        token = item.get("token_str", item.get("token", ""))
        logprob = item.get("logprob", -100)
        cand_dict[token] = math.exp(logprob)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    
    # Collect all tokens
    all_tokens = set(ref_dict.keys()) | set(cand_dict.keys())
    
    kld = 0.0
    for token in all_tokens:
        p = ref_dict.get(token, eps)
        q = cand_dict.get(token, eps)
        if p > eps and q > eps:
            kld += p * math.log(p / q)
    
    return kld


def main():
    ap = argparse.ArgumentParser(description="D2 KV KLD Test")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO.gguf"))
    ap.add_argument("--ngl", type=int, default=33)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--n-probs", type=int, default=10)
    ap.add_argument("--port-ref", type=int, default=8098)
    ap.add_argument("--port-cand", type=int, default=8099)
    ap.add_argument("--output", default=os.path.join(HERE, "d2_kv_kld_report.json"))
    args = ap.parse_args()

    print("=" * 70)
    print("  D2 KV KLD TEST — f16 vs q4_0 KV Cache")
    print("=" * 70)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  ngl: {args.ngl}, ctx: {args.ctx}")
    print(f"  Prompts: {len(PROMPTS)}")
    print()

    # Step 1: Generate reference logits with f16 KV
    print("[1/3] Starting f16 KV server (reference)...")
    proc_ref = start_server(args.model, "f16", "f16", args.port_ref, args.ngl, args.ctx)
    if not proc_ref:
        sys.exit("Failed to start reference server")
    
    ref_results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"  Prompt {i+1}/{len(PROMPTS)}...")
        result = get_completion_with_logprobs(args.port_ref, prompt, args.n_predict, args.n_probs)
        if result:
            ref_results.append({"prompt": prompt, "result": result})
    
    # Stop reference server
    proc_ref.terminate()
    proc_ref.wait(timeout=5)
    time.sleep(3)

    # Step 2: Generate comparison logits with q4_0 KV
    print("\n[2/3] Starting q4_0 KV server (candidate)...")
    proc_cand = start_server(args.model, "q4_0", "q4_0", args.port_cand, args.ngl, args.ctx)
    if not proc_cand:
        sys.exit("Failed to start candidate server")
    
    cand_results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"  Prompt {i+1}/{len(PROMPTS)}...")
        result = get_completion_with_logprobs(args.port_cand, prompt, args.n_predict, args.n_probs)
        if result:
            cand_results.append({"prompt": prompt, "result": result})
    
    # Stop candidate server
    proc_cand.terminate()
    proc_cand.wait(timeout=5)

    # Step 3: Compute KLD
    print("\n[3/3] Computing KLD...")
    
    token_klds = []
    total_tokens = 0
    
    for ref, cand in zip(ref_results, cand_results):
        ref_result = ref["result"]
        cand_result = cand["result"]
        
        # Get completion data
        ref_completion = ref_result.get("completion", ref_result)
        cand_completion = cand_result.get("completion", cand_result)
        
        # Try different response formats
        ref_probs = ref_completion.get("probs", ref_result.get("probs", []))
        cand_probs = cand_completion.get("probs", cand_result.get("probs", []))
        
        if not ref_probs or not cand_probs:
            # Try tokens format
            ref_tokens = ref_completion.get("tokens", ref_result.get("tokens", []))
            cand_tokens = cand_completion.get("tokens", cand_result.get("tokens", []))
            
            if ref_probs and cand_probs:
                for rp, cp in zip(ref_probs, cand_probs):
                    kld = compute_token_kld(rp, cp)
                    token_klds.append(kld)
                    total_tokens += 1
    
    # Compute aggregate stats
    if token_klds:
        mean_kld = sum(token_klds) / len(token_klds)
        max_kld = max(token_klds)
        median_kld = sorted(token_klds)[len(token_klds) // 2]
        
        # Bits of surprise
        mean_bps = mean_kld / math.log(2)
    else:
        mean_kld = max_kld = median_kld = mean_bps = 0
    
    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  KV types compared: f16 (reference) vs q4_0 (candidate)")
    print(f"  Tokens analyzed: {total_tokens}")
    print(f"  Mean KLD:   {mean_kld:.6f} nats")
    print(f"  Max KLD:    {max_kld:.6f} nats")
    print(f"  Median KLD: {median_kld:.6f} nats")
    print(f"  Mean BPS:   {mean_bps:.6f} bits/token")
    print()
    
    # Interpretation
    if mean_kld < 0.001:
        verdict = "NEGLIGEABLE — q4_0 KV est statistiquement identique à f16"
    elif mean_kld < 0.01:
        verdict = "TRÈS FAIBLE — q4_0 KV est quasi-identique à f16"
    elif mean_kld < 0.05:
        verdict = "FAIBLE — q4_0 KV a un impact minimal sur la qualité"
    elif mean_kld < 0.1:
        verdict = "MODÉRÉ — q4_0 KV a un impact mesurable mais acceptable"
    else:
        verdict = "ÉLEVÉ — q4_0 KV dégrade significativement la qualité"
    
    print(f"  Verdict: {verdict}")
    print("=" * 70)
    
    # Save report
    report = {
        "model": os.path.basename(args.model),
        "reference_kv": "f16",
        "candidate_kv": "q4_0",
        "ngl": args.ngl,
        "ctx": args.ctx,
        "n_predict": args.n_predict,
        "n_probs": args.n_probs,
        "n_prompts": len(PROMPTS),
        "total_tokens": total_tokens,
        "mean_kld_nats": mean_kld,
        "max_kld_nats": max_kld,
        "median_kld_nats": median_kld,
        "mean_bps": mean_bps,
        "verdict": verdict,
        "token_klds": token_klds,
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Report saved to {args.output}")


if __name__ == "__main__":
    main()
