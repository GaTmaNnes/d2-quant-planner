#!/usr/bin/env python3
"""D2 Axis #4: Real calibration from FP8 safetensors using torch."""
import sys, io, os, json, time
import numpy as np
import torch
from safetensors.torch import load_file
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

FP8_DIR = "C:/Users/videl/Desktop/lama 1080-5070/hf_weights_35b_fp8"
OUTDIR = "C:/Users/videl/Desktop/lama 1080-5070/models/35b_exp"
CORPUS = "C:/Users/videl/Desktop/lama 1080-5070/ppl_test_small.txt"

N_EXPERTS = 256
TOP_K = 8


def softmax(x):
    e = torch.exp(x - x.max())
    return e / e.sum()


if __name__ == '__main__':
    print("=" * 70)
    print("  D2 CALIBRATION — REAL weights from safetensors (torch)")
    print("=" * 70)

    # Step 1: Load embeddings from outside.safetensors
    print("\nStep 1: Loading embeddings...")
    outside = load_file(os.path.join(FP8_DIR, "outside.safetensors"))
    for name, tensor in outside.items():
        print(f"  {name}: {tensor.shape} {tensor.dtype}")
    
    embd_key = [k for k in outside if 'token_embd' in k][0]
    embd = outside[embd_key].float()  # [vocab, hidden] or [hidden, vocab]
    print(f"  Embedding: {embd.shape}")
    
    # Step 2: Load gate projection from layer 0
    print("\nStep 2: Loading gate projection from layer 0...")
    layer0 = load_file(os.path.join(FP8_DIR, "layers-0.safetensors"))
    gate_keys = [k for k in layer0 if 'gate_inp' in k]
    print(f"  Gate keys: {gate_keys}")
    
    if gate_keys:
        gate_proj = layer0[gate_keys[0]].float()
        print(f"  Gate projection: {gate_proj.shape}")
    else:
        print("  ERROR: no gate_inp found!")
        sys.exit(1)
    
    # Step 3: Tokenize corpus
    print("\nStep 3: Tokenizing...")
    with open(CORPUS, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Simple byte-level tokenization
    tokens = [b for b in text.encode('utf-8', errors='replace')[:3000]]
    # Remap to vocab range (248320)
    tokens = [t % embd.shape[0] for t in tokens]
    print(f"  {len(tokens)} tokens from {len(text)} chars")
    
    # Step 4: Run routing simulation
    print("\nStep 4: Routing simulation with REAL weights...")
    
    # Determine shape convention
    # embd: [vocab, hidden] — select row per token
    # gate: [hidden, n_experts] or [n_experts, hidden]
    
    if embd.shape[1] == gate_proj.shape[0]:
        # embd[vocab, hidden] @ gate[hidden, experts] → [vocab, experts]
        W = gate_proj  # [hidden, experts]
    elif embd.shape[1] == gate_proj.shape[1]:
        W = gate_proj.T  # transpose to [hidden, experts]
    else:
        # Try matching dimensions
        for i in [0, 1]:
            for j in [0, 1]:
                if embd.shape[i] == gate_proj.shape[j]:
                    print(f"  Match: embd dim {i} ({embd.shape[i]}) = gate dim {j} ({gate_proj.shape[j]})")
    
    print(f"  embd: {embd.shape}, gate_W: {W.shape}")
    
    expert_counts = torch.zeros(N_EXPERTS, dtype=torch.int64)
    
    t0 = time.time()
    for tid in tokens:
        h = embd[tid]  # [hidden]
        scores = W.T @ h  # [n_experts]
        top8 = torch.argsort(scores)[-TOP_K:]
        expert_counts[top8] += 1
    
    elapsed = time.time() - t0
    counts = expert_counts.numpy()
    total = counts.sum()
    freq = counts / total * 100
    
    print(f"  Done in {elapsed:.1f}s")
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"  RESULTS — Expert Activation Frequency")
    print(f"{'='*70}")
    
    n_active = (counts > 0).sum()
    sorted_freq = np.sort(freq)[::-1]
    cumsum = np.cumsum(sorted_freq)
    
    print(f"\n  Routing decisions: {total:,}")
    print(f"  Active experts: {n_active} / {N_EXPERTS}")
    
    print(f"\n  --- Cumulative Coverage ---")
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        c = cumsum[min(n-1, len(cumsum)-1)]
        bar = '#' * int(c / 2)
        print(f"  Top-{n:>3}: {c:>6.1f}% {bar}")
    
    hot = (freq > 0.5).sum()
    warm = ((freq > 0.01) & (freq <= 0.5)).sum()
    cold = (freq <= 0.01).sum()
    
    print(f"\n  --- Classification ---")
    print(f"  Hot   (>0.5%):   {hot:>3} ({hot/N_EXPERTS*100:.1f}%)")
    print(f"  Warm  (0.01-0.5%): {warm:>3} ({warm/N_EXPERTS*100:.1f}%)")
    print(f"  Cold  (<0.01%):  {cold:>3} ({cold/N_EXPERTS*100:.1f}%)")
    
    # Quantization plan
    EXPERT_MB = 2048 * 512 * 4.5 / 8 / 1e6
    PAIR = EXPERT_MB * 2
    
    print(f"\n  --- Optimal Quantization Plan ---")
    gb_hot = hot * PAIR / 1024
    gb_warm = warm * PAIR * 3.44/4.5 / 1024
    gb_cold = cold * PAIR * 2.94/4.5 / 1024
    print(f"  Hot  ({hot:>3}): IQ4_NL 4.5bpw → {gb_hot:.2f} GB")
    print(f"  Warm ({warm:>3}): Q3_K   3.4bpw → {gb_warm:.2f} GB")
    print(f"  Cold ({cold:>3}): Q2_K   2.9bpw → {gb_cold:.2f} GB")
    print(f"  TOTAL: {gb_hot+gb_warm+gb_cold:.2f} GB (vs 12.61 GB all IQ4)")
    print(f"  Savings: {12.61 - (gb_hot+gb_warm+gb_cold):.2f} GB")
    
    # Top experts
    print(f"\n  --- Top 15 Experts ---")
    top_idx = np.argsort(freq)[::-1][:15]
    for i, idx in enumerate(top_idx):
        print(f"  #{i+1:>2} Expert {idx:>3}: {freq[idx]:.3f}% ({counts[idx]:,} hits)")
    
    # Save
    output = {
        'expert_freq': counts.tolist(),
        'n_active': int(n_active),
        'n_hot': int(hot), 'n_warm': int(warm), 'n_cold': int(cold),
        'total_gb_optimal': gb_hot+gb_warm+gb_cold,
    }
    outpath = os.path.join(OUTDIR, 'd2_calibration_real.json')
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {outpath}")
