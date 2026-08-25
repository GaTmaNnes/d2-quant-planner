#!/usr/bin/env python3
"""D2 Axis #4 FINAL: Real calibration from safetensors."""
import sys, io, os, json, time
import numpy as np
import torch
from safetensors.torch import load_file

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

FP8_DIR = "C:/Users/videl/Desktop/lama 1080-5070/hf_weights_35b_fp8"
OUTDIR = "C:/Users/videl/Desktop/lama 1080-5070/models/35b_exp"
CORPUS = "C:/Users/videl/Desktop/lama 1080-5070/ppl_test_small.txt"
N_EXPERTS = 256
TOP_K = 8
N_LAYERS = 40

if __name__ == '__main__':
    print("=" * 70)
    print("  D2 CALIBRATION FINAL — Real safetensors weights")
    print("=" * 70)

    # Step 1: Load embedding
    print("\nStep 1: Loading embedding...")
    outside = load_file(os.path.join(FP8_DIR, "outside.safetensors"))
    embd = outside['model.language_model.embed_tokens.weight'].float()  # [248320, 2048]
    print(f"  Embedding: {embd.shape}")

    # Step 2: Load gate weights for ALL 40 layers
    print("\nStep 2: Loading gate weights for all layers...")
    layer_gates = {}
    for layer_idx in range(N_LAYERS):
        layer_file = os.path.join(FP8_DIR, f"layers-{layer_idx}.safetensors")
        if os.path.exists(layer_file):
            d = load_file(layer_file)
            gate_key = [k for k in d if 'mlp.gate.weight' in k and 'expert' not in k]
            if gate_key:
                layer_gates[layer_idx] = d[gate_key[0]].float()  # [256, 2048]
    
    print(f"  Loaded gates for {len(layer_gates)} layers")
    if layer_gates:
        sample_key = list(layer_gates.keys())[0]
        print(f"  Sample gate shape: {layer_gates[sample_key].shape}")

    # Step 3: Tokenize corpus
    print("\nStep 3: Tokenizing corpus...")
    with open(CORPUS, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    tokens = [b % embd.shape[0] for b in text.encode('utf-8', errors='replace')[:3000]]
    print(f"  {len(tokens)} tokens")

    # Step 4: Run routing for each layer
    print("\nStep 4: Running routing simulation...")
    expert_total = torch.zeros(N_EXPERTS, dtype=torch.int64)
    expert_per_layer = {}
    
    t0 = time.time()
    for layer_idx, gate in layer_gates.items():
        counts = torch.zeros(N_EXPERTS, dtype=torch.int64)
        for tid in tokens:
            h = embd[tid]  # [2048]
            scores = gate @ h  # [256]
            top8 = torch.argsort(scores)[-TOP_K:]
            counts[top8] += 1
        expert_per_layer[layer_idx] = counts.numpy()
        expert_total += counts
    
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Analysis
    counts = expert_total.numpy()
    total = counts.sum()
    freq = counts / total * 100
    n_active = (counts > 0).sum()
    sorted_freq = np.sort(freq)[::-1]
    cumsum = np.cumsum(sorted_freq)

    print(f"\n{'='*70}")
    print(f"  EXPERT ACTIVATION FREQUENCY (REAL weights)")
    print(f"{'='*70}")
    print(f"  Routing decisions: {total:,}")
    print(f"  Active experts: {n_active} / {N_EXPERTS}")

    print(f"\n  --- Cumulative Coverage ---")
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        c = cumsum[min(n-1, len(cumsum)-1)]
        bar = '#' * int(c / 2)
        print(f"  Top-{n:>3}: {c:>6.1f}% {bar}")

    hot = (freq > 0.5).sum()
    warm = ((freq > 0.01) & (freq <= 0.5)).sum()
    cold = (freq <= 0.01).sum()
    never = (freq == 0).sum()

    print(f"\n  --- Classification ---")
    print(f"  Hot   (>0.5%):   {hot:>3} ({hot/N_EXPERTS*100:.1f}%)")
    print(f"  Warm  (0.01-0.5%): {warm:>3} ({warm/N_EXPERTS*100:.1f}%)")
    print(f"  Cold  (<0.01%):  {cold:>3} ({cold/N_EXPERTS*100:.1f}%)")
    print(f"  Never (0%):      {never:>3} ({never/N_EXPERTS*100:.1f}%)")

    # Quantization plan
    EXPERT_MB = 2048 * 512 * 4.5 / 8 / 1e6
    PAIR = EXPERT_MB * 2

    print(f"\n  --- Optimal Quantization Plan ---")
    gb_hot = hot * PAIR / 1024
    gb_warm = warm * PAIR * 3.44/4.5 / 1024
    gb_cold = cold * PAIR * 2.94/4.5 / 1024
    print(f"  Hot  ({hot:>3}): IQ4_NL 4.5bpw -> {gb_hot:.2f} GB")
    print(f"  Warm ({warm:>3}): Q3_K   3.4bpw -> {gb_warm:.2f} GB")
    print(f"  Cold ({cold:>3}): Q2_K   2.9bpw -> {gb_cold:.2f} GB")
    total_gb = gb_hot + gb_warm + gb_cold
    print(f"  TOTAL: {total_gb:.2f} GB (vs 12.61 GB all IQ4)")
    print(f"  Savings: {12.61 - total_gb:.2f} GB")

    # Top experts
    print(f"\n  --- Top 20 Experts ---")
    top_idx = np.argsort(freq)[::-1][:20]
    for i, idx in enumerate(top_idx):
        print(f"  #{i+1:>2} Expert {idx:>3}: {freq[idx]:.3f}% ({counts[idx]:,} hits)")

    # Per-layer diversity
    print(f"\n  --- Per-Layer Expert Diversity ---")
    print(f"  {'Layer':>6} {'Type':<12} {'Active':>7} {'Top1%':>7} {'Entropy':>8}")
    print(f"  {'-'*40}")
    for li in sorted(expert_per_layer.keys()):
        lc = expert_per_layer[li]
        lt = lc.sum()
        lf = lc / lt * 100
        na = (lc > 0).sum()
        top1 = lf.max()
        prob = lc / lt
        prob = prob[prob > 0]
        ent = -np.sum(prob * np.log2(prob))
        norm_ent = ent / np.log2(N_EXPERTS)
        ltype = "linear" if li < 30 else "full_attn"
        print(f"  {li:>6} {ltype:<12} {na:>5}/256 {top1:>6.2f}% {norm_ent:>7.3f}")

    # Save
    output = {
        'expert_freq': counts.tolist(),
        'per_layer': {str(k): v.tolist() for k, v in expert_per_layer.items()},
        'n_active': int(n_active),
        'n_hot': int(hot), 'n_warm': int(warm), 'n_cold': int(cold), 'n_never': int(never),
        'total_gb_optimal': total_gb,
    }
    outpath = os.path.join(OUTDIR, 'd2_calibration_final.json')
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {outpath}")
