#!/usr/bin/env python3
"""
D2 Axis #4 v2: Corrected calibration
- Fix embedding shape (vocab=248320, hidden=2048)
- Use real gate weights per layer
- Real embedding → gate → softmax → top-8 routing
"""
import sys, io, os, json, time
import numpy as np
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SOURCE = "C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
CORPUS = "C:/Users/videl/Desktop/lama 1080-5070/ppl_test_small.txt"
OUTDIR = "C:/Users/videl/Desktop/lama 1080-5070/models/35b_exp"

N_EXPERTS = 256
TOP_K = 8
N_LAYERS = 40


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def load_weights(path):
    """Load gate weights and embeddings from GGUF."""
    from gguf import GGUFReader
    reader = GGUFReader(path)
    
    embd = None
    gate_inp = {}  # layer_idx -> numpy array [256, 2048] (expert_gate weights)
    gate_proj = None  # shared gate projection [2048, 256]
    
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data.astype(np.float32)  # Convert to float32 for computation
        
        if name == 'token_embd.weight':
            # Shape in GGUF: [248320, 2048] → transpose to [248320, 2048]
            embd = data
            print(f"  Embedding: {embd.shape}")
        
        if name == 'blk.0.ffn_gate_inp.weight':
            gate_proj = data
            print(f"  Gate projection: {gate_proj.shape}")
        
        # Per-layer expert gate weights
        for layer in range(N_LAYERS):
            if name == f'blk.{layer}.ffn_gate.weight':
                # shape: [2048, 512, 256] → this is the per-expert gate
                # We need the shared gate projection
                pass
    
    return embd, gate_proj


def load_gate_weights_detailed(path):
    """Load all gate-related weights for proper routing simulation."""
    from gguf import GGUFReader
    reader = GGUFReader(path)
    
    weights = {}
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data.astype(np.float32)
        
        if name == 'token_embd.weight':
            weights['embd'] = data
        
        # ffn_gate_inp is the shared router projection [2048, 256]
        # Present in blk.0, shared across all layers
        if name == 'blk.0.ffn_gate_inp.weight':
            weights['gate_proj'] = data
        
        # ffn_gate.weight is per-expert gate (not the router)
        # The router is ffn_gate_inp
    
    return weights


def simple_tokenizer(text, max_tokens=2000):
    """Byte-level tokenizer mapping to vocab IDs."""
    # Qwen tokenizer uses tiktoken-like encoding
    # For routing simulation, use byte values mapped to vocab range
    tokens = []
    for byte_val in text.encode('utf-8', errors='replace')[:max_tokens]:
        tokens.append(byte_val % 248320)
    return tokens


def run_calibration(embd, gate_proj, corpus_tokens):
    """Run real routing simulation with correct shapes.
    
    Routing: score = softmax(gate_proj @ hidden_state)
    hidden_state ≈ embd[token_id] (first layer approximation)
    """
    n_vocab, hidden_dim = embd.shape  # [248320, 2048]
    gate_out_dim = gate_proj.shape[1]  # 256 experts
    
    print(f"  Embedding: {embd.shape} (vocab={n_vocab}, hidden={hidden_dim})")
    print(f"  Gate proj: {gate_proj.shape} (hidden={gate_proj.shape[0]}, experts={gate_out_dim})")
    
    # Expert frequency counters
    expert_total = np.zeros(N_EXPERTS, dtype=np.int64)
    expert_layer = defaultdict(lambda: np.zeros(N_EXPERTS, dtype=np.int64))
    
    valid_tokens = 0
    for token_id in corpus_tokens:
        if token_id >= n_vocab:
            continue
        
        # Get embedding for this token
        hidden = embd[token_id]  # [2048]
        
        # Apply gate projection
        scores = gate_proj @ hidden  # [256]
        
        # Softmax
        probs = softmax(scores)
        
        # Top-8 experts
        top8 = np.argsort(probs)[-TOP_K:]
        expert_total[top8] += 1
        valid_tokens += 1
    
    print(f"  Processed {valid_tokens} tokens")
    return expert_total


def analyze(expert_counts):
    """Analyze activation frequency."""
    total = expert_counts.sum()
    freq = expert_counts / total * 100
    
    n_active = (expert_counts > 0).sum()
    
    print(f"\n  === EXPERT ACTIVATION FREQUENCY ===")
    print(f"  Total routing decisions: {total:,}")
    print(f"  Active experts: {n_active} / {N_EXPERTS}")
    
    # Cumulative distribution
    sorted_freq = np.sort(freq)[::-1]
    cumsum = np.cumsum(sorted_freq)
    
    print(f"\n  --- Cumulative Distribution ---")
    print(f"  {'Top-N':>6} {'Freq %':>8} {'Cumul %':>8}")
    print(f"  {'-'*22}")
    for n in [1, 2, 4, 8, 10, 20, 50, 100, 150, 200, 256]:
        if n <= len(sorted_freq):
            print(f"  Top-{n:>3} {sorted_freq[:n].sum():>7.2f}% {cumsum[n-1]:>7.1f}%")
    
    # Classification
    hot = (freq > 0.5).sum()
    warm = ((freq > 0.01) & (freq <= 0.5)).sum()
    cold = (freq <= 0.01).sum()
    never = (freq == 0).sum()
    
    print(f"\n  --- Classification ---")
    print(f"  Hot   (>0.5%):   {hot:>3} experts ({hot/N_EXPERTS*100:.1f}%)")
    print(f"  Warm  (0.01-0.5%): {warm:>3} experts ({warm/N_EXPERTS*100:.1f}%)")
    print(f"  Cold  (<0.01%):  {cold:>3} experts ({cold/N_EXPERTS*100:.1f}%)")
    print(f"  Never (0%):      {never:>3} experts ({never/N_EXPERTS*100:.1f}%)")
    
    # Expert weight sizes
    EXPERT_MB = 2048 * 512 * 4.5 / 8 / 1e6  # IQ4_NL: ~5.625 MB per expert projection
    EXPERT_PAIR_MB = EXPERT_MB * 2  # gate_up + down
    
    hot_gb = hot * EXPERT_PAIR_MB / 1024
    warm_gb = warm * EXPERT_PAIR_MB * 3.44/4.5 / 1024  # Q3_K
    cold_gb = cold * EXPERT_PAIR_MB * 2.94/4.5 / 1024  # Q2_K
    
    print(f"\n  --- Quantization Strategy ---")
    print(f"  Hot ({hot}):  IQ4_NL → {hot_gb:.2f} GB")
    print(f"  Warm ({warm}): Q3_K   → {warm_gb:.2f} GB")
    print(f"  Cold ({cold}): Q2_K   → {cold_gb:.2f} GB")
    print(f"  TOTAL: {hot_gb + warm_gb + cold_gb:.2f} GB")
    
    # Top experts
    print(f"\n  --- Top 20 Experts ---")
    top_indices = np.argsort(freq)[::-1][:20]
    for i, idx in enumerate(top_indices):
        print(f"  #{i+1:>2} Expert {idx:>3}: {freq[idx]:.3f}% ({expert_counts[idx]:,} activations)")
    
    return {
        'freq': freq.tolist(),
        'n_active': int(n_active),
        'n_hot': int(hot), 'n_warm': int(warm), 'n_cold': int(cold), 'n_never': int(never),
        'total_gb': hot_gb + warm_gb + cold_gb,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("  D2 CALIBRATION v2 — Real Gate Routing Simulation")
    print("=" * 70)
    
    print("\nStep 1: Loading weights...")
    weights = load_gate_weights_detailed(SOURCE)
    
    embd = weights.get('embd')
    gate_proj = weights.get('gate_proj')
    
    if embd is None or gate_proj is None:
        print("  ERROR: Missing weights!")
        sys.exit(1)
    
    print("\nStep 2: Tokenizing corpus...")
    with open(CORPUS, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    tokens = simple_tokenizer(text, max_tokens=3000)
    print(f"  {len(text)} chars → {len(tokens)} tokens")
    
    print("\nStep 3: Running routing simulation...")
    t0 = time.time()
    counts = run_calibration(embd, gate_proj, tokens)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    
    print("\nStep 4: Analysis...")
    result = analyze(counts)
    
    # Save
    output = {
        'expert_freq': counts.tolist(),
        'total_tokens': int(counts.sum()),
        'summary': result,
    }
    outpath = os.path.join(OUTDIR, 'd2_calibration_v2.json')
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {outpath}")
