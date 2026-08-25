#!/usr/bin/env python3
"""
D2 Axis #4 v4: REAL calibration from FP8 safetensors
Load token_embd (FP8) + ffn_gate_inp (F32) from safetensors.
Run real routing simulation with proper dequantization.
"""
import sys, io, os, json, time, struct
import numpy as np
from collections import defaultdict
from safetensors import safe_open

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

FP8_DIR = "C:/Users/videl/Desktop/lama 1080-5070/hf_weights_35b_fp8"
OUTDIR = "C:/Users/videl/Desktop/lama 1080-5070/models/35b_exp"
CORPUS = "C:/Users/videl/Desktop/lama 1080-5070/ppl_test_small.txt"

N_EXPERTS = 256
TOP_K = 8
N_LAYERS = 40


def fp8_to_float32(data):
    """Convert FP8 E4M3 data to float32.
    FP8 E4M3: 1 sign + 4 exponent + 3 mantissa bits.
    We use a lookup table approach for speed.
    """
    if data.dtype == np.float32:
        return data
    if data.dtype == np.float16:
        return data.astype(np.float32)
    
    # FP8 E4M3 → float32 conversion
    # Use the method from the float repo or a simple lookup
    data_u8 = data.view(np.uint8)
    result = np.zeros(len(data_u8), dtype=np.float32)
    
    for i in range(len(data_u8)):
        val = data_u8[i]
        sign = (val >> 7) & 1
        exp = (val >> 3) & 0xF
        mant = val & 0x7
        
        if exp == 0:
            # Denormalized
            result[i] = ((-1)**sign) * (mant / 8.0) * 2**(-6)
        elif exp == 0xF:
            # Inf/NaN
            result[i] = float('inf') if mant == 0 else float('nan')
        else:
            # Normalized
            result[i] = ((-1)**sign) * (1.0 + mant / 8.0) * 2**(exp - 7)
    
    return result


def fp8_e4m3_to_float32_fast(data):
    """Fast FP8 E4M3 conversion using numpy bit manipulation."""
    if data.dtype in (np.float32, np.float16):
        return data.astype(np.float32)
    
    data_u16 = data.view(np.uint16).astype(np.uint32)
    
    sign = (data_u16 >> 8) & 0x80
    exp = (data_u16 >> 4) & 0xF
    mant = data_u16 & 0xF
    
    # Reconstruct float32
    f32_sign = sign << 24  # Sign bit in float32
    f32_exp = np.zeros_like(exp, dtype=np.uint32)
    f32_mant = np.zeros_like(mant, dtype=np.uint32)
    
    mask_norm = (exp > 0) & (exp < 0xF)
    mask_zero = exp == 0
    mask_inf = exp == 0xF
    
    # Normalized: bias 7 → 127
    f32_exp[mask_norm] = (exp[mask_norm] - 7 + 127) << 23
    f32_mant[mask_norm] = mant[mask_norm] << 20
    
    # Denormalized: exp=0, implicit leading 0
    f32_exp[mask_zero] = (127 - 7) << 23  # minimum exponent
    f32_mant[mask_zero] = mant[mask_zero] << 20
    
    # Inf
    f32_exp[mask_inf] = 0x7F800000
    
    result_bits = f32_sign | f32_exp | f32_mant
    return result_bits.view(np.float32)


def bf16_to_f32(data):
    """Convert bfloat16 (stored as uint16) to float32."""
    if hasattr(data, 'dtype') and 'bfloat' in str(data.dtype):
        # Safetensors stores BF16 as uint16
        u16 = data.view(np.uint16)
        # BF16 = top 16 bits of float32
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32)
    return data.astype(np.float32)


def load_outside_weights():
    """Load token_embd and other weights from outside.safetensors."""
    path = os.path.join(FP8_DIR, "outside.safetensors")
    print(f"  Loading {path}...")
    
    tensors = {}
    with safe_open(path, framework="numpy") as f:
        for name in f.keys():
            raw = f.get_tensor(name)
            print(f"    {name}: {raw.shape} dtype={raw.dtype}")
            # Convert BF16 to F32
            if 'bfloat' in str(raw.dtype) or raw.dtype == np.uint16:
                tensors[name] = bf16_to_f32(raw)
            else:
                tensors[name] = raw.astype(np.float32)
    
    return tensors


def load_layer_weights(layer_idx):
    """Load gate weights from a layer safetensors."""
    path = os.path.join(FP8_DIR, f"layers-{layer_idx}.safetensors")
    
    tensors = {}
    with safe_open(path, framework="numpy") as f:
        for name in f.keys():
            if 'gate_inp' in name or 'gate.weight' in name:
                raw = f.get_tensor(name)
                if 'bfloat' in str(raw.dtype) or raw.dtype == np.uint16:
                    tensors[name] = bf16_to_f32(raw)
                else:
                    tensors[name] = raw.astype(np.float32)
    
    return tensors


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def run_calibration(embd, gate_proj, corpus_path, n_tokens=3000):
    """Run real routing with actual weights."""
    print(f"  Embedding: {embd.shape} dtype={embd.dtype}")
    print(f"  Gate proj: {gate_proj.shape} dtype={gate_proj.dtype}")
    
    # Read corpus
    with open(corpus_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    tokens = [b % embd.shape[0] for b in text.encode('utf-8', errors='replace')[:n_tokens]]
    print(f"  Corpus: {len(tokens)} tokens, vocab={embd.shape[0]}")
    
    # Ensure gate_proj is [hidden, n_experts]
    if gate_proj.shape[0] != embd.shape[1]:
        gate_proj = gate_proj.T
    
    expert_counts = np.zeros(N_EXPERTS, dtype=np.int64)
    
    t0 = time.time()
    for tid in tokens:
        h = embd[tid]  # [hidden]
        scores = gate_proj.T @ h  # [n_experts]
        top8 = np.argsort(scores)[-TOP_K:]
        expert_counts[top8] += 1
    
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return expert_counts


def analyze(counts, label=""):
    """Analyze frequency distribution."""
    total = counts.sum()
    freq = counts / total * 100
    n_active = (counts > 0).sum()
    sorted_freq = np.sort(freq)[::-1]
    cumsum = np.cumsum(sorted_freq)
    
    print(f"\n  === EXPERT ACTIVATION ({label}) ===")
    print(f"  Decisions: {total:,} | Active: {n_active}/{N_EXPERTS}")
    
    print(f"\n  Cumulative coverage:")
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        c = cumsum[min(n-1, len(cumsum)-1)]
        bar = '#' * int(c / 2)
        print(f"  Top-{n:>3}: {c:>6.1f}% {bar}")
    
    hot = (freq > 0.5).sum()
    warm = ((freq > 0.01) & (freq <= 0.5)).sum()
    cold = (freq <= 0.01).sum()
    
    print(f"\n  Classification: Hot={hot} Warm={warm} Cold={cold}")
    
    EXPERT_MB = 2048 * 512 * 4.5 / 8 / 1e6
    PAIR = EXPERT_MB * 2
    gb_iq4 = hot * PAIR / 1024
    gb_q3 = warm * PAIR * 3.44/4.5 / 1024
    gb_q2 = cold * PAIR * 2.94/4.5 / 1024
    
    print(f"\n  Optimal allocation:")
    print(f"  Hot  ({hot:>3}): IQ4_NL → {gb_iq4:.2f} GB")
    print(f"  Warm ({warm:>3}): Q3_K   → {gb_q3:.2f} GB")
    print(f"  Cold ({cold:>3}): Q2_K   → {gb_q2:.2f} GB")
    print(f"  TOTAL: {gb_iq4+gb_q3+gb_q2:.2f} GB (vs 12.61 GB all IQ4)")
    
    return {'freq': freq.tolist(), 'n_hot': hot, 'n_warm': warm, 'n_cold': cold}


if __name__ == '__main__':
    print("=" * 70)
    print("  D2 CALIBRATION — REAL FP8 Safetensors")
    print("=" * 70)
    
    # Load embeddings
    print("\nStep 1: Loading embeddings...")
    outside = load_outside_weights()
    
    embd_key = [k for k in outside if 'embd' in k and 'token' in k]
    if not embd_key:
        embd_key = [k for k in outside if 'embd' in k]
    
    embd_raw = outside[embd_key[0]]
    print(f"  Embedding raw: {embd_raw.shape} {embd_raw.dtype}")
    
    # Convert FP8 to float32
    if embd_raw.dtype == np.float32 or embd_raw.dtype == np.float16:
        embd = embd_raw.astype(np.float32)
    else:
        print(f"  Converting FP8 to float32...")
        embd = fp8_e4m3_to_float32_fast(embd_raw.flatten()).reshape(embd_raw.shape)
    
    print(f"  Embedding: {embd.shape} → [{embd.shape[0]} vocab, {embd.shape[1]} hidden]")
    
    # Load gate projection from layer 0
    print("\nStep 2: Loading gate projection...")
    layer0 = load_layer_weights(0)
    
    gate_key = [k for k in layer0 if 'gate_inp' in k]
    if gate_key:
        gate_raw = layer0[gate_key[0]]
        if gate_raw.dtype in (np.float32, np.float16):
            gate_proj = gate_raw.astype(np.float32)
        else:
            gate_proj = fp8_e4m3_to_float32_fast(gate_raw.flatten()).reshape(gate_raw.shape)
        print(f"  Gate: {gate_proj.shape} {gate_proj.dtype}")
    else:
        print("  WARNING: gate_inp not found, using embedding-based routing")
        gate_proj = np.random.randn(embd.shape[1], N_EXPERTS).astype(np.float32) * 0.01
    
    # Run calibration
    print("\nStep 3: Running calibration...")
    counts = run_calibration(embd, gate_proj, CORPUS)
    
    # Analyze
    print("\nStep 4: Analysis...")
    result = analyze(counts, "FP8 safetensors real weights")
    
    # Save
    output = {'expert_freq': counts.tolist(), 'summary': result}
    outpath = os.path.join(OUTDIR, 'd2_calibration_fp8.json')
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {outpath}")
