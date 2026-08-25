#!/usr/bin/env python3
"""
D2-MOE DFlash Hidden State Inspector
=====================================
Diagnostic tool to understand why DFlash acceptance = 0%.

Tests:
1. Tensor shape validation (D2-MOE vs DFlash compatibility)
2. fc.weight projection simulation with synthetic data
3. Server log analysis for hidden state capture issues
4. Numerical health check of DFlash weights
"""

import sys, os, struct, time, json
import numpy as np

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================================
# PART 1: GGUF Tensor Inspection
# ============================================================

def inspect_gguf(path, label):
    """Inspect a GGUF file for DFlash compatibility."""
    import gguf
    print(f"\n{'='*60}")
    print(f"  {label}: {os.path.basename(path)}")
    print(f"{'='*60}")
    
    reader = gguf.GGUFReader(path, 'r')
    
    # KV metadata
    print("\n--- Metadata ---")
    for name, field in sorted(reader.fields.items()):
        if not name.startswith('tensor_') and not name.startswith('tokenizer'):
            try:
                data = field.parts[field.data]
                if len(data) == 1:
                    val = data[0]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8', errors='replace')[:60]
                    print(f"  {name} = {val}")
                elif len(data) <= 10:
                    print(f"  {name} = {[int(x) for x in data[:10]]}")
            except:
                pass
    
    # Tensor shapes
    print("\n--- Key Tensors ---")
    for tensor in reader.tensors:
        name = tensor.name
        shape = list(tensor.shape)
        if any(x in name for x in ['token_embd', 'output_norm', 'output.weight',
                                     'fc.weight', 'hidden_norm',
                                     'blk.0.attn_q', 'blk.0.ffn_gate_exps',
                                     'blk.0.ffn_down_exps', 'blk.0.attn_output',
                                     'blk.0.ffn_gate_inp']):
            data = tensor.data
            stats = f"mean={np.mean(data):.6f} std={np.std(data):.6f} min={np.min(data):.6f} max={np.max(data):.6f}"
            has_nan = np.any(np.isnan(data.astype(np.float32)))
            has_inf = np.any(np.isinf(data.astype(np.float32)))
            health = "[FAIL] NaN/Inf!" if has_nan or has_inf else "[OK]"
            print(f"  {name}: {shape} {stats} {health}")
    
    return reader

# ============================================================
# PART 2: fc.weight Projection Test
# ============================================================

def test_fc_projection(d2moe_path, dflash_path):
    """Simulate the DFlash fc.weight projection with synthetic hidden states."""
    import gguf
    
    print(f"\n{'='*60}")
    print("  FC.WEIGHT PROJECTION TEST")
    print(f"{'='*60}")
    
    # Load DFlash fc.weight
    dflash_reader = gguf.GGUFReader(dflash_path, 'r')
    fc_weight = None
    for t in dflash_reader.tensors:
        if 'fc.weight' in t.name:
            fc_weight = t.data.astype(np.float32)
            break
    
    if fc_weight is None:
        print("  ❌ fc.weight not found in DFlash GGUF!")
        return
    
    print(f"\n  fc.weight shape: {fc_weight.shape}")
    print(f"  Expected: (16384, 2048) or (2048, 16384)")
    
    # The fc.weight projects concatenated hidden states [16384, ctx] → [2048, ctx]
    # In ggml: result = target_hidden @ fc.weight^T
    # target_hidden: [16384, ctx] (8 layers × 2048)
    # fc.weight in ggml: ne[0]=2048, ne[1]=16384
    # Result: [2048, ctx]
    
    # Simulate with random data
    np.random.seed(42)
    ctx_len = 10
    n_target_features = 16384  # 8 * 2048
    n_embd = 2048
    
    # Create synthetic target hidden states (simulating 8 layers of 2048)
    target_hidden = np.random.randn(n_target_features, ctx_len).astype(np.float32)
    
    # Simulate the projection: result = fc_weight @ target_hidden
    # (In ggml, fc_weight has ne[0]=2048, ne[1]=16384, so the matmul is fc_weight @ target_hidden)
    if fc_weight.shape == (2048, 16384):
        result = fc_weight @ target_hidden  # [2048, 16384] @ [16384, ctx] = [2048, ctx]
        print(f"  Projection: [2048,16384] @ [16384,{ctx_len}] → [{result.shape[0]},{result.shape[1]}]")
    elif fc_weight.shape == (16384, 2048):
        # Transpose for ggml convention
        result = fc_weight.T @ target_hidden  # [2048,16384] @ [16384, ctx] = [2048, ctx]
        print(f"  Projection: [2048,16384] @ [16384,{ctx_len}] → [{result.shape[0]},{result.shape[1]}]")
    else:
        print(f"  [FAIL] Unexpected fc.weight shape: {fc_weight.shape}")
        return
    
    # Analyze result
    has_nan = np.any(np.isnan(result))
    has_inf = np.any(np.isinf(result))
    print(f"\n  Result stats:")
    print(f"    mean:  {np.mean(result):.6f}")
    print(f"    std:   {np.std(result):.6f}")
    print(f"    min:   {np.min(result):.6f}")
    print(f"    max:   {np.max(result):.6f}")
    print(f"    NaN:   {has_nan}")
    print(f"    Inf:   {has_inf}")
    
    # Test with zeros (simulating missing hidden states)
    zero_hidden = np.zeros((n_target_features, ctx_len), dtype=np.float32)
    if fc_weight.shape == (2048, 16384):
        zero_result = fc_weight @ zero_hidden
    else:
        zero_result = fc_weight.T @ zero_hidden
    
    print(f"\n  With ZERO hidden states (simulating CPU fallback failure):")
    print(f"    mean:  {np.mean(zero_result):.6f}")
    print(f"    std:   {np.std(zero_result):.6f}")
    print(f"    All zero: {np.allclose(zero_result, 0)}")
    
    if np.allclose(zero_result, 0):
        print(f"\n  [WARN] If hidden states are zero (CPU fallback failure),")
        print(f"     DFlash fc.weight projection produces zeros -> wrong drafts -> 0%% acceptance!")

# ============================================================
# PART 3: Layer Mapping Verification
# ============================================================

def verify_layer_mapping(dflash_path):
    """Verify that target_layer_ids match actual model layers."""
    import gguf
    
    print(f"\n{'='*60}")
    print("  LAYER MAPPING VERIFICATION")
    print(f"{'='*60}")
    
    reader = gguf.GGUFReader(dflash_path, 'r')
    
    # Get target_layer_ids
    target_ids = None
    n_embd = None
    for name, field in sorted(reader.fields.items()):
        if 'target_layer_ids' in name:
            try:
                data = field.parts[field.data]
                target_ids = [int(x) for x in data]
            except:
                pass
        if 'embedding_length' in name:
            try:
                data = field.parts[field.data]
                n_embd = int(data[0])
            except:
                pass
    
    if target_ids is None:
        print("  ❌ target_layer_ids not found!")
        return
    
    print(f"\n  target_layer_ids: {target_ids}")
    print(f"  n_embd: {n_embd}")
    print(f"  n_target_features: {len(target_ids) * n_embd}")
    
    # Check which layers are GDN vs full attention
    print(f"\n  Layer types (Qwen3.6-35B-A3B):")
    print(f"  Layers where (i+1)%4==0 are FULL ATTENTION, others are GDN (linear)")
    
    for i, lid in enumerate(target_ids):
        layer_type = "FULL" if (lid + 1) % 4 == 0 else "GDN"
        on_gpu_ngl15 = "GPU" if lid < 15 else "CPU"
        print(f"    DFlash slot {i}: target layer {lid:2d} ({layer_type}) → {on_gpu_ngl15} with ngl=15")
    
    # Check: with ngl=15, how many target layers are on GPU?
    gpu_count = sum(1 for lid in target_ids if lid < 15)
    cpu_count = len(target_ids) - gpu_count
    print(f"\n  With ngl=15: {gpu_count} layers on GPU, {cpu_count} layers on CPU")
    
    if cpu_count > 0:
        print(f"\n  [WARN] {cpu_count} target layers are on CPU!")
        print(f"     DFlash hidden state capture uses 'callback hidden fallback' for these.")
        print(f"     This may produce incorrect/zero hidden states -> 0%% acceptance.")
    
    # What ngl would cover ALL target layers?
    max_target = max(target_ids)
    print(f"\n  To cover ALL target layers on GPU: ngl >= {max_target + 1}")
    print(f"  (D2-MOE = 17.5 GB, VRAM = 8 GB → ngl={max_target + 1} needs ~{17.5 * (max_target + 1) / 40:.1f} GB)")

# ============================================================
# PART 4: Server Log Analysis
# ============================================================

def analyze_server_log(log_path):
    """Analyze DFlash server logs for hidden state capture issues."""
    print(f"\n{'='*60}")
    print("  SERVER LOG ANALYSIS")
    print(f"{'='*60}")
    
    if not os.path.exists(log_path):
        print(f"  [FAIL] Log file not found: {log_path}")
        return
    
    with open(log_path, 'r', errors='replace') as f:
        lines = f.readlines()
    
    # Count key events
    hidden_warnings = [l for l in lines if 'allocate_hidden_gpu' in l]
    accept_lines = [l for l in lines if 'draft acceptance' in l]
    stat_lines = [l for l in lines if 'dflash:' in l and 'statistics' in l]
    commit_failures = [l for l in lines if 'commit failed' in l]
    
    print(f"\n  Total log lines: {len(lines)}")
    print(f"  Hidden state warnings: {len(hidden_warnings)}")
    print(f"  Commit failures: {len(commit_failures)}")
    print(f"  Acceptance reports: {len(accept_lines)}")
    
    if hidden_warnings:
        # Extract unique warning patterns
        patterns = set()
        for l in hidden_warnings:
            # Extract "hidden layer N device X"
            import re
            m = re.search(r'hidden layer (\d+) device (\w+)', l)
            if m:
                patterns.add(f"layer {m.group(1)} → {m.group(2)}")
        
        print(f"\n  Unique hidden state warnings:")
        for p in sorted(patterns):
            print(f"    {p}")
    
    if accept_lines:
        print(f"\n  Acceptance reports:")
        for l in accept_lines[-5:]:
            print(f"    {l.strip()}")
    
    if stat_lines:
        print(f"\n  DFlash statistics:")
        for l in stat_lines[-3:]:
            print(f"    {l.strip()}")

# ============================================================
# MAIN
# ============================================================

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    
    d2moe_path = os.path.join(base, 'models', 'Qwen3.6-35B-A3B-D2-MOE.gguf')
    dflash_path = os.path.join(base, 'models', 'Qwen3.6-35B-A3B-DFlash-D2FIX.gguf')
    log_path = os.path.join(base, 'server_dflash_fixed_ids.log')
    
    print("=" * 60)
    print("  D2-MOE DFlash Hidden State Inspector")
    print("=" * 60)
    
    # 1. Inspect both GGUFs
    if os.path.exists(d2moe_path):
        inspect_gguf(d2moe_path, "D2-MOE Target")
    
    if os.path.exists(dflash_path):
        inspect_gguf(dflash_path, "DFlash Draft")
    
    # 2. Test fc.weight projection
    if os.path.exists(dflash_path):
        test_fc_projection(d2moe_path, dflash_path)
    
    # 3. Verify layer mapping
    if os.path.exists(dflash_path):
        verify_layer_mapping(dflash_path)
    
    # 4. Analyze server logs
    analyze_server_log(log_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("  DIAGNOSIS SUMMARY")
    print(f"{'='*60}")
    print("""
  Root cause of 0% DFlash acceptance:

  1. With ngl=15, 5 of 8 target layers (16,22,27,32,37) are on CPU
  2. DFlash hidden capture uses 'callback hidden fallback' for CPU layers
  3. The fallback may produce incorrect/zero hidden states
  4. fc.weight projection of zero/wrong states → wrong draft tokens
  5. Target model rejects all draft tokens → 0% acceptance

  Solutions:
  A. Use ngl >= 38 (all target layers on GPU) → needs >14 GB VRAM
  B. Use a smaller D2-MOE (Q3) that fits with ngl >= 38 in 8 GB
  C. Fix the CPU fallback to properly capture hidden states
  D. Accept D2-MOE baseline (27.2 t/s) without DFlash
""")

if __name__ == '__main__':
    main()
