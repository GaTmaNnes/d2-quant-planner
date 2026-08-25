#!/usr/bin/env python3
"""VRAM decomposition using gguf library with proper offset-based size calculation."""
import sys, os, io
from pathlib import Path
from gguf import GGUFReader

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Actual BPW calibrated from llama.cpp source code
# Format: (name, bytes_per_block, block_size) -> bpw = bytes_per_block * 8 / block_size
# [CORRIGÉ 25/08/2026] Mapping conforme beellama.cpp/ggml/include/ggml.h :
# l'ancien tableau avait tous les IDs décalés (20="IQ2_S", 26="IQ4_NL",
# 13="Q3_K_L", 14="Q4_K_S"...). Vraies valeurs : 11=Q3_K 3.4375 bpw,
# 12=Q4_K 4.5, 13=Q5_K 5.5, 14=Q6_K 6.5625, 15=Q8_K, 20=IQ4_NL 4.5,
# 22=IQ2_S 2.5, 26=I32.
QUANT = {
    0:  ("F32",     32.0),   # 4 bytes/elem
    1:  ("F16",     16.0),   # 2 bytes/elem
    2:  ("Q4_0",    4.5),    # 32-block, 18 bytes
    3:  ("Q4_1",    5.0),    # 32-block, 20 bytes
    6:  ("Q5_0",    5.5),    # 32-block, 22 bytes
    7:  ("Q5_1",    6.0),    # 32-block, 24 bytes
    8:  ("Q8_0",    8.5),    # 32-block, 34 bytes
    9:  ("Q8_1",    9.0),    # 32-block, 36 bytes
    10: ("Q2_K",    2.625),  # 256-block, 84 bytes
    11: ("Q3_K",    3.4375), # 256-block, 110 bytes
    12: ("Q4_K",    4.5),    # 256-block, 144 bytes
    13: ("Q5_K",    5.5),    # 256-block, 176 bytes
    14: ("Q6_K",    6.5625), # 256-block, 210 bytes
    15: ("Q8_K",    8.125),  # 256-block, 260 bytes
    16: ("IQ2_XXS", 2.0625), # 256-block, 66 bytes
    17: ("IQ2_XS",  2.3125), # 256-block, 74 bytes
    18: ("IQ3_XXS", 3.0625), # 256-block
    19: ("IQ1_S",   1.5625),
    20: ("IQ4_NL",  4.5),    # 32-block, 18 bytes
    21: ("IQ3_S",   3.4375),
    22: ("IQ2_S",   2.5),    # 256-block, 80 bytes
    23: ("IQ4_XS",  4.25),   # 256-block, 136 bytes
    24: ("I8",      8.0),
    25: ("I16",     16.0),
    26: ("I32",     32.0),
    27: ("I64",     64.0),
    28: ("F64",     64.0),
    29: ("IQ1_M",   1.75),
    30: ("BF16",    16.0),
}


def categorize(name):
    nl = name.lower()
    layer = None
    if 'layers.' in name:
        try:
            layer = int(name.split('layers.')[1].split('.')[0])
        except: pass
    
    if 'token_embd' in nl or 'output_norm' in nl:
        return 'EMBED/NORM', layer
    elif 'output.' in nl and 'weight' in nl:
        return 'LM_HEAD', layer
    elif 'ffn_gate_exps' in nl or 'ffn_up_exps' in nl:
        return 'EXPERT_GATE_UP', layer
    elif 'ffn_down_exps' in nl:
        return 'EXPERT_DOWN', layer
    elif 'attn_qkv' in nl:
        return 'ATTN_QKV', layer
    elif 'attn_o' in nl:
        return 'ATTN_OUT', layer
    elif 'ssm_' in nl or 'conv1d' in nl:
        return 'SSM/CONV', layer
    elif 'ffn_gate' in nl or 'ffn_up' in nl:
        return 'FFN_GATE_UP', layer
    elif 'ffn_down' in nl:
        return 'FFN_DOWN', layer
    elif 'norm' in nl:
        return 'NORMS', layer
    else:
        return 'OTHER', layer


def analyze(path, label):
    print(f"\n{'='*85}")
    print(f"  {label}")
    print(f"  File: {Path(path).name} ({os.path.getsize(path)/1e9:.2f} GB on disk)")
    print(f"{'='*85}")

    reader = GGUFReader(path)

    # Read file size
    file_size = os.path.getsize(path)

    # Compute sizes from actual element count and quant BPW
    tensors = []
    for tensor in reader.tensors:
        ne = 1
        for s in tensor.shape:
            ne *= int(s)
        qtype = int(tensor.tensor_type)
        qname, bpw = QUANT.get(qtype, (f"UNK({qtype})", 4.0))
        calc_size = ne * bpw / 8.0
        tensors.append({
            'name': tensor.name,
            'ne': ne,
            'qtype': qtype,
            'qname': qname,
            'bpw': bpw,
            'calc_size': calc_size,
        })

    total_calc = sum(t['calc_size'] for t in tensors)
    # [CORRIGÉ 25/08/2026] NOTE : l'overhead inclut header + paddings
    # d'alignement 32 o (non modélisés) ; si on calculait le dernier tenseur
    # par file_size-offset, il faudrait lui soustraire le padding final.
    overhead = file_size - total_calc
    overhead_pct = overhead / file_size * 100

    print(f"  Calculated weights: {total_calc/1e9:.3f} GB ({total_calc/1e6:.0f} MB)")
    print(f"  File overhead:      {overhead/1e9:.3f} GB ({overhead_pct:.1f}%)")
    print(f"  Num tensors:        {len(tensors)}")

    # Group by category
    cats = {}
    quants = {}
    cat_quant = {}
    layers = {}

    for t in tensors:
        cat, layer = categorize(t['name'])
        qn = t['qname']
        
        if cat not in cats:
            cats[cat] = {'n': 0, 'bytes': 0, 'ne': 0}
        cats[cat]['n'] += 1
        cats[cat]['bytes'] += t['calc_size']
        cats[cat]['ne'] += t['ne']
        
        if qn not in quants:
            quants[qn] = {'n': 0, 'bytes': 0, 'ne': 0, 'bpw': t['bpw']}
        quants[qn]['n'] += 1
        quants[qn]['bytes'] += t['calc_size']
        quants[qn]['ne'] += t['ne']
        
        kq = (cat, qn)
        if kq not in cat_quant:
            cat_quant[kq] = {'n': 0, 'bytes': 0}
        cat_quant[kq]['n'] += 1
        cat_quant[kq]['bytes'] += t['calc_size']
        
        if layer is not None:
            if layer not in layers:
                layers[layer] = {'bytes': 0, 'cats': {}}
            layers[layer]['bytes'] += t['calc_size']
            if cat not in layers[layer]['cats']:
                layers[layer]['cats'][cat] = 0
            layers[layer]['cats'][cat] += t['calc_size']

    # Print by category
    print(f"\n  --- By Category ---")
    print(f"  {'Category':<20} {'Count':>5} {'MB':>8} {'GB':>7} {'%':>5}")
    print(f"  {'-'*45}")
    for cat in sorted(cats.keys(), key=lambda c: -cats[c]['bytes']):
        v = cats[cat]
        pct = v['bytes'] / total_calc * 100
        print(f"  {cat:<20} {v['n']:>5} {v['bytes']/1e6:>8.0f} {v['bytes']/1e9:>7.3f} {pct:>4.1f}%")

    # Print by quant
    print(f"\n  --- By Quantization ---")
    print(f"  {'Format':<12} {'Count':>5} {'MB':>8} {'GB':>7} {'BPW':>5}")
    print(f"  {'-'*37}")
    for qn in sorted(quants.keys(), key=lambda q: -quants[q]['bytes']):
        v = quants[qn]
        print(f"  {qn:<12} {v['n']:>5} {v['bytes']/1e6:>8.0f} {v['bytes']/1e9:>7.3f} {v['bpw']:>5.2f}")

    # Print category x quant
    print(f"\n  --- Category x Quantization ---")
    print(f"  {'Category':<20} {'Quant':<10} {'MB':>8} {'Count':>5}")
    print(f"  {'-'*43}")
    for kq in sorted(cat_quant.keys(), key=lambda k: -cat_quant[k]['bytes']):
        cat, qn = kq
        v = cat_quant[kq]
        print(f"  {cat:<20} {qn:<10} {v['bytes']/1e6:>8.0f} {v['n']:>5}")

    # Per-layer
    if layers:
        nl = len(layers)
        print(f"\n  --- Per-Layer (first 3 + last 3 of {nl}) ---")
        ls = sorted(layers.keys())
        show = ls[:3] + ['...'] + ls[-3:] if nl > 6 else ls
        for l in show:
            if l == '...':
                print(f"  ...")
                continue
            v = layers[l]
            top = sorted(v['cats'].items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{k}={v2/1e6:.0f}MB" for k, v2 in top)
            print(f"  L{l:>2}: {v['bytes']/1e6:>6.0f} MB  ({top_str})")

    return cats, quants, cat_quant, layers


def compare(path_a, label_a, path_b, label_b):
    print(f"\n{'='*85}")
    print(f"  COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*85}")

    ra = GGUFReader(path_a)
    rb = GGUFReader(path_b)

    da = {}
    for t in ra.tensors:
        ne = 1
        for s in t.shape:
            ne *= int(s)
        qtype = int(t.tensor_type)
        qname, bpw = QUANT.get(qtype, (f"UNK({qtype})", 4.0))
        da[t.name] = {'ne': ne, 'qtype': qtype, 'bpw': bpw, 'size': ne * bpw / 8, 'qname': qname}

    db = {}
    for t in rb.tensors:
        ne = 1
        for s in t.shape:
            ne *= int(s)
        qtype = int(t.tensor_type)
        qname, bpw = QUANT.get(qtype, (f"UNK({qtype})", 4.0))
        db[t.name] = {'ne': ne, 'qtype': qtype, 'bpw': bpw, 'size': ne * bpw / 8, 'qname': qname}

    common = set(da.keys()) & set(db.keys())
    print(f"  Common: {len(common)}, Only-A: {len(set(da)-set(db))}, Only-B: {len(set(db)-set(da))}")

    total_a = sum(da[n]['size'] for n in common)
    total_b = sum(db[n]['size'] for n in common)

    print(f"\n  {label_a} weight size: {total_a/1e9:.3f} GB")
    print(f"  {label_b} weight size: {total_b/1e9:.3f} GB")
    print(f"  Delta: {(total_a-total_b)/1e6:+.0f} MB")

    # Deltas per category
    cat_delta = {}
    for n in common:
        cat, _ = categorize(n)
        d = da[n]['size'] - db[n]['size']
        if cat not in cat_delta:
            cat_delta[cat] = 0
        cat_delta[cat] += d

    print(f"\n  --- Delta by Category ---")
    for cat in sorted(cat_delta.keys(), key=lambda c: -abs(cat_delta[c])):
        print(f"  {cat:<20} {cat_delta[cat]/1e6:>+8.0f} MB")

    # Top individual tensor deltas
    diffs = []
    for n in common:
        d = da[n]['size'] - db[n]['size']
        if abs(d) > 1e5:
            diffs.append((n, da[n], db[n], d))
    diffs.sort(key=lambda x: -abs(x[3]))

    print(f"\n  --- Top 10 Tensor Deltas ---")
    print(f"  {'Tensor':<42} {label_a:<10} {label_b:<10} {'dMB':>8}")
    print(f"  {'-'*70}")
    for n, va, vb, delta in diffs[:10]:
        print(f"  {n[:41]:<42} {va['qname']:<10} {vb['qname']:<10} {delta/1e6:>+8.0f}")
    
    # VRAM estimation
    print(f"\n  --- VRAM Estimation (ngl=15, 15/40 layers GPU) ---")
    gpu_fraction = 15.0 / 40.0  # 37.5% layers on GPU
    
    # For MoE: GPU loads all layers' weight buffers but only executes some
    # Actually in partial offload, GPU loads the weight buffers for GPU layers
    # The weights are: expert (90%) + attn/ssm/norms (10%)
    # For ngl=15: layers 0-14 are GPU (15/40)
    # Each layer has same weight ~ total/40
    
    layer_avg = total_a / 40
    gpu_weight_a = layer_avg * 15
    gpu_weight_b = total_b / 40 * 15
    
    print(f"  {label_a}: ~{gpu_weight_a/1e9:.2f} GB weight on GPU + compute/KV overhead")
    print(f"  {label_b}: ~{gpu_weight_b/1e9:.2f} GB weight on GPU + compute/KV overhead")


if __name__ == '__main__':
    iq4 = "C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
    d2eco = "C:/Users/videl/Desktop/lama 1080-5070/models/Qwen3.6-35B-A3B-D2-ECO.gguf"
    q3km = "C:/Users/videl/Desktop/lama 1080-5070/models/35b_quant/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"

    r1 = analyze(iq4, "IQ4_NL (HauhauCS Aggressive) - ON DISK 19.78 GB")
    r2 = analyze(d2eco, "D2-ECO (custom Q3/Q4 mix) - ON DISK 15.50 GB")
    compare(iq4, "IQ4_NL", d2eco, "D2-ECO")

    if os.path.exists(q3km):
        r3 = analyze(q3km, "Q3_K_M (Unsloth) - ON DISK 16.60 GB")
        compare(iq4, "IQ4_NL", q3km, "Q3_K_M")
