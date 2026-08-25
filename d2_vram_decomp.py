#!/usr/bin/env python3
"""VRAM decomposition for 35B MoE GGUF files.
Uses safetensors GGUF reader to compute exact sizes per tensor type."""
import sys, os, json
from pathlib import Path

try:
    from gguf import GGUFReader
except ImportError:
    print("Installing gguf...")
    os.system("pip install gguf")
    from gguf import GGUFReader

# Actual BPW (bits per weight) based on GGUF block sizes
# block_bytes * 8 / block_elements = bpw
# [CORRIGÉ 25/08/2026] Mapping conforme beellama.cpp/ggml/include/ggml.h :
# l'ancien code avait des IDs décalés et Q4_0 marqué 32.0 bpw (contradiction
# avec son propre commentaire "20 bytes -> 5.0"). Vraies valeurs ci-dessous.
QUANT_BPW = {
    0: 32.0,    # F32 (4 bytes/elem)
    1: 16.0,    # F16 (2 bytes/elem)
    2: 4.5,     # Q4_0 (block=32, 18 bytes)
    3: 5.0,     # Q4_1 (block=32, 20 bytes)
    6: 5.5,     # Q5_0 (block=32, 22 bytes)
    7: 6.0,     # Q5_1 (block=32, 24 bytes)
    8: 8.5,     # Q8_0 (block=32, 34 bytes)
    9: 9.0,     # Q8_1 (block=32, 36 bytes)
    10: 2.625,  # Q2_K (block=256, 84 bytes -> 2.625 bpw)
    11: 3.4375, # Q3_K (block=256, 110 bytes -> 3.4375 bpw)
    12: 4.5,    # Q4_K (block=256, 144 bytes -> 4.5 bpw)
    13: 5.5,    # Q5_K (block=256, 176 bytes -> 5.5 bpw)
    14: 6.5625, # Q6_K (block=256, 210 bytes -> 6.5625 bpw)
    15: 8.125,  # Q8_K (block=256, 260 bytes)
    16: 2.0625, # IQ2_XXS (block=256, 66 bytes)
    17: 2.3125, # IQ2_XS (block=256, 74 bytes)
    18: 3.0625, # IQ3_XXS (block=256, 78+... bytes)
    19: 1.5625, # IQ1_S
    20: 4.5,    # IQ4_NL (block=32, 18 bytes -> 4.5 bpw)
    21: 3.4375, # IQ3_S
    22: 2.5,    # IQ2_S (block=256, 80 bytes)
    23: 4.25,   # IQ4_XS (block=256, 136 bytes -> 4.25 bpw)
    24: 8.0,    # I8
    25: 16.0,   # I16
    26: 32.0,   # I32
    27: 64.0,   # I64
    28: 64.0,   # F64
    29: 1.75,   # IQ1_M
    30: 16.0,   # BF16
}

QUANT_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
}

# Fix Windows console encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def categorize(name):
    """Categorize tensor for 35B MoE."""
    nl = name.lower()
    layer = None
    if 'layers.' in name:
        try:
            layer = int(name.split('layers.')[1].split('.')[0])
        except:
            pass

    if 'token_embd' in nl or 'output_norm' in nl or 'output' in nl:
        return 'EMBED/HEAD/NORM', layer
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
        return 'FFN_GATE_UP(shared)', layer
    elif 'ffn_down' in nl:
        return 'FFN_DOWN(shared)', layer
    elif 'attn_norm' in nl or 'ffn_norm' in nl:
        return 'NORMS', layer
    elif 'blk.' in nl:
        return 'BLOCK_OTHER', layer
    else:
        return 'MISC', layer


def analyze(path, label):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  File: {Path(path).name} ({os.path.getsize(path)/1e9:.2f} GB)")
    print(f"{'='*80}")

    reader = GGUFReader(path)

    cats = {}
    quants = {}
    cat_quant = {}
    layers = {}
    total = 0

    # [CORRIGÉ 25/08/2026] boucle morte supprimée (for field ... pass) :
    # elle ne faisait rien, les tensors sont lus via reader.tensors ci-dessous.
    # Read tensors properly
    for tensor in reader.tensors:
        name = tensor.name
        n_elements = 1
        for s in tensor.shape:
            n_elements *= int(s)
        # GGUF quant type from tensor
        qtype = int(tensor.tensor_type)
        bpw = QUANT_BPW.get(qtype, 32.0)
        qname = QUANT_NAMES.get(qtype, f"T{qtype}")
        size_bytes = n_elements * bpw / 8
        total += size_bytes

        cat, layer = categorize(name)

        if cat not in cats:
            cats[cat] = {'n': 0, 'bytes': 0, 'raw_bytes': 0}
        cats[cat]['n'] += 1
        cats[cat]['bytes'] += size_bytes
        cats[cat]['raw_bytes'] += n_elements * 4  # F32 equivalent

        if qname not in quants:
            quants[qname] = {'n': 0, 'bytes': 0, 'raw_bytes': 0, 'bpw': bpw}
        quants[qname]['n'] += 1
        quants[qname]['bytes'] += size_bytes
        quants[qname]['raw_bytes'] += n_elements * 4

        kq = (cat, qname)
        if kq not in cat_quant:
            cat_quant[kq] = 0
        cat_quant[kq] += size_bytes

        if layer is not None:
            if layer not in layers:
                layers[layer] = {'bytes': 0, 'cats': {}}
            layers[layer]['bytes'] += size_bytes
            if cat not in layers[layer]['cats']:
                layers[layer]['cats'][cat] = 0
            layers[layer]['cats'][cat] += size_bytes

    # Print
    print(f"\n  TOTAL WEIGHTS: {total/1e9:.3f} GB ({total/1e6:.0f} MB)")

    print(f"\n  --- By Category ---")
    print(f"  {'Category':<25} {'Count':>6} {'MB':>8} {'GB':>8} {'%':>6}")
    print(f"  {'-'*53}")
    for cat in sorted(cats.keys(), key=lambda c: -cats[c]['bytes']):
        v = cats[cat]
        pct = v['bytes'] / total * 100
        print(f"  {cat:<25} {v['n']:>6} {v['bytes']/1e6:>8.0f} {v['bytes']/1e9:>8.3f} {pct:>5.1f}%")

    print(f"\n  --- By Quantization ---")
    print(f"  {'Format':<15} {'Count':>6} {'MB':>8} {'GB':>8} {'BPW':>5}")
    print(f"  {'-'*42}")
    for qn in sorted(quants.keys(), key=lambda q: -quants[q]['bytes']):
        v = quants[qn]
        print(f"  {qn:<15} {v['n']:>6} {v['bytes']/1e6:>8.0f} {v['bytes']/1e9:>8.3f} {v['bpw']:>5.1f}")

    print(f"\n  --- Category × Quant ---")
    print(f"  {'Category':<25} {'Quant':<12} {'MB':>8} {'Count':>6}")
    print(f"  {'-'*51}")
    for kq in sorted(cat_quant.keys(), key=lambda k: -cat_quant[k]):
        cat, qn = kq
        # count
        v = cats.get(cat, {})
        print(f"  {cat:<25} {qn:<12} {cat_quant[kq]/1e6:>8.0f}")

    # Per-layer summary
    if layers:
        nl = len(layers)
        print(f"\n  --- Per-Layer ({nl} layers) ---")
        ls = sorted(layers.keys())
        show = ls[:3] + ['...'] + ls[-3:] if nl > 6 else ls
        for l in show:
            if l == '...':
                print(f"  ...")
                continue
            v = layers[l]
            top = sorted(v['cats'].items(), key=lambda x: -x[1])[:3]
            top_str = " | ".join(f"{k}={v2/1e6:.0f}MB" for k, v2 in top)
            print(f"  L{l:>2}: {v['bytes']/1e6:>6.0f} MB  ({top_str})")

    return cats, quants, cat_quant, layers


def compare(path_a, label_a, path_b, label_b):
    """Size-by-size comparison."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*80}")

    ra = GGUFReader(path_a)
    rb = GGUFReader(path_b)

    da = {}
    for t in ra.tensors:
        ne = 1
        for s in t.shape:
            ne *= int(s)
        bpw = QUANT_BPW.get(int(t.tensor_type), 32)
        da[t.name] = {'ne': ne, 'qtype': int(t.tensor_type), 'bpw': bpw,
                       'size': ne * bpw / 8, 'qname': QUANT_NAMES.get(int(t.tensor_type), '???')}

    db = {}
    for t in rb.tensors:
        ne = 1
        for s in t.shape:
            ne *= int(s)
        bpw = QUANT_BPW.get(int(t.tensor_type), 32)
        db[t.name] = {'ne': ne, 'qtype': int(t.tensor_type), 'bpw': bpw,
                       'size': ne * bpw / 8, 'qname': QUANT_NAMES.get(int(t.tensor_type), '???')}

    common = set(da.keys()) & set(db.keys())
    print(f"  Common: {len(common)}, Only-A: {len(set(da)-set(db))}, Only-B: {len(set(db)-set(da))}")

    diffs = []
    for name in common:
        sa = da[name]['size']
        sb = db[name]['size']
        delta = sa - sb
        if abs(delta) > 1e5:  # >100KB
            diffs.append((name, da[name], db[name], delta))
    diffs.sort(key=lambda x: -abs(x[3]))

    total_a = sum(da[n]['size'] for n in common)
    total_b = sum(db[n]['size'] for n in common)
    total_delta = total_a - total_b

    print(f"\n  {label_a} total: {total_a/1e9:.3f} GB")
    print(f"  {label_b} total: {total_b/1e9:.3f} GB")
    print(f"  Delta: {total_delta/1e9:+.3f} GB ({total_delta/1e6:+.0f} MB)")

    print(f"\n  --- Biggest Deltas (top 25) ---")
    print(f"  {'Tensor':<42} {label_a:<12} {label_b:<12} {'dMB':>8}")
    print(f"  {'-'*74}")
    for name, va, vb, delta in diffs[:25]:
        print(f"  {name[:41]:<42} {va['qname']:<12} {vb['qname']:<12} {delta/1e6:>+8.0f}")
    print(f"  {'TOTAL':<42} {'':12} {'':12} {total_delta/1e6:>+8.0f}")


    # Category delta
    cat_delta = {}
    for name in common:
        cat, _ = categorize(name)
        delta = da[name]['size'] - db[name]['size']
        if cat not in cat_delta:
            cat_delta[cat] = 0
        cat_delta[cat] += delta

    print(f"\n  --- Delta by Category ---")
    for cat in sorted(cat_delta.keys(), key=lambda c: -abs(cat_delta[c])):
        d = cat_delta[cat]
        if abs(d) > 1e5:
            print(f"  {cat:<25} {d/1e6:>+8.0f} MB")

    # Summary
    print(f"\n  KEY FINDING:")
    print(f"  IQ4_NL file ({Path(path_a).name}) is 19.78 GB on disk")
    print(f"  D2-ECO file ({Path(path_b).name}) is 15.50 GB on disk")
    print(f"  Weight calculation: IQ4_NL={total_a/1e9:.2f} GB, D2-ECO={total_b/1e9:.2f} GB")
    print(f"  Difference: {total_delta/1e6:+.0f} MB in weight data")


if __name__ == '__main__':
    iq4 = "C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
    d2eco = "C:/Users/videl/Desktop/lama 1080-5070/models/Qwen3.6-35B-A3B-D2-ECO.gguf"
    q3km = "C:/Users/videl/Desktop/lama 1080-5070/models/35b_quant/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"

    r1 = analyze(iq4, "IQ4_NL BASELINE (18.4 GB)")
    r2 = analyze(d2eco, "D2-ECO (14.4 GB)")
    compare(iq4, "IQ4_NL", d2eco, "D2-ECO")

    if os.path.exists(q3km):
        r3 = analyze(q3km, "Q3_K_M (15.5 GB)")
        compare(iq4, "IQ4_NL", q3km, "Q3_K_M")
