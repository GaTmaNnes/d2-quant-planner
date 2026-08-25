#!/usr/bin/env python3
"""VRAM decomposition: calculate weights per tensor type for 35B MoE models.
Compare IQ4_NL vs D2-ECO to understand where memory goes."""
import json, os, struct, sys
from pathlib import Path

GGUF_MAGIC = 0x46554747

# [CORRIGÉ 25/08/2026] Types KV GGUF conformes beellama.cpp/ggml/include/gguf.h
# (lignes 54-66) : 7=BOOL (1 octet, l'ancien code le lisait comme une chaîne),
# 8=STRING ([len u64][octets]), 9=ARRAY ([elem_type u32][count u64][data] —
# ordre confirmé par gguf-py gguf_reader.py _get_field_parts).
_KV_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                    10: 8, 11: 8, 12: 8}


def _skip_kv_value(f, val_type):
    """Saute une valeur KV GGUF dans le flux."""
    if val_type == 8:      # STRING : [len u64][octets]
        slen = struct.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif val_type == 9:    # ARRAY : [type i32][count u64][valeurs...]
        etype, count = struct.unpack('<IQ', f.read(12))
        for _ in range(count):
            _skip_kv_value(f, etype)
    else:
        size = _KV_SCALAR_SIZES.get(val_type)
        if size is None:
            raise ValueError(f"type KV GGUF inconnu: {val_type}")
        f.read(size)

# [CORRIGÉ 25/08/2026] Mapping conforme beellama.cpp/ggml/include/ggml.h (lignes 390-440) :
# 11=Q3_K(3.4375 bpw), 12=Q4_K(4.5), 13=Q5_K(5.5), 14=Q6_K(6.5625), 15=Q8_K,
# 20=IQ4_NL(4.5), 22=IQ2_S(2.5), 26=I32. L'ancien mapping était décalé/faux.
QUANT_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
}

# BPW = bytes_per_block * 8 / block_size (blocs 32 ou 256 selon type)
QUANT_BPW = {
    0: 32.0,    1: 16.0,    2: 4.5,     3: 5.0,
    6: 5.5,     7: 6.0,     8: 8.5,     9: 9.0,
    10: 2.625,  11: 3.4375, 12: 4.5,    13: 5.5,
    14: 6.5625, 15: 8.125,  16: 2.0625, 17: 2.3125,
    18: 3.0625, 19: 1.5625, 20: 4.5,    21: 3.4375,
    22: 2.5,    23: 4.25,   24: 8.0,    25: 16.0,
    26: 32.0,   27: 64.0,   28: 64.0,   29: 1.75,
    30: 16.0,
}


def read_gguf_tensors(gguf_path):
    """Parse GGUF file and extract tensor info."""
    tensors = []
    with open(gguf_path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == GGUF_MAGIC, f"Not GGUF: {gguf_path}"

        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        # Read key-value pairs
        for _ in range(n_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)
            val_type = struct.unpack('<I', f.read(4))[0]
            _skip_kv_value(f, val_type)

        # Read tensor info
        for _ in range(n_tensors):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            shape = []
            for _ in range(n_dims):
                shape.append(struct.unpack('<Q', f.read(8))[0])
            n_dims_read = n_dims
            # GGUF stores shape in reverse
            shape = shape[::-1]
            data_offset = struct.unpack('<Q', f.read(8))[0]
            ggml_type = struct.unpack('<I', f.read(4))[0]
            ne = 1
            for s in shape:
                ne *= s
            tensors.append({
                'name': name,
                'shape': shape,
                'ne': ne,
                'ggml_type': ggml_type,
                'data_offset': data_offset,
            })

        # Get data start offset
        # After all tensor info, data starts at next page-aligned offset
        # Actually in GGUF the data section starts right after the last tensor info
        # We need to figure out data_size
        # Read current position, then skip to data start
        # The offset in tensor info is relative to start of data section
        data_start = f.tell()

        # Align to 32 bytes
        data_start = ((data_start + 31) // 32) * 32

    # Calculate file sizes
    # [CORRIGÉ 25/08/2026] NOTE : data_size calculé via ne*bpw n'inclut PAS le
    # padding d'alignement (32 o par défaut entre tenseurs + fin de fichier) —
    # les tailles sont donc une approximation basse du fichier réel.
    file_size = os.path.getsize(gguf_path)

    # Calculate each tensor's size
    for t in tensors:
        t['data_size'] = (t['ne'] * QUANT_BPW.get(t['ggml_type'], 32)) / 8
        t['quant_name'] = QUANT_NAMES.get(t['ggml_type'], f"T{t['ggml_type']}")
        t['bpw'] = QUANT_BPW.get(t['ggml_type'], 32)

    return tensors, file_size


def categorize_tensor(name):
    """Categorize tensor into semantic groups for 35B MoE."""
    name_lower = name.lower()
    # Layer index
    layer = None
    if 'layers.' in name:
        try:
            layer = int(name.split('layers.')[1].split('.')[0])
        except (ValueError, IndexError):
            pass

    # Type categories
    if 'token_embd' in name_lower or 'output' in name_lower or 'norm' in name_lower:
        return 'embed_head_norm', layer
    elif 'attn_qkv' in name_lower or 'attn_q' in name_lower or 'attn_k' in name_lower or 'attn_v' in name_lower:
        return 'attention_qkv', layer
    elif 'attn_o' in name_lower or 'attn_output' in name_lower:
        return 'attention_out', layer
    elif 'ffn_gate_exps' in name_lower or 'ffn_up_exps' in name_lower:
        return 'expert_gate_up', layer
    elif 'ffn_down_exps' in name_lower:
        return 'expert_down', layer
    elif 'ffn_gate' in name_lower or 'ffn_up' in name_lower:
        return 'ffn_gate_up_shared', layer
    elif 'ffn_down' in name_lower:
        return 'ffn_down_shared', layer
    elif 'ssm_' in name_lower or 'conv1d' in name_lower:
        return 'ssm_conv', layer
    elif 'attn_norm' in name_lower or 'ffn_norm' in name_lower or 'ln' in name_lower:
        return 'norms', layer
    elif 'blk.' in name_lower or 'block.' in name_lower:
        return 'other_block', layer
    else:
        return 'misc', layer


def analyze_gguf(path, label):
    """Analyze a GGUF file and print categorized breakdown."""
    print(f"\n{'='*70}")
    print(f"  {label}: {Path(path).name}")
    print(f"  File: {os.path.getsize(path) / 1e9:.2f} GB")
    print(f"{'='*70}")

    tensors, file_size = read_gguf_tensors(path)

    # Group by category
    by_category = {}
    by_quant = {}
    by_layer = {}
    by_category_quant = {}

    for t in tensors:
        cat, layer = categorize_tensor(t['name'])
        qname = t['quant_name']
        sz_mb = t['data_size'] / 1e6

        if cat not in by_category:
            by_category[cat] = {'count': 0, 'total_bytes': 0}
        by_category[cat]['count'] += 1
        by_category[cat]['total_bytes'] += t['data_size']

        if qname not in by_quant:
            by_quant[qname] = {'count': 0, 'total_bytes': 0}
        by_quant[qname]['count'] += 1
        by_quant[qname]['total_bytes'] += t['data_bytes'] if 'data_bytes' in t else t['data_size']

        if layer is not None:
            if layer not in by_layer:
                by_layer[layer] = {'count': 0, 'total_bytes': 0, 'cats': {}}
            by_layer[layer]['count'] += 1
            by_layer[layer]['total_bytes'] += t['data_size']
            if cat not in by_layer[layer]['cats']:
                by_layer[layer]['cats'][cat] = 0
            by_layer[layer]['cats'][cat] += t['data_size']

        key = (cat, qname)
        if key not in by_category_quant:
            by_category_quant[key] = {'count': 0, 'total_bytes': 0}
        by_category_quant[key]['count'] += 1
        by_category_quant[key]['total_bytes'] += t['data_size']

    # Print by category
    print(f"\n  --- By Category ---")
    print(f"  {'Category':<25} {'Count':>6} {'Size (MB)':>10} {'Size (GB)':>10} {'%':>6}")
    print(f"  {'-'*57}")
    total_bytes = sum(v['total_bytes'] for v in by_category.values())
    for cat in sorted(by_category.keys()):
        v = by_category[cat]
        pct = v['total_bytes'] / total_bytes * 100
        print(f"  {cat:<25} {v['count']:>6} {v['total_bytes']/1e6:>10.1f} {v['total_bytes']/1e9:>10.3f} {pct:>5.1f}%")
    print(f"  {'TOTAL':<25} {sum(v['count'] for v in by_category.values()):>6} {total_bytes/1e6:>10.1f} {total_bytes/1e9:>10.3f}")

    # Print by quantization
    print(f"\n  --- By Quantization ---")
    print(f"  {'Format':<15} {'Count':>6} {'Size (MB)':>10} {'Size (GB)':>10} {'BPW':>5}")
    print(f"  {'-'*46}")
    for qname in sorted(by_quant.keys()):
        v = by_quant[qname]
        # BPW from the tensors
        bpw_samples = [t['bpw'] for t in tensors if t['quant_name'] == qname]
        bpw = bpw_samples[0] if bpw_samples else 0
        print(f"  {qname:<15} {v['count']:>6} {v['total_bytes']/1e6:>10.1f} {v['total_bytes']/1e9:>10.3f} {bpw:>5.1f}")

    # Print per-category quantization
    print(f"\n  --- Category × Quantization ---")
    print(f"  {'Category':<25} {'Quant':<12} {'Size (MB)':>10} {'Count':>6}")
    print(f"  {'-'*53}")
    for (cat, qname) in sorted(by_category_quant.keys()):
        v = by_category_quant[(cat, qname)]
        print(f"  {cat:<25} {qname:<12} {v['total_bytes']/1e6:>10.1f} {v['count']:>6}")

    # Print per-layer breakdown (first 5 + last 5)
    print(f"\n  --- Per-Layer (first 5 + last 5 of {len(by_layer)} layers) ---")
    layers_sorted = sorted(by_layer.keys())
    display_layers = layers_sorted[:5] + ['...'] + layers_sorted[-5:] if len(layers_sorted) > 10 else layers_sorted
    for layer in display_layers:
        if layer == '...':
            print(f"  ...")
            continue
        v = by_layer[layer]
        cats_str = ", ".join(f"{k}={v2/1e6:.0f}MB" for k, v2 in sorted(v['cats'].items(), key=lambda x: -x[1])[:4])
        print(f"  Layer {layer:>2}: {v['total_bytes']/1e6:>7.1f} MB | {cats_str}")

    return by_category, by_category_quant, by_layer


def compare_models(path1, label1, path2, label2):
    """Compare two models tensor by tensor."""
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {label1} vs {label2}")
    print(f"{'='*70}")

    t1, _ = read_gguf_tensors(path1)
    t2, _ = read_gguf_tensors(path2)

    # Build lookup by name
    d1 = {t['name']: t for t in t1}
    d2 = {t['name']: t for t in t2}

    common = set(d1.keys()) & set(d2.keys())
    only1 = set(d1.keys()) - set(d2.keys())
    only2 = set(d2.keys()) - set(d1.keys())

    print(f"  Common tensors: {len(common)}")
    print(f"  Only in {label1}: {len(only1)}")
    print(f"  Only in {label2}: {len(only2)}")

    if only1:
        print(f"\n  --- Only in {label1} ---")
        for name in sorted(only1)[:10]:
            t = d1[name]
            print(f"  {name}: {t['quant_name']} ({t['ne']} elem, {t['data_size']/1e6:.1f} MB)")
    if only2:
        print(f"\n  --- Only in {label2} ---")
        for name in sorted(only2)[:10]:
            t = d2[name]
            print(f"  {name}: {t['quant_name']} ({t['ne']} elem, {t['data_size']/1e6:.1f} MB)")

    # Compare common tensors
    print(f"\n  --- Biggest Differences (by size delta) ---")
    diffs = []
    for name in common:
        t1v = d1[name]
        t2v = d2[name]
        delta_bytes = t1v['data_size'] - t2v['data_size']
        if abs(delta_bytes) > 1e6:  # >1 MB difference
            diffs.append((name, t1v, t2v, delta_bytes))

    diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    print(f"  {'Tensor':<45} {label1:<12} {label2:<12} {'Delta MB':>10}")
    print(f"  {'-'*79}")
    total_delta = 0
    for name, t1v, t2v, delta in diffs[:30]:
        total_delta += delta
        print(f"  {name[:44]:<45} {t1v['quant_name']:<12} {t2v['quant_name']:<12} {delta/1e6:>+10.1f}")
    print(f"  {'TOTAL DELTA':<45} {'':12} {'':12} {total_delta/1e6:>+10.1f}")

    return diffs


if __name__ == '__main__':
    base = Path("C:/Users/videl/Desktop/lama 1080-5070")

    iq4_path = "C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
    d2eco_path = str(base / "models" / "Qwen3.6-35B-A3B-D2-ECO.gguf")
    q3km_path = str(base / "models" / "35b_quant" / "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf")

    # Analyze each
    b1 = analyze_gguf(iq4_path, "IQ4_NL BASELINE")
    b2 = analyze_gguf(d2eco_path, "D2-ECO")

    # Compare IQ4_NL vs D2-ECO
    compare_models(iq4_path, "IQ4_NL", d2eco_path, "D2-ECO")

    # Also analyze Q3_K_M if exists
    if os.path.exists(q3km_path):
        b3 = analyze_gguf(q3km_path, "Q3_K_M")
        compare_models(iq4_path, "IQ4_NL", q3km_path, "Q3_K_M")
