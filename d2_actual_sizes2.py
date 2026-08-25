#!/usr/bin/env python3
"""Compute actual stored tensor sizes from GGUF offsets using gguf library."""
import sys, io, os, struct
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from gguf import GGUFReader

def get_actual_sizes(path):
    """Use GGUFReader to get offsets, then compute sizes from consecutive offsets."""
    reader = GGUFReader(path)
    
    # Get tensor info: name, offset, type, ne
    tensor_info = []
    for tensor in reader.tensors:
        ne = 1
        for s in tensor.shape:
            ne *= int(s)
        # [CORRIGÉ 25/08/2026] gguf-py n'a pas d'attribut `tensor.offset` :
        # c'est `data_offset` (ReaderTensor NamedTuple). Fallback getattr.
        off = getattr(tensor, 'data_offset', None)
        if off is None:
            off = getattr(tensor, 'offset', 0)
        tensor_info.append({
            'name': tensor.name,
            'offset': int(off),
            'type': int(tensor.tensor_type),
            'ne': ne,
        })
    
    # Sort by offset
    tensor_info.sort(key=lambda x: x['offset'])
    
    # Compute actual stored sizes
    file_size = os.path.getsize(path)
    
    for i, t in enumerate(tensor_info):
        if i + 1 < len(tensor_info):
            t['actual_size'] = tensor_info[i+1]['offset'] - t['offset']
        else:
            # [CORRIGÉ 25/08/2026] NOTE : dernier tenseur — file_size - offset
            # inclut header + padding final d'alignement → taille surestimée
            # (approximation conservée, à soustraire si besoin de l'octet près).
            t['actual_size'] = file_size - t['offset']
    
    return tensor_info, file_size


# [CORRIGÉ 25/08/2026] Mapping conforme beellama.cpp/ggml/include/ggml.h :
# 11=Q3_K 3.4375 bpw, 12=Q4_K 4.5, 13=Q5_K 5.5, 14=Q6_K 6.5625, 15=Q8_K,
# 20=IQ4_NL 4.5, 22=IQ2_S 2.5, 26=I32. L'ancien mapping était décalé/faux.
QUANT = {
    0: ("F32", 32.0), 1: ("F16", 16.0), 2: ("Q4_0", 4.5), 3: ("Q4_1", 5.0),
    6: ("Q5_0", 5.5), 7: ("Q5_1", 6.0), 8: ("Q8_0", 8.5), 9: ("Q8_1", 9.0),
    10: ("Q2_K", 2.625), 11: ("Q3_K", 3.4375), 12: ("Q4_K", 4.5),
    13: ("Q5_K", 5.5), 14: ("Q6_K", 6.5625), 15: ("Q8_K", 8.125),
    16: ("IQ2_XXS", 2.0625), 17: ("IQ2_XS", 2.3125), 18: ("IQ3_XXS", 3.0625),
    19: ("IQ1_S", 1.5625), 20: ("IQ4_NL", 4.5), 21: ("IQ3_S", 3.4375),
    22: ("IQ2_S", 2.5), 23: ("IQ4_XS", 4.25), 24: ("I8", 8.0),
    25: ("I16", 16.0), 26: ("I32", 32.0), 27: ("I64", 64.0),
    28: ("F64", 64.0), 29: ("IQ1_M", 1.75), 30: ("BF16", 16.0),
}


def analyze(path, label):
    print("\n" + "=" * 85)
    print("  %s" % label)
    print("  File: %s (%.2f GB)" % (os.path.basename(path), os.path.getsize(path) / 1e9))
    print("=" * 85)
    
    tensors, file_size = get_actual_sizes(path)
    
    by_type = defaultdict(lambda: {'count': 0, 'actual': 0, 'ne': 0, 'names': []})
    total_actual = 0
    
    for t in tensors:
        tid = t['type']
        qname = QUANT.get(tid, ("T%d" % tid, 0))[0]
        by_type[qname]['count'] += 1
        by_type[qname]['actual'] += t['actual_size']
        by_type[qname]['ne'] += t['ne']
        if len(by_type[qname]['names']) < 3:
            by_type[qname]['names'].append(t['name'])
        total_actual += t['actual_size']
    
    print("\n  ACTUAL STORED SIZES (from GGUF data offsets):")
    print("  %-12s %6s %12s %12s %6s" % ("Format", "Count", "Actual GB", "Elements", "BPW"))
    print("  " + "-" * 50)
    for qn in sorted(by_type.keys(), key=lambda q: -by_type[q]['actual']):
        v = by_type[qn]
        bpw = v['actual'] * 8.0 / v['ne'] if v['ne'] > 0 else 0
        print("  %-12s %6d %12.3f %12s %6.2f" % (
            qn, v['count'], v['actual'] / 1e9, format(v['ne'], ','), bpw))
    
    print()
    print("  Total actual data: %.3f GB" % (total_actual / 1e9))
    print("  File size:         %.3f GB" % (file_size / 1e9))
    print("  Overhead:          %.3f GB" % ((file_size - total_actual) / 1e9))
    
    # Show sample tensors for each type
    print()
    for qn in sorted(by_type.keys(), key=lambda q: -by_type[q]['actual']):
        print("  Sample %s tensors:" % qn)
        for name in by_type[qn]['names'][:2]:
            t = next(x for x in tensors if x['name'] == name)
            bpw = t['actual_size'] * 8.0 / t['ne'] if t['ne'] > 0 else 0
            print("    %-45s ne=%12s actual=%10d bpw=%.3f" % (
                name, format(t['ne'], ','), t['actual_size'], bpw))
    
    return by_type


if __name__ == '__main__':
    iq4 = "C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
    d2eco = "C:/Users/videl/Desktop/lama 1080-5070/models/Qwen3.6-35B-A3B-D2-ECO.gguf"
    
    r1 = analyze(iq4, "IQ4_NL (HauhauCS Aggressive) - 19.78 GB on disk")
    r2 = analyze(d2eco, "D2-ECO (custom) - 15.50 GB on disk")
