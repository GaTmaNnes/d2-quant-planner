#!/usr/bin/env python3
"""Compute ACTUAL stored tensor sizes from GGUF offsets."""
import struct, os, sys, io
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

path = 'C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf'

# [CORRIGÉ 25/08/2026] Types KV GGUF conformes beellama.cpp/ggml/include/gguf.h
# (lignes 54-66) : 7=BOOL (1 octet — l'ancien code le traitait comme une chaîne),
# 8=STRING, 9=ARRAY ([elem_type i32][count u64][data], ordre confirmé gguf-py).
_KV_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                    10: 8, 11: 8, 12: 8}


def skip_kv_value(f, vtype):
    if vtype == 8:      # STRING : [len u64][octets]
        sl = struct.unpack('<Q', f.read(8))[0]
        f.read(sl)
    elif vtype == 9:    # ARRAY : [type i32][count u64][valeurs...]
        atype, na = struct.unpack('<IQ', f.read(12))
        for _ in range(na):
            skip_kv_value(f, atype)
    else:
        size = _KV_SCALAR_SIZES.get(vtype)
        if size is None:
            raise ValueError(f"type KV GGUF inconnu: {vtype}")
        f.read(size)

with open(path, 'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    
    for _ in range(n_kv):
        kl = struct.unpack('<Q', f.read(8))[0]
        f.read(kl)
        vtype = struct.unpack('<I', f.read(4))[0]
        skip_kv_value(f, vtype)
    
    tensors = []
    for _ in range(n_tensors):
        nl = struct.unpack('<Q', f.read(8))[0]
        name = f.read(nl).decode('utf-8')
        nd = struct.unpack('<I', f.read(4))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        shape = shape[::-1]
        offset = struct.unpack('<Q', f.read(8))[0]
        gtype = struct.unpack('<I', f.read(4))[0]
        ne = 1
        for s in shape:
            ne *= s
        tensors.append((name, ne, gtype, offset))
    
    data_section_start = f.tell()
    data_section_start = ((data_section_start + 31) // 32) * 32

file_size = os.path.getsize(path)
data_size = file_size - data_section_start

print("File size:       %.3f GB" % (file_size / 1e9))
print("Data section:    %.3f GB" % (data_size / 1e9))
print("Header:          %.1f MB" % (data_section_start / 1e6))
print()

tensors.sort(key=lambda x: x[3])

by_type = defaultdict(lambda: {'count': 0, 'bytes': 0, 'ne': 0})
for i, (name, ne, gtype, offset) in enumerate(tensors):
    if i + 1 < len(tensors):
        actual = tensors[i + 1][3] - offset
    else:
        # [CORRIGÉ 25/08/2026] NOTE : dernier tenseur = data_size - offset
        # inclut le padding final d'alignement (32 o) → taille légèrement
        # surestimée pour ce tenseur.
        actual = data_size - offset
    by_type[gtype]['count'] += 1
    by_type[gtype]['bytes'] += actual
    by_type[gtype]['ne'] += ne

# [CORRIGÉ 25/08/2026] vrais noms ggml.h (l'ancien : 13="Q3_K_L", 14="Q4_K_S",
# 20="???" — tous faux). Table complète 0-30.
type_names = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
              8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
              13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS',
              18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S',
              22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32',
              27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16'}
print("ACTUAL stored sizes (from data offsets):")
print("-" * 80)
total_actual = 0
for tid in sorted(by_type.keys()):
    v = by_type[tid]
    bpw = v['bytes'] * 8.0 / v['ne'] if v['ne'] > 0 else 0
    tname = type_names.get(tid, 'T%d' % tid)
    total_actual += v['bytes']
    print("  Type %2d (%8s): %4d tensors, %8.3f GB actual, %14s elements, %.3f bpw" % (
        tid, tname, v['count'], v['bytes'] / 1e9, format(v['ne'], ','), bpw))

print()
print("Total actual data: %.3f GB" % (total_actual / 1e9))
print("Overhead:          %.1f MB" % ((file_size - data_section_start - total_actual) / 1e6))

# Show per-tensor examples for type 20
print()
print("Sample type-20 tensors (showing ne and actual size):")
for i, (name, ne, gtype, offset) in enumerate(tensors):
    if gtype == 20 and i < len(tensors) - 1:
        actual = tensors[i + 1][3] - offset
        bpw = actual * 8.0 / ne if ne > 0 else 0
        print("  %-50s ne=%12s actual=%10d bpw=%.3f" % (name, format(ne, ','), actual, bpw))
        break  # Just show first

# Show a few type 20 tensors to check consistency
print()
print("First 5 type-20 tensors:")
count = 0
for i, (name, ne, gtype, offset) in enumerate(tensors):
    if gtype == 20:
        actual = tensors[i + 1][3] - offset if i + 1 < len(tensors) else data_size - offset
        bpw = actual * 8.0 / ne if ne > 0 else 0
        print("  %-50s ne=%12s stored=%10d bpw=%.3f" % (name, format(ne, ','), actual, bpw))
        count += 1
        if count >= 5:
            break
