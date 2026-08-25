#!/usr/bin/env python3
"""Compute actual tensor sizes from GGUF data offsets."""
import struct, os, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

GGUF_MAGIC = 0x46554747

# From ggml.h — actual stored bytes per block * 8 / block_size
# [CORRIGÉ 25/08/2026] Mapping complet conforme beellama.cpp/ggml/include/ggml.h
# (lignes 390-440), tous IDs 0-30. L'ancienne table "ACCURATE" était décalée :
# 20="IQ2_S" (faux, =IQ4_NL), 26="IQ4_NL" (faux, =I32), bpw inventés.
QUANT_INFO = {
    0:  ("F32",     32.0),
    1:  ("F16",     16.0),
    2:  ("Q4_0",    4.5),    # block=32, 18 bytes -> 4.5 bpw
    3:  ("Q4_1",    5.0),    # block=32, 20 bytes
    6:  ("Q5_0",    5.5),    # block=32, 22 bytes
    7:  ("Q5_1",    6.0),    # block=32, 24 bytes
    8:  ("Q8_0",    8.5),    # block=32, 34 bytes
    9:  ("Q8_1",    9.0),    # block=32, 36 bytes
    10: ("Q2_K",    2.625),  # block=256, 84 bytes
    11: ("Q3_K",    3.4375), # block=256, 110 bytes
    12: ("Q4_K",    4.5),    # block=256, 144 bytes
    13: ("Q5_K",    5.5),    # block=256, 176 bytes
    14: ("Q6_K",    6.5625), # block=256, 210 bytes
    15: ("Q8_K",    8.125),  # block=256, 260 bytes
    16: ("IQ2_XXS", 2.0625), # block=256, 66 bytes
    17: ("IQ2_XS",  2.3125), # block=256, 74 bytes
    18: ("IQ3_XXS", 3.0625),
    19: ("IQ1_S",   1.5625),
    20: ("IQ4_NL",  4.5),    # block=32, 18 bytes -> 4.5 bpw
    21: ("IQ3_S",   3.4375),
    22: ("IQ2_S",   2.5),    # block=256, 80 bytes
    23: ("IQ4_XS",  4.25),   # block=256, 136 bytes
    24: ("I8",      8.0),
    25: ("I16",     16.0),
    26: ("I32",     32.0),
    27: ("I64",     64.0),
    28: ("F64",     64.0),
    29: ("IQ1_M",   1.75),
    30: ("BF16",    16.0),
}


def read_gguf_sizes(path):
    """Read GGUF and compute actual tensor sizes from type + ne (element count)."""
    tensors = []
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == GGUF_MAGIC, f"Not GGUF: {path}"
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        # Skip KV pairs
        # [CORRIGÉ 25/08/2026] Types conformes gguf.h (7=BOOL, 8=STRING,
        # 9=ARRAY) et ordre ARRAY corrigé : [elem_type i32][count u64][data]
        # (l'ancien code lisait [count][type] — inversé). Récursif pour les
        # chaînes imbriquées dans les tableaux.
        _KV_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                            10: 8, 11: 8, 12: 8}

        def skip_kv_value(fh, vt):
            if vt == 8:      # STRING
                sl = struct.unpack('<Q', fh.read(8))[0]
                fh.read(sl)
            elif vt == 9:    # ARRAY
                atype, na = struct.unpack('<IQ', fh.read(12))
                for _ in range(na):
                    skip_kv_value(fh, atype)
            else:
                size = _KV_SCALAR_SIZES.get(vt)
                if size is None:
                    raise ValueError(f"type KV GGUF inconnu: {vt}")
                fh.read(size)

        for _ in range(n_kv):
            kl = struct.unpack('<Q', f.read(8))[0]
            f.read(kl)
            vtype = struct.unpack('<I', f.read(4))[0]
            skip_kv_value(f, vtype)

        # Read tensor info
        data_offsets = []
        for _ in range(n_tensors):
            nlen = struct.unpack('<Q', f.read(8))[0]
            name = f.read(nlen).decode('utf-8')
            ndims = struct.unpack('<I', f.read(4))[0]
            shape = []
            for _ in range(ndims):
                shape.append(struct.unpack('<Q', f.read(8))[0])
            shape = shape[::-1]
            data_offset = struct.unpack('<Q', f.read(8))[0]
            ggml_type = struct.unpack('<I', f.read(4))[0]

            ne = 1
            for s in shape:
                ne *= s

            qname, bpw = QUANT_INFO.get(ggml_type, (f"T{ggml_type}", 4.0))
            size_bytes = ne * bpw / 8.0

            tensors.append({
                'name': name,
                'shape': shape,
                'ne': ne,
                'type': ggml_type,
                'qname': qname,
                'bpw': bpw,
                'size_bytes': size_bytes,
                'data_offset': data_offset,
            })
            data_offsets.append(data_offset)

        # File position = start of data section (with alignment)
        data_section_start = f.tell()
        # GGUF aligns data to 32 bytes
        data_section_start = ((data_section_start + 31) // 32) * 32

    file_size = os.path.getsize(path)
    data_section_size = file_size - data_section_start

    return tensors, file_size, data_section_start, data_section_size


def analyze_full(path, label):
    """Full analysis with actual file offsets."""
    tensors, file_size, data_start, data_size = read_gguf_sizes(path)
    
    # Calculate sizes from offsets (ground truth)
    offsets_sorted = sorted(enumerate(tensors), key=lambda x: x[1]['data_offset'])

    # Last tensor's end = data_size
    actual_sizes = []
    for i, (idx, t) in enumerate(offsets_sorted):
        if i + 1 < len(offsets_sorted):
            next_off = offsets_sorted[i+1][1]['data_offset']
        else:
            next_off = data_size
        # [CORRIGÉ 25/08/2026] NOTE : pour le dernier tenseur, data_size -
        # offset inclut le padding final d'alignement → taille surestimée.
        actual_size = next_off - t['data_offset']
        actual_sizes.append((idx, t, actual_size))
    
    total_actual = sum(s for _, _, s in actual_sizes)
    total_calc = sum(t['size_bytes'] for t in tensors)
    
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  File: {Path(path).name}")
    print(f"  File size:    {file_size/1e9:.3f} GB ({file_size/1e6:.0f} MB)")
    print(f"  Data section: {data_size/1e9:.3f} GB ({data_size/1e6:.0f} MB)")
    print(f"  Header:       {(data_start)/1e6:.1f} MB")
    print(f"  Calc weights: {total_calc/1e9:.3f} GB")
    print(f"  Actual data:  {total_actual/1e9:.3f} GB")
    ratio = total_actual / total_calc if total_calc > 0 else 0
    print(f"  Ratio actual/calc: {ratio:.3f}x")
    
    # Show biggest tensors by actual size
    actual_sizes.sort(key=lambda x: -x[2])
    print(f"\n  --- Top 30 tensors by ACTUAL stored size ---")
    print(f"  {'Tensor':<48} {'Quant':<10} {'Actual MB':>10} {'Calc MB':>10} {'Ratio':>6}")
    print(f"  {'-'*84}")
    for idx, t, actual in actual_sizes[:30]:
        calc = t['size_bytes']
        r = actual / calc if calc > 0 else 0
        print(f"  {t['name'][:47]:<48} {t['qname']:<10} {actual/1e6:>10.1f} {calc/1e6:>10.1f} {r:>6.2f}")
    
    # Calibrate BPW from actual sizes
    by_type = {}
    for idx, t, actual in actual_sizes:
        qn = t['qname']
        if qn not in by_type:
            by_type[qn] = {'count': 0, 'actual_bytes': 0, 'calc_bytes': 0, 'ne': 0}
        by_type[qn]['count'] += 1
        by_type[qn]['actual_bytes'] += actual
        by_type[qn]['calc_bytes'] += t['size_bytes']
        by_type[qn]['ne'] += t['ne']
    
    print(f"\n  --- Calibrated BPW by Quantization ---")
    print(f"  {'Format':<12} {'Count':>5} {'Actual GB':>10} {'Calc GB':>10} {'Calib BPW':>10} {'Old BPW':>8} {'Elements':>12}")
    print(f"  {'-'*67}")
    for qn in sorted(by_type.keys(), key=lambda q: -by_type[q]['actual_bytes']):
        v = by_type[qn]
        calib_bpw = v['actual_bytes'] * 8 / v['ne'] if v['ne'] > 0 else 0
        # Find old BPW
        old_bpw = 0
        for tid, (name, bpw) in QUANT_INFO.items():
            if name == qn:
                old_bpw = bpw
                break
        print(f"  {qn:<12} {v['count']:>5} {v['actual_bytes']/1e9:>10.3f} {v['calc_bytes']/1e9:>10.3f} {calib_bpw:>10.3f} {old_bpw:>8.1f} {v['ne']:>12,}")
    
    return by_type


if __name__ == '__main__':
    iq4 = "C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
    d2eco = "C:/Users/videl/Desktop/lama 1080-5070/models/Qwen3.6-35B-A3B-D2-ECO.gguf"
    
    r1 = analyze_full(iq4, "IQ4_NL (HauhauCS Aggressive)")
    r2 = analyze_full(d2eco, "D2-ECO (custom Q3)")
