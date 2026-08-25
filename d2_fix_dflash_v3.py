#!/usr/bin/env python3
"""
Fix DFlash GGUF: transpose fc.weight and add dflash.n_target_features metadata.
Uses raw binary patching since gguf Python lib metadata is tricky.
"""
import struct, os, sys

# ============================================================
# [CORRIGÉ 25/08/2026] ⚠ BANDEAU DANGER — SCRIPT DE CORRUPTION CONNUE ⚠
# 1. Transpose le bloc binaire BRUT de fc.weight quantisé Q8_0 : les octets
#    quantisés sont réinterprétés sans déquant/requant => poids invalides.
# 2. Re-type les métadonnées à l'aveugle via isinstance() : uint64/float32/
#    arrays gguf deviennent des strings => metadata corrompue.
# Résultat vérifié 24/08/2026 : le draft « D2FIX » produit est INUTILISABLE.
# Alternative saine : models/Qwen3.6-35B-A3B-DFlash-OFFICIAL-BF16.gguf
# ============================================================
if "--i-know-this-corrupts-tensors" not in sys.argv:
    print("[REFUSÉ] Ce script CORROMPT les tenseurs (transposition brute Q8_0")
    print("         + métadonnées stringifiées). Il ne tournera pas sans preuve")
    print("         que tu comprends : relance avec --i-know-this-corrupts-tensors.")
    sys.exit(1)

def read_gguf_header(data):
    """Parse GGUF header, return (header_end_offset, tensor_info_list, kv_dict)"""
    pos = 0
    magic = data[pos:pos+4]; pos += 4
    assert magic == b'GGUF', f"Not GGUF: {magic}"
    version = struct.unpack_from('<I', data, pos)[0]; pos += 4
    n_tensors = struct.unpack_from('<I', data, pos)[0]; pos += 4
    n_kv = struct.unpack_from('<I', data, pos)[0]; pos += 4
    
    print(f"  magic={magic} version={version} n_tensors={n_tensors} n_kv={n_kv}")
    return n_tensors, n_kv

def main():
    src = 'models/Qwen3.6-35B-A3B-DFlash-Q8_0.gguf'
    
    # First, let's use the GGUF Python library properly for the rewrite
    import gguf
    import numpy as np
    
    print("Reading source GGUF...")
    reader = gguf.GGUFReader(src, 'r')
    
    # Collect all info
    tensors = []
    fc_idx = None
    for i, t in enumerate(reader.tensors):
        data = t.data.copy()
        tensors.append({'name': t.name, 'shape': list(t.shape), 'data': data})
        if 'fc.weight' in t.name:
            fc_idx = i
            print(f"  Found fc.weight: shape={t.shape}, numpy shape={data.shape}")
    
    if fc_idx is None:
        print("ERROR: fc.weight not found!")
        return
    
    # Transpose fc.weight: (2048, 16384) -> (16384, 2048)
    fc = tensors[fc_idx]
    original_shape = fc['data'].shape
    fc['data'] = np.ascontiguousarray(fc['data'].T)
    fc['shape'] = list(fc['data'].shape)
    print(f"  Transposed fc.weight: {original_shape} -> {fc['data'].shape}")
    
    # Now write using a fresh writer
    # [CORRIGÉ 25/08/2026] Sortie distincte « -v3 » : v3 et proper écrasaient
    # mutuellement DFlash-D2FIX.gguf (impossible de savoir quel patcheur avait
    # produit le fichier présent).
    dst = 'models/Qwen3.6-35B-A3B-DFlash-D2FIX-v3.gguf'
    if os.path.exists(dst):
        os.remove(dst)
    
    # Get architecture from reader metadata
    arch = 'dflash'
    for k, v in reader.metadata.items():
        if k == 'general.architecture':
            arch = str(v)
            break
    
    print(f"  Architecture: {arch}")
    writer = gguf.GGUFWriter(dst, arch)
    
    # Write ALL metadata from source, adding n_target_features
    n_target_added = False
    for key, val in reader.metadata.items():
        # The gguf library stores values differently depending on type
        # We need to handle them properly
        try:
            # Try to detect type from the value
            if isinstance(val, int) or (hasattr(val, 'item') and isinstance(val.item(), int)):
                writer.add_uint32(key, int(val))
            elif isinstance(val, float) or (hasattr(val, 'item') and isinstance(val.item(), float)):
                writer.add_float32(key, float(val))
            elif isinstance(val, bool):
                writer.add_bool(key, val)
            elif isinstance(val, str):
                writer.add_string(key, val)
            elif isinstance(val, bytes):
                writer.add_string(key, val.decode('utf-8', errors='replace'))
            else:
                writer.add_string(key, str(val))
        except Exception as e:
            print(f"  Warning: could not add KV {key}: {e}")
    
    # Add dflash.n_target_features = 16384
    try:
        writer.add_uint32('dflash.n_target_features', 16384)
        n_target_added = True
        print("  Added dflash.n_target_features = 16384")
    except Exception as e:
        print(f"  ERROR adding n_target_features: {e}")
    
    # Write tensors
    for t in tensors:
        try:
            writer.add_tensor(t['name'], t['data'])
        except Exception as e:
            print(f"  ERROR writing tensor {t['name']}: {e}")
            # Try with raw_shape
            try:
                writer.add_tensor(t['name'], t['data'], raw_shape=t['shape'])
                print(f"  OK with raw_shape for {t['name']}")
            except Exception as e2:
                print(f"  FATAL: {t['name']}: {e2}")
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"\nOutput: {dst}")
    print(f"Size: {os.path.getsize(dst) / 1024 / 1024:.1f} MB")
    
    # Verify
    print("\nVerifying...")
    reader2 = gguf.GGUFReader(dst, 'r')
    for t in reader2.tensors:
        if 'fc.weight' in t.name:
            print(f"  fc.weight: shape={t.shape}")
            break
    # Check n_target_features
    for k, v in reader2.metadata.items():
        if 'n_target' in k or 'dflash' in k:
            print(f"  {k} = {v}")

if __name__ == '__main__':
    main()
