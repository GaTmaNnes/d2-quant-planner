#!/usr/bin/env python3
"""
Fix the DFlash GGUF for Qwen3.6-35B-A3B:
1. fc.weight in safetensors is (2048, 16384) — needs to be (16384, 2048) for llama.cpp
2. Add dflash.n_target_features = 16384 metadata
"""
import struct, sys, os, shutil
import numpy as np

# ============================================================
# [CORRIGÉ 25/08/2026] ⚠ BANDEAU DANGER — SCRIPT DE CORRUPTION CONNUE ⚠
# 1. Transposition BRUTE du bloc quantisé Q8_0 de fc.weight : octets
#    réinterprétés sans déquant/requant => poids invalides.
# 2. TOUTES les métadonnées sont réécrites via add_string(key, str(val)) :
#    uint32/float/arrays gguf deviennent des strings => metadata corrompue.
# Résultat vérifié 24/08/2026 : draft « D2FIX » INUTILISABLE.
# Alternative saine : models/Qwen3.6-35B-A3B-DFlash-OFFICIAL-BF16.gguf
# ============================================================
if "--i-know-this-corrupts-tensors" not in sys.argv:
    print("[REFUSÉ] Ce script CORROMPT les tenseurs (transposition brute Q8_0")
    print("         + métadonnées toutes stringifiées). Il ne tournera pas sans")
    print("         preuve que tu comprends : relance avec --i-know-this-corrupts-tensors.")
    sys.exit(1)

def fix_dflash_gguf(input_path, output_path):
    """
    Read the GGUF, find fc.weight tensor data, transpose it, and add KV metadata.
    
    GGUF v3 format:
    - Header: magic(4) + version(4) + n_tensors(4) + n_kv(4)
    - KV pairs
    - Tensor info
    - Padding to alignment
    - Tensor data
    """
    import gguf
    
    reader = gguf.GGUFReader(input_path, 'r')
    
    # Print all tensor shapes
    print("=== Tensor shapes ===")
    fc_data = None
    fc_name_idx = None
    for i, tensor in enumerate(reader.tensors):
        if 'fc.weight' in tensor.name:
            print(f"  {tensor.name}: shape={tensor.shape}, dtype={tensor.data.dtype}")
            fc_data = tensor.data.copy()
            fc_name_idx = i
        elif i < 5 or 'fc' in tensor.name.lower() or 'hidden' in tensor.name.lower():
            print(f"  {tensor.name}: shape={tensor.shape}")
    
    if fc_data is None:
        print("ERROR: fc.weight not found!")
        return
    
    print(f"\n=== fc.weight original shape: {fc_data.shape} ===")
    
    # Transpose fc.weight: (2048, 16384) -> (16384, 2048)
    fc_transposed = fc_data.T.copy()
    print(f"=== fc.weight transposed shape: {fc_transposed.shape} ===")
    
    # Now write a new GGUF using the writer
    writer = gguf.GGUFWriter(output_path, arch=reader.metadata.get('general.architecture', 'dflash'))
    
    # Copy all metadata
    for key, val in reader.metadata.items():
        try:
            writer.add_string(key, str(val))
        except:
            pass
    
    # Add the missing dflash.n_target_features
    writer.add_uint32('dflash.n_target_features', 16384)
    print("Added dflash.n_target_features = 16384")
    
    # Copy all tensors, replacing fc.weight with transposed version
    for tensor in reader.tensors:
        if 'fc.weight' in tensor.name:
            print(f"  Replacing {tensor.name}: {tensor.shape} -> {fc_transposed.shape}")
            writer.add_tensor(tensor.name, fc_transposed, raw_shape=tuple(fc_transposed.shape))
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_shape=tuple(tensor.shape))
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"\n=== Output: {output_path} ===")
    print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

if __name__ == '__main__':
    src = 'models/Qwen3.6-35B-A3B-DFlash-Q8_0.gguf'
    # [CORRIGÉ 25/08/2026] Sortie distincte « -proper » : v3 et proper écrasaient
    # mutuellement DFlash-D2FIX.gguf.
    dst = 'models/Qwen3.6-35B-A3B-DFlash-D2FIX-proper.gguf'
    fix_dflash_gguf(src, dst)
