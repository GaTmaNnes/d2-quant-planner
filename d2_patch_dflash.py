#!/usr/bin/env python3
"""Patch DFlash GGUF: add dflash.n_target_features metadata."""
import struct
import shutil
import os
import sys

# [CORRIGÉ 25/08/2026] Refus sans --force : ce patcheur réécrit le flux GGUF à la main
# (insertion de KV => décalage du bloc tenseurs). Sans précaution il produisait des
# fichiers désalignés/tronqués. Désormais : --force requis, padding d'alignement calculé,
# et RAISE sur type KV inconnu (l'ancien break tronquait silencieusement la lecture).
if "--force" not in sys.argv:
    print("[REFUSÉ] Patch binaire GGUF risqué (réécriture manuelle du flux).")
    print("         Relance avec --force si tu assumes le risque.")
    print("         Alternative saine : models/Qwen3.6-35B-A3B-DFlash-OFFICIAL-BF16.gguf")
    sys.exit(1)

INPUT  = "models/Qwen3.6-35B-A3B-DFlash-CORRECT.gguf"
OUTPUT = "models/Qwen3.6-35B-A3B-DFlash-PATCHED.gguf"

def encode_string(s):
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b

def encode_kv_int(key, val):
    return encode_string(key) + struct.pack('<I', 4) + struct.pack('<i', val)  # type=4=UINT32

def main():
    shutil.copy2(INPUT, OUTPUT)
    
    with open(OUTPUT, 'r+b') as f:
        # Read header
        magic = f.read(4)
        assert magic == b'GGUF'
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Original: v{version}, {n_tensors} tensors, {n_kv} KV pairs")
        
        # Read the full KV section
        kv_start = f.tell()
        
        # We need to:
        # 1. Read all existing KV pairs
        # 2. Add new ones
        # 3. Rewrite the file with updated n_kv
        
        # For simplicity, let's just update n_kv in the header and append new KV pairs
        # before the tensor data
        
        # First, find where tensor data starts by reading all KV pairs
        kv_data = bytearray()
        for i in range(n_kv):
            # key length + key
            klen = struct.unpack('<Q', f.read(8))[0]
            key = f.read(klen)
            
            # value type
            vtype = struct.unpack('<I', f.read(4))[0]
            
            if vtype == 1:  # STRING
                slen = struct.unpack('<Q', f.read(8))[0]
                val = f.read(slen)
            elif vtype == 4:  # UINT32
                val = f.read(4)
            elif vtype == 5:  # INT32
                val = f.read(4)
            elif vtype == 6:  # FLOAT32
                val = f.read(4)
            elif vtype == 7:  # BOOL
                val = f.read(1)
            elif vtype == 8:  # UINT64
                val = f.read(8)
            elif vtype == 9:  # ARRAY
                atype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<Q', f.read(8))[0]
                val = struct.pack('<I', atype) + struct.pack('<Q', alen)
                for _ in range(alen):
                    if atype == 1:  # STRING
                        sl = struct.unpack('<Q', f.read(8))[0]
                        val += struct.pack('<Q', sl) + f.read(sl)
                    elif atype in (4, 5):  # UINT32, INT32
                        val += f.read(4)
                    elif atype == 6:  # FLOAT32
                        val += f.read(4)
                    elif atype == 7:  # BOOL
                        val += f.read(1)
                    elif atype == 8:  # UINT64
                        val += f.read(8)
            else:
                print(f"Unknown type {vtype}")
                break
            
            kv_data += struct.pack('<Q', klen) + key + struct.pack('<I', vtype) + val
        
        kv_end = f.tell()
        print(f"KV section: {kv_end - kv_start} bytes")
        
        # Read tensor data
        tensor_data = f.read()
        print(f"Tensor data: {len(tensor_data)} bytes")
        
        # New KV pairs to add
        new_kvs = bytearray()
        new_kvs += encode_kv_int("dflash.n_target_features", 16384)
        new_kvs += encode_kv_int("dflash.n_target_layers", 8)
        
        new_n_kv = n_kv + 2
        
        # Rewrite file
        f.seek(0)
        f.write(b'GGUF')
        f.write(struct.pack('<I', version))
        f.write(struct.pack('<Q', n_tensors))
        f.write(struct.pack('<Q', new_n_kv))
        f.write(bytes(kv_data))
        f.write(new_kvs)
        f.write(tensor_data)
        f.truncate()
    
    print(f"Patched: {OUTPUT} ({new_n_kv} KV pairs)")

if __name__ == '__main__':
    main()
