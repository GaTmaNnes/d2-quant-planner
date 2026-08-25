#!/usr/bin/env python3
"""Fix DFlash GGUF metadata: add n_target_features = n_target_layers × hidden_size."""
import struct
import sys
import shutil

INPUT  = "models/Qwen3.6-35B-A3B-DFlash-CORRECT.gguf"
OUTPUT = "models/Qwen3.6-35B-A3B-DFlash-FIXED.gguf"

# Values to set
OVERRIDES = {
    b"dflash.n_target_features": (16384, "uint32"),
    b"dflash.n_target_layers":   (8,      "uint32"),
}

# GGUF value types
TYPE_UINT32 = 4
TYPE_UINT64 = 9

def main():
    shutil.copy2(INPUT, OUTPUT)
    
    with open(OUTPUT, "r+b") as f:
        # Read header
        magic = f.read(4)
        assert magic == b"GGUF", f"Not a GGUF file: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        
        print(f"GGUF v{version}, {n_tensors} tensors, {n_kv} KV pairs")
        
        # Read all KV pairs
        kv_pairs = []
        for i in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen)
            vtype = struct.unpack("<I", f.read(4))[0]
            
            if vtype == 1:  # STRING
                slen = struct.unpack("<Q", f.read(8))[0]
                val = f.read(slen)
                kv_pairs.append((key, vtype, val))
            elif vtype == 4:  # UINT32
                val = struct.unpack("<I", f.read(4))[0]
                kv_pairs.append((key, vtype, val))
            elif vtype == 5:  # INT32
                val = struct.unpack("<i", f.read(4))[0]
                kv_pairs.append((key, vtype, val))
            elif vtype == 6:  # FLOAT32
                val = struct.unpack("<f", f.read(4))[0]
                kv_pairs.append((key, vtype, val))
            elif vtype == 7:  # BOOL
                val = struct.unpack("<B", f.read(1))[0]
                kv_pairs.append((key, vtype, val))
            elif vtype == 8:  # UINT64
                val = struct.unpack("<Q", f.read(8))[0]
                kv_pairs.append((key, vtype, val))
            elif vtype == 9:  # ARRAY
                atype = struct.unpack("<I", f.read(4))[0]
                alen = struct.unpack("<Q", f.read(8))[0]
                # Skip array elements
                for _ in range(alen):
                    if atype == 1:  # STRING
                        slen = struct.unpack("<Q", f.read(8))[0]
                        f.read(slen)
                    elif atype == 4:  # UINT32
                        f.read(4)
                    elif atype == 5:  # INT32
                        f.read(4)
                    elif atype == 6:  # FLOAT32
                        f.read(4)
                    elif atype == 7:  # BOOL
                        f.read(1)
                    elif atype == 8:  # UINT64
                        f.read(8)
                    else:
                        print(f"Unknown array element type: {atype}")
                        break
                kv_pairs.append((key, vtype, (atype, alen)))
            else:
                print(f"Unknown type {vtype} for key {key}")
                break
        
        # Check if overrides already exist
        existing_keys = {kv[0] for kv in kv_pairs}
        added = 0
        for key, (val, typ) in OVERRIDES.items():
            if key in existing_keys:
                print(f"  Key '{key.decode()}' already exists, updating...")
                # Update in place
                for i, (k, vt, v) in enumerate(kv_pairs):
                    if k == key:
                        kv_pairs[i] = (k, TYPE_UINT32, val)
                        break
            else:
                print(f"  Adding key '{key.decode()}' = {val}")
                kv_pairs.append((key, TYPE_UINT32, val))
                added += 1
        
        print(f"Added {added} keys, total KV pairs: {len(kv_pairs)}")
        
        # Rebuild the GGUF file
        # This is complex - we need to rewrite the entire header
        # For now, let's just verify the approach works
        print("\nNote: Full GGUF rewrite needed. Using a simpler approach...")
        
        # Actually, let's just check if we can use the --override-kv with COPY
        # The issue might be that the binary expects the args in a specific order
        print("\nTrying llama-quantize with different arg order...")
        
    # Clean up
    import os
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)

if __name__ == "__main__":
    main()
