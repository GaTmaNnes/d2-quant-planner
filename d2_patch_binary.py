#!/usr/bin/env python3
"""Binary-patch DFlash GGUF to add metadata."""
import struct
import shutil

INPUT  = "models/Qwen3.6-35B-A3B-DFlash-CORRECT.gguf"
OUTPUT = "models/Qwen3.6-35B-A3B-DFlash-PATCHED.gguf"

shutil.copy2(INPUT, OUTPUT)

with open(OUTPUT, 'r+b') as f:
    # Read GGUF header
    magic = f.read(4)
    assert magic == b'GGUF'
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    
    print(f"Original: v{version}, {n_tensors} tensors, {n_kv} KV pairs")
    
    # Read KV pairs to find the end of KV section
    kv_start = f.tell()
    
    # Parse all KV pairs to find their total size
    total_kv_size = 0
    for i in range(n_kv):
        # Read key length
        klen_bytes = f.read(8)
        klen = struct.unpack('<Q', klen_bytes)[0]
        total_kv_size += 8 + klen  # key length + key data
        
        # Read value type
        vtype = struct.unpack('<I', f.read(4))[0]
        total_kv_size += 4
        
        # Read value based on type
        if vtype == 1:  # STRING
            slen = struct.unpack('<Q', f.read(8))[0]
            f.read(slen)
            total_kv_size += 8 + slen
        elif vtype == 4:  # UINT32
            f.read(4)
            total_kv_size += 4
        elif vtype == 5:  # INT32
            f.read(4)
            total_kv_size += 4
        elif vtype == 6:  # FLOAT32
            f.read(4)
            total_kv_size += 4
        elif vtype == 7:  # BOOL
            f.read(1)
            total_kv_size += 1
        elif vtype == 8:  # UINT64
            f.read(8)
            total_kv_size += 8
        elif vtype == 9:  # ARRAY
            atype = struct.unpack('<I', f.read(4))[0]
            alen = struct.unpack('<Q', f.read(8))[0]
            total_kv_size += 4 + 8
            for _ in range(alen):
                if atype == 1:  # STRING
                    sl = struct.unpack('<Q', f.read(8))[0]
                    f.read(sl)
                    total_kv_size += 8 + sl
                elif atype in (4, 5):  # UINT32, INT32
                    f.read(4)
                    total_kv_size += 4
                elif atype == 6:  # FLOAT32
                    f.read(4)
                    total_kv_size += 4
                elif atype == 7:  # BOOL
                    f.read(1)
                    total_kv_size += 1
                elif atype == 8:  # UINT64
                    f.read(8)
                    total_kv_size += 8
        else:
            print(f"Unknown type {vtype} at KV pair {i}")
            break
    
    kv_end = f.tell()
    print(f"KV section ends at offset {kv_end}")
    
    # Build new KV pairs to append
    def encode_string(s):
        b = s.encode('utf-8')
        return struct.pack('<Q', len(b)) + b
    
    new_kv = bytearray()
    # dflash.n_target_features = 16384 (UINT32)
    new_kv += encode_string("dflash.n_target_features")
    new_kv += struct.pack('<I', 4)  # type UINT32
    new_kv += struct.pack('<I', 16384)
    
    # dflash.n_target_layers = 8 (UINT32)
    new_kv += encode_string("dflash.n_target_layers")
    new_kv += struct.pack('<I', 4)  # type UINT32
    new_kv += struct.pack('<I', 8)
    
    new_n_kv = n_kv + 2
    
    # Now we need to:
    # 1. Go back to the header
    # 2. Update n_kv
    # 3. Read everything after KV section
    # 4. Write: header (with updated n_kv) + old KV data + new KV data + tensor data
    
    # Read tensor data
    tensor_data = f.read()
    
    # Rewrite the file
    f.seek(0)
    f.write(b'GGUF')
    f.write(struct.pack('<I', version))
    f.write(struct.pack('<Q', n_tensors))
    f.write(struct.pack('<Q', new_n_kv))
    # Write old KV data
    f.seek(kv_end)
    f.write(bytes(new_kv))
    # Write tensor data
    f.write(tensor_data)
    f.truncate()
    
    print(f"Patched: {OUTPUT} ({new_n_kv} KV pairs)")

# Verify
with open(OUTPUT, 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    print(f"Verified: v{version}, {n_tensors} tensors, {n_kv} KV pairs")
