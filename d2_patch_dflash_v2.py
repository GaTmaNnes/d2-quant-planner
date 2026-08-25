#!/usr/bin/env python3
"""Patch DFlash GGUF: add dflash.n_target_features using gguf library."""
import gguf
import numpy as np
import sys

INPUT  = "models/Qwen3.6-35B-A3B-DFlash-CORRECT.gguf"
OUTPUT = "models/Qwen3.6-35B-A3B-DFlash-PATCHED.gguf"

reader = gguf.GGUFReader(INPUT)

# Get arch from metadata
arch = None
for name, field in reader.fields.items():
    if name == "general.architecture":
        data = reader.get_field(name)
        if data and data.data is not None:
            # String type: data is the string bytes
            arch = data.parts[data.data[0]].tobytes().decode('utf-8') if hasattr(data.parts[data.data[0]], 'tobytes') else str(data.data)
            break

print(f"Architecture: {arch}")

# Create writer
writer = gguf.GGUFWriter(OUTPUT, arch=arch)

# Copy ALL existing KV fields
for name, field in reader.fields.items():
    data = reader.get_field(name)
    if data is None or name == "general.architecture":
        continue
    
    types = list(data.types) if data.types else []
    raw_data = list(data.data) if data.data is not None else []
    
    if not types:
        continue
    
    val_type = types[0]
    
    try:
        if val_type == gguf.GGUFValueType.STRING:
            # Get string from parts
            idx = raw_data[0] if raw_data else 0
            if idx < len(data.parts):
                part = data.parts[idx]
                s = part.tobytes().decode('utf-8') if hasattr(part, 'tobytes') else bytes(part).decode('utf-8')
                writer.add_key(name, s)
        elif val_type == gguf.GGUFValueType.UINT32:
            writer.add_key(name, int(raw_data[0]) if raw_data else 0)
        elif val_type == gguf.GGUFValueType.INT32:
            writer.add_key(name, int(raw_data[0]) if raw_data else 0)
        elif val_type == gguf.GGUFValueType.FLOAT32:
            writer.add_key(name, float(raw_data[0]) if raw_data else 0.0)
        elif val_type == gguf.GGUFValueType.BOOL:
            writer.add_key(name, bool(raw_data[0]) if raw_data else False)
        elif val_type == gguf.GGUFValueType.UINT64:
            writer.add_key(name, int(raw_data[0]) if raw_data else 0)
    except Exception as e:
        print(f"  Skip {name}: {e}")

# Add the missing metadata
print("Adding dflash.n_target_features = 16384")
writer.add_key("dflash.n_target_features", 16384)
print("Adding dflash.n_target_layers = 8")
writer.add_key("dflash.n_target_layers", 8)

# Copy all tensors
for tensor in reader.tensors:
    writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)

# Write
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

print(f"Written to {OUTPUT}")

# Verify
r2 = gguf.GGUFReader(OUTPUT)
print(f"\nVerification:")
for name in sorted(r2.fields.keys()):
    if 'n_target' in name or 'arch' in name:
        data = r2.get_field(name)
        if data and data.data is not None:
            print(f"  {name}: {list(data.data)}")
