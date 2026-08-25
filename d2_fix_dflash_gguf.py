#!/usr/bin/env python3
"""Add dflash.n_target_features to DFlash GGUF using gguf library."""
import gguf
import sys

INPUT  = "models/Qwen3.6-35B-A3B-DFlash-CORRECT.gguf"
OUTPUT = "models/Qwen3.6-35B-A3B-DFlash-FIXED.gguf"

# Read existing GGUF
reader = gguf.GGUFReader(INPUT)

# Create writer with same arch
writer = gguf.GGUFWriter(OUTPUT, arch="dflash")

# Copy all existing KV pairs
for name, field in reader.fields.items():
    data = reader.get_field(name)
    if data is None:
        continue
    
    types = data.types
    raw_data = data.data
    
    if not types:
        continue
    
    val_type = types[0]
    
    if val_type == gguf.GGUFValueType.STRING:
        # Get the string value
        parts = data.parts
        # The string data is in the last parts
        str_bytes = b''
        for i in range(1, len(parts)):
            part = parts[i]
            if hasattr(part, 'tobytes'):
                str_bytes += part.tobytes()
            else:
                str_bytes += bytes(part)
        writer.add_key(name, str_bytes.decode('utf-8'))
    elif val_type == gguf.GGUFValueType.UINT32:
        val = int(raw_data[0]) if len(raw_data) > 0 else 0
        writer.add_key(name, val)
    elif val_type == gguf.GGUFValueType.INT32:
        val = int(raw_data[0]) if len(raw_data) > 0 else 0
        writer.add_key(name, val)
    elif val_type == gguf.GGUFValueType.FLOAT32:
        val = float(raw_data[0]) if len(raw_data) > 0 else 0.0
        writer.add_key(name, val)
    elif val_type == gguf.GGUFValueType.BOOL:
        val = bool(raw_data[0]) if len(raw_data) > 0 else False
        writer.add_key(name, val)
    elif val_type == gguf.GGUFValueType.UINT64:
        val = int(raw_data[0]) if len(raw_data) > 0 else 0
        writer.add_key(name, val)
    elif val_type == gguf.GGUFValueType.ARRAY:
        # Skip arrays for now
        pass

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
