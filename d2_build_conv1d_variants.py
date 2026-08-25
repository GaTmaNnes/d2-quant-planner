#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a GGUF variant where ONLY the ssm_conv1d tensors are re-typed.
Everything else is copied byte-for-byte (metadata + tensor bytes).

Usage:
  python d2_build_conv1d_variants.py --conv1d-type f16
  python d2_build_conv1d_variants.py --conv1d-type q8_0   # structural-constraint test

The base model stores conv1d as F32 shape [4, 8192] (ne[0] = 4).
 - F16 : block_size 1 -> valid.
 - Q8_0: block_size 32 -> ne[0]=4 is not a multiple of 32 -> llama.cpp will reject it.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "beellama.cpp", "gguf-py"))
from gguf import GGUFReader, GGUFWriter
from gguf import quants
from gguf.constants import GGUFValueType, GGMLQuantizationType

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "models", "Qwen3.5-9B-Q4_K_S.gguf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv1d-type", default="f16", help="f16 | q8_0 | q4_0")
    ap.add_argument("--src", default=SRC)
    args = ap.parse_args()

    conv_type = getattr(GGMLQuantizationType, args.conv1d_type.upper())
    dst = os.path.join(
        os.path.dirname(args.src),
        f"Qwen3.5-9B-Q4_K_S.conv1d_{args.conv1d_type}.gguf",
    )

    r = GGUFReader(args.src)
    arch = str(r.fields.get("general.architecture").contents())
    w = GGUFWriter(dst, arch=arch, use_temp_file=True)

    # 1) copy all metadata verbatim (arch/version/counts auto-written by the writer)
    for key, field in r.fields.items():
        if key in ("general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        w.add_key_value(key, field.contents(), vtype, sub_type)

    # 2) tensors: pass-through except conv1d
    n_conv = 0
    for t in r.tensors:
        is_conv = ("ssm_conv1d" in t.name) and (t.tensor_type == GGMLQuantizationType.F32)
        if is_conv:
            if conv_type == GGMLQuantizationType.F16:
                data = np.asarray(t.data, dtype=np.float32).astype(np.float16)
                w.add_tensor(t.name, data, raw_shape=data.shape, raw_dtype=GGMLQuantizationType.F16)
            else:
                # flat block quantization (for the structural-constraint test)
                flat = np.asarray(t.data, dtype=np.float32).ravel()
                q = quants.quantize(flat, conv_type)
                w.add_tensor(t.name, q, raw_shape=q.shape, raw_dtype=conv_type)
            n_conv += 1
        else:
            w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"[+] variant écrit : {dst}")
    print(f"    conv1d re-typés : {n_conv}")

    # 3) verify: compare non-conv1d tensors byte-for-byte + report conv1d types
    r2 = GGUFReader(dst)
    assert len(r2.tensors) == len(r.tensors), "tensor count mismatch"
    mismatches = 0
    for a, b in zip(r.tensors, r2.tensors):
        if "ssm_conv1d" in a.name:
            continue
        if a.tensor_type != b.tensor_type or a.n_bytes != b.n_bytes or \
           not np.array_equal(np.asarray(a.data).ravel(), np.asarray(b.data).ravel()):
            mismatches += 1
            if mismatches <= 5:
                print(f"    MISMATCH: {a.name}")
    print(f"    non-conv1d tensors byte-identiques : {'OK' if mismatches == 0 else f'{mismatches} différences!'}")
    for b in r2.tensors:
        if "ssm_conv1d" in b.name:
            print(f"    {b.name}: {b.tensor_type.name} {b.n_bytes} bytes")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
