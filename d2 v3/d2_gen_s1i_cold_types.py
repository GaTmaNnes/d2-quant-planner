#!/usr/bin/env python3
"""D2 S1I-COLD (28/08/2026) — variant conservateur gate/up Q3_K sur couches froides.

Couches froides = blk.0-5 (imatrix gate/up 4-6x moins importante que profondeur).
Recette = S1I + gate/up exps -> Q3_K sur ces couches uniquement.
Source: s1i_types_all.txt (recette S1I validee).
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "s1i_types_all.txt")
OUT = os.path.join(HERE, "s1i_cold_types.txt")
COLD_LAYERS = [0, 1, 2, 3, 4, 5]

def main():
    cold = {f"blk.{l}" for l in COLD_LAYERS}
    lines = open(SRC, encoding="ascii").read().splitlines()
    out, changed = [], 0
    for ln in lines:
        name, typ = ln.split("=", 1)
        lname = name.rsplit(".", 2)[0]
        if lname in cold and (name.endswith("ffn_gate_exps.weight") or name.endswith("ffn_up_exps.weight")):
            typ = "Q3_K"; changed += 1
        out.append(f"{name}={typ}")
    open(OUT, "w", encoding="ascii").write("\n".join(out) + "\n")
    print(f"[ok] {OUT} ({len(out)} tensors, {changed} overrides gate/up->Q3_K sur {COLD_LAYERS})")
    import gguf, math
    r = gguf.GGUFReader(r"C:\Users\videl\Desktop\lama 1080-5070\models\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf")
    qs = gguf.GGML_QUANT_SIZES
    tot = 0
    for t in r.tensors:
        lv = dict(l.split("=") for l in out)[t.name]
        q = qs[getattr(gguf.GGMLQuantizationType, lv)]
        tot += int(math.prod(t.shape)) // q[0] * q[1]
    print(f"taille estimee: {tot/1024/1024:.0f} MiB")

if __name__ == "__main__":
    main()