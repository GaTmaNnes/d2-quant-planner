#!/usr/bin/env python3
"""D2 Q2 VARIANTS (28/08/2026) — gate/up -> Q2_K sur couches COLD vs HOT.

Base = recette S1I (s1i_types_all.txt : gate/up IQ3_S, down Q3_K, router Q6_K).
Variant A (COLD) : gate/up -> Q2_K sur couches 0-5   (experts freq=0, imatrix basse)
Variant B (HOT)  : gate/up -> Q2_K sur couches 10,13,14,15,17,35 (experts freq>1000)
Même nb de couches -> économie mémoire identique -> ΔPPL comparable.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "s1i_types_all.txt")
COLD = [0, 1, 2, 3, 4, 5]
HOT = [10, 13, 14, 15, 17, 35]
MEDIUM = [18, 31, 6, 23, 12, 19]   # freq intermediaire, SANS overlap avec TIER3
TIER3 = [20, 21, 25, 27, 29, 30]   # imatrix gate/up 2.63-3.22e8 (bande 2.5-3.5e8)

def gen(layers, out):
    cold = {f"blk.{l}" for l in layers}
    lines = open(SRC, encoding="ascii").read().splitlines()
    out_lines, changed = [], 0
    for ln in lines:
        name, typ = ln.split("=", 1)
        lname = name.rsplit(".", 2)[0]
        if lname in cold and (name.endswith("ffn_gate_exps.weight") or name.endswith("ffn_up_exps.weight")):
            typ = "Q2_K"; changed += 1
        out_lines.append(f"{name}={typ}")
    open(out, "w", encoding="ascii").write("\n".join(out_lines) + "\n")
    print(f"[ok] {out} ({changed} gate/up->Q2_K sur {layers})")

gen(COLD, os.path.join(HERE, "s1i_coldq2_types.txt"))
gen(HOT, os.path.join(HERE, "s1i_hotq2_types.txt"))
gen(MEDIUM, os.path.join(HERE, "s1i_medq2_types.txt"))
gen(TIER3, os.path.join(HERE, "s1i_tier3_types.txt"))