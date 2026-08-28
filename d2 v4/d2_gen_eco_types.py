#!/usr/bin/env python3
"""D2 ECO series (28/08/2026) — descendre les tensors tolerants vers Q2_K.
ECO-1 : down_exps Q3_K -> Q2_K sur TOUTES les couches (SNR ~76 dB, tolerant).
Base : recette COLDQ2 (s1i_coldq2_types.txt).
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "s1i_coldq2_types.txt")

def gen(out, down_all_q2=False, gate_q2_extra=None):
    gate_q2_extra = gate_q2_extra or []
    gl = {f"blk.{l}" for l in gate_q2_extra}
    lines = open(SRC, encoding="ascii").read().splitlines()
    out_lines, changed = [], 0
    for ln in lines:
        name, typ = ln.split("=", 1)
        lname = name.rsplit(".", 2)[0]
        if down_all_q2 and name.endswith("ffn_down_exps.weight"):
            typ = "Q2_K"; changed += 1
        elif lname in gl and (name.endswith("ffn_gate_exps.weight") or name.endswith("ffn_up_exps.weight")):
            typ = "Q2_K"; changed += 1
        out_lines.append(f"{name}={typ}")
    open(out, "w", encoding="ascii").write("\n".join(out_lines) + "\n")
    print(f"[ok] {out} ({changed} overrides)")

gen(os.path.join(HERE, "s1i_eco1_types.txt"), down_all_q2=True)
if "--eco2" in sys.argv:
    gen(os.path.join(HERE, "s1i_eco2_types.txt"), down_all_q2=True, gate_q2_extra=[6,7,8,9,11,12])