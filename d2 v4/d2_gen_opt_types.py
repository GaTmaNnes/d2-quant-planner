#!/usr/bin/env python3
"""D2-OPT series (28/08/2026) — budget constant.
A: down_exps -> Q2_K sur couches 0-5   (liberation)
B: gate/up   -> Q4_K sur couches 30-38 (reinvestissement)
C: A + B (D2-OPT complet, budget ~constant)
Base: recette COLDQ2 (s1i_coldq2_types.txt).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "s1i_coldq2_types.txt")
COLD_DOWN = list(range(0, 6))
HOT_GATE = list(range(30, 39))

def gen(out, down_q2=None, gate_q4=None):
    down_q2 = down_q2 or []
    gate_q4 = gate_q4 or []
    dl = {f"blk.{l}" for l in down_q2}
    gl = {f"blk.{l}" for l in gate_q4}
    lines = open(SRC, encoding="ascii").read().splitlines()
    out_lines, changed = [], 0
    for ln in lines:
        name, typ = ln.split("=", 1)
        lname = name.rsplit(".", 2)[0]
        if lname in dl and name.endswith("ffn_down_exps.weight"):
            typ = "Q2_K"; changed += 1
        elif lname in gl and (name.endswith("ffn_gate_exps.weight") or name.endswith("ffn_up_exps.weight")):
            typ = "Q4_K"; changed += 1
        out_lines.append(f"{name}={typ}")
    open(out, "w", encoding="ascii").write("\n".join(out_lines) + "\n")
    print(f"[ok] {out} ({changed} overrides)")

gen(os.path.join(HERE, "s1i_optA_types.txt"), down_q2=COLD_DOWN)
gen(os.path.join(HERE, "s1i_optB_types.txt"), gate_q4=HOT_GATE)
gen(os.path.join(HERE, "s1i_optC_types.txt"), down_q2=COLD_DOWN, gate_q4=HOT_GATE)