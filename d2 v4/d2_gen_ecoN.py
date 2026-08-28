#!/usr/bin/env python3
"""D2-ECO intermediate: down Q2 sur couches 0..N (genou du Pareto)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "s1i_coldq2_types.txt")

def gen(out, n):
    dl = {f"blk.{l}" for l in range(0, n + 1)}
    lines = open(SRC, encoding="ascii").read().splitlines()
    out_lines, changed = [], 0
    for ln in lines:
        name, typ = ln.split("=", 1)
        lname = name.rsplit(".", 2)[0]
        if lname in dl and name.endswith("ffn_down_exps.weight"):
            typ = "Q2_K"; changed += 1
        out_lines.append(f"{name}={typ}")
    open(out, "w", encoding="ascii").write("\n".join(out_lines) + "\n")
    print(f"[ok] {out} ({changed} down->Q2 sur 0..{n})")

if __name__ == "__main__":
    gen(os.path.join(HERE, "s1i_ecoN11_types.txt"), 11)