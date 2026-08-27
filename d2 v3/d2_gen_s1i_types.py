#!/usr/bin/env python3
"""Génère s1i_types_all.txt depuis UD-IQ4_NL (26/08/2026).

Recette S-1I = recette D2 validée, reconstruite proprement :
  - ffn_down_exps : Q3_K   (LA modification — depuis du 4.5 bpw propre)
  - gate/up experts, shexp, attn, embd, output : IQ4_NL INCHANGÉS (zéro dégradation)
  - routeurs ffn_gate_inp : Q6_K (protection router — littérature MoE)
  - normes + ssm_conv1d : F16 (verrou structurel)

Source : UD-IQ4_NL uniforme (16.79 GiB) → sortie ≈ 15.7 GiB estimé.
"""
import gguf, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "models", "Qwen3.6-35B-A3B-UD-IQ4_NL.gguf")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s1i_types_all.txt")

r = gguf.GGUFReader(SRC)
lines = []
counts = {}
for t in r.tensors:
    name, cur = t.name, t.tensor_type.name
    if "norm" in name:                       # inclut output_norm (sous-ensemble)
        lv = "F16"
    elif name.endswith("ssm_conv1d.weight"):
        lv = "F16"
    elif "ffn_gate_inp" in name:            # routeurs MoE (protégés, littérature)
        lv = "Q6_K"
    elif "ffn_down_exps" in name:
        lv = "Q3_K"                          # ← LE changement S-1I
    else:
        lv = cur                             # gate/up/shexp/attn/embd/output : inchangés
    counts[lv] = counts.get(lv, 0) + 1
    lines.append(f"{name}={lv}")

open(OUT, "w", encoding="ascii").write("\n".join(lines) + "\n")
qs = gguf.GGML_QUANT_SIZES
lv_by_name = dict(l.split("=") for l in lines)   # construit UNE fois (évite le O(n²))
tot = 0
for t in r.tensors:
    q = qs[getattr(gguf.GGMLQuantizationType, lv_by_name[t.name])]
    tot += int(math.prod(t.shape)) // q[0] * q[1]
print(f"[ok] {OUT} ({len(lines)} tenseurs)")
print("mix:", dict(sorted(counts.items())))
print(f"taille estimee: {tot/1024/1024:.0f} MiB")
