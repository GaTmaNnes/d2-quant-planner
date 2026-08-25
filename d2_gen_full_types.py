#!/usr/bin/env python3
"""Generate complete tensor-type files for gate_up vs down experiments.
Preserves original quantization for all non-target tensors."""
import sys, io, os
from gguf import GGUFReader, GGMLQuantizationType

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

QT = GGMLQuantizationType
TYPE_STR = {
    QT.F32: "F32", QT.F16: "F16", QT.Q4_0: "Q4_0", QT.Q4_1: "Q4_1",
    QT.Q5_0: "Q5_0", QT.Q5_1: "Q5_1", QT.Q8_0: "Q8_0", QT.Q8_1: "Q8_1",
    QT.Q2_K: "Q2_K", QT.Q3_K: "Q3_K", QT.Q4_K: "Q4_K", QT.Q5_K: "Q5_K",
    QT.Q6_K: "Q6_K", QT.Q8_K: "Q8_K",
    QT.IQ2_XXS: "IQ2_XXS", QT.IQ2_XS: "IQ2_XS", QT.IQ3_XXS: "IQ3_XXS",
    QT.IQ1_S: "IQ1_S", QT.IQ4_NL: "IQ4_NL", QT.IQ3_S: "IQ3_S",
    QT.IQ2_S: "IQ2_S", QT.IQ4_XS: "IQ4_XS",
    QT.I8: "I8", QT.I16: "I16", QT.I32: "I32", QT.I64: "I64",
    QT.F64: "F64", QT.IQ1_M: "IQ1_M", QT.BF16: "BF16",
}

SOURCE = "C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
OUTDIR = "C:/Users/videl/Desktop/lama 1080-5070/models/35b_exp"

print("Reading source GGUF...")
reader = GGUFReader(SOURCE)

# Get all tensors with their types
tensors = [(t.name, int(t.tensor_type)) for t in reader.tensors]
print(f"  {len(tensors)} tensors read")

variants = {
    "A_baseline_iq4nl": {},  # no overrides
    "B_down_q3k": {},        # ffn_down_exps -> Q3_K
    "C_gate_q3k": {},        # ffn_gate_exps + ffn_up_exps -> Q3_K
    "D_down_q3k_gate_iq4": {},  # down -> Q3_K, gate_up -> IQ4_NL (explicit)
    "E_down_q2k": {},        # ffn_down_exps -> Q2_K (extreme compression)
    "F_down_iq3xxs": {},     # ffn_down_exps -> IQ3_XXS (even more aggressive)
}

for name, qtype in tensors:
    qstr = TYPE_STR.get(qtype, "F32")
    
    for vname, overrides in variants.items():
        # Start with original type
        overrides[name] = qstr
    
    # Apply overrides for each variant
    if 'ffn_down_exps.weight' in name:
        variants["B_down_q3k"][name] = "Q3_K"
        variants["D_down_q3k_gate_iq4"][name] = "Q3_K"
        variants["E_down_q2k"][name] = "Q2_K"
        variants["F_down_iq3xxs"][name] = "IQ3_XXS"
    
    if 'ffn_gate_exps.weight' in name:
        variants["C_gate_q3k"][name] = "Q3_K"
        variants["D_down_q3k_gate_iq4"][name] = "IQ4_NL"
    
    if 'ffn_up_exps.weight' in name:
        variants["C_gate_q3k"][name] = "Q3_K"
        variants["D_down_q3k_gate_iq4"][name] = "IQ4_NL"

# Write files
os.makedirs(OUTDIR, exist_ok=True)
for vname, overrides in variants.items():
    path = os.path.join(OUTDIR, f"types_{vname}.txt")
    with open(path, 'w') as f:
        for name, qstr in sorted(overrides.items()):
            f.write(f"{name}={qstr}\n")
    n_overrides = sum(1 for n, q in overrides.items() 
                      if q != TYPE_STR.get(
                          next((qt for tn, qt in tensors if tn == n), QT.F32), "F32"))
    print(f"  {vname}: {len(overrides)} tensors, {n_overrides} overrides -> {path}")

print("\nDone. Files in:", OUTDIR)
