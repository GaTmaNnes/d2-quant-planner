#!/usr/bin/env python3
"""
D2 Per-Layer Down Allocation
Linear attn (0-29) = less sensitive → Q2/IQ3_XXS on down
Full attn (30-39) = more sensitive → Q3_K on down
gate_up stays IQ4_NL everywhere.
"""
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
}

SOURCE = "C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
OUTDIR = "C:/Users/videl/Desktop/lama 1080-5070/models/35b_exp"

LINEAR = set(range(0, 30))
FULL = set(range(30, 40))


def get_layer(name):
    for prefix in ['blk.', 'layers.']:
        if prefix in name:
            try:
                return int(name.split(prefix)[1].split('.')[0])
            except:
                pass
    return None


print("Reading source GGUF...")
reader = GGUFReader(SOURCE)
tensors = [(t.name, int(t.tensor_type)) for t in reader.tensors]
print(f"  {len(tensors)} tensors")

# Define per-layer strategies
# Key insight: linear_attn layers are 4.2x less sensitive → more aggressive
strategies = {
    # Variant B: all down = Q3_K (measured baseline)
    "J_linearQ2_fullQ3": {
        "desc": "linear down=Q2_K, full down=Q3_K",
        "linear_down": "Q2_K",
        "full_down": "Q3_K",
    },
    # Variant: linear down = IQ3_XXS, full down = Q3_K
    "K_linearIQ3_fullQ3": {
        "desc": "linear down=IQ3_XXS, full down=Q3_K",
        "linear_down": "IQ3_XXS",
        "full_down": "Q3_K",
    },
    # Variant: linear down = Q2_K, full down = Q4_K (less aggressive on full)
    "L_linearQ2_fullQ4": {
        "desc": "linear down=Q2_K, full down=Q4_K",
        "linear_down": "Q2_K",
        "full_down": "Q4_K",
    },
    # Variant: linear down = IQ3_XXS, full down = IQ4_NL (most aggressive)
    "M_linearIQ3_fullIQ4": {
        "desc": "linear down=IQ3_XXS, full down=IQ4_NL",
        "linear_down": "IQ3_XXS",
        "full_down": "IQ4_NL",
    },
    # Variant: all down = Q2_K (uniform aggressive)
    "N_allDownQ2": {
        "desc": "all down=Q2_K",
        "linear_down": "Q2_K",
        "full_down": "Q2_K",
    },
}

for sname, sinfo in strategies.items():
    overrides = {}
    n_changed = 0

    for name, qtype in tensors:
        qstr = TYPE_STR.get(qtype, "F32")
        overrides[name] = qstr

        layer = get_layer(name)
        if layer is None:
            continue

        # Only override ffn_down_exps.weight
        if 'ffn_down_exps.weight' not in name:
            continue

        if layer in LINEAR:
            overrides[name] = sinfo["linear_down"]
            n_changed += 1
        elif layer in FULL:
            overrides[name] = sinfo["full_down"]
            n_changed += 1

    path = os.path.join(OUTDIR, f"types_{sname}.txt")
    with open(path, 'w') as f:
        for n, q in sorted(overrides.items()):
            f.write(f"{n}={q}\n")

    print(f"  {sname}: {sinfo['desc']} ({n_changed} overrides)")

print("\nDone. Quantize with:")
print("  llama-quantize --allow-requantize --tensor-type-file types_<variant>.txt <source> <output> Q4_K")
