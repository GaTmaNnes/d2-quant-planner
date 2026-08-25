#!/usr/bin/env python3
"""
D2 Axis #9: linear_attn vs full_attn sensitivity
The 35B MoE has 30 linear_attn + 10 full_attn layers.
Test: which type is more sensitive to quantization?
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
    QT.I8: "I8", QT.I16: "I16", QT.I32: "I32", QT.I64: "I64",
    QT.F64: "F64", QT.IQ1_M: "IQ1_M", QT.BF16: "BF16",
}

SOURCE = "C:/Users/videl/.lmstudio/models/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
OUTDIR = "C:/Users/videl/Desktop/lama 1080-5070/models/35b_exp"

# Layer types
# [CORRIGÉ 25/08/2026] full_attn = toutes les 4 couches (1 couche sur 4),
# PAS un bloc continu 30-39.
FULL_ATTN_LAYERS = set(range(3, 40, 4))          # 3,7,...,39 (10 couches)
LINEAR_LAYERS = set(range(40)) - FULL_ATTN_LAYERS

# Tensors that are linear_attn specific
LINEAR_TENSOR_KEYWORDS = ['linear_attn', 'linear_conv', 'ssm_', 'conv1d']
# Tensors that are full_attn specific
FULL_ATTN_TENSOR_KEYWORDS = ['attn_q', 'attn_k', 'attn_v', 'attn_o']
# Shared tensors (present in all layers)
SHARED_TENSOR_KEYWORDS = ['ffn_', 'norm', 'post_attention']


def get_layer(name):
    """Extract layer index from tensor name."""
    if 'layers.' in name:
        try:
            return int(name.split('layers.')[1].split('.')[0])
        except:
            pass
    if 'blk.' in name:
        try:
            return int(name.split('blk.')[1].split('.')[0])
        except:
            pass
    return None


def is_linear_attn_tensor(name):
    """Check if tensor belongs to linear_attn computation."""
    return any(kw in name.lower() for kw in LINEAR_TENSOR_KEYWORDS)


def is_full_attn_tensor(name):
    """Check if tensor belongs to full attention computation."""
    return any(kw in name.lower() for kw in FULL_ATTN_TENSOR_KEYWORDS)


print("Reading source GGUF...")
reader = GGUFReader(SOURCE)
tensors = [(t.name, int(t.tensor_type)) for t in reader.tensors]
print(f"  {len(tensors)} tensors")

# Create variants
# [CORRIGÉ 25/08/2026] CONTRÔLE DE MASSE : les variantes G/H historiques
# quantifiaient TOUS les experts des couches ciblées (linéaires: 30 couches ×3
# tenseurs = 90 vs full_attn: 10×3 = 30) → la comparaison confondait la
# sensibilité du type d'attention avec la MASSE d'experts dégradés (3× plus).
# ⚠️ Les résultats historiques G vs H sont INVALIDES pour cette question.
# Maintenant : le MÊME NOMBRE de tenseurs experts Q3_K dans G et H
# (échantillonnage déterministe en pas régulier sur les couches linéaires).
print("⚠️ [CORRIGÉ 25/08/2026] Résultats historiques G/H INVALIDES : masse "
      "d'experts inégale (90 vs 30 tenseurs). Nouvelles variantes à masse égale.")
variants = {}

# Variant G: linear_attn → Q3_K, full_attn stays IQ4_NL
# (tests: are linear_attn layers safe to compress?)
variants["G_linear_q3k"] = {}

# Variant H: full_attn → Q3_K, linear_attn stays IQ4_NL
# (tests: are full_attn layers safe to compress?)
variants["H_fullattn_q3k"] = {}

# Variant I: BOTH → Q3_K (baseline for comparison = our D2-ECO approach)
variants["I_both_q3k"] = {}

EXPERT_KEYWORDS = ['ffn_gate_exps', 'ffn_down_exps', 'ffn_up_exps']
linear_expert_tensors = []   # experts des couches linear_attn
fullattn_expert_tensors = [] # experts des couches full_attn

for name, qtype in tensors:
    qstr = TYPE_STR.get(qtype, "F32")
    layer = get_layer(name)

    for vname in variants:
        variants[vname][name] = qstr  # default: keep original

    if layer is None:
        continue
    if any(kw in name.lower() for kw in EXPERT_KEYWORDS):
        if layer in LINEAR_LAYERS:
            linear_expert_tensors.append(name)
        elif layer in FULL_ATTN_LAYERS:
            fullattn_expert_tensors.append(name)

# Masse égale : autant de tenseurs experts Q3_K dans G que dans H.
n_common = min(len(linear_expert_tensors), len(fullattn_expert_tensors))
if n_common > 0:
    stride = max(1, len(linear_expert_tensors) // n_common)
    g_experts = set(linear_expert_tensors[::stride][:n_common])
    h_experts = set(fullattn_expert_tensors)  # tous (déjà == n_common si 30 vs 90)
else:
    g_experts = h_experts = set()

for name in g_experts:
    variants["G_linear_q3k"][name] = "Q3_K"
for name in h_experts:
    variants["H_fullattn_q3k"][name] = "Q3_K"

for name, qtype in tensors:
    qstr = TYPE_STR.get(qtype, "F32")
    layer = get_layer(name)
    if layer is None:
        continue

    # Tenseurs spécifiques au TYPE D'ATTENTION (variable réellement testée)
    # Variant G: linear_attn → Q3_K
    if layer in LINEAR_LAYERS and is_linear_attn_tensor(name):
        variants["G_linear_q3k"][name] = "Q3_K"
    # Variant H: full_attn → Q3_K
    if layer in FULL_ATTN_LAYERS and is_full_attn_tensor(name):
        variants["H_fullattn_q3k"][name] = "Q3_K"

    # Variant I: both → Q3_K (tous les experts des deux groupes)
    if any(kw in name.lower() for kw in EXPERT_KEYWORDS):
        if layer in LINEAR_LAYERS or layer in FULL_ATTN_LAYERS:
            variants["I_both_q3k"][name] = "Q3_K"

# Write files
os.makedirs(OUTDIR, exist_ok=True)
for vname, overrides in variants.items():
    path = os.path.join(OUTDIR, f"types_{vname}.txt")
    n_changed = sum(1 for n, q in overrides.items() if q == "Q3_K")
    with open(path, 'w') as f:
        for name, qstr in sorted(overrides.items()):
            f.write(f"{name}={qstr}\n")
    print(f"  {vname}: {n_changed} tensors → Q3_K -> {path}")

# Vérification du contrôle de masse
g_n = sum(1 for n, q in variants["G_linear_q3k"].items() if q == "Q3_K" and any(kw in n.lower() for kw in EXPERT_KEYWORDS))
h_n = sum(1 for n, q in variants["H_fullattn_q3k"].items() if q == "Q3_K" and any(kw in n.lower() for kw in EXPERT_KEYWORDS))
print(f"\n  Contrôle de masse experts : G={g_n} tenseurs experts Q3_K, H={h_n} "
      f"(égaux: {g_n == h_n})")

print("\nNext: quantize with llama-quantize --allow-requantize --tensor-type-file")
print("Then: measure PPL + t/s for each variant")
print("⚠️ Rappel : les anciens résultats G/H (masse inégale) sont INVALIDES.")
