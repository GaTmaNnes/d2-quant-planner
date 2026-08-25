"""Traduit une config RAMP (group_name -> QType) en fichier --tensor-type-file
pour notre llama-quantize (beellama.cpp), en reutilisant le GGUFAnalyzer de
RAMP pour retrouver les tenseurs GGUF reels de chaque groupe de decision."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# RAMP s'installe par défaut dans un dossier temporaire ; surchargeable via env.
RAMP_PATH = os.environ.get("RAMP_QUANT_PATH", r"C:\Users\videl\AppData\Local\Temp\ramp-quant")
sys.path.insert(0, RAMP_PATH)
try:
    from gguf_analyzer import GGUFAnalyzer
except ImportError as e:
    sys.exit(f"[!] gguf_analyzer introuvable dans {RAMP_PATH} "
             f"(installer RAMP ou définir RAMP_QUANT_PATH). Détail : {e}")

GGUF_PATH = os.path.join(HERE, "models", "Qwen3.8-27B-hybrid.gguf")
RAMP_JSON = os.path.join(HERE, "ramp_27b_best.json")
OUT = os.path.join(HERE, "d2_tensor_types_ramp.txt")

TYPE_MAP = {
    "IQ2_XXS": "iq2_xxs", "IQ2_XS": "iq2_xs", "IQ3_XXS": "iq3_xxs", "IQ3_S": "iq3_s",
    "Q4_K": "q4_K", "Q5_K": "q5_K", "Q6_K": "q6_K", "Q8_0": "q8_0",
}

with open(RAMP_JSON, encoding="utf-8") as fh:
    d = json.load(fh)
config = d["config"]

a = GGUFAnalyzer(GGUF_PATH)

lines = []
missing_groups = []
for group_name, qtype in config.items():
    grp = a.groups.get(group_name)
    if grp is None:
        missing_groups.append(group_name)
        continue
    ggml_type = TYPE_MAP[qtype]
    for tname in grp.tensor_names:
        lines.append(f"{tname}={ggml_type}")

print(f"{len(lines)} tenseurs mappes depuis {len(config)} groupes RAMP, {len(missing_groups)} groupes non trouves")
if missing_groups:
    print("groupes non trouves:", missing_groups[:5])

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("->", OUT)

from collections import Counter
c = Counter(l.split("=")[1] for l in lines)
print("repartition:", dict(c))
