#!/usr/bin/env python3
"""D2 TRANSPORT PROFILER (28/08/2026)

Mesure le working-set expert actif par token pour chaque modèle D2,
directement depuis les shapes/tensotypes GGUF reels + routing top-8.

Métrique dominante en decode MoE sous contrainte VRAM 8 GB :
    bytes d'experts actifs transferés / token
    (au lieu de la taille GGUF totale).

Usage:
  python d2_transport_profiler.py [--json]
"""
import gguf, math, os, sys, json

# ROOT corrigé : le vrai projet est lama-tensorRT 1050-5070, pas "lama 1080-5070".
HERE = os.path.dirname(os.path.abspath(__file__))   # .../d2 v3/d2 v3
ROOT = os.path.dirname(os.path.dirname(HERE))        # racine du projet


def find_models():
    """Détecte les GGUF présents dans le projet (aucun chemin codé en dur)."""
    found = {}
    for base in (ROOT, os.path.join(ROOT, "models"),
                 os.path.join(ROOT, "d2 v3"), os.path.join(ROOT, "d2 v3", "d2 v3")):
        if not os.path.isdir(base):
            continue
        for f in sorted(os.listdir(base)):
            if f.lower().endswith(".gguf") and f not in found:
                found[f] = os.path.join(base, f)
    return found

# Modèles attendus (D2 35B) — seuls ceux réellement présents seront mesurés.
WANTED = ["Qwen3.6-35B-A3B-UD-IQ4_NL.gguf", "Qwen3.6-35B-A3B-D2-S1I.gguf",
          "Qwen3.6-35B-A3B-D2-OFFICIAL-SPEED3.gguf", "Qwen3.6-35B-A3B-D2-MOE.gguf"]
MODELS = {os.path.basename(p): p for f, p in find_models().items()}
QS = gguf.GGML_QUANT_SIZES
N_EXPERTS, TOP_K = 256, 8

def tbytes(t):
    n = math.prod(t.shape)
    bs, bb = QS[t.tensor_type.value]
    return n / bs * bb

def load(m):
    p = os.path.join(ROOT, m)
    r = gguf.GGUFReader(p)
    by = {t.name: t for t in r.tensors}
    return by

out = {}
print(f"{'model':<10} {'GGUF(GB)':>9} {'gate/ex':>9} {'up/ex':>9} {'down/ex':>9} "
      f"{'MB/LAYER':>9} {'MB/TOKEN':>9} {'MB/TOKEN':>9}")
print(f"{'':10} {'':>9} {'':>9} {'':>9} {'':>9} {'':>9} {'(8ex,40L)':>9} {'(8ex,26L)':>9}")

if not MODELS:
    print("[!] Aucun fichier .gguf trouvé dans le projet — rien à mesurer.")
    print(f"    Chemin cherché : {ROOT} (+ models/, d2 v3/)")
    sys.exit(0)

for name, m in MODELS.items():
    try:
        by = load(m)
    except Exception as e:
        print(f"{name:<10} ERROR {e}")
        continue
    tot = sum(tbytes(t) for t in by.values()) / 1e9
    g = u = d = 0.0
    per_layer = {}
    for li in range(40):
        keys = {k: f"blk.{li}.ffn_{k}_exps.weight" for k in ("gate", "up", "down")}
        bg = tbytes(by[keys["gate"]]) / N_EXPERTS / 1e6 if keys["gate"] in by else 0
        bu = tbytes(by[keys["up"]])   / N_EXPERTS / 1e6 if keys["up"] in by else 0
        bd = tbytes(by[keys["down"]]) / N_EXPERTS / 1e6 if keys["down"] in by else 0
        g += bg; u += bu; d += bd
        per_layer[li] = TOP_K * (bg + bu + bd)
    mb_all = sum(per_layer.values())          # 40 couches experts CPU
    mb_26  = sum(per_layer[i] for i in range(26))  # ncmoe=26 (26 premieres couches CPU)
    print(f"{name:<10} {tot:>9.2f} {g:>9.3f} {u:>9.3f} {d:>9.3f} "
          f"{mb_all/40:>9.1f} {mb_all:>9.1f} {mb_26:>9.1f}")
    out[name] = {"gguf_gb": round(tot,2), "gate_mb_ex": round(g,3), "up_mb_ex": round(u,3),
                 "down_mb_ex": round(d,3), "mb_per_layer": round(mb_all/40,1),
                 "mb_token_40L": round(mb_all,1), "mb_token_26L": round(mb_26,1)}

if "--json" in sys.argv:
    print(json.dumps(out, indent=2))