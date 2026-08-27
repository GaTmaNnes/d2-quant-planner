#!/usr/bin/env python3
"""D2 PLANNER PARETO (28/08/2026) — allocation décisionnelle Q2/Q3/Q4/Q6.

Fusionne 3 profils existants (ZERO build) :
  - d2_fp8_expert_report.json   : SNR/expert (FP8 vs SPEED3) + freq routing + classe
  - SPEED3 GGUF                 : shapes + types réels (le référentiel du SNR)
  - (option) imatrix            : importance activation

Métrique décisionnelle :
  crush_score = snr_norm * coldness
    snr_norm   : SNR/expert élevé = headroom de compression (robuste)
    coldness   : 1/(1+ln(freq)) = expert peu routé = impact faible
  => crush d'abord les experts robustes + froids ; PROTEGE les fragiles (SNR bas).

Sortie : liste ordonnée de recommandations + bytes économisés par palier.
"""
import json, math, os, gguf, argparse

ROOT = r"C:\Users\videl\Desktop\lama 1080-5070"
REPORT = os.path.join(ROOT, "d2 v3", "d2_fp8_expert_report.json")
ap = argparse.ArgumentParser()
ap.add_argument("--gguf", default=os.path.join(ROOT, "models", "Qwen3.6-35B-A3B-D2-S1I-COLDQ2-IMATRIX-OFF.gguf"))
ap.add_argument("--report", default=REPORT)
args = ap.parse_args()
SPEED3 = args.gguf
REPORT = args.report
if not os.path.exists(SPEED3):
    print(f"[!] GGUF introuvable: {SPEED3} — passez --gguf <fichier> (SPEED3/S1I supprimés)")
    raise SystemExit(1)
if not os.path.exists(REPORT):
    print(f"[!] report introuvable: {REPORT}")
    raise SystemExit(1)
QS = gguf.GGML_QUANT_SIZES
N_EXPERTS = 256

def tbytes(t):
    n = math.prod(t.shape)
    bs, bb = QS[t.tensor_type.value]
    return n / bs * bb

rep = json.load(open(REPORT))
exp = rep["experts"]
print(f"report: {len(exp)} experts | {rep['method']} | gguf={rep['gguf'].split('/')[-1]}")

r = gguf.GGUFReader(SPEED3)
by = {t.name: t for t in r.tensors}

# bytes per expert per (layer,type) from SPEED3 shapes
size_cache = {}
for li in range(40):
    for ty in ("gate", "up", "down"):
        t = by.get(f"blk.{li}.ffn_{ty}_exps.weight")
        if t:
            size_cache[(li, ty)] = tbytes(t) / N_EXPERTS / 1e6  # MB/expert

def quant_mb(expert_mb, bpw_from, bpw_to):
    return expert_mb * bpw_to / bpw_from

BPW = {"gate": 5.5, "up": 5.5, "down": 2.7}   # SPEED3 approx (Q5_K gate/up, Q2_K down)
# cibles de compression
TARGETS = [("Q3_K", 3.4), ("Q2_K", 2.6)]

rows = []
for e in exp:
    key = (e["layer"], e["type"])
    if key not in size_cache:
        continue
    mb = size_cache[key]
    cold = 1.0 / (1.0 + math.log1p(e["freq"]))
    snr_norm = max(0.0, min(1.0, e["snr_db"] / 100.0))   # unifié avec d2_decision_matrix.py
    crush = snr_norm * cold
    rows.append({**e, "mb": mb, "crush": crush, "snr_norm": snr_norm, "cold": cold})

if not rows:
    print("[!] aucun expert exploitable (report vide ou size_cache vide)")
    raise SystemExit(1)

# tri par crush décroissant
rows.sort(key=lambda x: -x["crush"])

print(f"\n{'rank':>4} {'layer':>5} {'type':<6} {'expert':>6} {'snr':>6} {'freq':>5} {'class':<9} {'MB/ex':>6} {'crush':>6}")
for i, x in enumerate(rows[:14]):
    print(f"{i+1:>4} {x['layer']:>5} {x['type']:<6} {x['expert']:>6} {x['snr_db']:>6.1f} {x['freq']:>5} "
          f"{x['classe']:<9} {x['mb']:>6.2f} {x['crush']:>6.3f}")

# fragiles (à protéger) : SNR bas
frag = sorted([x for x in rows if x["snr_db"] < 15], key=lambda x: x["snr_db"])[:10]
print(f"\n--- FRAGILES (SNR<15dB) — à protéger / remonter en précision ---")
for x in frag:
    print(f"  blk.{x['layer']}.{x['type']} exp.{x['expert']:>3} snr={x['snr_db']:.1f} freq={x['freq']} mb={x['mb']:.2f}")

# bilan compressibilité : combien d'octets si on cruse le top-25% crush-score vers Q3_K/Q2_K
rows_sorted = sorted(rows, key=lambda x: -x["crush"])
n_crush = int(len(rows_sorted) * 0.25)
saved_q3 = saved_q2 = 0.0
for x in rows_sorted[:n_crush]:
    src = BPW[x["type"]]
    saved_q3 += x["mb"] * (1 - 3.4/src)
    saved_q2 += x["mb"] * (1 - 2.6/src)
tot_expert_mb = sum(x["mb"] for x in rows)
print(f"\n--- BILAN compression top-25% experts (crush score) ---")
print(f"  {n_crush} experts / {len(rows)} -> vers Q3_K: {-saved_q3/1024:.2f} GB  | vers Q2_K: {-saved_q2/1024:.2f} GB")
print(f"  experts totaux: {tot_expert_mb:.2f} GB")

hot = [x for x in rows if x["freq"] > 0]
never = [x for x in rows if x["freq"] == 0]
print(f"\n--- HOT (freq>0) vs NEVER (freq=0) ---")
print(f"  hot: {len(hot)} ({len(hot)/len(rows)*100:.1f}%)  never: {len(never)} ({len(never)/len(rows)*100:.1f}%)")
if hot and never:
    print(f"  SNR hot: {sum(x['snr_db'] for x in hot)/len(hot):.1f} dB | SNR never: {sum(x['snr_db'] for x in never)/len(never):.1f} dB")
elif hot:
    print(f"  SNR hot: {sum(x['snr_db'] for x in hot)/len(hot):.1f} dB (never vide)")
else:
    print("  (hot vide)")

# TRANSPORT sur base S1I (gate/up IQ3_S 0.45MB, down Q3_K 0.45MB, 8 actifs, 26 couches CPU)
GATE_S1I, GATE_Q2 = 0.45, 0.45*2.6/3.4
per_tok_all = 26 * 8 * 2 * (GATE_S1I - GATE_Q2)   # gate+up, toutes couches
per_tok_6   = 6  * 8 * 2 * (GATE_S1I - GATE_Q2)   # couches froides 0-5
print(f"\n--- TRANSPORT gate/up IQ3_S -> Q2_K (base S1I, 281 MB/token) ---")
print(f"  toutes les 26 couches CPU: -{per_tok_all:.1f} MB/token -> {281-per_tok_all:.0f} MB/token (-{per_tok_all/281*100:.0f}%)")
print(f"  couches froides 0-5 uniquement: -{per_tok_6:.1f} MB/token -> {281-per_tok_6:.0f} MB/token (-{per_tok_6/281*100:.1f}%)")

print(f"\n--- ROUTERS (ffn_gate_inp) ---")
rout_bytes = 0
for li in range(40):
    t = by.get(f"blk.{li}.ffn_gate_inp.weight")
    if t:
        rout_bytes += tbytes(t)
print(f"  total router F32: {rout_bytes/1e6:.1f} MB -> F16: {rout_bytes/2e6:.1f} MB (gain {-rout_bytes/2e6:.1f} MB, négligeable)")