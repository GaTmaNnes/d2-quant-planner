#!/usr/bin/env python3
"""D2 DECISION MATRIX (28/08/2026) — fichier de décision tensor/source/target.

Pure analyse sur profils existants (pas de build) :
  - d2_fp8_expert_report.json : SNR/expert + freq routing (FP8 vs SPEED3)
  - S1I GGUF                  : types réels actuels + shapes (base de décision)
  - imatrix                   : (option) importance activation

Pour chaque tensor expert et chaque transformation descendante :
  saved_mb, snr (headroom qualité), freq, hot/cold, score ROI.

Score ROI (transport) = snr_norm * bytes_saved * freq_norm
  - snr_norm élevé  = headroom de compression (robuste)
  - bytes_saved     = gain transport si l'expert est actif
  - freq_norm élevé = expert souvent routé -> plus de bytes/token économisés

Sortie : d2_decision_matrix.csv + top transformations + ratio t/s/MB.
"""
import json, math, os, gguf, csv, argparse

ROOT = r"C:\Users\videl\Desktop\lama 1080-5070"
ap = argparse.ArgumentParser()
ap.add_argument("--gguf", default=os.path.join(ROOT, "models", "Qwen3.6-35B-A3B-D2-S1I-COLDQ2-IMATRIX-OFF.gguf"))
ap.add_argument("--report", default=os.path.join(ROOT, "d2 v3", "d2_fp8_expert_report.json"))
args = ap.parse_args()
REPORT = args.report
S1I = args.gguf
if not os.path.exists(S1I):
    print(f"[!] GGUF introuvable: {S1I} — passez --gguf <fichier> (S1I supprimé)")
    raise SystemExit(1)
if not os.path.exists(REPORT):
    print(f"[!] report introuvable: {REPORT}")
    raise SystemExit(1)
QS = gguf.GGML_QUANT_SIZES
N_EXPERTS = 256
BPW_Q = {"Q2_K": 2.6, "Q3_K": 3.4, "Q4_K": 4.5, "Q5_K": 5.5, "Q6_K": 6.6}

def tbytes(t):
    n = math.prod(t.shape)
    bs, bb = QS[t.tensor_type.value]
    return n / bs * bb

rep = json.load(open(REPORT))
exp = rep["experts"]
r = gguf.GGUFReader(S1I)
by = {t.name: t for t in r.tensors}

# type S1I actuel par (layer, type)
cur = {}
for li in range(40):
    for ty in ("gate", "up", "down"):
        t = by.get(f"blk.{li}.ffn_{ty}_exps.weight")
        if t:
            cur[(li, ty)] = (tbytes(t) / N_EXPERTS / 1e6, t.tensor_type.name)

BPW_CUR = {"IQ3_S": 3.4, "Q3_K": 3.4, "IQ4_NL": 4.5}
DOWNGRADES = {  # source type -> targets (bpw inférieurs seulement)
    "IQ3_S": [("Q2_K", 2.6)],
    "Q3_K":  [("Q2_K", 2.6)],
    "IQ4_NL":[("Q3_K", 3.4), ("Q2_K", 2.6)],
}

# freq max pour normalisation
freqs = [e["freq"] for e in exp]
fmax = max(freqs) or 1

rows = []
for e in exp:
    key = (e["layer"], e["type"])
    if key not in cur:
        continue
    mb, tcur = cur[key]
    if tcur not in DOWNGRADES:
        continue
    snr_norm = max(0.0, min(1.0, e["snr_db"] / 100.0))
    freq_norm = e["freq"] / fmax
    cold = "COLD" if e["freq"] <= 1 else ("WARM" if e["freq"] < fmax*0.02 else "HOT")
    for tgt, bpw in DOWNGRADES[tcur]:
        saved_mb = mb * (1 - bpw / BPW_CUR[tcur])
        if saved_mb <= 0:
            continue
        # qualite: risque = fragilite x exposition (freq). +snr = -fragile.
        quality_risk = (1.0 - snr_norm) * (freq_norm + 0.1)
        crush_score = saved_mb / max(quality_risk, 1e-6)   # MB economises / risque qualite
        rows.append({
            "tensor": f"blk.{e['layer']}.ffn_{e['type']}_exps.weight", "expert": e["expert"],
            "source": tcur, "target": tgt, "saved_mb": round(saved_mb, 4),
            "snr_db": e["snr_db"], "freq": e["freq"], "class": cold,
            "quality_risk": round(quality_risk, 5), "crush_score": round(crush_score, 6),
        })

rows.sort(key=lambda x: -x["crush_score"])
if not rows:
    print("[!] aucun candidat ne matche DOWNGRADES / size_cache — sortie vide")
    raise SystemExit(1)
outcsv = os.path.join(ROOT, "d2 v3", "d2_decision_matrix.csv")
with open(outcsv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"[ok] {outcsv} ({len(rows)} candidats) - classe par MB/risque-qualite (freq-neutral)")
print(f"\n{'tensor':<30} {'exp':>4} {'src':<7} {'tgt':<5} {'savedMB':>8} {'snr':>6} {'freq':>6} {'class':<5} {'risk':>7} {'score':>8}")
for x in rows[:16]:
    print(f"{x['tensor']:<30} {x['expert']:>4} {x['source']:<7} {x['target']:<5} {x['saved_mb']:>8.4f} "
          f"{x['snr_db']:>6.1f} {x['freq']:>6} {x['class']:<5} {x['quality_risk']:>7.4f} {x['crush_score']:>8.4f}")

# agrégats par transformation
from collections import defaultdict
agg = defaultdict(lambda: [0, 0.0])
for x in rows:
    agg[(x["source"], x["target"])][0] += 1
    agg[(x["source"], x["target"])][1] += x["saved_mb"]
print(f"\n--- AGREGAT par transformation (tous experts) ---")
for k, (n, mb) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"  {k[0]:<7} -> {k[1]:<5} : {n:>6} experts, {-mb:>8.1f} MB total")

print(f"\n--- ratio t/s / MB actifs (base mesures) ---")
for name, ts, mb in [("S1I", 29.78, 281), ("D2-MOE", 22.63, 339), ("UD-IQ4_NL", 8.46, 310)]:
    print(f"  {name:<10} {ts:>6.2f} t/s / {mb:>3} MB = {ts/mb:>6.3f} t/s per MB")