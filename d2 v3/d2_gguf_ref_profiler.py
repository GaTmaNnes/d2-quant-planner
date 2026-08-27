#!/usr/bin/env python3
"""D2 GGUF-REF PROFILER (28/08/2026) — SNR par expert sans safetensors FP8.

Référence = GGUF quasi-lossless (ex: Q8_0 racine) au lieu des safetensors FP8
supprimés. Compare expert par expert le GGUF cible (D2) vs la référence.

Réutilise les helpers de d2_fp8_expert_profiler (même format de sortie).
Usage:
  python d2_gguf_ref_profiler.py --gguf <D2.gguf> --ref-gguf <Q8_0.gguf>
      [--layers 0-9] [--calib ...] [--out report_q8.json]
"""
import argparse, json, math, os, sys, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d2_fp8_expert_profiler import prepare_layer_gguf, dequant_one_expert, SNR_THR

ROOT = os.path.dirname(HERE)
DEFAULT_CALIB = os.path.join(ROOT, "models", "35b_exp", "d2_calibration_final.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref-gguf", required=True)
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--out", default=os.path.join(HERE, "d2_gguf_ref_report.json"))
    args = ap.parse_args()
    if not os.path.exists(args.gguf):
        print(f"[!] GGUF cible introuvable: {args.gguf}"); raise SystemExit(1)
    if not os.path.exists(args.ref_gguf):
        print(f"[!] GGUF référence introuvable: {args.ref_gguf}"); raise SystemExit(1)
    lo, hi = (int(x) for x in args.layers.split("-"))
    layers = list(range(lo, hi + 1))
    calib = json.load(open(args.calib)) if os.path.exists(args.calib) else {}
    calib = calib.get("per_layer", calib)          # structure: {"per_layer": {layer: {expert: count}}}
    calib = {int(k): v for k, v in calib.items() if str(k).isdigit()}

    results = []
    for li in layers:
        ref_pre = prepare_layer_gguf(args.ref_gguf, li)   # reference (Q8_0)
        tgt_pre = prepare_layer_gguf(args.gguf, li)       # cible D2
        for kind in ("gate", "up", "down"):
            if kind not in ref_pre or kind not in tgt_pre:
                continue
            n_experts = min(ref_pre[kind][0].shape[0] if False else 256,
                            len(tgt_pre[kind][0]) // tgt_pre[kind][1], 256)
            for e in range(256):
                try:
                    ref_e = dequant_one_expert(ref_pre[kind], e)
                    tgt_e = dequant_one_expert(tgt_pre[kind], e)
                except Exception:
                    continue
                # alignement layout (transpose si nécessaire)
                if ref_e.shape != tgt_e.shape:
                    if ref_e.T.shape == tgt_e.shape:
                        tgt_e = tgt_e.T.copy()
                    else:
                        continue
                import numpy as np
                r = ref_e.astype("float32").ravel(); g = tgt_e.astype("float32").ravel()
                dot = float((r * g).sum())
                n1 = float((r * r).sum() ** 0.5); n2 = float((g * g).sum() ** 0.5)
                cos = dot / (n1 * n2) if n1 * n2 > 0 else 0.0
                err = float((((r - g) ** 2).mean()) ** 0.5)
                snr = 20 * math.log10(n1 / (err + 1e-9)) if err > 1e-12 else 99.0
                fl = calib.get(li, [])
                freq = fl[e] if isinstance(fl, list) and e < len(fl) else (fl.get(e, 0.0) if isinstance(fl, dict) else 0.0)
                cls = "robuste" if snr >= SNR_THR["robuste"] else (
                    "moyen" if snr >= SNR_THR["moyen"] else "sensible")
                results.append({"layer": li, "expert": e, "type": kind,
                                "snr_db": round(snr, 2), "cos": round(cos, 4),
                                "freq": round(freq, 6), "classe": cls})
        print(f"[i] couche {li}: {len([r for r in results if r['layer']==li])} tenseurs profilés")

    if os.path.exists(args.out):
        try: os.replace(args.out, args.out + ".bak")
        except OSError: pass
    json.dump({"generated_at": "2026-08-28", "method": "gguf_vs_gguf",
               "gguf": os.path.basename(args.gguf), "ref": os.path.basename(args.ref_gguf),
               "n_layers": len(layers), "thresholds_db": SNR_THR, "experts": results},
              open(args.out, "w", encoding="utf-8"), indent=1)
    from collections import Counter, defaultdict
    cl = Counter((r["classe"], r["type"]) for r in results)
    per_type = defaultdict(list)
    for r in results: per_type[r["type"]].append(r["snr_db"])
    print("\n== SYNTHESE ==")
    for t, snrs in per_type.items():
        print(f"  {t:9s} N={len(snrs):4d} SNR moy={statistics.mean(snrs):6.2f} min={min(snrs):7.2f} max={max(snrs):7.2f}")
    print("  classes:", dict(cl))
    print(f"[+] {args.out}")

if __name__ == "__main__":
    main()