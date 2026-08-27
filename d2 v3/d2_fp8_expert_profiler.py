#!/usr/bin/env python3
"""D2 FP8 Expert Profiler (26/08/2026) — carte de sensibilité par expert.

Profiles les safetensors FP8 officiels (hf_weights_35b_fp8) expert par expert
et les croise avec :
  - le GGUF quantifié D2 (ex: D2-OFFICIAL-SPEED3) : SNR/cos FP8->GGUF par expert
  - la calibration réelle de routing (models/35b_exp/d2_calibration_final.json)
    : fréquence d'activation par expert (axe AWQ/imatrix)

Format FP8 détecté (vérifié sur le disque) : poids torch.float8_e4m3fn,
scale BF16 par bloc (4×16) NVFP4-style -> W_f32 = fp8 * scale, tile (128,128).

Sorties : JSON par expert {couche, expert, type(gate/up/down), snr_db, cos,
freq, classe(robuste/moyen/sensible)} + rapport MD.

Usage :
  python d2_fp8_expert_profiler.py --gguf models/Qwen3.6-35B-A3B-D2-OFFICIAL-SPEED3.gguf
      [--layers 0-9] [--calib models/35b_exp/d2_calibration_final.json]
"""
import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FP8_DIR = os.path.join(ROOT, "hf_weights_35b_fp8")

SNR_THR = {"robuste": 20.0, "moyen": 12.0}   # seuils dB par classe (littérature MoEQuant)


def dequant_fp8_tensor(t, scale):
    """FP8 E4M3 + scale BF16 par bloc 128x128 (scale shape = R/128, C/128) -> f32."""
    t = t.float()
    s = scale.float()
    rows, cols = t.shape
    rb, cb = s.shape                    # (rows/128, cols/128)
    return (t.reshape(rb, rows // rb, cb, cols // cb)
            * s.unsqueeze(1).unsqueeze(3)).reshape(rows, cols)


def load_layer_shards(layer, keys_need):
    """Charge depuis les 42 shards les clés du layer demandé."""
    import glob
    import safetensors.torch
    out = {}
    for shard in sorted(glob.glob(os.path.join(FP8_DIR, "*.safetensors"))):
        try:
            sd = safetensors.torch.load_file(shard, device="cpu")
        except Exception:
            continue
        hit = {k: sd[k] for k in sd if k in keys_need}
        out.update(hit)
        if len(out) >= len(keys_need):
            break
    return out


def prepare_layer_gguf(gguf_path, layer):
    """Retourne {kind: (raw_bytes, bytes_par_expert, dtype_gguf, shape_f32)}.
    Lit les octets DIRECTEMENT dans le fichier via data_offset/n_bytes : t.data
    du reader est tronque sur ce fork (18 Mo vus au lieu de ~144)."""
    import gguf
    import numpy as np
    r = gguf.GGUFReader(gguf_path)
    out = {}
    for kind in ("gate", "up", "down"):
        name = f"blk.{layer}.ffn_{kind}_exps.weight"
        t = next((x for x in r.tensors if x.name == name), None)
        if t is None:
            continue
        shp = [int(x) for x in t.shape]
        # axe expert = DERNIER axe gguf (layout C: contigu par expert)
        n_expert = shp[-1]
        with open(gguf_path, "rb") as f:
            f.seek(int(t.data_offset))
            # [CORRIGÉ] t.n_bytes est TRONQUE sur ce fork pour les gros tenseurs
            # experts -> recalcule depuis prod(shape) x type_size/block_size
            qs = gguf.GGML_QUANT_SIZES[t.tensor_type]
            nbytes_full = int(math.prod(shp)) // qs[0] * qs[1]
            raw = np.frombuffer(f.read(nbytes_full), dtype=np.uint8)
        bpe = len(raw) // n_expert
        small = (shp[1], shp[0])                     # matrice logique (rows=n_ff?, cols=n_embd?)
        out[kind] = (raw, bpe, t.tensor_type, small)
    return out


def dequant_one_expert(entry, e):
    """Déquantifie l'expert e -> f32 (shape logique (rows, cols))."""
    import gguf
    import numpy as np
    raw, bpe, dtype, shape_small = entry
    sl = raw[e * bpe:(e + 1) * bpe]
    flat = np.asarray(gguf.dequantize(sl, dtype), dtype=np.float32)
    # ordre gguf: ne=[a,b,c]; un expert = matrice (ne[1], ne[0]) si axe expert=ne[2]
    return flat.reshape(shape_small)


def main():
    global FP8_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.path.join(ROOT, "models",
                     "Qwen3.6-35B-A3B-D2-OFFICIAL-SPEED3.gguf"))
    ap.add_argument("--fp8", default=FP8_DIR)
    ap.add_argument("--layers", default="0-39", help="ex: 0-9 ou 0,1,7")
    ap.add_argument("--calib", default=os.path.join(ROOT, "models", "35b_exp",
                     "d2_calibration_final.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "d2_fp8_expert_report.json"))
    args = ap.parse_args()

    FP8_DIR = args.fp8      # redirige le chargement des shards vers --fp8

    # parser la plage de couches
    layers = []
    for part in args.layers.split(","):
        if "-" in part:
            a, b = part.split("-")
            layers += list(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    layers = sorted(set(l for l in layers if 0 <= l < 40))

    calib = {}
    if os.path.exists(args.calib):
        c = json.load(open(args.calib, encoding="utf-8"))
        for li, vec in (c.get("per_layer") or {}).items():
            calib[int(li)] = {i: f for i, f in enumerate(vec)}
    print(f"[i] FP8={os.path.basename(args.fp8)} GGUF={os.path.basename(args.gguf)} "
          f"layers={layers} calib={'oui' if calib else 'non'}")

    results = []
    for li in layers:
        prefix = f"model.language_model.layers.{li}.mlp.experts."
        kinds = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
        keys_need = set()
        for e in range(256):
            for proj in kinds:
                keys_need.add(f"{prefix}{e}.{proj}.weight")
                keys_need.add(f"{prefix}{e}.{proj}.weight_scale_inv")
        sd = load_layer_shards(li, keys_need)
        print(f"[i] couche {li}: {len(sd)//6} experts charges (shards)")
        # déquantifier UNE FOIS chaque tenseur fusionné du GGUF pour cette couche
        gguf_pre = prepare_layer_gguf(args.gguf, li)
        for e in range(256):
            for proj, kind in kinds.items():
                w = sd.get(f"{prefix}{e}.{proj}.weight")
                ws = sd.get(f"{prefix}{e}.{proj}.weight_scale_inv")
                if w is None or ws is None:
                    continue
                wf = dequant_fp8_tensor(w, ws)          # reference FP8 dequantisee
                if kind not in gguf_pre:
                    continue
                g_e = dequant_one_expert(gguf_pre[kind], e)
                wfs = tuple(wf.shape)
                if tuple(g_e.shape) != wfs:
                    if tuple(g_e.T.shape) == wfs:
                        g_e = g_e.T.copy()      # layout transpose GGUF<->HF
                    else:
                        continue
                # SNR / cos entre FP8 et notre quant GGUF (numpy pur, torch optionnel)
                import numpy as np
                wf_a = wf.detach().cpu().numpy().astype("float32").ravel()
                g_a = g_e.astype("float32").ravel()
                dot = float((wf_a * g_a).sum())
                n1 = float((wf_a * wf_a).sum() ** 0.5); n2 = float((g_a * g_a).sum() ** 0.5)
                cos = dot / (n1 * n2) if n1 * n2 > 0 else 0.0
                err = float((((wf_a - g_a) ** 2).mean()) ** 0.5)
                snr = 20 * math.log10(n1 / (err + 1e-9)) if err > 1e-12 else 99.0
                freq = calib.get(li, {}).get(e, 0.0)
                cls = "robuste" if snr >= SNR_THR["robuste"] else (
                    "moyen" if snr >= SNR_THR["moyen"] else "sensible")
                results.append({"layer": li, "expert": e, "type": kind,
                                "snr_db": round(snr, 2), "cos": round(cos, 4),
                                "freq": round(freq, 6), "classe": cls})
        print(f"    -> {len([r for r in results if r['layer'] == li])} tenseurs profilés")

    # synthèse
    from collections import Counter, defaultdict
    cl = Counter((r["classe"], r["type"]) for r in results)
    per_type = defaultdict(list)
    for r in results:
        per_type[r["type"]].append(r["snr_db"])
    print("\n== SYNTHESE ==")
    for t, snrs in per_type.items():
        import statistics
        print(f"  {t:9s} N={len(snrs):4d} SNR moy={statistics.mean(snrs):6.2f} dB "
              f"min={min(snrs):7.2f} max={max(snrs):7.2f}")
    print("  classes:", dict(cl))

    if os.path.exists(args.out):                      # backup avant overwrite (évite perte de données)
        try:
            os.replace(args.out, args.out + ".bak")
        except OSError:
            pass
    json.dump({"generated_at": "2026-08-26", "method": "fp8_safetensors_vs_gguf",
               "gguf": os.path.basename(args.gguf), "n_layers": len(layers),
               "thresholds_db": SNR_THR, "experts": results},
              open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"[+] {args.out}")


if __name__ == "__main__":
    main()