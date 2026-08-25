#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 EXPERT PROFILER FAST — version optimisée (batch memory, pas slice-by-slice).
Lit chaque tenseur fusionné d'un coup, profile les 256 experts en mémoire.
~2-5s par couche au lieu de 500s.
"""

import argparse
import json
import os
import struct
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD_DIR_35B = os.path.join(HERE, "hf_weights_35b")
INDEX_PATH = os.path.join(SHARD_DIR_35B, "model.safetensors.index.json")

# --- Import measure from d2_streaming_profiler ---
sys.path.insert(0, HERE)
from d2_streaming_profiler import measure

FORMATS = ["FP16", "FP8", "INT8", "NVFP4", "Q4", "Q3", "Q2"]
BYTES_PER_FMT = {"FP16": 2.0, "FP8": 1.0, "INT8": 1.0, "NVFP4": 0.5,
                 "Q4": 0.5, "Q3": 0.375, "Q2": 0.25}


def read_tensor_f32(path, hlen, info):
    """Lit un tenseur ENTIER et le retourne en float32."""
    dtype = info["dtype"]
    shape = info["shape"]
    nelems = 1
    for s in shape:
        nelems *= s
    off = info["data_offsets"][0]
    
    if dtype == "F32":
        bpe = 4
        with open(path, "rb") as fh:
            fh.seek(8 + hlen + off)
            raw = fh.read(nelems * bpe)
        return np.frombuffer(raw, dtype="<f4").astype(np.float32).reshape(shape)
    elif dtype in ("BF16", "F16"):
        bpe = 2
        with open(path, "rb") as fh:
            fh.seek(8 + hlen + off)
            raw = fh.read(nelems * bpe)
        u = np.frombuffer(raw, dtype="<u2")
        W = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float32)
        return W.reshape(shape)
    elif dtype == "F8_E4M3":
        bpe = 1
        with open(path, "rb") as fh:
            fh.seek(8 + hlen + off)
            raw = fh.read(nelems * bpe)
        # FP8 E4M3: reconstruct float32
        u = np.frombuffer(raw, dtype=np.uint8).astype(np.uint32)
        # E4M3 format: [s][e3e2e1e0][m2m1m0] -> sign=bit7, exp=bits6-3, mantissa=bits2-0
        sign = (u >> 7) & 1
        exp = (u >> 3) & 0xF
        mant = u & 0x7
        # Handle special cases
        is_denorm = (exp == 0)
        is_naninf = (exp == 0xF)
        # Normal numbers: value = (-1)^s * 2^(exp-7) * (1 + mant/8)
        # Denormals: value = (-1)^s * 2^(-6) * (mant/8)
        val = np.where(is_naninf, np.nan,
                np.where(is_denorm,
                    ((-1.0)**sign) * (2.0**(-6)) * (mant / 8.0),
                    ((-1.0)**sign) * (2.0**(exp.astype(np.float32) - 7)) * (1.0 + mant / 8.0)))
        val = np.nan_to_num(val, nan=0.0)
        return val.reshape(shape)
    else:
        raise ValueError(f"Dtype inconnu: {dtype}")


def profile_fused_expert(path, hlen, info, tensor_name, layer):
    """Profile un tenseur fusionné [256, d1, d2] en batch."""
    shape = info["shape"]
    E = shape[0]
    proj = "gate_up" if "gate_up_proj" in tensor_name else "down"
    
    t0 = time.time()
    W_all = read_tensor_f32(path, hlen, info)  # [256, d1, d2]
    t_load = time.time() - t0
    
    results = []
    t_measure = 0
    for e in range(E):
        W_e = W_all[e]  # [d1, d2]
        if proj == "gate_up" and W_e.shape[0] < W_e.shape[1]:
            W_e = W_e.T
        t1 = time.time()
        snr = measure(W_e)
        t_measure += time.time() - t1
        results.append({
            "layer": layer, "expert": e, "proj": proj, "source": "fused",
            "n_elems": int(W_e.size),
            "snr": {k: round(v, 1) for k, v in snr.items()}
        })
    
    del W_all
    print(f"    Layer {layer:2d} {proj:7s}: load={t_load:.1f}s measure={t_measure:.1f}s total={t_load+t_measure:.1f}s "
          f"({E} experts, {results[0]['n_elems']//1000}K elems)")
    return results


def profile_layer(shard_dir, layer_num):
    """Profile tous les experts d'une couche (gate_up + down) en lisant le bon shard."""
    # Trouver quel shard contient cette couche
    with open(INDEX_PATH) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    
    gate_up_key = None
    down_key = None
    for tname, shard in wm.items():
        if f".layers.{layer_num}.mlp.experts.gate_up_proj" in tname:
            gate_up_key = (tname, shard)
        if f".layers.{layer_num}.mlp.experts.down_proj" in tname:
            down_key = (tname, shard)
    
    if not gate_up_key and not down_key:
        print(f"  Layer {layer_num}: aucun tenseur expert trouvé")
        return []
    
    results = []
    
    for tname, shard in [gate_up_key, down_key]:
        if not tname:
            continue
        path = os.path.join(shard_dir, shard)
        with open(path, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
        info = hdr[tname]
        results.extend(profile_fused_expert(path, hlen, info, tname, layer_num))
    
    return results


def aggregate(jsonl_path, report_path, router_report_path, min_snr=20.0, trust_proxy=False):
    """Agrège le JSONL en rapport par expert, croisé avec fréquence routeur.
    [CORRIGÉ 25/08/2026] promote/demote basés sur l'importance routeur = PROXY
    STATIQUE (~25% fiable ; 256/256 experts actifs, entropie 0.998) : ils ne sont
    appliqués que si --trust-proxy est passé explicitement."""
    rows = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    
    if not rows:
        print("[!] JSONL vide")
        return
    
    # Charger données routeur
    router_importance = {}
    if os.path.exists(router_report_path):
        with open(router_report_path) as f:
            rdata = json.load(f)
        for e in rdata.get("experts", []):
            router_importance[e["expert"]] = e.get("importance_rel", 0.5)
    
    # Moyenne par expert x proj
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        e = r["expert"]
        proj = r["proj"]
        for fmt in FORMATS:
            if fmt in r["snr"]:
                acc[e][proj][fmt].append(r["snr"][fmt])
    
    # Table finale
    table = []
    proxy_warn_shown = [False]  # [CORRIGÉ 25/08/2026] warning proxy statique
    for e in sorted(acc):
        row = {"expert": e}
        
        # SNR par projection
        for proj in ["gate_up", "down"]:
            if proj in acc[e]:
                for fmt in FORMATS:
                    vals = acc[e][proj][fmt]
                    row[f"{proj}_{fmt}_snr"] = round(float(np.mean(vals)), 1) if vals else None
                row[f"{proj}_slices"] = sum(len(acc[e][proj][fmt]) for fmt in FORMATS)
        
        # SNR global (moyenne gate_up + down)
        for fmt in FORMATS:
            vals = []
            for proj in ["gate_up", "down"]:
                if proj in acc[e] and fmt in acc[e][proj]:
                    vals.extend(acc[e][proj][fmt])
            row[f"{fmt}_snr"] = round(float(np.mean(vals)), 1) if vals else None
        
        # Importance routeur
        row["router_importance"] = round(router_importance.get(e, 0.5), 4)
        
        # Score combiné : SNR pondéré par importance
        # Plus l'expert est important, plus il faut de SNR
        row["weighted_snr_q4"] = round(row.get("Q4_snr", 0) * row["router_importance"], 1)
        row["weighted_snr_q3"] = round(row.get("Q3_snr", 0) * row["router_importance"], 1)
        row["weighted_snr_q2"] = round(row.get("Q2_snr", 0) * row["router_importance"], 1)
        
        # Choix optimal : plus petit format dont le SNR pondéré dépasse le seuil
        TIERS = [
            ("Q2", 0.25, "Q2_snr"),
            ("Q3", 0.375, "Q3_snr"),
            ("Q4", 0.5, "Q4_snr"),
            ("INT8", 1.0, "INT8_snr"),
            ("FP16", 2.0, "FP16_snr"),
        ]
        chosen = "FP16"
        chosen_bpe = 2.0
        for fmt, bpe, snr_key in TIERS:
            snr = row.get(snr_key, 0)
            if snr and snr >= min_snr:
                chosen = fmt
                chosen_bpe = bpe
                break
        
        # Ajustement par importance : experts très importants → +1 tier
        # [CORRIGÉ 25/08/2026] GATED par --trust-proxy : l'importance est un PROXY
        # statique (~25% fiable), PAS une fréquence d'activation réelle. Les faits
        # mesurés (256/256 actifs, entropie 0.998) interdisent de le traiter comme
        # une fréquence réelle sans consentement explicite.
        imp = row["router_importance"]
        if not trust_proxy:
            proxy_warn_shown[0] = True
        elif imp > 0.85:
            # Promote: INT8 becomes FP16, Q4 becomes INT8
            promote = {"Q2": "Q3", "Q3": "Q4", "Q4": "INT8", "INT8": "FP16", "FP16": "FP16"}
            chosen = promote.get(chosen, chosen)
            chosen_bpe = BYTES_PER_FMT.get(chosen, 2.0)
        elif imp < 0.75:
            # Demote: FP16 becomes INT8, INT8 becomes Q4, Q4 becomes Q3
            demote = {"FP16": "INT8", "INT8": "Q4", "Q4": "Q3", "Q3": "Q2", "Q2": "Q2"}
            chosen = demote.get(chosen, chosen)
            chosen_bpe = BYTES_PER_FMT.get(chosen, 2.0)
        
        row["chosen"] = chosen
        row["bytes_per_elem"] = chosen_bpe
        table.append(row)
    
    # Stats
    from collections import Counter
    fmt_count = Counter(r["chosen"] for r in table)

    if proxy_warn_shown[0]:
        print()
        print("  ⚠️ AVERTISSEMENT [CORRIGÉ 25/08/2026] : router_importance = PROXY STATIQUE")
        print("  (gate norm, fiabilité ~25%). FAITS MESURÉS : 256/256 experts actifs,")
        print("  entropie routeur 0.998 → PAS une fréquence d'activation réelle.")
        print("  Promote/demote DÉSACTIVÉS. Passer --trust-proxy pour les appliquer quand même.")

    # Taille totale
    n_layers = 40
    per_expert_gate_up = 1024 * 2048
    per_expert_down = 2048 * 512

    total_gb = 0
    for r in table:
        bpe = r["bytes_per_elem"]
        e_bytes = (per_expert_gate_up + per_expert_down) * bpe * n_layers
        total_gb += e_bytes / (1024**3)
    # [CORRIGÉ 25/08/2026] BUG : la division par len(table) donnait le poids
    # D'UN SEUL expert réparti sur 256 lignes (poids ÷256). C'est bien la SOMME.
    # Si tous les experts n'ont pas été profilés, on extrapole à 256.
    if table and len(table) < 256:
        total_gb *= 256 / len(table)
        extrapolated = True
    else:
        extrapolated = False
    
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({
            "min_snr": min_snr,
            "n_experts": len(table),
            "format_distribution": dict(fmt_count),
            "estimated_expert_gb": round(total_gb, 1),
            "expert_gb_extrapolated_to_256": extrapolated,
            "trust_proxy_applied": trust_proxy,
            "note": "SNR croise avec importance routeur (PROXY statique) pour allocation par expert",
            "experts": table
        }, fh, ensure_ascii=False, indent=2)

    print()
    print("=" * 100)
    print(f"  D2 EXPERT PROFILER FAST — agrégation ({len(table)} experts, seuil={min_snr} dB)")
    print("=" * 100)
    print(f"  Répartition: {dict(fmt_count)}")
    print(f"  Poids experts estimés (somme réelle, tous experts): {total_gb:.1f} GB"
          + (" [extrapolé à 256]" if extrapolated else ""))
    print(f"  Top 5 experts (importance):")
    top5 = sorted(table, key=lambda r: r["router_importance"], reverse=True)[:5]
    for r in top5:
        print(f"    E{r['expert']:>3}: imp={r['router_importance']:.3f} Q4={r.get('Q4_snr',0):.1f} "
              f"Q3={r.get('Q3_snr',0):.1f} Q2={r.get('Q2_snr',0):.1f} → {r['chosen']}")
    print(f"  Bottom 5 experts (importance):")
    bot5 = sorted(table, key=lambda r: r["router_importance"])[:5]
    for r in bot5:
        print(f"    E{r['expert']:>3}: imp={r['router_importance']:.3f} Q4={r.get('Q4_snr',0):.1f} "
              f"Q3={r.get('Q3_snr',0):.1f} Q2={r.get('Q2_snr',0):.1f} → {r['chosen']}")
    print(f"  [+] Report: {report_path}")
    print("=" * 100)


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    
    ap = argparse.ArgumentParser(description="D2 Expert Profiler FAST")
    ap.add_argument("--shard-dir", default=SHARD_DIR_35B)
    ap.add_argument("--layers", default=None, help="ex: 0-39 ou 0,5,10")
    ap.add_argument("--jsonl", default=os.path.join(HERE, "d2_expert_fast.jsonl"))
    ap.add_argument("--report", default=os.path.join(HERE, "d2_expert_fast_report.json"))
    ap.add_argument("--router-report", default=os.path.join(HERE, "d2_router_static_report.json"))
    ap.add_argument("--aggregate", action="store_true", help="Agréger le JSONL en rapport")
    ap.add_argument("--min-snr", type=float, default=20.0)
    ap.add_argument("--resume", action="store_true", help="Skip layers déjà profilées")
    # [CORRIGÉ 25/08/2026] --trust-proxy requis pour appliquer promote/demote
    ap.add_argument("--trust-proxy", action="store_true",
                    help="Appliquer promote/demote sur l'importance routeur "
                         "(PROXY statique ~25%% fiable — déconseillé)")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.jsonl, args.report, args.router_report, args.min_snr,
                  trust_proxy=args.trust_proxy)
        return
    
    # Déterminer les layers à profiler
    if args.layers:
        layers = set()
        for part in args.layers.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                layers.update(range(int(a), int(b) + 1))
            else:
                layers.add(int(part))
        layers = sorted(layers)
    else:
        layers = list(range(40))
    
    # Reprise
    seen_layers = set()
    if args.resume and os.path.exists(args.jsonl):
        with open(args.jsonl) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    seen_layers.add(r["layer"])
        print(f"Reprise: {len(seen_layers)} couches déjà profilées")
    
    layers = [l for l in layers if l not in seen_layers]
    if not layers:
        print("Toutes les couches sont déjà profilées. Lance --aggregate.")
        return
    
    print(f"Profiling {len(layers)} couches: {layers[:5]}...{layers[-3:] if len(layers) > 5 else ''}")
    print()
    
    t_start = time.time()
    total_experts = 0
    for i, layer in enumerate(layers):
        results = profile_layer(args.shard_dir, layer)
        with open(args.jsonl, "a", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        total_experts += len(results)
        
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * len(layers) - elapsed
        print(f"  [{i+1}/{len(layers)}] Progress: {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
    
    print(f"\nTerminé: {total_experts} entrées en {time.time()-t_start:.1f}s")
    print(f"JSONL: {args.jsonl}")
    print(f"Lance --aggregate pour le rapport final")


if __name__ == "__main__":
    main()