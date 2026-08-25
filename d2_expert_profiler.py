#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 EXPERT PROFILER — profile les 256 experts du Qwen3.5-35B-A3B, un à un.
===========================================================================
Le checkpoint stocke les experts de façon FUSIONNÉE par couche :

    model.language_model.layers.N.mlp.experts.gate_up_proj   [256, 1024, 2048]  (1.07 Go)
    model.language_model.layers.N.mlp.experts.down_proj      [256, 2048,  512]  (0.54 Go)

L'expert i vit donc dans la tranche [i, :, :] de chaque tenseur. Ce profileur
ne charge JAMAIS le tenseur entier : il lit uniquement la tranche de l'expert
(≈ 4 Mo + 2 Mo) via un seek ciblé, la mesure, écrit le résultat, puis passe au
suivant. C'est le « profiler au fur et à mesure » appliqué aux experts.

Le layer MTP, lui, stocke ses experts SÉPARÉMENT :
    mtp.layers.0.mlp.experts.{0..255}.{gate,up,down}_proj.weight   (256 x 3 tensors)

Flux adapté au téléchargement par shard :
  1. télécharger le shard nécessaire
  2. python d2_expert_profiler.py --shard-file <shard>   (append au JSONL)
  3. supprimer le shard
  4. shard suivant
  ... puis : python d2_expert_profiler.py --report       (agrège la table finale)

Sorties :
  - d2_expert_profiler.jsonl   (une ligne par (couche, expert, projection), ajoutée au fil de l'eau)
  - d2_expert_profiler_report.json  (table expert -> SNR moyen par format -> choix)

Usage :
  python d2_expert_profiler.py --shard-file hf_weights_35b/model.safetensors-00001-of-00014.safetensors
  python d2_expert_profiler.py --report
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
SHARD_DIR = os.path.join(HERE, "hf_weights_35b")
INDEX_PATH = os.path.join(SHARD_DIR, "model.safetensors.index.json")
OUT_JSONL = os.path.join(HERE, "d2_expert_profiler.jsonl")
OUT_REPORT = os.path.join(HERE, "d2_expert_profiler_report.json")

from d2_streaming_profiler import measure  # réutilise measure() à 7 formats

FORMATS = ["FP16", "FP8", "INT8", "NVFP4", "Q4", "Q3", "Q2"]
BYTES_PER_FMT = {"FP16": 2.0, "FP8": 1.0, "INT8": 1.0, "NVFP4": 0.5,
                 "Q4": 0.5, "Q3": 0.375, "Q2": 0.25}


# ---------------------------------------------------------------------------
# Lecture ciblée d'une tranche d'expert
# ---------------------------------------------------------------------------
def read_header(path):
    with open(path, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hlen))
    return hlen, hdr


def read_expert_slice(path, hlen, info, i_expert):
    """Lit la tranche [i_expert, :, :] d'un tenseur fusionné [E, d1, d2] -> float32 (d1, d2)."""
    shape = info["shape"]
    if len(shape) == 2:
        # tenseur non fusionné (ex. expert MTP séparé) : lire en entier
        d1, d2 = shape[0], shape[1]
        off = info["data_offsets"][0]
        nread = d1 * d2
    else:
        E, d1, d2 = shape[0], shape[1], shape[2]
        bpe = 4 if info["dtype"] == "F32" else 2
        off = info["data_offsets"][0] + i_expert * d1 * d2 * bpe
        nread = d1 * d2

    dt = info["dtype"]
    bpe = 4 if dt == "F32" else 2
    with open(path, "rb") as fh:
        fh.seek(8 + hlen + off)
        raw = fh.read(nread * bpe)
    if dt == "F32":
        W = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:  # BF16
        u = np.frombuffer(raw, dtype="<u2")
        W = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float32)
    return W.reshape(d1, d2)


def layer_of(name):
    if ".layers." in name:
        return int(name.split(".layers.")[1].split(".")[0])
    return -1  # ex. lm_head / embed / mtp


def is_fused_expert(name):
    return (".mlp.experts.gate_up_proj" in name) or (".mlp.experts.down_proj" in name)


def is_mtp_expert(name):
    return ("mtp." in name) and (".mlp.experts." in name)


# ---------------------------------------------------------------------------
# Reprise (skip des tranches déjà profilées)
# ---------------------------------------------------------------------------
def load_seen(jsonl):
    """Retourne l'ensemble des clés (source, layer, expert, proj) déjà dans le JSONL."""
    seen = set()
    if os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    seen.add((r["source"], str(r["layer"]), r["expert"], r["proj"]))
                except Exception:
                    continue
    return seen


# ---------------------------------------------------------------------------
# Traitement d'un shard (streaming par expert)
# ---------------------------------------------------------------------------
def process_shard(path, experts, layers, jsonl, seen=None, limit=None):
    """Lit le shard, profile chaque tranche d'expert demandée, écrit au JSONL."""
    if not os.path.exists(path):
        print(f"[!] shard absent : {path}")
        return 0
    if seen is None:
        seen = set()
    hlen, hdr = read_header(path)
    n = 0
    n_skipped = 0
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        L = layer_of(name)
        if L < 0:
            continue  # pas une couche du modèle (lm_head/embed/mtp gérés à part)
        if layers and L not in layers:
            continue

        if is_fused_expert(name):
            proj = "gate_up" if "gate_up_proj" in name else "down"
            E = info["shape"][0]
            for e in range(E):
                if experts and e not in experts:
                    continue
                key = ("fused", str(L), e, proj)
                if key in seen:
                    n_skipped += 1
                    continue
                W = read_expert_slice(path, hlen, info, e)
                snr = measure(W)
                rec = {"layer": L, "expert": e, "proj": proj, "source": "fused",
                       "n_elems": int(W.size), "snr": {k: round(v, 1) for k, v in snr.items()}}
                with open(jsonl, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                seen.add(key)
                n += 1
                del W
                if limit and n >= limit:
                    return n

        elif is_mtp_expert(name):
            # mtp.layers.0.mlp.experts.<e>.<proj>.weight  (un tensor = un expert)
            parts = name.split(".mlp.experts.")[1].split(".")
            e = int(parts[0])
            proj = parts[1]
            if experts and e not in experts:
                continue
            key = ("mtp", "mtp", e, proj)
            if key in seen:
                n_skipped += 1
                continue
            W = read_expert_slice(path, hlen, info, e)
            snr = measure(W)
            rec = {"layer": "mtp", "expert": e, "proj": proj, "source": "mtp",
                   "n_elems": int(W.size), "snr": {k: round(v, 1) for k, v in snr.items()}}
            with open(jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            seen.add(key)
            n += 1
            del W
            if limit and n >= limit:
                return n
    if n_skipped:
        print(f"    ({n_skipped} tranches déjà profilées, ignorées)")
    return n


# ---------------------------------------------------------------------------
# Agrégation par expert
# ---------------------------------------------------------------------------
def aggregate(jsonl, report_path, min_snr=20.0):
    rows = []
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("[!] JSONL vide — lance d'abord --shard-file sur au moins un shard.")
        return 1

    # [CORRIGÉ 25/08/2026] Accumulation PAR PROJECTION : l'ancienne moyenne
    # unique gate_up+down masquait l'asymétrie mesurée 13.5× (gate_up très
    # sensible, down très tolérant). Les deux stats sont maintenant reportées
    # séparément ; le choix de format est gouverné par la projection la plus
    # fragile (min), pas par une moyenne qui noie le signal.
    acc_proj = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # expert -> proj -> fmt -> [snr]
    n_slices = defaultdict(int)
    for r in rows:
        e = r["expert"]
        n_slices[e] += 1
        proj = r.get("proj", "?")
        for fmt in FORMATS:
            if fmt in r["snr"]:
                acc_proj[e][proj][fmt].append(r["snr"][fmt])

    table = []
    for e in sorted(acc_proj):
        row = {"expert": e, "slices": n_slices[e]}
        for proj in ("gate_up", "down"):
            if proj not in acc_proj[e]:
                continue
            for fmt in FORMATS:
                vals = acc_proj[e][proj][fmt]
                row[f"{proj}_{fmt}_snr"] = round(float(np.mean(vals)), 1) if vals else None

        def best_fmt_for(proj):
            """Plus petit palier dont le meilleur format atteint min_snr pour cette proj."""
            TIERS = [
                (0.25, ["Q2"]),
                (0.375, ["Q3"]),
                (0.5, ["Q4", "NVFP4"]),
                (1.0, ["INT8", "FP8"]),
                (2.0, ["FP16"]),
            ]
            for size, fmts in TIERS:
                snrs = [(f, row.get(f"{proj}_{f}_snr")) for f in fmts]
                snrs = [(f, v) for f, v in snrs if v is not None]
                if not snrs:
                    continue
                best_fmt, best_snr = max(snrs, key=lambda x: x[1])
                if best_snr >= min_snr:
                    return best_fmt
            return "FP16"

        # Choix conservateur : le format doit satisfaire les DEUX projections
        # (asymétrie 13.5× → gouvernée par la pire).
        choices = {best_fmt_for(p) for p in ("gate_up", "down") if p in acc_proj[e]}
        ORDER = ["FP16", "INT8", "FP8", "NVFP4", "Q4", "Q3", "Q2"]
        chosen = max(choices, key=lambda f: ORDER.index(f)) if choices else "FP16"
        row["chosen"] = chosen
        row["chosen_gate_up"] = best_fmt_for("gate_up") if "gate_up" in acc_proj[e] else None
        row["chosen_down"] = best_fmt_for("down") if "down" in acc_proj[e] else None
        row["bytes_per_elem"] = BYTES_PER_FMT[chosen]
        table.append(row)

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({"min_snr": min_snr,
                   "note": "[CORRIGÉ 25/08/2026] stats gate_up/down séparées (asymétrie 13.5×) ; "
                           "'chosen' = format satisfaisant les 2 projections",
                   "experts": table}, fh, ensure_ascii=False, indent=2)

    # affichage — colonnes gate_up et down SÉPARÉES
    print("=" * 110)
    print(f"  D2 EXPERT PROFILER — agrégation ({len(table)} experts, seuil min_snr={min_snr} dB)")
    print("  [CORRIGÉ 25/08/2026] gate_up et down reportés SÉPARÉMENT (asymétrie mesurée 13.5×)")
    print("=" * 110)
    hdr = (f"  {'Expert':>7} | {'#slices':>7} | {'GU Q4':>6} | {'DN Q4':>6} | "
           f"{'GU Q3':>6} | {'DN Q3':>6} | {'GU INT8':>7} | {'DN INT8':>7} | "
           f"{'GU FP16':>7} | {'DN FP16':>7} | {'choix':<6}")
    print(hdr)
    print("  " + "-" * 104)

    def cell(r, key, width):
        v = r.get(key)
        return f"{v:>{width}.1f}" if isinstance(v, float) else f"{'—':>{width}}"

    for r in table:
        print(f"  E{r['expert']:>6} | {r['slices']:>7} | "
              f"{cell(r, 'gate_up_Q4_snr', 6)} | {cell(r, 'down_Q4_snr', 6)} | "
              f"{cell(r, 'gate_up_Q3_snr', 6)} | {cell(r, 'down_Q3_snr', 6)} | "
              f"{cell(r, 'gate_up_INT8_snr', 7)} | {cell(r, 'down_INT8_snr', 7)} | "
              f"{cell(r, 'gate_up_FP16_snr', 7)} | {cell(r, 'down_FP16_snr', 7)} | "
              f"{r['chosen']:<6}")

    from collections import Counter
    c = Counter(r["chosen"] for r in table)
    print("  " + "-" * 104)
    print("  Répartition des choix : " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    # Asymétrie agrégée gate_up vs down (le fait 13.5× doit rester visible)
    gu_q4 = [r["gate_up_Q4_snr"] for r in table if r.get("gate_up_Q4_snr")]
    dn_q4 = [r["down_Q4_snr"] for r in table if r.get("down_Q4_snr")]
    if gu_q4 and dn_q4 and np.mean(dn_q4) > 0:
        ratio = np.mean(gu_q4) / np.mean(dn_q4)
        print(f"  Asymétrie moyenne SNR Q4 gate_up/down : {np.mean(gu_q4):.1f}/{np.mean(dn_q4):.1f} "
              f"(ratio {ratio:.1f}× — cohérent avec l'asymétrie 13.5× PPL si >1)")
    print(f"  [+] Report : {report_path}")
    print("=" * 110)
    return 0


def parse_range(s):
    """'0-7' -> set(range(0,8)), '0,5,255' -> {0,5,255}, '' -> None (tous)."""
    if not s or s.lower() in ("all", "none"):
        return None
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 Expert Profiler (Qwen3.5-35B-A3B)")
    ap.add_argument("--shard-dir", default=SHARD_DIR)
    ap.add_argument("--shard-file", default=None, help="traiter UN seul shard")
    ap.add_argument("--experts", default="", help="ex. '0-7' ou '0,5,255' (vide = tous)")
    ap.add_argument("--layers", default="", help="ex. '0-3' (vide = toutes)")
    ap.add_argument("--limit", type=int, default=0, help="max tranches (test rapide)")
    ap.add_argument("--jsonl", default=OUT_JSONL)
    ap.add_argument("--report", default=OUT_REPORT)
    ap.add_argument("--min-snr", type=float, default=20.0, help="SNR min pour le choix")
    ap.add_argument("--aggregate", action="store_true", help="agréger le JSONL existant")
    args = ap.parse_args()

    if args.aggregate:
        return aggregate(args.jsonl, args.report, args.min_snr)

    experts = parse_range(args.experts)
    layers = parse_range(args.layers)
    t0 = time.time()

    if args.shard_file:
        shards = [args.shard_file]
    else:
        shards = sorted(os.path.join(args.shard_dir, f)
                        for f in os.listdir(args.shard_dir) if f.endswith(".safetensors"))

    seen = load_seen(args.jsonl)
    total = 0
    for sh in shards:
        print(f"[*] shard : {sh}")
        n = process_shard(sh, experts, layers, args.jsonl, seen=seen, limit=args.limit)
        total += n
        print(f"    -> {n} tranches profilées (total {total})")
        if args.limit and total >= args.limit:
            break

    print(f"\n[+] {total} tranches profilées en {time.time()-t0:.1f} s -> {args.jsonl}")
    print("    Agrège avec : python d2_expert_profiler.py --aggregate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
