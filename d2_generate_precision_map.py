#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 GENERATE PRECISION MAP
=========================
Génère `precision_map.json` AUTOMATIQUEMENT depuis le score UTILITY, en
remplaçant la politique statique.

Pipeline de décision par tensor (dans l'ordre) :

  1. STRUCTURAL PASS  — incompatibilité matérielle/kernel :
        - ssm_conv1d      -> FP16 (kernel ssm-conv exige src1 nb[0]=4B ; ne[0]=4 < bloc 32/64)
                             REASON = STRUCTURAL_INCOMPATIBILITY (pas NUMERICAL_QUALITY)
        - norm.*          -> FP16 (couche de normalisation)
        - ssm_a / ssm_dt  -> FP16 (paramètre récurrent minuscule)
        - bloc géométrie  : ne[0] % block_size == 0 pour chaque candidat
  2. ECONOMIC PASS    — gain VRAM négligeable :
        - tenseur < 1 Mo en FP16 -> FP16
  3. NUMERICAL PASS   — score UTILITY :
        UTILITY(p) = VRAM_gain_normalisé - (rel_err(p) * amplification / tolérance)
        -> argmax sur les candidats compatibles.
  4. MEMORY BUDGET    — si le total dépasse --vram-budget (défaut 6 Go) :
        downgrade INT8 -> NVFP4 par ordre de moindre perte d'UTILITY par Go économisé.

Entrées :
  - models/Qwen3.5-9B-Q4_K_S.gguf           (shapes réelles, ne[0], n_elems)
  - hf_weights/model.safetensors-00002..    (SNR FFN mesuré)
  - precision_map.json existant             (alpha / density préservés)

Sorties :
  - precision_map.json          (écrasé, avec sauvegarde .bak)
  - precision_map_diag.json     (raison + SNR + UTILITY complets)

Usage :
  python d2_generate_precision_map.py [--dry-run] [--vram-budget GB]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "beellama.cpp", "gguf-py"))
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "Qwen3.5-9B-Q4_K_S.gguf")
MAP_PATH = os.path.join(HERE, "precision_map.json")
SHARD2 = os.path.join(HERE, "hf_weights", "model.safetensors-00002-of-00004.safetensors")
SHARD4 = os.path.join(HERE, "hf_weights", "model.safetensors-00004-of-00004.safetensors")

from d2_autonomous_profiler import (compute_utility, DEFAULT_SNR, spectral_factor,
                                    DENSITY_FP16_THRESH, ALPHA_INT8_THRESH)
from d2_noise_profiler import (load_gguf_conv1d, load_hf_ffn, quant_int8,
                               quant_e2m1_block, full_metrics)
from d2_precision_optimizer import load_hf_generic, hf_to_gguf, layer_num

BLOCK_SIZE = {
    "FP16":  GGML_QUANT_SIZES[GGMLQuantizationType.F16][0],
    "INT8":  GGML_QUANT_SIZES[GGMLQuantizationType.Q8_0][0],
    "NVFP4": GGML_QUANT_SIZES[GGMLQuantizationType.NVFP4][0],
}
LABEL = {"FP16": "FP16_REQUIRED", "INT8": "INT8_SAFE", "NVFP4": "NVFP4_SAFE"}
BYTES_PER = {"FP16_REQUIRED": 2.0, "INT8_SAFE": 1.0, "NVFP4_SAFE": 0.5}

ECON_MIN_BYTES = 1.0e6   # tensors < 1 Mo (FP16) => gain économique nul


def _snr_safe(W, Wq):
    m = full_metrics(W, Wq)
    s = m.get("snr", np.nan)
    return float(s) if np.isfinite(s) else np.nan


def build_measurements():
    """SNR mesuré par tensor (clé = nom GGUF). Sources fiables uniquement."""
    meas = {}

    # 1) conv1d (GGUF F32) — de toute façon F32-forcé, mais mesuré pour le diag
    for name, W in load_gguf_conv1d(MODEL).items():
        Wt = W.T if (W.ndim > 1 and W.shape[0] < W.shape[1]) else W.reshape(1, -1)
        meas[name] = {
            "FP16": _snr_safe(Wt, Wt.astype(np.float16).astype(np.float32)),
            "INT8": _snr_safe(Wt, quant_int8(Wt)),
            "NVFP4": _snr_safe(Wt, quant_e2m1_block(Wt, scale_e4m3=True)),
        }

    # 2) FFN (shard 2) — le vrai enjeu mémoire
    if os.path.exists(SHARD2):
        for name, W in load_hf_ffn(SHARD2).items():
            gg = hf_to_gguf(name)
            if not gg:
                continue
            meas[f"blk.{layer_num(name)}.{gg}"] = {
                "FP16": 150.0,
                "INT8": _snr_safe(W, quant_int8(W)),
                "NVFP4": _snr_safe(W, quant_e2m1_block(W, scale_e4m3=True)),
            }

    # 3) ssm_out (shard 4, out_proj -> ssm_out : forme [4096,4096] fiable)
    if os.path.exists(SHARD4):
        for name, W in load_hf_generic(SHARD4, keep=("linear_attn.out_proj",)).items():
            gg = hf_to_gguf(name)
            if not gg or gg != "ssm_out.weight":
                continue
            meas[f"blk.{layer_num(name)}.{gg}"] = {
                "FP16": 150.0,
                "INT8": _snr_safe(W, quant_int8(W)),
                "NVFP4": _snr_safe(W, quant_e2m1_block(W, scale_e4m3=True)),
            }

    # sanitize : SNR non-fini -> DEFAULT
    for name, m in meas.items():
        for k in ("FP16", "INT8", "NVFP4"):
            if not np.isfinite(m[k]):
                m[k] = DEFAULT_SNR[k]
    return meas


def static_policy(name, alpha, density):
    """Reconstruit la politique statique documentée (precision_map.json d'origine).

    Référence de concordance :
      - ssm_conv1d : density >= 0.31 -> FP16, sinon INT8
      - ssm_alpha/beta : alpha >= 1.4 -> INT8, sinon NVFP4
      - norm / ssm_a / ssm_dt -> FP16
      - reste -> NVFP4
    """
    ln = name.lower()
    if "ssm_conv1d" in ln:
        return "FP16_REQUIRED" if density >= DENSITY_FP16_THRESH else "INT8_SAFE"
    if "ssm_alpha" in ln or "ssm_beta" in ln:
        return "INT8_SAFE" if alpha >= ALPHA_INT8_THRESH else "NVFP4_SAFE"
    if "norm" in ln or ln.endswith(".ssm_a") or ".ssm_dt" in ln:
        return "FP16_REQUIRED"
    return "NVFP4_SAFE"


def decide(name, ne0, n_elems, snr_map, alpha=1.0, density=0.0):
    """Retourne (label, reason, utility_dict_or_None)."""
    ln = name.lower()

    # ---- 1. STRUCTURAL PASS ----
    if "ssm_conv1d" in ln:
        spec = ""
        if density >= DENSITY_FP16_THRESH:
            spec = f" ; SPECTRAL: density {density:.3f} >= {DENSITY_FP16_THRESH} -> FP16"
        else:
            spec = f" ; SPECTRAL: density {density:.3f} (statique=INT8) MAIS kernel F32 -> FP16"
        return ("FP16_REQUIRED",
                "STRUCTURAL_INCOMPATIBILITY: kernel ssm-conv exige F32 (src1 nb[0]=4B); ne[0]=4 < bloc 32/64" + spec,
                None)
    if "norm" in ln:
        return "FP16_REQUIRED", "STRUCTURAL: couche de normalisation (critique)", None
    if ln.endswith(".ssm_a") or ".ssm_dt" in ln:
        return "FP16_REQUIRED", "STRUCTURAL: paramètre SSM récurrent (minuscule)", None

    # bloc géométrie : candidats compatibles avec ne[0]
    candidates = ["FP16"]
    if ne0 % BLOCK_SIZE["INT8"] == 0:
        candidates.append("INT8")
    if ne0 % BLOCK_SIZE["NVFP4"] == 0:
        candidates.append("NVFP4")

    # ---- 2. ECONOMIC PASS ----
    fp16_bytes = n_elems * 2.0
    if fp16_bytes < ECON_MIN_BYTES:
        return "FP16_REQUIRED", f"ECONOMIC: tenseur minuscule ({fp16_bytes/1e6:.3f} Mo FP16, gain nul)", None

    # ---- 3. NUMERICAL PASS (UTILITY + sensibilité spectrale) ----
    sf = spectral_factor(alpha, density, name)
    u = compute_utility(name, n_elems, snr_map, spectral_factor=sf)
    best = max((p for p in candidates), key=lambda p: u[p]["utility"])
    label = LABEL[best]
    reason = (f"UTILITY={u[best]['utility']:+.3f} "
              f"(VRAM -{u[best]['vram_gain_mb']:.2f} Mo vs FP16 ; "
              f"SNR INT8={snr_map['INT8']:.1f} dB NVFP4={snr_map['NVFP4']:.1f} dB ; "
              f"SF={sf:.2f})")
    return label, reason, u


def total_size_gb(m, tensors):
    """Taille estimée des poids. FP16_REQUIRED (ou absent) -> conserve le type stocké."""
    tot = 0.0
    for name, info in tensors.items():
        entry = m.get(name)
        if entry is None or entry.get("precision") == "FP16_REQUIRED":
            t = getattr(GGMLQuantizationType, info["current_type"], GGMLQuantizationType.F32)
            bs, ts = GGML_QUANT_SIZES[t]
            bpe = ts / bs
        else:
            bpe = BYTES_PER.get(entry["precision"], 2.0)
        tot += info["n_elems"] * bpe
    return tot / 1e9


def apply_budget(m, utils, tensors, budget_gb):
    """Downgrade INT8->NVFP4 par moindre perte d'UTILITY par Go économisé."""
    candidates = []
    for name, u in utils.items():
        if m[name]["precision"] != "INT8_SAFE":
            continue
        saved_gb = (BYTES_PER["INT8_SAFE"] - BYTES_PER["NVFP4_SAFE"]) * tensors[name]["n_elems"] / 1e9
        loss = u["INT8"]["utility"] - u["NVFP4"]["utility"]
        ratio = loss / max(saved_gb, 1e-12)   # perte d'utility par Go économisé
        candidates.append((name, ratio, saved_gb, u))

    candidates.sort(key=lambda x: x[1])  # plus petit ratio d'abord
    n_down = 0
    while total_size_gb(m, tensors) > budget_gb and candidates:
        name, ratio, saved_gb, u = candidates.pop(0)
        m[name]["precision"] = "NVFP4_SAFE"
        m[name]["reason"] += f" | BUDGET {budget_gb:.1f}G (utility {u['NVFP4']['utility']:+.3f}, -{saved_gb*1e3:.0f} Mo)"
        n_down += 1
    return n_down, total_size_gb(m, tensors)


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="ne pas écrire precision_map.json")
    ap.add_argument("--vram-budget", type=float, default=6.0, help="budget poids en Go (défaut 6.0)")
    args = ap.parse_args()

    # GGUF : liste complète des tensors
    r = GGUFReader(MODEL)
    tensors = {}
    for t in r.tensors:
        tensors[t.name] = {"ne0": int(t.shape[0]), "n_elems": int(t.n_elements),
                           "current_type": t.tensor_type.name}

    meas = build_measurements()

    existing = {}
    if os.path.exists(MAP_PATH):
        with open(MAP_PATH, encoding="utf-8") as fh:
            existing = json.load(fh)

    # décision (hors budget)
    new_map = {}
    utils = {}
    diag = {}
    for name, info in tensors.items():
        snr_map = meas.get(name, dict(DEFAULT_SNR))
        ex = existing.get(name, {})
        alpha = float(ex.get("alpha", 1.0))
        density = float(ex.get("density", 0.0))
        label, reason, u = decide(name, info["ne0"], info["n_elems"], snr_map, alpha, density)
        new_map[name] = {
            "precision": label,
            "alpha": alpha,
            "density": density,
            "reason": reason,
        }
        if u is not None:
            utils[name] = u
        diag[name] = {
            "precision": label, "reason": reason,
            "ne0": info["ne0"], "n_elems": info["n_elems"],
            "snr": {k: round(v, 1) for k, v in snr_map.items()},
            "spectral_factor": spectral_factor(alpha, density, name),
        }
        if u is not None:
            diag[name]["utility"] = u

    size_before = total_size_gb(new_map, tensors)

    # ---- 4. MEMORY BUDGET ----
    n_down, size_after = apply_budget(new_map, utils, tensors, args.vram_budget)

    # ---- rapport ----
    from collections import Counter
    c = Counter(v["precision"] for v in new_map.values())
    print("=" * 100)
    print("  D2 GENERATE PRECISION MAP — politique auto (STRUCTURAL + ECONOMIC + UTILITY + BUDGET)")
    print("=" * 100)
    print(f"  Tensors       : {len(new_map)}")
    print(f"  Répartition   : FP16_REQUIRED {c.get('FP16_REQUIRED',0):>4} | "
          f"INT8_SAFE {c.get('INT8_SAFE',0):>4} | NVFP4_SAFE {c.get('NVFP4_SAFE',0):>4}")
    print(f"  Taille (UTILITY pur)  : {size_before:,.2f} Go")
    print(f"  Taille (budget {args.vram_budget:.1f} Go) : {size_after:,.2f} Go "
          f"({n_down} tensors INT8->NVFP4)")
    if existing:
        print(f"  Taille (politique statique) : {total_size_gb(existing, tensors):,.2f} Go")
    print("=" * 100)

    # ---- concordance vs politique statique reconstruite (alpha/density) ----
    _match = _tot = 0
    _by_cat = {}
    for name, info in tensors.items():
        a = new_map[name]["alpha"]
        d = new_map[name]["density"]
        s = static_policy(name, a, d)
        g = new_map[name]["precision"]
        ln = name.lower()
        if "ssm_conv1d" in ln:
            cat = "ssm_conv1d (densite spectrale)"
        elif "ssm_alpha" in ln or "ssm_beta" in ln:
            cat = "ssm_alpha/beta (alpha spectral)"
        elif "norm" in ln or ln.endswith(".ssm_a") or ".ssm_dt" in ln:
            cat = "norm/ssm_a/ssm_dt (structural)"
        else:
            cat = "defaut attn/ffn/ssm_out/emb/output"
        m = _by_cat.setdefault(cat, [0, 0])
        m[1] += 1
        _tot += 1
        if s == g:
            m[0] += 1
            _match += 1
    print(f"\n  Concordance vs politique statique (reconstruite alpha/density) : "
          f"{_match}/{_tot} ({100*_match/max(_tot,1):.0f} %)")
    print(f"  {'Categorie':<38} {'match':>6} {'total':>6} {'%':>5}")
    for cat in sorted(_by_cat):
        m = _by_cat[cat]
        print(f"    {cat:<38} {m[0]:>6} {m[1]:>6} {100*m[0]/max(m[1],1):>4.0f}%")
    print("=" * 100)

    # changements vs existant
    if existing:
        changed = [(n, existing.get(n, {}).get("precision"), new_map[n]["precision"])
                   for n in new_map
                   if existing.get(n, {}).get("precision") != new_map[n]["precision"]]
        print(f"\n  Changements vs politique statique : {len(changed)}")
        for n, old, new in changed[:40]:
            print(f"    {n:<30} {str(old):<14} -> {new}")
        if len(changed) > 40:
            print(f"    ... ({len(changed)-40} autres)")
        print("=" * 100)

    # ---- écriture ----
    if not args.dry_run:
        if os.path.exists(MAP_PATH):
            shutil.copy2(MAP_PATH, MAP_PATH + ".bak")
            print(f"\n[+] Sauvegarde : {MAP_PATH}.bak")
        with open(MAP_PATH, "w", encoding="utf-8") as fh:
            json.dump(new_map, fh, ensure_ascii=False, indent=2)
        print(f"[+] precision_map.json régénéré ({len(new_map)} tensors)")

    with open(os.path.join(HERE, "precision_map_diag.json"), "w", encoding="utf-8") as fh:
        json.dump(diag, fh, ensure_ascii=False, indent=2)
    print(f"[+] Diagnostic complet : precision_map_diag.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
