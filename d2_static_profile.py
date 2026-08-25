#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 STATIC PROFILE — inventaire structurel du checkpoint (sans charger les poids).
==================================================================================
Lit config.json + les en-tetes safetensors des 66 shards d'un checkpoint HF.

Produit :
  - architecture (config)
  - inventaire tensors : nom / dtype / shape / octets / role
  - CARTE FP8 OFFICIELLE : quels tensors sont restes BF16 (modules_to_not_convert)
    vs convertis en F8_E4M3 -> source de verite pour FP8_OFFICIAL_SAFE / FP8_FORBIDDEN
  - poids par (role, dtype) + quelques stats echantillon (proxy HEURISTIC)

Confidence D2 : chaque champ est etiquete measured / calculated / heuristic / unknown
(spec D2 — ne jamais presenter une estimation comme une mesure).

Exemple :
  python d2_static_profile.py --model models/Qwen3.8-27B-FP8
"""

import argparse
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "Qwen3.8-27B-FP8")
OUT = os.path.join(HERE, "d2_static_profile_27B.json")

ROLES = [
    ("embed", r"\.embed_tokens$"),
    ("lm_head", r"^lm_head$"),
    ("norm", r"layernorm|\.norm$|^model\.norm$"),
    ("gate", r"mlp\.(shared_expert_gate|gate)$|attn_output_gate"),
    ("linear_attn", r"linear_attn\."),
    ("self_attn", r"self_attn\."),
    ("mlp", r"mlp\.(gate_proj|up_proj|down_proj|fc1|fc2)"),
    ("mtp", r"^mtp\."),
    ("vision", r"visual\."),
    ("other", r".*"),
]


def classify(name):
    for role, pat in ROLES:
        if re.search(pat, name):
            return role
    return "other"


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return hdr, 8 + n  # header + data_start


def dtype_bytes(dtype):
    m = {"F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
         "I8": 1, "I16": 2, "I32": 4, "I64": 8, "U8": 1}
    return m.get(dtype, 2)


def scan(model_dir):
    shards = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    tensors = []
    per_shard = {}
    for sh in shards:
        hdr, _ = read_header(os.path.join(model_dir, sh))
        n = 0
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            shape = meta["shape"]
            dtype = meta["dtype"]
            nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
            tensors.append({
                "name": name, "dtype": dtype, "shape": shape,
                "bytes": nbytes, "role": classify(name), "shard": sh,
            })
            n += 1
        per_shard[sh] = n
    return tensors, per_shard


def aggregate(tensors):
    by_role_dtype = {}
    by_dtype = {}
    by_role = {}
    total = 0
    for t in tensors:
        total += t["bytes"]
        by_role_dtype.setdefault((t["role"], t["dtype"]), 0)
        by_role_dtype[(t["role"], t["dtype"])] += t["bytes"]
        by_dtype.setdefault(t["dtype"], 0)
        by_dtype[t["dtype"]] += t["bytes"]
        by_role.setdefault(t["role"], 0)
        by_role[t["role"]] += t["bytes"]
    return by_role_dtype, by_dtype, by_role, total


def _precision_classes(tensors, forbidden=None):
    """Classe chaque tensor pour la variation de precision SANS re-profilage structurel:
    - fp8_payload : converti FP8 dans le checkpoint -> candidat Q4/Q2
    - forbidden_bf16 : BF16 protégé (carte officielle) -> fixe en BF16
    - scales : weight_scale_inv (metadonnees FP8) -> fixe
    """
    forbidden = forbidden or []
    def kept_by_prefix(name):
        variants = (name, name.replace("model.language_model.", "model.", 1))
        return any(v == p or v.startswith(p + ".") for v in variants for p in forbidden)
    cls = {"fp8_payload": {"count": 0, "bytes": 0},
           "forbidden_bf16": {"count": 0, "bytes": 0},
           "scales": {"count": 0, "bytes": 0}}
    for t in tensors:
        name = t["name"]
        if name.endswith("weight_scale_inv"):
            c = "scales"
        elif t["dtype"] == "F8_E4M3":
            c = "fp8_payload"
        elif kept_by_prefix(name):
            c = "forbidden_bf16"
        else:
            c = "forbidden_bf16"  # non-FP8 et non scale = BF16 protégé (embed/norm/lm_head...)
        cls[c]["count"] += 1
        cls[c]["bytes"] += t["bytes"]
    out = {}
    for c, v in cls.items():
        out[c] = {"count": v["count"], "bytes": v["bytes"], "gb": round(v["bytes"] / 1e9, 3)}
    return out


def fp8_official_map(tensors, cfg_forbidden):
    """Carte reelle : tensor BF16 (non converti) vs F8_E4M3 (converti)."""
    converted = [t for t in tensors if t["dtype"] == "F8_E4M3"]
    kept = [t for t in tensors if t["dtype"] != "F8_E4M3"]
    kept_bytes = sum(t["bytes"] for t in kept)
    converted_bytes = sum(t["bytes"] for t in converted)
    # coherence avec modules_to_not_convert : ce sont des PREFIXES de MODULES
    # (named_modules -> 'p' matche name == p ou name.startswith(p + ".")),
    # et les FP8 portent des scales BF16 (weight_scale_inv) a distinguer des poids.
    forbidden = sorted(cfg_forbidden or [])
    def kept_by_prefix(name):
        # le config declare parfois le chemin avec ou sans le wrapper language_model
        variants = (name, name.replace("model.language_model.", "model.", 1))
        for v in variants:
            if any(v == p or v.startswith(p + ".") for p in forbidden):
                return True
        return False
    scale_tensors = [t for t in tensors if t["name"].endswith("weight_scale_inv")]
    scale_bytes = sum(t["bytes"] for t in scale_tensors)
    kept_weights = [t for t in kept if not t["name"].endswith("weight_scale_inv")]
    pred = [t for t in kept_weights if not kept_by_prefix(t["name"])]
    pred_bytes = sum(t["bytes"] for t in pred)
    conv_by_prefix = [t for t in converted if kept_by_prefix(t["name"])]
    conv_bytes = sum(t["bytes"] for t in conv_by_prefix)
    # roles des vraies anomalies
    pred_roles = {}
    for t in pred:
        pred_roles[t["role"]] = pred_roles.get(t["role"], 0) + t["bytes"]
    return {
        "kept_bf16_count": len(kept), "kept_bf16_bytes": kept_bytes,
        "converted_fp8_count": len(converted), "converted_fp8_bytes": converted_bytes,
        "declared_modules_to_not_convert": len(forbidden),
        "detected_kept_tensors": len(kept),
        "fp8_scales_bf16": len(scale_tensors), "fp8_scales_bf16_bytes": scale_bytes,
        "fp8_scales_bf16_gb": round(scale_bytes / 1e9, 3),
        "kept_weights_without_declared_prefix": len(pred), "kept_weights_without_declared_prefix_bytes": pred_bytes,
        "kept_weights_without_declared_prefix_gb": round(pred_bytes / 1e9, 3),
        "kept_weights_without_declared_prefix_sample": [t["name"] for t in pred[:15]],
        "kept_weights_without_declared_prefix_roles_gb": {k: round(v / 1e9, 3) for k, v in sorted(pred_roles.items())},
        "converted_fp8_but_declared_bf16": len(conv_by_prefix), "converted_fp8_but_declared_bf16_bytes": conv_bytes,
        "converted_fp8_but_declared_bf16_gb": round(conv_bytes / 1e9, 3),
        "converted_fp8_but_declared_bf16_sample": [t["name"] for t in conv_by_prefix[:15]],
    }


def sample_weight_stats(tensors, model_dir, limit=200):
    """Stats echantillon (proxy HEURISTIC, pas une mesure qualite) : RMS, max, ratio outliers.
    Lecture limitee : tensors BF16/F32/F8_E4M3 de quelques couches representatives.
    Les octets bruts sont lus directement aux offsets du header (pas de conversion safetensors)."""
    import numpy as np
    import os
    _hdr_cache = {}

    def _shards_of(tensors):
        return sorted({t["shard"] for t in tensors})

    def offsets(shard, name):
        return _hdr_cache[shard][0][name]["data_offsets"]

    def data_start(shard):
        return _hdr_cache[shard][1]

    def raw_bytes(shard, name):
        a, b = offsets(shard, name)
        with open(os.path.join(model_dir, shard), "rb") as fh:
            fh.seek(data_start(shard) + a)
            return fh.read(b - a)

    for sh in _shards_of(tensors):
        _hdr_cache[sh] = read_header(os.path.join(model_dir, sh))

    layers = {"model.language_model.layers.0.", "model.language_model.layers.63.",
              "model.language_model.layers.3."}
    sel = [t for t in tensors if any(t["name"].startswith(p) for p in layers)
           and t["dtype"] in ("BF16", "F32", "F8_E4M3")][:limit]
    out = []
    for t in sel:
        try:
            raw = np.frombuffer(raw_bytes(t["shard"], t["name"]), dtype=np.uint8)
            if t["dtype"] == "BF16":  # uint16 << 16 -> pattern f32 (spec D2 §5)
                u16 = raw.view(np.uint16)
                f = (u16.astype(np.uint32) << 16).view(np.float32)
            elif t["dtype"] == "F32":
                f = raw.view(np.float32)
            elif t["dtype"] == "F16":
                f = raw.view(np.float16).astype(np.float32)
            else:  # F8_E4M3 -> decode manuel
                f = _f8e4m3_to_f32(raw)
            rms = float(np.sqrt(np.mean(f.astype(np.float64) ** 2)))
            mx = float(np.max(np.abs(f)))
            n_out = int(np.sum(np.abs(f) > 4 * rms))
            out.append({"name": t["name"], "dtype": t["dtype"], "shape": t["shape"],
                        "rms": round(rms, 5), "max": round(mx, 3),
                        "outlier_rate": round(n_out / f.size, 6),
                        "confidence": "heuristic"})
        except Exception as e:
            out.append({"name": t["name"], "error": str(e), "confidence": "unknown"})
    return out


def _f8e4m3_to_f32(raw):
    """F8_E4M3FN (float8_e4m3fn torch) -> float32.
    NB: exp=15 avec man<7 est FINI (256..448) ; seul exp=15 man=7 = NaN. PAS d'infini.
    Valide contre torch.float8_e4m3fn (0 desaccord sur 0x00..0xFF)."""
    import numpy as np
    sign = ((raw >> 7) & 1).astype(np.float32)
    exp = ((raw >> 3) & 0xF).astype(np.int32)
    man = (raw & 0x7).astype(np.float32)
    f = np.zeros(raw.size, dtype=np.float32)
    nan = (exp == 15) & (man == 7)
    sub = exp == 0
    f[sub] = (1.0 - 2.0 * sign[sub]) * (2.0 ** (1 - 7)) * (man[sub] / 8.0)
    norm = (exp > 0) & (~nan)
    f[norm] = (1.0 - 2.0 * sign[norm]) * (2.0 ** (exp[norm] - 7)) * (1.0 + man[norm] / 8.0)
    f[nan] = np.nan
    return f


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 STATIC PROFILE — inventaire structurel checkpoint")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--json", default=OUT)
    args = ap.parse_args()

    cfg_path = os.path.join(args.model, "config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    tc = cfg.get("text_config", {})
    quant = cfg.get("quantization_config", {})
    forbidden = quant.get("modules_to_not_convert", [])

    tensors, per_shard = scan(args.model)
    by_role_dtype, by_dtype, by_role, total = aggregate(tensors)

    # ---- carte officielle ----
    fmap = fp8_official_map(tensors, forbidden)

    # ---- couches hybrides (spec : 64 couches, pattern linear/full) ----
    layer_types = tc.get("layer_types", [])
    n_full = sum(1 for t in layer_types if t == "full_attention")
    n_lin = sum(1 for t in layer_types if t == "linear_attention")

    profile = {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "source": args.model,
        "revision_hint": "pin HF 017b9c7 (Qwen/Qwen3.8-27B-FP8)",
        "architecture": {
            "confidence": "measured",
            "model_type": cfg.get("model_type"),
            "arch": cfg.get("architectures"),
            "multimodal": bool(cfg.get("vision_config")),
            "language": {
                "layers": tc.get("num_hidden_layers"),
                "attention_layers_full": n_full,
                "linear_attn_layers": n_lin,
                "hidden": tc.get("hidden_size"),
                "intermediate": tc.get("intermediate_size"),
                "heads": tc.get("num_attention_heads"),
                "kv_heads": tc.get("num_key_value_heads"),
                "head_dim": tc.get("head_dim"),
                "vocab": tc.get("vocab_size"),
                "native_context": tc.get("max_position_embeddings"),
                "linear_attn": {
                    "conv_kernel": tc.get("linear_conv_kernel_dim"),
                    "key_heads": tc.get("linear_num_key_heads"),
                    "key_head_dim": tc.get("linear_key_head_dim"),
                    "value_heads": tc.get("linear_num_value_heads"),
                    "value_head_dim": tc.get("linear_value_head_dim"),
                    "ssm_dtype": tc.get("mamba_ssm_dtype"),
                },
                "mtp_layers": tc.get("mtp_num_hidden_layers"),
                "output_gate": tc.get("output_gate_type"),
                "tie_embeddings": tc.get("tie_word_embeddings"),
            },
            "vision": {
                "depth": (cfg.get("vision_config") or {}).get("depth"),
                "hidden": (cfg.get("vision_config") or {}).get("hidden_size"),
                "heads": (cfg.get("vision_config") or {}).get("num_heads"),
                "patch": (cfg.get("vision_config") or {}).get("patch_size"),
                "out_hidden": (cfg.get("vision_config") or {}).get("out_hidden_size"),
            } if cfg.get("vision_config") else None,
        },
        "quantization": {
            "confidence": "measured",
            "method": quant.get("quant_method"),
            "fmt": quant.get("fmt"),
            "activation_scheme": quant.get("activation_scheme"),
            "official_forbidden_count": len(forbidden),
            "official_forbidden_sample": forbidden[:10],
            "fp8_map": fmap,
        },
        "tensor_inventory": {
            "confidence": "measured",
            "count": len(tensors),
            "shards": len(per_shard),
            "shards_ok": {k: v for k, v in per_shard.items()},
            "total_bytes": total,
            "total_gb": round(total / 1e9, 2),
            "by_dtype": {k: round(v / 1e9, 2) for k, v in sorted(by_dtype.items())},
            "by_role_gb": {k: round(v / 1e9, 2) for k, v in sorted(by_role.items())},
            "by_role_dtype_gb": {f"{r}/{d}": round(v / 1e9, 2)
                                 for (r, d), v in sorted(by_role_dtype.items())},
            "by_role_dtype_bytes": {f"{r}/{d}": v for (r, d), v in sorted(by_role_dtype.items())},
        },
        "precision_classes": {
            "confidence": "measured",
            "note": "classe par tensor: fp8_payload (convertible Q4/Q2), forbidden_bf16 (fixe), scales (fixe)",
            **_precision_classes(tensors, forbidden),
        },
        "kv_model": {
            "confidence": "calculated",
            "kv_bytes_per_token_fp16": (2 * tc.get("num_key_value_heads", 0)
                                        * tc.get("head_dim", 0) * n_full * 2),
            "note": "attention seulement (n_full=16) ; etat recurrent GDN exclu (mamba_ssm_dtype=float32)",
        },
        "weight_stats_sample": sample_weight_stats(tensors, args.model),
        "caveats": [
            "weight_stats_sample = proxy HEURISTIC (RMS/max/outlier), PAS une mesure de qualite",
            "FP8 map = mesure (dtypes reels des shards), a croiser avec modules_to_not_convert",
            "BF16/F16/F8_E4M3 lus avec leurs vrais dtypes (jamais BF16->F16)",
        ],
    }

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, ensure_ascii=False, indent=2)

    print(f"[+] {len(tensors)} tensors / {len(per_shard)} shards / {profile['tensor_inventory']['total_gb']} GB")
    print(f"    hybrid : {n_full} full-attention + {n_lin} linear-attention (sur {tc.get('num_hidden_layers')})")
    print(f"    dtypes : {profile['tensor_inventory']['by_dtype']}")
    print(f"    FP8 convertis : {fmap['converted_fp8_count']} tensors, {round(fmap['converted_fp8_bytes']/1e9,1)} GB")
    print(f"    BF16 conserves: {fmap['kept_bf16_count']} tensors, {round(fmap['kept_bf16_bytes']/1e9,1)} GB")
    print(f"    scales FP8 (BF16 weight_scale_inv) : {fmap['fp8_scales_bf16']} tensors, {fmap['fp8_scales_bf16_gb']} GB")
    print(f"    kept poids sans prefixe declare : {fmap['kept_weights_without_declared_prefix']} ({fmap['kept_weights_without_declared_prefix_gb']} GB)")
    print(f"    convertis FP8 mais declares BF16 : {fmap['converted_fp8_but_declared_bf16']} ({fmap['converted_fp8_but_declared_bf16_gb']} GB)")
    print(f"[+] -> {args.json}")


if __name__ == "__main__":
    main()