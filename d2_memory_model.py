#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 MEMORY / TRANSPORT MODEL — Qwen3.8-27B-FP8 (precision-dependent)
===================================================================
La structure du modele est FIXEE (profil statique) ; on fait varier la precision
SANS re-profilage structurel. Seul le payload FP8 (24.7 GB, 407 tensors) est
re-quantifiable ; forbidden_bf16 (6.16 GB) et scales (3 MB) restent fixes.

  PRECISIONS:  FP8 / Q4 / Q2   (+ Q8 optionnel)
  TRAFFIC:     N'EST PAS deduit des seuls poids.
               decode = weights_read + kv_read + kv_write
                        + gdn_state_read + gdn_state_write   (etat recurrent lu ET ecrit chaque token)
                        + activation_transfer + router
  OFFLOW:      all_vram | vram_ram | vram_ram_pcie
  QUALITY:     contrainte QUALITY_LOSS <= MAX_ALLOWED_LOSS (d2_quant_reference.json)
  CONFIDENCE:  MEASURED / CALCULATED / HEURISTIC / UNKNOWN sur chaque valeur.

Usage:
  python d2_memory_model.py [--ctx 32768] [--max-loss 1.0]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(HERE, "d2_static_profile_27B.json")
REF = os.path.join(HERE, "d2_quant_reference.json")
OUT = os.path.join(HERE, "d2_memory_27B.json")

B = 10 ** 9  # octets/GiB decimal (aligné sur fichiers HF)

# ---------------------------------------------------------------------------
# machines (capacites / chemins)
# ---------------------------------------------------------------------------
MACHINES = {
    "rtx5070": {"vram_gb": 8, "ram_gb": 33.6, "ctx_max": 32768,
                "vram_bw_gbs": 306.0, "pcie_gbs": 14.0,
                "bw_src": "MEASURED (memset VRAM pic 306 Go/s ~ débit réel de décodage 290 Go/s)", "mem_src": "MEASURED"},
    "rtx4090": {"vram_gb": 24, "ram_gb": 64, "ctx_max": 65536,
                "vram_bw_gbs": 1008.0, "pcie_gbs": 28.0,
                "bw_src": "HEURISTIC(spec)", "mem_src": "HEURISTIC"},
    "rtx5090": {"vram_gb": 32, "ram_gb": 64, "ctx_max": 131072,
                "vram_bw_gbs": 1792.0, "pcie_gbs": 32.0,
                "bw_src": "HEURISTIC(spec)", "mem_src": "HEURISTIC"},
    "cpu32":   {"vram_gb": 0, "ram_gb": 32, "ctx_max": 32768,
                "vram_bw_gbs": 33.0, "pcie_gbs": 0.0,
                "bw_src": "MEASURED", "mem_src": "MEASURED"},
    "cpu64":   {"vram_gb": 0, "ram_gb": 64, "ctx_max": 65536,
                "vram_bw_gbs": 51.2, "pcie_gbs": 0.0,
                "bw_src": "HEURISTIC", "mem_src": "HEURISTIC"},
}

CTXS = [8192, 16384, 32768, 65536, 131072, 262144]

# bytes/weight supplementaires pour scales+metadata+alignment par precision
# (GGUF-like: group 32, scale fp16 -> ~0.0625 B/w ; alignment 64B)
SCALE_OVERHEAD = {
    "FP8": 0.001,   # scales 128x128, quasiment nul (mesure 3MB/24.7GB)
    "Q8": 0.05,
    "Q4": 0.07,
    "Q2": 0.09,
}


def quality_by_family(ref):
    """top1 par famille : MEILLEUR top1 de la famille (plafond atteignable), publisher-reported.
    Evite d'ecraser avec la reference 9B (top1=None) et ne prend pas le premier match."""
    best = {"Q8": None, "Q4": None, "Q2": None}
    for m in ref.get("models", []):
        for q in m.get("quants", []):
            name = q.get("quant", "")
            if q.get("top1") is None:
                continue
            for f in ("Q8", "Q4", "Q2"):
                if f in name and (best.get(f) is None or q["top1"] > best[f]["top1"]):
                    best[f] = {"top1": q["top1"], "kl": q.get("kl"),
                               "size_gb": q.get("size_gb"), "quant": name,
                               "source": "publisher-reported"}
    out = {f: best[f] for f in ("Q8", "Q4", "Q2")}
    out["FP8"] = {"top1": 99.5, "kl": 0.0001, "size_gb": None, "quant": "FP8_E4M3",
                  "source": "reference FP8 natif"}
    return out


def weight_sizes(profile, precision):
    """Payload par precision — CALCULATED. Seul fp8_payload varie ; forbidden+scales fixes."""
    pc = profile["precision_classes"]
    fp8_pay = pc["fp8_payload"]["bytes"]
    forbidden = pc["forbidden_bf16"]["bytes"]
    scales = pc["scales"]["bytes"]
    bits = {"FP8": 8, "Q8": 8, "Q4": 4, "Q2": 2}[precision]
    payload = fp8_pay * bits / 8
    payload += fp8_pay * SCALE_OVERHEAD[precision]  # scales+metadata du payload requantifie
    total = payload + forbidden + scales
    return {
        "precision": precision,
        "payload_bytes": payload, "payload_gb": round(payload / B, 3),
        "forbidden_bf16_bytes": forbidden, "forbidden_bf16_gb": round(forbidden / B, 3),
        "scales_bytes": scales, "scales_gb": round(scales / B, 5),
        "total_bytes": total, "total_gb": round(total / B, 2),
        "confidence": "CALCULATED",
        "note": "forbidden_bf16 + scales fixes ; seul fp8_payload(24.7GB) est requantifie",
    }


def kv_calculated(p, ctx):
    bpt = 2 * p["kv_heads"] * p["head_dim"] * p["n_full"] * 2  # fp16
    return {"bytes": bpt * ctx, "gb": round(bpt * ctx / B, 4),
            "bpt_fp16": bpt, "confidence": "CALCULATED"}


def gdn_state_calculated(p):
    per_layer = p["lin_k_heads"] * p["lin_k_dim"] * p["lin_v_heads"] * p["lin_v_dim"]
    db = 4 if p["ssm_dtype"] == "float32" else 2
    total = per_layer * p["n_lin"] * db
    return {"bytes": total, "gb": round(total / B, 3), "dtype": p["ssm_dtype"],
            "confidence": "CALCULATED"}


def activations(p, ctx, mode="decode"):
    """HEURISTIC : pic en vol. decode ~hidden; prefill ~hidden*ctx*3 couches."""
    if mode == "decode":
        bytes_ = p["hidden"] * 2 * 3
    else:
        bytes_ = (p["hidden"] * ctx * 2 * 3 + p["intermediate"] * 2)
    return {"bytes": bytes_, "gb": round(bytes_ / B, 4), "confidence": "HEURISTIC",
            "note": "depend batch/seq/hidden/dtype/impl — a calibrer"}


def workspace_by_precision(p, precision, resident_no_ws):
    """HEURISTIC : FP8/Q4/Q2 ont des buffers temporaires differents."""
    factor = {"FP8": 0.02, "Q8": 0.015, "Q4": 0.015, "Q2": 0.012}[precision]
    ws = max(resident_no_ws * factor, 512 * 10 ** 6)
    rt = 256 * 10 ** 6
    return {"workspace_bytes": ws, "workspace_gb": round(ws / B, 3),
            "runtime_bytes": rt, "runtime_gb": round(rt / B, 3),
            "confidence": "HEURISTIC"}


def traffic_decode(p, w, kv, state, act):
    """TRAFFIC decode (batch=1) — CALCULATED. Chaque composant compte.
    - weights_read : tous les poids residents (1x/token)
    - kv_read : contexte entier ; kv_write : 1 token
    - gdn_state : LU ET ECRIT a chaque token (etat recurrent)
    - activation_transfer : sorties inter-couches (hidden)
    - router : n/a (dense) — champ present pour MoE
    """
    return {
        "weights_read_gb": round(w["total_bytes"] / B, 3),
        "kv_read_gb": round(kv["bytes"] / B, 4),
        "kv_write_gb": round(kv["bpt_fp16"] / B, 6),
        "gdn_state_read_gb": round(state["bytes"] / B, 3),
        "gdn_state_write_gb": round(state["bytes"] / B, 3),
        "activation_gb": round(act["bytes"] / B, 5),
        "router_gb": 0.0,
        "total_gb": round((w["total_bytes"] + kv["bytes"] + kv["bpt_fp16"] +
                           2 * state["bytes"] + act["bytes"]) / B, 3),
        "confidence": "CALCULATED",
        "note": "poids lus 1x/token ; etat GDN lu+ecrit chaque token ; KV ctx+1",
    }


def traffic_prefill(p, w, kv, state, act, ctx):
    tot = (w["total_bytes"] / ctx + kv["bpt_fp16"] + 2 * state["bytes"] / ctx + act["bytes"] / ctx)
    return {"gb_per_token": round(tot / B, 5), "confidence": "CALCULATED",
            "note": "amorti sur seq=ctx : poids/ctx, etat GDN/ctx"}


def fit_and_tps(m, resident, offload_mode, traffic, w, kv, state, act):
    """Calcule tps et mode de fonctionnement pour une machine.
    Modes: all_vram (tout doit tenir), vram_ram (offload RAM), vram_ram_pcie (streaming)."""
    vram = m["vram_gb"] * B
    ram = m["ram_gb"] * B
    if m["vram_gb"] == 0:
        # CPU pur
        fits_ram = resident <= ram
        t = traffic["total_gb"] * B / (m["vram_bw_gbs"] * B)
        return {"fit": "YES" if fits_ram else "NO",
                "offload_gb": max(0.0, (resident - ram) / B),
                "est_token_s": round(t * 1000, 1), "est_tps": round(1.0 / t, 2),
                "mode": "ram"}
    resident_in_vram = resident
    if offload_mode == "all_vram":
        ok = resident_in_vram <= vram
        off = max(0.0, (resident_in_vram - vram) / B)
        # tout le trafic lu de VRAM si ok, sinon PCIe pour le surplus
        if ok:
            t = traffic["total_gb"] * B / (m["vram_bw_gbs"] * B)
            mode = "vram"
        else:
            offload_b = resident_in_vram - vram
            t_vram = (traffic["total_gb"] * B - offload_b) / (m["vram_bw_gbs"] * B)
            t_pcie = offload_b / (m["pcie_gbs"] * B)
            t = t_vram + t_pcie
            mode = "vram+pcie"
        return {"fit": "YES" if ok else "NO", "offload_gb": round(off, 2),
                "est_token_s": round(t * 1000, 1), "est_tps": round(1.0 / t, 2), "mode": mode}
    if offload_mode == "vram_ram":
        offload_b = max(0.0, resident - vram)
        ram_ok = offload_b <= ram
        t_vram = (traffic["total_gb"] * B - offload_b) / (m["vram_bw_gbs"] * B) if offload_b > 0 \
            else traffic["total_gb"] * B / (m["vram_bw_gbs"] * B)
        t_pcie = offload_b / (m["pcie_gbs"] * B) if offload_b > 0 else 0
        t = t_vram + t_pcie
        return {"fit": "YES" if ram_ok else "NO", "offload_gb": round(offload_b / B, 2),
                "est_token_s": round(t * 1000, 1), "est_tps": round(1.0 / t, 2),
                "mode": "vram_ram" if offload_b > 0 else "vram"}
    # vram_ram_pcie : streaming sans limite (disk swap)
    offload_b = max(0.0, resident - vram)
    t_vram = (traffic["total_gb"] * B - offload_b) / (m["vram_bw_gbs"] * B) if offload_b > 0 \
        else traffic["total_gb"] * B / (m["vram_bw_gbs"] * B)
    t_pcie = offload_b / (m["pcie_gbs"] * B) if offload_b > 0 else 0
    return {"fit": "YES", "offload_gb": round(offload_b / B, 2),
            "est_token_s": round((t_vram + t_pcie) * 1000, 1),
            "est_tps": round(1.0 / (t_vram + t_pcie), 2), "mode": "streaming"}


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 MEMORY/TRANSPORT (precision-dependent) — 27B FP8")
    ap.add_argument("--json", default=OUT)
    ap.add_argument("--ctx", type=int, default=32768, choices=CTXS)
    ap.add_argument("--max-loss", type=float, default=1.0,
                    help="perte top1 max autorisee vs reference FP8 (quality gate)")
    args = ap.parse_args()

    with open(PROFILE, encoding="utf-8") as fh:
        profile = json.load(fh)
    with open(os.path.join(HERE, "models", "Qwen3.8-27B-FP8", "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    tc = cfg.get("text_config", {})
    with open(REF, encoding="utf-8") as fh:
        ref = json.load(fh)
    quality = quality_by_family(ref)

    p = {
        "layers": tc.get("num_hidden_layers"),
        "n_full": sum(1 for t in tc.get("layer_types", []) if t == "full_attention"),
        "n_lin": sum(1 for t in tc.get("layer_types", []) if t == "linear_attention"),
        "kv_heads": tc.get("num_key_value_heads"), "head_dim": tc.get("head_dim"),
        "lin_k_heads": tc.get("linear_num_key_heads"), "lin_k_dim": tc.get("linear_key_head_dim"),
        "lin_v_heads": tc.get("linear_num_value_heads"), "lin_v_dim": tc.get("linear_value_head_dim"),
        "ssm_dtype": tc.get("mamba_ssm_dtype"), "hidden": tc.get("hidden_size"),
        "intermediate": tc.get("intermediate_size"),
    }

    kv = kv_calculated(p, args.ctx)
    state = gdn_state_calculated(p)
    act = activations(p, args.ctx, "decode")
    act_prefill = activations(p, args.ctx, "prefill")

    precisions = ["FP8", "Q8", "Q4", "Q2"]
    rows = {}
    report = {"generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
              "model": "Qwen3.8-27B-FP8 (natif FP8 fine-grained, pin 017b9c7)",
              "ctx": args.ctx, "max_allowed_loss_top1": args.max_loss,
              "architecture": p,
              "kv": kv, "gdn_state": state,
              "activations_decode": act, "activations_prefill": act_prefill,
              "quality_by_family": quality,
              "decisions": {}, "machines_detail": {}}

    for prec in precisions:
        w = weight_sizes(profile, prec)
        traffic_d = traffic_decode(p, w, kv, state, act)
        traffic_p = traffic_prefill(p, w, kv, state, act_prefill, args.ctx)
        resident_no_ws = w["total_bytes"] + kv["bytes"] + state["bytes"] + act["bytes"]
        ws_rt = workspace_by_precision(p, prec, resident_no_ws)
        resident = resident_no_ws + ws_rt["workspace_bytes"] + ws_rt["runtime_bytes"]

        # quality gate
        q = quality.get(prec)
        loss = max(0.0, 99.5 - q["top1"]) if q and q.get("top1") else 99.5
        gate = "PASS" if loss <= args.max_loss else "FAIL"

        fits = {}
        for name, m in MACHINES.items():
            fits[name] = {mode: fit_and_tps(m, resident, mode, traffic_d, w, kv, state, act)
                          for mode in ("all_vram", "vram_ram", "vram_ram_pcie")}

        # decision: meilleure config par machine = fit YES + tps max + quality gate PASS
        decision = {}
        for name, m in MACHINES.items():
            if gate != "PASS":
                decision[name] = {"fit": "NO", "reason": "quality FAIL "
                                   f"(loss {loss:.1f} > {args.max_loss})"}
                continue
            best = None
            for mode, f in fits[name].items():
                if f["fit"] == "YES":
                    if best is None or f["est_tps"] > best["est_tps"]:
                        best = {"mode": mode, **f}
            decision[name] = best or {"fit": "NO", "reason": "no fit (resident > capacite)"}

        rows[prec] = {
            "weights": w,
            "resident_gb": round(resident / B, 2),
            "traffic_decode": traffic_d,
            "traffic_prefill": traffic_p,
            "quality": {"top1": q["top1"] if q else None, "kl": q["kl"] if q else None,
                        "loss_vs_fp8": loss, "gate": gate},
            "workspace_runtime": ws_rt,
            "fit": fits,
        }
        report["decisions"][prec] = decision
        report["machines_detail"][prec] = rows[prec]

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"[+] D2 MEMORY/TRANSPORT 27B — ctx={args.ctx}, max_loss={args.max_loss}")
    print(f"    KV@ctx={kv['gb']} GB | GDN state={state['gb']} GB ({state['dtype']}) | activ decode={act['gb']} GB")
    print()
    hdr = f"    {'prec':6s} {'poids':>8s} {'resident':>9s} {'traffic':>9s} {'top1':>7s} {'loss':>6s} gate"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for prec in precisions:
        r = rows[prec]
        print(f"    {prec:6s} {r['weights']['total_gb']:8.2f} {r['resident_gb']:9.2f} "
              f"{r['traffic_decode']['total_gb']:9.2f} {r['quality']['top1'] or 0:7.1f} "
              f"{r['quality']['loss_vs_fp8']:6.2f} {r['quality']['gate']:4s}")
    print()
    print("    FIT par machine (meilleur mode / tps est):")
    for name in MACHINES:
        parts = []
        for prec in precisions:
            d = report["decisions"][prec][name]
            if d.get("fit") == "NO":
                parts.append(f"{prec}=NO")
            else:
                parts.append(f"{prec}={d.get('mode')}~{d.get('est_tps')}")
        print(f"      {name:9s}  " + "  ".join(parts))
    print()
    print("    RECOMMENDATION:")
    for name in MACHINES:
        cands = [(prec, report["decisions"][prec][name]) for prec in precisions
                 if report["decisions"][prec][name].get("fit") == "YES"]
        if not cands:
            print(f"      {name:9s}: aucun fit (resident > capacite memoire)")
        else:
            best = max(cands, key=lambda c: c[1].get("est_tps", 0))
            loss = rows[best[0]]['quality']['loss_vs_fp8']
            print(f"      {name:9s}: {best[0]} (mode {best[1].get('mode')}, ~{best[1].get('est_tps')} t/s, "
                  f"offload {best[1].get('offload_gb')} GB, loss top1 {loss:.2f})")
    print(f"[+] -> {args.json}")


if __name__ == "__main__":
    main()