#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 SELECTIVE QUANTIZATION OPTIMIZER — Qwen3.8-27B-FP8
======================================================
Moteur de recherche de precision mixte par ROLE.

Structure fixe (profil statique) ; seul le payload FP8 (24.7 GB) est requantifiable.
Roles: linear_attn (SSM, 5.54 GB) | self_attn (1.78 GB) | mlp (17.38 GB)
forbidden_bf16 (6.16 GB) + scales (3 MB) restent fixes.

  OBJECTIF   : minimiser (resident + transport_cost)
  CONTRAINTES: quality_loss <= max_loss ; quality_loss << Q2 ; hardware_fit
  PREFERENCE : poids dans la zone cible [15, 18] GB (entre Q2 et Q4)
  QUALITE    : modele additif calibre sur les references publisher-reported
               (uniform Q4 -> loss 3.07, uniform Q2 -> 12.32, Q3 -> 7.09)
               perte(role,prec) = part_payload_role x sensibilite_role x loss_uniforme(prec)
               sensibilite par role = outlier_rate moyen (profil statique), normalise
               pour que la moyenne ponderee = 1 (les unif formes reproduisent la reference)
  CONFIDENCE : HEURISTIC (quality additif) — validation finale = golden suite

Usage:
  python d2_selective_optimizer.py [--max-loss 3.5] [--machine all]
"""

import argparse
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(HERE, "d2_static_profile_27B.json")
REF = os.path.join(HERE, "d2_quant_reference.json")
OUT = os.path.join(HERE, "d2_selective_map.json")

B = 10 ** 9

# pertes uniformes (top1) — publisher-reported (d2_quant_reference.json)
UNIFORM_LOSS = {"FP8": 0.0, "Q8": 0.57, "Q4": 3.07, "Q3": 7.09, "Q2": 12.32}
BITS = {"FP8": 8, "Q8": 8, "Q4": 4, "Q3": 3, "Q2": 2}
SCALE_OVERHEAD = {"FP8": 0.001, "Q8": 0.05, "Q4": 0.07, "Q3": 0.08, "Q2": 0.09}
PRECISIONS = ["FP8", "Q8", "Q4", "Q3", "Q2"]

MACHINES = {
    "rtx5070": {"vram_gb": 8, "ram_gb": 33.6, "vram_bw_gbs": 306.0, "pcie_gbs": 14.0, "bw_src": "MEASURED"},
    "rtx4090": {"vram_gb": 24, "ram_gb": 64, "vram_bw_gbs": 1008.0, "pcie_gbs": 28.0, "bw_src": "HEURISTIC"},
    "rtx5090": {"vram_gb": 32, "ram_gb": 64, "vram_bw_gbs": 1792.0, "pcie_gbs": 32.0, "bw_src": "HEURISTIC"},
    "cpu32":   {"vram_gb": 0, "ram_gb": 32, "vram_bw_gbs": 33.0, "pcie_gbs": 0.0, "bw_src": "MEASURED"},
    "cpu64":   {"vram_gb": 0, "ram_gb": 64, "vram_bw_gbs": 51.2, "pcie_gbs": 0.0, "bw_src": "HEURISTIC"},
}

# role -> octets payload FP8 (mesure depuis le profil statique)
ROLES = ["linear_attn", "self_attn", "mlp"]


def role_payload_bytes(profile):
    b = profile["tensor_inventory"]["by_role_dtype_bytes"]
    return {r: b.get(f"{r}/F8_E4M3", 0) for r in ROLES}


def role_sensitivity(profile, role_bytes):
    """Sensibilite par role = outlier_rate moyen (profil statique, proxy HEURISTIC),
    normalisee pour que la moyenne ponderee par octets = 1."""
    ws = [w for w in profile["weight_stats_sample"] if "error" not in w]
    raw = {"linear_attn": [], "self_attn": [], "mlp": [], "norm": [], "other": []}
    for w in ws:
        n = w["name"]
        if "linear_attn" in n: r = "linear_attn"
        elif "self_attn" in n: r = "self_attn"
        elif "mlp" in n: r = "mlp"
        elif "layernorm" in n or n.endswith("norm.weight"): r = "norm"
        else: r = "other"
        raw[r].append(w["outlier_rate"])
    sens = {}
    for r in ROLES:
        l = raw.get(r, [])
        sens[r] = (sum(l) / len(l)) if l else 0.001
    total = sum(role_bytes[r] for r in ROLES)
    wavg = sum(sens[r] * role_bytes[r] for r in ROLES) / total
    for r in ROLES:
        sens[r] /= wavg  # moyenne ponderee = 1 -> uniform reproduit la reference
    return sens


def mixed_weights(profile, role_bytes, assignment):
    """Poids mixte — CALCULATED. forbidden + scales fixes ; payload par role requantifie."""
    pc = profile["precision_classes"]
    forbidden = pc["forbidden_bf16"]["bytes"]
    scales = pc["scales"]["bytes"]
    payload = 0.0
    detail = {}
    for r in ROLES:
        prec = assignment[r]
        bytes_role = role_bytes[r] * BITS[prec] / 8
        bytes_role += role_bytes[r] * SCALE_OVERHEAD[prec]
        payload += bytes_role
        detail[r] = {"precision": prec, "bytes": bytes_role,
                     "gb": round(bytes_role / B, 3),
                     "src_gb": round(role_bytes[r] / B, 3)}
    total = payload + forbidden + scales
    return {"total_bytes": total, "total_gb": round(total / B, 2),
            "payload_gb": round(payload / B, 3),
            "forbidden_gb": round(forbidden / B, 3),
            "scales_gb": round(scales / B, 5),
            "detail": detail, "confidence": "CALCULATED",
            "assignment": assignment}


def quality_loss(profile, role_bytes, sens, assignment):
    """Modele additif HEURISTIC — calibre pour reproduire uniform Q4=3.07, Q2=12.32."""
    total = sum(role_bytes[r] for r in ROLES)
    loss = 0.0
    detail = {}
    for r in ROLES:
        part = role_bytes[r] / total
        prec = assignment[r]
        l = part * sens[r] * UNIFORM_LOSS[prec]
        loss += l
        detail[r] = {"precision": prec, "part": round(part, 4),
                     "sens": round(sens[r], 3), "loss": round(l, 3)}
    return {"loss": round(loss, 2), "detail": detail, "confidence": "HEURISTIC",
            "model": "sum(part x sens x uniform_loss) calibre sur publisher-reported"}


def resident_traffic(w, p, kv_bytes, state_bytes, act_bytes, ws_rt):
    resident = w["total_bytes"] + kv_bytes + state_bytes + act_bytes + ws_rt["workspace_bytes"] + ws_rt["runtime_bytes"]
    traffic = (w["total_bytes"] + kv_bytes + state_bytes * 2 + act_bytes +
               kv_bytes / (p["ctx"]))  # kv_write ~ 1 token, approx
    return resident, traffic


def fit_machine(m, resident_bytes, traffic_bytes):
    """Retourne (fit, mode, est_tps). 3 modes: all_vram / vram_ram / streaming."""
    vram = m["vram_gb"] * B
    ram = m["ram_gb"] * B
    if m["vram_gb"] == 0:
        fits = resident_bytes <= ram
        t = traffic_bytes / (m["vram_bw_gbs"] * B)
        return {"fit": "YES" if fits else "NO", "mode": "ram",
                "est_tps": round(1.0 / t, 2), "est_token_s": round(t * 1000, 1),
                "offload_gb": round(max(0.0, (resident_bytes - ram) / B), 2)}
    off = max(0.0, (resident_bytes - vram) / B)
    best = None
    for mode in ("all_vram", "vram_ram", "streaming"):
        if mode == "all_vram":
            if off > 0:
                continue  # ne tient pas en tout VRAM
            t = traffic_bytes / (m["vram_bw_gbs"] * B)
            f = "YES"
        else:
            if mode == "vram_ram" and off > ram / B:
                continue
            t_vram = max(0.0, traffic_bytes - off * B) / (m["vram_bw_gbs"] * B)
            t_pcie = (off * B / (m["pcie_gbs"] * B)) if off > 0 and m["pcie_gbs"] > 0 else 0.0
            t = t_vram + t_pcie
            f = "YES"
        cand = {"fit": f, "mode": mode, "est_tps": round(1.0 / t, 2),
                "est_token_s": round(t * 1000, 1), "offload_gb": round(off, 2)}
        if best is None or cand["est_tps"] > best["est_tps"]:
            best = cand
    return best


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 SELECTIVE QUANTIZATION — 27B FP8")
    ap.add_argument("--json", default=OUT)
    ap.add_argument("--max-loss", type=float, default=3.5,
                    help="perte top1 max autorisee (quality gate)")
    ap.add_argument("--target-min", type=float, default=15.0)
    ap.add_argument("--target-max", type=float, default=18.0)
    ap.add_argument("--machine", default="all", choices=["all"] + list(MACHINES))
    args = ap.parse_args()

    with open(PROFILE, encoding="utf-8") as fh:
        profile = json.load(fh)
    with open(os.path.join(HERE, "models", "Qwen3.8-27B-FP8", "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    tc = cfg.get("text_config", {})
    p = {
        "ctx": args_machine_ctx(args, tc),
        "kv_heads": tc.get("num_key_value_heads"), "head_dim": tc.get("head_dim"),
        "n_full": sum(1 for t in tc.get("layer_types", []) if t == "full_attention"),
        "lin_k_heads": tc.get("linear_num_key_heads"), "lin_k_dim": tc.get("linear_key_head_dim"),
        "lin_v_heads": tc.get("linear_num_value_heads"), "lin_v_dim": tc.get("linear_value_head_dim"),
        "n_lin": sum(1 for t in tc.get("layer_types", []) if t == "linear_attention"),
        "ssm_dtype": tc.get("mamba_ssm_dtype"), "hidden": tc.get("hidden_size"),
        "intermediate": tc.get("intermediate_size"),
    }

    role_bytes = role_payload_bytes(profile)
    sens = role_sensitivity(profile, role_bytes)
    kv_bytes = 2 * p["kv_heads"] * p["head_dim"] * p["n_full"] * 2 * p["ctx"]
    state_bytes = p["lin_k_heads"] * p["lin_k_dim"] * p["lin_v_heads"] * p["lin_v_dim"] * p["n_lin"] * 4
    act_bytes = p["hidden"] * 2 * 3
    ws_rt = {"workspace_bytes": 512 * 10 ** 6, "runtime_bytes": 256 * 10 ** 6}

    results = []
    for assignment in itertools.product(PRECISIONS, repeat=len(ROLES)):
        a = dict(zip(ROLES, assignment))
        w = mixed_weights(profile, role_bytes, a)
        q = quality_loss(profile, role_bytes, sens, a)
        resident, traffic = resident_traffic(w, p, kv_bytes, state_bytes, act_bytes, ws_rt)
        if q["loss"] > args.max_loss:
            continue  # quality gate FAIL
        machines = {}
        for name in ([args.machine] if args.machine != "all" else MACHINES):
            m = MACHINES[name]
            machines[name] = fit_machine(m, resident, traffic)
        results.append({"assignment": a, "weights_gb": w["total_gb"],
                        "resident_gb": round(resident / B, 2),
                        "traffic_gb": round(traffic / B, 2),
                        "loss": q["loss"], "quality_detail": q["detail"],
                        "machines": machines,
                        "in_target_zone": args.target_min <= w["total_gb"] <= args.target_max})

    results.sort(key=lambda r: (r["weights_gb"], r["loss"]))

    report = {"generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
              "model": "Qwen3.8-27B-FP8", "ctx": p["ctx"],
              "max_loss": args.max_loss, "target_zone_gb": [args.target_min, args.target_max],
              "uniform_loss_reference": UNIFORM_LOSS,
              "role_bytes_gb": {r: round(role_bytes[r] / B, 3) for r in ROLES},
              "role_sensitivity": {r: round(sens[r], 3) for r in ROLES},
              "kv_gb": round(kv_bytes / B, 3), "gdn_state_gb": round(state_bytes / B, 3),
              "candidates": results[:200],
              "summary": summarize(results, args),
              "caveats": [
                  "quality = modele additif HEURISTIC (part x sens x uniform_loss) calibre sur publisher-reported",
                  "sensibilite par role = outlier_rate moyen du profil statique (proxy, echantillon 3 couches)",
                  "validation finale requise: PPL + logits KL + task benchmark (golden suite)",
                  "loss top1 = proxy, pas une mesure linguistique",
              ]}

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"[+] D2 SELECTIVE — ctx={p['ctx']}, max_loss={args.max_loss}, target [{args.target_min},{args.target_max}] GB")
    print(f"    sensibilite (norm): " + ", ".join(f"{r}={sens[r]:.2f}" for r in ROLES))
    print(f"    candidates passant le gate: {len(results)}")
    print()
    best_zone = [r for r in results if r["in_target_zone"]]
    best_global = results[:1]
    show = (best_zone or best_global)[:6]
    for r in show:
        tag = "TARGET" if r["in_target_zone"] else "      "
        print(f"  [{tag}] {r['weights_gb']:5.2f} GB | resident {r['resident_gb']:6.2f} | traffic {r['traffic_gb']:6.2f} "
              f"| loss {r['loss']:5.2f} | {r['assignment']}")
    print()
    print("    per-machine best:")
    for name in ([args.machine] if args.machine != "all" else MACHINES):
        ok = [r for r in results if r["machines"].get(name, {}).get("fit") == "YES"]
        if not ok:
            print(f"      {name:9s}: aucun candidat ne tient")
        else:
            b = min(ok, key=lambda r: r["weights_gb"])
            mm = b["machines"][name]
            print(f"      {name:9s}: {b['weights_gb']:5.2f} GB loss {b['loss']:5.2f} "
                  f"mode {mm['mode']:9s} ~{mm['est_tps']} t/s offload {mm['offload_gb']} GB  {b['assignment']}")
    print(f"[+] -> {args.json}")


def args_machine_ctx(args, tc):
    # Contexte natif du modèle (max_position_embeddings), repli 32768.
    return int(tc.get("max_position_embeddings") or 32768)


def summarize(results, args):
    zone = [r for r in results if r["in_target_zone"]]
    return {
        "candidates_gate_pass": len(results),
        "in_target_zone": len(zone),
        "best_in_zone": (zone[0] if zone else None),
        "best_global": (results[0] if results else None),
        "note": "tries par (poids, loss) croissants",
    }


if __name__ == "__main__":
    main()