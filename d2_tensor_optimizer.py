#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 TENSOR-LEVEL OPTIMIZER — Qwen3.8-27B-FP8
=============================================
Selection de precision mixte par TENSOR (granularite layer x tensor), optimisant
les 4 axes simultanement: RAM / transport / precision / goulot reel.

Sources:
  - d2_tensor_profile_full.json  (per-tensor rel_err mesure pour Q4/Q3/Q2/Q1 + channel proxy)
  - d2_static_profile_27B.json   (precision_classes: fp8_payload/forbidden/scales)
  - d2_memory_model.json         (KV, GDN state, activations, workspace, trafic)

Principe (spec D2-27B):
  D2 SCORE = qualite - l1*resident - l2*bytes_per_token - l3*dma - l4*pcie - l5*sync
  -> pas un simple min(weight_size). Le GDN 2.42GB est FIXE quel que soit les poids:
  on ne cherche pas le gain uniquement dans les poids.

Qualite: modele additif par tensor, calibre sur publisher-reported
  loss = sum_i part_i x sens_i x uniform_loss(p_i)
  part_i = bytes_i / payload_total ; sens_i = rel_err_i moyen / rel_err moyen (normalise,
  moyenne ponderee = 1 -> uniform reproduit la reference Q4=3.07 / Q2=12.32).
  Q1 = ternaire {-1,0,+1}, CONDITIONNEL (jamais baseline).

Trois profils (le moteur choisit selon la machine):
  D2-ECO        ~16 GB     priorite RAM
  D2-BALANCED   ~17-19 GB  priorite precision + transport
  D2-PERFORMANCE~19-21 GB  priorite TPS

Usage:
  python d2_tensor_optimizer.py [--json d2_tensor_optim.json] [--machine all]
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TENSOR_PROFILE = os.path.join(HERE, "d2_tensor_profile_full.json")
STATIC = os.path.join(HERE, "d2_static_profile_27B.json")
OUT = os.path.join(HERE, "d2_tensor_optim.json")

B = 10 ** 9

# pertes uniformes (top1) — publisher-reported
UNIFORM_LOSS = {"FP8": 0.0, "Q8": 0.57, "Q4": 3.07, "Q3": 7.09, "Q2": 12.32}
# Q1 = ternaire, CONDITIONNEL : perte extrapolee (HEURISTIC, a valider golden suite)
UNIFORM_LOSS["Q1"] = 18.0
BITS = {"FP8": 8, "Q8": 8, "Q4": 4, "Q3": 3, "Q2": 2, "Q1": 1}
SCALE_OVERHEAD = {"FP8": 0.001, "Q8": 0.05, "Q4": 0.07, "Q3": 0.08, "Q2": 0.09, "Q1": 0.12}
PRECISIONS = ["FP8", "Q8", "Q4", "Q3", "Q2", "Q1"]
# echelle de precision (rang decroissant)
PREC_RANK = {p: i for i, p in enumerate(PRECISIONS)}

MACHINES = {
    "rtx5070": {"vram_gb": 8, "ram_gb": 33.6, "vram_bw_gbs": 306.0, "pcie_gbs": 14.0, "bw_src": "MEASURED"},
    "rtx4090": {"vram_gb": 24, "ram_gb": 64, "vram_bw_gbs": 1008.0, "pcie_gbs": 28.0, "bw_src": "HEURISTIC"},
    "rtx5090": {"vram_gb": 32, "ram_gb": 64, "vram_bw_gbs": 1792.0, "pcie_gbs": 32.0, "bw_src": "HEURISTIC"},
    "cpu32":   {"vram_gb": 0, "ram_gb": 32, "vram_bw_gbs": 33.0, "pcie_gbs": 0.0, "bw_src": "MEASURED"},
    "cpu64":   {"vram_gb": 0, "ram_gb": 64, "vram_bw_gbs": 51.2, "pcie_gbs": 0.0, "bw_src": "HEURISTIC"},
}

# machines -> profil recommande (spec D2)
MACHINE_PROFILE = {
    "rtx5070": "ECO",          # 8 GB VRAM  -> ECO
    "rtx4090": "PERFORMANCE",  # 24 GB      -> PERFORMANCE
    "rtx5090": "PERFORMANCE",  # 32 GB      -> FP8 / PERFORMANCE
    "cpu32":   "ECO",          # 32 GB RAM  -> ECO/BALANCED
    "cpu64":   "BALANCED",     # 64 GB RAM  -> BALANCED/PERFORMANCE
}

PROFILES = {
    #            cible_gb      lambdas (qualite, resident, trafic)  max_loss
    "ECO":        {"target": 16.0, "l_q": 1.0, "l_res": 0.30, "l_tra": 0.03, "max_loss": 8.0},
    "BALANCED":   {"target": 18.0, "l_q": 1.0, "l_res": 0.18, "l_tra": 0.06, "max_loss": 4.0},
    "PERFORMANCE": {"target": 20.0, "l_q": 1.0, "l_res": 0.10, "l_tra": 0.10, "max_loss": 2.5},
}

ROLE_OF = {"linear_attn": "SSM/GDN", "self_attn": "attn", "mlp": "MLP", "norm": "norm", "other": "other"}


def load_payload_tensors(tprofile, static):
    """Tensors FP8 payload (requantifiables) avec rel_err mesure par precision.
    Retourne liste de dict {name, bytes, role, layer, rel:{prec:rel_err}, chan:{prec:ratio}}."""
    # ensemble des payload FP8 (profil statique -> les F8_E4M3 hors forbidden)
    pc = static["precision_classes"]
    forbidden_bytes = pc["forbidden_bf16"]["bytes"]
    scales_bytes = pc["scales"]["bytes"]
    payload = []
    for t in tprofile.get("tensors", []):
        if "error" in t or "precision" not in t or not t["precision"]:
            continue
        if t.get("dtype") != "F8_E4M3":
            continue
        name = t["name"]
        role = "linear_attn" if "linear_attn" in name else \
               ("self_attn" if "self_attn" in name else ("mlp" if ".mlp." in name else
                ("norm" if "norm" in name else "other")))
        rel = {p: t["precision"][p]["rel_err"] for p in ("Q4", "Q3", "Q2", "Q1") if p in t["precision"]}
        chan = {p: t["precision"][p]["channel_weighted_rel"] for p in ("Q4", "Q3", "Q2", "Q1")
                if p in t["precision"] and t["precision"][p].get("channel_weighted_rel") is not None}
        payload.append({
            "name": name, "bytes": t["bytes"], "role": role,
            "layer": (name.split(".layers.")[1].split(".")[0] if ".layers." in name else None),
            "rel": rel, "chan": chan,
        })
    return payload, forbidden_bytes, scales_bytes


def tensor_sensitivity(payload):
    """Sensibilite par tensor — derivee du channel_weighted_rel (proxy activation-aware,
    RMS de ligne, meilleur discriminant que rel_err brut) quand dispo, sinon rel_err.
    Normalisee (moyenne ponderee = 1). HEURISTIC."""
    tot = sum(t["bytes"] for t in payload)
    # moyenne ponderee de la metrique de sensibilite par precision
    metric = "chan" if any(t.get("chan") for t in payload) else "rel"
    mean_m = {}
    for p in ("Q4", "Q3", "Q2", "Q1"):
        s = sum(t["bytes"] * (t[metric].get(p, 0) if t[metric] else 0) for t in payload)
        mean_m[p] = s / tot if tot else 0.0
    sens = {}
    for t in payload:
        vals = []
        for p in ("Q4", "Q3", "Q2"):
            m = (t[metric].get(p) if t[metric] else None)
            if m is not None and mean_m[p] > 0:
                vals.append(m / mean_m[p])
        sens[t["name"]] = (sum(vals) / len(vals)) if vals else 1.0
    wavg = sum(t["bytes"] * sens[t["name"]] for t in payload) / tot
    for t in payload:
        sens[t["name"]] /= wavg
    return sens, mean_m


def quality_loss(payload, sens, assignment):
    """Modele additif par tensor — calibre pour reproduire uniform Q4=3.07, Q2=12.32.
    HEURISTIC. loss = sum_i part_i x sens_i x uniform_loss(p_i)."""
    tot = sum(t["bytes"] for t in payload)
    loss = 0.0
    detail = []
    for t in payload:
        p = assignment[t["name"]]
        part = t["bytes"] / tot
        l = part * sens[t["name"]] * UNIFORM_LOSS[p]
        loss += l
        detail.append({"name": t["name"], "prec": p, "gb": round(t["bytes"] / B, 5),
                       "part": round(part, 6), "sens": round(sens[t["name"]], 3),
                       "loss": round(l, 5)})
    return {"loss": round(loss, 3), "detail": detail, "confidence": "HEURISTIC",
            "model": "sum(part x sens x uniform_loss), sens = rel_err mesure normalise"}


def tensor_bytes(prec, src_bytes):
    return src_bytes * BITS[prec] / 8 + src_bytes * SCALE_OVERHEAD[prec]


def resident_traffic(weights_bytes, extra):
    """Resident et trafic decode — CALCULATED. GDN state lu+ecrit chaque token."""
    kv, state, act, ws_rt = extra["kv"], extra["state"], extra["act"], extra["ws_rt"]
    resident = weights_bytes + kv + state + act + ws_rt["workspace_bytes"] + ws_rt["runtime_bytes"]
    traffic = weights_bytes + kv + state * 2 + act + kv / extra["ctx"]
    return resident, traffic


def fit_machine(m, resident_bytes, traffic_bytes):
    vram = m["vram_gb"] * B
    ram = m["ram_gb"] * B
    if m["vram_gb"] == 0:
        fits = resident_bytes <= ram
        t = traffic_bytes / (m["vram_bw_gbs"] * B)
        return {"fit": "YES" if fits else "NO", "mode": "ram",
                "est_tps": round(1.0 / t, 2), "offload_gb": round(max(0.0, (resident_bytes - ram) / B), 2)}
    off = max(0.0, (resident_bytes - vram) / B)
    best = None
    for mode in ("all_vram", "vram_ram", "streaming"):
        if mode == "all_vram":
            if off > 0:
                continue
            t = traffic_bytes / (m["vram_bw_gbs"] * B)
        else:
            if mode == "vram_ram" and off > ram / B:
                continue
            t_vram = max(0.0, traffic_bytes - off * B) / (m["vram_bw_gbs"] * B)
            t_pcie = (off * B / (m["pcie_gbs"] * B)) if off > 0 and m["pcie_gbs"] > 0 else 0.0
            t = t_vram + t_pcie
        cand = {"fit": "YES", "mode": mode, "est_tps": round(1.0 / t, 2),
                "offload_gb": round(off, 2)}
        if best is None or cand["est_tps"] > best["est_tps"]:
            best = cand
    return best


def greedy_assign(payload, sens, profile_cfg, fixed_bytes=0.0):
    """Optimisation sous contrainte (knapsack greed) : part de tout-Q4 (reference
    uniforme = loss 3.07), puis applique les pas marginaux par meilleur ratio.

    - downgrade (Q4->Q3->Q2->Q1) : ratio = octets_sauves / loss_ajoutee  (le plus grand d'abord)
    - upgrade   (Q4->Q8->FP8)    : ratio = loss_sauvee / octets_coutes (le plus grand d'abord)
    - budget : total loss <= max_loss (gate). Cible : approcher target_gb de poids TOTAUX
      (payload + forbidden + scales). fixed_bytes = forbidden + scales.

    La granularite TENSOR (pas role) permet de descendre selectivement des tensors
    robustes du MLP sans sacrifier les tensors sensibles — casse le plafond role-level.

    NB (correction) : le delta de PERTE et de TAILLE d'un pas est recalcule par rapport
    a l'ETAT COURANT de chaque tensor (pas a la baseline Q4). Sinon un tensor degrade en
    plusieurs pas (Q4->Q3->Q2) voyait sa perte double-comptee."""
    start = {t["name"]: "Q4" for t in payload}
    target_total = profile_cfg["target"] * B
    target_payload = target_total - fixed_bytes
    max_loss = profile_cfg["max_loss"]

    def total_loss(assign):
        return quality_loss(payload, sens, assign)["loss"]

    def total_weights(assign):
        return sum(tensor_bytes(assign[t["name"]], t["bytes"]) for t in payload) + fixed_bytes

    # precompute contributions par tensor (part x sens)
    tot = sum(t["bytes"] for t in payload)
    contrib = {t["name"]: t["bytes"] / tot * sens[t["name"]] for t in payload}
    bytes_of = {t["name"]: t["bytes"] for t in payload}

    # etapes candidates: (tensor, prec, delta_bytes, delta_loss) vs baseline Q4
    # (db/dl servent uniquement a CLASSER les pas ; les deltas appliques sont recalculés)
    steps = []
    for t in payload:
        nm = t["name"]
        w0 = tensor_bytes("Q4", t["bytes"])
        for p in PRECISIONS:
            if p == "Q4":
                continue
            db = tensor_bytes(p, t["bytes"]) - w0
            dl = contrib[nm] * (UNIFORM_LOSS[p] - UNIFORM_LOSS["Q4"])
            steps.append({"name": nm, "prec": p, "db": db, "dl": dl})

    assign = dict(start)
    loss = total_loss(assign)

    def step_dl(nm, from_p, to_p):
        return contrib[nm] * (UNIFORM_LOSS[to_p] - UNIFORM_LOSS[from_p])

    def apply_step(s):
        nonlocal loss
        prev = assign[s["name"]]
        assign[s["name"]] = s["prec"]
        loss += step_dl(s["name"], prev, s["prec"])

    weights = total_weights(assign)

    # --- phase downgrade : au-dessus de la cible -> descendre les moins sensibles
    dg = [s for s in steps if s["db"] < 0]
    dg.sort(key=lambda s: (-s["db"]) / max(s["dl"], 1e-9), reverse=True)
    for s in dg:
        if weights <= target_total * 0.98:
            break
        cur = assign[s["name"]]
        if PREC_RANK[cur] >= PREC_RANK[s["prec"]]:
            continue
        if loss + step_dl(s["name"], cur, s["prec"]) > max_loss:
            continue
        apply_step(s)
        weights = total_weights(assign)

    # --- phase upgrade : sous la cible ou budget restant -> remonter les plus sensibles.
    # Un upgrade REDUIT la perte (dl<0) : le gate max_loss ne s'applique qu'aux downgrades.
    up = [s for s in steps if s["db"] > 0]
    up.sort(key=lambda s: (-s["dl"]) / max(s["db"], 1e-9), reverse=True)
    for s in up:
        cur = assign[s["name"]]
        if PREC_RANK[cur] <= PREC_RANK[s["prec"]]:
            continue
        db = tensor_bytes(s["prec"], bytes_of[s["name"]]) - tensor_bytes(cur, bytes_of[s["name"]])
        if weights + db > target_total * 1.05:
            continue  # depassement de cible : essayer les pas suivants (plus petits)
        apply_step(s)
        weights = total_weights(assign)

    return assign


def apply_max_loss(assign, payload, sens, max_loss, budget=None):
    """Si le loss depasse max_loss, releve les tensors les plus sensibles vers une
    precision superieure jusqu'a repasser sous le budget (passe locale).
    budget = nombre max d'iterations (1 tensor par cran par iteration) ; par defaut
    assez grand pour converger meme sur un gros payload (le plus contributif d'abord)."""
    if budget is None:
        budget = len(payload) * len(PRECISIONS)
    cur = quality_loss(payload, sens, assign)
    for _ in range(budget):
        if cur["loss"] <= max_loss:
            break
        # tensor le plus contributif (loss x part) a relever d'un cran
        worst = max(cur["detail"], key=lambda d: d["loss"])
        p = assign[worst["name"]]
        up = PRECISIONS[max(0, PREC_RANK[p] - 1)]
        if up == p:
            break
        assign[worst["name"]] = up
        cur = quality_loss(payload, sens, assign)
    return assign


def build_report(payload, forbidden_bytes, scales_bytes, sens, extra, machines, profile):
    cfg = PROFILES[profile]
    assign = greedy_assign(payload, sens, cfg, fixed_bytes=forbidden_bytes + scales_bytes)
    # enforce le gate max_loss (le greedy peut depasser le gate via la phase upgrade)
    assign = apply_max_loss(assign, payload, sens, cfg["max_loss"])
    q = quality_loss(payload, sens, assign)
    weights = sum(tensor_bytes(assign[t["name"]], t["bytes"]) for t in payload) + forbidden_bytes + scales_bytes
    resident, traffic = resident_traffic(weights, extra)
    # repartition par precision
    by_prec = {}
    for t in payload:
        p = assign[t["name"]]
        by_prec[p] = by_prec.get(p, 0) + tensor_bytes(p, t["bytes"])
    fit = {name: fit_machine(MACHINES[name], resident, traffic) for name in machines}
    # D2 SCORE (spec) = qualite - l1*resident - l2*traffic
    score = q["loss"] - cfg["l_res"] * resident / B - cfg["l_tra"] * traffic / B
    return {
        "profile": profile, "target_gb": cfg["target"],
        "weights_gb": round(weights / B, 2), "resident_gb": round(resident / B, 2),
        "traffic_gb": round(traffic / B, 2),
        "loss": q["loss"], "max_loss_gate": cfg["max_loss"],
        "d2_score": round(score, 2),
        "by_precision_gb": {p: round(b / B, 3) for p, b in sorted(by_prec.items())},
        "recommended_machines": [n for n in machines if MACHINE_PROFILE.get(n) == profile],
        "fit": fit,
        "quality_detail": q["detail"],
        "assignment": assign,
        "confidence": {"weights": "CALCULATED", "quality": "HEURISTIC",
                       "transport": "CALCULATED", "fit": "CALCULATED/HEURISTIC"},
        "caveats": [
            "greedy marginal par tensor (pas global optimal) ; granularite tensor casse le plafond role",
            "Q1 = ternaire conditionnel, perte extrapolee 18.0 HEURISTIC (a valider golden suite)",
            "quality additif calibre sur publisher-reported (uniform Q4/Q2 reproduits)",
            "channel proxy (RMS ligne) disponible mais non utilise pour le tri -> activation-aware reelle plus tard",
            "GDN state 2.42GB fixe (present quel que soit les poids) — le gain ne vient pas que des poids",
            "d2_score = loss - l_res*resident - l_tra*traffic (RAM/trafic en avant, pas min(poids) seul)",
        ],
    }


def load_context(tprofile_path=TENSOR_PROFILE):
    """Charge les sources et construit le contexte partage (payload, sensibilite,
    extra transport, tailles KV/GDN) utilise par main() et par les tests de
    non-regression."""
    with open(tprofile_path, encoding="utf-8") as fh:
        tprofile = json.load(fh)
    with open(STATIC, encoding="utf-8") as fh:
        static = json.load(fh)
    with open(os.path.join(HERE, "models", "Qwen3.8-27B-FP8", "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    tc = cfg.get("text_config", {})

    payload, forbidden_bytes, scales_bytes = load_payload_tensors(tprofile, static)
    sens, mean_rel = tensor_sensitivity(payload)

    p = {
        "ctx": 32768,
        "kv_heads": tc.get("num_key_value_heads"), "head_dim": tc.get("head_dim"),
        "n_full": sum(1 for t in tc.get("layer_types", []) if t == "full_attention"),
        "lin_k_heads": tc.get("linear_num_key_heads"), "lin_k_dim": tc.get("linear_key_head_dim"),
        "lin_v_heads": tc.get("linear_num_value_heads"), "lin_v_dim": tc.get("linear_value_head_dim"),
        "n_lin": sum(1 for t in tc.get("layer_types", []) if t == "linear_attention"),
        "hidden": tc.get("hidden_size"),
    }
    kv_bytes = 2 * p["kv_heads"] * p["head_dim"] * p["n_full"] * 2 * p["ctx"]
    state_bytes = p["lin_k_heads"] * p["lin_k_dim"] * p["lin_v_heads"] * p["lin_v_dim"] * p["n_lin"] * 4
    act_bytes = p["hidden"] * 2 * 3
    extra = {"kv": kv_bytes, "state": state_bytes, "act": act_bytes,
             "ws_rt": {"workspace_bytes": 512 * 10 ** 6, "runtime_bytes": 256 * 10 ** 6}, "ctx": p["ctx"]}

    return {
        "payload": payload, "forbidden_bytes": forbidden_bytes,
        "scales_bytes": scales_bytes, "sens": sens, "mean_rel": mean_rel,
        "extra": extra, "kv_bytes": kv_bytes, "state_bytes": state_bytes,
        "ctx": p["ctx"],
    }


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 TENSOR-LEVEL OPTIMIZER — 27B FP8")
    ap.add_argument("--json", default=OUT)
    ap.add_argument("--profile", choices=list(PROFILES) + ["all"], default="all")
    ap.add_argument("--machine", default="all", choices=["all"] + list(MACHINES))
    ap.add_argument("--tprofile", default=TENSOR_PROFILE)
    args = ap.parse_args()

    ctx = load_context(args.tprofile)
    payload = ctx["payload"]
    forbidden_bytes = ctx["forbidden_bytes"]
    scales_bytes = ctx["scales_bytes"]
    sens = ctx["sens"]
    mean_rel = ctx["mean_rel"]
    extra = ctx["extra"]
    kv_bytes = ctx["kv_bytes"]
    state_bytes = ctx["state_bytes"]

    machines = [args.machine] if args.machine != "all" else list(MACHINES)
    profiles = [args.profile] if args.profile != "all" else list(PROFILES)

    reports = [build_report(payload, forbidden_bytes, scales_bytes, sens, extra, machines, pr) for pr in profiles]
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen3.8-27B-FP8", "ctx": ctx["ctx"],
        "payload_tensors": len(payload),
        "payload_gb": round(sum(t["bytes"] for t in payload) / B, 2),
        "forbidden_gb": round(forbidden_bytes / B, 2), "scales_gb": round(scales_bytes / B, 5),
        "kv_gb": round(kv_bytes / B, 3), "gdn_state_gb": round(state_bytes / B, 3),
        "mean_rel_err": {k: round(v, 6) for k, v in mean_rel.items()},
        "uniform_loss_reference": UNIFORM_LOSS,
        "machine_profile_recommendation": MACHINE_PROFILE,
        "profiles": reports,
        "caveats": reports[0]["caveats"] if reports else [],
    }
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"[+] D2 TENSOR-LEVEL — payload {len(payload)} tensors / {report['payload_gb']:.1f} GB FP8")
    for r in reports:
        fit_ok = [n for n in machines if r["fit"].get(n, {}).get("fit") == "YES"]
        print(f"  [{r['profile']:11s}] weights {r['weights_gb']:5.2f} GB | resident {r['resident_gb']:5.2f} | "
              f"traffic {r['traffic_gb']:5.2f} | loss {r['loss']:5.2f} (gate {r['max_loss_gate']}) | "
              f"score {r['d2_score']:6.2f} | fits {','.join(fit_ok) if fit_ok else 'NONE'}")
        bp = r["by_precision_gb"]
        print(f"       prec: " + ", ".join(f"{k}={v}" for k, v in sorted(bp.items())))
        for name in machines:
            if r["fit"].get(name, {}).get("fit") == "YES":
                mm = r["fit"][name]
                print(f"       {name:9s} -> ~{mm['est_tps']} t/s mode {mm['mode']:9s} offload {mm['offload_gb']} GB")
    print(f"[+] -> {args.json}")


if __name__ == "__main__":
    main()