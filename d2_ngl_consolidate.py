#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consolide tous les JSON de sweep ngl (v1, fine, 24-27, 11-32, q4km) en un
rapport unique d2_ngl_sweep_final.md + d2_ngl_sweep_final.json."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [
    "d2_ngl_sweep_report.json",        # v1 : D2-ECO 33/40/48/56/64/99 + Q4_K_M 33/40
    "d2_ngl_sweep_fine_d2eco.json",    # fine : D2-ECO 2/20/22/24/26/28
    "d2_ngl_sweep_24_27.json",         # D2-ECO + Q4_K_M 24/27
    "d2_ngl_sweep_11_32.json",         # D2-ECO 11-40 + fit
    "d2_ngl_sweep_q4km_11_32.json",    # Q4_K_M 14-99 + fit (complet)
]

MODELS = {"D2-ECO", "Q4_K_M"}


def load(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("results", [])


def normalize(r):
    """Normalise v1 (clé 'ngl' int) et v2 (clé 'config' str) vers un format
    commun avec 'config' (str)."""
    r = dict(r)
    if "config" not in r and "ngl" in r:
        r["config"] = str(r["ngl"])
    elif "config" not in r:
        r["config"] = "?"
    return r


def cfg_sort_key(cfg):
    """Tri : les entiers d'abord (numérique), puis 'fit:...' (alpha)."""
    if cfg.isdigit():
        return (0, int(cfg))
    return (1, cfg)


def main():
    results = {}
    for f in FILES:
        for r in load(f):
            if "error" in r or r.get("tg128_tps") is None:
                continue
            r = normalize(r)
            results[(r["model"], r["config"])] = r
    print(f"total configs uniques: {len(results)}")

    lines = ["# SWEEP NGL FINAL — llama-bench pp512/tg128 (-r 3, KV f16, FA on)",
             "",
             "| Modèle | config | tg128 (t/s) | pp512 (t/s) | VRAM max (MiB) | GPU % | CPU llama % | RAM sys (Go) | wall (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m_tag, m_label in [("D2-ECO", "D2-ECO"), ("Q4_K_M", "Q4_K_M")]:
        rows = [r for r in results.values() if m_tag in (r.get("model") or "")]
        rows.sort(key=lambda r: cfg_sort_key(r["config"]))
        for r in rows:
            tg = r.get("tg128_tps")
            pp = r.get("pp512_tps")
            vram = r.get("vram_max_mb")
            gpu = r.get("gpu_util_max_pct")
            cpu = r.get("cpu_bench_max_pct")
            ram = r.get("ram_used_max_gb")
            wall = r.get("wall_s")
            lines.append(f"| {m_label} | {r['config']} | "
                         f"{tg if tg is not None else '—'} | "
                         f"{pp if pp is not None else '—'} | "
                         f"{vram if vram is not None else '—'} | "
                         f"{gpu if gpu is not None else '—'} | "
                         f"{cpu if cpu is not None else '—'} | "
                         f"{ram if ram is not None else '—'} | "
                         f"{wall if wall is not None else '—'} |")
    out_md = os.path.join(HERE, "d2_ngl_sweep_final.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out_json = os.path.join(HERE, "d2_ngl_sweep_final.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({f"{m}|{c}": v for (m, c), v in results.items()}, f, indent=2, ensure_ascii=False)
    print(f"[+] -> {out_md}")
    print(f"[+] -> {out_json}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
