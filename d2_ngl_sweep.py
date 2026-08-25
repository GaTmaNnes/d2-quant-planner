#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 NGL SWEEP v2 — balayage -ngl fin + fit auto, avec VRAM/RAM/GPU%/CPU%.
=========================================================================
Mesure, pour chaque modèle x chaque config :
  - pp512 / tg128  (tokens/s, llama-bench -r 3, sortie JSON officielle)
  - VRAM max       (nvidia-smi échantillonné pendant tout le run)
  - GPU util max   (% nvidia-smi)
  - CPU % max      (psutil : système + process llama-bench)
  - RAM max        (psutil, système)
  - ngl réellement retenu par llama.cpp (utile pour les configs fit)

Configs supportées dans --ngl :
  - entiers : 33, 40, ...
  - "fit:<marge>" : placement automatique --fit-target <marge> MiB
                    (équivalent du fit=on des docs llama.cpp ; `-ngl auto`
                     n'existe PAS dans ce build — vérifié 22/08/2026)

Le rapport JSON est réécrit après CHAQUE config : une coupure ne perd rien.
Le GPU doit être libre (aucun autre process llama) sinon mesures polluées.

Exemples :
  python d2_ngl_sweep.py --ngl 20,24,28,30,32,33,34,36,38,40,fit:0,fit:200
  python d2_ngl_sweep.py --model models/Qwen3.8-27B-D2-ECO.gguf --ngl 33,34,35
  python d2_ngl_sweep.py --reps 1 --ngl 33
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_EXE = os.path.join(HERE, "llama-bench.exe")
OUT_JSON = os.path.join(HERE, "d2_ngl_sweep_report.json")
OUT_MD = os.path.join(HERE, "d2_ngl_sweep_report.md")

DEFAULT_MODELS = [
    ("D2-ECO", os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO.gguf")),
    ("Q4_K_M", os.path.join(HERE, "models", "Qwen3.8-27B-Q4_K_M.gguf")),
]
# Grille fine autour du sweet spot (33) + tests fit auto (D2-ECO)
DEFAULT_CONFIGS = ["2", "20", "22", "24", "26", "28", "30", "31", "32",
                   "33", "34", "35", "36", "38", "40", "fit:0", "fit:200"]
# Sous-ensemble pour Q4_K_M (v1 couvre déjà 33/40/48/56/64/99)
Q4KM_CONFIGS = ["2", "24", "28", "30", "32", "34", "fit:0", "fit:200"]


class SysSampler:
    """Échantillonne VRAM / util GPU / RAM / CPU pendant un run (thread)."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.stop = threading.Event()
        self.vram_max_mb = 0
        self.gpu_util_max = 0
        self.ram_used_max_gb = 0.0
        self.ram_percent_max = 0.0
        self.cpu_sys_max = 0.0
        self.cpu_bench_max = 0.0
        self.bench_rss_max_gb = 0.0
        self._thread = None
        try:
            import psutil  # noqa: F401
            self._psutil = psutil
        except Exception:
            self._psutil = None

    def _bench_info(self):
        """Retourne (cpu_max, rss_max_gb) des processus llama-bench actifs."""
        if not self._psutil:
            return 0.0, 0.0
        cpu_mx = 0.0
        rss_mx = 0.0
        try:
            for pr in self._psutil.process_iter(["name", "cpu_percent", "memory_info"]):
                try:
                    if "llama-bench" in (pr.info.get("name") or "").lower():
                        cpu_mx = max(cpu_mx, pr.info.get("cpu_percent") or 0)
                        mi = pr.info.get("memory_info")
                        if mi and getattr(mi, "rss", None):
                            rss_mx = max(rss_mx, mi.rss / (1024 ** 3))
                except Exception:
                    pass
        except Exception:
            pass
        return cpu_mx, rss_mx

    def _sample_once(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip().splitlines()
            if out:
                parts = out[0].split(",")
                vram = float(parts[0].strip())
                util = float(parts[1].strip())
                self.vram_max_mb = max(self.vram_max_mb, vram)
                self.gpu_util_max = max(self.gpu_util_max, util)
        except Exception:
            pass
        if self._psutil:
            try:
                vm = self._psutil.virtual_memory()
                self.ram_used_max_gb = max(self.ram_used_max_gb,
                                           vm.used / (1024 ** 3))
                self.ram_percent_max = max(self.ram_percent_max, vm.percent)
                self.cpu_sys_max = max(self.cpu_sys_max,
                                       self._psutil.cpu_percent(interval=None))
                cbm, rss = self._bench_info()
                self.cpu_bench_max = max(self.cpu_bench_max, cbm)
                self.bench_rss_max_gb = max(self.bench_rss_max_gb, rss)
            except Exception:
                pass

    def start(self):
        def loop():
            while not self.stop.is_set():
                self._sample_once()
                time.sleep(self.interval)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def join(self):
        self.stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def parse_bench_json(stdout, stderr):
    """Extrait les résultats tg128/pp512 du JSON de llama-bench (-o json).

    Ce build (beellama.cpp) émet un **tableau brut** d'objets (pas un objet
    avec clé `results`) et nomme la vitesse `avg_ts`/`stddev_ts` (pas
    `tokens_per_second`). On gère les deux formes par robustesse.
    """
    text = stdout
    try:
        data = json.loads(text)
    except Exception:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end < start:
            return None, (stderr or text)[-1500:]
        try:
            data = json.loads(text[start:end + 1])
        except Exception:
            return None, (stderr or text)[-1500:]

    if isinstance(data, dict):
        results = data.get("results", [])
    elif isinstance(data, list):
        results = data
    else:
        return None, (stderr or text)[-1500:]

    pp = tg = None
    model_size = 0
    chosen_ngl = None
    for r in results:
        model_size = max(model_size, r.get("model_size") or 0)
        if r.get("n_gpu_layers") is not None:
            chosen_ngl = r.get("n_gpu_layers")
        if r.get("n_prompt") == 512 and r.get("n_gen") == 0:
            pp = r
        elif r.get("n_prompt") == 0 and r.get("n_gen") == 128:
            tg = r
    return {"pp512": pp, "tg128": tg, "model_size": model_size,
            "chosen_ngl": chosen_ngl}, None


def tps_of(r):
    v = r.get("avg_ts")
    if v is None:
        v = r.get("tokens_per_second")
    return round(v, 2) if v is not None else None


def stddev_of(r):
    v = r.get("stddev_ts")
    if v is None:
        v = r.get("stddev")
    return round(v, 2) if v is not None else None


def build_cmd(model, config, reps):
    """Construit la ligne de commande llama-bench pour une config."""
    cmd = [BENCH_EXE, "-m", model, "-p", "512", "-n", "128",
           "-r", str(reps), "-fa", "on", "-o", "json"]
    if config.startswith("fit:"):
        margin = config.split(":", 1)[1]
        cmd += ["-fitt", margin]          # fit auto (pas de -ngl)
    else:
        cmd += ["-ngl", str(config)]
    return cmd


def run_one_config(model, config, reps, timeout):
    cmd = build_cmd(model, config, reps)
    entry = {"model": os.path.basename(model), "config": config,
             "cmd": " ".join(cmd)}

    sampler = SysSampler(interval=0.5)
    sampler.start()
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout)
        wall_s = time.time() - t0
    except subprocess.TimeoutExpired:
        sampler.join()
        entry["error"] = f"timeout ({timeout}s)"
        return entry
    except FileNotFoundError:
        sampler.join()
        entry["error"] = f"{BENCH_EXE} introuvable"
        return entry
    finally:
        sampler.join()

    entry["wall_s"] = round(wall_s, 1)
    entry["vram_max_mb"] = round(sampler.vram_max_mb, 0)
    entry["gpu_util_max_pct"] = round(sampler.gpu_util_max, 0)
    entry["ram_used_max_gb"] = round(sampler.ram_used_max_gb, 2)
    entry["ram_percent_max"] = round(sampler.ram_percent_max, 1)
    entry["cpu_sys_max_pct"] = round(sampler.cpu_sys_max, 0)
    entry["cpu_bench_max_pct"] = round(sampler.cpu_bench_max, 0)
    entry["bench_rss_max_gb"] = round(sampler.bench_rss_max_gb, 2)

    data, err = parse_bench_json(p.stdout, p.stderr)
    if err:
        entry["error"] = f"exit={p.returncode}, sortie non parsée"
        entry["tail"] = err[-1200:]
        return entry

    pp, tg = data["pp512"], data["tg128"]
    entry["model_size_gb"] = round((data.get("model_size") or 0) / (1024 ** 3), 2)
    entry["chosen_ngl"] = data["chosen_ngl"]
    if pp:
        entry["pp512_tps"] = tps_of(pp)
    if tg:
        entry["tg128_tps"] = tps_of(tg)
        entry["tg128_stddev"] = stddev_of(tg)
    return entry


def save_report(path, report):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def render_md(report):
    lines = ["# D2 NGL SWEEP v2 — llama-bench pp512/tg128 (-r %d)"
             % report["reps"], ""]
    lines.append(f"- Généré : {report['generated_at']}")
    lines.append("- KV cache : f16 (défaut llama-bench) · flash-attn : on")
    lines.append(f"- VRAM GPU : {report['gpu_total_mb']:.0f} MiB")
    lines.append("- `-ngl auto` NON supporté par ce build → fit via `-fitt`")
    lines.append("")
    for model in report["models"]:
        res = [r for r in report["results"] if r["model"] == model and "error" not in r]
        fail = [r for r in report["results"] if r["model"] == model and "error" in r]
        base = next((r for r in res if r["config"] == "33"), None)
        base_tg = base["tg128_tps"] if base and base.get("tg128_tps") else None
        base_vram = base["vram_max_mb"] if base else None
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| config | ngl réel | tg128 | pp512 | VRAM max | Δtg vs 33 | ΔVRAM | t/s per GiB VRAM | RAM sys | RAM proc | GPU % | CPU sys | CPU llama | wall |")
        lines.append("|--------|----------|-------|-------|----------|-----------|-------|-----------------|---------|----------|-------|---------|-----------|------|")
        for r in res:
            tg = r.get("tg128_tps")
            vram = r.get("vram_max_mb")
            dtg = f"{(tg - base_tg):+.2f}" if tg is not None and base_tg else "—"
            dvram = f"{(vram - base_vram):+.0f}" if vram is not None and base_vram else "—"
            eff = f"{tg / (vram / 1024):.2f}" if tg is not None and vram else "—"
            lines.append(f"| {r['config']:>7} | {str(r.get('chosen_ngl') or '—'):>7} | "
                         f"{tg or 0:>6.2f} | {r.get('pp512_tps') or 0:>6.0f} | "
                         f"{vram or 0:>8.0f} | {dtg:>8} | {dvram:>6} | {eff:>15} | "
                         f"{r.get('ram_used_max_gb') or 0:>6.2f} | {r.get('bench_rss_max_gb') or 0:>7.2f} | "
                         f"{r.get('gpu_util_max_pct') or 0:>5.0f} | {r.get('cpu_sys_max_pct') or 0:>5.0f} | "
                         f"{r.get('cpu_bench_max_pct') or 0:>9.0f} | {r.get('wall_s') or 0:>5.0f} |")
        for r in fail:
            lines.append(f"| {r['config']:>7} | **ÉCHEC** : {r.get('error')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="D2 NGL Sweep v2 — ngl fin/fit auto + VRAM/RAM/GPU/CPU")
    ap.add_argument("--model", action="append", default=[], help="chemin GGUF (répétable)")
    ap.add_argument("--name", action="append", default=[], help="nom court (aligné avec --model)")
    ap.add_argument("--ngl", default=",".join(DEFAULT_CONFIGS),
                    help="configs : entiers et/ou fit:<marge>")
    ap.add_argument("--reps", type=int, default=3, help="répétitions llama-bench (défaut 3)")
    ap.add_argument("--timeout", type=int, default=1800, help="secondes max par config")
    ap.add_argument("--json", default=OUT_JSON)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not os.path.isfile(BENCH_EXE):
        sys.exit(f"[!] {BENCH_EXE} introuvable.")

    models = list(args.model) or [p for _, p in DEFAULT_MODELS]
    names = list(args.name)
    if names and len(names) != len(models):
        sys.exit("[!] --name doit avoir le même nombre d'entrées que --model")
    if not names:
        names = [os.path.basename(m).replace(".gguf", "") for m in models]

    configs = [c.strip() for c in args.ngl.split(",") if c.strip()]
    for c in configs:
        if not (c.isdigit() or c.startswith("fit:")):
            sys.exit(f"[!] Config invalide : '{c}' (entier ou fit:<marge>)")
    if not configs:
        sys.exit("[!] Aucune config (--ngl)")

    for m in models:
        if not os.path.isfile(m):
            sys.exit(f"[!] Modèle introuvable : {m}")

    gpu_total = 0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        gpu_total = float(out.strip().splitlines()[0])
    except Exception:
        pass

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reps": args.reps,
        "gpu_total_mb": gpu_total,
        "models": names,
        "configs": configs,
        "results": [],
    }
    save_report(args.json, report)

    total = len(names) * len(configs)
    done = 0
    for name, model in zip(names, models):
        for cfg in configs:
            done += 1
            print(f"[{done}/{total}] {name} config={cfg} ...", flush=True)
            r = run_one_config(model, cfg, args.reps, args.timeout)
            report["results"].append(r)
            save_report(args.json, report)  # incrémental : rien n'est perdu
            if "error" in r:
                print(f"    [!] ÉCHEC : {r['error']}")
            else:
                print(f"    tg128={r['tg128_tps']} t/s · pp512={r['pp512_tps']} t/s · "
                      f"ngl={r.get('chosen_ngl')} · VRAM={r['vram_max_mb']:.0f}/{gpu_total:.0f} MiB · "
                      f"GPU={r['gpu_util_max_pct']:.0f}% · CPU={r['cpu_bench_max_pct']:.0f}% · "
                      f"RAM={r['ram_used_max_gb']:.2f} Go ({r['wall_s']:.0f}s)", flush=True)

    print()
    md = render_md(report)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"[+] -> {args.json}")
    print(f"[+] -> {OUT_MD}")


if __name__ == "__main__":
    main()
