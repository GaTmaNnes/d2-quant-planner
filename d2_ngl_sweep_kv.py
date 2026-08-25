#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 KV SWEEP — turbo3/turbo4/q4_0 vs baseline f16 sur D2-ECO ngl 28→35.
========================================================================
Sweep croisé : 8 ngl × 4 modes KV = 32 configs, ~25 minutes.
Sauvegarde incrémentale dans d2_kv_sweep.json + rapport MD.
"""
import json, os, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_EXE = os.path.join(HERE, "llama-bench.exe")
MODEL = os.path.join(HERE, "models", "Qwen3.8-27B-D2-ECO.gguf")
OUT_JSON = os.path.join(HERE, "d2_kv_sweep.json")
OUT_MD = os.path.join(HERE, "d2_kv_sweep.md")

KV_MODES = [
    ("f16",     None,          "baseline (sans flag cache-type)"),
    ("turbo3",  "turbo3",      "TurboQuant 3 (Pascal)"),
    ("turbo4",  "turbo4",      "TurboQuant 4 + QJL (Blackwell)"),
    ("q4_0",    "q4_0",        "Quantisation uniforme q4_0"),
]
NGL_SWEET = [28, 29, 30, 31, 32, 33, 34, 35]
REPS = 3
TIMEOUT = 600


# ── SysSampler (identique à d2_ngl_sweep.py) ────────────────────────────
class SysSampler:
    def __init__(self, interval=0.5):
        self.interval = interval; self.stop = threading.Event()
        self.vram_max_mb = 0; self.gpu_util_max = 0
        self.ram_used_max_gb = 0.0; self.ram_percent_max = 0.0
        self.cpu_sys_max = 0.0; self.cpu_bench_max = 0.0; self.bench_rss_max_gb = 0.0
        self._thread = None
        try:
            import psutil; self._psutil = psutil
        except Exception:
            self._psutil = None

    def _bench_info(self):
        if not self._psutil: return 0.0, 0.0
        cm, rm = 0.0, 0.0
        try:
            for pr in self._psutil.process_iter(["name", "cpu_percent", "memory_info"]):
                try:
                    if "llama-bench" in (pr.info.get("name") or "").lower():
                        cm = max(cm, pr.info.get("cpu_percent") or 0)
                        mi = pr.info.get("memory_info")
                        if mi and getattr(mi, "rss", None):
                            rm = max(rm, mi.rss / (1024**3))
                except Exception: pass
        except Exception: pass
        return cm, rm

    def _sample_once(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3).stdout.strip().splitlines()
            if out:
                p = out[0].split(",")
                self.vram_max_mb = max(self.vram_max_mb, float(p[0].strip()))
                self.gpu_util_max = max(self.gpu_util_max, float(p[1].strip()))
        except Exception: pass
        if self._psutil:
            try:
                vm = self._psutil.virtual_memory()
                self.ram_used_max_gb = max(self.ram_used_max_gb, vm.used/(1024**3))
                self.ram_percent_max = max(self.ram_percent_max, vm.percent)
                self.cpu_sys_max = max(self.cpu_sys_max, self._psutil.cpu_percent(interval=None))
                cb, rs = self._bench_info()
                self.cpu_bench_max = max(self.cpu_bench_max, cb)
                self.bench_rss_max_gb = max(self.bench_rss_max_gb, rs)
            except Exception: pass

    def start(self):
        def loop():
            while not self.stop.is_set():
                self._sample_once(); time.sleep(self.interval)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def join(self):
        self.stop.set()
        if self._thread: self._thread.join(timeout=5)


# ── Parsing llama-bench JSON ────────────────────────────────────────────
def parse_bench_json(stdout, stderr):
    text = stdout
    try:
        data = json.loads(text)
    except Exception:
        s = text.find("["); e = text.rfind("]")
        if s < 0 or e < s: return None, (stderr or text)[-1500:]
        try: data = json.loads(text[s:e+1])
        except Exception: return None, (stderr or text)[-1500:]
    results = data.get("results", []) if isinstance(data, dict) else data
    pp = tg = None; chosen_ngl = None
    for r in results:
        if r.get("n_gpu_layers") is not None: chosen_ngl = r["n_gpu_layers"]
        if r.get("n_prompt") == 512 and r.get("n_gen") == 0: pp = r
        elif r.get("n_prompt") == 0 and r.get("n_gen") == 128: tg = r
    return {"pp512": pp, "tg128": tg, "chosen_ngl": chosen_ngl}, None


def tps_of(r):
    v = r.get("avg_ts"); v = v if v is not None else r.get("tokens_per_second")
    return round(v, 2) if v is not None else None


def stddev_of(r):
    v = r.get("stddev_ts"); v = v if v is not None else r.get("stddev")
    return round(v, 2) if v is not None else None


# ── Build & run ─────────────────────────────────────────────────────────
def build_cmd(path, ngl, kv_tag):
    cmd = [BENCH_EXE, "-m", path, "-p", "512", "-n", "128",
           "-r", str(REPS), "-fa", "on", "-o", "json", "-ngl", str(ngl)]
    if kv_tag:
        cmd += ["--cache-type-k", kv_tag, "--cache-type-v", kv_tag]
    return cmd


def run_one(path, ngl, kv_label, kv_tag, gpu_total):
    cmd = build_cmd(path, ngl, kv_tag)
    entry = {"model": os.path.basename(path), "ngl": ngl, "kv": kv_label,
             "cmd": " ".join(cmd)}
    sampler = SysSampler(0.5); sampler.start(); t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=TIMEOUT)
        wall = time.time() - t0
    except subprocess.TimeoutExpired:
        sampler.join(); entry["error"] = f"timeout ({TIMEOUT}s)"; return entry
    except FileNotFoundError:
        sampler.join(); entry["error"] = f"{BENCH_EXE} introuvable"; return entry
    finally:
        sampler.join()

    entry["wall_s"] = round(wall, 1)
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
        entry["tail"] = err[-1200:]; return entry
    pp, tg = data["pp512"], data["tg128"]
    entry["chosen_ngl"] = data["chosen_ngl"]
    if pp: entry["pp512_tps"] = tps_of(pp)
    if tg:
        entry["tg128_tps"] = tps_of(tg)
        entry["tg128_stddev"] = stddev_of(tg)
    return entry


# ── Rapport MD ──────────────────────────────────────────────────────────
def render_md(report, baseline_map):
    """Tableau croisé ngl × KV, avec un bloc par métrique."""
    metrics = [
        ("tg128_tps",    "tg128 (t/s)",     "t/s"),
        ("pp512_tps",    "pp512 (t/s)",     "t/s"),
        ("vram_max_mb",  "VRAM max (MiB)",  "MiB"),
        ("tg128_stddev", "tg128 stddev",    "σ"),
    ]
    kv_labels = [m[0] for m in KV_MODES]
    lines = [
        f"# D2 KV SWEEP — turbo3 / turbo4 / q4_0 vs f16 sur D2-ECO",
        f"",
        f"- Généré : {report['generated_at']}",
        f"- Modèle  : Qwen3.8-27B-D2-ECO (11.64 GB), -r {REPS}",
        f"- GPU     : {report['gpu_total_mb']:.0f} MiB, FA on",
        f"- Modes   : f16 (baseline), turbo3, turbo4, q4_0",
        f"",
    ]
    for key, title, unit in metrics:
        lines.append(f"## {title}")
        lines.append("")
        header = "| ngl | " + " | ".join(kv_labels) + " |"
        sep = "|---|" + "|".join(["---"] * len(kv_labels)) + "|"
        lines.append(header)
        lines.append(sep)
        for ngl in NGL_SWEET:
            vals = []
            for kvl in kv_labels:
                r = report["results"].get((ngl, kvl))
                if r is None or "error" in r:
                    vals.append("—")
                else:
                    v = r.get(key)
                    if v is None:
                        vals.append("—")
                    elif key == "vram_max_mb":
                        vals.append(f"{v:.0f}")
                    elif key == "tg128_stddev":
                        vals.append(f"{v:.2f}")
                    else:
                        vals.append(f"{v:.2f}")
            lines.append("| " + str(ngl) + " | " + " | ".join(vals) + " |")
        lines.append("")

    # Bloc comparaison vs baseline (%)
    lines.append("## Δ vs f16 baseline (%)")
    lines.append("")
    for met_key, met_title, _ in [("tg128_tps", "tg128", "t/s"), ("pp512_tps", "pp512", "t/s")]:
        lines.append(f"### {met_title} — perte vs f16")
        lines.append("")
        header2 = "| ngl | " + " | ".join(kv_labels[1:]) + " |"
        sep2 = "|---|" + "|".join(["---"] * (len(kv_labels)-1)) + "|"
        lines.append(header2); lines.append(sep2)
        for ngl in NGL_SWEET:
            base = report["results"].get((ngl, "f16"))
            if base is None or "error" in base:
                continue
            base_v = base.get(met_key)
            if base_v is None: continue
            vals2 = []
            for kvl in kv_labels[1:]:
                r = report["results"].get((ngl, kvl))
                if r is None or "error" in r:
                    vals2.append("—"); continue
                v = r.get(met_key)
                if v is None: vals2.append("—")
                else: vals2.append(f"{(v-base_v)/base_v*100:+.1f}%")
            lines.append("| " + str(ngl) + " | " + " | ".join(vals2) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ── Main ────────────────────────────────────────────────────────────────
def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

    if not os.path.isfile(BENCH_EXE):
        sys.exit(f"[!] {BENCH_EXE} introuvable.")
    if not os.path.isfile(MODEL):
        sys.exit(f"[!] Modèle introuvable: {MODEL}")

    gpu_total = 0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
            "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
        gpu_total = float(out.strip().splitlines()[0])
    except Exception: pass

    # Charger les résultats existants (reprise si interrompu)
    results = {}
    if os.path.isfile(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            saved = json.load(f)
            for k, v in saved.get("results", {}).items():
                # k = "ngl|kv", e.g. "30|turbo4"
                parts = k.split("|")
                results[(int(parts[0]), parts[1])] = v
        print(f"[i] Reprise: {len(results)} configs déjà faites\n")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_total_mb": gpu_total,
        "results": {},
    }

    total = len(NGL_SWEET) * len(KV_MODES)
    done = 0
    baseline_map = {}  # ngl -> baseline entry

    for kv_label, kv_tag, kv_desc in KV_MODES:
        for ngl in NGL_SWEET:
            done += 1
            key = (ngl, kv_label)
            if key in results:
                print(f"[{done}/{total}] SKIP ngl={ngl} kv={kv_label} (déjà fait)")
                if kv_label == "f16" and "error" not in results[key]:
                    baseline_map[ngl] = results[key]
                continue

            print(f"[{done}/{total}] ngl={ngl} kv={kv_label} ({kv_desc}) ...", flush=True)
            r = run_one(MODEL, ngl, kv_label, kv_tag, gpu_total)
            results[key] = r

            # Sauvegarde incrémentale
            report["results"] = {f"{n}|{k}": v for (n, k), v in results.items()}
            report["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            if kv_label == "f16" and "error" not in r:
                baseline_map[ngl] = r

            if "error" in r:
                print(f"    [!] ÉCHEC : {r['error']}")
            else:
                print(f"    tg128={r['tg128_tps']} t/s · pp512={r['pp512_tps']} t/s · "
                      f"VRAM={r['vram_max_mb']:.0f}/{gpu_total:.0f} MiB · "
                      f"GPU={r['gpu_util_max_pct']:.0f}% · wall={r['wall_s']:.0f}s", flush=True)

    # Rapport final
    report["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["results"] = {f"{n}|{k}": v for (n, k), v in results.items()}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = render_md(report, baseline_map)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[+] -> {OUT_JSON}")
    print(f"[+] -> {OUT_MD}")
    print()
    print(md)


if __name__ == "__main__":
    main()