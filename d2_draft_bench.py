#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 DRAFT BENCH — VRAM reelle, scan ngl et taux d'acceptation avec un draft model.
====================================================================================
Comble 3 des axes identifies manquants d'un coup (ils partagent le meme besoin :
lancer un serveur target+draft et mesurer) :
  - "VRAM reelle avec draft"     : placer target + draft + KV dans le budget VRAM
  - "ngl optimal avec draft"     : scanner -ngl du target avec le draft actif
  - "Acceptance rate MTP/draft"  : mesurer le taux d'acceptation reel par config

N'invente aucun chiffre : la VRAM vient de nvidia-smi (avant/apres generation),
le taux d'acceptation vient de `timings.draft_n`/`timings.draft_n_accepted` que
llama-server renvoie nativement dans la reponse JSON de /completion (verifie
dans tools/server/server-context.cpp — pas une estimation).

Exemples :
  python d2_draft_bench.py --target models/Qwen3.8-27B-Q4_K_M.gguf ^
                            --draft models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf ^
                            --ngl-list 20,40,60,99
  python d2_draft_bench.py --target models/target.gguf --draft models/draft.gguf --spec-type draft-mtp
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_EXE = os.path.join(HERE, "llama-server.exe")
OUT_JSON = os.path.join(HERE, "d2_draft_bench_report.json")
DEFAULT_PROMPT = "Explique en trois phrases ce qu'est la quantification d'un modèle de langage."


def query_vram_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        return float(out[0]) if out else None
    except Exception:
        return None


def wait_ready(port, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1.5) as r:
                data = json.loads(r.read().decode())
                if data.get("data") or data.get("models"):
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def wait_port_free(port, timeout=30):
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return True
        time.sleep(0.5)
    return False


def run_one_config(target, draft, spec_type, ngl, ngld, port, prompt, n_predict, ctx, timeout):
    args = [
        "-m", target, "--model-draft", draft, "--spec-type", spec_type,
        "--spec-draft-ngl", str(ngld), "-ngl", str(ngl),
        "--ctx-size", str(ctx), "--batch-size", "512",
        "--host", "127.0.0.1", "--port", str(port), "--flash-attn", "on",
    ]
    print(f"    $ llama-server -ngl {ngl} --spec-type {spec_type} ...")
    env = dict(os.environ)
    env["PATH"] = HERE + os.pathsep + env.get("PATH", "")
    proc = subprocess.Popen([SERVER_EXE] + args, cwd=HERE, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
    result = {"ngl": ngl, "spec_type": spec_type}
    try:
        if not wait_ready(port, timeout=timeout):
            proc.terminate()
            tail = proc.stdout.read(4000) if proc.stdout else ""
            result["error"] = "serveur non prêt dans le délai imparti"
            result["log_tail"] = tail
            return result

        vram_loaded_mb = query_vram_mb()
        result["vram_loaded_mb"] = vram_loaded_mb

        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/completion",
                data=json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode())
        except urllib.error.URLError as e:
            result["error"] = f"requête /completion échouée : {e}"
            return result
        wall_s = time.perf_counter() - t0

        vram_after_gen_mb = query_vram_mb()
        result["vram_after_gen_mb"] = vram_after_gen_mb

        timings = resp.get("timings", {}) or {}
        draft_n = timings.get("draft_n")
        draft_n_accepted = timings.get("draft_n_accepted")
        result.update({
            "predicted_n": timings.get("predicted_n"),
            "predicted_per_second": timings.get("predicted_per_second"),
            "wall_s": round(wall_s, 3),
            "draft_n": draft_n,
            "draft_n_accepted": draft_n_accepted,
            "draft_acceptance_pct": round(100.0 * draft_n_accepted / draft_n, 2)
                                     if draft_n else None,
        })
        return result
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        wait_port_free(port)


def main():
    ap = argparse.ArgumentParser(description="D2 Draft Bench — VRAM/ngl/acceptance avec speculative decoding")
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--spec-type", default="dflash",
                     help="none,draft-simple,draft-eagle3,draft-mtp,dflash,... (défaut: dflash)")
    ap.add_argument("--ngl-list", default="99", help="valeurs -ngl du target séparées par des virgules")
    ap.add_argument("--ngld", default="auto", help="-ngl du draft (défaut: auto)")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--server-timeout", type=int, default=180, help="secondes max de chargement serveur")
    ap.add_argument("--json", default=OUT_JSON)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not os.path.isfile(SERVER_EXE):
        sys.exit(f"[!] {SERVER_EXE} introuvable.")
    if not os.path.isfile(args.target):
        sys.exit(f"[!] Target introuvable : {args.target}")
    if not os.path.isfile(args.draft):
        sys.exit(f"[!] Draft introuvable : {args.draft}")

    try:
        ngl_values = [v.strip() for v in args.ngl_list.split(",") if v.strip()]
    except Exception:
        sys.exit("[!] --ngl-list invalide.")

    results = []
    for i, ngl in enumerate(ngl_values, 1):
        print(f"[{i}/{len(ngl_values)}] -ngl={ngl}")
        r = run_one_config(args.target, args.draft, args.spec_type, ngl, args.ngld,
                            args.port, args.prompt, args.n_predict, args.ctx, args.server_timeout)
        results.append(r)
        if "error" in r:
            print(f"    [!] {r['error']}")
        else:
            print(f"    VRAM chargé={r['vram_loaded_mb']:.0f} Mo  après-gen={r['vram_after_gen_mb']:.0f} Mo  "
                  f"tok/s={r['predicted_per_second']:.2f}  "
                  f"acceptance={r['draft_acceptance_pct']}%  "
                  f"({r['draft_n_accepted']}/{r['draft_n']})" if r.get("draft_n") else
                  f"    VRAM chargé={r['vram_loaded_mb']:.0f} Mo  tok/s={r.get('predicted_per_second')}  "
                  f"(pas de stats draft — spéculatif inactif ou non déclenché)")

    print("\n" + "=" * 100)
    print(f"  D2 DRAFT BENCH — target={os.path.basename(args.target)}  draft={os.path.basename(args.draft)}  "
          f"spec-type={args.spec_type}")
    print("=" * 100)
    print(f"  {'ngl':>6} {'VRAM chargé':>12} {'VRAM +gen':>12} {'tok/s':>8} {'accept %':>10} {'accepté/total':>15}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"  {str(r['ngl']):>6}  ÉCHEC : {r['error']}")
            continue
        print(f"  {str(r['ngl']):>6} "
              f"{r['vram_loaded_mb']:>10.0f} Mo "
              f"{r['vram_after_gen_mb']:>10.0f} Mo "
              f"{(r['predicted_per_second'] or 0):>8.2f} "
              f"{(r['draft_acceptance_pct'] if r['draft_acceptance_pct'] is not None else '—'):>10} "
              f"{(str(r['draft_n_accepted']) + '/' + str(r['draft_n'])) if r.get('draft_n') else '—':>15}")
    print("=" * 100)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({"target": args.target, "draft": args.draft, "spec_type": args.spec_type,
                    "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] -> {args.json}")


if __name__ == "__main__":
    main()
