#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 FREETOKEN BENCH — tokens/s et VRAM reels d'un serveur FreeToken (WSL2),
comparables au format de rapport de d2_draft_bench.py / d2_ppl_sweep.py.

N'invente aucun chiffre : lance `ft serve` dans WSL2 Ubuntu (le seul OS que
FreeToken supporte - voir "d2 v3/FREETOKEN_INTEGRATION.md" §1), attend qu'il
soit pret via /v1/models, mesure le tok/s reel depuis la reponse
/v1/chat/completions (champ usage), et la VRAM via nvidia-smi (Windows,
meme GPU physique que WSL2 -> lecture valide).

Rappel important (voir FREETOKEN_INTEGRATION.md §3, mesure reelle du
25/08/2026 sur ce GPU) : le checkpoint FP8 deja present
(hf_weights_35b_fp8/) n'a PAS de noyau CPU-MoE -> --moe-backend hybrid est
indisponible pour ce fichier precis (echouera ou retombera sur offload cote
FreeToken). Seul "offload" est utilisable avec ce checkpoint. "hybrid" ne
redevient utilisable qu'avec un checkpoint bf16 ou nvfp4 (ex :
nvidia/Qwen3.6-35B-A3B-NVFP4, pas telecharge a ce jour).

Exemples :
  python d2_freetoken_bench.py --moe-backend offload
  python d2_freetoken_bench.py --model /mnt/c/Users/videl/Desktop/other-ckpt --moe-backend offload,hybrid
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
OUT_JSON = os.path.join(HERE, "d2_freetoken_bench_report.json")
DEFAULT_PROMPT = "Explique en trois phrases ce qu'est la quantification d'un modèle de langage."

WSL_DISTRO = "Ubuntu"
WSL_FREETOKEN_DIR = "~/FreeToken"
# Checkpoint deja present sur disque (FP8 -> offload seulement, voir docstring).
DEFAULT_WINDOWS_MODEL = r"C:\Users\videl\Desktop\lama 1080-5070\hf_weights_35b_fp8"


def win_to_wsl_path(path):
    """Traduit un chemin Windows (C:\\...) en chemin WSL (/mnt/c/...)."""
    path = os.path.abspath(path)
    drive, rest = os.path.splitdrive(path)
    if not drive:
        return path.replace("\\", "/")
    drive_letter = drive.rstrip(":").lower()
    rest = rest.replace("\\", "/").lstrip("/")
    return f"/mnt/{drive_letter}/{rest}"


def query_vram_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        return float(out[0]) if out else None
    except Exception:
        return None


def wait_ready(port, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1.5) as r:
                data = json.loads(r.read().decode())
                if data.get("data"):
                    return True
        except Exception:
            pass
        time.sleep(2)
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


def wait_generative(port, timeout):
    """Attend que le serveur accepte VRAIMENT une génération (pas seulement /v1/models).
    FreeToken répond 503 tant que les experts ne sont pas tous chargés en RAM."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({
                    "model": "default",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1, "temperature": 0.0,
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def run_one_config(model_path_wsl, backend, port, prompt, n_predict, load_timeout, gen_timeout):
    serve_cmd = (
        f"cd {WSL_FREETOKEN_DIR} && source .venv/bin/activate && "
        f"exec ft serve --model-path '{model_path_wsl}' --moe-backend {backend} "
        f"--host 0.0.0.0 --port {port}"
    )
    print(f"    $ wsl -d {WSL_DISTRO} -e bash -lc \"{serve_cmd}\"")
    proc = subprocess.Popen(
        ["wsl.exe", "-d", WSL_DISTRO, "-e", "bash", "-lc", serve_cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    result = {"backend": backend}
    try:
        if not wait_ready(port, timeout=load_timeout):
            proc.terminate()
            tail = proc.stdout.read(4000) if proc.stdout else ""
            result["error"] = "serveur FreeToken non prêt dans le délai imparti"
            result["log_tail"] = tail
            return result

        # [CORRIGÉ 25/08] /v1/models répond pendant le chargement des experts ;
        # attendre qu'une génération minimale passe (sinon HTTP 503 sur la vraie mesure).
        if not wait_generative(port, timeout=load_timeout):
            proc.terminate()
            tail = proc.stdout.read(4000) if proc.stdout else ""
            result["error"] = "FreeToken prêt sur /v1/models mais incapable de générer (503 persistant)"
            result["log_tail"] = tail
            return result

        vram_loaded_mb = query_vram_mb()
        result["vram_loaded_mb"] = vram_loaded_mb

        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": n_predict,
                    "temperature": 0.0,
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=gen_timeout) as r:
                resp = json.loads(r.read().decode())
        except urllib.error.URLError as e:
            result["error"] = f"requête /v1/chat/completions échouée : {e}"
            return result
        wall_s = time.perf_counter() - t0

        vram_after_gen_mb = query_vram_mb()
        result["vram_after_gen_mb"] = vram_after_gen_mb

        usage = resp.get("usage", {}) or {}
        completion_tokens = usage.get("completion_tokens")
        result.update({
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion_tokens,
            "wall_s": round(wall_s, 3),
            "tokens_per_second": round(completion_tokens / wall_s, 2)
                                  if completion_tokens and wall_s > 0 else None,
        })
        return result
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        # ft serve tourne dans WSL, pas dans ce process Windows : un terminate()
        # sur wsl.exe ne garantit pas la mort du process serveur cote Linux.
        # Filet de securite explicite, avec tolerance aux timeouts WSL.
        try:
            subprocess.run(["wsl.exe", "-d", WSL_DISTRO, "-e", "pkill", "-f", "ft serve"],
                            capture_output=True, timeout=30)
        except Exception:
            pass
        try:
            wait_port_free(port, timeout=45)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="D2 FreeToken Bench — tok/s et VRAM réels via ft serve (WSL2)")
    ap.add_argument("--model", default=DEFAULT_WINDOWS_MODEL,
                     help=r"Chemin Windows du checkpoint HF safetensors (défaut : hf_weights_35b_fp8)")
    ap.add_argument("--moe-backend", default="offload",
                     help="backends à comparer, séparés par des virgules : offload,hybrid,cpu,fused,auto "
                          "(hybrid indisponible pour un checkpoint FP8 — voir docstring)")
    ap.add_argument("--port", type=int, default=1919)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--load-timeout", type=int, default=600,
                     help="secondes max pour le chargement d'un MoE 35B (défaut 10 min)")
    ap.add_argument("--gen-timeout", type=int, default=300)
    ap.add_argument("--json", default=OUT_JSON)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not os.path.isdir(args.model):
        sys.exit(f"[!] Checkpoint introuvable : {args.model}")
    model_path_wsl = win_to_wsl_path(args.model)

    backends = [b.strip() for b in args.moe_backend.split(",") if b.strip()]

    results = []
    for i, backend in enumerate(backends, 1):
        print(f"[{i}/{len(backends)}] --moe-backend {backend}")
        try:
            r = run_one_config(model_path_wsl, backend, args.port, args.prompt,
                                args.n_predict, args.load_timeout, args.gen_timeout)
        except Exception as e:
            r = {"backend": backend, "error": f"exception interne bench : {e}"}
        results.append(r)
        if "error" in r:
            print(f"    [!] {r['error']}")
        else:
            print(f"    VRAM chargé={r['vram_loaded_mb']:.0f} Mo  après-gen={r['vram_after_gen_mb']:.0f} Mo  "
                  f"tok/s={r['tokens_per_second']}  ({r['completion_tokens']} tokens en {r['wall_s']}s)")

    print("\n" + "=" * 100)
    print(f"  D2 FREETOKEN BENCH — modèle={os.path.basename(args.model)}")
    print("=" * 100)
    print(f"  {'backend':>10} {'VRAM chargé':>12} {'VRAM +gen':>12} {'tok/s':>8} {'tokens':>8} {'wall_s':>8}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"  {r['backend']:>10}  ÉCHEC : {r['error']}")
            continue
        print(f"  {r['backend']:>10} "
              f"{r['vram_loaded_mb']:>10.0f} Mo "
              f"{r['vram_after_gen_mb']:>10.0f} Mo "
              f"{(r['tokens_per_second'] or 0):>8.2f} "
              f"{(r['completion_tokens'] or 0):>8} "
              f"{r['wall_s']:>8}")
    print("=" * 100)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "model_path_wsl": model_path_wsl,
                    "prompt": args.prompt, "n_predict": args.n_predict,
                    "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] -> {args.json}")


if __name__ == "__main__":
    main()
