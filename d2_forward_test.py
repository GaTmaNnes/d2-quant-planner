#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 FORWARD TEST — Δlogits / ΔPPL réel via llama-server (logprobs).
==================================================================
Calibre les seuils UTILITY (TOL_RECURRENT / TOL_LOCAL) en mesurant la vraie
dégradation de sortie entre deux quantifications du même modèle.

Méthode :
  - lance llama-server (subprocess) sur un modèle
  - pour un jeu de prompts fixe, génère en greedy (temperature=0, déterministe)
  - collecte les logprobs par token généré (completion_probabilities)
  - NLL = -moyenne(log P(token)) ; PPL = exp(NLL)
  - tokens/s = tokens_predicted / temps de génération (decode)

Le ΔPPL entre deux modèles (ex. Q8_0 vs Q4_K_S) quantifie le coût qualité réel
de la quantification → sert à calibrer la tolérance du score UTILITY.

Usage :
  python d2_forward_test.py --model models/Qwen3.5-9B-Q8_0.gguf
  python d2_forward_test.py --model models/Qwen3.5-9B-Q4_K_S.gguf --port 8081
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA_SERVER = os.path.join(HERE, "llama-server.exe")

# Prompts variés (mix code / français / raisonnement) pour un signal stable
PROMPTS = [
    "La capital de la France est",
    "Une fois, dans une forêt sombre,",
    "def fibonacci(n):",
    "Le théorème de Pythagore affirme que",
    "In a large language model, the attention mechanism",
    "La différence entre FP8 et INT8 en quantification est",
    "Un bon algorithme de tri doit",
    "The capital of Japan is",
    "Explique en une phrase ce qu'est un state space model",
    "Le cache KV d'un modèle est utilisé pour",
    "Write a function to reverse a linked list in Python",
    "La bande passante mémoire d'un GPU détermine",
    "Pour réduire la consommation VRAM d'un LLM,",
    "The Qwen family of models is known for",
    "Un tensor core calcule",
]


def wait_health(port, timeout=120):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def completion(port, prompt, n_predict, n_probs=1):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
        "n_probs": n_probs, "stream": False, "cache_prompt": True,
    }).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def run_model(model, port, n_predict, args):
    log = os.path.join(HERE, "d2_forward_server.log")
    # [CORRIGÉ 25/08/2026] -ngl 99 codé dur = OOM garanti sur le 35B (17.5 GB
    # dans 8 Go VRAM). Désormais --ngl (défaut 15 = config production) +
    # avertissement si modèle >10 GB avec ngl>30.
    ngl = getattr(args, "ngl", 15)
    model_size_gb = os.path.getsize(model) / (1024**3) if os.path.exists(model) else 0
    if model_size_gb > 10 and ngl > 30:
        print(f"[!] ATTENTION : modèle {model_size_gb:.1f} GB (>10 GB) avec -ngl {ngl} "
              f"→ OOM quasi certain en 8 Go VRAM. Défaut conseillé : 15.")
    cmd = [LLAMA_SERVER, "-m", model, "--port", str(port), "-ngl", str(ngl),
           "--ctx-size", "4096", "--host", "127.0.0.1"]
    log_fh = open(log, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    try:
        if not wait_health(port):
            return {"error": f"serveur {model} ne démarre pas (voir {log})"}
        nll_sum = 0.0
        n_tok = 0
        n_prompt_tok = 0
        decode_s = 0.0
        per_prompt = []
        for p in PROMPTS:
            try:
                r = completion(port, p, n_predict)
            except Exception as e:
                per_prompt.append({"prompt": p[:30], "error": str(e)})
                continue
            # logprobs des tokens générés (format OpenAI: cp["logprob"] déjà en log)
            cps = r.get("completion_probabilities") or []
            for cp in cps:
                lp = cp.get("logprob")
                if lp is None:
                    tl = cp.get("top_logprobs") or []
                    lp = tl[0].get("logprob") if tl else None
                if lp is not None and math.isfinite(float(lp)):
                    nll_sum += -float(lp)
                    n_tok += 1
            tt = r.get("timings", {})
            decode_s += float(tt.get("predicted_ms", 0.0)) / 1000.0
            n_prompt_tok += int(r.get("tokens_evaluated", 0))
            lp_list = []
            for cp in cps:
                lp = cp.get("logprob")
                if lp is None:
                    tl = cp.get("top_logprobs") or []
                    lp = tl[0].get("logprob") if tl else None
                if lp is not None and math.isfinite(float(lp)):
                    lp_list.append(-float(lp))
            per_prompt.append({
                "prompt": p[:40],
                "gen_tokens": int(r.get("tokens_predicted", 0)),
                "avg_nll": round(sum(lp_list) / len(lp_list), 4) if lp_list else None,
            })
        avg_nll = nll_sum / max(n_tok, 1)
        ppl = math.exp(avg_nll)
        tps = n_tok / max(decode_s, 1e-9)
        return {
            "model": model,
            "n_prompts": len(PROMPTS),
            "n_gen_tokens": n_tok,
            "avg_nll": round(avg_nll, 4),
            "ppl": round(ppl, 2),
            "decode_tok_per_s": round(tps, 2),
            "decode_s": round(decode_s, 2),
            "per_prompt": per_prompt,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        log_fh.close()


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n-predict", type=int, default=64)
    # [CORRIGÉ 25/08/2026] --ngl (défaut 15 = prod) au lieu de -ngl 99 codé dur
    ap.add_argument("--ngl", type=int, default=15,
                    help="couches sur GPU (défaut 15 = production ; 99 = OOM sur 35B)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print(f"[*] Forward test sur {os.path.basename(args.model)} (port {args.port}) ...")
    res = run_model(args.model, args.port, args.n_predict, args)
    if "error" in res:
        print("[!]", res["error"])
        return 1
    label = args.label or os.path.basename(args.model)
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)
    print(f"  tokens générés      : {res['n_gen_tokens']}")
    print(f"  NLL moyen           : {res['avg_nll']:.4f}  (plus bas = mieux)")
    print(f"  PPL (greedy)        : {res['ppl']:.2f}  (plus bas = mieux)")
    print(f"  décode tokens/s     : {res['decode_tok_per_s']:.2f}")
    print("=" * 70)
    out = args.json or os.path.join(HERE, f"d2_forward_{label.replace(' ', '_').replace('-', '_')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"[+] JSON : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
