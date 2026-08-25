#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [CORRIGÉ 25/08/2026] La tokenisation MD5 produit des IDs arbitraires = BRUIT
# SEMANTIQUE (pas de lien avec l'embedding reel du mot). Toutes les sorties sont
# re-titrees « ROUTING PROXY (tokens hashés) » pour ne pas confondre avec du
# routing réel. FAIT MESURÉ 24/08 : 256/256 experts actifs, entropie routeur
# 0.998 — ce script est un PROXY, pas une mesure du routing en production.
"""
D2 MOE ROUTING TRACER — PROXY de routing (tokens hashés), PAS un routing réel.
===============================================================================
Lit les 40 tenseurs mlp.gate.weight [256, 2048] depuis les safetensors,
charge le corpus, et pour chaque token simule :
  hidden_state[2048] -> gate @ hidden -> softmax -> top-8

Sans forward reel (pas assez de RAM), on simule le hidden state :
  - Option A (best): utiliser les embeddings du token courant comme proxy
  - Option B: bruit gaussien centre (proxy naif)
  - Option C: distribution basee sur la norme des gates (statique)

Ce script utilise l'Option A : embedding(token) comme proxy du hidden state.
ATTENTION [CORRIGÉ 25/08/2026] : par defaut le corpus est "tokenisé" par MD5
(word -> hash % vocab). Ces IDs sont SEMANTIQUEMENT INVALIDES (l'embedding
recupéré ne correspond pas au mot). Résultat = PROXY statistique uniquement.
Pour un routing plus proche du réel : --real-tokens FILE avec un fichier de
token IDs produits par llama-server /tokenize sur wiki.test.raw (un ID par ligne).

Sorties :
  - d2_moe_routing_trace.jsonl : trace complete (token, layer, top-8 experts, gate_probs)
  - d2_moe_routing_report.json : stats agreeees (frequence, co-occurrence, locality)
"""

import json
import os
import struct
import sys
import time
from collections import defaultdict, Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD_DIR = os.path.join(HERE, "hf_weights_35b")
INDEX_PATH = os.path.join(SHARD_DIR, "model.safetensors.index.json")

# On utilise des mots reels du corpus (tokenises), et on fait le routing
# avec l'embedding comme proxy du hidden state
# Chargement du tokenizer GGUF via subprocess llama-tokenize ou fallback
CORPUS_PATH = os.path.join(HERE, "ppl_test_small.txt")


def read_tensor_bf16(path, info):
    """Lit un tenseur BF16 et le retourne en float32."""
    hlen, hdr = read_header_raw(path)
    dtype = info["dtype"]
    shape = info["shape"]
    nelems = 1
    for s in shape:
        nelems *= s
    off = info["data_offsets"][0]
    bpe = 2 if dtype in ("BF16", "F16") else 4
    
    with open(path, "rb") as fh:
        fh.seek(8 + hlen + off)
        raw = fh.read(nelems * bpe)
    
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").astype(np.float32).reshape(shape)
    else:  # BF16
        u = np.frombuffer(raw, dtype="<u2")
        W = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float32)
        return W.reshape(shape)


def read_header_raw(path):
    with open(path, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hlen))
    return hlen, hdr


def load_gates(shard_dir, index_path):
    """Charge les 40 tenseurs mlp.gate.weight [256, 2048]."""
    with open(index_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]

    gates = {}  # layer -> np.array [256, 2048]
    for tname, shard in wm.items():
        if ".mlp.gate.weight" in tname and ".shared" not in tname:
            layer = int(tname.split(".layers.")[1].split(".")[0])
            path = os.path.join(shard_dir, shard)
            with open(path, "rb") as fh:
                hlen = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(hlen))
            info = hdr[tname]
            W = read_tensor_bf16(path, info)  # [256, 2048]
            gates[layer] = W.astype(np.float32)
            print("  Layer {}: gate shape={} loaded".format(layer, W.shape))

    return gates


def load_embedding(shard_dir, index_path):
    """Charge l'embedding token [vocab_size, 2048] pour servir de proxy hidden state."""
    with open(index_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]

    for tname, shard in wm.items():
        if tname.endswith("embed_tokens.weight"):
            path = os.path.join(shard_dir, shard)
            with open(path, "rb") as fh:
                hlen = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(hlen))
            info = hdr[tname]
            W = read_tensor_bf16(path, info)  # [vocab_size, hidden]
            print("  Embedding: shape={} loaded".format(W.shape))
            # Normaliser (les embeddings ont des normes tres variables)
            norms = np.linalg.norm(W, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            W = W / norms * 10.0  # scale pour donner du signal au softmax
            return W.astype(np.float32)

    # Fallback: utiliser le bruit gaussien
    print("  Embedding non trouve, fallback bruit")
    return None


DEFAULT_N_TOKENS = 4000  # [CORRIGÉ 25/08/2026] 500 -> 4000 (échantillon plus stable)


def load_real_tokens(path):
    """[CORRIGÉ 25/08/2026] Lit un fichier de tokens REELS (wiki.test.raw tokenisé
    par llama-server /tokenize) : JSON {"tokens": [...]} ou IDs séparés par
    espaces/lignes. Ces IDs ont une vraie sémantique, contrairement au hash MD5."""
    import re as _re
    with open(path, encoding="utf-8") as fh:
        content = fh.read().strip()
    tokens = []
    if content.startswith("{"):
        data = json.loads(content)
        raw = data.get("tokens", [])
        for t in raw:
            if isinstance(t, bool):
                continue
            if isinstance(t, int):
                tokens.append(t)
            else:
                try:
                    tokens.append(int(str(t)))
                except ValueError:
                    continue
    else:
        tokens = [int(m) for m in _re.findall(r"\d+", content)]
    print("  Tokens REELS charges depuis {}: {} tokens".format(path, len(tokens)))
    return tokens


def tokenize_corpus(path, n_tokens=DEFAULT_N_TOKENS, vocab_size=248320):
    """Tokenise le corpus via subprocess (llama-tokenize) ou fallback simple."""
    # Fallback: tokenisation naive par mots (pas parfaite mais donne des ID distincts)
    with open(path, encoding="utf-8") as f:
        text = f.read()

    # [CORRIGÉ 25/08/2026] Tokenisation MD5 = BRUIT SEMANTIQUE : l'ID produit ne
    # pointe vers aucun embedding porteur de sens. Ne pas interpreter les
    # resultats comme du routing reel — uniquement comme proxy statistique.
    # Pour du reel : --real-tokens FILE (sortie llama-server /tokenize).
    import hashlib
    tokens = []
    for word in text.split():
        h = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        token_id = h % vocab_size
        tokens.append(token_id)

    print("  Corpus 'tokenise' par MD5 (BRUIT SEMANTIQUE, proxy uniquement): "
          "{} tokens".format(len(tokens)))
    return tokens[:n_tokens]


def simulate_routing(gates, embedding, tokens, temperature=1.0):
    """
    Simule le routing pour une sequence de tokens.
    
    Pour chaque token t et chaque couche L :
      hidden ~ embedding[token[t]]  (proxy du hidden state)
      logits = gate[L] @ hidden     [256]
      probs = softmax(logits / temperature)
      top8 = indices des 8 plus grandes probs
    
    C'est une approximation : le vrai hidden state depend de toute la sequence,
    pas seulement du token courant. Mais ca capture la dependance semantique
    du routing au contenu du token.
    """
    n_layers = len(gates)
    trace = []

    # Pre-calculer les logits pour tous les tokens et toutes les couches
    # Pour accelerer, on fait un batch de tokens contre chaque gate
    all_token_ids = tokens
    n_tokens = len(all_token_ids)

    if embedding is not None:
        # Utiliser les embeddings reels
        h = embedding[all_token_ids]  # [n_tokens, 2048]
    else:
        # Fallback: bruit gaussien avec seed par token (reproductible)
        np.random.seed(42)
        h = np.random.randn(n_tokens, 2048).astype(np.float32)
    
    print("  Simulating routing: {} tokens x {} layers...".format(n_tokens, n_layers))
    t0 = time.time()

    for L in sorted(gates):
        gate = gates[L]  # [256, 2048]
        logits = np.dot(h, gate.T)  # [n_tokens, 256]
        
        # Softmax
        logits_max = np.max(logits, axis=1, keepdims=True)
        logits_exp = np.exp((logits - logits_max) / temperature)
        probs = logits_exp / np.sum(logits_exp, axis=1, keepdims=True)

        # Top-8 par token
        top8_idx = np.argsort(-probs, axis=1)[:, :8]
        top8_probs = np.take_along_axis(probs, top8_idx, axis=1)

        for t in range(n_tokens):
            trace.append({
                "token_idx": t,
                "token_id": int(all_token_ids[t]),
                "layer": L,
                "top8_experts": top8_idx[t].tolist(),
                "top8_probs": [round(float(p), 6) for p in top8_probs[t]],
                "entropy": round(float(-np.sum(probs[t] * np.log(probs[t] + 1e-10)) / np.log(256)), 6),
            })

    dt = time.time() - t0
    print("  Trace generee en {:.1f}s ({} entrees)".format(dt, len(trace)))

    return trace


def aggregate_trace(trace, output_report):
    """Agrege la trace en statistiques exploitables."""
    # Frequence par expert
    expert_freq = Counter()
    expert_per_layer = defaultdict(Counter)
    # Co-occurrence (dans la meme couche, meme token)
    cooccur = Counter()
    # Consecutive overlap (meme layer, tokens t et t+1)
    consecutive_overlaps = []

    # Organiser par token
    entries_by_token = defaultdict(lambda: defaultdict(list))
    for e in trace:
        entries_by_token[e["token_idx"]][e["layer"]] = e

    n_tokens = len(entries_by_token)
    n_layers = max(e["layer"] for e in trace) + 1

    for t in range(n_tokens):
        for L in range(n_layers):
            if L not in entries_by_token[t]:
                continue
            e = entries_by_token[t][L]
            experts = e["top8_experts"]
            probs = e["top8_probs"]

            for exp, prob in zip(experts, probs):
                expert_freq[exp] += prob
                expert_per_layer[L][exp] += prob

            # Co-occurrence dans cette couche
            for i in range(len(experts)):
                for j in range(i + 1, len(experts)):
                    pair = tuple(sorted([experts[i], experts[j]]))
                    cooccur[pair] += probs[i] * probs[j]

        # Consecutive overlap avec le token precedent
        if t > 0 and t - 1 in entries_by_token:
            overlap_count = 0
            for L in range(n_layers):
                if L in entries_by_token[t] and L in entries_by_token[t - 1]:
                    e_curr = set(entries_by_token[t][L]["top8_experts"])
                    e_prev = set(entries_by_token[t - 1][L]["top8_experts"])
                    overlap_count += len(e_curr & e_prev)
            max_possible = n_layers * 8
            consecutive_overlaps.append(overlap_count / max_possible if max_possible > 0 else 0)

    # Normaliser les frequences
    total_prob = sum(expert_freq.values())
    expert_freq_norm = {e: round(p / total_prob * 100, 2) for e, p in expert_freq.most_common()}

    # Top experts global
    top_global = expert_freq.most_common(16)

    # Top experts par couche
    top_per_layer = {}
    for L in sorted(expert_per_layer):
        top_per_layer[str(L)] = [
            {"expert": e, "freq_pct": round(p / sum(expert_per_layer[L].values()) * 100, 2)}
            for e, p in expert_per_layer[L].most_common(8)
        ]

    # Co-occurrence top paires
    top_pairs = cooccur.most_common(15)

    # Statistiques de locality
    avg_overlap = np.mean(consecutive_overlaps) * 100 if consecutive_overlaps else 0
    # Compte combien d'experts uniques sont actives au total
    active_experts = len([e for e, p in expert_freq.items() if p > 0])
    pct_active = active_experts / 256 * 100

    report = {
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "temperature": 1.0,
        # [CORRIGÉ 25/08/2026] method renommée : PROXY, pas routing réel
        "method": "ROUTING PROXY (tokens hashés MD5 = bruit sémantique ; "
                  "--real-tokens pour des tokens réels llama-server)",
        "active_experts": active_experts,
        "pct_experts_active": round(pct_active, 1),
        "avg_consecutive_overlap_pct": round(avg_overlap, 1),
        "top_experts_global": [
            {"expert": int(e), "freq_pct": p}
            for e, p in top_global
        ],
        "top_experts_per_layer": top_per_layer,
        "top_cooccurrence_pairs": [
            {"pair": list(p), "score": round(s, 4)}
            for p, s in top_pairs
        ],
        "expert_frequencies_full": {str(k): v for k, v in sorted(expert_freq_norm.items(), key=lambda x: -x[1])[:50]},
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # [CORRIGÉ 25/08/2026] argparse : --real-tokens (tokens reels llama-server),
    # --n-tokens (4000 par defaut au lieu de 500)
    import argparse
    ap = argparse.ArgumentParser(description="D2 MoE Routing Tracer — PROXY (tokens hashés)")
    ap.add_argument("--real-tokens", default=None,
                    help="Fichier de tokens REELS (wiki.test.raw tokenisé par "
                         "llama-server /tokenize). Sans ce fichier : MD5 = bruit sémantique.")
    ap.add_argument("--n-tokens", type=int, default=DEFAULT_N_TOKENS,
                    help="Taille de l'échantillon (défaut {})".format(DEFAULT_N_TOKENS))
    args = ap.parse_args()

    print("=" * 80)
    print("  D2 MOE ROUTING TRACER — ROUTING PROXY (tokens hashés, PAS un routing réel)")
    print("=" * 80)

    # 1. Charger les gates
    print("\n  [1/5] Chargement des 40 gates de routage...")
    gates = load_gates(SHARD_DIR, INDEX_PATH)
    print("  {} gates chargees".format(len(gates)))

    # 2. Charger l'embedding
    print("\n  [2/5] Chargement de l'embedding...")
    embedding = load_embedding(SHARD_DIR, INDEX_PATH)

    # 3. Tokeniser le corpus
    print("\n  [3/5] Tokenisation du corpus...")
    if args.real_tokens and os.path.exists(args.real_tokens):
        tokens = load_real_tokens(args.real_tokens)[:args.n_tokens]
        method_label = "ROUTING PROXY (tokens REELS llama-server)"
    else:
        if args.real_tokens:
            print("  [!] Fichier absent: {} — fallback MD5".format(args.real_tokens))
        tokens = tokenize_corpus(CORPUS_PATH, n_tokens=args.n_tokens)
        method_label = "ROUTING PROXY (tokens hashés — BRUIT SÉMANTIQUE)"

    # 4. Simuler le routing
    print("\n  [4/5] Simulation du routing forward...")
    trace = simulate_routing(gates, embedding, tokens, temperature=1.0)

    # 5. Agreger
    trace_path = os.path.join(HERE, "d2_moe_routing_trace.jsonl")
    report_path = os.path.join(HERE, "d2_moe_routing_real_report.json")

    with open(trace_path, "w", encoding="utf-8") as f:
        for e in trace:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    report = aggregate_trace(trace, report_path)

    # 6. Affichage
    # [CORRIGÉ 25/08/2026] « ROUTING REEL » -> « ROUTING PROXY » : la tokenisation
    # MD5 injecte du bruit semantique ; ces chiffres ne sont PAS des mesures reelles.
    print("\n  --- {} ---".format(method_label))
    print("  Tokens simules: {}".format(report["n_tokens"]))
    print("  Experts actifs: {} / 256 ({:.1f}%)".format(
        report["active_experts"], report["pct_experts_active"]))
    print("  Overlap moyen consecutif: {:.1f}%".format(
        report["avg_consecutive_overlap_pct"]))
    print("  Top-8 experts globaux:")
    for e in report["top_experts_global"][:8]:
        print("    E{:>3}: {:.2f}%".format(e["expert"], e["freq_pct"]))
    
    # Verifier si le routing est different de la norme statique
    print("\n  --- COMPARAISON STATIQUE vs EMBEDDING-PROXY ---")
    # Les experts par norme statique (du routeur)
    with open(os.path.join(HERE, "d2_router_static_report.json")) as f:
        static = json.load(f)
    static_top8 = [e["expert"] for e in static["experts"][:8]]
    routing_top8 = [e["expert"] for e in report["top_experts_global"][:8]]
    overlap = set(static_top8) & set(routing_top8)
    print("  Top-8 statique (gate norm): {}".format(static_top8))
    print("  Top-8 embedding-proxy:      {}".format(routing_top8))
    print("  Overlap: {} / 8 experts".format(len(overlap)))
    
    print("\n  [+] Trace: {}".format(trace_path))
    print("  [+] Rapport: {}".format(report_path))

    # Verdict
    print("\n  --- VERDICT ROUTING (PROXY — fiabilité ~25% sur proxy statique) ---")
    print("  FAIT MESURÉ (24/08) : 256/256 experts actifs, aucun expert froid,")
    print("  entropie routeur 0.998. Ce proxy ne peut PAS contredire ces faits.")
    if report["pct_experts_active"] > 50:
        print("  Le routing active >50% des experts → le modele utilise VRAIMENT")
        print("  la plupart des 256 experts. L'hypothese '8 experts froids'")
        print("  n'est validee QUE si la distribution est tres asymetrique.")
    else:
        print("  Le routing active <50% des experts → concentration reelle.")
    
    if report["avg_consecutive_overlap_pct"] > 70:
        print("  Overlap >70% → forte localite → cout de chargement amorti")
        print("  → Meme les experts Q4/Q3 changent rarement entre tokens")
    else:
        print("  Overlap <70% → le routing change significativement entre tokens")


if __name__ == "__main__":
    main()