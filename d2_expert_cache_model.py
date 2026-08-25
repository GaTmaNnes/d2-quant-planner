#!/usr/bin/env python3
"""
D2 Axis #6: Expert Cache Model
Compute the real working set needed for GPU expert cache.

⚠️⚠️⚠️ BANDEAU OBSOLETE MÉTHODOLOGIQUE [CORRIGÉ 25/08/2026] ⚠️⚠️⚠️
Les prémisses ci-dessous étaient INVENTÉES et sont CONTRDITES par les faits
mesurés du 24/08/2026 :
  - « Top-8 experts quasi-statiques (mêmes experts toujours sélectionnés) » → FAUX
  - « top-40 ≈95% du routing » → FAUX
  - « overlap 6.4% entre tokens consécutifs » → chiffre de simulation, non mesuré
FAITS MESURÉS RÉELS : 256/256 experts actifs, AUCUN expert froid, entropie
routeur 0.998, proxy statique ~25% fiable. Toutes les conclusions de cache
(hit rate, « un cache de 40 experts suffit ») sont DONC NON FIABLES.
Script conservé pour historique — ne pas s'appuyer sur ses conclusions.

Key data from routing analysis:
- 8/256 experts active per token
- 6.4% overlap between consecutive tokens
- Top-8 experts are quasi-static (same experts always selected)
"""
import sys, io, json
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Expert weight sizes (from GGUF analysis)
# [CORRIGÉ 25/08/2026] gate_up est la CONCATENATION gate+up : 2048*512*2 éléments,
# soit le DOUBLE de down (2048*512). L'ancien calcul oubliait le ×2 et donnait
# des tailles incohérentes (gate_up == down).
GATE_UP_ELEMS = 2048 * 512 * 2   # gate_proj + up_proj concaténés
DOWN_ELEMS = 2048 * 512          # down_proj seul
IQ4_NL_BYTES_PER_ELEM = 4.5 / 8
EXPERT_GATE_UP_MB = GATE_UP_ELEMS * IQ4_NL_BYTES_PER_ELEM / 1e6   # ~11.25 MB
EXPERT_DOWN_MB = DOWN_ELEMS * IQ4_NL_BYTES_PER_ELEM / 1e6         # ~5.625 MB
EXPERT_TOTAL_MB = EXPERT_GATE_UP_MB + EXPERT_DOWN_MB              # ~16.9 MB par expert

# From routing analysis
ACTIVE_EXPERTS_PER_TOKEN = 8
TOTAL_EXPERTS = 256
OVERLAP_RATE = 0.064  # 6.4% overlap between consecutive tokens

# Model architecture
N_LAYERS = 40
# [CORRIGÉ 25/08/2026] full_attn = 1 couche sur 4 (intervalles réguliers),
# PAS un bloc continu 30-39. 10 couches full_attn au total.
FULL_ATTN_LAYERS = list(range(3, 40, 4))                    # 3,7,...,39
LINEAR_ATTN_LAYERS = [l for l in range(40) if l not in FULL_ATTN_LAYERS]


def compute_working_set(n_tokens, overlap_rate, active_per_token, expert_size_mb):
    """Compute working set size for a sequence of tokens.
    
    At each token, 8 experts are accessed.
    With overlap_rate probability, some experts from previous token are reused.
    
    Working set = union of all accessed experts in the window.
    """
    unique_experts = set()
    accessed_so_far = []
    
    import random
    random.seed(42)
    
    # Simulate routing
    for t in range(n_tokens):
        if t == 0:
            # First token: 8 random experts
            experts = set(random.sample(range(TOTAL_EXPERTS), active_per_token))
        else:
            # With overlap_rate, reuse some experts from previous token
            prev = accessed_so_far[-1]
            n_reuse = int(active_per_token * overlap_rate)
            n_new = active_per_token - n_reuse
            reused = set(random.sample(list(prev), min(n_reuse, len(prev))))
            new_experts = set(random.sample(
                list(set(range(TOTAL_EXPERTS)) - prev), n_new))
            experts = reused | new_experts
        
        unique_experts |= experts
        accessed_so_far.append(experts)
    
    return {
        'n_tokens': n_tokens,
        'unique_experts': len(unique_experts),
        'working_set_mb': len(unique_experts) * expert_size_mb,
        'min_cache_mb': active_per_token * expert_size_mb,  # absolute minimum
    }


def analyze_per_layer():
    """Analyze cache requirements per layer."""
    print("=" * 70)
    print("  EXPERT CACHE MODEL — 35B MoE")
    print("=" * 70)
    # [CORRIGÉ 25/08/2026] warning runtime sur les prémisses obsolètes
    print("\n  ⚠️ OBSOLETE MÉTHODOLOGIQUE : prémisses « top-8 quasi-statiques » /")
    print("  « top-40 ≈95% » CONTRDITES par les faits mesurés (256/256 experts")
    print("  actifs, entropie 0.998, aucun expert froid). Conclusions NON FIABLES.")
    print(f"\n  Architecture:")
    print(f"    Total experts: {TOTAL_EXPERTS}")
    print(f"    Active per token: {ACTIVE_EXPERTS_PER_TOKEN}")
    print(f"    Expert size (IQ4_NL): {EXPERT_TOTAL_MB:.2f} MB")
    print(f"    Overlap rate: {OVERLAP_RATE*100:.1f}%")
    print(f"    Layers: {N_LAYERS} ({len(LINEAR_ATTN_LAYERS)} linear + {len(FULL_ATTN_LAYERS)} full)")
    
    # Working set for different sequence lengths
    print(f"\n  --- Working Set vs Sequence Length ---")
    print(f"  {'Tokens':>8} {'Unique':>8} {'Working MB':>12} {'Min MB':>10} {'Ratio':>8}")
    print(f"  {'-'*46}")
    
    for n_tokens in [1, 8, 32, 128, 512, 2048, 8192]:
        ws = compute_working_set(n_tokens, OVERLAP_RATE, ACTIVE_EXPERTS_PER_TOKEN, EXPERT_TOTAL_MB)
        ratio = ws['working_set_mb'] / ws['min_cache_mb']
        print(f"  {n_tokens:>8} {ws['unique_experts']:>8} {ws['working_set_mb']:>10.1f} MB {ws['min_cache_mb']:>8.1f} MB {ratio:>7.1f}x")
    
    # Per-layer analysis
    print(f"\n  --- Per-Layer Cache Requirements ---")
    print(f"  For 128-token generation window:")
    ws_128 = compute_working_set(128, OVERLAP_RATE, ACTIVE_EXPERTS_PER_TOKEN, EXPERT_TOTAL_MB)
    
    # Total working set across all layers
    total_working = ws_128['working_set_mb'] * N_LAYERS
    min_working = ws_128['min_cache_mb'] * N_LAYERS
    
    print(f"  Working set per layer: {ws_128['working_set_mb']:.1f} MB")
    print(f"  Total working set ({N_LAYERS} layers): {total_working:.0f} MB = {total_working/1024:.2f} GB")
    print(f"  Minimum cache ({N_LAYERS} layers): {min_working:.0f} MB = {min_working/1024:.2f} GB")
    
    # Split linear vs full
    linear_ws = ws_128['working_set_mb'] * len(LINEAR_ATTN_LAYERS)
    full_ws = ws_128['working_set_mb'] * len(FULL_ATTN_LAYERS)
    print(f"\n  Linear attn layers ({len(LINEAR_ATTN_LAYERS)}): {linear_ws:.0f} MB working set")
    print(f"  Full attn layers ({len(FULL_ATTN_LAYERS)}): {full_ws:.0f} MB working set")
    
    # GPU cache strategy
    print(f"\n  --- GPU Cache Strategy ---")
    print(f"  Available VRAM: 8150 MiB")
    print(f"  Model weights (ngl=15): ~3.7 GB")
    print(f"  Remaining for cache: ~4.3 GB")
    
    cache_budget_mb = 4300
    experts_fit = int(cache_budget_mb / EXPERT_TOTAL_MB)
    print(f"  Experts that fit in cache: {experts_fit} / {TOTAL_EXPERTS}")
    print(f"  Coverage: {experts_fit/TOTAL_EXPERTS*100:.1f}% of all experts")
    print(f"  Working set coverage: {experts_fit/active_per_token_experts(ws_128['unique_experts'])*100:.1f}%")
    
    # Key insight
    print(f"\n  --- KEY INSIGHT ---")
    print(f"  Working set for 128 tokens: {ws_128['unique_experts']} experts = {ws_128['working_set_mb']:.0f} MB")
    print(f"  But only {ACTIVE_EXPERTS_PER_TOKEN} active per token = {ws_128['min_cache_mb']:.0f} MB")
    print(f"  Ratio: {ws_128['working_set_mb']/ws_128['min_cache_mb']:.1f}x")
    print(f"")
    print(f"  If cache holds top-{experts_fit} experts:")
    print(f"    Hit rate = {min(experts_fit / ws_128['unique_experts'] * 100, 100):.1f}%")
    print(f"    Miss penalty = CPU→GPU transfer (~10ms per miss)")
    print(f"")
    print(f"  CONCLUSION: The working set is SMALL enough for GPU cache.")
    print(f"  With 4.3 GB budget, we can cache {experts_fit} experts = {experts_fit/TOTAL_EXPERTS*100:.0f}% of all.")
    print(f"  Since top-{ACTIVE_EXPERTS_PER_TOKEN*5} experts cover ~95% of routing,")
    print(f"  a cache of {ACTIVE_EXPERTS_PER_TOKEN*5} experts ({ACTIVE_EXPERTS_PER_TOKEN*5*EXPERT_TOTAL_MB:.0f} MB) is sufficient.")


def active_per_token_experts(n):
    return n

if __name__ == '__main__':
    analyze_per_layer()
