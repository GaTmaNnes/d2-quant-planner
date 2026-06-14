#!/usr/bin/env python3
"""
D2 Production — Minimal ILP Quantization Planner
=================================================
Formulation log-homogène (V2) :

  score(i,q) = G(q) - λ · layer_weight(i) · risk(q)

  G(q) = log(BPE_FP16 / BPE(q))          ∈ [0, log(4)] ≈ [0, 1.386]
       = log(TPS_q / TPS_FP16)  [Roofline, mémoire-bound, S=1]
  risk(q)  ∈ {0.00, 0.35, 1.00}           proxy de perte qualité

  Les deux termes sont dans le même espace log / borné → λ est
  physiquement interprétable comme un ratio information/qualité.

  bpe GGUF réels (pas théoriques) :
    q8_0  : 32 × 1B + 1 fp16 scale = 34B/32 = 1.0625 bpe
    q4_K_M: 256 × 4b + scales/mins  = 144B/256 = 0.5625 bpe

  Seuils d'indifférence (lw=1.0) :
    λ_indiff(q4_K_M) = log(2/0.5625) / 1.00  ≈ 1.27
    λ_indiff(q8_0)   = log(2/1.0625) / 0.35  ≈ 1.81

  Règle opérationnelle :
    λ < 1.27  → INT4 et INT8 > FP16
    1.27 ≤ λ < 1.81  → INT8 > FP16, INT4 rejeté
    λ ≥ 1.81  → FP16 (conservateur)

  ILP :
    min Σ x[i,q] · (-score(i,q))
    s.t. Σ_q x[i,q] = 1         ∀ i
         Σ_{i,q} x[i,q]·VRAM[i,q] ≤ budget
         x[i,q] ∈ {0,1}
"""

import json
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ortools.linear_solver import pywraplp

# ─── Propriétés dtype (GGUF réels, Roofline mémoire-bound) ──────────────────
# bpe = bytes par élément tels que mesurés dans les blocs GGUF :
#
#   f16   : 2.0000  (exact)
#   q8_0  : 1.0625  (32B weights + 2B scale fp16 par bloc de 32)
#   q4_K_M: 0.5625  (256B×4b=128B weights + 12B scales + 4B dmin/d = 144B/256)
#
# G(q) = log(bpe_f16 / bpe(q)) : gain log-TPS Roofline, S=1, mémoire-bound
# risk : proxy perte qualité [0,1], empirique (perplexité, 3B modèles)
#
# λ_indiff(q, lw) = G(q) / (lw · risk(q))
#   q4_K_M, lw=1.0 : log(2/0.5625)/1.00 ≈ 1.27
#   q8_0,   lw=1.0 : log(2/1.0625)/0.35 ≈ 1.81

_BPE_F16 = 2.0

DTYPE_PROPS: Dict[str, Dict] = {
    'FP16': {'bpe': 2.0000, 'G': 0.0,                          'risk': 0.00,
             'gguf': 'f16'},
    'INT8': {'bpe': 1.0625, 'G': math.log(_BPE_F16 / 1.0625),  'risk': 0.35,
             'gguf': 'q8_0'},    # G ≈ 0.633
    'INT4': {'bpe': 0.5625, 'G': math.log(_BPE_F16 / 0.5625),  'risk': 1.00,
             'gguf': 'q4_K_M'},  # G ≈ 1.268
}
DTYPES = list(DTYPE_PROPS.keys())


# ─── Classification des couches + layer-type weight ─────────────────────────
# layer_weight : amplificateur du risque par type de couche
#   Embed / Head / Norm → 99.0 → forcé FP16 (x[i,'FP16']=1 hard constraint)
#   Attn : 1.3   (plus sensible à la quantification que FFN)
#   FFN  : 0.8   (robuste, bonne cible INT4)
#   Bias : 0.0   (négligeable)

LAYER_WEIGHT: Dict[str, float] = {
    'embed': 99.0,
    'head' : 99.0,
    'norm' : 99.0,
    'attn' : 1.30,
    'ffn'  : 0.80,
    'bias' : 0.00,
    'other': 1.00,
}

def classify(name: str) -> str:
    n = name.lower()
    if any(p in n for p in ('embed', 'wte', 'wpe', 'tok_embed', 'position_embed')):
        return 'embed'
    if any(p in n for p in ('lm_head', 'head.weight', 'output.weight', 'cls')):
        return 'head'
    if any(p in n for p in ('norm', 'ln_', 'layer_norm', 'rms_norm', 'layernorm')):
        return 'norm'
    if any(p in n for p in ('attn', 'attention', 'q_proj', 'k_proj', 'v_proj',
                             'o_proj', 'c_attn', 'c_proj', 'wq', 'wk', 'wv', 'wo',
                             'query', 'key', 'value')):
        return 'attn'
    if any(p in n for p in ('mlp', 'ffn', 'fc', 'gate_proj', 'up_proj', 'down_proj',
                             'c_fc', 'w1', 'w2', 'w3', 'dense')):
        return 'ffn'
    if 'bias' in n:
        return 'bias'
    return 'other'


# ─── Solver principal ────────────────────────────────────────────────────────

def solve_quantization_plan(
    layers: List[Dict],
    vram_budget_gb: float,
    lam: float = 1.0,
    # Alias historique — remplacé par lam (λ physiquement interprétable)
    w_speed: float = 1.0,   # ignoré (G déjà normalisé)
    w_risk: float = None,    # si fourni, utilisé comme alias de lam
) -> List[Dict]:
    """
    ILP via OR-Tools (SCIP).

    Un seul knob : lam (λ).
      score(i,q) = G(q) - λ · lw(i) · risk(q)
      G(q)  = log(BPE_FP16 / BPE(q))  ∈ [0, 1.386]
      risk(q) ∈ [0, 1.00]  (empirique)

    Seuils opérationnels (lw=1.0) :
      λ < 1.39  → INT4 wins
      1.39 ≤ λ < 1.98  → INT8 wins
      λ ≥ 1.98  → FP16 conservative
    """
    # compat alias
    if w_risk is not None:
        lam = w_risk

    budget_bytes = vram_budget_gb * 1e9
    n = len(layers)
    P = len(DTYPES)

    # Pré-calculer scores et VRAM
    lw    = [LAYER_WEIGHT[classify(l['name'])] for l in layers]
    score = []   # (n·P,) à maximiser → minimiser -score
    vram  = []   # (n·P,) bytes

    for i, layer in enumerate(layers):
        sz = layer['shape'][0] * layer['shape'][1]
        for q in DTYPES:
            s = DTYPE_PROPS[q]['G'] - lam * lw[i] * DTYPE_PROPS[q]['risk']
            score.append(s)
            vram.append(sz * DTYPE_PROPS[q]['bpe'])

    # ── OR-Tools SCIP ───────────────────────────────────────────────────
    solver = pywraplp.Solver.CreateSolver('SCIP')

    x = [[solver.BoolVar(f'x_{i}_{q}') for q in range(P)] for i in range(n)]

    # C1 : un seul dtype par couche
    for i in range(n):
        solver.Add(sum(x[i]) == 1)

    # C2 : VRAM ≤ budget
    solver.Add(
        sum(x[i][q] * vram[i*P+q]
            for i in range(n) for q in range(P)) <= budget_bytes
    )

    # C3 : hard constraints couches forcées (lw ≥ 99 → FP16 uniquement)
    for i, layer in enumerate(layers):
        if lw[i] >= 99.0:
            solver.Add(x[i][0] == 1)   # index 0 = FP16

    # Objectif : maximiser score
    obj = solver.Objective()
    for i in range(n):
        for q in range(P):
            obj.SetCoefficient(x[i][q], score[i*P+q])
    obj.SetMaximization()

    status = solver.Solve()
    ok = (status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE))

    plan = []
    for i, layer in enumerate(layers):
        if ok:
            qi = max(range(P), key=lambda q: x[i][q].solution_value())
        else:
            # Greedy fallback
            qi = max(range(P), key=lambda q: score[i*P+q])
        dtype = DTYPES[qi]
        plan.append({
            **layer,
            'dtype'       : dtype,
            'layer_type'  : classify(layer['name']),
            'layer_weight': lw[i],
            'score'       : round(score[i*P+qi], 4),
            'G'           : round(DTYPE_PROPS[dtype]['G'], 4),
            'risk'        : round(DTYPE_PROPS[dtype]['risk'], 3),
            'vram_gb'     : round(layer['shape'][0]*layer['shape'][1]
                                  * DTYPE_PROPS[dtype]['bpe'] / 1e9, 5),
        })
    return plan


# ─── GGUF / llama.cpp export ─────────────────────────────────────────────────
#
# Problème critique : les noms tensors HF safetensors ≠ noms GGUF.
# llama.cpp --override-tensor attend les noms GGUF (post-conversion).
# On implémente un mapping par architecture détectée à partir des noms HF.

import re as _re

def detect_arch(layer_names: List[str]) -> str:
    """Détecte l'architecture depuis les noms de tensors HF."""
    names = ' '.join(layer_names[:10])
    if 'h.0.attn' in names or 'transformer.h' in names:
        return 'gpt2'
    if 'model.layers.0.self_attn' in names:
        return 'llama'
    if 'model.layers.0.self_attention' in names:
        return 'falcon'
    if 'transformer.blocks.0' in names:
        return 'mpt'
    return 'unknown'


def hf_to_gguf_name(hf_name: str, arch: str) -> str:
    """
    Traduit un nom de tensor HF safetensors → nom GGUF llama.cpp.

    Retourne le nom HF original si aucun mapping trouvé (fallback sûr :
    llama.cpp ignorera les noms inconnus plutôt que d'errorer).
    """
    if arch == 'gpt2':
        # Embeddings
        if hf_name == 'wte.weight':           return 'token_embd.weight'
        if hf_name == 'wpe.weight':           return 'position_embd.weight'
        if hf_name in ('ln_f.weight', 'transformer.ln_f.weight'):
            return 'output_norm.weight'
        if hf_name in ('ln_f.bias',  'transformer.ln_f.bias'):
            return 'output_norm.bias'
        # Blocs h.N.xxx
        m = _re.match(r'(?:transformer\.)?h\.(\d+)\.(attn|mlp|ln_1|ln_2)\.(.+)',
                      hf_name)
        if m:
            n, block, rest = m.groups()
            if block == 'attn':
                sub = {'c_attn.weight': 'attn_qkv.weight',
                       'c_attn.bias':   'attn_qkv.bias',
                       'c_proj.weight': 'attn_out.weight',
                       'c_proj.bias':   'attn_out.bias'}.get(rest)
                if sub: return f'blk.{n}.{sub}'
            elif block == 'mlp':
                sub = {'c_fc.weight':   'ffn_up.weight',
                       'c_fc.bias':     'ffn_up.bias',
                       'c_proj.weight': 'ffn_down.weight',
                       'c_proj.bias':   'ffn_down.bias'}.get(rest)
                if sub: return f'blk.{n}.{sub}'
            elif block == 'ln_1':
                return f'blk.{n}.attn_norm.{rest}'
            elif block == 'ln_2':
                return f'blk.{n}.ffn_norm.{rest}'

    elif arch == 'llama':
        if hf_name == 'model.embed_tokens.weight': return 'token_embd.weight'
        if hf_name == 'model.norm.weight':         return 'output_norm.weight'
        if hf_name == 'lm_head.weight':            return 'output.weight'
        m = _re.match(r'model\.layers\.(\d+)\.(.+)', hf_name)
        if m:
            n, rest = m.groups()
            LLAMA_MAP = {
                'self_attn.q_proj.weight':              'attn_q.weight',
                'self_attn.k_proj.weight':              'attn_k.weight',
                'self_attn.v_proj.weight':              'attn_v.weight',
                'self_attn.o_proj.weight':              'attn_output.weight',
                'mlp.gate_proj.weight':                 'ffn_gate.weight',
                'mlp.up_proj.weight':                   'ffn_up.weight',
                'mlp.down_proj.weight':                 'ffn_down.weight',
                'input_layernorm.weight':               'attn_norm.weight',
                'post_attention_layernorm.weight':      'ffn_norm.weight',
            }
            sub = LLAMA_MAP.get(rest)
            if sub: return f'blk.{n}.{sub}'

    return hf_name   # fallback : nom HF brut (llama.cpp l'ignorera sans erreur)


def export_gguf(plan: List[Dict], path: str = 'quant_plan.json',
                arch: str = None) -> str:
    """
    Exporte le plan en JSON + génère --override-tensor pour llama.cpp.

    Les noms tensors sont traduits HF→GGUF via hf_to_gguf_name().
    arch détectée automatiquement si non fournie.

    Usage :
      llama-cli --model model.gguf $(cat quant_plan_cmd.txt)
    """
    if arch is None:
        arch = detect_arch([e['name'] for e in plan])

    rows = []
    for e in plan:
        gguf_name = hf_to_gguf_name(e['name'], arch)
        gguf_type = DTYPE_PROPS[e['dtype']]['gguf']
        rows.append({
            'hf_name'  : e['name'],
            'gguf_name': gguf_name,
            'gguf_type': gguf_type,
            'dtype'    : e['dtype'],
            'layer_type': e['layer_type'],
        })

    cmd = ' \\\n  '.join(
        f"--override-tensor {r['gguf_name']}={r['gguf_type']}"
        for r in rows
    )
    data = {
        'format' : 'llama_cpp_override_tensor',
        'version': '2.0',
        'arch'   : arch,
        'mapping': rows,
        'llama_cpp_cmd': cmd,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    with open(path.replace('.json', '_cmd.txt'), 'w') as f:
        f.write(cmd)
    return cmd


# ─── Summary ─────────────────────────────────────────────────────────────────

def summarize(plan: List[Dict], vram_budget_gb: float,
              lam: float = 1.0,
              w_speed: float = 1.0, w_risk: float = None) -> str:
    if w_risk is not None:
        lam = w_risk
    counts  = Counter(e['dtype'] for e in plan)
    by_type = Counter(f"{e['layer_type']}:{e['dtype']}" for e in plan)
    vram_gb = sum(e['vram_gb'] for e in plan)
    avg_G   = sum(DTYPE_PROPS[e['dtype']]['G'] for e in plan) / len(plan)

    by_type_str = '  '.join(
        f"{lt}→{dt}:{c}" for (lt_dt, c) in sorted(by_type.items())
        for lt, dt in [lt_dt.split(':')]
    )

    # λ position relative aux seuils d'indifférence (bpe GGUF réels)
    regime = ('INT4-dominant' if lam < 1.27
               else 'INT8-dominant' if lam < 1.81
               else 'FP16-conservative')

    return '\n'.join([
        f"\n{'='*54}",
        f"  D2 Production -- Quantization Plan (log space)",
        f"{'='*54}",
        f"  Layers     : {len(plan)}  lam={lam:.2f} [{regime}]",
        f"  Budget     : {vram_budget_gb:.1f} GB   Used: {vram_gb:.3f} GB",
        f"  FP16       : {counts.get('FP16',0)}",
        f"  INT8       : {counts.get('INT8',0)}",
        f"  INT4       : {counts.get('INT4',0)}",
        f"  Avg G(q)   : {avg_G:.3f}  (log-TPS gain vs FP16 baseline)",
        f"  lam_indiff : q4_K_M~1.27  q8_0~1.81  (lw=1.0, real GGUF bpe)",
        f"  By type    : {by_type_str}",
    ])


# --- Demo -------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    try:
        from tabulate import tabulate
        TAB = True
    except ImportError:
        TAB = False

    CACHE_DIR = '/tmp/hf_cache'

    def _load_gpt2():
        from huggingface_hub import hf_hub_download
        import safetensors.torch as st
        local = hf_hub_download('gpt2', 'model.safetensors', cache_dir=CACHE_DIR)
        tensors = st.load_file(local)
        layers = []
        for name, tensor in tensors.items():
            if tensor.ndim < 2: continue
            t = tensor.float().numpy()
            if t.ndim > 2: t = t.reshape(t.shape[0], -1)
            m, n = t.shape
            if m < 8 or n < 8: continue
            layers.append({'name': name, 'shape': [m, n]})
        return layers

    try:
        layers = _load_gpt2()
        print(f"\n  GPT-2 loaded : {len(layers)} layers")
    except Exception as e:
        print(f"\n  [HF] fallback ({e})")
        layers = (
            [{'name': 'wte.weight',  'shape': [50257, 768]}] +
            [{'name': f'h.{i}.{t}', 'shape': s}
             for i in range(12)
             for t, s in [('attn.c_attn.weight', [768, 2304]),
                           ('attn.c_proj.weight', [768, 768]),
                           ('mlp.c_fc.weight',    [768, 3072]),
                           ('mlp.c_proj.weight',  [3072, 768])]] +
            [{'name': 'lm_head.weight', 'shape': [50257, 768]}]
        )

    # -- Test 1: divergence at real GGUF thresholds --------------------------
    print(f"\n{'#'*54}")
    print(f"  TEST: lambda divergence (real GGUF bpe)")
    print(f"  G(q4_K_M)={math.log(2.0/0.5625):.4f}  G(q8_0)={math.log(2.0/1.0625):.4f}")
    print(f"  lam_indiff: q4_K_M~1.27  q8_0~1.81  (lw=1.0)")
    print(f"{'#'*54}")

    results = {}
    for lam in [0.5, 1.27, 1.81, 3.0]:
        plan = solve_quantization_plan(layers, vram_budget_gb=6.0, lam=lam)
        c = Counter(e['dtype'] for e in plan)
        vg = sum(e['vram_gb'] for e in plan)
        results[lam] = (c, vg, plan)

    rows = [[f"lam={w:.2f}", c.get('FP16',0), c.get('INT8',0), c.get('INT4',0),
             f"{v:.3f}"] for w, (c,v,_) in results.items()]
    if TAB:
        print(tabulate(rows, headers=['Lambda','FP16','INT8','INT4','VRAM_GB'],
                       tablefmt='psql'))
    else:
        print(f"  {'Lambda':<12} FP16  INT8  INT4  VRAM_GB")
        for r in rows:
            print(f"  {r[0]:<12} {r[1]:>4}  {r[2]:>4}  {r[3]:>4}  {r[4]:>7}")

    # -- Test 2: GGUF name mapping -------------------------------------------
    print(f"\n{'#'*54}")
    print(f"  TEST: HF -> GGUF name mapping")
    print(f"{'#'*54}")

    plan_127 = results[1.27][2]
    arch = detect_arch([e['name'] for e in plan_127])
    print(f"  arch: {arch}")
    print()

    for e in plan_127[:10]:
        gguf = hf_to_gguf_name(e['name'], arch)
        ok = 'OK' if gguf != e['name'] else '??(fallback)'
        print(f"  {e['dtype']:<5} {e['name'][:36]:<36} -> {gguf[:36]:<36} {ok}")

    # -- Summary + export ----------------------------------------------------
    print(summarize(plan_127, 6.0, lam=1.27))

    cmd = export_gguf(plan_127, '/tmp/hf_cache/quant_plan.json', arch=arch)
    lines_cmd = cmd.split('\n')
    print(f"\n  === llama.cpp --override-tensor (first 6) ===")
    for l in lines_cmd[:6]:
        print(f"  {l}")
    print(f"  ... ({len(lines_cmd)} tensors total)")
    print(f"  -> quant_plan.json + quant_plan_cmd.txt written")
