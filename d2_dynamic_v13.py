"""
D2-DYNAMIC ENGINE V13 — Neural Kernel Router + Precision Hysteresis
====================================================================
Patches BS (cross-reference arXiv 2026-06) :
  BS-01  SINK_THRESHOLD 128 -> 4  (StreamingLLM/KVQuant)
         + dynamique via --sink-threshold ou ctx_len
  BS-02  best_quant batch-aware CORRIGE :
         batch 1   -> Q4_K_M
         batch 2-7 -> Q4_K_M (conservateur, tâche inconnue)
         batch >=8 -> NVFP4_SAFE (throughput prime)
  BS-06  SINK_DTYPE=BF16 (Flash Attention 2/3 natif)

Bugs corrigés (critique 2026-06-14) :
  FIX-1  EMA signal basé sur actual_switch (pas desired_switch)
         => switch bloqué par hysteresis ne pénalise plus la stabilité
  FIX-2  batch 2-7 -> Q4_K_M (conservateur) au lieu de NVFP4_SAFE
  FIX-3  STAB_EMA_ALPHA exposé en CLI (--ema-alpha)
  FIX-4  ATTN peut utiliser best_quant si complexity < 0.8 ET age >= SINK_THRESHOLD
  FIX-5  Export JSON complet : alpha_w, best_quant, op_class, stability_history
"""

import json
import sys
import argparse
import math
from collections import Counter

# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────
DEFAULT_PLAN    = "d2_unified_plan_FULL.json"

# BS-01 : StreamingLLM/KVQuant -> 1-4 tokens suffisent
# Note: 4 est défendable, pas prouvé optimal; f(ctx_len) serait plus rigoureux
DEFAULT_SINK    = 4

STAB_EMA_ALPHA  = 0.15    # exposé via --ema-alpha (FIX-3)
HYSTERESIS_MIN  = 0.60

# BS-06 : FA2/3 natif BF16 sur Ampere+
SINK_DTYPE      = "BF16"

KERNEL_MAP = {
    "NVFP4":      "MARLIN",
    "NVFP4_SAFE": "MARLIN",
    "INT4":       "MARLIN",
    "Q4_K_M":     "MARLIN",
    "INT8":       "CUTLASS",
    "BF16":       "CUBLASLT",
    "FP16":       "CUBLASLT",
    "FP32":       "CUBLASLT",
}

ATTN_OPS = {"ATTN_INPUT", "ATTN_OUTPUT"}


# ─────────────────────────────────────────────────────────────
# BS-01 : sink threshold dynamique (optionnel, non prouvé optimal)
# ─────────────────────────────────────────────────────────────
def compute_sink_threshold(ctx_len: int = 0, override: int = 0) -> int:
    """
    Calcule le sink threshold.
    Si override > 0, l'utilise directement.
    Sinon heuristique très conservatrice basée sur ctx_len.
    Note: la littérature valide 1-4 tokens. Toute valeur > 4 est spéculative.
    """
    if override > 0:
        return override
    # 4 pour la majorité des contextes (StreamingLLM)
    # Légèrement plus conservateur sur très long contexte (spéculatif)
    if ctx_len > 131072:
        return 8   # non validé par la littérature, conservateur
    return 4


# ─────────────────────────────────────────────────────────────
# BS-02 : batch-aware — CORRIGÉ (FIX-2)
# ─────────────────────────────────────────────────────────────
def _batch_best_quant(node_best_quant: str, batch_size: int = 1) -> str:
    """
    Sélection batch-aware du format optimal.

    Basé sur arxiv 2601.14277 (qualité NVFP4 dégradée sur tâches difficiles).
    NOTE: le batch n'est pas la variable dominante (task difficulty l'est),
    mais c'est le proxy disponible à l'exécution.

    FIX-2: batch 2-7 -> Q4_K_M (conservateur),
    pas NVFP4_SAFE comme avant (dead code corrigé).
    """
    if batch_size <= 1:
        return "Q4_K_M"      # single-user: qualité prime
    if batch_size >= 8:
        return node_best_quant   # production: throughput prime (NVFP4_SAFE)
    # batch 2-7 : tâche inconnue -> conservateur
    return "Q4_K_M"


# ─────────────────────────────────────────────────────────────
# Plan demo inline
# ─────────────────────────────────────────────────────────────
def _make_demo_plan(n_blocks=4):
    nodes = {}
    for b in range(n_blocks):
        specs = [
            ("attn_q",      "ATTN_INPUT"),
            ("attn_k",      "ATTN_INPUT"),
            ("attn_v",      "ATTN_INPUT"),
            ("attn_output", "ATTN_OUTPUT"),
            ("ffn_gate",    "FFN_INPUT"),
            ("ffn_up",      "FFN_INPUT"),
            ("ffn_down",    "FFN_OUTPUT"),
        ]
        for op, cls in specs:
            name = "blk.{}.{}.weight".format(b, op)
            nodes[name] = {
                "precision":  "INT8",
                "best_quant": "NVFP4_SAFE",
                "alpha_w":    2.1 + b * 0.05,
                "op_class":   cls,
            }
    return {"nodes": nodes, "global_stats": {}}


def _plan_exists(path):
    try:
        open(path).close()
        return True
    except FileNotFoundError:
        return False


# ─────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────
class D2DynamicEngineV13:
    """
    V13 : Precision Hysteresis + Dynamic Cache Slicing.

    Bugs corrigés :
      FIX-1  EMA pénalise uniquement les vrais switches (actual_switch)
      FIX-2  batch 2-7 -> Q4_K_M
      FIX-3  ema_alpha configurable
      FIX-4  ATTN peut utiliser best_quant hors sink + faible complexité
      FIX-5  dump_plan() exporte metadata complet
    """

    def __init__(self, plan_path=DEFAULT_PLAN, demo=False,
                 batch_size=1, sink_threshold=DEFAULT_SINK,
                 ema_alpha=STAB_EMA_ALPHA, ctx_len=0):
        if demo:
            raw = _make_demo_plan()
        else:
            with open(plan_path, "r") as f:
                raw = json.load(f)

        self.base_plan = raw["nodes"]
        self.precision_state   = {n: d["precision"] for n, d in self.base_plan.items()}
        self.stability_history = {n: 1.0 for n in self.base_plan}
        self.step_count   = 0
        self.batch_size   = batch_size
        self.ema_alpha    = ema_alpha    # FIX-3
        self.sink_threshold = compute_sink_threshold(ctx_len, sink_threshold)

    # ── sélection adaptative ─────────────────────────────────
    def _target_precision(self, name, node, complexity, token_age):
        """
        Priorités (FIX-4 : ATTN peut maintenant utiliser best_quant) :

          1. Attention sink (age < sink_threshold + ATTN)  -> SINK_DTYPE (BF16)
          2. Requête complexe (>0.8) + ATTN               -> SINK_DTYPE (BF16)
          3. best_quant dispo + stable (>= 0.75)          -> best_quant (batch-aware)
             FIX-4 : s'applique aussi aux ATTN si complexity <= 0.8 ET age >= sink_threshold
          4. Fallback: precision du plan de base
        """
        op_cls  = node.get("op_class", "")
        is_attn = op_cls in ATTN_OPS

        # Règle 1 : attention sink strict
        if token_age < self.sink_threshold and is_attn:
            return SINK_DTYPE

        # Règle 2 : haute complexité + ATTN -> BF16
        # FIX-4 : seulement si complexity > 0.8, sinon ATTN peut aussi quantizer
        if complexity > 0.8 and is_attn:
            return SINK_DTYPE

        # Règle 3 : best_quant si couche stable
        # FIX-4 : s'applique maintenant aux ATTN hors sink ET faible complexité
        best = node.get("best_quant", "")
        if best and self.stability_history[name] >= 0.75:
            return _batch_best_quant(best, self.batch_size)

        return node["precision"]

    # ── hysteresis — FIX-1 ──────────────────────────────────
    def _apply_hysteresis(self, name, target_p):
        """
        FIX-1 : EMA basée sur actual_switch (après décision blocage),
        pas sur desired_switch.

        Avant (bug) : un switch BLOQUÉ signalait quand même signal=0
        -> stabilité continuait de baisser même sans changement réel.

        Après : si le switch est bloqué, aucun changement d'état réel
        -> signal=1 (couche stable) -> stabilité ne baisse pas.
        """
        current = self.precision_state[name]
        desired_switch = (current != target_p)

        if desired_switch:
            if self.stability_history[name] < HYSTERESIS_MIN:
                # switch bloqué : target revient à current, aucun changement réel
                target_p = current
            else:
                # switch autorisé
                self.precision_state[name] = target_p

        # FIX-1 : actual_switch = y-a-t-il eu un vrai changement ?
        actual_switch = (self.precision_state[name] != current)
        signal = 0.0 if actual_switch else 1.0

        # EMA de stabilité (FIX-3 : ema_alpha configurable)
        self.stability_history[name] = (
            (1.0 - self.ema_alpha) * self.stability_history[name]
            + self.ema_alpha * signal
        )

        return target_p

    # ── plan step ─────────────────────────────────────────────
    def update_plan(self, request_complexity, token_age):
        """
        Génère le plan dynamique pour un step.

        Args:
            request_complexity: float [0,1]
            token_age:          int (position token dans le contexte)

        Returns:
            dict {name: {"precision": str, "kernel": str}}
        """
        self.step_count += 1
        dynamic_plan = {}

        for name, node in self.base_plan.items():
            target  = self._target_precision(name, node, request_complexity, token_age)
            final_p = self._apply_hysteresis(name, target)
            dynamic_plan[name] = {
                "precision": final_p,
                "kernel":    KERNEL_MAP.get(final_p, "CUTLASS"),
            }

        return dynamic_plan

    # ── stats ─────────────────────────────────────────────────
    def plan_stats(self, plan):
        prec = [v["precision"] for v in plan.values()]
        kern = [v["kernel"]    for v in plan.values()]
        return {
            "precision_dist": dict(Counter(prec)),
            "kernel_dist":    dict(Counter(kern)),
            "n_layers":       len(plan),
        }

    # ── FIX-5 : export JSON complet ──────────────────────────
    def dump_plan(self, dynamic_plan, out_path):
        """
        FIX-5 : Export complet incluant :
          - precision + kernel (plan dynamique)
          - alpha_w, best_quant, op_class (du plan de base)
          - stability_history (état du routeur pour reprise)
        Permet à un outil downstream de reconstruire l'état complet.
        """
        full = {}
        for name, v in dynamic_plan.items():
            base = self.base_plan.get(name, {})
            full[name] = {
                # plan dynamique
                "precision":         v["precision"],
                "kernel":            v["kernel"],
                # metadata plan de base
                "base_precision":    base.get("precision", ""),
                "best_quant":        base.get("best_quant", ""),
                "alpha_w":           base.get("alpha_w", 0.0),
                "op_class":          base.get("op_class", ""),
                # état routeur
                "stability":         round(self.stability_history.get(name, 1.0), 4),
            }
        meta = {
            "step":           self.step_count,
            "batch_size":     self.batch_size,
            "sink_threshold": self.sink_threshold,
            "ema_alpha":      self.ema_alpha,
            "sink_dtype":     SINK_DTYPE,
        }
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "layers": full}, f, indent=2)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def _print_sample(plan, label, n=4):
    print("\n  [{}]".format(label))
    for name, v in list(plan.items())[:n]:
        print("    {:<42} {:<12} {}".format(name, v["precision"], v["kernel"]))
    if len(plan) > n:
        print("    ... ({} couches supplémentaires)".format(len(plan) - n))


def main():
    parser = argparse.ArgumentParser(description="D2 Dynamic Engine V13 (bugs corriges)")
    parser.add_argument("--plan",        default=DEFAULT_PLAN,
                        help="Plan JSON d'entree")
    parser.add_argument("--demo",        action="store_true",
                        help="Mode demo sans fichier plan")
    parser.add_argument("--out",         default="d2_v13_dynamic_plan.json",
                        help="Fichier JSON de sortie")
    parser.add_argument("--steps",       type=int, default=3,
                        help="Nombre de steps a simuler")
    parser.add_argument("--batch-size",  type=int, default=1,
                        help="Batch (1=Q4_K_M, 2-7=Q4_K_M conservateur, >=8=NVFP4)")
    parser.add_argument("--sink-threshold", type=int, default=0,
                        help="Sink threshold (0=auto: 4 pour ctx<=128K, 8 pour >128K)")
    parser.add_argument("--ctx-len",     type=int, default=0,
                        help="Longueur contexte pour calcul auto du sink threshold")
    parser.add_argument("--ema-alpha",   type=float, default=STAB_EMA_ALPHA,
                        help="EMA alpha stabilite (defaut: 0.15, FIX-3)")
    args = parser.parse_args()

    demo = args.demo or not _plan_exists(args.plan)
    if demo and not args.demo:
        print("  [!] {} introuvable — mode demo actif".format(args.plan))

    sink = compute_sink_threshold(args.ctx_len, args.sink_threshold)
    best_q_label = _batch_best_quant("NVFP4_SAFE", args.batch_size)

    print("=" * 64)
    print("  D2 Dynamic Engine V13 — Hystéresis + Kernel Router")
    print("  FIX: EMA actual_switch | batch 2-7→Q4_K_M | export complet")
    print("=" * 64)

    engine = D2DynamicEngineV13(
        plan_path=args.plan,
        demo=demo,
        batch_size=args.batch_size,
        sink_threshold=args.sink_threshold,
        ema_alpha=args.ema_alpha,
        ctx_len=args.ctx_len,
    )

    print("  Plan          : {}".format("demo inline" if demo else args.plan))
    print("  Couches       : {}".format(len(engine.base_plan)))
    print("  Batch         : {}  -> best_quant={}".format(args.batch_size, best_q_label))
    print("  Sink threshold: {} tokens -> {} (BS-01+BS-06)".format(sink, SINK_DTYPE))
    print("  EMA alpha     : {}".format(args.ema_alpha))

    scenarios = [
        (0.9, 2,   "RAG/Reasoning  (c=0.9, age=2)    sink actif"),
        (0.5, 256, "Generation mid (c=0.5, age=256)"),
        (0.2, 512, "Chit-chat      (c=0.2, age=512)"),
    ][:args.steps]

    last_plan = None
    for complexity, age, label in scenarios:
        plan  = engine.update_plan(complexity, age)
        stats = engine.plan_stats(plan)
        _print_sample(plan, label)
        print("    Précisions : {}".format(stats["precision_dist"]))
        print("    Kernels    : {}".format(stats["kernel_dist"]))
        last_plan = plan

    if last_plan:
        # FIX-5 : export complet avec metadata
        engine.dump_plan(last_plan, args.out)
        print("\n  Plan exporté  : {} (complet: alpha_w + stability + meta)".format(args.out))

    print("  V13 v2 OK — bugs corrigés (FIX 1-5)")


if __name__ == "__main__":
    main()
