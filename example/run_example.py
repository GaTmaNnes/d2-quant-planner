#!/usr/bin/env python3
"""
D2 — Exemple d'utilisation en ligne de commande
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from d2_production import solve_quantization_plan, summarize, export_gguf


def main():
    print("=== D2 Quantization Planner - Exemple ===")
    print()
    
    # Exemple de layers (simulé - en vrai chargé depuis safetensors)
    example_layers = [
        {"name": "model.layers.0.self_attn.q_proj.weight", "shape": [4096, 4096]},
        {"name": "model.layers.0.self_attn.k_proj.weight", "shape": [1024, 4096]},
        {"name": "model.layers.0.self_attn.v_proj.weight", "shape": [1024, 4096]},
        {"name": "model.layers.0.self_attn.o_proj.weight", "shape": [4096, 4096]},
        {"name": "model.layers.0.mlp.gate_proj.weight", "shape": [14336, 4096]},
        {"name": "model.layers.0.mlp.up_proj.weight", "shape": [14336, 4096]},
        {"name": "model.layers.0.mlp.down_proj.weight", "shape": [4096, 14336]},
        {"name": "lm_head.weight", "shape": [128256, 4096]},
    ]
    
    # Paramètres
    vram_budget = 6.0      # GB
    w_risk = 0.45          # plus élevé = plus conservateur
    
    print(f"Budget VRAM : {vram_budget} GB | w_risk = {w_risk}")
    print("Calcul du plan de quantization...")
    print()
    
    plan = solve_quantization_plan(
        example_layers,
        vram_budget_gb=vram_budget,
        w_speed=1.0,
        w_risk=w_risk
    )
    
    # Affichage
    print("=== PLAN DE QUANTIZATION ===")
    for item in plan:
        print(f"{item['name'][:45]:45} → {item['dtype']:6} | "
              f"Score: {item['score']:+.3f} | VRAM: {item['vram_gb']:.4f} GB")
    
    print("\n" + "="*60)
    print(summarize(plan, vram_budget, 1.0, w_risk))
    
    # Export
    export_gguf(plan, "quant_plan_example.json")
    print("\n✅ Fichier exporté : quant_plan_example.json")
    print("Vous pouvez maintenant l'utiliser avec llama.cpp")


if __name__ == "__main__":
    main()
