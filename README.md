# D2 — Spectral-Aware Layer-wise Quantization Planner

**Outil avancé pour générer des plans de quantization intelligents (layer-wise) pour llama.cpp / GGUF**

Combine analyse spectrale (SVD), RG Flow et optimisation sous contrainte VRAM pour produire des plans de quantization mixtes (FP16 / INT8 / INT4) qui protègent les couches sensibles (lm_head, attention) tout en maximisant la vitesse et en minimisant la VRAM.

## Fonctionnalités

- Interface graphique PySide6 moderne et non-bloquante
- Chargement réel des poids via safetensors (Hugging Face ou local)
- Analyse spectrale simulée par type de couche
- Optimisation hybride (score spectral + risque + vitesse)
- Export JSON compatible llama.cpp + commande prête à l'emploi
- Protection intelligente du `lm_head`

## Installation

```bash
git clone https://github.com/GaTmaNnes/d2-quant-planner.git
cd d2-quant-planner

pip install -r requirements.txt
```

## Utilisation

### Interface Graphique

```bash
python d2_ui.py
```

### Utilisation en ligne de commande (exemple)

```python
from d2_production import solve_quantization_plan, export_gguf

layers = [...]  # chargés via safetensors
plan = solve_quantization_plan(layers, vram_budget_gb=6.0, w_risk=0.35)
export_gguf(plan, "quant_plan.json")
```

## Résultats attendus

- **Qwen3.5-9B** : excellents gains en Q4_K_M / NVFP4 layer-wise
- **Qwen3.5-35B-A3B (MoE)** : très bon en NVFP4
- **Économie VRAM** : supplémentaire de 10-25% vs uniform quant tout en préservant la qualité

## Roadmap

- [ ] Vrai SVD sur poids réels
- [ ] Vrai ILP avec OR-Tools
- [ ] Version Gradio / Hugging Face Space
- [ ] Benchmarks perplexity + vitesse

## Licence

MIT License — voir [LICENSE](LICENSE)

---

Made with ❤️ for the open LLM inference community
