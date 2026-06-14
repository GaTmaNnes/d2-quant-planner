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
git clone https://github.com/tonusername/d2-quant-planner.git
cd d2-quant-planner

pip install -r requirements.txt
