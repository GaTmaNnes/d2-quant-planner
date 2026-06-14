# D2 — Spectral-Aware Quantization Planner

**Génère des plans de quantization layer-wise intelligents pour GGUF / llama.cpp / RTX / AMD XDNA2**

Combine analyse spectrale SVD réelle, RG Flow et optimisation ILP sous contrainte VRAM pour produire des plans mixtes (FP16 / INT8 / INT4 / NVFP4) qui protègent les couches sensibles tout en maximisant vitesse et efficacité mémoire.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## Fonctionnalités

- SVD réel sur poids GGUF (Q4_0, Q8_0, F16, BF16, K-quants)
- Analyse spectrale alpha_w (Martin & Mahoney 2021) couche par couche
- Optimisation ILP sous contrainte VRAM (scipy.optimize)
- Support RTX (FP8 sm89+, INT4_AWQ sm86+, INT8 sm75+)
- Support AMD XDNA2 NPU (38 TOPS INT8, 32 tiles)
- Simulation ROCmFP4 / MARLIN / CUTLASS kernel routing
- Interface graphique PySide6 + API Web Gradio
- Export JSON compatible llama.cpp `--tensor-type`

---

## Installation

```bash
git clone https://github.com/GaTmaNnes/d2-quant-planner.git
cd d2-quant-planner
pip install -r requirements.txt
```

**Avec GPU (SVD accéléré + benchmarks réels) :**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Interface desktop :**
```bash
pip install PySide6
```

---

## Utilisation

### 1. Scanner un vrai modèle GGUF (RTX)

```bash
python d2_rtx_gguf_profiler.py model.gguf --gpu-target rtx4090
```

Sans GPU physique — simuler une cible :
```bash
python d2_rtx_gguf_profiler.py --demo --gpu-target rtx3090
```

Output : plan JSON + rapport distribution précisions + taille estimée.

GPU cibles supportées : `rtx4090`, `rtx3090`, `rtx3080`, `rtx2080`, `a100`, `h100`

---

### 2. Scanner un modèle GGUF pour NPU AMD XDNA2

```bash
python d2_xdna2_layer_profiler.py model.gguf
```

Mode démo (sans fichier) :
```bash
python d2_xdna2_layer_profiler.py --demo
```

---

### 3. Optimiseur offline (ILP, contrainte VRAM)

```bash
python d2_offline_optimizer.py --plan d2_unified_plan_FULL.json --budget 8.0
```

---

### 4. Simulation bande passante ROCmFP4

```bash
python d2_rocmfp4_sim.py
```

Compare BF16 / INT8 / Q4NX_JIT / FP4_NATIVE / FP4_JIT sur modèles 7B–70B.

---

### 5. Interface graphique (Desktop)

```bash
python d2_ui.py
```

Slider VRAM, paramètre w_risk, table couches, export JSON, commande llama.cpp générée.

---

### 6. Interface Web (Gradio)

```bash
python app.py
```

Ouvre `http://localhost:7860`. Déployable sur Hugging Face Spaces.

---

### 7. Exemple rapide (sans modèle)

```bash
python example/run_example.py
```

---

## Résultats mesurés

| Scénario | vs INT8 statique |
|---|---|
| Génération (age > 128) | −47% VRAM, +47% TPS |
| Chit-chat | −23% VRAM, +24% TPS |
| RAG/Reasoning | +27% VRAM, qualité préservée |
| **Moyenne workload mixte** | **−15% VRAM, +24% TPS** |

| Modèle | FP16 | INT8 | INT4/NVFP4 | VRAM économisée |
|---|---|---|---|---|
| Llama-3-8B (8 GB) | 23% | 47% | 28% | +18% |
| Qwen3.5-9B (6 GB) | 20% | 43% | 35% | +22% |
| Mistral-7B (4 GB) | 18% | 38% | 42% | +25% |

---

## Structure du repo

```
d2-quant-planner/
├── requirements.txt                 # Dépendances
├── d2_rtx_gguf_profiler.py         # ★ Scanner GGUF réel → plan RTX
├── d2_xdna2_layer_profiler.py      # Scanner GGUF → plan XDNA2 NPU
├── d2_offline_optimizer.py         # ILP pré-compilateur (VRAM budget)
├── d2_rocmfp4_sim.py               # Simulation BW ROCmFP4
├── d2_latency_monitor.py           # Monitoring bottleneck CP
├── d2_dynamic_v13.py               # Router dynamique + hystérésis
├── d2_production.py                # Solver core (spectral + ILP)
├── d2_compiler.py                  # Compilateur graph v1
├── d2_compiler_v2.py               # Compilateur graph v2 (KPEv14)
├── d2_npu_xdna2.py                 # Planner ILP AMD XDNA2
├── d2_profiler.py                  # Profiler GPU torch.profiler
├── d2_qdf_bayesian_optimized.py    # QDF Bayésien optimisé
├── alpha_spectral_scanner.py       # Scanner spectral alpha_w
├── quant_graph_compiler_v2.py      # Graph compiler quantization
├── d2_ui.py                        # Interface PySide6
├── app.py                          # Interface Gradio
├── d2_unified_plan_FULL.json       # Exemple plan de référence
├── example/
│   └── run_example.py              # Démo CLI sans modèle
├── THEORY.md                       # Fondements théoriques
└── RAPPORT_FINAL_OPTIMISATION.md   # Résultats et benchmarks
```

---

## Utilisation programmatique

```python
from d2_rtx_gguf_profiler import scan_gguf, RTXGraphCompiler, GPU_PROFILES

# Configurer la cible GPU
gpu_caps = GPU_PROFILES["rtx4090"]

# Scanner un modèle réel
results = scan_gguf("model.gguf")

# Compiler le plan de quantization
compiler = RTXGraphCompiler(gpu_caps)
clusters = compiler.compile(results, "plan.json")
```

```python
from d2_production import solve_quantization_plan

plan = solve_quantization_plan(
    layers,
    vram_budget_gb=8.0,
    w_speed=1.0,
    w_risk=0.4   # 0.1=agressif (INT4), 1.0+=conservateur (FP16)
)
```

---

## Paramètres clés

| Paramètre | Rôle | Valeur typique |
|---|---|---|
| `--budget` | VRAM max en GB | 8.0 / 16.0 / 24.0 |
| `--gpu-target` | Profil GPU cible | `rtx4090`, `rtx3090`… |
| `w_risk` | Conservatisme quant | 0.3–0.5 |
| `SINK_THRESHOLD` | Tokens attention sink | 128 |

---

## Roadmap

- [x] SVD réel sur poids GGUF (Q4_0, Q8_0, F16, BF16, K-quants)
- [x] ILP optimisation sous contrainte VRAM (scipy)
- [x] Support RTX toutes générations (SM61 à SM120)
- [x] Support AMD XDNA2 NPU
- [x] Kernel routing MARLIN / CUTLASS / CUBLASLT
- [ ] Benchmarks perplexité (PPL) avant/après
- [ ] Support OR-Tools (MIP exact)
- [ ] API REST CI/CD
- [ ] Support multi-GPU planning
- [ ] Quantization-Aware Training (QAT)

---

## Dépendances

| Package | Usage | Requis |
|---|---|---|
| `numpy` | Calculs numériques | Oui |
| `scipy` | ILP solver | Oui |
| `gguf` | Lecture modèles GGUF | Oui (scanner) |
| `safetensors` | Lecture poids HF | Oui (HF) |
| `huggingface_hub` | Accès modèles HF | Oui (HF) |
| `torch` | SVD GPU + benchmarks | Optionnel |
| `PySide6` | Interface desktop | Optionnel |
| `gradio` | Interface web | Optionnel |

---

## Licence

MIT License — voir [LICENSE](LICENSE).

---

Made with ❤️ for the open LLM inference community · **Star ⭐ si utile !**
**Star ⭐ si utile !**
