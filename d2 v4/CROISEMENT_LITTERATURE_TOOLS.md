# D2 × Littérature × FreeToken — croisement consolidé (26/08/2026)

Sources citées en clair (pas de lien masqué). Chaque outil externe est confronté
aux outils D2 déjà présents dans `d2 corrige/`, avec un verdict d'action.

---

## 1. Outils de quantification & profiling externes vs D2

| Outil | URL | Équivalent D2 | Verdict |
|---|---|---|---|
| llama.cpp (officiel) | https://github.com/ggml-org/llama.cpp | beellama.cpp (fork Anbeeld) | base commune ; DFlash/MTP/TurboQuant en plus |
| ik_llama.cpp (quants IQK, CPU/hybrid) | https://github.com/ikawrakow/ik_llama.cpp | K-quants SPEED3 (Q2_K..Q6_K) | ⚠️ levier non exploité — CPU-bound = notre goulot |
| Unsloth Dynamic GGUFs | https://unsloth.ai/docs/basics/dynamic-3.0-ggufs | d2_multicriteria_planner.py | même idée, D2 plus fin (5 axes mesurés) |
| moe-scalpel (pruning experts) | https://github.com/lucafulgenzi/moe-scalpel | calibration réelle routing | croisement : 1.4 % experts froids → pruning inutile |
| llama-quantize-cost (MSE/tenseur) | https://github.com/jimbothigpen/llama-quantize-cost | d2_kld_real_profiler + precision_map_diag | équivalent déjà construit |
| llama-optimize (auto-params) | https://github.com/bigattichouse/llama-optimize | sweeps ngl/KV/MTP | équivalent |
| imatrix (importance activation) | https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md | calibration gate (activation réelle) | ⚠️ axe à connecter au planner |
| Custom quant mixes (tensor-type/IQK) | https://github.com/ggml-org/llama.cpp/discussions/12741 · https://www.mintlify.com/ikawrakow/ik_llama.cpp/quantization/iqk-quants | --tensor-type-file (speed_types_*.txt) | IQK en piste pour couches CPU |
| Target-bpw | https://github.com/ggml-org/llama.cpp/pull/15550 · https://github.com/ggml-org/llama.cpp/discussions/15576 | planner budget → niveaux | validation externe de l'approche |
| Guide MoE offload | https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide | sweeps ngl + --cpu-moe | validé (cliff ngl17, CPU-bound) |
| Expert profiling REAP-style | https://github.com/ggml-org/llama.cpp/pull/20454 | d2_moe_routing_tracer (proxy) | manque le forward réel |
| Layer bottleneck profiles | https://huggingface.co/datasets/sjakek/qwen36-coding-layer-bottleneck-profiles | d2_35b_allocation / roofline | comparaison possible |
| Beyond Perplexity (KV mixte) | https://www.qeios.com/read/RGD04F | KV q4_0, K-q8/V-q4 | ✅ valide notre découverte (K>V sensible) |
| Unsloth Dynamic NVFP4 (Blackwell) | https://unsloth.ai/docs/basics/nvfp4 | precision_map NVFP4_SAFE + FreeToken nvfp4 | ✅ notre GPU = Blackwell sm_120 → voie FreeToken hybrid |

## 2. Faits mesurés du projet (croisement littérature)

| Littérature | Mesure D2 réelle (26/08) |
|---|---|
| MoEQuant (ICML 2025) : sensibilité ≠ par expert | top-32 ≈ 18 %, top-128 ≈ 58 % (calibration 3000 tokens réels) |
| AWQ : activations, pas seulement poids | calibration gate réelle = l'axe activation manquant à imatrix |
| REAP : pruning par usage | 142/10240 experts froids (1.4 %) → peu de gras, identifiables |
| IMPQ (2025) : interactions inter-couches | non couvert par le planner actuel (manuel par tenseur) |
| MXSens / SFMoE (2026) | direction confirmée : sensibilité mesurée, pas uniforme |

## 3. Conséquences pour FreeToken × Windows

- Pruning mort (1.4 % froids) → pas de gain RAM par experts
- FP8 36 Go > RAM WSL2 15 Go → offload infaisable sur ce PC
- **NVFP4 natif Blackwell ≈ 18 Go** → tient en RAM WSL2 + backend `hybrid` disponible
  (mesure FreeToken du 25/08 : CPU/PCIe 2.48× ; bf16 4.25× mais pas de CPU-FP8)
- LRU inadapté (overlap consécutif 6.4 %) → `hybrid` est le seul backend qui ne
  dépend pas des hits de cache → choisi pour la voie Windows

## 4. Plan d'action (tous les outils utilisés)

1. **Corriger le mapping FP8→GGUF** du profiler KLD (safetensors NVFP4-style)
2. **Carte de sensibilité par expert** sur les safetensors FP8 (40×256×{gate,up,down})
3. **Croiser carte + calibration réelle** → classes robuste/sensible par expert
4. **Émettre un checkpoint D2 mixte FP8/NVFP4 en safetensors** → format FreeToken
5. **Bench FreeToken hybrid** sur ce checkpoint (WSL2, NVFP4 tient en RAM)
6. **Tester ik_llama IQK** sur les couches CPU si le temps le permet