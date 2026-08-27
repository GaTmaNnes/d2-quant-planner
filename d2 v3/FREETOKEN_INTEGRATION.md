# FreeToken × projet D2 — analyse d'intégration (25/08/2026)

Tout ce document cite des fichiers réels du projet ou des sorties de commandes
réellement exécutées aujourd'hui (pas d'estimation). Sources citées entre
parenthèses.

## 1. Constat de départ : le bon modèle, au bon endroit

Le premier `registry.json` consulté (`d2_ecosystem/registry.json`, généré le
22/08) décrit `Qwen3.8-27B`, un modèle **dense** (0 tenseur expert/router sur
1599). Sur cette base, FreeToken (moteur *MoE-only*) n'aurait servi à rien.

Mais le projet a évolué depuis : `models/` contient maintenant
`Qwen3.6-35B-A3B-D2-ECO.gguf`, `Qwen3.6-35B-A3B-D2-MOE.gguf` et une dizaine de
variantes DFlash — un vrai MoE. Le checkpoint HF source
(`hf_weights_35b_fp8/config.json`) le confirme :

```
"architectures": ["Qwen3_5MoeForConditionalGeneration"]
"model_type": "qwen3_5_moe"
```

FreeToken liste explicitement cette famille comme supportée
(`docs/models.md` : *"Qwen3.6 / Qwen3.5 MoE — Qwen/Qwen3.6-35B-A3B [...],
nvidia/Qwen3.6-35B-A3B-NVFP4"*), et son CLI a même un préréglage de géométrie
dédié : `ft bench bw --model qwen3.6-moe`. Ce n'est pas une correspondance
approximative — c'est le modèle nommé en dur dans leurs outils.

## 2. Ce que FreeToken apporte, concrètement

FreeToken n'est pas un concurrent de llama.cpp/llama-server en général — c'est
un moteur spécialisé dans une seule chose : servir un MoE trop gros pour la
VRAM en gardant les experts inactifs en RAM et en ne déplaçant que ce qui est
nécessaire. Sur un modèle dense (comme leur ancien 27B), il n'apporte rien.
Sur `Qwen3.6-35B-A3B` avec 8 Go de VRAM (`nvidia-smi` : 8151 MiB total), c'est
exactement le problème qu'ils ont.

Quatre backends (`ft serve --moe-backend`) :
- `offload` : experts en RAM hôte, cache LRU sur GPU, les *misses* sont
  transférés par PCIe.
- `cpu` : les *misses* sont calculés sur CPU au lieu d'être transférés.
- `hybrid` : par étape, une partie des *misses* est transférée par PCIe,
  le reste calculé sur CPU **en parallèle** (recouvrement calcul/transfert).
  Calibré une fois par machine via `ft bench bw`.
- `fused` : experts résidents en VRAM — inutile ici (ne rentre pas en 8 Go).

C'est l'équivalent — en plus fin — de leur propre `--cpu-moe`/`--n-cpu-moe`
dans `llama_ui.py`/`build_args()` : eux font un split CPU/GPU **statique** par
couche, FreeToken adapte dynamiquement le split par étape de décodage, calibré
sur la bande passante réelle de la machine.

## 3. Mesure réelle faite aujourd'hui sur leur GPU

Commande exécutée : `ft bench bw --model qwen3.6-moe` (WSL2 Ubuntu, RTX 5070
Laptop, `~/FreeToken` venv). Sortie brute :

```
host nitro   gpu cuda:0 (NVIDIA GeForce RTX 5070 Laptop GPU)   cpu 10c/7t
ceilings: CPU STREAM read 67.1  |  PCIe linear H2D 14.4  D2H 14.3  GB/s

qwen3.6-moe  H=2048 I=512 E=256 top_k=8
  format      expert       CPU-MoE   PCIe-gather  CPU/PCIe  backend
  bf16       6.00 MB     55.9 GB/s     13.2 GB/s     4.25x  hybrid
     overlapped: CPU-MoE 44.1 + PCIe 13.1 GB/s -> hybrid fetches 22.9% of misses
  nvfp4      1.69 MB     32.6 GB/s     13.1 GB/s     2.48x  hybrid
     overlapped: CPU-MoE 24.0 + PCIe 13.0 GB/s -> hybrid fetches 35.2% of misses
  fp8        3.00 MB           n/a     13.1 GB/s         —  offload
         └─ CPU MoE has no fp8_block weight path; hybrid unavailable
```

**Point critique pour ce projet précisément** : le checkpoint qu'ils ont déjà
sur disque (`hf_weights_35b_fp8/`, ~36 Go) est au format **FP8**. Or FreeToken
n'a **aucun noyau CPU pour FP8** — `hybrid` est donc indisponible pour ce
fichier précis, malgré le ratio CPU/PCIe favorable mesuré (4.25x en bf16).
Avec le FP8 existant, seul `offload` (cache LRU pur) est utilisable.

Pour profiter de `hybrid`, il faudrait soit reconvertir en bf16 (36 Go →
~70 Go, pas d'intérêt), soit récupérer le checkpoint NVFP4 déjà publié
(`nvidia/Qwen3.6-35B-A3B-NVFP4` — nettement plus petit que le FP8 actuel,
et `hybrid` fonctionne dessus, 2.48x mesuré).

## 4. Confrontation avec leurs propres mesures de routing

Le projet a déjà, de façon autonome, mesuré (puis corrigé) le comportement du
routeur MoE. Deux fichiers pertinents :

- `d2_moe_routing_real_report.json` (racine du projet) — **attention**, le nom
  contient « real » mais le champ `"method"` dit littéralement
  `"embedding_proxy (embedding(token) comme proxy hidden state)"` : ce n'est
  **pas** un forward réel à travers les 40 couches, mais une simulation
  (embedding brut du token comme proxy du hidden state). Les bannières
  `[CORRIGÉ 25/08/2026]` dans `d2_expert_cache_model.py` et
  `d2_moe_routing_tracer.py` le confirment et invalident une version encore
  plus grossière (hash MD5 des tokens) utilisée avant.
- Chiffres de ce proxy (500 tokens, 40 couches) : **256/256 experts actifs**
  (`pct_experts_active: 100.0`), chevauchement entre tokens consécutifs
  seulement **6.4%** (`avg_consecutive_overlap_pct`) — mauvais signe pour un
  cache **LRU** qui mise sur la localité temporelle immédiate.
- Mais `expert_frequencies_full` montre un déséquilibre réel non négligeable :
  l'expert le plus utilisé (57) apparaît dans **65%** des couches×tokens
  échantillonnés, contre ~3.1% attendu si le choix top-8/256 était uniforme.
  Il y a donc un déséquilibre de popularité agrégée, même si la récence
  immédiate (consécutif à consécutif) n'aide pas beaucoup un cache LRU.

**Conclusion honnête** : ces chiffres sont un proxy (pas un forward réel), à
prendre comme indication et non comme vérité mesurée à l'exécution. Mais leur
sens converge avec la mesure bande-passante indépendante ci-dessus : un cache
LRU pur (`offload`) profitera peu de la récence par token, alors que `hybrid`
(qui ne dépend pas du tout du taux de hit du cache — il calcule les misses sur
CPU au lieu d'espérer un hit) est structurellement mieux adapté à ce modèle,
**si** le format du checkpoint le permet (donc pas avec le FP8 actuel).

Le seul moyen de trancher définitivement est de mesurer le vrai routing
pendant un forward réel — ce que `d2_moe_runtime_profiler.py --bench-only`
sait déjà faire dans ce projet, et que FreeToken expose aussi nativement via
`--enable-cache-report` une fois un serveur `ft serve` lancé.

## 5. Autres apports de FreeToken indépendants du MoE

- **Cache KV sémantique** (ancrages pour éditions agentiques — tool calls,
  blocs de réflexion) et `--enable-special-token-ckpt` : orthogonal au MoE,
  pertinent pour leur intégration opencode existante
  (`llama_ui.py` → carte « Intégration opencode »). Aucun équivalent dans leur
  pipeline llama.cpp actuel.
- **Réallocation VRAM élastique** entre cache d'experts et KV
  (`--kv-reserve-tokens`, `--moe-cache-auto`) : pertinent sur une carte à 8 Go
  partagée entre modèle cible et modèle brouillon DFlash2 — problème qu'ils
  ont déjà rencontré (VRAM serrée, `-ngl auto` forcé dans `llama_ui.py` pour
  cette raison).
- **`--moe-cpu-layers`** documente explicitly le cas WSL (pinning CUDA
  plafonné par quota sous WSL) — preuve que FreeToken a été conçu/testé en
  connaissance de cause pour tourner sous WSL2, pas juste "ça devrait marcher".

## 6. Limites et ce que FreeToken NE remplace PAS

- Format d'entrée : safetensors HF (ou GGUF natif pour Gemma-4 seulement) —
  **pas** les GGUF quantifiés D2-ECO/D2-MOE qu'ils ont produits. Passer à
  FreeToken n'est pas un remplacement de leur pipeline de quantification
  GGUF, c'est un moteur de service alternatif à côté, sur le checkpoint HF
  d'origine (qu'ils ont déjà : `hf_weights_35b_fp8/`).
  Leur travail de quantification sélective (D2-ECO/BALANCED, imatrix, KLD)
  reste entièrement valable pour le pipeline llama.cpp — FreeToken a son
  propre système de format bas-débit (NVFP4/FP8/MXFP4) mais pas de sélection
  fine par tenseur comme leur `d2_tensor_optimizer.py`.
- `--moe-cache-policy` n'accepte que `lru` (pas de politique par fréquence),
  alors que leurs propres mesures (§4) suggèrent qu'une politique par
  popularité agrégée serait mieux adaptée que la seule récence.
- Le checkpoint FP8 déjà sur disque perd l'accès à `hybrid` (§3) — c'est le
  goulot le plus immédiat à lever avant de tirer une conclusion de vitesse.

## 7. Prochaine étape concrète (pas encore faite)

`d2_freetoken_bench.py` (dans ce dossier) mesure réellement, sur le serveur
FreeToken lancé en local, tokens/s et VRAM utilisée — même format de rapport
JSON que `d2_draft_bench.py`/`d2_ppl_sweep.py`. Il n'a pas encore été exécuté
avec le modèle complet (chargement d'un MoE 35B = plusieurs minutes, et le
format FP8 disponible ne permet que `offload` — voir §3) : à lancer
volontairement quand vous voulez la vraie comparaison tok/s face à
`D2-ECO.gguf`/`D2-MOE.gguf` sous llama-server.
