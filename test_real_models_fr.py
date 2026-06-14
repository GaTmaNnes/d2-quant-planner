#!/usr/bin/env python3
"""
D2 Quantization Planner — Suite de Tests Réels (VERSION FRANÇAISE)

Teste D2 sur les vrais modèles HuggingFace :
- GPT-2 (petit, 124M paramètres)
- TinyLlama (1.1B)
- Llama 2 7B (si disponible)
- Modèles Qwen

Mesure :
- Exposant spectral (α_w) par couche
- Composantes du C-Vector
- Type de quantization recommandé
- Estimation utilisation VRAM
- Métriques de fragmentation
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from typing import Dict, List, Tuple
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE SPECTRALE (depuis spectral_theory.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_alpha_w(W: np.ndarray) -> float:
    """Calcule l'exposant spectral α_w à partir de la matrice de poids."""
    try:
        U, s, Vt = svd(W, full_matrices=False)
    except:
        return 2.0
    
    s = s[s > 0.01 * s[0]]
    if len(s) < 2:
        return 2.0
    
    log_i = np.log(np.arange(1, len(s) + 1))
    log_s = np.log(s)
    
    try:
        coeffs = np.polyfit(log_i, log_s, 1)
        return float(max(-2.0 * coeffs[0], 1.0))
    except:
        return 2.0


def compute_stable_rank(W: np.ndarray) -> float:
    """Calcule le rang stable r_s = ||W||_F² / ||W||_2²."""
    try:
        U, s, Vt = svd(W, full_matrices=False)
        return float(np.sum(s**2) / (s[0]**2 + 1e-10))
    except:
        return 1.0


def compute_spectral_entropy(W: np.ndarray) -> float:
    """Calcule l'entropie spectrale H = -Σ p_i log(p_i)."""
    try:
        U, s, Vt = svd(W, full_matrices=False)
        s = s[s > 0.01 * s[0]]
        p = s**2 / np.sum(s**2)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))
    except:
        return 1.0


def compute_spectral_radius(W: np.ndarray) -> float:
    """Calcule le rayon spectral ρ = plus grande valeur singulière."""
    try:
        U, s, Vt = svd(W, full_matrices=False)
        return float(s[0] if len(s) > 0 else 1.0)
    except:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# POLITIQUE DE QUANTIZATION (Corrigée)
# ─────────────────────────────────────────────────────────────────────────────

def quantization_policy(alpha_w: float, density: float) -> str:
    """
    Mapping de quantization corrigé.
    
    instabilité = α_w (1 + ρ_d)
    
    Instabilité FAIBLE → NVFP4 (compressible)
    Instabilité ÉLEVÉE → FP16 (sensible)
    """
    instability = alpha_w * (1.0 + density)
    
    if instability < 1.2:
        return "NVFP4"
    elif instability < 1.6:
        return "INT8"
    elif instability < 2.0:
        return "FP8"
    else:
        return "FP16"


# ─────────────────────────────────────────────────────────────────────────────
# COÛTS DTYPE (VRAM + Temps)
# ─────────────────────────────────────────────────────────────────────────────

DTYPE_INFO = {
    "FP16": {"bytes": 2.0, "tps_gain": 1.0},
    "FP8": {"bytes": 1.0, "tps_gain": 1.5},
    "INT8": {"bytes": 1.0, "tps_gain": 1.35},
    "NVFP4": {"bytes": 0.5, "tps_gain": 1.75},
}


def estimate_layer_vram(shape: Tuple, dtype: str) -> float:
    """Estime l'utilisation VRAM en GB."""
    numel = np.prod(shape)
    bytes_per_param = DTYPE_INFO[dtype]["bytes"]
    return (numel * bytes_per_param) / (1024**3)


# ─────────────────────────────────────────────────────────────────────────────
# FRAMEWORK DE TEST DE MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

class AnalyseurD2:
    """Analyse un modèle pour la planification de quantization."""
    
    def __init__(self, nom_modele: str, device: str = "cpu"):
        """
        Initialise l'analyseur.
        
        Args:
            nom_modele: Identifiant de modèle HuggingFace
            device: CPU ou CUDA
        """
        self.nom_modele = nom_modele
        self.device = device
        self.modele = None
        self.tokenizer = None
        self.analyses_couches = []
    
    def charger_modele(self):
        """Charge le modèle depuis HuggingFace."""
        print(f"📥 Chargement de {self.nom_modele}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.nom_modele)
            self.modele = AutoModelForCausalLM.from_pretrained(
                self.nom_modele,
                torch_dtype=torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            total_params = sum(p.numel() for p in self.modele.parameters())
            print(f"✓ Modèle chargé : {total_params / 1e6:.1f}M paramètres")
        except Exception as e:
            print(f"❌ Erreur de chargement : {e}")
            raise
    
    def analyser_couches(self) -> List[Dict]:
        """
        Analyse tous les poids des couches.
        
        Returns:
            Liste de dictionnaires d'analyse de couches
        """
        print("🔍 Analyse des couches en cours...")
        self.analyses_couches = []
        
        total_params = 0
        nombre_couches = 0
        
        for nom, param in self.modele.named_parameters():
            # Ignore les biais et normalisation
            if "bias" in nom or "norm" in nom or "ln_" in nom:
                continue
            
            # Analyse seulement les matrices de poids
            if param.dim() < 2:
                continue
            
            nombre_couches += 1
            W = param.data.detach().cpu().numpy()
            
            # Reshape si nécessaire
            if W.ndim > 2:
                W = W.reshape(W.shape[0], -1)
            
            # Saute les petites couches
            if W.shape[0] < 8 or W.shape[1] < 8:
                continue
            
            numel = np.prod(W.shape)
            total_params += numel
            
            # Calcule les propriétés spectrales
            alpha_w = compute_alpha_w(W)
            rang_stable = compute_stable_rank(W)
            entropie = compute_spectral_entropy(W)
            rho = compute_spectral_radius(W)
            
            # Densité (sparsité)
            densite = float(np.count_nonzero(W) / W.size) if W.size > 0 else 0.0
            
            # Décision de quantization
            dtype = quantization_policy(alpha_w, densite)
            
            # Estimation VRAM
            vram_fp16 = estimate_layer_vram(W.shape, "FP16")
            vram_dtype = estimate_layer_vram(W.shape, dtype)
            
            info_couche = {
                "nom": nom,
                "forme": W.shape,
                "parametres": int(numel),
                "alpha_w": float(alpha_w),
                "rang_stable": float(rang_stable),
                "entropie": float(entropie),
                "rho": float(rho),
                "densite": float(densite),
                "dtype_recommande": dtype,
                "vram_fp16_mb": float(vram_fp16 * 1024),
                "vram_dtype_mb": float(vram_dtype * 1024),
                "economie_vram_pct": float(100 * (1 - vram_dtype / vram_fp16)) if vram_fp16 > 0 else 0.0,
            }
            
            self.analyses_couches.append(info_couche)
        
        print(f"✓ {nombre_couches} couches analysées, {total_params / 1e6:.1f}M paramètres totaux")
        return self.analyses_couches
    
    def calculer_statistiques(self) -> Dict:
        """Calcule les statistiques d'ensemble."""
        if not self.analyses_couches:
            return {}
        
        alpha_ws = [l["alpha_w"] for l in self.analyses_couches]
        entropies = [l["entropie"] for l in self.analyses_couches]
        vrays = [l["vram_dtype_mb"] for l in self.analyses_couches]
        
        # Distribution dtype
        distribution_dtype = {}
        for couche in self.analyses_couches:
            dtype = couche["dtype_recommande"]
            distribution_dtype[dtype] = distribution_dtype.get(dtype, 0) + 1
        
        # Fragmentation (couches consécutives avec dtypes différents)
        fragmentation = 0
        for i in range(len(self.analyses_couches) - 1):
            if self.analyses_couches[i]["dtype_recommande"] != self.analyses_couches[i+1]["dtype_recommande"]:
                fragmentation += 1
        
        return {
            "nombre_couches": len(self.analyses_couches),
            "alpha_w_moyenne": float(np.mean(alpha_ws)),
            "alpha_w_ecart_type": float(np.std(alpha_ws)),
            "alpha_w_min": float(np.min(alpha_ws)),
            "alpha_w_max": float(np.max(alpha_ws)),
            "entropie_moyenne": float(np.mean(entropies)),
            "entropie_ecart_type": float(np.std(entropies)),
            "distribution_dtype": distribution_dtype,
            "vram_totale_mb": float(np.sum(vrays)),
            "score_fragmentation": float(fragmentation / max(1, len(self.analyses_couches) - 1)),
            "evenements_fragmentation": int(fragmentation),
        }
    
    def generer_rapport(self, dossier_sortie: str = "resultats"):
        """Génère un rapport d'analyse."""
        Path(dossier_sortie).mkdir(exist_ok=True)
        
        # Analyse complète
        rapport = {
            "modele": self.nom_modele,
            "couches": self.analyses_couches,
            "statistiques": self.calculer_statistiques(),
        }
        
        fichier_sortie = Path(dossier_sortie) / f"{self.nom_modele.replace('/', '_')}_analyse.json"
        with open(fichier_sortie, "w", encoding="utf-8") as f:
            json.dump(rapport, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Rapport sauvegardé : {fichier_sortie}")
        return rapport
    
    def afficher_resume(self):
        """Affiche un résumé lisible."""
        stats = self.calculer_statistiques()
        
        print("\n" + "=" * 80)
        print(f"📊 RÉSUMÉ D'ANALYSE D2 — {self.nom_modele}")
        print("=" * 80)
        print(f"Couches analysées : {stats['nombre_couches']}")
        print(f"α_w : {stats['alpha_w_min']:.2f} — {stats['alpha_w_max']:.2f} (moyenne : {stats['alpha_w_moyenne']:.2f})")
        print(f"Entropie : {stats['entropie_moyenne']:.2f} ± {stats['entropie_ecart_type']:.2f}")
        print(f"VRAM totale (quantifiée) : {stats['vram_totale_mb'] / 1024:.2f} GB")
        print(f"Fragmentation : {stats['score_fragmentation']:.2%} ({stats['evenements_fragmentation']} événements)")
        
        print("\n📈 Distribution des types de quantization :")
        for dtype in sorted(stats['distribution_dtype'].keys()):
            count = stats['distribution_dtype'][dtype]
            pct = 100 * count / stats['nombre_couches']
            print(f"  {dtype:6} : {count:3d} couches ({pct:5.1f}%)")
        
        # Économies VRAM
        vram_fp16_total = sum(l["vram_fp16_mb"] for l in self.analyses_couches)
        economie_pct = 100 * (1 - stats['vram_totale_mb'] / vram_fp16_total) if vram_fp16_total > 0 else 0
        print(f"\n💾 Économies VRAM : {economie_pct:.1f}% (FP16 baseline)")
        
        print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# SUITE DE TESTS
# ─────────────────────────────────────────────────────────────────────────────

MODELES_A_TESTER = [
    "gpt2",
    "distilgpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


def executer_suite_tests(modeles: List[str] = None, device: str = "cpu"):
    """Lance l'analyse sur plusieurs modèles."""
    if modeles is None:
        modeles = MODELES_A_TESTER
    
    resultats = {}
    
    for nom_modele in modeles:
        print(f"\n\n{'🔬 ' * 25}")
        print(f"Test en cours : {nom_modele}")
        print(f"{'🔬 ' * 25}\n")
        
        try:
            analyseur = AnalyseurD2(nom_modele, device=device)
            analyseur.charger_modele()
            analyseur.analyser_couches()
            analyseur.afficher_resume()
            resultats[nom_modele] = analyseur.generer_rapport()
        
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse de {nom_modele} : {e}")
            resultats[nom_modele] = {"erreur": str(e)}
    
    # Résumé comparatif
    print("\n\n" + "=" * 80)
    print("📋 RÉSUMÉ COMPARATIF")
    print("=" * 80)
    
    for nom_modele, rapport in resultats.items():
        if "erreur" in rapport:
            print(f"{nom_modele}: ❌ ÉCHEC")
        else:
            stats = rapport["statistiques"]
            print(f"\n{nom_modele}:")
            print(f"  α_w : {stats['alpha_w_moyenne']:.2f} ± {stats['alpha_w_ecart_type']:.2f}")
            print(f"  VRAM : {stats['vram_totale_mb'] / 1024:.2f} GB")
            print(f"  Fragmentation : {stats['score_fragmentation']:.2%}")
            
            # Top 3 couches les plus instables
            couches_instables = sorted(
                rapport['couches'],
                key=lambda x: x['alpha_w'] * (1 + x['densite']),
                reverse=True
            )[:3]
            print(f"  ⚠️  Couches critiques (top 3) :")
            for i, couche in enumerate(couches_instables, 1):
                instabilite = couche['alpha_w'] * (1 + couche['densite'])
                print(f"     {i}. {couche['nom'][:50]:50} → {couche['dtype_recommande']:6} (instabilité={instabilite:.2f})")
    
    print("=" * 80)
    
    return resultats


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE LIGNE DE COMMANDE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Suite de Tests de Modèles D2")
    parser.add_argument("--modele", type=str, help="Modèle spécifique à tester")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tous", action="store_true", help="Tester tous les modèles par défaut")
    
    args = parser.parse_args()
    
    if args.modele:
        modeles = [args.modele]
    elif args.tous:
        modeles = MODELES_A_TESTER
    else:
        # Par défaut : test GPT-2 et DistilGPT-2 (rapides)
        modeles = ["gpt2", "distilgpt2"]
    
    print("=" * 80)
    print("🚀 D2 QUANTIZATION PLANNER — SUITE DE TESTS FRANÇAISE")
    print("=" * 80)
    print(f"Device : {args.device}")
    print(f"Modèles : {', '.join(modeles)}")
    print("=" * 80)
    
    executer_suite_tests(modeles, device=args.device)
    
    print("\n" + "=" * 80)
    print("✅ TESTS TERMINÉS")
    print("=" * 80)
    print("📁 Résultats sauvegardés dans le dossier 'resultats/'")
    print("=" * 80)
