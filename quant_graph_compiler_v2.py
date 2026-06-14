import json
import networkx as nx
import numpy as np

class QuantGraphCompilerV2:
    """
    Compilateur de Graphe de Quantification V2 (Graph IR + GPU Cost Model).
    Intègre les contraintes de forme, l'alignement et les barrières de switch.
    """

    def __init__(self, map_v3="precision_map_v3.json"):
        with open(map_v3, "r") as f:
            self.raw_data = json.load(f)
        self.G = nx.DiGraph()
        
        # Coûts relatifs (Compute + Latence)
        self.precision_costs = {
            "NVFP4_SAFE": 0.2,    # Très rapide (Hardware native)
            "INT4_AWQ": 0.4,      # Logiciel simulé
            "INT8_SAFE": 0.6,
            "FP8": 0.7,
            "FP16_REQUIRED": 1.0  # Référence
        }
        
        # Pénalités
        self.SWITCH_PENALTY = 0.5  # Coût d'un switch de précision (Memory Roundtrip)
        self.ALIGN_PENALTY = 0.3   # Coût d'un désalignement (Padding/Speed drop)

    def build_graph(self):
        print("🏗️ Pass 1: Generating Quantization Graph IR...")
        nodes = list(self.raw_data.keys())
        
        def get_sort_key(name):
            parts = name.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                return (int(parts[1]), name)
            return (0, name)
        
        sorted_nodes = sorted(nodes, key=get_sort_key)
        
        prev_node = None
        for name in sorted_nodes:
            data = self.raw_data[name]
            shape = data["shape"]
            
            # Calcul du score d'alignement (multiple de 32 pour Blackwell/Pascal)
            is_aligned = (shape[-1] % 32 == 0)
            
            # Initialisation avec la recommandation Alpha brute
            # Mais on ajoute un biais pour Blackwell
            alpha = data["alpha_w"]
            density = data["density"]
            
            # Politique initiale (NVIDIA Policy)
            if alpha > 1.8 and not (density > 0.3):
                init_p = "NVFP4_SAFE"
            elif alpha > 1.4:
                init_p = "INT8_SAFE"
            else:
                init_p = "FP16_REQUIRED"

            self.G.add_node(name, 
                           precision=init_p,
                           alpha=alpha,
                           density=density,
                           shape=shape,
                           is_aligned=is_aligned,
                           op_class=data["op_class"])
            
            if prev_node:
                self.G.add_edge(prev_node, name)
            prev_node = name

    def optimize(self):
        """Optimisation globale minimisant Latency = sum(compute) + sum(switches)."""
        print("💰 Pass 2: Optimizing GPU Cost Model (Global Optimization)...")
        nodes = list(self.G.nodes)
        
        # Approche glissante pour le clustering
        for i in range(1, len(nodes)):
            curr_name = nodes[i]
            prev_name = nodes[i-1]
            
            curr = self.G.nodes[curr_name]
            prev = self.G.nodes[prev_name]
            
            # Calcul du coût SI on switche vs SI on unifie
            cost_switch = self.precision_costs[curr["precision"]] + self.SWITCH_PENALTY
            
            # Coût d'unification (on prend la précision la plus sûre des deux)
            # Ranks: NVFP4=0, INT8=1, FP16=2
            ranks = {"NVFP4_SAFE": 0, "INT8_SAFE": 1, "FP16_REQUIRED": 2}
            unified_p = prev["precision"] if ranks[prev["precision"]] > ranks[curr["precision"]] else curr["precision"]
            cost_unified = self.precision_costs[unified_p]
            
            # Si le switch coûte plus cher que l'unification, on unifie le domaine
            if cost_switch > cost_unified:
                self.G.nodes[curr_name]["precision"] = unified_p

    def check_alignment(self):
        """Pass 3: Legalization Pass (Alignement physique)."""
        print("⚖️ Pass 3: Legalization & Shape Alignment...")
        for node in self.G.nodes:
            n = self.G.nodes[node]
            if not n["is_aligned"] and n["precision"] == "NVFP4_SAFE":
                # Le FP4 sur Blackwell demande un alignement strict
                # Si désaligné, on fallback en INT8 (plus tolérant)
                n["precision"] = "INT8_SAFE"

    def compile(self, output_path="final_sm120_graph_plan.json"):
        self.build_graph()
        self.optimize()
        self.check_alignment()
        
        final_map = {}
        clusters = []
        curr_c = {"p": None, "layers": []}
        
        for node in self.G.nodes:
            n = self.G.nodes[node]
            final_map[node] = {
                "precision": n["precision"],
                "alpha_w": n["alpha"],
                "op_class": n["op_class"],
                "shape": n["shape"]
            }
            
            if n["precision"] != curr_c["p"]:
                if curr_c["layers"]: clusters.append(curr_c)
                curr_c = {"p": n["precision"], "layers": [node]}
            else:
                curr_c["layers"].append(node)
        
        if curr_c["layers"]: clusters.append(curr_c)

        with open(output_path, "w") as f:
            json.dump(final_map, f, indent=2)
            
        print(f"\n✅ Compilation IR Graph V2 terminée.")
        print(f"📦 Modèle structuré en {len(clusters)} clusters d'exécution.")
        return clusters

if __name__ == "__main__":
    compiler = QuantGraphCompilerV2()
    clusters = compiler.compile()
    
    print("\n--- PLAN D'EXÉCUTION BLACKWELL OPTIMISÉ (V2) ---")
    for i, c in enumerate(clusters):
        print(f"Cluster {i+1:02} | {c['p']:15} | {len(c['layers']):3} couches (Type: {compiler.G.nodes[c['layers'][0]]['op_class']})")
