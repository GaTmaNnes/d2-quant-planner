import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

import ttkbootstrap as tb
from tkinter import filedialog, messagebox

if getattr(sys, "frozen", False):
    # PyInstaller : __file__ pointe vers le dossier d'extraction temporaire,
    # pas vers l'emplacement réel du .exe. Il faut utiliser sys.executable
    # pour retrouver llama-server.exe posé à côté du lanceur compilé.
    ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    ROOT = os.path.dirname(os.path.abspath(__file__))

SERVER_EXE = os.path.join(ROOT, "llama-server.exe")
MULTI_GPU_DIR = os.path.join(ROOT, "multi-gpu")
# [CORRIGÉ 25/08/2026] Ajout du build fixé DFlash : le binaire racine llama-server.exe
# est un build du 23/08 SANS fixes DFlash (crash draft « invalid vector subscript ») ;
# le SEUL binaire DFlash fonctionnel est beellama.cpp/build-cuda/bin/llama-server.exe
# (cudart64_13.dll copiée à côté, self-contained).
BUILD_CUDA_DIR = os.path.join(ROOT, "beellama.cpp", "build-cuda", "bin")
BUILDS = {
    "standard": {"label": "Standard (CUDA)", "dir": ROOT},
    "multi-gpu": {"label": "Multi-GPU (CUDA + Vulkan)", "dir": MULTI_GPU_DIR},
    "build-cuda": {"label": "build-cuda (fixé DFlash 24/08)", "dir": BUILD_CUDA_DIR},
}
MODELS_DIR = os.path.join(ROOT, "models")
DFLASH_MAX_CTX = 8192          # [CORRIGÉ 25/08/2026] DFlash crash si ctx > 8192 (-c 8192 validé)
# [CORRIGÉ 25/08/2026] Draft par défaut = OFFICIAL-BF16 (seul draft validé : 90,9% acceptance).
# Les drafts « D2FIX » produits par les patcheurs maison sont corrompus.
DEFAULT_DRAFT = os.path.join(MODELS_DIR, "Qwen3.6-35B-A3B-DFlash-OFFICIAL-BF16.gguf")
CONFIG_PATH = os.path.join(ROOT, "lama_ui_config.json")
DEFAULT_PORT = 8080
METRICS_URL = "http://127.0.0.1:8765/metrics"

DARK_THEME = "tokyo-night-dark"
LIGHT_THEME = "tokyo-night-light"

# BUG (terminal qui s'ouvre/se coupe) : sans ce flag, chaque subprocess.run /
# Popen lancé depuis une appli Tk/pythonw fait apparaitre puis disparaitre
# une fenetre console noire. query_gpu_memory() etant appelee toutes les 3s
# via _vram_poll(), ca donnait un clignotement recurrent en boucle tant que
# l'appli tournait. CREATE_NO_WINDOW n'existe que sur Windows.
CREATIONFLAGS_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# Bornes du widget de log (voir _append) : sans plafond, une session longue
# avec un serveur verbeux fait grossir le Text tkinter indefiniment -> c'est
# la fuite de memoire principale de l'appli.
LOG_MAX_LINES = 5000
LOG_TRIM_TO = 4000


def _classify_gpu(name, cap):
    if cap.startswith("10.") or cap.startswith("12."):
        return {"name": name, "cache": "turbo4", "ctx": 32768, "cap": cap}
    if cap.startswith("6."):
        return {"name": name, "cache": "turbo3", "ctx": 16384, "cap": cap}
    return {"name": name, "cache": "turbo3", "ctx": 16384, "cap": cap}


def detect_gpu():
    info = {"name": "CPU (aucun GPU NVIDIA)", "cache": "", "ctx": 4096, "cap": "", "count": 0, "index": None}
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATIONFLAGS_NO_WINDOW,
        ).stdout.strip().splitlines()
        gpus = []
        for line in raw:
            line = line.replace("\r", "").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            g = _classify_gpu(parts[1].strip(), parts[2].strip())
            g["index"] = int(parts[0].strip())
            gpus.append(g)
        if gpus:
            # Choisit le GPU le plus capable (compute capability la plus haute), pas juste
            # le premier renvoyé par nvidia-smi (ordre PCI bus id, pas ordre de performance).
            best = max(gpus, key=lambda g: tuple(int(x) for x in g["cap"].split(".") if x.isdigit()))
            best["count"] = len(gpus)
            info = best
    except Exception:
        pass
    return info


def query_gpu_memory(index=0):
    """Retourne (utilisé_MiB, total_MiB) du GPU d'index `index`, ou None si indisponible.

    `index` doit correspondre au meme GPU que celui choisi par detect_gpu() : sur une
    machine multi-GPU, interroger toujours le premier (raw[0]) alors que detect_gpu()
    a pu selectionner un autre GPU (le plus capable) ferait afficher la VRAM du
    mauvais GPU dans l'UI.
    """
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=CREATIONFLAGS_NO_WINDOW,
        ).stdout.strip().splitlines()
        for line in raw:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            if int(parts[0]) == index:
                return float(parts[1]), float(parts[2])
    except Exception:
        pass
    return None


def query_local_metrics():
    """Interroge le capteur de telemetrie local (bande passante NPU, goulot,
    charge CPU/GPU/RAM) s'il tourne sur ce poste. Renvoie None silencieusement
    s'il est absent (machine sans ce capteur, ou service pas demarre) : cette
    carte doit degrader proprement, pas faire planter l'UI."""
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=1.0) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def find_models():
    found = []
    for base in (MODELS_DIR, ROOT):
        if os.path.isdir(base):
            for f in sorted(os.listdir(base)):
                fl = f.lower()
                if not fl.endswith(".gguf"):
                    continue
                # Exclut les fichiers GGUF qui ne sont pas des modeles chargeables par
                # llama-server (imatrix genere par llama-imatrix : meme extension .gguf
                # mais schema different -> le selectionner et lancer plante le serveur).
                if "imatrix" in fl:
                    continue
                found.append(os.path.join(base, f))
    return found


def _file_gb(path):
    """Taille d'un fichier en Go, ou None si inaccessible."""
    try:
        return os.path.getsize(path) / (1024 ** 3)
    except OSError:
        return None


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(data):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


class LlamaUI(tb.Window):
    def __init__(self):
        self.cfg = load_config()
        theme = self.cfg.get("theme", DARK_THEME)
        super().__init__(title="LAMA TOUR — Lanceur GGUF", themename=theme,
                          size=(1000, 900), minsize=(820, 640))

        self.gpu = detect_gpu()
        self.proc = None
        self.launch_time = None
        self.web_auto_opened = False
        self._model_display_map = {}

        self._build()
        self._vram_poll()
        self._oc_poll()
        self._uptime_tick()
        self._metrics_poll()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        if not os.path.isfile(self.server_exe()):
            messagebox.showwarning(
                "llama-server.exe introuvable",
                f"Le binaire serveur est introuvable :\n{self.server_exe()}\n\n"
                "Placez llama-server.exe à côté de ce programme avant de lancer.",
            )
            self.btn_launch.configure(state="disabled")

    # ------------------------------------------------------------------ UI

    def _build(self):
        header = tb.Frame(self, padding=(16, 14, 16, 8))
        header.pack(fill="x")
        tb.Label(header, text="🦙 LAMA TOUR", font="-size 18 -weight bold").pack(side="left")
        tb.Label(header, text="  Lanceur GGUF pour llama.cpp", bootstyle="secondary").pack(side="left")
        self.theme_btn = tb.Button(header, text="☀" if self._is_dark() else "🌙", width=3,
                                    bootstyle="secondary-outline", command=self.toggle_theme)
        self.theme_btn.pack(side="right")

        cards = tb.Frame(self, padding=(16, 0, 16, 8))
        cards.pack(fill="x")
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)

        gpu_card = tb.Labelframe(cards, text="GPU", padding=12, bootstyle="info")
        gpu_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        gpu_row = tb.Frame(gpu_card)
        gpu_row.pack(fill="x")
        self.vram_meter = tb.Meter(
            gpu_row, metersize=110, amounttotal=100, amountused=0, subtext="VRAM",
            textright="%", bootstyle="success", stripethickness=6,
        )
        self.vram_meter.pack(side="left", padx=(0, 12))
        gpu_text = tb.Frame(gpu_row)
        gpu_text.pack(side="left", fill="both", expand=True)
        name = self.gpu["name"]
        if self.gpu.get("count", 0) > 1:
            name += f"  (+{self.gpu['count'] - 1} autre(s) ignoré(s))"
        tb.Label(gpu_text, text=name, font="-weight bold", wraplength=260, justify="left").pack(anchor="w")
        tb.Label(gpu_text, text=f"Compute {self.gpu['cap'] or '—'}  ·  cache {self.gpu['cache'] or 'CPU'}",
                 bootstyle="secondary").pack(anchor="w")
        self.vram_text_var = tb.StringVar(value="VRAM : —")
        tb.Label(gpu_text, textvariable=self.vram_text_var, bootstyle="secondary").pack(anchor="w", pady=(4, 0))

        status_card = tb.Labelframe(cards, text="Statut", padding=12, bootstyle="info")
        status_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.status_dot_var = tb.StringVar(value="○")
        self.status_text_var = tb.StringVar(value="Serveur arrêté")
        row1 = tb.Frame(status_card)
        row1.pack(fill="x", anchor="w")
        self.status_dot_lbl = tb.Label(row1, textvariable=self.status_dot_var, bootstyle="secondary",
                                        font="-size 14")
        self.status_dot_lbl.pack(side="left")
        tb.Label(row1, textvariable=self.status_text_var, font="-weight bold").pack(side="left", padx=(4, 0))
        self.uptime_var = tb.StringVar(value="")
        tb.Label(status_card, textvariable=self.uptime_var, bootstyle="secondary").pack(anchor="w", pady=(4, 0))
        self.oc_status_var = tb.StringVar(value="opencode : indisponible")
        self.oc_lbl = tb.Label(status_card, textvariable=self.oc_status_var, bootstyle="secondary")
        self.oc_lbl.pack(anchor="w", pady=(6, 0))
        self.oc_model_var = tb.StringVar(value="")
        tb.Label(status_card, textvariable=self.oc_model_var, bootstyle="secondary").pack(anchor="w")

        perf_card = tb.Labelframe(cards, text="Bande passante & goulot (capteur local :8765)",
                                   padding=12, bootstyle="info")
        perf_card.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self.perf_unavailable_var = tb.StringVar(value="Capteur local indisponible (port 8765 injoignable)")
        self.perf_unavailable_lbl = tb.Label(perf_card, textvariable=self.perf_unavailable_var,
                                              bootstyle="secondary")
        self.perf_row = tb.Frame(perf_card)

        def stat(parent, title):
            box = tb.Frame(parent)
            box.pack(side="left", padx=(0, 22))
            tb.Label(box, text=title, bootstyle="secondary", font="-size 8").pack(anchor="w")
            var = tb.StringVar(value="—")
            tb.Label(box, textvariable=var, font="-weight bold").pack(anchor="w")
            return var

        self.perf_cpu_var = stat(self.perf_row, "CPU")
        self.perf_ram_var = stat(self.perf_row, "RAM")
        self.perf_gpu_var = stat(self.perf_row, "GPU")
        self.perf_gpuclk_var = stat(self.perf_row, "Horloge mémoire GPU")
        self.perf_npu_bw_var = stat(self.perf_row, "Bande passante NPU")
        self.perf_bottleneck_var = stat(self.perf_row, "Goulot")

        nb = tb.Notebook(self, padding=(16, 0))
        nb.pack(fill="both", expand=True)
        tab_launch = tb.Frame(nb, padding=12)
        tab_log = tb.Frame(nb, padding=12)
        nb.add(tab_launch, text="  Lancer  ")
        nb.add(tab_log, text="  Journal  ")

        self._build_launch_tab(tab_launch)
        self._build_log_tab(tab_log)

        bottom = tb.Frame(self, padding=16)
        bottom.pack(fill="x")
        self.btn_launch = tb.Button(bottom, text="▶  Lancer", bootstyle="success", command=self.launch)
        self.btn_launch.pack(side="left", padx=(0, 6))
        self.btn_restart = tb.Button(bottom, text="⟲  Redémarrer", bootstyle="warning-outline",
                                      command=self.restart, state="disabled")
        self.btn_restart.pack(side="left", padx=6)
        self.btn_stop = tb.Button(bottom, text="■  Arrêter", bootstyle="danger-outline",
                                   command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        self.btn_web = tb.Button(bottom, text="🌐 Ouvrir l'interface web", bootstyle="secondary-outline",
                                  command=self.open_web, state="disabled")
        self.btn_web.pack(side="left", padx=6)
        self.btn_cmd = tb.Button(bottom, text="📋 Copier la commande", bootstyle="secondary-outline",
                                  command=self.copy_cmd)
        self.btn_cmd.pack(side="right")

    def _build_launch_tab(self, outer):
        # Le nombre de cartes (Modele/Options serveur/Options avancees/Speculative
        # decoding/opencode) depasse desormais la hauteur de fenetre disponible sur
        # beaucoup d'ecrans -> sans conteneur scrollable, le bas de l'onglet (et les
        # boutons Lancer/Arreter en dessous, hors de cet onglet) devient inaccessible.
        scroll = tb.ScrolledFrame(outer, autohide=True)
        scroll.pack(fill="both", expand=True)
        parent = scroll

        model_card = tb.Labelframe(parent, text="Modèle", padding=12, bootstyle="secondary")
        model_card.pack(fill="x", pady=(0, 10))
        model_card.columnconfigure(0, weight=1)
        self.model_var = tb.StringVar()
        self.model_combo = tb.Combobox(model_card, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=0, column=0, sticky="we", padx=(0, 8))
        self._refresh_models(keep_selection=self.cfg.get("model"))
        tb.Button(model_card, text="Parcourir…", bootstyle="secondary",
                  command=self.browse).grid(row=0, column=1)
        tb.Button(model_card, text="Dossier modèles", bootstyle="secondary-link",
                  command=self.open_models_folder).grid(row=0, column=2, padx=(6, 0))
        tb.Button(model_card, text="🔄", width=3, bootstyle="secondary-outline",
                  command=self.refresh_all_models).grid(row=0, column=3, padx=(6, 0))

        build_row = tb.Frame(model_card)
        build_row.grid(row=1, column=0, columnspan=3, sticky="we", pady=(8, 0))
        tb.Label(build_row, text="Build serveur").pack(side="left")
        default_build = self.cfg.get("build", "standard")
        if default_build not in BUILDS:
            default_build = "standard"
        self.build_var = tb.StringVar(value=default_build)
        build_combo = tb.Combobox(build_row, textvariable=self.build_var, state="readonly", width=26,
                                   values=list(BUILDS.keys()))
        build_combo.set(default_build)
        # affiche le label lisible plutot que la clef interne, tout en gardant
        # la clef comme valeur reelle du combobox (via mapping ci-dessous)
        build_combo.configure(values=[BUILDS[k]["label"] for k in BUILDS])
        build_combo.set(BUILDS[default_build]["label"])
        self._build_label_to_key = {v["label"]: k for k, v in BUILDS.items()}
        build_combo.bind("<<ComboboxSelected>>", lambda e: self.build_var.set(
            self._build_label_to_key.get(build_combo.get(), "standard")))
        build_combo.pack(side="left", padx=(8, 8))
        self.btn_detect = tb.Button(build_row, text="Détecter les GPU", bootstyle="secondary-outline",
                                     command=self.detect_devices)
        self.btn_detect.pack(side="left")
        self.devices_var = tb.StringVar(value="")
        tb.Label(model_card, textvariable=self.devices_var, bootstyle="secondary",
                 wraplength=850, justify="left").grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        opt_card = tb.Labelframe(parent, text="Options serveur", padding=12, bootstyle="secondary")
        opt_card.pack(fill="x", pady=(0, 10))
        for c in (1, 3):
            opt_card.columnconfigure(c, weight=1)

        def field(row, col, label, var, width=10):
            tb.Label(opt_card, text=label).grid(row=row, column=col, sticky="w", padx=(0 if col == 0 else 16, 4), pady=6)
            e = tb.Entry(opt_card, textvariable=var, width=width)
            e.grid(row=row, column=col + 1, sticky="w", pady=6)
            return e

        self.cache_var = tb.StringVar(value=self.cfg.get("cache", self.gpu["cache"] or "off"))
        tb.Label(opt_card, text="Cache KV").grid(row=0, column=0, sticky="w", pady=6)
        tb.Combobox(opt_card, textvariable=self.cache_var, state="readonly", width=9,
                    values=["off", "turbo3", "turbo4", "q4_0", "q8_0"]).grid(row=0, column=1, sticky="w", pady=6)

        self.ctx_var = tb.StringVar(value=str(self.cfg.get("ctx", self.gpu["ctx"])))
        field(0, 2, "Contexte", self.ctx_var)

        self.port_var = tb.StringVar(value=str(self.cfg.get("port", DEFAULT_PORT)))
        field(1, 0, "Port HTTP", self.port_var)

        self.threads_var = tb.StringVar(value=str(self.cfg.get("threads", "")))
        field(1, 2, "Threads (-t)", self.threads_var)

        self.parallel_var = tb.StringVar(value=str(self.cfg.get("parallel", "1")))
        field(2, 0, "Slots parallèles (-np)", self.parallel_var)

        moe_row = tb.Frame(opt_card)
        moe_row.grid(row=3, column=0, columnspan=4, sticky="we", pady=(10, 0))
        tb.Label(moe_row, text="Experts MoE sur CPU").pack(side="left")
        self.cpu_moe_var = tb.BooleanVar(value=bool(self.cfg.get("cpu_moe", False)))
        tb.Checkbutton(moe_row, text="Tout", variable=self.cpu_moe_var, bootstyle="round-toggle",
                        command=self._sync_moe_state).pack(side="left", padx=(12, 4))
        tb.Label(moe_row, text="ou N premières couches :").pack(side="left", padx=(12, 4))
        self.n_cpu_moe_var = tb.StringVar(value=str(self.cfg.get("n_cpu_moe", "")))
        self.n_cpu_moe_entry = tb.Entry(moe_row, textvariable=self.n_cpu_moe_var, width=6)
        self.n_cpu_moe_entry.pack(side="left")
        self._sync_moe_state()

        self.auto_open_var = tb.BooleanVar(value=bool(self.cfg.get("auto_open", True)))
        tb.Checkbutton(opt_card, text="Ouvrir automatiquement l'interface web une fois prêt",
                        variable=self.auto_open_var, bootstyle="round-toggle").grid(
            row=4, column=0, columnspan=4, sticky="w", pady=(10, 0))

        # --- Options avancées ---
        adv_card = tb.Labelframe(parent, text="Options avancées", padding=12, bootstyle="secondary")
        adv_card.pack(fill="x", pady=(0, 10))
        for c in (1, 3):
            adv_card.columnconfigure(c, weight=1)

        def afield(row, col, label, var, width=12, show=None):
            tb.Label(adv_card, text=label).grid(row=row, column=col, sticky="w",
                                                 padx=(0 if col == 0 else 16, 4), pady=6)
            e = tb.Entry(adv_card, textvariable=var, width=width, show=show)
            e.grid(row=row, column=col + 1, sticky="w", pady=6)
            return e

        self.alias_var = tb.StringVar(value=str(self.cfg.get("alias", "")))
        afield(0, 0, "Alias (-a)", self.alias_var, width=18)

        self.ubatch_var = tb.StringVar(value=str(self.cfg.get("ubatch", "")))
        afield(0, 2, "Ubatch (-ub)", self.ubatch_var)

        self.device_var = tb.StringVar(value=str(self.cfg.get("device", "")))
        afield(1, 0, "Device(s) (-dev)", self.device_var, width=18)

        self.tensor_split_var = tb.StringVar(value=str(self.cfg.get("tensor_split", "")))
        afield(1, 2, "Tensor-split (-ts)", self.tensor_split_var)

        self.main_gpu_var = tb.StringVar(value=str(self.cfg.get("main_gpu", "")))
        afield(2, 0, "GPU principal (-mg)", self.main_gpu_var)

        self.reasoning_var = tb.StringVar(value=str(self.cfg.get("reasoning_format", "auto")))
        tb.Label(adv_card, text="Raisonnement").grid(row=2, column=2, sticky="w", padx=16, pady=6)
        tb.Combobox(adv_card, textvariable=self.reasoning_var, state="readonly", width=13,
                    values=["auto", "none", "deepseek", "deepseek-legacy"]).grid(
            row=2, column=3, sticky="w", pady=6)

        self.api_key_var = tb.StringVar(value=str(self.cfg.get("api_key", "")))
        afield(3, 0, "Clé API", self.api_key_var, width=18, show="*")

        self.log_file_var = tb.StringVar(value=str(self.cfg.get("log_file", "")))
        tb.Label(adv_card, text="Fichier de log").grid(row=3, column=2, sticky="w", padx=16, pady=6)
        log_row = tb.Frame(adv_card)
        log_row.grid(row=3, column=3, sticky="we", pady=6)
        tb.Entry(log_row, textvariable=self.log_file_var, width=16).pack(side="left")
        tb.Button(log_row, text="…", width=2, bootstyle="secondary",
                  command=self.browse_log_file).pack(side="left", padx=(4, 0))

        flags_row = tb.Frame(adv_card)
        flags_row.grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.mlock_var = tb.BooleanVar(value=bool(self.cfg.get("mlock", False)))
        tb.Checkbutton(flags_row, text="mlock", variable=self.mlock_var,
                        bootstyle="round-toggle").pack(side="left", padx=(0, 16))
        self.no_mmap_var = tb.BooleanVar(value=bool(self.cfg.get("no_mmap", False)))
        tb.Checkbutton(flags_row, text="Désactiver mmap", variable=self.no_mmap_var,
                        bootstyle="round-toggle").pack(side="left", padx=(0, 16))
        self.server_metrics_var = tb.BooleanVar(value=bool(self.cfg.get("server_metrics", False)))
        tb.Checkbutton(flags_row, text="Métriques serveur (/metrics)", variable=self.server_metrics_var,
                        bootstyle="round-toggle").pack(side="left", padx=(0, 16))
        self.jinja_var = tb.BooleanVar(value=bool(self.cfg.get("jinja", True)))
        tb.Checkbutton(flags_row, text="Jinja (template chat)", variable=self.jinja_var,
                        bootstyle="round-toggle").pack(side="left")

        # --- Speculative decoding (DFlash2 draft model) ---
        spec_card = tb.Labelframe(parent, text="Speculative decoding (DFlash2)", padding=12, bootstyle="secondary")
        spec_card.pack(fill="x", pady=(0, 10))
        spec_card.columnconfigure(1, weight=1)

        self.draft_var = tb.BooleanVar(value=bool(self.cfg.get("draft_enabled", False)))
        tb.Checkbutton(spec_card, text="Activer DFlash2 draft",
                        variable=self.draft_var, bootstyle="round-toggle").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        tb.Label(spec_card, text="Draft model").grid(row=1, column=0, sticky="w", pady=6)
        self.draft_model_var = tb.StringVar(value=str(self.cfg.get("draft_model", "")))
        self.draft_model_combo = tb.Combobox(spec_card, textvariable=self.draft_model_var,
                                              state="readonly")
        self.draft_model_combo.grid(row=1, column=1, sticky="we", padx=(0, 8))
        tb.Button(spec_card, text="Parcourir…", bootstyle="secondary",
                  command=self.browse_draft).grid(row=1, column=2)
        self._refresh_draft_models(keep_selection=self.cfg.get("draft_model"))

        # [CORRIGÉ 25/08/2026] n-max défaut 5 → 2 : meilleur acceptance (90,9%) ET
        # meilleur t/s mesurés (n-max 4/8 font chuter l'acceptation sans gain de vitesse).
        self.draft_n_max_var = tb.StringVar(value=str(self.cfg.get("draft_n_max", "2")))
        tb.Label(spec_card, text="Tokens draftés (n-max)").grid(row=2, column=0, sticky="w", pady=6)
        tb.Entry(spec_card, textvariable=self.draft_n_max_var, width=6).grid(row=2, column=1, sticky="w", pady=6)

        self.draft_ctx_var = tb.StringVar(value=str(self.cfg.get("draft_ctx", "512")))
        tb.Label(spec_card, text="Ctx draft (cross-attn)").grid(row=3, column=0, sticky="w", pady=6)
        tb.Entry(spec_card, textvariable=self.draft_ctx_var, width=6).grid(row=3, column=1, sticky="w", pady=6)

        # [CORRIGÉ 25/08/2026] Label d'avertissement (clamp ctx DFlash, etc.)
        self.dflash_warn_var = tb.StringVar(value="")
        tb.Label(spec_card, textvariable=self.dflash_warn_var, bootstyle="warning",
                 wraplength=700, justify="left").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))

        note = tb.Label(parent, text="(pas de support precision_map.json / adaptive-quant dans ce build)",
                         bootstyle="secondary")
        note.pack(anchor="w", pady=(0, 10))

        oc_card = tb.Labelframe(parent, text="Intégration opencode (provider llamacpp)", padding=12,
                                 bootstyle="secondary")
        oc_card.pack(fill="x")
        tb.Button(oc_card, text="Copier la référence modèle opencode", bootstyle="secondary",
                  command=self.copy_oc_ref).pack(anchor="w")

    def _build_log_tab(self, parent):
        toolbar = tb.Frame(parent)
        toolbar.pack(fill="x", pady=(0, 8))
        tb.Button(toolbar, text="🗑 Effacer", bootstyle="secondary-outline",
                  command=self.clear_log).pack(side="left")
        tb.Button(toolbar, text="📋 Copier tout", bootstyle="secondary-outline",
                  command=self.copy_log).pack(side="left", padx=6)
        self.autoscroll_var = tb.BooleanVar(value=True)
        tb.Checkbutton(toolbar, text="Défilement auto", variable=self.autoscroll_var,
                        bootstyle="round-toggle").pack(side="right")

        self.out = tb.ScrolledText(parent, height=18, font=("Consolas", 9), autohide=True)
        self.out.pack(fill="both", expand=True)
        self.out.text.configure(state="disabled")

    # ----------------------------------------------------------- helpers

    def _alive(self):
        """True si la fenetre existe encore. A utiliser dans toute boucle
        self.after(...) et dans append() avant de toucher un widget : une
        fois l'interpreteur Tcl completement detruit (fin de on_close()),
        winfo_exists() peut lever TclError au lieu de renvoyer False -
        differe selon si c'est un widget enfant ou la fenetre racine elle
        meme qui est interrogee. On avale cette exception ici plutot que de
        dupliquer un try/except a chaque appelant."""
        try:
            return bool(self.winfo_exists())
        except Exception:
            return False

    def _is_dark(self):
        return "dark" in self.style.theme.name

    def toggle_theme(self):
        new_theme = LIGHT_THEME if self._is_dark() else DARK_THEME
        self.style.theme_use(new_theme)
        self.theme_btn.configure(text="☀" if self._is_dark() else "🌙")
        self.cfg["theme"] = new_theme

    def _refresh_models(self, keep_selection=None):
        self.models = find_models()
        self._model_display_map = {}
        values = []
        for m in self.models:
            try:
                size_gb = os.path.getsize(m) / (1024 ** 3)
                label = f"{os.path.basename(m)}   ·   {size_gb:.1f} Go"
            except OSError:
                label = os.path.basename(m)
            if label in self._model_display_map:
                # Meme nom de fichier + meme taille affichee dans deux dossiers
                # differents (models/ et racine) : sans ca, le second ecraserait
                # silencieusement le premier dans la table et deviendrait
                # inaccessible depuis la liste deroulante.
                label = f"{label}   [{os.path.basename(os.path.dirname(m))}]"
            values.append(label)
            self._model_display_map[label] = m
        self.model_combo["values"] = values
        if keep_selection:
            for label, path in self._model_display_map.items():
                if path == keep_selection:
                    self.model_combo.set(label)
                    return
            if os.path.isfile(keep_selection):
                self.model_combo.set(keep_selection)
                return
        if values:
            self.model_combo.current(0)

    def refresh_all_models(self):
        """Recharge la liste des .gguf (dossier models/ + racine) sans relancer l'appli.
        Garde la selection courante si le fichier existe toujours."""
        self._refresh_models(keep_selection=self._resolve_model_path())
        self._refresh_draft_models(keep_selection=self._resolve_draft_path())
        self.append("[modèles] liste rafraîchie\n")

    def browse(self):
        path = filedialog.askopenfilename(
            title="Choisir un GGUF", filetypes=[("GGUF", "*.gguf"), ("Tous", "*.*")]
        )
        if path:
            self.model_combo.set(path)

    def open_models_folder(self):
        target = MODELS_DIR if os.path.isdir(MODELS_DIR) else ROOT
        os.startfile(target)

    def browse_log_file(self):
        path = filedialog.asksaveasfilename(
            title="Fichier de log du serveur", defaultextension=".log",
            filetypes=[("Log", "*.log"), ("Tous", "*.*")],
        )
        if path:
            self.log_file_var.set(path)

    # ---- draft model helpers ----

    def _refresh_draft_models(self, keep_selection=None):
        """Remplit la liste déroulante des draft models (même dossier que les modèles principaux)."""
        # [CORRIGÉ 25/08/2026] Filtre taille < 3 Go : les targets (D2-MOE 17,1 Go, UD-IQ4_NL
        # 18 Go…) apparaissaient dans la liste et étaient sélectionnables comme « draft »,
        # ce qui chargeait un second modèle complet en VRAM.
        self.draft_models = [m for m in find_models() if _file_gb(m) is not None and _file_gb(m) < 3.0]
        self._draft_display_map = {}
        values = []
        for m in self.draft_models:
            try:
                size_gb = os.path.getsize(m) / (1024 ** 3)
                label = f"{os.path.basename(m)}   ·   {size_gb:.1f} Go"
            except OSError:
                label = os.path.basename(m)
            values.append(label)
            self._draft_display_map[label] = m
        self.draft_model_combo["values"] = values
        if keep_selection:
            for label, path in self._draft_display_map.items():
                if path == keep_selection:
                    self.draft_model_combo.set(label)
                    return
            if os.path.isfile(keep_selection):
                if _file_gb(keep_selection) is None or _file_gb(keep_selection) < 3.0:
                    self.draft_model_combo.set(keep_selection)
                    return
                # [CORRIGÉ 25/08/2026] ancien choix = un target > 3 Go : on retombe
                # sur le défaut ci-dessous au lieu de le recharger comme draft.
        # [CORRIGÉ 25/08/2026] Défaut : draft OFFICIAL-BF16 (seul draft DFlash validé).
        if os.path.isfile(DEFAULT_DRAFT):
            for label, path in self._draft_display_map.items():
                if path == DEFAULT_DRAFT:
                    self.draft_model_combo.set(label)
                    return
            self.draft_model_combo.set(DEFAULT_DRAFT)
            return
        if values:
            self.draft_model_combo.current(0)

    def browse_draft(self):
        path = filedialog.askopenfilename(
            title="Choisir un GGUF draft", filetypes=[("GGUF", "*.gguf"), ("Tous", "*.*")]
        )
        if path:
            self.draft_model_combo.set(path)

    def _resolve_draft_path(self):
        val = self.draft_model_var.get()
        if not val:
            return None
        if val in self._draft_display_map:
            return self._draft_display_map[val]
        if os.path.isfile(val):
            return os.path.abspath(val)
        return val

    def build_dir(self):
        key = getattr(self, "build_var", None).get() if hasattr(self, "build_var") else "standard"
        return BUILDS.get(key, BUILDS["standard"])["dir"]

    def server_exe(self):
        """Chemin de llama-server.exe pour le build actuellement selectionne
        (Standard CUDA-only a la racine, ou Multi-GPU CUDA+Vulkan dans multi-gpu/)."""
        return os.path.join(self.build_dir(), "llama-server.exe")

    def detect_devices(self):
        exe = self.server_exe()
        if not os.path.isfile(exe):
            messagebox.showerror("Erreur", f"Introuvable pour ce build :\n{exe}")
            return
        self.btn_detect.configure(state="disabled")
        self.devices_var.set("Détection en cours…")
        threading.Thread(target=self._detect_devices_worker, args=(exe,), daemon=True).start()

    def _detect_devices_worker(self, exe):
        try:
            p = subprocess.run([exe, "--list-devices"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=20,
                                creationflags=CREATIONFLAGS_NO_WINDOW)
            out = (p.stdout or "") + (p.stderr or "")
            lines = [l.strip() for l in out.splitlines()
                     if ":" in l and ("CUDA" in l or "Vulkan" in l or "ROCm" in l)]
            err = None
        except Exception as e:
            lines, err = [], str(e)
        if self._alive():
            self.after(0, self._detect_devices_done, lines, err)

    def _detect_devices_done(self, lines, err):
        self.btn_detect.configure(state="normal")
        if err:
            messagebox.showerror("Erreur", f"Échec de la détection : {err}")
            return
        if lines:
            self.devices_var.set("Périphériques : " + "  ·  ".join(lines))
            self.append("[détection GPU]\n" + "\n".join(lines) + "\n")
        else:
            self.devices_var.set("Aucun périphérique GPU détecté pour ce build.")

    def _resolve_model_path(self):
        val = self.model_var.get()
        if not val:
            return None
        if val in self._model_display_map:
            return self._model_display_map[val]
        if os.path.isfile(val):
            return os.path.abspath(val)
        return val

    def _sync_moe_state(self):
        self.n_cpu_moe_entry.configure(state="disabled" if self.cpu_moe_var.get() else "normal")

    # ------------------------------------------------------------ launch

    def build_args(self):
        model = self._resolve_model_path()
        if not model:
            messagebox.showerror("Erreur", "Choisis un modèle.")
            return None
        if not os.path.isfile(model):
            messagebox.showerror("Erreur", f"Modèle introuvable : {model}")
            return None

        try:
            ctx = int(self.ctx_var.get())
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror("Erreur", "Contexte et port doivent être des nombres.")
            return None

        # [CORRIGÉ 25/08/2026] Garde-fou DFlash : ctx > 8192 => crash draft
        # « invalid vector subscript » (config validée 24/08 : -c 8192).
        # Clamp automatique + avertissement visible dans la carte DFlash2.
        if self.draft_var.get() and ctx > DFLASH_MAX_CTX:
            ctx = DFLASH_MAX_CTX
            self.ctx_var.set(str(ctx))
            self.dflash_warn_var.set(
                f"⚠ Contexte ramené à {DFLASH_MAX_CTX} : DFlash plante au-delà "
                "(crash draft validé 24/08/2026).")

        # [CORRIGÉ 25/08/2026] turbo3/turbo4 exigent TOUTES les couches KV sur GPU :
        # incompatible avec un -ngl partiel/auto. Seul cas toléré : modèle <= 6 Go qui
        # tient entièrement en VRAM (RTX 5070 8 Go) → on force alors -ngl 99.
        full_gpu_forced = False
        if self.cache_var.get().startswith("turbo"):
            model_gb = _file_gb(model)
            if model_gb is None or model_gb > 6.0:
                taille = f"{model_gb:.1f} Go" if model_gb is not None else "taille inconnue"
                messagebox.showerror(
                    "Cache Turbo incompatible",
                    f"Le cache {self.cache_var.get()} exige tout le modèle sur GPU (-ngl 99).\n"
                    f"Modèle sélectionné : {taille} (> 6 Go) → offload CPU partiel obligatoire.\n\n"
                    "Utilise plutôt un cache q4_0 / q8_0 / off pour ce modèle."
                )
                return None
            full_gpu_forced = True

        args = ["-m", model, "--ctx-size", str(ctx), "--batch-size", "512",
                "--host", "127.0.0.1", "--port", str(port)]
        if self.cache_var.get() != "off":
            args += ["--cache-type-k", self.cache_var.get(),
                     "--cache-type-v", self.cache_var.get()]
        if self.gpu["cache"]:
            # Ne PAS forcer "-ngl 99" : un -ngl explicite desactive le mecanisme --fit
            # de llama.cpp (qui n'ajuste que les parametres non definis par l'utilisateur).
            # Verifie empiriquement sur ce projet : avec -ngl 99 force, un modele qui ne
            # tient pas entierement en VRAM soit sur-alloue via VMM (marche par chance,
            # lent), soit plante net en OOM (confirme sur un GGUF de 30 Go). Comme les
            # gros modeles (jusqu'a 30 Go) sont maintenant courants dans ce dossier, il
            # faut laisser le defaut "-ngl auto" + "--fit on" (defauts de llama-server)
            # decider combien de couches tiennent reellement en VRAM.
            if full_gpu_forced:
                # [CORRIGÉ 25/08/2026] cache turbo accepté uniquement parce que le
                # modèle tient en VRAM : -ngl 99 explicite garantit le tout-GPU
                # exigé par turbo (sinon KV partiellement CPU = refus du build).
                args += ["-ngl", "99", "--flash-attn", "on"]
            else:
                args += ["-ngl", "auto", "--flash-attn", "on"]
        else:
            args += ["-ngl", "0"]

        for var, flag, label in (
            (self.threads_var, "--threads", "Threads"),
            (self.parallel_var, "--parallel", "Slots parallèles"),
        ):
            val = var.get().strip()
            if val:
                try:
                    n = int(val)
                except ValueError:
                    messagebox.showerror("Erreur", f"« {label} » doit être un entier.")
                    return None
                args += [flag, str(n)]

        if self.cpu_moe_var.get():
            args += ["--cpu-moe"]
        else:
            n_moe = self.n_cpu_moe_var.get().strip()
            if n_moe:
                try:
                    n_moe_i = int(n_moe)
                except ValueError:
                    messagebox.showerror("Erreur", "Le nombre de couches MoE sur CPU doit être un entier.")
                    return None
                args += ["--n-cpu-moe", str(n_moe_i)]

        # --- Options avancées ---
        alias = self.alias_var.get().strip()
        if alias:
            args += ["-a", alias]

        ubatch = self.ubatch_var.get().strip()
        if ubatch:
            try:
                args += ["-ub", str(int(ubatch))]
            except ValueError:
                messagebox.showerror("Erreur", "« Ubatch » doit être un entier.")
                return None

        device = self.device_var.get().strip()
        if device:
            args += ["-dev", device]

        tensor_split = self.tensor_split_var.get().strip()
        if tensor_split:
            args += ["-ts", tensor_split]

        main_gpu = self.main_gpu_var.get().strip()
        if main_gpu:
            try:
                args += ["-mg", str(int(main_gpu))]
            except ValueError:
                messagebox.showerror("Erreur", "« GPU principal » doit être un entier.")
                return None

        if self.reasoning_var.get() != "auto":
            args += ["--reasoning-format", self.reasoning_var.get()]

        api_key = self.api_key_var.get().strip()
        if api_key:
            args += ["--api-key", api_key]

        log_file = self.log_file_var.get().strip()
        if log_file:
            args += ["--log-file", log_file]

        if self.mlock_var.get():
            args += ["--mlock"]
        if self.no_mmap_var.get():
            args += ["--no-mmap"]
        if self.server_metrics_var.get():
            args += ["--metrics"]
        if not self.jinja_var.get():
            args += ["--no-jinja"]

        # --- Speculative decoding (DFlash2 draft model) ---
        if self.draft_var.get():
            draft_path = self._resolve_draft_path()
            if draft_path and os.path.isfile(draft_path):
                args += ["--model-draft", draft_path,
                         "--spec-type", "dflash",
                         "--spec-draft-ngl", "99"]
                n_max = self.draft_n_max_var.get().strip()
                if n_max:
                    try:
                        args += ["--spec-draft-n-max", str(int(n_max))]
                    except ValueError:
                        pass
                draft_ctx = self.draft_ctx_var.get().strip()
                if draft_ctx:
                    try:
                        args += ["--spec-draft-ctx-size", str(int(draft_ctx))]
                    except ValueError:
                        pass
            elif draft_path:
                messagebox.showwarning("Draft model",
                    f"Draft model introuvable : {draft_path}\n"
                    "Le draft sera ignoré.")
        return args

    def _current_settings(self):
        return {
            "model": self._resolve_model_path(),
            "build": getattr(self, "build_var", None).get() if hasattr(self, "build_var") else "standard",
            "cache": self.cache_var.get(),
            "ctx": self.ctx_var.get(),
            "port": self.port_var.get(),
            "threads": self.threads_var.get(),
            "parallel": self.parallel_var.get(),
            "cpu_moe": self.cpu_moe_var.get(),
            "n_cpu_moe": self.n_cpu_moe_var.get(),
            "alias": self.alias_var.get(),
            "ubatch": self.ubatch_var.get(),
            "device": self.device_var.get(),
            "tensor_split": self.tensor_split_var.get(),
            "main_gpu": self.main_gpu_var.get(),
            "reasoning_format": self.reasoning_var.get(),
            "api_key": self.api_key_var.get(),
            "log_file": self.log_file_var.get(),
            "mlock": self.mlock_var.get(),
            "no_mmap": self.no_mmap_var.get(),
            "server_metrics": self.server_metrics_var.get(),
            "jinja": self.jinja_var.get(),
            "draft_enabled": self.draft_var.get(),
            "draft_model": self.draft_model_var.get(),
            "draft_n_max": self.draft_n_max_var.get(),
            "draft_ctx": self.draft_ctx_var.get(),
            "auto_open": self.auto_open_var.get(),
            "theme": self.cfg.get("theme", DARK_THEME),
        }

    def launch(self):
        args = self.build_args()
        if not args:
            return
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Info", "Le serveur tourne déjà.")
            return

        port = self.port_var.get()
        if self._port_busy(port):
            messagebox.showerror(
                "Port occupé",
                f"Le port {port} est déjà utilisé par un autre processus.\n"
                "Arrête-le ou choisis un autre port avant de lancer.",
            )
            return

        server_exe = self.server_exe()
        build_dir = self.build_dir()
        if not os.path.isfile(server_exe):
            messagebox.showerror("Erreur", f"llama-server.exe introuvable :\n{server_exe}")
            return

        self.append(f"$ {server_exe} {' '.join(args)}\n")
        self.append("Lancement…\n")
        self.btn_launch.configure(state="disabled")
        self.btn_restart.configure(state="normal")
        self.btn_stop.configure(state="normal")
        self.btn_web.configure(state="normal")
        self.web_auto_opened = False
        self.status_dot_var.set("●")
        self.status_dot_lbl.configure(bootstyle="warning")
        self.status_text_var.set(f"Démarrage… (port {port})")

        env = dict(os.environ)
        env["PATH"] = build_dir + os.pathsep + env.get("PATH", "")
        try:
            self.proc = subprocess.Popen(
                [server_exe] + args,
                cwd=build_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=CREATIONFLAGS_NO_WINDOW,
            )
        except OSError as exc:
            self.append(f"[erreur au lancement] {exc}\n")
            messagebox.showerror("Erreur", f"Impossible de lancer le serveur :\n{exc}")
            self._done()
            return
        self.launch_time = time.time()
        save_config({**self.cfg, **self._current_settings()})
        threading.Thread(target=self._pump, daemon=True).start()

    def restart(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.append("[redémarrage…]\n")
            self.after(800, self._restart_wait, 0)
        else:
            self.launch()

    def _restart_wait(self, attempts):
        # BUG (manque un garde-fou) : sans limite, un process qui ignore
        # terminate() (I/O bloquant, gros modele en cours de mmap, etc.)
        # faisait boucler _restart_wait indefiniment sans jamais relancer.
        # Au bout de ~10s, on force un kill() plutot que d'attendre a l'infini.
        if self.proc and self.proc.poll() is None:
            if attempts == 33:  # ~10s (33 * 300ms) apres le premier delai de 800ms
                self.append("[redémarrage] le serveur ne répond pas, arrêt forcé\n")
                self.proc.kill()
            self.after(300, self._restart_wait, attempts + 1)
        else:
            self.launch()

    def _port_busy(self, port):
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                return s.connect_ex(("127.0.0.1", int(port))) == 0
        except Exception:
            return False

    # ---------------------------------------------------------- polling

    def _vram_poll(self):
        if not self._alive():
            return
        index = self.gpu.get("index")
        # Machine sans GPU NVIDIA (index=None) : ne pas relancer nvidia-smi
        # en boucle toutes les 3s pour rien, ca echoue proprement mais c'est
        # un subprocess inutile a repetition.
        mem = query_gpu_memory(index) if index is not None else None
        if mem:
            used, total = mem
            pct = (used / total * 100) if total else 0
            style = "success" if pct < 60 else ("warning" if pct < 85 else "danger")
            self.vram_meter.configure(amountused=round(pct), bootstyle=style)
            self.vram_text_var.set(f"VRAM : {used/1024:.1f} / {total/1024:.1f} Go")
        else:
            self.vram_text_var.set("VRAM : indisponible (CPU)")
        self.after(3000, self._vram_poll)

    def _metrics_poll(self):
        if not self._alive():
            return
        m = query_local_metrics()
        if m is None:
            self.perf_row.pack_forget()
            self.perf_unavailable_lbl.pack(anchor="w")
        else:
            self.perf_unavailable_lbl.pack_forget()
            self.perf_row.pack(anchor="w")

            cpu = m.get("cpu") or {}
            self.perf_cpu_var.set(
                f"{cpu.get('load_pct', 0):.0f}%  ·  {cpu.get('temp_c', 0):.0f}°C  ·  {cpu.get('power_w', 0):.0f} W"
            )

            ram = m.get("ram") or {}
            self.perf_ram_var.set(
                f"{ram.get('used_gb', 0):.1f} / {ram.get('total_gb', 0):.1f} Go  ({ram.get('pct', 0):.0f}%)"
            )

            gpu = m.get("gpu") or {}
            self.perf_gpu_var.set(
                f"{gpu.get('load_pct', 0):.0f}%  ·  {gpu.get('temp_c', 0):.0f}°C  ·  {gpu.get('power_w', 0):.1f} W"
            )

            clocks = ((m.get("optimus") or {}).get("dgpu") or {}).get("clocks_mhz") or {}
            if clocks:
                self.perf_gpuclk_var.set(f"{clocks.get('mem', 0):.0f} MHz")
            else:
                self.perf_gpuclk_var.set("—")

            npu = m.get("npu") or {}
            bw = npu.get("bw_gbs", 0) or 0
            state = npu.get("state", "—")
            if bw > 0:
                self.perf_npu_bw_var.set(f"{bw:.1f} GB/s  ({state})")
            else:
                self.perf_npu_bw_var.set(f"— ({state})" if state != "—" else "—")

            bottleneck = npu.get("bottleneck", "—")
            self.perf_bottleneck_var.set(bottleneck)

        self.after(2000, self._metrics_poll)

    def _oc_poll(self):
        if not self._alive():
            return
        port = self.port_var.get()
        ok = False
        model_name = None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1.5) as r:
                data = json.loads(r.read().decode())
                models = data.get("data") or data.get("models") or []
                if models:
                    ok = True
                    model_name = models[0].get("id") or models[0].get("name")
        except Exception:
            ok = False

        if ok:
            self.status_dot_var.set("●")
            self.status_dot_lbl.configure(bootstyle="success")
            self.status_text_var.set(f"Serveur prêt (port {port})")
            self.oc_status_var.set("opencode : provider llamacpp actif")
            self.oc_lbl.configure(bootstyle="success")
            self.oc_model_var.set(f"modèle : llamacpp/{model_name}" if model_name else "")
            if self.auto_open_var.get() and not self.web_auto_opened:
                self.web_auto_opened = True
                self.open_web()
        else:
            if self.proc and self.proc.poll() is None:
                self.status_dot_lbl.configure(bootstyle="warning")
                self.status_text_var.set(f"Démarrage… (port {port})")
            else:
                self.status_dot_var.set("○")
                self.status_dot_lbl.configure(bootstyle="secondary")
                self.status_text_var.set("Serveur arrêté")
            self.oc_status_var.set("opencode : indisponible")
            self.oc_lbl.configure(bootstyle="secondary")
            self.oc_model_var.set("")
        self.after(2000, self._oc_poll)

    def _uptime_tick(self):
        if not self._alive():
            return
        if self.proc and self.proc.poll() is None and self.launch_time:
            elapsed = int(time.time() - self.launch_time)
            self.uptime_var.set(f"Actif depuis {elapsed // 60:02d}:{elapsed % 60:02d}")
        else:
            self.uptime_var.set("")
        self.after(1000, self._uptime_tick)

    # ------------------------------------------------------------- misc

    def copy_oc_ref(self):
        model = self._resolve_model_path()
        if not model and self.models:
            model = self.models[0]
        if not model:
            messagebox.showerror("Erreur", "Aucun modèle disponible à copier.")
            return
        name = os.path.basename(model)
        self.clipboard_clear()
        self.clipboard_append(f"llamacpp/{name}")
        self.append(f"[opencode] modèle copié : llamacpp/{name}\n")

    def _pump(self):
        proc = self.proc  # snapshot : identifie le process de CE thread precisement
        for line in proc.stdout:
            self.append(line)
        # BUG (fuite) : sans wait(), le handle du process n'est jamais reclame
        # (returncode jamais lu) -> ResourceWarning et fuite de handle Windows
        # a chaque lancement/redemarrage sur une session longue.
        proc.wait()
        proc.stdout.close()
        self.append("\n[serveur arrêté]\n")
        self.after(0, self._done, proc)

    def _done(self, proc=None):
        # Un ancien thread _pump (process precedent, ex: apres "Redemarrer") peut
        # terminer et appeler _done() APRES qu'un nouveau process ait deja demarre.
        # Sans cette garde, ca remet l'UI en etat "arrete" alors que le nouveau
        # serveur tourne (boutons/etat incoherents avec la realite).
        if proc is not None and proc is not self.proc:
            return
        self.btn_launch.configure(state="normal")
        self.btn_restart.configure(state="disabled")
        self.btn_stop.configure(state="disabled")
        self.btn_web.configure(state="disabled")
        self.launch_time = None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def open_web(self):
        webbrowser.open(f"http://127.0.0.1:{self.port_var.get()}")

    def copy_cmd(self):
        args = self.build_args()
        if args:
            # [CORRIGÉ 25/08/2026] ROOT contient des espaces (« lama 1080-5070 ») :
            # sans guillemets, la commande collée dans un shell était inexploitable.
            # Chaque chemin est donc quoté (exe + valeurs de type chemin).
            parts = [self.server_exe()] + [str(a) for a in args]
            cmd = " ".join(f'"{p}"' if (" " in p or "\t" in p) else p for p in parts)
            self.clipboard_clear()
            self.clipboard_append(cmd)
            self.append("[commande copiée]\n")

    def clear_log(self):
        self.out.text.configure(state="normal")
        self.out.text.delete("1.0", "end")
        self.out.text.configure(state="disabled")

    def copy_log(self):
        self.clipboard_clear()
        self.clipboard_append(self.out.text.get("1.0", "end"))
        self.append("[journal copié]\n")

    def append(self, text):
        # Si la fenetre a ete fermee (serveur laisse tourner en arriere-plan via
        # "Non" a la question de fermeture), le thread _pump continue de lire la
        # sortie du process et appellerait .after() sur une fenetre detruite.
        if not self._alive():
            return
        self.after(0, self._append, text)

    def _append(self, text):
        self.out.text.configure(state="normal")
        self.out.text.insert("end", text)
        # BUG (fuite memoire) : sans plafond, le Text tkinter accumule toutes
        # les lignes de log indefiniment. Un serveur verbeux (ou --metrics /
        # streaming de tokens) sur une session de plusieurs heures fait
        # grossir la RAM sans limite. On tronque le debut au-dela d'un seuil.
        line_count = int(self.out.text.index("end-1c").split(".")[0])
        if line_count > LOG_MAX_LINES:
            cut = line_count - LOG_TRIM_TO
            self.out.text.delete("1.0", f"{cut}.0")
        if self.autoscroll_var.get() if hasattr(self, "autoscroll_var") else True:
            self.out.text.see("end")
        self.out.text.configure(state="disabled")

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            resp = messagebox.askyesnocancel(
                "Serveur en cours",
                "Le serveur llama-server tourne encore.\nL'arrêter avant de quitter ?",
            )
            if resp is None:
                return
            if resp:
                self.proc.terminate()
                save_config({**self.cfg, **self._current_settings()})
                self.destroy()
                return
            # [CORRIGÉ 25/08/2026] Réponse « Non » : le serveur restait orphelin et
            # invisible (impossible à arrêter). On affiche désormais son PID dans une
            # fenêtre dédiée avec un bouton taskkill /PID.
            self._show_orphan_window()
            return
        save_config({**self.cfg, **self._current_settings()})
        self.destroy()

    def _show_orphan_window(self):
        """[CORRIGÉ 25/08/2026] Fenêtre de supervision d'un serveur laissé orphelin :
        affiche le PID et propose un kill propre via `taskkill /PID <pid> /F`
        (taskkill plutôt que proc.terminate() : la fenêtre principale est masquée
        et le handle peut survivre à l'UI, autant tuer par PID comme en console).
        """
        pid = self.proc.pid
        self.withdraw()  # cache l'UI principale, garde le mainloop vivant

        win = tb.Toplevel(self)
        win.title("Serveur laissé actif")
        tb.Label(win, text="llama-server tourne encore en arrière-plan.",
                 font="-weight bold").pack(anchor="w", padx=16, pady=(16, 2))
        tb.Label(win, text=f"PID : {pid}", font="-size 12").pack(anchor="w", padx=16)
        status_var = tb.StringVar(value="")
        tb.Label(win, textvariable=status_var, bootstyle="secondary").pack(anchor="w", padx=16)

        def kill_orphan():
            r = subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True,
                               text=True)
            if r.returncode == 0:
                status_var.set(f"Processus {pid} terminé.")
            else:
                status_var.set((r.stderr or r.stdout or "taskkill a échoué.").strip())
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass

        def quit_all():
            save_config({**self.cfg, **self._current_settings()})
            win.destroy()
            self.destroy()

        btns = tb.Frame(win)
        btns.pack(fill="x", padx=16, pady=(10, 16))
        tb.Button(btns, text=f"Terminer (taskkill /PID {pid})", bootstyle="danger",
                  command=kill_orphan).pack(side="left")
        tb.Button(btns, text="Laisser tourner & quitter l'UI", bootstyle="secondary-outline",
                  command=quit_all).pack(side="left", padx=(8, 0))
        win.protocol("WM_DELETE_WINDOW", quit_all)


if __name__ == "__main__":
    LlamaUI().mainloop()
