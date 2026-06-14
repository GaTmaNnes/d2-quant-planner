#!/usr/bin/env python3
"""
D2 Production UI — PySide6 non-bloquant
=========================================
Interface pour d2_production.py :

  - Champ model_id  (HuggingFace ID ou chemin local)
  - Slider VRAM budget (2–48 GB)
  - Slider w_risk = λ (0.1–5.0)
  - Table plan (layer / type / dtype / score / vram)
  - Bouton Solve   (thread non-bloquant)
  - Bouton Export  (JSON + cmd.txt pour llama.cpp)
  - Status bar (progression, erreurs)

Dépendances :
  pip install PySide6 ortools huggingface_hub safetensors

Usage :
  python3 d2_ui.py
"""

import json
import os
import sys
import threading
from collections import Counter
from typing import List, Dict, Optional

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QTableWidget,
    QTableWidgetItem, QHeaderView, QStatusBar, QFileDialog,
    QGroupBox, QGridLayout, QSplitter, QTextEdit,
)

# ─── Worker thread (non-bloquant) ────────────────────────────────────────────

class SolverWorker(QObject):
    """Charge les poids et résout l'ILP dans un thread séparé."""

    progress = Signal(str)      # messages de statut
    finished = Signal(list)     # plan résolu
    error    = Signal(str)      # erreur

    def __init__(self, model_id: str, vram_budget: float,
                 w_speed: float, w_risk: float,
                 cache_dir: str = '/tmp/hf_cache'):
        super().__init__()
        self.model_id    = model_id
        self.vram_budget = vram_budget
        self.w_speed     = w_speed
        self.w_risk      = w_risk
        self.cache_dir   = cache_dir

    def run(self):
        try:
            from d2_production import solve_quantization_plan

            # ── Chargement poids ─────────────────────────────────────────
            self.progress.emit(f"Chargement {self.model_id} ...")
            layers = self._load_layers()

            # ── ILP ──────────────────────────────────────────────────────
            self.progress.emit(f"ILP solver ({len(layers)} couches) ...")
            plan = solve_quantization_plan(
                layers,
                vram_budget_gb=self.vram_budget,
                w_speed=self.w_speed,
                w_risk=self.w_risk,
            )
            self.finished.emit(plan)

        except Exception as e:
            self.error.emit(str(e))

    def _load_layers(self) -> List[Dict]:
        """Charge depuis HuggingFace (safetensors) ou chemin local."""
        # Vérifier chemin local d'abord
        local_st = os.path.join(self.model_id, 'model.safetensors')
        if os.path.isfile(self.model_id):
            local_st = self.model_id
        elif not os.path.isfile(local_st):
            from huggingface_hub import hf_hub_download, list_repo_files
            self.progress.emit("Téléchargement depuis HuggingFace ...")
            files = [f for f in list_repo_files(self.model_id)
                     if f.endswith('.safetensors') and 'onnx' not in f]
            if not files:
                raise FileNotFoundError(f"Pas de .safetensors pour {self.model_id}")
            local_st = hf_hub_download(self.model_id, files[0],
                                        cache_dir=self.cache_dir)

        import safetensors.torch as st
        tensors = st.load_file(local_st)
        layers = []
        for name, tensor in tensors.items():
            if tensor.ndim < 2: continue
            t = tensor.float().numpy() if hasattr(tensor, 'numpy') else tensor
            if t.ndim > 2: t = t.reshape(t.shape[0], -1)
            m, n = t.shape
            if m < 8 or n < 8: continue
            layers.append({'name': name, 'shape': [m, n]})
        return layers


# ─── Palette couleurs dtype ───────────────────────────────────────────────────

DTYPE_COLORS = {
    'FP16': QColor(70,  130, 180),   # steel blue
    'INT8': QColor(60,  179, 113),   # medium sea green
    'INT4': QColor(255, 140,   0),   # dark orange
}
DTYPE_TEXT = {
    'FP16': QColor(255, 255, 255),
    'INT8': QColor(255, 255, 255),
    'INT4': QColor(255, 255, 255),
}


# ─── Fenêtre principale ───────────────────────────────────────────────────────

class D2Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("D2 Production — Quantization Planner")
        self.resize(1100, 700)
        self._plan: Optional[List[Dict]] = None
        self._thread: Optional[QThread] = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)

        # ── Controls ─────────────────────────────────────────────────────
        ctrl_box = QGroupBox("Paramètres")
        ctrl_layout = QGridLayout(ctrl_box)

        # Model ID
        ctrl_layout.addWidget(QLabel("Model ID / path :"), 0, 0)
        self.model_edit = QLineEdit("gpt2")
        self.model_edit.setPlaceholderText("gpt2 | TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        ctrl_layout.addWidget(self.model_edit, 0, 1, 1, 3)

        # VRAM Budget
        ctrl_layout.addWidget(QLabel("VRAM budget (GB) :"), 1, 0)
        self.vram_slider = QSlider(Qt.Horizontal)
        self.vram_slider.setRange(2, 48)
        self.vram_slider.setValue(8)
        self.vram_slider.setTickInterval(4)
        self.vram_slider.setTickPosition(QSlider.TicksBelow)
        self.vram_label = QLabel("8 GB")
        self.vram_slider.valueChanged.connect(
            lambda v: self.vram_label.setText(f"{v} GB"))
        ctrl_layout.addWidget(self.vram_slider, 1, 1, 1, 2)
        ctrl_layout.addWidget(self.vram_label, 1, 3)

        # w_risk (λ)
        ctrl_layout.addWidget(QLabel("w_risk (λ) :"), 2, 0)
        self.risk_slider = QSlider(Qt.Horizontal)
        self.risk_slider.setRange(1, 50)    # ×0.1 → [0.1, 5.0]
        self.risk_slider.setValue(10)       # défaut 1.0
        self.risk_slider.setTickInterval(5)
        self.risk_slider.setTickPosition(QSlider.TicksBelow)
        self.risk_label = QLabel("1.0")
        self.risk_slider.valueChanged.connect(
            lambda v: self.risk_label.setText(f"{v/10:.1f}"))
        ctrl_layout.addWidget(self.risk_slider, 2, 1, 1, 2)
        ctrl_layout.addWidget(self.risk_label, 2, 3)

        # Hint
        hint = QLabel("⬅ INT4-agressif   ⬆ λ = risque/vitesse   FP16-conservateur ➡")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        ctrl_layout.addWidget(hint, 3, 1, 1, 3)

        # Buttons
        btn_row = QHBoxLayout()
        self.solve_btn = QPushButton("▶ Solve")
        self.solve_btn.setFixedHeight(36)
        self.solve_btn.setStyleSheet("font-weight: bold; background: #2196F3; color: white;")
        self.solve_btn.clicked.connect(self._on_solve)
        btn_row.addWidget(self.solve_btn)

        self.export_btn = QPushButton("⬇ Export GGUF")
        self.export_btn.setFixedHeight(36)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self.export_btn)
        ctrl_layout.addLayout(btn_row, 4, 0, 1, 4)

        root.addWidget(ctrl_box)

        # ── Splitter : table + summary ────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        # Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ['Layer', 'Type', 'Dtype', 'Score', 'VRAM GB'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        font_m = QFont("Monospace", 9)
        self.table.setFont(font_m)
        splitter.addWidget(self.table)

        # Summary panel
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setFont(QFont("Monospace", 10))
        self.summary_box.setFixedWidth(280)
        splitter.addWidget(self.summary_box)
        splitter.setSizes([800, 280])

        root.addWidget(splitter)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Prêt. Entrez un model_id et cliquez Solve.")

    def _on_solve(self):
        model_id    = self.model_edit.text().strip()
        vram_budget = float(self.vram_slider.value())
        w_speed     = 1.0
        w_risk      = self.risk_slider.value() / 10.0

        if not model_id:
            self.status.showMessage("Erreur : model_id vide.")
            return

        self.solve_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.table.setRowCount(0)
        self.summary_box.clear()

        # Worker dans un QThread
        self._thread  = QThread()
        self._worker  = SolverWorker(model_id, vram_budget, w_speed, w_risk)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda m: self.status.showMessage(m))
        self._worker.finished.connect(self._on_plan_ready)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self.solve_btn.setEnabled(True))

        self._thread.start()

    def _on_plan_ready(self, plan: List[Dict]):
        self._plan = plan
        self._populate_table(plan)
        self._update_summary(plan)
        self.export_btn.setEnabled(True)
        vram_used = sum(e['vram_gb'] for e in plan)
        self.status.showMessage(
            f"Résolu — {len(plan)} couches, {vram_used:.3f} GB utilisés")

    def _on_error(self, msg: str):
        self.status.showMessage(f"Erreur : {msg}")
        self.solve_btn.setEnabled(True)

    def _populate_table(self, plan: List[Dict]):
        self.table.setRowCount(len(plan))
        for row, e in enumerate(plan):
            items = [
                e['name'],
                e['layer_type'],
                e['dtype'],
                f"{e['score']:+.4f}",
                f"{e['vram_gb']:.5f}",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignVCenter |
                                       (Qt.AlignLeft if col == 0 else Qt.AlignCenter))
                if col == 2:  # dtype
                    item.setBackground(DTYPE_COLORS.get(text, QColor(200,200,200)))
                    item.setForeground(DTYPE_TEXT.get(text, QColor(0,0,0)))
                    font = QFont()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(row, col, item)
        self.table.resizeRowsToContents()

    def _update_summary(self, plan: List[Dict]):
        from d2_production import summarize
        w_risk = self.risk_slider.value() / 10.0
        text   = summarize(plan, float(self.vram_slider.value()),
                            w_speed=1.0, w_risk=w_risk)
        counts = Counter(e['dtype'] for e in plan)
        n = len(plan)

        bar_html = ""
        for dtype, color in [('INT4','#FF8C00'), ('INT8','#3CB371'), ('FP16','#4682B4')]:
            c = counts.get(dtype, 0)
            pct = int(c / n * 100) if n else 0
            bar_html += (f"<span style='background:{color};color:white;"
                         f"padding:2px 6px;border-radius:3px;margin:2px;'>"
                         f"{dtype} {c} ({pct}%)</span> ")

        self.summary_box.setHtml(
            f"<pre style='font-size:10px'>{text}</pre>"
            f"<p style='margin:8px'>{bar_html}</p>"
        )

    def _on_export(self):
        if not self._plan:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter plan GGUF", "quant_plan.json",
            "JSON (*.json)")
        if not path:
            return
        from d2_production import export_gguf
        cmd = export_gguf(self._plan, path)
        lines = cmd.split('\n')
        self.status.showMessage(
            f"Exporté → {path}  ({len(lines)} tensors)")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark-ish palette
    palette = app.palette()
    palette.setColor(palette.Window, QColor(45, 45, 45))
    palette.setColor(palette.WindowText, QColor(220, 220, 220))
    palette.setColor(palette.Base, QColor(35, 35, 35))
    palette.setColor(palette.AlternateBase, QColor(50, 50, 50))
    palette.setColor(palette.ToolTipBase, QColor(60, 60, 60))
    palette.setColor(palette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(palette.Text, QColor(220, 220, 220))
    palette.setColor(palette.Button, QColor(60, 60, 60))
    palette.setColor(palette.ButtonText, QColor(220, 220, 220))
    palette.setColor(palette.BrightText, QColor(255, 80, 80))
    palette.setColor(palette.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    win = D2Window()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
