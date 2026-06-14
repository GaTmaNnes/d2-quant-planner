#!/usr/bin/env python3
"""
D2 UI — PySide6 Interface for Spectral Layer-wise Quantization Planner
"""
import sys
import json
from pathlib import Path
from typing import List, Dict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QSlider, QPushButton, QTableWidget, QTableWidgetItem,
    QTextEdit, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
    QProgressBar, QGroupBox, QGridLayout
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from d2_production import solve_quantization_plan, summarize, export_gguf


class QuantizationWorker(QThread):
    """Worker thread for non-blocking quantization planning."""
    finished = Signal(list, str, str)
    error = Signal(str)
    
    def __init__(self, layers: List[Dict], vram_budget: float, w_risk: float):
        super().__init__()
        self.layers = layers
        self.vram_budget = vram_budget
        self.w_risk = w_risk
    
    def run(self):
        try:
            plan = solve_quantization_plan(
                self.layers,
                vram_budget_gb=self.vram_budget,
                w_speed=1.0,
                w_risk=self.w_risk
            )
            summary_text = summarize(plan, self.vram_budget, 1.0, self.w_risk)
            json_cmd = export_gguf(plan, "current_plan.json")
            self.finished.emit(plan, summary_text, json_cmd)
        except Exception as e:
            self.error.emit(str(e))


class D2QuantPlannerUI(QMainWindow):
    """Main UI window for D2 Quantization Planner."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("D2 — Spectral-Aware Quantization Planner")
        self.setGeometry(100, 100, 1400, 900)
        self.current_plan = None
        self.current_layers = self._get_example_layers()
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("D2 — Spectral Layer-wise Quantization Planner")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # Control Panel
        control_group = QGroupBox("Configuration")
        control_layout = QGridLayout()
        
        # VRAM Budget
        control_layout.addWidget(QLabel("VRAM Budget (GB):"), 0, 0)
        self.vram_spinbox = QDoubleSpinBox()
        self.vram_spinbox.setRange(2, 128)
        self.vram_spinbox.setValue(8.0)
        self.vram_spinbox.setSingleStep(0.5)
        control_layout.addWidget(self.vram_spinbox, 0, 1)
        
        # w_risk Parameter
        control_layout.addWidget(QLabel("w_risk (λ):"), 0, 2)
        self.risk_spinbox = QDoubleSpinBox()
        self.risk_spinbox.setRange(0.1, 3.0)
        self.risk_spinbox.setValue(0.4)
        self.risk_spinbox.setSingleStep(0.05)
        control_layout.addWidget(self.risk_spinbox, 0, 3)
        
        # Model ID Input
        control_layout.addWidget(QLabel("Model ID (optional):"), 1, 0)
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("e.g., Qwen/Qwen3.5-9B")
        control_layout.addWidget(self.model_input, 1, 1, 1, 3)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("🚀 Generate Plan")
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.run_button.clicked.connect(self.run_planner)
        button_layout.addWidget(self.run_button)
        
        self.export_button = QPushButton("💾 Export JSON")
        self.export_button.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        self.export_button.clicked.connect(self.export_plan)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.export_button)
        
        self.load_button = QPushButton("📂 Load Layers")
        self.load_button.setStyleSheet("background-color: #FF9800; color: white; padding: 8px;")
        self.load_button.clicked.connect(self.load_layers)
        button_layout.addWidget(self.load_button)
        
        main_layout.addLayout(button_layout)
        
        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)
        
        # Results Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Layer Name", "Type", "Dtype", "Score", "VRAM (GB)"])
        self.table.setColumnWidth(0, 400)
        main_layout.addWidget(self.table)
        
        # Summary
        summary_label = QLabel("Summary:")
        summary_font = QFont()
        summary_font.setBold(True)
        summary_label.setFont(summary_font)
        main_layout.addWidget(summary_label)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        main_layout.addWidget(self.summary_text)
        
        # Export Command
        export_label = QLabel("llama.cpp Command:")
        export_label.setFont(summary_font)
        main_layout.addWidget(export_label)
        
        self.export_text = QTextEdit()
        self.export_text.setReadOnly(True)
        self.export_text.setMaximumHeight(100)
        main_layout.addWidget(self.export_text)
    
    def _get_example_layers(self) -> List[Dict]:
        """Get example layers for demonstration."""
        return [
            {"name": "model.layers.0.self_attn.q_proj.weight", "shape": [4096, 4096]},
            {"name": "model.layers.0.self_attn.k_proj.weight", "shape": [1024, 4096]},
            {"name": "model.layers.0.self_attn.v_proj.weight", "shape": [1024, 4096]},
            {"name": "model.layers.0.self_attn.o_proj.weight", "shape": [4096, 4096]},
            {"name": "model.layers.0.mlp.gate_proj.weight", "shape": [14336, 4096]},
            {"name": "model.layers.0.mlp.up_proj.weight", "shape": [14336, 4096]},
            {"name": "model.layers.0.mlp.down_proj.weight", "shape": [4096, 14336]},
            {"name": "lm_head.weight", "shape": [128256, 4096]},
        ]
    
    def run_planner(self):
        """Run the quantization planner."""
        self.run_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate
        
        vram_budget = self.vram_spinbox.value()
        w_risk = self.risk_spinbox.value()
        
        self.worker = QuantizationWorker(self.current_layers, vram_budget, w_risk)
        self.worker.finished.connect(self.on_plan_finished)
        self.worker.error.connect(self.on_plan_error)
        self.worker.start()
    
    def on_plan_finished(self, plan: List[Dict], summary: str, export_cmd: str):
        """Handle finished quantization plan."""
        self.current_plan = plan
        
        # Update table
        self.table.setRowCount(len(plan))
        for i, item in enumerate(plan):
            self.table.setItem(i, 0, QTableWidgetItem(item["name"][:60]))
            self.table.setItem(i, 1, QTableWidgetItem(item["layer_type"]))
            self.table.setItem(i, 2, QTableWidgetItem(item["dtype"]))
            self.table.setItem(i, 3, QTableWidgetItem(str(item["score"])))
            self.table.setItem(i, 4, QTableWidgetItem(f"{item['vram_gb']:.6f}"))
        
        # Update summary
        self.summary_text.setPlainText(summary)
        self.export_text.setPlainText(export_cmd)
        
        # Re-enable buttons
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.progress.setVisible(False)
        
        QMessageBox.information(self, "Success", "Quantization plan generated successfully!")
    
    def on_plan_error(self, error_msg: str):
        """Handle planner error."""
        self.run_button.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Error", f"Planner error: {error_msg}")
    
    def export_plan(self):
        """Export the current plan to JSON."""
        if self.current_plan is None:
            QMessageBox.warning(self, "Warning", "No plan to export. Generate a plan first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Quantization Plan",
            "quant_plan.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            export_gguf(self.current_plan, file_path)
            QMessageBox.information(self, "Success", f"Plan exported to {file_path}")
    
    def load_layers(self):
        """Load layer definitions from JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Layer Definitions",
            "",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.current_layers = data
                    QMessageBox.information(self, "Success", f"Loaded {len(data)} layers.")
                else:
                    QMessageBox.warning(self, "Error", "JSON must contain a list of layers.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {e}")


def main():
    """Run the application."""
    app = QApplication(sys.argv)
    window = D2QuantPlannerUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
