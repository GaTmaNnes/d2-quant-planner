#!/usr/bin/env python3
"""
D2 Gradio Web Interface
=======================

Easy-to-use web UI for quantization planning.
Supports HuggingFace model IDs and local safetensors files.

Usage:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import json
import os
import tempfile
from typing import List, Dict, Tuple
import gradio as gr

from d2_production import solve_quantization_plan, summarize, export_gguf


def run_planner(
    model_id: str,
    vram_budget: float,
    w_risk: float,
    example_layers: str = None
) -> Tuple[str, str, str, str]:
    """
    Run the quantization planner.
    
    Args:
        model_id: HuggingFace model ID or path to safetensors file
        vram_budget: VRAM budget in GB
        w_risk: Risk weight parameter (λ)
        example_layers: Optional JSON string with layer definitions
    
    Returns:
        (table_html, summary_text, export_command, status_message)
    """
    try:
        # Load or generate layers
        if example_layers and example_layers.strip():
            try:
                layers = json.loads(example_layers)
            except json.JSONDecodeError as e:
                return "", "", "", f"❌ Invalid JSON: {e}"
        else:
            # Load from model
            layers = _load_layers_from_model(model_id)
        
        if not layers:
            return "", "", "", "❌ No valid layers found"
        
        # Solve quantization plan
        plan = solve_quantization_plan(
            layers,
            vram_budget_gb=vram_budget,
            w_speed=1.0,
            w_risk=w_risk
        )
        
        # Generate outputs
        table_html = _generate_table_html(plan)
        summary_text = summarize(plan, vram_budget, 1.0, w_risk)
        
        # Export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        export_cmd = export_gguf(plan, temp_path)
        
        vram_used = sum(e['vram_gb'] for e in plan)
        status = (
            f"✅ Plan solved successfully!\n"
            f"• {len(plan)} layers\n"
            f"• {vram_used:.3f} / {vram_budget:.1f} GB ({vram_used/vram_budget*100:.1f}%)\n"
            f"• Plan exported to {temp_path}"
        )
        
        return table_html, summary_text, export_cmd, status
        
    except Exception as e:
        return "", "", "", f"❌ Error: {str(e)}"


def _load_layers_from_model(model_id: str) -> List[Dict]:
    """Load layer definitions from HuggingFace or local safetensors."""
    try:
        import safetensors.torch as st
        
        # Check if local file
        if os.path.isfile(model_id):
            tensors = st.load_file(model_id)
        else:
            # Download from HuggingFace
            from huggingface_hub import hf_hub_download, list_repo_files
            
            files = [f for f in list_repo_files(model_id)
                     if f.endswith('.safetensors') and 'onnx' not in f]
            
            if not files:
                raise FileNotFoundError(f"No .safetensors files in {model_id}")
            
            local_path = hf_hub_download(
                model_id,
                files[0],
                cache_dir=os.path.expanduser("~/.cache/huggingface/hub")
            )
            tensors = st.load_file(local_path)
        
        # Extract layer shapes
        layers = []
        for name, tensor in tensors.items():
            if tensor.ndim < 2:
                continue
            
            # Reshape if needed
            shape = list(tensor.shape)
            if len(shape) > 2:
                shape = [shape[0], -1]
            
            m, n = shape[0], shape[1]
            
            # Filter small layers
            if m < 8 or n < 8:
                continue
            
            layers.append({'name': name, 'shape': [m, n]})
        
        return layers
        
    except Exception as e:
        # Return example layers on error
        return _get_example_layers()


def _get_example_layers() -> List[Dict]:
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


def _generate_table_html(plan: List[Dict]) -> str:
    """Generate HTML table for the plan."""
    html = '<table style="width:100%; border-collapse: collapse;">'
    html += '<thead style="background: #f0f0f0;">'
    html += '<tr>'
    html += '<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Layer Name</th>'
    html += '<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Type</th>'
    html += '<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Dtype</th>'
    html += '<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Score</th>'
    html += '<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">VRAM (GB)</th>'
    html += '</tr>'
    html += '</thead>'
    html += '<tbody>'
    
    dtype_colors = {
        'FP16': '#4682B4',
        'INT8': '#3CB371',
        'INT4': '#FF8C00',
    }
    
    for item in plan:
        dtype = item['dtype']
        color = dtype_colors.get(dtype, '#999999')
        html += '<tr>'
        html += f'<td style="border: 1px solid #ddd; padding: 8px;">{item["name"][:60]}</td>'
        html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{item["layer_type"]}</td>'
        html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center; background: {color}; color: white; font-weight: bold;">{dtype}</td>'
        html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{item["score"]:+.4f}</td>'
        html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{item["vram_gb"]:.6f}</td>'
        html += '</tr>'
    
    html += '</tbody>'
    html += '</table>'
    return html


# ─── Gradio Interface ────────────────────────────────────────────────────────

with gr.Blocks(title="D2 Quantization Planner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 D2 — Spectral-Aware Quantization Planner
    
    Advanced layer-wise quantization planning for **llama.cpp / GGUF**
    
    Combine spectral analysis, RG Flow, and VRAM-constrained optimization to produce mixed quantization plans (FP16 / INT8 / INT4).
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 📝 Configuration")
            
            model_id_input = gr.Textbox(
                value="gpt2",
                label="Model ID (HuggingFace) or Local Path",
                placeholder="e.g., gpt2, TinyLlama/TinyLlama-1.1B-Chat-v1.0, or /path/to/model.safetensors",
                lines=1
            )
            
            vram_budget_slider = gr.Slider(
                minimum=2,
                maximum=128,
                value=8.0,
                step=0.5,
                label="VRAM Budget (GB)",
                info="Available VRAM for quantized model"
            )
            
            w_risk_slider = gr.Slider(
                minimum=0.1,
                maximum=5.0,
                value=0.4,
                step=0.1,
                label="w_risk (λ)",
                info="Higher = more conservative (FP16), Lower = more aggressive (INT4)"
            )
            
            gr.Markdown("## 📋 Advanced")
            
            example_layers_input = gr.Textbox(
                label="Custom Layers (JSON - optional)",
                placeholder='[{"name": "layer1", "shape": [4096, 4096]}, ...]',
                lines=3,
                info="Leave empty to auto-load from model"
            )
            
            solve_button = gr.Button("▶ Generate Plan", variant="primary", scale=2)
    
    with gr.Column():
        gr.Markdown("## 📊 Results")
        
        status_output = gr.Textbox(
            label="Status",
            lines=4,
            interactive=False,
            value="Click 'Generate Plan' to start"
        )
        
        summary_output = gr.Textbox(
            label="Summary",
            lines=6,
            interactive=False
        )
    
    gr.Markdown("## 📈 Quantization Plan")
    table_output = gr.HTML()
    
    gr.Markdown("## 💾 Export Command (llama.cpp)")
    export_output = gr.Textbox(
        label="Export Command",
        interactive=False,
        lines=4
    )
    
    # Connect button
    solve_button.click(
        run_planner,
        inputs=[model_id_input, vram_budget_slider, w_risk_slider, example_layers_input],
        outputs=[table_output, summary_output, export_output, status_output]
    )
    
    gr.Markdown("""
    ---
    
    ## 📚 Usage Tips
    
    1. **Model ID**: Use any HuggingFace model ID (e.g., `gpt2`, `TinyLlama/TinyLlama-1.1B`) or path to local `.safetensors`
    2. **VRAM Budget**: Set to your available VRAM
    3. **w_risk**: Control aggressiveness:
       - 0.1–0.3: Aggressive (more INT4)
       - 0.4–0.6: Balanced
       - 0.8–1.0+: Conservative (more FP16)
    4. **Custom Layers**: Advanced users can provide their own layer definitions as JSON
    
    ## 🔗 Links
    
    - [GitHub Repository](https://github.com/GaTmaNnes/d2-quant-planner)
    - [Report Issues](https://github.com/GaTmaNnes/d2-quant-planner/issues)
    - [llama.cpp](https://github.com/ggerganov/llama.cpp)
    """)


if __name__ == "__main__":
    demo.launch(share=True)
