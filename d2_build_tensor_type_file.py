"""Traduit le profil ECO de d2_tensor_optim.json (precision par tenseur HF)
en fichier --tensor-type-file pour llama-quantize (noms de tenseurs GGUF).

Mapping HF -> GGUF etabli a partir des logs de conversion reels
(beellama.cpp/conversion/qwen.py, architecture QWEN35) :
  self_attn.{q,k,v}_proj      -> attn_{q,k,v}
  self_attn.o_proj            -> attn_output
  mlp.gate_proj / up_proj / down_proj -> ffn_gate / ffn_up / ffn_down
  linear_attn.in_proj_qkv     -> attn_qkv   (couches lineaires uniquement)
  linear_attn.in_proj_z       -> attn_gate
  linear_attn.out_proj        -> ssm_out
  mtp.layers.0.*              -> blk.<block_count>.*  (block_count = 64, meme
                                 convention de noms que les couches normales,
                                 confirme par le log de quantize : blk.64.attn_q...)
"""
import argparse
import json
import os
import re

BLOCK_COUNT = 64  # qwen35.block_count(65) - 1 = index de la couche MTP (blk.64)

HERE = os.path.dirname(os.path.abspath(__file__))

ROLE_MAP = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
}

PREC_MAP = {"Q2": "q2_K", "Q3": "q3_K", "Q4": "q4_K", "FP8": "q8_0"}

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
MTP_RE = re.compile(r"^mtp\.layers\.(\d+)\.(.+)$")


def hf_to_gguf(name: str) -> str | None:
    m = LAYER_RE.match(name)
    if m:
        idx, role = m.group(1), m.group(2)
        gguf_role = ROLE_MAP.get(role)
        return f"blk.{idx}.{gguf_role}" if gguf_role else None
    m = MTP_RE.match(name)
    if m:
        role = m.group(2)
        gguf_role = ROLE_MAP.get(role)
        return f"blk.{BLOCK_COUNT}.{gguf_role}" if gguf_role else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="ECO", choices=["ECO", "BALANCED", "PERFORMANCE"])
    args = ap.parse_args()

    with open(os.path.join(HERE, "d2_tensor_optim.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    profile = next(p for p in d["profiles"] if p["profile"] == args.profile)

    lines = []
    unmapped = []
    for item in profile["quality_detail"]:
        gguf_name = hf_to_gguf(item["name"])
        if gguf_name is None:
            unmapped.append(item["name"])
            continue
        ggml_type = PREC_MAP[item["prec"]]
        lines.append(f"{gguf_name}={ggml_type}")

    print(f"{len(lines)} tenseurs mappes, {len(unmapped)} non mappes")
    if unmapped:
        print("non mappes (exemples):", unmapped[:5])

    out_name = os.path.join(HERE, f"d2_tensor_types_{args.profile.lower()}.txt")
    with open(out_name, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"-> {out_name}")

    from collections import Counter
    c = Counter(l.split("=")[1] for l in lines)
    print("repartition:", dict(c))
    ffn_up = [l for l in lines if ".ffn_up." in l]
    q2_ffn_up = [l for l in ffn_up if "=q2_K" in l]
    print(f"ffn_up: {len(ffn_up)} tenseurs, {len(q2_ffn_up)} en q2_K")


if __name__ == "__main__":
    main()
