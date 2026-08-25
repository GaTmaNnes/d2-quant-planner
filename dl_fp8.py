import os, sys, time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3.8-27B-FP8")
os.makedirs(DEST, exist_ok=True)
print(f"[dl] target: {DEST}", flush=True)
t0 = time.time()
try:
    path = snapshot_download(
        repo_id="Qwen/Qwen3.8-27B-FP8",
        revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        local_dir=DEST,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.md", "*.jinja", "*tokenizer*", "*.model", "*.py"],
        max_workers=8,
    )
    print(f"[dl] OK {path} en {time.time()-t0:.0f}s", flush=True)
except Exception as e:
    print(f"[dl] ERROR {e}", flush=True)
    sys.exit(1)