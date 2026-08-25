# ============================================================
#  Test reel du 27B dense (Qwen3.8-27B-FP8) sur RTX 5070 8 Go
#  1) HF safetensors (FP8) -> GGUF (fp8-as-q8, auto pour le reste)
#  2) Quantize -> Q4_K_M (pour faire tenir plus de couches en VRAM)
#  3) llama-bench reel avec offload partiel (-ngl auto, --fit on)
# ============================================================
$ErrorActionPreference = "Stop"
$Root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$Conv    = "C:\Users\videl\miniconda3\envs\AMD_Quark\python.exe"
$SrcDir  = "$Root\models\Qwen3.8-27B-FP8"
$GgufOut = "$Root\models\Qwen3.8-27B-hybrid.gguf"
$QuantOut= "$Root\models\Qwen3.8-27B-Q4_K_M.gguf"

Set-Location "$Root\beellama.cpp"

Write-Output "[1/3] Conversion HF -> GGUF (fp8-as-q8, auto)..."
& $Conv convert_hf_to_gguf.py "$SrcDir" --outtype auto --fp8-as-q8 --outfile "$GgufOut"
if ($LASTEXITCODE -ne 0) { Write-Output "CONVERT_FAILED"; exit 1 }
Write-Output "CONVERT_OK"

Write-Output "[2/3] Quantize -> Q4_K_M..."
& "$Root\llama-quantize.exe" --allow-requantize "$GgufOut" "$QuantOut" Q4_K_M
if ($LASTEXITCODE -ne 0) { Write-Output "QUANTIZE_FAILED"; exit 1 }
Write-Output "QUANTIZE_OK"

Write-Output "[3/3] llama-bench reel (offload partiel automatique via -fitt)..."
& "$Root\llama-bench.exe" -m "$QuantOut" -ngl 99 -fitt 1024 -p 512 -n 128 -fa 1
if ($LASTEXITCODE -ne 0) { Write-Output "BENCH_FAILED"; exit 1 }

Write-Output "ALL_OK"
