$ErrorActionPreference = "Stop"
$Root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$GgufOut = "$Root\models\Qwen3.8-27B-hybrid.gguf"
$QuantOut= "$Root\models\Qwen3.8-27B-Q4_K_M.gguf"

Write-Output "[2/3] Quantize -> Q4_K_M (--allow-requantize)..."
& "$Root\llama-quantize.exe" --allow-requantize "$GgufOut" "$QuantOut" Q4_K_M
if ($LASTEXITCODE -ne 0) { Write-Output "QUANTIZE_FAILED"; exit 1 }
Write-Output "QUANTIZE_OK"

Write-Output "[3/3] llama-bench reel (offload partiel automatique via -fitt)..."
& "$Root\llama-bench.exe" -m "$QuantOut" -ngl 99 -fitt 1024 -p 512 -n 128 -fa 1
if ($LASTEXITCODE -ne 0) { Write-Output "BENCH_FAILED"; exit 1 }

Write-Output "ALL_OK"
