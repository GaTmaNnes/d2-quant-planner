$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Corpus = "$Root\corpus\wiki.test.raw"
$Exe = "$Root\llama-perplexity.exe"

$models = @(
    @{ name = "REF-hybrid-Q8_0-BF16"; path = "$Root\models\Qwen3.8-27B-hybrid.gguf" },
    @{ name = "Q4_K_M-uniform";       path = "$Root\models\Qwen3.8-27B-Q4_K_M.gguf" },
    @{ name = "D2-ECO-selectif";      path = "$Root\models\Qwen3.8-27B-D2-ECO.gguf" }
)

foreach ($m in $models) {
    Write-Output "=============================================="
    Write-Output "PPL START: $($m.name)"
    Write-Output "=============================================="
    & $Exe -m $m.path -f $Corpus -c 1024 --chunks 50 -fitt 1024 -fa 1
    Write-Output "PPL DONE: $($m.name)"
}
Write-Output "ALL_PPL_OK"
