# ============================================================
#  LAMA TOUR - REBUILD 23/08/2026
#  Rebuild frais de beellama.cpp avec GGML_CUDA_FA_ALL_QUANTS=ON
#  (flag crucial pour TurboQuant — absent du build précédent !)
# ============================================================
$ErrorActionPreference = "Stop"

$Root      = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "beellama.cpp"
$CondaEnv  = "C:\Users\videl\miniconda3\envs\cudabuild\Library"
$VcVars    = "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"

Write-Output "[*] Chargement de l'environnement MSVC (vcvars64)..."
$envDump = cmd /c "`"$VcVars`" >nul 2>&1 && set"
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

$env:PATH             = "$CondaEnv\bin;$env:PATH"
$env:CUDA_PATH        = $CondaEnv
$env:CUDAToolkit_ROOT = $CondaEnv

Write-Output "[*] nvcc :"
nvcc --version

Set-Location $Root

# Nettoyer l'ancien build
if (Test-Path "build-cuda") {
    Write-Output "[*] Nettoyage build-cuda..."
    Remove-Item -Recurse -Force "build-cuda"
}

Write-Output "[*] Configuration CMake (Ninja, CUDA natif, FA_ALL_QUANTS, Release)..."
cmake -B build-cuda -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON `
    -DGGML_NATIVE=ON `
    -DGGML_CUDA_FA=ON `
    -DGGML_CUDA_FA_ALL_QUANTS=ON `
    -DLLAMA_BUILD_TOOLS=ON `
    -DLLAMA_BUILD_SERVER=ON `
    -DLLAMA_BUILD_EXAMPLES=ON `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_APP=ON
if ($LASTEXITCODE -ne 0) { Write-Output "CONFIGURE_FAILED"; exit 1 }

Write-Output "[*] Compilation (peut prendre 15-40 min pour les kernels CUDA)..."
cmake --build build-cuda -j 20
if ($LASTEXITCODE -ne 0) { Write-Output "BUILD_FAILED"; exit 1 }

Write-Output "BUILD_OK"