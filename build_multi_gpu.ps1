# ============================================================
#  LAMA TOUR - Build CUDA + Vulkan (RTX 5070 SM120 + Radeon 880M)
#  Un seul binaire, deux backends GPU, dossier dedie "multi-gpu/".
# ============================================================
$ErrorActionPreference = "Stop"

$Root      = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "beellama.cpp"
$CondaEnv  = "C:\Users\videl\miniconda3\envs\cudabuild\Library"
$VcVars    = "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
$BuildDir  = "build-multi"

Write-Output "[*] Chargement de l'environnement MSVC (vcvars64)..."
$envDump = cmd /c "`"$VcVars`" >nul 2>&1 && set"
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

$env:PATH             = "$CondaEnv\bin;$CondaEnv\nvvm\bin;$env:PATH"
$env:CUDA_PATH        = $CondaEnv
$env:CUDAToolkit_ROOT = $CondaEnv
$env:VULKAN_SDK       = $CondaEnv

Write-Output "[*] nvcc :"
nvcc --version
Write-Output "[*] glslc :"
glslc --version

Set-Location $Root

Write-Output "[*] Configuration CMake (CUDA natif sm_120 + Vulkan, Release)..."
cmake -B $BuildDir -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON `
    -DGGML_VULKAN=ON `
    -DGGML_NATIVE=ON `
    -DLLAMA_BUILD_TOOLS=ON `
    -DLLAMA_BUILD_SERVER=ON `
    -DLLAMA_BUILD_EXAMPLES=ON `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_APP=ON
if ($LASTEXITCODE -ne 0) { Write-Output "CONFIGURE_FAILED"; exit 1 }

Write-Output "[*] Compilation (CUDA + shaders Vulkan, peut prendre 20-50 min)..."
cmake --build $BuildDir -j 20
if ($LASTEXITCODE -ne 0) { Write-Output "BUILD_FAILED"; exit 1 }

Write-Output "BUILD_OK"
