# ============================================================
# FreeToken Windows — installation complète SAUF reboot
# Usage :
#   .\freetoken_windows.ps1              -> configure tout, n'essaie pas de servir
#   .\freetoken_windows.ps1 -Launch      -> configure PUIS lance ft serve sur le 35B FP8
# ============================================================
param([switch]$Launch)

$ErrorActionPreference = "Continue"
$FT_DIR   = "C:\Users\videl\Desktop\FreeToken"
$FT_VENV  = "$FT_DIR\.venv-win\Scripts"
$CUDA     = "C:\Users\videl\miniconda3\envs\cudabuild\Library"
$MODELS   = "C:\Users\videl\Desktop\lama 1080-5070\hf_weights_35b_fp8"

function Commit-State {
    $os = Get-CimInstance Win32_OperatingSystem
    "{0:N1} GB libres / {1:N1} GB" -f ($os.FreeVirtualMemory/1MB), ($os.TotalVirtualMemorySize/1MB)
}

Write-Host "== 1. Liberation du commit memoire ==" -ForegroundColor Cyan
Get-Process ft, llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
wsl.exe --shutdown 2>$null          # libere vmmem (WSL prend plusieurs GB de commit)
Start-Sleep 3
"Commit apres nettoyage : $(Commit-State)"

Write-Host "`n== 2. Pagefile 64 GB ==" -ForegroundColor Cyan
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($isAdmin) {
    $cs = Get-CimInstance Win32_ComputerSystem
    if ($cs.AutomaticManagedPagefile) {
        Set-CimInstance -InputObject $cs -Property @{AutomaticManagedPagefile=$false}
        "gestion automatique desactivee"
    }
    Set-CimInstance -Query "SELECT * FROM Win32_PageFileSetting WHERE Name='C:\\pagefile.sys'" `
        -Property @{InitialSize=65536; MaximumSize=65536} -ErrorAction SilentlyContinue
    if (-not $?) {
        New-CimInstance -ClassName Win32_PageFileSetting -Property @{Name="C:\pagefile.sys"; InitialSize=65536; MaximumSize=65536} | Out-Null
    }
    "pagefile C: programmee a 64 GB (prendra effet au prochain reboot)" -replace '.',""
    Write-Host "[OK] pagefile 65536 MB programmee - effet complet au prochain reboot" -ForegroundColor Green
} else {
    Write-Host "[SKIP] pas admin -> pagefile non modifiee." -ForegroundColor Yellow
    Write-Host "Commande a lancer dans un PowerShell ADMINISTRateur :"
    Write-Host '  Set-CimInstance -Query "SELECT * FROM Win32_PageFileSetting WHERE Name=''C:\\pagefile.sys''" -Property @{InitialSize=65536; MaximumSize=65536}'
}

Write-Host "`n== 3. Variables d'environnement persistantes ==" -ForegroundColor Cyan
setx CUDA_HOME $CUDA | Out-Null
setx FREETOKEN_ALLOW_CUDA_MISMATCH 1 | Out-Null
"[OK] CUDA_HOME + FREETOKEN_ALLOW_CUDA_MISMATCH=1 persistes (nouveaux shells)"
$env:CUDA_HOME = $CUDA
$env:FREETOKEN_ALLOW_CUDA_MISMATCH = "1"
$env:PATH = "$CUDA\bin;$env:PATH"

Write-Host "`n== 4. Launcher ft-serve-35b.cmd ==" -ForegroundColor Cyan
$launcher = @"
@echo off
REM FreeToken natif Windows - Qwen3.6-35B-A3B FP8 - backend offload
set CUDA_HOME=$CUDA
set FREETOKEN_ALLOW_CUDA_MISMATCH=1
set PATH=%CUDA_HOME%\bin;%PATH%
echo [FreeToken] chargement du 35B FP8 (~5 min au premier demarrage)...
"$FT_VENV\ft.exe" serve --model-path "$MODELS" --moe-backend offload --host 127.0.0.1 --port 1919
"@
Set-Content "C:\Users\videl\Desktop\lama 1080-5070\ft-serve-35b.cmd" $launcher -Encoding ASCII
Write-Host "[OK] C:\Users\videl\Desktop\lama 1080-5070\ft-serve-35b.cmd"

Write-Host "`n== 5. Verification installation ==" -ForegroundColor Cyan
& "$FT_VENV\python.exe" -c "import torch; assert torch.cuda.is_available(), 'torch sans CUDA !'; import freetoken; print('torch', torch.__version__, '| CUDA OK | freetoken OK')"

if (-not $Launch) {
    Write-Host "`n[TERMINE] Pour lancer le serveur : .\ft-serve-35b.cmd   (ou relance ce script avec -Launch)" -ForegroundColor Green
    exit 0
}

Write-Host "`n== 6. Lancement ft serve 35B FP8 (offload) ==" -ForegroundColor Cyan
if ((Commit-State) -match "^(\d+)\.") { }
$p = Start-Process -FilePath "$FT_VENV\ft.exe" -ArgumentList 'serve', "--model-path=`"$MODELS`"", '--moe-backend','offload','--host','127.0.0.1','--port','1919' `
     -RedirectStandardOutput "$env:TEMP\ft35_out.log" -RedirectStandardError "$env:TEMP\ft35_err.log" -PassThru
"PID $($p.Id) - attente pret (jusqu'a 12 min)..."
$ready = $false
foreach ($i in 1..72) {
    Start-Sleep 10
    try { $null = Invoke-RestMethod -Uri "http://127.0.0.1:1919/v1/models" -TimeoutSec 3; $ready=$true; break } catch {}
    if ($p.HasExited) { break }
}
if (-not $ready) { Write-Host "[ECHEC] serveur non pret. Log:" -ForegroundColor Red; Get-Content "$env:TEMP\ft35_err.log" -Tail 8; exit 1 }

Write-Host "== 7. Generation de test ==" -ForegroundColor Cyan
$b = '{"model":"default","messages":[{"role":"user","content":"Explique la quantification MoE en deux phrases."}],"max_tokens":128,"temperature":0}'
$r = Invoke-RestMethod -Uri "http://127.0.0.1:1919/v1/chat/completions" -Method Post -Body $b -ContentType "application/json" -TimeoutSec 600
$t = $r.usage.completion_tokens
"Reponse : $($r.choices[0].message.content.Substring(0,[Math]::Min(160,$r.choices[0].message.content.Length)))..."
Write-Host ("[RESULTAT] {0} tokens en {1:N1}s" -f $t, $r.usage.total_tokens) 
$vram = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
"VRAM utilisee : $([math]::Round($vram/1024,2)) GB"
"Serveur laisse tourner (PID $($p.Id)) - http://127.0.0.1:1919"
