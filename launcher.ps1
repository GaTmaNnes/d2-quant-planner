# ============================================================
#  LAMA TOUR - Lanceur interactif (PowerShell)
#  - Lancer un LLM optimisé pour le GPU détecté
#  - Quantifier un GGUF
#  - Fine-tuner un modèle (llama-finetune)
#
#  Lancement :  clic droit > "Exécuter avec PowerShell"
#  ou en console :  powershell -ExecutionPolicy Bypass -File .\launcher.ps1
# ============================================================

$ErrorActionPreference = "Continue"

# --- Racine du script (gère les chemins avec espaces) ---
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

$ServerExe    = Join-Path $Root "llama-server.exe"
# [CORRIGE 25/08/2026] SEUL binaire DFlash fonctionnel : build beellama.cpp\build-cuda\bin
# (fixes draft 24/08, cudart64_13.dll a cote). Le binaire racine est un build 23/08
# SANS fixes -> crash draft « invalid vector subscript ».
$ServerExeFixed = Join-Path $Root "beellama.cpp\build-cuda\bin\llama-server.exe"
$QuantizeExe  = Join-Path $Root "tools\llama-quantize.exe"
$FinetuneExe  = Join-Path $Root "tools\llama-finetune.exe"
$ModelsDir    = Join-Path $Root "models"
$CorpusDir    = Join-Path $Root "corpus"

# ============================================================
#  Détection GPU
# ============================================================
function Get-GpuInfo {
    $gpu = @{ Name = "CPU (aucun GPU NVIDIA)"; CacheType = ""; HasCuda = $false; Label = "CPU" }
    try {
        $raw = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null
        if ($raw -and $raw[0]) {
            $line  = ($raw | Select-Object -First 1) -replace "`r|`n", ""
            $parts = $line -split ","
            $gpu.Name = $parts[0].Trim()
            $cap  = $parts[1].Trim()
            $gpu.HasCuda = $true
            if ($cap -match "^6\.") {
                $gpu.CacheType = "turbo3"
                $gpu.Label = "Pascal (GTX 10xx)"
            }
            elseif ($cap -match "^10\.|^12\.") {
                $gpu.CacheType = "turbo4"
                $gpu.Label = "Blackwell (RTX 50xx / sm_100+)"
            }
            else {
                $gpu.CacheType = "turbo3"
                $gpu.Label = "NVIDIA générique"
            }
        }
    } catch {}
    return $gpu
}

# ============================================================
#  Aides
# ============================================================
function Read-Choice($prompt, $min, $max) {
    while ($true) {
        $v = Read-Host $prompt
        if ($v -match '^\d+$' -and [int]$v -ge $min -and [int]$v -le $max) {
            return [int]$v
        }
        Write-Host "  Choix invalide (entre $min et $max)." -ForegroundColor Red
    }
}

function Get-Models {
    if (Test-Path $ModelsDir) {
        return @(Get-ChildItem -Path $ModelsDir -Filter *.gguf | Sort-Object Name)
    }
    return @()
}

function Confirm-Run($exe, $args) {
    Write-Host ""
    Write-Host "  Commande :" -ForegroundColor Cyan
    Write-Host ("  " + $exe + " " + ($args -join " ")) -ForegroundColor DarkGray
    Write-Host ""
    $go = Read-Host "  Lancer ? (O/n)"
    return ($go -ne "n" -and $go -ne "N")
}

# [CORRIGE 25/08/2026] Test de port occupe avant lancement (comme l'UI Tkinter) :
# lancer llama-server sur un port deja pris = echec obscur en plein demarrage.
function Test-PortBusy {
    param([int]$Port)
    try {
        return (Test-NetConnection -ComputerName 127.0.0.1 -Port $Port `
            -InformationLevel Quiet -WarningAction SilentlyContinue)
    } catch {
        return $false
    }
}

# ============================================================
#  MENU 1 : Lancer un LLM
# ============================================================
function Menu-Launch {
    Write-Host ""
    Write-Host "  ===== LANCER UN LLM =====" -ForegroundColor Yellow

    if (-not (Test-Path $ServerExe)) {
        Write-Host "  [ERREUR] $ServerExe introuvable." -ForegroundColor Red
        return
    }

    $gpu = Get-GpuInfo
    Write-Host ("  GPU détecté : {0}  ({1})" -f $gpu.Name, $gpu.Label) -ForegroundColor Green
    if ($gpu.HasCuda) {
        Write-Host ("  Cache KV     : {0}" -f $gpu.CacheType) -ForegroundColor Green
    }

    # --- Choix du modèle ---
    $models = Get-Models
    if ($models.Count -eq 0) {
        Write-Host "  Aucun .gguf trouvé dans $ModelsDir" -ForegroundColor Red
        $m = Read-Host "  Chemin complet du modèle (.gguf)"
        if (-not $m) { return }
    } else {
        Write-Host ""
        Write-Host "  Modèles disponibles :" -ForegroundColor White
        for ($i = 0; $i -lt $models.Count; $i++) {
            $sz = [math]::Round($models[$i].Length / 1GB, 2)
            Write-Host ("    {0}. {1}  ({2} Go)" -f ($i + 1), $models[$i].Name, $sz)
        }
        Write-Host ("    {0}. Autre chemin..." -f ($models.Count + 1))
        $c = Read-Choice "  Choix" 1 ($models.Count + 1)
        if ($c -le $models.Count) {
            $m = $models[$c - 1].FullName
        } else {
            $m = Read-Host "  Chemin complet du modèle (.gguf)"
            if (-not $m) { return }
        }
    }

    # --- Couches GPU selon la taille du modele ---
    # [CORRIGE 25/08/2026] -ngl 99 force quel que soit le modele => OOM / sur-allocation
    # VMM sur les gros GGUF. Prod mesuree (24/08) : -ngl 15 pour > 10 Go (D2-MOE 17,1 Go),
    # -ngl 99 seulement si le modele tient entierement en VRAM.
    $modelGB = 0.0
    if ($m -and (Test-Path $m)) { $modelGB = [math]::Round((Get-Item $m).Length / 1GB, 2) }
    $ngl = if ($modelGB -gt 10) { "15" } else { "99" }

    # --- Taille de contexte ---
    if ($gpu.HasCuda) {
        $defCtx = if ($gpu.CacheType -eq "turbo4") { 32768 } else { 16384 }
    } else {
        $defCtx = 4096
    }
    $ctx = Read-Host "  Taille de contexte (défaut $defCtx)"
    if ($ctx -eq "") { $ctx = $defCtx }

    # --- Port ---
    $port = Read-Host "  Port HTTP (défaut 8080)"
    if ($port -eq "") { $port = "8080" }

    # --- Draft model (DFlash2 speculative decoding) ---
    $draftEnabled = $false
    $draftPath = ""
    $useDraft = Read-Host "  Activer DFlash2 draft ? (o/N)"
    if ($useDraft -eq "o" -or $useDraft -eq "O" -or $useDraft -eq "y" -or $useDraft -eq "Y") {
        $draftEnabled = $true
        # [CORRIGE 25/08/2026] Ancienne heuristique « Q2_K|draft » : AUCUN GGUF present
        # ne matchait (pas de Q2_K dans models\). Defaut : draft OFFICIAL-BF16, le seul
        # valide (90,9% acceptance) ; filtre taille < 3 Go pour eviter les targets.
        $draftCandidates = @()
        if (Test-Path $ModelsDir) {
            $official = @(Get-ChildItem -Path $ModelsDir -Filter *.gguf |
                Where-Object { $_.Name -match "DFlash-OFFICIAL-BF16" } | Select-Object -First 1)
            if ($official) { $draftCandidates += $official }
            $draftCandidates += @(Get-ChildItem -Path $ModelsDir -Filter *.gguf |
                Where-Object { ($_.Length -lt 3GB) -and ($_.Name -match "draft|DFlash") } |
                Where-Object { -not $official -or $_.FullName -ne $official.FullName })
        }
        if ($draftCandidates.Count -gt 0) {
            Write-Host "  Draft models trouves :" -ForegroundColor White
            for ($i = 0; $i -lt $draftCandidates.Count; $i++) {
                Write-Host ("    {0}. {1}" -f ($i + 1), $draftCandidates[$i].Name)
            }
            $dc = Read-Choice "  Choix draft" 1 $draftCandidates.Count
            $draftPath = $draftCandidates[$dc - 1].FullName
        } else {
            $draftPath = Read-Host "  Chemin du draft model (.gguf)"
        }
        if (-not $draftPath -or -not (Test-Path $draftPath)) {
            Write-Host "  Draft model introuvable, mode standard." -ForegroundColor Yellow
            $draftEnabled = $false
        }
    }

    # [CORRIGE 25/08/2026] DFlash exige ctx <= 8192 (crash draft au-dela, valide 24/08)
    if ($draftEnabled -and [int]$ctx -gt 8192) {
        Write-Host "  Contexte $ctx > 8192 : ramene a 8192 (limite DFlash)." -ForegroundColor Yellow
        $ctx = "8192"
    }

    # --- Construction de la commande ---
    $args = @(
        "-m", $m,
        "--ctx-size", "$ctx",
        "--batch-size", "512",
        "--host", "127.0.0.1",
        "--port", "$port",
        "--parallel", "1"
    )
    if ($gpu.HasCuda) {
        # [CORRIGE 25/08/2026] $ngl depend de la taille du modele (voir plus haut)
        $args += @("-ngl", "$ngl", "--flash-attn", "on")
        $args += @("--cache-type-k", $gpu.CacheType)
        $args += @("--cache-type-v", $gpu.CacheType)
    } else {
        $args += @("-ngl", "0")
    }
    if ($draftEnabled) {
        # [CORRIGE 25/08/2026] n-max 5 -> 2 : meilleur acceptance (90,9%) et meilleur t/s.
        # Cache KV turbo incompatible avec un -ngl partiel (gros modele) -> bascule q4_0 :
        if ($gpu.CacheType -match "^turbo" -and $modelGB -gt 10) {
            Write-Host "  Cache $($gpu.CacheType) impossible (-ngl partiel sur > 10 Go) -> q4_0." -ForegroundColor Yellow
            $cleaned = @()
            for ($ai = 0; $ai -lt $args.Count; $ai++) {
                if ($args[$ai] -in @("--cache-type-k", "--cache-type-v")) { $ai++; continue }
                $cleaned += $args[$ai]
            }
            $args = $cleaned
            $args += @("--cache-type-k", "q4_0", "--cache-type-v", "q4_0")
        }
        $args += @("--model-draft", $draftPath, "--spec-type", "dflash", "--spec-draft-ngl", "99", `
                   "--spec-draft-n-max", "2", "--spec-draft-ctx-size", "512")
        Write-Host ("  DFlash2 draft : {0}" -f $draftPath) -ForegroundColor Cyan
    }

    # [CORRIGE 25/08/2026] Le mode DFlash doit tourner sur le binaire FIXE, pas le racine.
    $exeToRun = $ServerExe
    if ($draftEnabled) {
        if (Test-Path $ServerExeFixed) {
            $exeToRun = $ServerExeFixed
        } else {
            Write-Host "  [ERREUR] Binaire DFlash fixe introuvable :" -ForegroundColor Red
            Write-Host "    $ServerExeFixed" -ForegroundColor Red
            return
        }
    }

    # [CORRIGE 25/08/2026] Port occupe -> echec immediat explicite (comme l'UI).
    if (Test-PortBusy -Port ([int]$port)) {
        Write-Host ("  [ERREUR] Port {0} deja occupe par un autre processus." -f $port) -ForegroundColor Red
        Write-Host "  Arrete-le ou choisis un autre port avant de relancer." -ForegroundColor Red
        return
    }

    if (-not (Confirm-Run $exeToRun $args)) { return }

    Write-Host ""
    Write-Host "  Lancement du serveur... (Ctrl+C pour arrêter)" -ForegroundColor Green
    Write-Host ("  Interface : http://127.0.0.1:{0}" -f $port) -ForegroundColor Cyan
    Write-Host ""
    & $exeToRun @args
}

# ============================================================
#  MENU 2 : Quantifier un GGUF
# ============================================================
$QuantTypes = @(
    "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0",
    "IQ4_XS", "IQ3_M", "Q2_K", "F16", "BF16", "NVFP4"
)

function Menu-Quantize {
    Write-Host ""
    Write-Host "  ===== QUANTIFIER UN GGUF =====" -ForegroundColor Yellow

    if (-not (Test-Path $QuantizeExe)) {
        Write-Host "  [ERREUR] $QuantizeExe introuvable." -ForegroundColor Red
        return
    }

    # --- Modèle source ---
    $models = Get-Models
    if ($models.Count -gt 0) {
        Write-Host ""
        Write-Host "  Modèle source :" -ForegroundColor White
        for ($i = 0; $i -lt $models.Count; $i++) {
            Write-Host ("    {0}. {1}" -f ($i + 1), $models[$i].Name)
        }
        Write-Host ("    {0}. Autre chemin..." -f ($models.Count + 1))
        $c = Read-Choice "  Choix" 1 ($models.Count + 1)
        if ($c -le $models.Count) {
            $in = $models[$c - 1].FullName
        } else {
            $in = Read-Host "  Chemin complet du modèle source (.gguf)"
            if (-not $in) { return }
        }
    } else {
        $in = Read-Host "  Chemin complet du modèle source (.gguf)"
        if (-not $in) { return }
    }

    # --- Type de quantification ---
    Write-Host ""
    Write-Host "  Type de quantification :" -ForegroundColor White
    for ($i = 0; $i -lt $QuantTypes.Count; $i++) {
        Write-Host ("    {0}. {1}" -f ($i + 1), $QuantTypes[$i])
    }
    $c = Read-Choice "  Choix" 1 $QuantTypes.Count
    $qtype = $QuantTypes[$c - 1]

    # --- Nom de sortie ---
    $base = [System.IO.Path]::GetFileNameWithoutExtension($in)
    $defOut = Join-Path (Split-Path -Parent $in) ("{0}-{1}.gguf" -f $base, $qtype)
    $out = Read-Host "  Fichier de sortie (défaut $defOut)"
    if ($out -eq "") { $out = $defOut }

    # --- Threads ---
    $th = Read-Host "  Nombre de threads (défaut : tous)"
    if ($th -eq "") { $th = "" }

    $args = @($in, $out, $qtype)
    if ($th -ne "") { $args += $th }

    if (-not (Confirm-Run $QuantizeExe $args)) { return }

    Write-Host ""
    Write-Host "  Quantification en cours..." -ForegroundColor Green
    & $QuantizeExe @args
    Write-Host ""
    Write-Host "  Terminé : $out" -ForegroundColor Green
}

# ============================================================
#  MENU 3 : Fine-tuner un modèle
# ============================================================
function Menu-Finetune {
    Write-Host ""
    Write-Host "  ===== FINE-TUNER UN MODÈLE =====" -ForegroundColor Yellow

    if (-not (Test-Path $FinetuneExe)) {
        Write-Host "  [ERREUR] $FinetuneExe introuvable." -ForegroundColor Red
        Write-Host "  Compilez-le via beellama.cpp\build_finetune.bat" -ForegroundColor DarkGray
        return
    }

    Write-Host "  NOTE : seuls les poids F32 sont entraînables (FP16/BF16/FP8/Q4/Q8 = gelés)." -ForegroundColor Magenta
    Write-Host "         Prévoir ~24 Go de RAM (CPU). Convertir un safetensors en GGUF F32 d'abord." -ForegroundColor Magenta
    Write-Host ""

    # --- Modèle de base ---
    $models = Get-Models
    if ($models.Count -gt 0) {
        Write-Host "  Modèle de base (idéalement F32) :" -ForegroundColor White
        for ($i = 0; $i -lt $models.Count; $i++) {
            Write-Host ("    {0}. {1}" -f ($i + 1), $models[$i].Name)
        }
        Write-Host ("    {0}. Autre chemin..." -f ($models.Count + 1))
        $c = Read-Choice "  Choix" 1 ($models.Count + 1)
        if ($c -le $models.Count) {
            $model = $models[$c - 1].FullName
            if ($models[$c - 1].Name -match "Q\d|IQ\d|F16|BF16") {
                Write-Host "  [ATTENTION] Modèle quantifié détecté : le fine-tune risque d'échouer." -ForegroundColor Yellow
            }
        } else {
            $model = Read-Host "  Chemin complet du modèle (.gguf)"
            if (-not $model) { return }
        }
    } else {
        $model = Read-Host "  Chemin complet du modèle (.gguf)"
        if (-not $model) { return }
    }

    # --- Données d'entraînement ---
    Write-Host ""
    $corpus = @()
    if (Test-Path $CorpusDir) {
        $corpus = @(Get-ChildItem -Path $CorpusDir -Include *.raw,*.txt -File -Recurse | Sort-Object Name)
    }
    if ($corpus.Count -gt 0) {
        Write-Host "  Données d'entraînement :" -ForegroundColor White
        for ($i = 0; $i -lt $corpus.Count; $i++) {
            Write-Host ("    {0}. {1}" -f ($i + 1), $corpus[$i].FullName.Replace("$Root\", ""))
        }
        Write-Host ("    {0}. Autre chemin..." -f ($corpus.Count + 1))
        $c = Read-Choice "  Choix" 1 ($corpus.Count + 1)
        if ($c -le $corpus.Count) {
            $file = $corpus[$c - 1].FullName
        } else {
            $file = Read-Host "  Chemin du fichier de données (.txt/.raw)"
            if (-not $file) { return }
        }
    } else {
        $file = Read-Host "  Chemin du fichier de données (.txt/.raw)"
        if (-not $file) { return }
    }

    # --- Sortie ---
    $defOut = Join-Path $Root "finetuned-model.gguf"
    $out = Read-Host "  Fichier de sortie (défaut $defOut)"
    if ($out -eq "") { $out = $defOut }

    # --- Hyperparamètres ---
    $epochs = Read-Host "  Nombre d'époques (défaut 2)"
    if ($epochs -eq "") { $epochs = "2" }

    $opt = Read-Host "  Optimiseur adamw|sgd (défaut adamw)"
    if ($opt -eq "") { $opt = "adamw" }

    $lr = Read-Host "  Learning rate (défaut 1e-5)"
    if ($lr -eq "") { $lr = "1e-5" }

    $wd = Read-Host "  Weight decay (défaut 1e-9)"
    if ($wd -eq "") { $wd = "1e-9" }

    $ctx = Read-Host "  Taille de contexte (défaut 512)"
    if ($ctx -eq "") { $ctx = "512" }

    $batch = Read-Host "  Batch size (défaut 512)"
    if ($batch -eq "") { $batch = "512" }

    $args = @(
        "-m", $model,
        "-f", $file,
        "-o", $out,
        "-epochs", "$epochs",
        "-opt", $opt,
        "-lr", "$lr",
        "-wd", "$wd",
        "-c", "$ctx",
        "-b", "$batch",
        "-ub", "$batch",
        "-ngl", "0"
    )

    if (-not (Confirm-Run $FinetuneExe $args)) { return }

    Write-Host ""
    Write-Host "  Fine-tune en cours (peut être long)..." -ForegroundColor Green
    & $FinetuneExe @args
    Write-Host ""
    Write-Host "  Terminé : $out" -ForegroundColor Green
}

# ============================================================
#  MENU PRINCIPAL
# ============================================================
while ($true) {
    Clear-Host
    Write-Host "  ============================================" -ForegroundColor Cyan
    Write-Host "   LAMA TOUR - Lanceur" -ForegroundColor Cyan
    Write-Host "  ============================================" -ForegroundColor Cyan
    $gpu = Get-GpuInfo
    Write-Host ("  GPU : {0} ({1})" -f $gpu.Name, $gpu.Label) -ForegroundColor Green
    Write-Host ""
    Write-Host "  1. Lancer un LLM" -ForegroundColor White
    Write-Host "  2. Quantifier un GGUF" -ForegroundColor White
    Write-Host "  3. Fine-tuner un modèle" -ForegroundColor White
    Write-Host "  4. Quitter" -ForegroundColor White
    Write-Host ""

    $c = Read-Choice "  Choix" 1 4
    switch ($c) {
        1 { Menu-Launch }
        2 { Menu-Quantize }
        3 { Menu-Finetune }
        4 { break }
    }
    if ($c -eq 4) { break }
}
