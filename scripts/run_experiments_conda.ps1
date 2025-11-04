#!/usr/bin/env pwsh
# Run U-Net experiments on multiple datasets with Conda environment
# Usage: .\run_experiments_conda.ps1 -CondaEnv myenv [-TrainOnly] [-TestOnly] [-NoWandb]

param(
    [Parameter(Mandatory=$true)]
    [string]$CondaEnv,        # Conda environment name (e.g., myenv)
    
    [switch]$TrainOnly,       # Train only
    [switch]$TestOnly,        # Test only
    [switch]$NoWandb,         # Disable Wandb
    [string]$Datasets = "all" # Datasets to run
)

# Error on stop
$ErrorActionPreference = "Stop"

# Set UTF-8 encoding for output
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ============================================================================
# Check and activate Conda environment
# ============================================================================

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "Conda Environment Setup" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

# Initialize Conda (equivalent to Anaconda Prompt)
$condaPath = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $condaPath) {
    # If Conda not found, try default paths
    $possiblePaths = @(
        "$env:USERPROFILE\Anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "C:\ProgramData\Anaconda3\Scripts\conda.exe",
        "C:\ProgramData\Miniconda3\Scripts\conda.exe"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $condaPath = $path
            break
        }
    }
    
    if (-not $condaPath) {
        Write-Host "[ERROR] Conda not found. Please install Anaconda/Miniconda." -ForegroundColor Red
        Write-Host "Or run from Anaconda Prompt." -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "Conda: $condaPath" -ForegroundColor Green

# Check if Conda environment exists
$condaEnvList = & conda env list | Select-String -Pattern "^$CondaEnv\s"
if (-not $condaEnvList) {
    Write-Host "[ERROR] Conda environment '$CondaEnv' not found." -ForegroundColor Red
    Write-Host "Available environments:" -ForegroundColor Yellow
    & conda env list
    exit 1
}

Write-Host "[OK] Using Conda environment '$CondaEnv'" -ForegroundColor Green
Write-Host ""

# ============================================================================
# Configuration: Datasets for training and testing
# ============================================================================

$DATASET_CONFIGS = @(
    @{
        name = "real200_dataset"
        root = "./onomatopoeia_dataset/real200_dataset"
        epochs = 100
        batch = 8
        patience = 30
    },
    @{
        name = "syn200"
        root = "./onomatopoeia_dataset/syn200"
        epochs = 100
        batch = 8
        patience = 30
    },
    @{
        name = "syn200-panel"
        root = "./onomatopoeia_dataset/syn200-panel"
        epochs = 100
        batch = 8
        patience = 30
    },
    @{
        name = "syn500"
        root = "./onomatopoeia_dataset/syn500"
        epochs = 100
        batch = 8
        patience = 30
    },
    @{
        name = "syn500-panel"
        root = "./onomatopoeia_dataset/syn500-panel"
        epochs = 100
        batch = 8
        patience = 30
    }
)

# Test dataset
$TEST_DATA_ROOT = "./onomatopoeia_dataset/test100_dataset"

# Model save directory
$MODELS_DIR = "./onomatopoeia_models"

# Results save directory
$RESULTS_DIR = "./onomatopoeia_experiment_results"

# Wandb project name
$WANDB_PROJECT = "onomatopoeia-seg-experiments"

# ============================================================================
# Dataset filtering
# ============================================================================

if ($Datasets -ne "all") {
    $selectedNames = $Datasets -split ","
    $DATASET_CONFIGS = $DATASET_CONFIGS | Where-Object { $selectedNames -contains $_.name }
    
    if ($DATASET_CONFIGS.Count -eq 0) {
        Write-Host "[ERROR] Specified datasets not found: $Datasets" -ForegroundColor Red
        Write-Host "Available datasets: syn200-corner, syn500-corner, syn750-corner, syn1000-corner" -ForegroundColor Yellow
        exit 1
    }
}

# ============================================================================
# Create directories
# ============================================================================

if (-not (Test-Path $MODELS_DIR)) {
    New-Item -ItemType Directory -Path $MODELS_DIR | Out-Null
}

if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Path $RESULTS_DIR | Out-Null
}

# ============================================================================
# Experiment log file
# ============================================================================

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_FILE = "$RESULTS_DIR/experiment_log_$timestamp.txt"

function Write-Log {
    param($Message)
    $logMessage = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host $logMessage
    Add-Content -Path $LOG_FILE -Value $logMessage
}

# ============================================================================
# Function to run Python with Conda environment
# ============================================================================

function Invoke-CondaPython {
    param(
        [string[]]$Arguments
    )
    
    # Run Python in Conda environment using conda run
    $condaRunArgs = @("run", "-n", $CondaEnv, "--no-capture-output", "python") + $Arguments
    
    Write-Log "  Command: conda run -n $CondaEnv --no-capture-output python $($Arguments -join ' ')"
    
    & conda @condaRunArgs 2>&1 | ForEach-Object {
        try {
            Write-Host $_
        } catch {
            # Ignore encoding errors
        }
    }
    return $LASTEXITCODE
}

# ============================================================================
# Start experiments
# ============================================================================

Write-Log "="*80
Write-Log "U-Net Automated Experiments (Conda Environment)"
Write-Log "="*80
Write-Log "Conda Environment: $CondaEnv"
$modeText = if($TrainOnly){"Train Only"}elseif($TestOnly){"Test Only"}else{"Train + Test"}
Write-Log "Mode: $modeText"
$wandbText = if($NoWandb){"Disabled"}else{"Enabled"}
Write-Log "Wandb: $wandbText"
Write-Log "Datasets: $($DATASET_CONFIGS.Count)"
Write-Log "Log File: $LOG_FILE"
Write-Log ""

# ============================================================================
# Run experiments on each dataset
# ============================================================================

$total_datasets = $DATASET_CONFIGS.Count
$completed = 0
$failed = 0

foreach ($config in $DATASET_CONFIGS) {
    $dataset_name = $config.name
    $dataset_root = $config.root
    $epochs = $config.epochs
    $batch = $config.batch
    $patience = $config.patience
    
    Write-Log ""
    Write-Log "="*80
    Write-Log "Dataset: $dataset_name"
    Write-Log "="*80
    Write-Log "  Root: $dataset_root"
    Write-Log "  Epochs: $epochs"
    Write-Log "  Batch Size: $batch"
    Write-Log "  Patience: $patience"
    Write-Log ""
    
    # ========================================================================
    # 1. Training
    # ========================================================================
    
    if (-not $TestOnly) {
        Write-Log "Training: $dataset_name"
        Write-Log "-"*80
        
        $train_args = @(
            "train_unet_split.py",
            "--root", $dataset_root,
            "--dataset", $dataset_name,
            "--models-dir", $MODELS_DIR,
            "--epochs", $epochs,
            "--batch", $batch,
            "--patience", $patience,
            "--wandb-proj", $WANDB_PROJECT,
            "--run-name", "${dataset_name}-train"
        )
        
        try {
            $exitCode = Invoke-CondaPython -Arguments $train_args
            
            if ($exitCode -eq 0) {
                Write-Log "[OK] Training completed: $dataset_name"
            } else {
                Write-Log "[FAIL] Training failed: $dataset_name (exit code: $exitCode)"
                $failed++
                continue
            }
        }
        catch {
            Write-Log "[ERROR] Training error: $dataset_name"
            Write-Log "   Error: $_"
            $failed++
            continue
        }
    }
    
    # ========================================================================
    # 2. Testing
    # ========================================================================
    
    if (-not $TrainOnly) {
        Write-Log ""
        Write-Log "Testing: $dataset_name"
        Write-Log "-"*80
        
        # Get model tag (latest version)
        $model_pattern = "$MODELS_DIR/${dataset_name}-unet-*.pt"
        $model_files = Get-ChildItem -Path $model_pattern -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        
        if ($model_files.Count -eq 0) {
            Write-Log "[WARN] Model not found: $model_pattern"
            Write-Log "   Skipping test"
            continue
        }
        
        $latest_model = $model_files[0]
        $model_tag = $latest_model.BaseName
        
        Write-Log "   Using model: $model_tag"
        
        $test_args = @(
            "test_unet.py",
            "--model-tag", $model_tag,
            "--models-dir", $MODELS_DIR,
            "--data-root", $TEST_DATA_ROOT,
            "--result-dir", $RESULTS_DIR,
            "--batch", $batch,
            "--wandb-proj", $WANDB_PROJECT,
            "--run-name", "${dataset_name}-test"
        )
        
        if ($NoWandb) {
            $test_args += "--no-wandb"
        }
        
        try {
            $exitCode = Invoke-CondaPython -Arguments $test_args
            
            if ($exitCode -eq 0) {
                Write-Log "[OK] Test completed: $dataset_name"
                $completed++
            } else {
                Write-Log "[FAIL] Test failed: $dataset_name (exit code: $exitCode)"
                $failed++
            }
        }
        catch {
            Write-Log "[ERROR] Test error: $dataset_name"
            Write-Log "   Error: $_"
            $failed++
        }
    }
    else {
        $completed++
    }
}

# ============================================================================
# Final results summary
# ============================================================================

Write-Log ""
Write-Log "="*80
Write-Log "Experiments Completed"
Write-Log "="*80
Write-Log "Total datasets: $total_datasets"
Write-Log "Success: $completed"
Write-Log "Failed: $failed"
Write-Log ""
Write-Log "Results: $RESULTS_DIR"
Write-Log "Log file: $LOG_FILE"
Write-Log "="*80

if ($failed -gt 0) {
    Write-Host "[WARN] Some experiments failed. Check log for details." -ForegroundColor Yellow
    exit 1
}
else {
    Write-Host "[OK] All experiments completed successfully!" -ForegroundColor Green
    exit 0
}
