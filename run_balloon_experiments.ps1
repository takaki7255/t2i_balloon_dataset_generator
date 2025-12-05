# =============================================================================
# Balloon U-Net Training & Testing Experiment Script (PowerShell版)
# =============================================================================
# train_unet_split.pyを使用して、balloon_dataset内の指定データセットで
# U-Netモデルを学習し、test_unet.pyでテストを実行します。
#
# 使用方法:
#   .\run_balloon_experiments.ps1
#
# オプション:
#   .\run_balloon_experiments.ps1 -SkipTraining   # 学習をスキップしてテストのみ
#   .\run_balloon_experiments.ps1 -SkipTest       # テストをスキップして学習のみ
# =============================================================================

param(
    [switch]$SkipTraining,
    [switch]$SkipTest
)

$ErrorActionPreference = "Stop"

# 基本設定
$DATASET_ROOT = ".\balloon_dataset"
$MODELS_DIR = ".\balloon_models"
$RESULTS_DIR = ".\balloon_results"
$WANDB_PROJECT = "balloon-seg"

# テストデータセット (評価用)
$TEST_DATASETS = @("test100_dataset")

# 学習対象データセット
$TRAIN_DATASETS = @(
    "syn200-corner",
    "syn500-corner",
    "syn750-corner",
    "syn1000-corner",
    "syn2000-corner",
    "syn5000-corner"
)

# モデル保存ディレクトリを作成
if (-not (Test-Path $MODELS_DIR)) {
    New-Item -ItemType Directory -Path $MODELS_DIR | Out-Null
}
if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Path $RESULTS_DIR | Out-Null
}

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Balloon U-Net Training & Testing" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dataset root: $DATASET_ROOT" -ForegroundColor Yellow
Write-Host "Models directory: $MODELS_DIR" -ForegroundColor Yellow
Write-Host "Results directory: $RESULTS_DIR" -ForegroundColor Yellow
Write-Host "Wandb project: $WANDB_PROJECT" -ForegroundColor Yellow
Write-Host ""
Write-Host "Training datasets ($($TRAIN_DATASETS.Count)):" -ForegroundColor Green
foreach ($ds in $TRAIN_DATASETS) {
    Write-Host "  - $ds"
}
Write-Host ""
Write-Host "Test datasets ($($TEST_DATASETS.Count)):" -ForegroundColor Magenta
foreach ($ds in $TEST_DATASETS) {
    Write-Host "  - $ds"
}
Write-Host ""

# 学習パラメータ
$EPOCHS = 100
$BATCH_SIZE = 8
$LEARNING_RATE = 1e-4
$PATIENCE = 30

# ============================================
# Training Loop
# ============================================
$successCount = 0
$failCount = 0
$skippedCount = 0
$trainedModels = @()

if (-not $SkipTraining) {
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  Phase 1: Training" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan

    foreach ($dataset in $TRAIN_DATASETS) {
        Write-Host ""
        Write-Host "----------------------------------------------" -ForegroundColor White
        Write-Host " Training: $dataset" -ForegroundColor Magenta
        Write-Host "----------------------------------------------" -ForegroundColor White
        
        $datasetPath = Join-Path $DATASET_ROOT $dataset
        
        # データセットが存在するか確認
        if (-not (Test-Path $datasetPath)) {
            Write-Host "  Dataset not found: $datasetPath - Skipping" -ForegroundColor Yellow
            $skippedCount++
            continue
        }
        
        # train/valディレクトリが存在するか確認
        $trainDir = Join-Path $datasetPath "train"
        $valDir = Join-Path $datasetPath "val"
        
        if (-not (Test-Path $trainDir) -or -not (Test-Path $valDir)) {
            Write-Host "  train/val directories not found in $datasetPath - Skipping" -ForegroundColor Yellow
            $skippedCount++
            continue
        }
        
        # 学習を実行
        Write-Host "Running: python train_unet_split.py --root $datasetPath --dataset $dataset" -ForegroundColor Gray
        
        try {
            python train_unet_split.py `
                --root "$datasetPath" `
                --dataset "$dataset" `
                --models-dir "$MODELS_DIR" `
                --epochs $EPOCHS `
                --batch $BATCH_SIZE `
                --lr $LEARNING_RATE `
                --patience $PATIENCE `
                --wandb-proj "$WANDB_PROJECT"
            
            Write-Host " Completed: $dataset" -ForegroundColor Green
            $successCount++
            $trainedModels += "$dataset-unet-01"
        }
        catch {
            Write-Host " FAILED: $dataset - $($_.Exception.Message)" -ForegroundColor Red
            $failCount++
        }
    }

    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  Training Summary" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Success: $successCount" -ForegroundColor Green
    Write-Host "Failed: $failCount" -ForegroundColor Red
    Write-Host "Skipped: $skippedCount" -ForegroundColor Yellow
} else {
    Write-Host "Skipping training phase..." -ForegroundColor Yellow
    # 既存のモデルを検索
    if (Test-Path "$MODELS_DIR\*.pt") {
        $trainedModels = Get-ChildItem "$MODELS_DIR\*.pt" | ForEach-Object { $_.BaseName }
    }
}

Write-Host ""
Write-Host "Trained models are saved in: $MODELS_DIR" -ForegroundColor White
Write-Host ""

# ============================================
# Testing Loop
# ============================================
if (-not $SkipTest) {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  Phase 2: Testing" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
    
    $testSuccessCount = 0
    $testFailCount = 0
    
    # 学習済みモデルを取得
    $modelFiles = Get-ChildItem "$MODELS_DIR\*.pt" -ErrorAction SilentlyContinue
    
    if ($modelFiles.Count -eq 0) {
        Write-Host "No trained models found in $MODELS_DIR" -ForegroundColor Yellow
    } else {
        Write-Host "Found $($modelFiles.Count) models to test" -ForegroundColor Green
        
        foreach ($modelFile in $modelFiles) {
            $modelTag = $modelFile.BaseName
            
            foreach ($testDataset in $TEST_DATASETS) {
                Write-Host ""
                Write-Host "----------------------------------------------" -ForegroundColor White
                Write-Host " Testing: $modelTag on $testDataset" -ForegroundColor Magenta
                Write-Host "----------------------------------------------" -ForegroundColor White
                
                $testDataPath = Join-Path $DATASET_ROOT $testDataset
                
                # テストデータセットが存在するか確認
                if (-not (Test-Path $testDataPath)) {
                    Write-Host "  Test dataset not found: $testDataPath - Skipping" -ForegroundColor Yellow
                    continue
                }
                
                # images/masksディレクトリが存在するか確認
                $imagesDir = Join-Path $testDataPath "images"
                $masksDir = Join-Path $testDataPath "masks"
                
                if (-not (Test-Path $imagesDir) -or -not (Test-Path $masksDir)) {
                    Write-Host "  images/masks directories not found in $testDataPath - Skipping" -ForegroundColor Yellow
                    continue
                }
                
                # テストを実行
                $runName = "$modelTag-test-$testDataset"
                Write-Host "Running: python test_unet.py --model-tag $modelTag --data-root $testDataPath" -ForegroundColor Gray
                
                try {
                    python test_unet.py `
                        --model-tag "$modelTag" `
                        --models-dir "$MODELS_DIR" `
                        --data-root "$testDataPath" `
                        --result-dir "$RESULTS_DIR" `
                        --batch $BATCH_SIZE `
                        --wandb-proj "$WANDB_PROJECT" `
                        --run-name "$runName"
                    
                    Write-Host " Completed: $runName" -ForegroundColor Green
                    $testSuccessCount++
                }
                catch {
                    Write-Host " FAILED: $runName - $($_.Exception.Message)" -ForegroundColor Red
                    $testFailCount++
                }
            }
        }
        
        Write-Host ""
        Write-Host "==============================================" -ForegroundColor Cyan
        Write-Host "  Testing Summary" -ForegroundColor Cyan
        Write-Host "==============================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Test Success: $testSuccessCount" -ForegroundColor Green
        Write-Host "Test Failed: $testFailCount" -ForegroundColor Red
    }
} else {
    Write-Host "Skipping testing phase..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Experiment Complete" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# モデル一覧を表示
Write-Host "Trained models:" -ForegroundColor Cyan
if (Test-Path "$MODELS_DIR\*.pt") {
    Get-ChildItem "$MODELS_DIR\*.pt" | ForEach-Object {
        Write-Host "  $($_.Name)" -ForegroundColor White
    }
} else {
    Write-Host "  No models found" -ForegroundColor Yellow
}

# 結果一覧を表示
Write-Host ""
Write-Host "Test results:" -ForegroundColor Cyan
if (Test-Path $RESULTS_DIR) {
    Get-ChildItem $RESULTS_DIR -Directory | ForEach-Object {
        Write-Host "  $($_.Name)" -ForegroundColor White
    }
} else {
    Write-Host "  No results found" -ForegroundColor Yellow
}
