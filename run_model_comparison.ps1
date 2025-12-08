# =============================================================================
# Multi-Model Balloon Segmentation Experiment Script (PowerShell版)
# =============================================================================
# real1000_datasetで ResNet-UNet, DeepLabv3+, SegFormer を学習し、
# test300_datasetでテストを実行してモデル比較を行います。
#
# 使用方法:
#   .\run_model_comparison.ps1
#
# オプション:
#   .\run_model_comparison.ps1 -SkipTraining   # 学習をスキップしてテストのみ
#   .\run_model_comparison.ps1 -SkipTest       # テストをスキップして学習のみ
# =============================================================================

param(
    [switch]$SkipTraining,
    [switch]$SkipTest
)

$ErrorActionPreference = "Continue"

# 基本設定
$DATASET_ROOT = ".\balloon_dataset"
$TRAIN_DATASET = "real200_dataset"
$TEST_DATASET = "test300_dataset"
$MODELS_DIR = ".\balloon_models"
$RESULTS_DIR = ".\balloon_results"
$WANDB_PROJECT = "balloon-model-comparison"

# 学習パラメータ
$EPOCHS = 100
$BATCH_SIZE = 8
$LEARNING_RATE = 1e-4
$PATIENCE = 20

# ディレクトリ作成
if (-not (Test-Path $MODELS_DIR)) {
    New-Item -ItemType Directory -Path $MODELS_DIR | Out-Null
}
if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Path $RESULTS_DIR | Out-Null
}

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Multi-Model Balloon Segmentation Experiment" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Training dataset: $TRAIN_DATASET" -ForegroundColor Green
Write-Host "Test dataset: $TEST_DATASET" -ForegroundColor Magenta
Write-Host "Models: ResNet34-UNet (scratch), ResNet34-UNet (pretrained), DeepLabv3+, SegFormer" -ForegroundColor Yellow
Write-Host ""

$trainDataPath = Join-Path $DATASET_ROOT $TRAIN_DATASET
$testDataPath = Join-Path $DATASET_ROOT $TEST_DATASET

# データセット確認
if (-not (Test-Path $trainDataPath)) {
    Write-Host "ERROR: Training dataset not found: $trainDataPath" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $testDataPath)) {
    Write-Host "ERROR: Test dataset not found: $testDataPath" -ForegroundColor Red
    exit 1
}

# ============================================
# Training Phase
# ============================================
if (-not $SkipTraining) {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  Phase 1: Training" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan

    # -----------------------------------------
    # 1. ResNet34-UNet (Scratch - No Pretrained)
    # -----------------------------------------
    Write-Host ""
    Write-Host "----------------------------------------------" -ForegroundColor White
    Write-Host " [1/4] Training: ResNet34-UNet (Scratch)" -ForegroundColor Magenta
    Write-Host "----------------------------------------------" -ForegroundColor White

    $resnetScratchModelTag = "$TRAIN_DATASET-resnet34-scratch-unet-01"
    
    try {
        python train_resnet_unet.py `
            --root "$trainDataPath" `
            --dataset "$TRAIN_DATASET" `
            --backbone resnet34 `
            --no-pretrained `
            --models-dir "$MODELS_DIR" `
            --epochs $EPOCHS `
            --batch $BATCH_SIZE `
            --lr $LEARNING_RATE `
            --patience $PATIENCE `
            --wandb-proj "$WANDB_PROJECT" `
            --run-name "$resnetScratchModelTag"
        
        Write-Host " Completed: ResNet34-UNet (Scratch)" -ForegroundColor Green
    }
    catch {
        Write-Host " FAILED: ResNet34-UNet (Scratch) - $($_.Exception.Message)" -ForegroundColor Red
    }

    # -----------------------------------------
    # 2. ResNet34-UNet (Pretrained)
    # -----------------------------------------
    Write-Host ""
    Write-Host "----------------------------------------------" -ForegroundColor White
    Write-Host " [2/4] Training: ResNet34-UNet (Pretrained)" -ForegroundColor Magenta
    Write-Host "----------------------------------------------" -ForegroundColor White

    $resnetModelTag = "$TRAIN_DATASET-resnet34-pretrained-unet-01"
    
    try {
        python train_resnet_unet.py `
            --root "$trainDataPath" `
            --dataset "$TRAIN_DATASET" `
            --backbone resnet34 `
            --pretrained `
            --models-dir "$MODELS_DIR" `
            --epochs $EPOCHS `
            --batch $BATCH_SIZE `
            --lr $LEARNING_RATE `
            --patience $PATIENCE `
            --wandb-proj "$WANDB_PROJECT" `
            --run-name "$resnetModelTag"
        
        Write-Host " Completed: ResNet34-UNet" -ForegroundColor Green
    }
    catch {
        Write-Host " FAILED: ResNet34-UNet - $($_.Exception.Message)" -ForegroundColor Red
    }

    # -----------------------------------------
    # 3. DeepLabv3+ (ResNet50, Pretrained, stride=16)
    # -----------------------------------------
    Write-Host ""
    Write-Host "----------------------------------------------" -ForegroundColor White
    Write-Host " [3/4] Training: DeepLabv3+ (ResNet50)" -ForegroundColor Magenta
    Write-Host "----------------------------------------------" -ForegroundColor White

    $deeplabModelTag = "$TRAIN_DATASET-deeplabv3plus-resnet50-pretrained-s16-01"
    
    try {
        python train_deeplabv3plus.py `
            --root "$trainDataPath" `
            --dataset "$TRAIN_DATASET" `
            --backbone resnet50 `
            --pretrained `
            --output-stride 16 `
            --models-dir "$MODELS_DIR" `
            --epochs $EPOCHS `
            --batch $BATCH_SIZE `
            --lr $LEARNING_RATE `
            --patience $PATIENCE `
            --wandb-proj "$WANDB_PROJECT" `
            --run-name "$deeplabModelTag"
        
        Write-Host " Completed: DeepLabv3+" -ForegroundColor Green
    }
    catch {
        Write-Host " FAILED: DeepLabv3+ - $($_.Exception.Message)" -ForegroundColor Red
    }

    # -----------------------------------------
    # 4. SegFormer (Boundary-Aware, Gray input)
    # -----------------------------------------
    Write-Host ""
    Write-Host "----------------------------------------------" -ForegroundColor White
    Write-Host " [4/4] Training: SegFormer (Boundary-Aware)" -ForegroundColor Magenta
    Write-Host "----------------------------------------------" -ForegroundColor White

    $segformerModelTag = "$TRAIN_DATASET-segformer-gray-01"
    $segformerSaveDir = Join-Path $MODELS_DIR $segformerModelTag
    
    try {
        python train_boundary_segformer.py `
            --train_images "$trainDataPath\train\images" `
            --train_masks "$trainDataPath\train\masks" `
            --val_images "$trainDataPath\val\images" `
            --val_masks "$trainDataPath\val\masks" `
            --input_type gray `
            --epochs $EPOCHS `
            --batch_size $BATCH_SIZE `
            --lr $LEARNING_RATE `
            --output_dir "$segformerSaveDir"
        
        Write-Host " Completed: SegFormer" -ForegroundColor Green
    }
    catch {
        Write-Host " FAILED: SegFormer - $($_.Exception.Message)" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  Training Complete" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
}
else {
    Write-Host "Skipping training phase..." -ForegroundColor Yellow
}

# ============================================
# Testing Phase
# ============================================
if (-not $SkipTest) {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  Phase 2: Testing on $TEST_DATASET" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan

    # -----------------------------------------
    # 1. Test ResNet34-UNet (Scratch)
    # -----------------------------------------
    Write-Host ""
    Write-Host "----------------------------------------------" -ForegroundColor White
    Write-Host " [1/4] Testing: ResNet34-UNet (Scratch)" -ForegroundColor Magenta
    Write-Host "----------------------------------------------" -ForegroundColor White

    $resnetScratchModelTag = "$TRAIN_DATASET-resnet34-scratch-unet-01"
    
    try {
        python test_resnet_unet.py `
            --model-tag "$resnetScratchModelTag" `
            --backbone resnet34 `
            --models-dir "$MODELS_DIR" `
            --data-root "$testDataPath" `
            --result-dir "$RESULTS_DIR" `
            --batch $BATCH_SIZE `
            --wandb-proj "$WANDB_PROJECT" `
            --run-name "$resnetScratchModelTag-test-$TEST_DATASET"
        
        Write-Host " Completed: ResNet34-UNet (Scratch) test" -ForegroundColor Green
    }
    catch {
        Write-Host " FAILED: ResNet34-UNet (Scratch) test - $($_.Exception.Message)" -ForegroundColor Red
    }

    # -----------------------------------------
    # 2. Test ResNet34-UNet (Pretrained)
    # -----------------------------------------
    Write-Host ""
    Write-Host "----------------------------------------------" -ForegroundColor White
    Write-Host " [2/4] Testing: ResNet34-UNet (Pretrained)" -ForegroundColor Magenta
    Write-Host "----------------------------------------------" -ForegroundColor White

    $resnetModelTag = "$TRAIN_DATASET-resnet34-pretrained-unet-01"
    
    try {
        python test_resnet_unet.py `
            --model-tag "$resnetModelTag" `
            --backbone resnet34 `
            --models-dir "$MODELS_DIR" `
            --data-root "$testDataPath" `
            --result-dir "$RESULTS_DIR" `
            --batch $BATCH_SIZE `
            --wandb-proj "$WANDB_PROJECT" `
            --run-name "$resnetModelTag-test-$TEST_DATASET"
        
        Write-Host " Completed: ResNet34-UNet test" -ForegroundColor Green
    }
    catch {
        Write-Host " FAILED: ResNet34-UNet test - $($_.Exception.Message)" -ForegroundColor Red
    }

    # -----------------------------------------
    # 3. Test DeepLabv3+
    # -----------------------------------------
    Write-Host ""
    Write-Host "----------------------------------------------" -ForegroundColor White
    Write-Host " [3/4] Testing: DeepLabv3+" -ForegroundColor Magenta
    Write-Host "----------------------------------------------" -ForegroundColor White

    $deeplabModelTag = "$TRAIN_DATASET-deeplabv3plus-resnet50-pretrained-s16-01"
    
    try {
        python test_deeplabv3plus.py `
            --model-tag "$deeplabModelTag" `
            --backbone resnet50 `
            --output-stride 16 `
            --models-dir "$MODELS_DIR" `
            --data-root "$testDataPath" `
            --result-dir "$RESULTS_DIR" `
            --batch $BATCH_SIZE `
            --wandb-proj "$WANDB_PROJECT" `
            --run-name "$deeplabModelTag-test-$TEST_DATASET"
        
        Write-Host " Completed: DeepLabv3+ test" -ForegroundColor Green
    }
    catch {
        Write-Host " FAILED: DeepLabv3+ test - $($_.Exception.Message)" -ForegroundColor Red
    }

    # -----------------------------------------
    # 4. Test SegFormer
    # -----------------------------------------
    Write-Host ""
    Write-Host "----------------------------------------------" -ForegroundColor White
    Write-Host " [4/4] Testing: SegFormer" -ForegroundColor Magenta
    Write-Host "----------------------------------------------" -ForegroundColor White

    $segformerModelTag = "$TRAIN_DATASET-segformer-gray-01"
    $segformerSaveDir = Join-Path $MODELS_DIR $segformerModelTag
    $segformerCheckpoint = Join-Path $segformerSaveDir "best_segformer_gray.pth"
    $segformerResultDir = Join-Path $RESULTS_DIR $segformerModelTag
    
    if (Test-Path $segformerCheckpoint) {
        try {
            python test_boundary_segformer.py `
                --checkpoint "$segformerCheckpoint" `
                --test_images "$testDataPath\images" `
                --test_masks "$testDataPath\masks" `
                --input_type gray `
                --output_dir "$segformerResultDir"
            
            Write-Host " Completed: SegFormer test" -ForegroundColor Green
        }
        catch {
            Write-Host " FAILED: SegFormer test - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    else {
        Write-Host " SegFormer checkpoint not found: $segformerCheckpoint" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  Testing Complete" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
}
else {
    Write-Host "Skipping testing phase..." -ForegroundColor Yellow
}

# ============================================
# Summary
# ============================================
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Experiment Summary" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Models trained on: $TRAIN_DATASET" -ForegroundColor White
Write-Host "Models tested on: $TEST_DATASET" -ForegroundColor White
Write-Host ""
Write-Host "Models:" -ForegroundColor Cyan
Write-Host "  1. ResNet34-UNet (Scratch)" -ForegroundColor White
Write-Host "  2. ResNet34-UNet (Pretrained)" -ForegroundColor White
Write-Host "  3. DeepLabv3+ (ResNet50, Pretrained, stride=16)" -ForegroundColor White
Write-Host "  4. SegFormer (Boundary-Aware, Gray)" -ForegroundColor White
Write-Host ""
Write-Host "Results saved in: $RESULTS_DIR" -ForegroundColor Yellow

# 結果ファイル一覧
if (Test-Path $RESULTS_DIR) {
    Write-Host ""
    Write-Host "Result directories:" -ForegroundColor Cyan
    Get-ChildItem $RESULTS_DIR -Directory | Where-Object { $_.Name -like "*$TRAIN_DATASET*" } | ForEach-Object {
        Write-Host "  $($_.Name)" -ForegroundColor White
    }
}
