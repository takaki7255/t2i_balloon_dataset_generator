#!/usr/bin/env pwsh
# 単一データセットで素早く学習・テストを実行
# 使用方法: .\quick_train_test.ps1 -Dataset syn500-corner [-SkipTrain] [-SkipTest]

param(
    [Parameter(Mandatory=$true)]
    [string]$Dataset,
    
    [switch]$SkipTrain,
    [switch]$SkipTest,
    [int]$Epochs = 100,
    [int]$Batch = 8,
    [int]$Patience = 15
)

$ErrorActionPreference = "Stop"

# データセット設定マッピング
$DATASET_CONFIGS = @{
    "syn200-corner" = "./balloon_dataset/syn200_dataset"
    "syn500-corner" = "./balloon_dataset/syn500_dataset"
    "syn750-corner" = "./balloon_dataset/syn750_dataset"
    "syn1000-corner" = "./balloon_dataset/syn1000_dataset"
}

if (-not $DATASET_CONFIGS.ContainsKey($Dataset)) {
    Write-Host "❌ 不明なデータセット: $Dataset" -ForegroundColor Red
    Write-Host "利用可能なデータセット:" -ForegroundColor Yellow
    $DATASET_CONFIGS.Keys | ForEach-Object { Write-Host "  - $_" }
    exit 1
}

$DATASET_ROOT = $DATASET_CONFIGS[$Dataset]
$MODELS_DIR = "./balloon_models"
$TEST_DATA_ROOT = "./test_dataset"

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "🚀 クイック実験: $Dataset" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "データセット: $DATASET_ROOT"
Write-Host "エポック: $Epochs | バッチ: $Batch | Patience: $Patience"
Write-Host ""

# 学習
if (-not $SkipTrain) {
    Write-Host "🏋️  学習開始..." -ForegroundColor Green
    python train_unet_split.py `
        --root $DATASET_ROOT `
        --dataset $Dataset `
        --models-dir $MODELS_DIR `
        --epochs $Epochs `
        --batch $Batch `
        --patience $Patience
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ 学習失敗" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ 学習完了`n" -ForegroundColor Green
}

# テスト
if (-not $SkipTest) {
    Write-Host "🧪 テスト開始..." -ForegroundColor Green
    
    # 最新モデルを取得
    $models = Get-ChildItem "$MODELS_DIR/${Dataset}-unet-*.pt" | Sort-Object Name -Descending
    if ($models.Count -eq 0) {
        Write-Host "❌ モデルが見つかりません" -ForegroundColor Red
        exit 1
    }
    
    $model_tag = $models[0].BaseName
    Write-Host "使用モデル: $model_tag"
    
    python test_unet.py `
        --model-tag $model_tag `
        --models-dir $MODELS_DIR `
        --data-root $TEST_DATA_ROOT `
        --batch $Batch
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ テスト失敗" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ テスト完了`n" -ForegroundColor Green
}

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "🎉 完了！" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
