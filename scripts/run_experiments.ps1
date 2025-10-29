#!/usr/bin/env pwsh
# 複数データセットでU-Net学習とテストを自動実行するスクリプト
# 使用方法: .\run_experiments.ps1 [-TrainOnly] [-TestOnly] [-NoWandb]

param(
    [switch]$TrainOnly,   # 学習のみ実行
    [switch]$TestOnly,    # テストのみ実行
    [switch]$NoWandb,     # Wandb無効化
    [string]$Datasets = "all"  # 実行するデータセット (all, syn200, syn500, syn750, syn1000 or カンマ区切り)
)

# エラー時に停止
$ErrorActionPreference = "Stop"

# ============================================================================
# 設定: 学習・テスト対象のデータセット
# ============================================================================

# データセット定義 (名前, ルートパス, エポック数, バッチサイズ)
$DATASET_CONFIGS = @(
    @{
        name = "syn200_allsize_dataset"
        root = "./balloon_dataset/syn200_allsize_dataset"
        epochs = 100
        batch = 8
        patience = 30
    },
    @{
        name = "syn200_dataset"
        root = "./balloon_dataset/syn200_dataset"
        epochs = 100
        batch = 8
        patience = 30
    },
    @{
        name = "syn500_dataset"
        root = "./balloon_dataset/syn500_dataset"
        epochs = 100
        batch = 8
        patience = 30
    },
    @{
        name = "syn750_dataset"
        root = "./balloon_dataset/syn750_dataset"
        epochs = 100
        batch = 8
        patience = 30
    },
    @{
        name = "syn1000_dataset"
        root = "./balloon_dataset/syn1000_dataset"
        epochs = 100
        batch = 8
        patience = 30
    }
    @{
        name = "syn200-corner"
        root = "./balloon_dataset/syn200-balloon-corner"
        epochs = 100
        batch = 8
        patience = 30
    },
    @{
        name = "syn500-corner"
        root = "./balloon_dataset/syn500-balloon-corner"
        epochs = 100
        batch = 8
        patience = 30
    },
    @{
        name = "syn750-corner"
        root = "./balloon_dataset/syn750-balloon-corner"
        epochs = 100
        batch = 8
        patience = 30
    },
    @{
        name = "syn1000-corner"
        root = "./balloon_dataset/syn1000-balloon-corner"
        epochs = 100
        batch = 8
        patience = 30
    }
)

# テスト用データセット
$TEST_DATA_ROOT = "./test_dataset"

# モデル保存ディレクトリ
$MODELS_DIR = "./balloon_models"

# 結果保存ディレクトリ
$RESULTS_DIR = "./experiment_results"

# Wandb プロジェクト名
$WANDB_PROJECT = "balloon-seg-experiments"

# ============================================================================
# データセットフィルタリング
# ============================================================================

if ($Datasets -ne "all") {
    $selectedNames = $Datasets -split ","
    $DATASET_CONFIGS = $DATASET_CONFIGS | Where-Object { $selectedNames -contains $_.name }
    
    if ($DATASET_CONFIGS.Count -eq 0) {
        Write-Host "❌ 指定されたデータセットが見つかりません: $Datasets" -ForegroundColor Red
        Write-Host "利用可能なデータセット: syn200-corner, syn500-corner, syn750-corner, syn1000-corner" -ForegroundColor Yellow
        exit 1
    }
}

# ============================================================================
# ディレクトリ作成
# ============================================================================

if (-not (Test-Path $MODELS_DIR)) {
    New-Item -ItemType Directory -Path $MODELS_DIR | Out-Null
}

if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Path $RESULTS_DIR | Out-Null
}

# ============================================================================
# 実験ログファイル
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
# 実験開始
# ============================================================================

Write-Log "="*80
Write-Log "🚀 U-Net 自動実験スクリプト開始"
Write-Log "="*80
Write-Log "実行モード: $(if($TrainOnly){'学習のみ'}elseif($TestOnly){'テストのみ'}else{'学習+テスト'})"
Write-Log "Wandb: $(if($NoWandb){'無効'}else{'有効'})"
Write-Log "対象データセット数: $($DATASET_CONFIGS.Count)"
Write-Log "ログファイル: $LOG_FILE"
Write-Log ""

# ============================================================================
# 各データセットで実験実行
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
    Write-Log "📊 データセット: $dataset_name"
    Write-Log "="*80
    Write-Log "  ルート: $dataset_root"
    Write-Log "  エポック: $epochs"
    Write-Log "  バッチサイズ: $batch"
    Write-Log "  Patience: $patience"
    Write-Log ""
    
    # ========================================================================
    # 1. 学習
    # ========================================================================
    
    if (-not $TestOnly) {
        Write-Log "🏋️  学習開始: $dataset_name"
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
            python @train_args
            
            if ($LASTEXITCODE -eq 0) {
                Write-Log "✅ 学習完了: $dataset_name"
            } else {
                Write-Log "❌ 学習失敗: $dataset_name (exit code: $LASTEXITCODE)"
                $failed++
                continue
            }
        }
        catch {
            Write-Log "❌ 学習エラー: $dataset_name"
            Write-Log "   エラー内容: $_"
            $failed++
            continue
        }
    }
    
    # ========================================================================
    # 2. テスト
    # ========================================================================
    
    if (-not $TrainOnly) {
        Write-Log ""
        Write-Log "🧪 テスト開始: $dataset_name"
        Write-Log "-"*80
        
        # モデルタグを取得（最新バージョン）
        $model_pattern = "$MODELS_DIR/${dataset_name}-unet-*.pt"
        $model_files = Get-ChildItem -Path $model_pattern -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        
        if ($model_files.Count -eq 0) {
            Write-Log "⚠️  モデルが見つかりません: $model_pattern"
            Write-Log "   テストをスキップします"
            continue
        }
        
        $latest_model = $model_files[0]
        $model_tag = $latest_model.BaseName
        
        Write-Log "   使用モデル: $model_tag"
        
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
            python @test_args
            
            if ($LASTEXITCODE -eq 0) {
                Write-Log "✅ テスト完了: $dataset_name"
                $completed++
            } else {
                Write-Log "❌ テスト失敗: $dataset_name (exit code: $LASTEXITCODE)"
                $failed++
            }
        }
        catch {
            Write-Log "❌ テストエラー: $dataset_name"
            Write-Log "   エラー内容: $_"
            $failed++
        }
    }
    else {
        $completed++
    }
}

# ============================================================================
# 最終結果サマリー
# ============================================================================

Write-Log ""
Write-Log "="*80
Write-Log "🎉 実験完了"
Write-Log "="*80
Write-Log "総データセット数: $total_datasets"
Write-Log "成功: $completed"
Write-Log "失敗: $failed"
Write-Log ""
Write-Log "結果保存先: $RESULTS_DIR"
Write-Log "ログファイル: $LOG_FILE"
Write-Log "="*80

if ($failed -gt 0) {
    Write-Host "⚠️  一部の実験が失敗しました。詳細はログを確認してください。" -ForegroundColor Yellow
    exit 1
}
else {
    Write-Host "✅ すべての実験が正常に完了しました！" -ForegroundColor Green
    exit 0
}
