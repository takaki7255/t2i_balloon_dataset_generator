#!/bin/bash
# 複数データセットでU-Net学習とテストを自動実行するスクリプト
# 使用方法: ./run_experiments.sh [--train-only] [--test-only] [--no-wandb] [--datasets=syn200,syn500]

set -e  # エラー時に停止

# ============================================================================
# オプション解析
# ============================================================================

TRAIN_ONLY=false
TEST_ONLY=false
NO_WANDB=false
DATASETS="all"

for arg in "$@"; do
    case $arg in
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --no-wandb)
            NO_WANDB=true
            shift
            ;;
        --datasets=*)
            DATASETS="${arg#*=}"
            shift
            ;;
        -h|--help)
            echo "使用方法: $0 [OPTIONS]"
            echo ""
            echo "OPTIONS:"
            echo "  --train-only        学習のみ実行"
            echo "  --test-only         テストのみ実行"
            echo "  --no-wandb          Wandb無効化"
            echo "  --datasets=NAMES    実行するデータセット (カンマ区切り, default: all)"
            echo "                      例: --datasets=syn200-corner,syn500-corner"
            echo ""
            echo "利用可能なデータセット:"
            echo "  syn200-corner, syn500-corner, syn750-corner, syn1000-corner"
            exit 0
            ;;
        *)
            echo "不明なオプション: $arg"
            echo "ヘルプ: $0 --help"
            exit 1
            ;;
    esac
done

# ============================================================================
# 設定: 学習・テスト対象のデータセット
# ============================================================================

# データセット定義配列
declare -a DATASET_NAMES=("syn200-corner" "syn500-corner" "syn750-corner" "syn1000-corner")
declare -a DATASET_ROOTS=("./balloon_dataset/syn200_dataset" "./balloon_dataset/syn500_dataset" "./balloon_dataset/syn750_dataset" "./balloon_dataset/syn1000_dataset")
declare -a DATASET_EPOCHS=(100 100 100 100)
declare -a DATASET_BATCH=(8 8 8 8)
declare -a DATASET_PATIENCE=(15 15 15 15)

# テスト用データセット
TEST_DATA_ROOT="./test_dataset"

# モデル保存ディレクトリ
MODELS_DIR="./balloon_models"

# 結果保存ディレクトリ
RESULTS_DIR="./experiment_results"

# Wandb プロジェクト名
WANDB_PROJECT="balloon-seg-experiments"

# ============================================================================
# データセットフィルタリング
# ============================================================================

if [ "$DATASETS" != "all" ]; then
    IFS=',' read -ra SELECTED <<< "$DATASETS"
    NEW_NAMES=()
    NEW_ROOTS=()
    NEW_EPOCHS=()
    NEW_BATCH=()
    NEW_PATIENCE=()
    
    for i in "${!DATASET_NAMES[@]}"; do
        for selected in "${SELECTED[@]}"; do
            if [ "${DATASET_NAMES[$i]}" == "$selected" ]; then
                NEW_NAMES+=("${DATASET_NAMES[$i]}")
                NEW_ROOTS+=("${DATASET_ROOTS[$i]}")
                NEW_EPOCHS+=("${DATASET_EPOCHS[$i]}")
                NEW_BATCH+=("${DATASET_BATCH[$i]}")
                NEW_PATIENCE+=("${DATASET_PATIENCE[$i]}")
            fi
        done
    done
    
    if [ ${#NEW_NAMES[@]} -eq 0 ]; then
        echo "❌ 指定されたデータセットが見つかりません: $DATASETS"
        echo "利用可能なデータセット: ${DATASET_NAMES[*]}"
        exit 1
    fi
    
    DATASET_NAMES=("${NEW_NAMES[@]}")
    DATASET_ROOTS=("${NEW_ROOTS[@]}")
    DATASET_EPOCHS=("${NEW_EPOCHS[@]}")
    DATASET_BATCH=("${NEW_BATCH[@]}")
    DATASET_PATIENCE=("${NEW_PATIENCE[@]}")
fi

# ============================================================================
# ディレクトリ作成
# ============================================================================

mkdir -p "$MODELS_DIR"
mkdir -p "$RESULTS_DIR"

# ============================================================================
# 実験ログファイル
# ============================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$RESULTS_DIR/experiment_log_$TIMESTAMP.txt"

log_message() {
    local MESSAGE="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo "$MESSAGE"
    echo "$MESSAGE" >> "$LOG_FILE"
}

# ============================================================================
# 実験開始
# ============================================================================

log_message "================================================================================"
log_message "🚀 U-Net 自動実験スクリプト開始"
log_message "================================================================================"
log_message "実行モード: $(if $TRAIN_ONLY; then echo '学習のみ'; elif $TEST_ONLY; then echo 'テストのみ'; else echo '学習+テスト'; fi)"
log_message "Wandb: $(if $NO_WANDB; then echo '無効'; else echo '有効'; fi)"
log_message "対象データセット数: ${#DATASET_NAMES[@]}"
log_message "ログファイル: $LOG_FILE"
log_message ""

# ============================================================================
# 各データセットで実験実行
# ============================================================================

TOTAL_DATASETS=${#DATASET_NAMES[@]}
COMPLETED=0
FAILED=0

for i in "${!DATASET_NAMES[@]}"; do
    DATASET_NAME="${DATASET_NAMES[$i]}"
    DATASET_ROOT="${DATASET_ROOTS[$i]}"
    EPOCHS="${DATASET_EPOCHS[$i]}"
    BATCH="${DATASET_BATCH[$i]}"
    PATIENCE="${DATASET_PATIENCE[$i]}"
    
    log_message ""
    log_message "================================================================================"
    log_message "📊 データセット: $DATASET_NAME"
    log_message "================================================================================"
    log_message "  ルート: $DATASET_ROOT"
    log_message "  エポック: $EPOCHS"
    log_message "  バッチサイズ: $BATCH"
    log_message "  Patience: $PATIENCE"
    log_message ""
    
    # ========================================================================
    # 1. 学習
    # ========================================================================
    
    if ! $TEST_ONLY; then
        log_message "🏋️  学習開始: $DATASET_NAME"
        log_message "--------------------------------------------------------------------------------"
        
        TRAIN_ARGS=(
            "train_unet_split.py"
            "--root" "$DATASET_ROOT"
            "--dataset" "$DATASET_NAME"
            "--models-dir" "$MODELS_DIR"
            "--epochs" "$EPOCHS"
            "--batch" "$BATCH"
            "--patience" "$PATIENCE"
            "--wandb-proj" "$WANDB_PROJECT"
            "--run-name" "${DATASET_NAME}-train"
        )
        
        if python "${TRAIN_ARGS[@]}"; then
            log_message "✅ 学習完了: $DATASET_NAME"
        else
            log_message "❌ 学習失敗: $DATASET_NAME (exit code: $?)"
            ((FAILED++))
            continue
        fi
    fi
    
    # ========================================================================
    # 2. テスト
    # ========================================================================
    
    if ! $TRAIN_ONLY; then
        log_message ""
        log_message "🧪 テスト開始: $DATASET_NAME"
        log_message "--------------------------------------------------------------------------------"
        
        # モデルタグを取得（最新バージョン）
        MODEL_PATTERN="$MODELS_DIR/${DATASET_NAME}-unet-*.pt"
        LATEST_MODEL=$(ls -t $MODEL_PATTERN 2>/dev/null | head -n 1)
        
        if [ -z "$LATEST_MODEL" ]; then
            log_message "⚠️  モデルが見つかりません: $MODEL_PATTERN"
            log_message "   テストをスキップします"
            continue
        fi
        
        MODEL_TAG=$(basename "$LATEST_MODEL" .pt)
        log_message "   使用モデル: $MODEL_TAG"
        
        TEST_ARGS=(
            "test_unet.py"
            "--model-tag" "$MODEL_TAG"
            "--models-dir" "$MODELS_DIR"
            "--data-root" "$TEST_DATA_ROOT"
            "--result-dir" "$RESULTS_DIR"
            "--batch" "$BATCH"
            "--wandb-proj" "$WANDB_PROJECT"
            "--run-name" "${DATASET_NAME}-test"
        )
        
        if $NO_WANDB; then
            TEST_ARGS+=("--no-wandb")
        fi
        
        if python "${TEST_ARGS[@]}"; then
            log_message "✅ テスト完了: $DATASET_NAME"
            ((COMPLETED++))
        else
            log_message "❌ テスト失敗: $DATASET_NAME (exit code: $?)"
            ((FAILED++))
        fi
    else
        ((COMPLETED++))
    fi
done

# ============================================================================
# 最終結果サマリー
# ============================================================================

log_message ""
log_message "================================================================================"
log_message "🎉 実験完了"
log_message "================================================================================"
log_message "総データセット数: $TOTAL_DATASETS"
log_message "成功: $COMPLETED"
log_message "失敗: $FAILED"
log_message ""
log_message "結果保存先: $RESULTS_DIR"
log_message "ログファイル: $LOG_FILE"
log_message "================================================================================"

if [ $FAILED -gt 0 ]; then
    echo "⚠️  一部の実験が失敗しました。詳細はログを確認してください。"
    exit 1
else
    echo "✅ すべての実験が正常に完了しました！"
    exit 0
fi
