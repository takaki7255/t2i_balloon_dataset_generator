#!/bin/bash
# =============================================================================
# Onomatopoeia U-Net Training & Testing Experiment Script
# =============================================================================
# train_unet_split.pyを使用して、onomatopoeia_dataset内の全データセットで
# U-Netモデルを学習し、test_unet.pyでテストを実行します。
#
# 使用方法:
#   chmod +x run_onomatopoeia_experiments.sh
#   ./run_onomatopoeia_experiments.sh
#
# オプション:
#   ./run_onomatopoeia_experiments.sh --skip-training  # 学習をスキップ
#   ./run_onomatopoeia_experiments.sh --skip-test      # テストをスキップ
#
# Windows (Git Bash) の場合:
#   bash run_onomatopoeia_experiments.sh
# =============================================================================

set -e  # エラーが発生したらスクリプトを停止

# オプション解析
SKIP_TRAINING=false
SKIP_TEST=false

for arg in "$@"; do
    case $arg in
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
    esac
done

# 基本設定
DATASET_ROOT="./onomatopoeia_dataset"
MODELS_DIR="./onomatopoeia_models"
RESULTS_DIR="./onomatopoeia_results"
WANDB_PROJECT="onomatopoeia-seg"

# テストデータセット (評価用)
TEST_DATASETS=("test300_dataset")

# 学習対象データセット
TRAIN_DATASETS=(
    "syn200"
    "syn200-aug"
    "syn500"
    "syn500-aug"
    "syn500-random"
    "syn1000"
    "syn1000-panel-aug"
    "syn2000"
    "syn2000-aug"
    "real200_dataset"
    "real1000_dataset"
    "real2000_dataset"
    "real3000_dataset"
)

# モデル保存ディレクトリを作成
mkdir -p "$MODELS_DIR"
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "  Onomatopoeia U-Net Training & Testing"
echo "=============================================="
echo ""
echo "📁 Dataset root: $DATASET_ROOT"
echo "💾 Models directory: $MODELS_DIR"
echo "📊 Results directory: $RESULTS_DIR"
echo "📊 Wandb project: $WANDB_PROJECT"
echo ""
echo "Training datasets (${#TRAIN_DATASETS[@]}):"
for ds in "${TRAIN_DATASETS[@]}"; do
    echo "  - $ds"
done
echo ""
echo "Test datasets (${#TEST_DATASETS[@]}):"
for ds in "${TEST_DATASETS[@]}"; do
    echo "  - $ds"
done
echo ""

# 学習パラメータ
EPOCHS=100
BATCH_SIZE=8
LEARNING_RATE=1e-4
PATIENCE=15

# ============================================
# Training Loop
# ============================================
if [ "$SKIP_TRAINING" = false ]; then
    echo "=============================================="
    echo "  Phase 1: Training"
    echo "=============================================="

    for dataset in "${TRAIN_DATASETS[@]}"; do
        echo ""
        echo "----------------------------------------------"
        echo "🚀 Training: $dataset"
        echo "----------------------------------------------"
        
        dataset_path="$DATASET_ROOT/$dataset"
        
        # データセットが存在するか確認
        if [ ! -d "$dataset_path" ]; then
            echo "⚠️  Dataset not found: $dataset_path - Skipping"
            continue
        fi
        
        # train/valディレクトリが存在するか確認
        if [ ! -d "$dataset_path/train" ] || [ ! -d "$dataset_path/val" ]; then
            echo "⚠️  train/val directories not found in $dataset_path - Skipping"
            continue
        fi
        
        # 学習を実行
        echo "📝 Running: python train_unet_split.py --root $dataset_path --dataset $dataset"
        
        python train_unet_split.py \
            --root "$dataset_path" \
            --dataset "$dataset" \
            --models-dir "$MODELS_DIR" \
            --epochs "$EPOCHS" \
            --batch "$BATCH_SIZE" \
            --lr "$LEARNING_RATE" \
            --patience "$PATIENCE" \
            --wandb-proj "$WANDB_PROJECT"
        
        echo "✅ Completed: $dataset"
    done

    echo ""
    echo "=============================================="
    echo "  Training Completed!"
    echo "=============================================="
else
    echo "Skipping training phase..."
fi

echo ""
echo "Trained models are saved in: $MODELS_DIR"
echo ""

# ============================================
# Testing Loop
# ============================================
if [ "$SKIP_TEST" = false ]; then
    echo ""
    echo "=============================================="
    echo "  Phase 2: Testing"
    echo "=============================================="
    
    # 学習済みモデルを検索
    model_files=$(ls "$MODELS_DIR"/*.pt 2>/dev/null || true)
    
    if [ -z "$model_files" ]; then
        echo "⚠️  No trained models found in $MODELS_DIR"
    else
        model_count=$(ls "$MODELS_DIR"/*.pt 2>/dev/null | wc -l)
        echo "Found $model_count models to test"
        
        for model_file in "$MODELS_DIR"/*.pt; do
            model_tag=$(basename "$model_file" .pt)
            
            for test_dataset in "${TEST_DATASETS[@]}"; do
                echo ""
                echo "----------------------------------------------"
                echo "🧪 Testing: $model_tag on $test_dataset"
                echo "----------------------------------------------"
                
                test_data_path="$DATASET_ROOT/$test_dataset"
                
                # テストデータセットが存在するか確認
                if [ ! -d "$test_data_path" ]; then
                    echo "⚠️  Test dataset not found: $test_data_path - Skipping"
                    continue
                fi
                
                # images/masksディレクトリが存在するか確認
                if [ ! -d "$test_data_path/images" ] || [ ! -d "$test_data_path/masks" ]; then
                    echo "⚠️  images/masks directories not found in $test_data_path - Skipping"
                    continue
                fi
                
                # テストを実行
                run_name="$model_tag-test-$test_dataset"
                echo "📝 Running: python test_unet.py --model-tag $model_tag --data-root $test_data_path"
                
                python test_unet.py \
                    --model-tag "$model_tag" \
                    --models-dir "$MODELS_DIR" \
                    --data-root "$test_data_path" \
                    --result-dir "$RESULTS_DIR" \
                    --batch "$BATCH_SIZE" \
                    --wandb-proj "$WANDB_PROJECT" \
                    --run-name "$run_name"
                
                echo "✅ Completed: $run_name"
            done
        done
        
        echo ""
        echo "=============================================="
        echo "  Testing Completed!"
        echo "=============================================="
    fi
else
    echo "Skipping testing phase..."
fi

echo ""
echo "=============================================="
echo "  Experiment Complete"
echo "=============================================="
echo ""

# モデル一覧を表示
echo "Trained models:"
ls -la "$MODELS_DIR"/*.pt 2>/dev/null || echo "  No models found"

# 結果一覧を表示
echo ""
echo "Test results:"
ls -d "$RESULTS_DIR"/*/ 2>/dev/null || echo "  No results found"
