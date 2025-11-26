# Boundary-Aware SegFormer

境界を意識した複合損失関数（BCE + Dice + Boundary Loss）を使用するSegFormerモデル。
Transformerベースの高性能セグメンテーションモデル。

## 特徴

- **Transformerベース**: MIT-B1バックボーン（階層的Transformer）
  - 局所・大域の両方の特徴を効率的に捉える
  - 長距離依存関係のモデリングが得意

- **複合損失関数**: BCE + Dice + Boundary Aware Loss
  - BCEで基本的なピクセル分類
  - Dice Lossでクラス不均衡に対応
  - Boundary Lossで境界領域を強調（境界付近を10倍重視）

- **柔軟な入力形式**:
  - **グレースケール (1ch)**: シンプルな1チャンネル入力
  - **LSD + SDF + Gray (3ch)**: 線分検出(LSD) + 符号付き距離関数(SDF) + グレースケール

- **ImageNet事前学習**: 転移学習で高精度を実現

- **固定入力サイズ**: 384x512（高解像度対応）

## U-Netとの比較

| 特徴 | SegFormer | U-Net |
|------|-----------|-------|
| アーキテクチャ | Transformer | CNN |
| パラメータ数 | ~13M (MIT-B1) | ~31M |
| 受容野 | グローバル（全画像） | 局所的 |
| 長距離依存 | 得意 | 苦手 |
| 計算効率 | やや重い | 軽い |
| 精度 | 高い（一般に） | 中程度 |

## ファイル構成

```
models/
  boundary_aware_segformer.py  # モデル定義、損失関数、特徴抽出
train_boundary_segformer.py    # 学習スクリプト
test_boundary_segformer.py     # テストスクリプト
```

## 使い方

### 1. 学習（グレースケール入力）

```bash
python train_boundary_segformer.py \
  --input_type gray \
  --train_images balloon_dataset/syn1000_dataset/images \
  --train_masks balloon_dataset/syn1000_dataset/masks \
  --val_images balloon_dataset/syn200_dataset/images \
  --val_masks balloon_dataset/syn200_dataset/masks \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.00006
```

### 2. 学習（LSD + SDF + Gray入力）

```bash
python train_boundary_segformer.py \
  --input_type lsd_sdf \
  --train_images balloon_dataset/syn1000_dataset/images \
  --train_masks balloon_dataset/syn1000_dataset/masks \
  --val_images balloon_dataset/syn200_dataset/images \
  --val_masks balloon_dataset/syn200_dataset/masks \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.00006
```

### 3. テスト

```bash
python test_boundary_segformer.py \
  --checkpoint models/best_segformer_gray.pth \
  --test_images balloon_dataset/syn200_dataset/images \
  --test_masks balloon_dataset/syn200_dataset/masks \
  --output_dir test_results/boundary_segformer
```

## パラメータ

### 学習パラメータ

- `--input_type`: 入力タイプ (`gray` or `lsd_sdf`)
- `--backbone`: SegFormerバックボーン (default: `nvidia/mit-b1`)
- `--pretrained`: ImageNet事前学習モデルを使用
- `--height`: 入力高さ (default: 384)
- `--width`: 入力幅 (default: 512)
- `--epochs`: エポック数 (default: 50)
- `--batch_size`: バッチサイズ (default: 4)
- `--lr`: 学習率 (default: 0.00006, SegFormer推奨値)

### 損失関数の重み

- `--bce_weight`: BCE損失の重み (default: 1.0)
- `--dice_weight`: Dice損失の重み (default: 1.0)
- `--boundary_weight`: Boundary損失の重み (default: 1.0)

## モデルアーキテクチャ

### SegFormer (MIT-B1)

```
Input (1ch or 3ch, 384x512)
    ↓
Hierarchical Transformer Encoder (MIT-B1)
    - Stage 1: H/4 × W/4, C=64
    - Stage 2: H/8 × W/8, C=128
    - Stage 3: H/16 × W/16, C=320
    - Stage 4: H/32 × W/32, C=512
    ↓
Lightweight All-MLP Decoder
    - Multi-scale feature fusion
    - Upsampling to input resolution
    ↓
Output (1ch, 384x512)
```

### MIT-B1 の詳細

- **パラメータ数**: 約13.7M
- **効率的な自己注意機構**: Shifted window attention
- **階層的特徴抽出**: CNNライクな多スケール特徴
- **事前学習**: ImageNet-1k

## 損失関数の詳細

U-Netと同じ複合損失関数を使用：

### 1. BCE (Binary Cross Entropy)
標準的なピクセル単位の分類損失

### 2. Dice Loss
```
Dice = 2 * |X ∩ Y| / (|X| + |Y|)
DiceLoss = 1 - Dice
```

### 3. Boundary Loss
境界領域を検出し、その領域のBCEを10倍に重み付け
```
BoundaryMask = Dilation(Mask) - Erosion(Mask)
WeightedBCE = BCE * (1 + BoundaryMask * 9)
```

## 入力特徴量

### グレースケール (1ch)
- 単純なグレースケール画像を正規化 [0, 1]
- 384x512にリサイズ

### LSD + SDF + Gray (3ch)
1. **グレースケール**: 正規化 [0, 1]
2. **LSD (Line Segment Detector)**: 線分検出による密度マップ
3. **SDF (Signed Distance Function)**: マスクの境界からの符号付き距離

すべて384x512にリサイズ

## 評価メトリクス

- **IoU (Intersection over Union)**: セグメンテーションの精度
- **Precision**: 予測した領域のうち正解の割合
- **Recall**: 正解領域のうち予測できた割合
- **F1 Score**: PrecisionとRecallの調和平均

## 出力

### 学習
- `models/best_segformer_{input_type}.pth`: ベストモデル
- `models/segformer_epoch_{N}_{input_type}.pth`: 定期チェックポイント

### テスト
- `test_results/boundary_segformer/test_results.txt`: 評価結果
- `test_results/boundary_segformer/visualizations/`: 可視化画像

## 可視化の見方

テスト結果の可視化画像は2行3列のグリッド:

**1行目（オーバーレイ）**:
- 左: 入力画像
- 中: 正解マスクのオーバーレイ（緑）
- 右: 予測マスクのオーバーレイ（カラーマップ）

**2行目（マスクと差分）**:
- 左: 正解マスク
- 中: 予測マスク
- 右: 差分
  - 緑: True Positive（正しく検出）
  - 赤: False Positive（誤検出）
  - 青: False Negative（見逃し）

## Tips

### 1. メモリ不足の場合
```bash
# バッチサイズを減らす
--batch_size 2

# 入力サイズを小さくする
--height 256 --width 384

# 軽量バックボーンに変更
--backbone nvidia/mit-b0
```

### 2. 精度を上げたい場合
```bash
# より大きなバックボーンを使用
--backbone nvidia/mit-b2

# エポック数を増やす
--epochs 100

# Boundary Lossの重みを増やす
--boundary_weight 2.0
```

### 3. 学習が不安定な場合
```bash
# 学習率を下げる
--lr 0.00003

# Weight decayを調整
--weight_decay 0.005
```

### 4. GPU使用率が低い場合
```bash
# バッチサイズを増やす
--batch_size 8

# ワーカー数を増やす
--num_workers 8
```

## バックボーン選択ガイド

| モデル | パラメータ数 | 精度 | 速度 | 推奨用途 |
|--------|-------------|------|------|---------|
| MIT-B0 | 3.7M | 中 | 速い | プロトタイピング、リアルタイム |
| MIT-B1 | 13.7M | 高 | 中 | **推奨・バランス型** |
| MIT-B2 | 24.7M | 非常に高 | やや遅い | 最高精度が必要 |
| MIT-B3 | 44.6M | 最高 | 遅い | ベンチマーク用 |

**推奨**: MIT-B1（精度と速度のバランスが良い）

## 依存ライブラリ

```bash
pip install torch torchvision transformers opencv-python numpy scipy tqdm
```

重要: `transformers`ライブラリが必要
```bash
pip install transformers
```

## モデルのテスト（簡易）

```bash
# モデル定義のテスト（初回は事前学習モデルをダウンロード）
python models/boundary_aware_segformer.py

# 出力例:
# === Boundary-Aware SegFormer Test ===
# [1/2] Testing gray (1ch) model...
# Gray Input: torch.Size([2, 1, 384, 512]) -> Output: torch.Size([2, 1, 384, 512])
# [2/2] Testing LSD+SDF+Gray (3ch) model...
# LSD+SDF+Gray Input: torch.Size([2, 3, 384, 512]) -> Output: torch.Size([2, 3, 384, 512])
# [Loss] Testing combined loss...
# Loss: X.XXXX
# Total parameters: 13,XXX,XXX
# ✓ All tests passed!
```

## SegFormer vs U-Net 選択ガイド

### SegFormerを選ぶべき場合:
- より高い精度が必要
- 長距離依存関係が重要（大きな吹き出しなど）
- GPU/メモリに余裕がある
- 最新の手法を試したい

### U-Netを選ぶべき場合:
- 高速な推論が必要
- メモリが限られている
- シンプルな実装が好ましい
- 局所的な特徴が重要

## トラブルシューティング

### 1. `transformers`のインストールエラー
```bash
pip install --upgrade transformers
```

### 2. 事前学習モデルのダウンロードエラー
```bash
# キャッシュをクリア
rm -rf ~/.cache/huggingface

# 再度実行
python models/boundary_aware_segformer.py
```

### 3. CUDA out of memory
- バッチサイズを減らす (`--batch_size 2`)
- 入力サイズを小さくする (`--height 256 --width 384`)
- グラデーションアキュムレーションを使用

### 4. 学習が収束しない
- 学習率を下げる (`--lr 0.00003`)
- 事前学習モデルを使用 (`--pretrained`)
- データセットを確認（正規化、ラベルの正確性）

## 実験例

```bash
# 実験1: グレースケール入力、MIT-B1、デフォルト設定
python train_boundary_segformer.py --input_type gray

# 実験2: LSD+SDF入力、MIT-B1、デフォルト設定
python train_boundary_segformer.py --input_type lsd_sdf

# 実験3: グレースケール入力、MIT-B0（軽量版）
python train_boundary_segformer.py --input_type gray --backbone nvidia/mit-b0

# 実験4: 境界強調（Boundary Loss重視）
python train_boundary_segformer.py --input_type gray --boundary_weight 2.0

# 実験5: 長時間学習
python train_boundary_segformer.py --input_type gray --epochs 100 --lr 0.00003
```

## ベンチマーク（参考）

典型的な性能（データセット・環境依存）:

| モデル | 入力 | IoU | Params | GPU Memory | 速度 (FPS) |
|--------|------|-----|--------|------------|-----------|
| U-Net | Gray 1ch | 0.85-0.90 | 31M | 4GB | ~30 |
| SegFormer-B1 | Gray 1ch | 0.87-0.92 | 14M | 6GB | ~20 |
| SegFormer-B1 | LSD+SDF 3ch | 0.88-0.93 | 14M | 6GB | ~18 |

※ 実際の性能はデータセット、ハイパーパラメータ、ハードウェアに依存

## 参考文献

- SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (NeurIPS 2021)
- Hugging Face Transformers: https://huggingface.co/docs/transformers
