# Boundary-Aware U-Net

境界を意識した複合損失関数（BCE + Dice + Boundary Loss）を使用するU-Netモデル。

## 特徴

- **複合損失関数**: BCE + Dice + Boundary Aware Loss
  - BCEで基本的なピクセル分類
  - Dice Lossでクラス不均衡に対応
  - Boundary Lossで境界領域を強調（境界付近を10倍重視）

- **柔軟な入力形式**:
  - **グレースケール (1ch)**: シンプルな1チャンネル入力
  - **LSD + SDF (3ch)**: 線分検出(LSD) + 符号付き距離関数(SDF) + グレースケール

- **詳細な可視化**: 入力、正解、予測、差分を並べて表示

## ファイル構成

```
models/
  boundary_aware_unet.py    # モデル定義、損失関数、特徴抽出
train_boundary_unet.py      # 学習スクリプト
test_boundary_unet.py       # テストスクリプト
```

## 使い方

### 1. 学習（グレースケール入力）

```bash
python train_boundary_unet.py \
  --input_type gray \
  --train_images balloon_dataset/syn1000_dataset/images \
  --train_masks balloon_dataset/syn1000_dataset/masks \
  --val_images balloon_dataset/syn200_dataset/images \
  --val_masks balloon_dataset/syn200_dataset/masks \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.0001
```

### 2. 学習（LSD + SDF入力）

```bash
python train_boundary_unet.py \
  --input_type lsd_sdf \
  --train_images balloon_dataset/syn1000_dataset/images \
  --train_masks balloon_dataset/syn1000_dataset/masks \
  --val_images balloon_dataset/syn200_dataset/images \
  --val_masks balloon_dataset/syn200_dataset/masks \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.0001
```

### 3. テスト

```bash
python test_boundary_unet.py \
  --checkpoint models/best_model_gray.pth \
  --test_images balloon_dataset/syn200_dataset/images \
  --test_masks balloon_dataset/syn200_dataset/masks \
  --output_dir test_results/boundary_unet
```

## パラメータ

### 学習パラメータ

- `--input_type`: 入力タイプ (`gray` or `lsd_sdf`)
- `--epochs`: エポック数 (default: 50)
- `--batch_size`: バッチサイズ (default: 4)
- `--lr`: 学習率 (default: 0.0001)
- `--base_channels`: ベースチャンネル数 (default: 64)

### 損失関数の重み

- `--bce_weight`: BCE損失の重み (default: 1.0)
- `--dice_weight`: Dice損失の重み (default: 1.0)
- `--boundary_weight`: Boundary損失の重み (default: 1.0)

## モデルアーキテクチャ

```
Input (1ch or 3ch)
    ↓
Encoder (5層のダウンサンプリング)
    - DoubleConv + MaxPool
    - Channels: 64 → 128 → 256 → 512 → 1024
    ↓
Decoder (4層のアップサンプリング)
    - ConvTranspose + Skip Connection + DoubleConv
    - Channels: 1024 → 512 → 256 → 128 → 64
    ↓
Output (1ch sigmoid)
```

## 損失関数の詳細

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

### LSD + SDF (3ch)
1. **グレースケール**: 正規化 [0, 1]
2. **LSD (Line Segment Detector)**: 線分検出による密度マップ
3. **SDF (Signed Distance Function)**: マスクの境界からの符号付き距離

## 評価メトリクス

- **IoU (Intersection over Union)**: セグメンテーションの精度
- **Precision**: 予測した領域のうち正解の割合
- **Recall**: 正解領域のうち予測できた割合
- **F1 Score**: PrecisionとRecallの調和平均

## 出力

### 学習
- `models/best_model_{input_type}.pth`: ベストモデル
- `models/checkpoint_epoch_{N}_{input_type}.pth`: 定期チェックポイント

### テスト
- `test_results/boundary_unet/test_results.txt`: 評価結果
- `test_results/boundary_unet/visualizations/`: 可視化画像
  - 入力、正解オーバーレイ、予測オーバーレイ
  - 正解マスク、予測マスク、差分（TP/FP/FN）

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

1. **学習が不安定な場合**:
   - 学習率を下げる (`--lr 0.00005`)
   - バッチサイズを増やす (`--batch_size 8`)

2. **境界の検出精度を上げたい場合**:
   - Boundary Lossの重みを増やす (`--boundary_weight 2.0`)

3. **メモリ不足の場合**:
   - バッチサイズを減らす (`--batch_size 2`)
   - ベースチャンネル数を減らす (`--base_channels 32`)

4. **LSD+SDF入力の利点**:
   - 線分情報により輪郭検出が向上
   - SDF情報により境界位置の精度が向上
   - 計算コストは若干増加

## 依存ライブラリ

```bash
pip install torch torchvision opencv-python numpy scipy tqdm
```

## モデルのテスト（簡易）

```bash
# モデル定義のテスト
python models/boundary_aware_unet.py

# 出力例:
# === Boundary-Aware U-Net Test ===
# Gray Input: torch.Size([2, 1, 256, 256]) -> Output: torch.Size([2, 1, 256, 256])
# LSD+SDF Input: torch.Size([2, 3, 256, 256]) -> Output: torch.Size([2, 1, 256, 256])
# Loss: 0.XXXX
# Total parameters: X,XXX,XXX
```
