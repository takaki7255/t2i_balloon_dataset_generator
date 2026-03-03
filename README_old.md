# マンガ吹き出し・オノマトペ セグメンテーション

マンガ画像から吹き出し（Speech Balloon）やオノマトペ（Onomatopoeia）を自動検出するセマンティックセグメンテーションモデルの学習・評価プロジェクト。

## 概要

本プロジェクトでは、合成データセットを用いてセグメンテーションモデルを学習し、実際のマンガ画像に対する汎化性能を評価します。

### 主な機能

- **合成データセット生成**: 吹き出し/オノマトペ画像を背景に合成してデータセットを作成
- **複数モデル対応**: U-Net, ResNet-UNet, DeepLabv3+, Boundary-Aware U-Net/SegFormer
- **実験自動化**: シェルスクリプトによるバッチ実験実行

---

## ディレクトリ構成

```
.
├── 📊 データセット作成
│   ├── create_syn_balloon_dataset.py      # 吹き出し合成データセット（統合版）
│   ├── create_syn_onomatopoeia_dataset.py # オノマトペ合成データセット（統合版）
│   ├── create_synreal_dataset.py          # 合成+実データ統合
│   ├── create_real_test_datasets.py       # 実データテストセット作成
│   ├── create_finetune_dataset.py         # ファインチューニング用
│   ├── create_limited_dataset.py          # 制限付きデータセット
│   └── create_mixed_dataset.py            # 混合データセット
│
├── 🎨 画像生成・合成
│   ├── balloon_generate.py                # 吹き出し画像生成（GPT-4o）
│   ├── back_generate.py                   # 背景画像生成（OpenAI API）
│   ├── composite_balloons.py              # 基本合成
│   ├── composite_balloons_optimized.py    # 最適化版合成
│   ├── composite_balloons_crop_optimized.py # クロップ最適化版
│   └── generate_random_composite.py       # ランダム合成
│
├── 🏋️ モデル学習
│   ├── train_unet_split.py                # U-Net
│   ├── train_resnet_unet.py               # ResNet-UNet
│   ├── train_deeplabv3plus.py             # DeepLabv3+
│   ├── train_boundary_unet.py             # Boundary-Aware U-Net
│   ├── train_boundary_segformer.py        # Boundary-Aware SegFormer
│   └── finetune_unet.py                   # U-Net微調整
│
├── 🧪 モデル評価
│   ├── test_unet.py                       # U-Netテスト
│   ├── test_resnet_unet.py                # ResNet-UNetテスト
│   ├── test_deeplabv3plus.py              # DeepLabv3+テスト
│   ├── test_boundary_unet.py              # Boundary-Aware U-Netテスト
│   └── test_boundary_segformer.py         # Boundary-Aware SegFormerテスト
│
├── 🔧 ユーティリティ
│   ├── generate_all_masks.py              # マスク一括生成
│   ├── create_onomatopeia_masks.py        # オノマトペマスク生成
│   ├── extract_real_balloons.py           # Manga109から吹き出し抽出
│   ├── seg_balloon_dataset_from_manga109seg.py # Manga109セグメンテーション抽出
│   └── rename.py                          # ファイル名一括変更
│
├── 📁 データ
│   ├── balloons/                          # 吹き出し画像（生成）
│   ├── balloons_real/                     # 吹き出し画像（実データ）
│   ├── onomatopoeias/                     # オノマトペ画像
│   ├── generated_double_backs_1536x1024/  # 背景画像
│   ├── balloon_dataset/                   # 吹き出しデータセット
│   ├── onomatopoeia_dataset/              # オノマトペデータセット
│   └── models/                            # 学習済みモデル
│
├── 📜 スクリプト・ドキュメント
│   ├── scripts/                           # 実験自動化シェルスクリプト
│   ├── tools/                             # 分析・検証ツール
│   └── docs/                              # ドキュメント
│
└── 📈 実験結果
    ├── balloon_experiment_results/        # 吹き出し実験結果
    ├── balloon_results/                   # 吹き出しモデル出力
    ├── onomatopoeia_experiment_results/   # オノマトペ実験結果
    └── onomatopoeia_results/              # オノマトペモデル出力
```

---

## クイックスタート

### 1. 合成データセット作成

**吹き出しデータセット:**
```bash
# デフォルト: パネル検出 + 角配置 + ランダム配置
python create_syn_balloon_dataset.py \
  --output-dir balloon_dataset/syn1000 \
  --target-images 1000

# データ拡張あり
python create_syn_balloon_dataset.py \
  --output-dir balloon_dataset/syn1000-aug \
  --target-images 1000 \
  --augmentation
```

**オノマトペデータセット:**
```bash
# デフォルト: パネル内配置
python create_syn_onomatopoeia_dataset.py \
  --output-dir onomatopoeia_dataset/syn1000 \
  --target-images 1000

# マンガ特有のaugmentation（せん断、透明度、ブラーなど）
python create_syn_onomatopoeia_dataset.py \
  --output-dir onomatopoeia_dataset/syn1000-aug \
  --target-images 1000 \
  --augmentation
```

### 2. モデル学習

```bash
# U-Net
python train_unet_split.py \
  --root balloon_dataset/syn1000 \
  --epochs 100 \
  --batch-size 8

# DeepLabv3+
python train_deeplabv3plus.py \
  --root balloon_dataset/syn1000 \
  --backbone resnet50 \
  --epochs 100

# Boundary-Aware U-Net
python train_boundary_unet.py \
  --root balloon_dataset/syn1000 \
  --epochs 100 \
  --input-channels 3
```

### 3. モデル評価

```bash
python test_unet.py \
  --model models/syn1000-unet.pt \
  --test-dir test_dataset \
  --output-dir results/
```

---

## 合成データセットの特徴

### 吹き出し配置（`create_syn_balloon_dataset.py`）

| モード | 説明 |
|--------|------|
| **デフォルト** | パネル検出 → 角配置(40%) + ランダム配置(60%) |
| `--no-panel-placement` | 完全ランダム配置 |

**データ拡張オプション:**
- 回転（±20度）
- 水平反転
- 端の切り取り
- 線の細化
- コマ角の四角化

### オノマトペ配置（`create_syn_onomatopoeia_dataset.py`）

| モード | 説明 |
|--------|------|
| **デフォルト** | パネル検出 → パネル内ランダム配置 |
| `--no-panel-placement` | 完全ランダム配置 |

**データ拡張オプション（マンガ特有）:**
- 回転（±30度）
- アスペクト比変更（0.9-1.1）
- せん断変換（±15度）- 斜体効果
- 透明度変化（0.7-1.0）
- ガウシアンブラー - 動きのブレ
- ランダム消去（5-15%）

---

## 対応モデル

| モデル | 学習スクリプト | 特徴 |
|--------|---------------|------|
| U-Net | `train_unet_split.py` | 基本的なEncoder-Decoder |
| ResNet-UNet | `train_resnet_unet.py` | ResNetバックボーン |
| DeepLabv3+ | `train_deeplabv3plus.py` | ASPP + Decoder |
| Boundary-Aware U-Net | `train_boundary_unet.py` | 境界認識損失 |
| Boundary-Aware SegFormer | `train_boundary_segformer.py` | Transformer + 境界認識 |

---

## 実験自動化

`scripts/` フォルダにシェルスクリプトを配置：

```bash
# 吹き出し実験
./scripts/run_balloon_experiments.ps1

# オノマトペ実験
./scripts/run_onomatopoeia_experiments.sh

# モデル比較
./scripts/run_model_comparison.ps1
```

---

## 統計ファイル

- `balloon_count_statistics.txt` - 吹き出し個数統計
- `balloon_size_statistics.txt` - 吹き出しサイズ統計
- `onomatopoeia_statistics.txt` - オノマトペ統計

これらの統計情報に基づいて、log-normal分布でサイズをサンプリングします。

---

## 依存関係

```
torch
torchvision
opencv-python
numpy
tqdm
segmentation-models-pytorch
transformers  # SegFormer用
```

---

## ライセンス

本プロジェクトは研究目的で開発されています。
- `syn_mihiraki300_dataset/`が最新の重要データセット
- 統計情報（`balloon_count_statistics.txt`）は削除厳禁
- 生成データ（`generated_*`）ディレクトリは学習に必要

## 🏆 今後の拡張

- より大規模なデータセット生成
- 追加のデータ拡張手法
- より精密な統計分析
- リアルタイム品質監視

## 🧠 セグメンテーションモデル

### U-Net
従来のエンコーダー・デコーダー構造を持つセグメンテーションモデル。
- スキップ接続による高解像度情報の保持
- 医療画像処理で実績あり
- 軽量で高速な推論

### DeepLab v3+ **（新規追加）**
Atrous Convolutionとマルチスケール処理を特徴とする最新セグメンテーションモデル。

#### 主要コンポーネント
1. **Encoder（エンコーダー）**
   - ResNet50/ResNet101バックボーン
   - Atrous Convolutionによる受容野拡大
   - Output Stride設定（8または16）

2. **ASPP（Atrous Spatial Pyramid Pooling）**
   - 5つの並列ブランチ（1×1 conv, 3×3 atrous conv×3, Global Average Pooling）
   - マルチスケール特徴抽出
   - Output Stride 16: dilation rates [6,12,18]
   - Output Stride 8: dilation rates [12,24,36]

3. **Decoder（デコーダー）**
   - 低レベル特徴との融合
   - バイリニア補間による解像度復元
   - 軽量な設計

#### 設定オプション
```python
CFG = {
    "BACKBONE": "resnet50",        # resnet50 or resnet101
    "OUTPUT_STRIDE": 16,           # 8 or 16
    "WANDB_PROJ": "balloon-seg-deeplabv3"
}
```

#### U-Netとの比較
| 特徴 | U-Net | DeepLab v3+ |
|------|-------|-------------|
| 受容野 | 限定的 | Atrous Convで大幅拡大 |
| マルチスケール | スキップ接続のみ | ASPP+低レベル特徴融合 |
| 計算効率 | 高い | 中程度（ASPP分） |
| 細部精度 | 高い（スキップ接続） | 高い（低レベル特徴融合） |
| 大域的理解 | 限定的 | 優秀（ASPP） |

#### 実験実行
```bash
# 単一設定での学習
python train_deeplabv3_split.py

# 複数設定でのバッチ実験
python run_deeplabv3_experiments.py

# アーキテクチャ分析
python analyze_deeplabv3_architecture.py
python compare_architectures.py
```
