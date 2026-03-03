# マンガ吹き出し・オノマトペ セグメンテーション

マンガ画像から吹き出し（Speech Balloon）やオノマトペ（Onomatopoeia）を自動検出するセマンティックセグメンテーションモデルの学習・評価プロジェクト。

---

## 目次

1. [プロジェクト概要](#プロジェクト概要)
2. [研究の背景と目的](#研究の背景と目的)
3. [環境構築](#環境構築)
4. [ディレクトリ構成](#ディレクトリ構成)
5. [データパイプライン](#データパイプライン)
6. [スクリプト詳細説明](#スクリプト詳細説明)
7. [モデル詳細](#モデル詳細)
8. [実験の進め方](#実験の進め方)
9. [既存データセット一覧](#既存データセット一覧)
10. [注意事項・Tips](#注意事項tips)
11. [トラブルシューティング](#トラブルシューティング)

---

## プロジェクト概要

### 研究テーマ
**合成データを用いたマンガセグメンテーションモデルの学習と実データへの汎化**

### 主な貢献
1. 生成AIを用いた吹き出し・背景画像の自動生成
2. 実際のマンガ統計に基づいたリアルな合成データセット生成
3. パネル検出を用いた自然な配置アルゴリズム
4. 複数のセグメンテーションモデルの比較評価

### 処理フロー（概要）
```
[素材生成] → [合成データセット作成] → [モデル学習] → [実データ評価]
    ↓              ↓                      ↓              ↓
 GPT-4o等で    吹き出しを背景に      U-Net等で学習   Manga109等で
 吹き出し生成   合成してペア作成                      精度検証
```

---

## 研究の背景と目的

### 背景
- マンガのデジタル化に伴い、吹き出しやオノマトペの自動検出需要が増加
- 実データのアノテーションは高コスト（1ページあたり数十分かかる）
- 合成データで学習し、実データに汎化できれば大幅なコスト削減が可能

### 研究課題
1. **Domain Gap問題**: 合成データと実マンガの見た目の違い
2. **配置の自然さ**: 吹き出しは通常コマの角や端に配置される
3. **サイズ分布**: 実際のマンガにおける吹き出しサイズの統計的特性

### 本プロジェクトのアプローチ
- **統計ベースサンプリング**: 実マンガから抽出した統計情報（log-normal分布）を使用
- **パネル検出配置**: 背景画像からコマを検出し、自然な位置に配置
- **データ拡張**: マンガ特有の変形（斜体、透明度変化など）を適用

---

## 環境構築

### 必要なPythonバージョン
- Python 3.8以上推奨

### 依存パッケージのインストール

```bash
# 基本パッケージ
pip install torch torchvision
pip install opencv-python numpy tqdm Pillow

# セグメンテーションモデル用
pip install segmentation-models-pytorch

# SegFormer用（オプション）
pip install transformers

# 実験管理（オプション）
pip install wandb

# OpenAI API使用時（画像生成）
pip install openai
```

### Conda環境の場合
```bash
conda create -n manga-seg python=3.10
conda activate manga-seg
pip install -r requirements.txt  # 要作成
```

### GPUの確認
```python
import torch
print(torch.cuda.is_available())  # Trueなら利用可能
print(torch.cuda.device_count())  # GPU数
```

---

## ディレクトリ構成

```
t2i_balloon_gen/
│
├── 【データセット作成スクリプト】
│   ├── create_syn_balloon_dataset.py      # ★メイン：吹き出し合成データセット
│   ├── create_syn_onomatopoeia_dataset.py # ★メイン：オノマトペ合成データセット
│   ├── create_synreal_dataset.py          # 合成+実データの統合
│   ├── create_real_test_datasets.py       # Manga109からテストセット作成
│   ├── create_finetune_dataset.py         # ファインチューニング用データセット
│   ├── create_limited_dataset.py          # 枚数制限付きデータセット
│   └── create_mixed_dataset.py            # 複数データセットの混合
│
├── 【画像生成・素材準備】
│   ├── balloon_generate.py                # OpenAI APIで吹き出し画像生成
│   ├── back_generate.py                   # OpenAI APIで背景画像生成
│   ├── extract_real_balloons.py           # Manga109から実吹き出し抽出
│   └── seg_balloon_dataset_from_manga109seg.py  # Manga109セグメンテーション利用
│
├── 【合成処理】
│   ├── composite_balloons.py              # 基本的な合成処理
│   ├── composite_balloons_optimized.py    # 最適化版（高速）
│   ├── composite_balloons_crop_optimized.py  # クロップ最適化版
│   └── generate_random_composite.py       # ランダム合成（デバッグ用）
│
├── 【モデル学習】
│   ├── train_unet_split.py                # U-Net学習
│   ├── train_resnet_unet.py               # ResNet-UNet学習
│   ├── train_deeplabv3plus.py             # DeepLabv3+学習
│   ├── train_boundary_unet.py             # 境界認識U-Net学習
│   ├── train_boundary_segformer.py        # 境界認識SegFormer学習
│   └── finetune_unet.py                   # 事前学習済みモデルの微調整
│
├── 【モデル評価】
│   ├── test_unet.py                       # U-Net評価
│   ├── test_resnet_unet.py                # ResNet-UNet評価
│   ├── test_deeplabv3plus.py              # DeepLabv3+評価
│   ├── test_boundary_unet.py              # 境界認識U-Net評価
│   └── test_boundary_segformer.py         # 境界認識SegFormer評価
│
├── 【ユーティリティ】
│   ├── generate_all_masks.py              # 複数画像のマスク一括生成
│   ├── create_onomatopeia_masks.py        # オノマトペ用マスク生成
│   ├── overlay_masks.py                   # マスクのオーバーレイ表示
│   └── rename.py                          # ファイル名一括変更
│
├── 【統計ファイル】★重要：削除禁止
│   ├── balloon_count_statistics.txt       # 吹き出し個数の統計
│   ├── balloon_size_statistics.txt        # 吹き出しサイズの統計
│   └── onomatopoeia_statistics.txt        # オノマトペの統計
│
├── 【サブディレクトリ】
│   ├── scripts/                           # 実験自動化シェルスクリプト
│   ├── tools/                             # 分析・検証ツール群
│   ├── docs/                              # 追加ドキュメント
│   │
│   ├── balloons/                          # 生成した吹き出し画像
│   ├── balloons_real/                     # 実データから抽出した吹き出し
│   ├── onomatopoeias/                     # オノマトペ画像
│   │
│   ├── backs_real/                        # 実マンガ背景
│   ├── generated_backs/                   # 生成した背景（単ページ）
│   ├── generated_double_backs/            # 生成した背景（見開き）
│   ├── generated_double_backs_1536x1024/  # ★主に使用：見開き背景
│   ├── generated_double_backs_gray/       # グレースケール背景
│   │
│   ├── balloon_dataset/                   # 吹き出しデータセット群
│   ├── onomatopoeia_dataset/              # オノマトペデータセット群
│   │
│   ├── balloon_experiment_results/        # 吹き出し実験結果
│   ├── balloon_results/                   # 吹き出しモデル出力
│   ├── onomatopoeia_experiment_results/   # オノマトペ実験結果
│   ├── onomatopoeia_results/              # オノマトペモデル出力
│   │
│   ├── models/                            # 学習済みモデル保存先
│   ├── bodies/                            # キャラクター体画像
│   └── body_masks/                        # キャラクターマスク
│
└── 【その他】
    ├── README.md                          # このファイル
    └── CONDA_GUIDE.md                     # Conda環境構築ガイド
```

---

## データパイプライン

### 全体の流れ

```
┌─────────────────────────────────────────────────────────────────────┐
│                        【素材準備フェーズ】                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐    │
│   │ GPT-4oで     │      │ GPT-4oで     │      │ Manga109から  │    │
│   │ 吹き出し生成  │      │ 背景生成     │      │ 実吹き出し抽出│    │
│   │              │      │              │      │              │    │
│   │balloon_      │      │back_         │      │extract_real_ │    │
│   │generate.py   │      │generate.py   │      │balloons.py   │    │
│   └──────┬───────┘      └──────┬───────┘      └──────┬───────┘    │
│          ↓                     ↓                     ↓            │
│   balloons/             generated_double_      balloons_real/     │
│   (PNG, 透過)           backs_1536x1024/       (PNG, 透過)        │
│                         (JPEG)                                    │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     【データセット生成フェーズ】                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   入力: 吹き出し画像 + 背景画像 + 統計ファイル                       │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │         create_syn_balloon_dataset.py                        │  │
│   │                                                              │  │
│   │  1. 背景画像を読み込み                                       │  │
│   │  2. パネル（コマ）を検出（二値化→輪郭検出）                  │  │
│   │  3. 統計ファイルから吹き出し個数・サイズをサンプリング       │  │
│   │  4. 各吹き出しを配置:                                        │  │
│   │     - 40%: パネルの角に配置（実マンガに近い）                │  │
│   │     - 60%: パネル内ランダム配置                              │  │
│   │  5. 合成画像とマスク画像を保存                               │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│   出力: balloon_dataset/synXXXX/                                    │
│         ├── images/     # 合成画像                                  │
│         ├── masks/      # 正解マスク（白=吹き出し、黒=背景）        │
│         ├── train.txt   # 学習用ファイルリスト                      │
│         └── val.txt     # 検証用ファイルリスト                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        【モデル学習フェーズ】                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐ │
│   │              train_unet_split.py など                         │ │
│   │                                                               │ │
│   │  入力: balloon_dataset/synXXXX/                               │ │
│   │  出力: models/synXXXX-unet.pt                                 │ │
│   │                                                               │ │
│   │  学習設定:                                                    │ │
│   │  - 損失関数: BCEWithLogitsLoss（二値分類）                   │ │
│   │  - 最適化: Adam（lr=1e-4）                                   │ │
│   │  - 画像サイズ: 512x512にリサイズ                             │ │
│   │  - バッチサイズ: 8（GPUメモリ依存）                          │ │
│   └──────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        【評価フェーズ】                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   テストデータ: Manga109の実マンガ画像（手動アノテーション）        │
│                                                                     │
│   評価指標:                                                         │
│   - IoU (Intersection over Union): 領域の重なり度合い              │
│   - Dice係数: 2 * |A∩B| / (|A| + |B|)                             │
│   - Precision / Recall / F1                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## スクリプト詳細説明

### データセット作成（最重要）

#### `create_syn_balloon_dataset.py` ★メインスクリプト

吹き出し合成データセットを生成する統合スクリプト。

**基本的な使い方:**
```bash
# デフォルト設定（パネル検出配置 ON）
python create_syn_balloon_dataset.py \
    --output-dir balloon_dataset/syn1000 \
    --target-images 1000

# データ拡張あり
python create_syn_balloon_dataset.py \
    --output-dir balloon_dataset/syn1000-aug \
    --target-images 1000 \
    --augmentation

# パネル検出なし（完全ランダム配置）
python create_syn_balloon_dataset.py \
    --output-dir balloon_dataset/syn1000-random \
    --target-images 1000 \
    --no-panel-placement
```

**主要な引数:**
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--output-dir` | 必須 | 出力ディレクトリ |
| `--target-images` | 1000 | 生成する画像数 |
| `--balloon-dir` | `balloons/` | 吹き出し画像フォルダ |
| `--back-dir` | `generated_double_backs_1536x1024/` | 背景画像フォルダ |
| `--augmentation` | False | データ拡張を有効化 |
| `--no-panel-placement` | False | パネル検出配置を無効化 |
| `--train-ratio` | 0.8 | 学習データの割合 |

**内部処理の詳細:**

1. **パネル検出アルゴリズム**
   ```python
   # 二値化でコマ境界を検出
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
   contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   # 面積フィルタリングで有効なパネルのみ抽出
   panels = [c for c in contours if cv2.contourArea(c) > min_area]
   ```

2. **吹き出し個数サンプリング**
   - `balloon_count_statistics.txt` の統計情報を使用
   - Log-normal分布からサンプリング（実マンガの分布に近似）

3. **サイズサンプリング**
   - `balloon_size_statistics.txt` の統計情報を使用
   - 背景幅の5%〜18%の範囲でサンプリング

4. **配置戦略**
   - **角配置 (40%)**: パネルの四隅のいずれかに配置
   - **ランダム配置 (60%)**: パネル内のランダムな位置に配置

**データ拡張オプション (`--augmentation`):**
- 回転: ±20度
- 水平反転: 50%確率
- 端の切り取り: 吹き出しの端を切って配置
- 線の細化: 輪郭線を細くする
- コマ角の四角化: 角を直角に

---

#### `create_syn_onomatopoeia_dataset.py` ★メインスクリプト

オノマトペ合成データセットを生成する統合スクリプト。

**基本的な使い方:**
```bash
# デフォルト設定（パネル内配置 ON）
python create_syn_onomatopoeia_dataset.py \
    --output-dir onomatopoeia_dataset/syn1000 \
    --target-images 1000

# マンガ特有のデータ拡張あり
python create_syn_onomatopoeia_dataset.py \
    --output-dir onomatopoeia_dataset/syn1000-aug \
    --target-images 1000 \
    --augmentation
```

**オノマトペ特有のデータ拡張:**
| 拡張 | パラメータ | 効果 |
|------|-----------|------|
| 回転 | ±30度 | 動きの表現 |
| アスペクト比変更 | 0.9〜1.1 | 縦横比の変化 |
| せん断変換 | ±15度 | 斜体効果（マンガでよく見られる） |
| 透明度変化 | 0.7〜1.0 | 薄いオノマトペの表現 |
| ガウシアンブラー | kernel 3-7 | 動きのブレ |
| ランダム消去 | 5-15% | 部分的な欠損 |

**吹き出しとの違い:**
- オノマトペはパネル内のどこにでも配置される（角に限定されない）
- より派手な変形（せん断、ブラー）が自然

---

### 画像生成

#### `balloon_generate.py`

OpenAI API (GPT-4o/DALL-E) を使用して吹き出し画像を生成。

```bash
# 環境変数にAPIキーを設定
export OPENAI_API_KEY="your-api-key"

python balloon_generate.py
```

**生成される画像:**
- 透過PNG形式
- 様々な形状の吹き出し（楕円、雲形、叫び形など）
- テキストなし（空の吹き出し）

#### `back_generate.py`

OpenAI APIを使用してマンガ風背景画像を生成。

```bash
python back_generate.py
```

**生成される画像:**
- 見開きサイズ (1536x1024)
- マンガのコマ割りを含む
- グレースケールまたはモノクロ

---

### モデル学習

#### `train_unet_split.py`

基本的なU-Netモデルの学習。

```bash
python train_unet_split.py \
    --root balloon_dataset/syn1000 \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4 \
    --save-path models/syn1000-unet.pt
```

**主要な設定:**
```python
# 内部設定
IMAGE_SIZE = 512  # 入力画像サイズ
ENCODER_NAME = "resnet34"  # エンコーダー（smpライブラリ使用）
IN_CHANNELS = 3  # RGB入力
CLASSES = 1  # 二値セグメンテーション
```

#### `train_boundary_unet.py`

境界認識損失を追加したU-Net。吹き出しの輪郭をより正確に検出。

```bash
python train_boundary_unet.py \
    --root balloon_dataset/syn1000 \
    --epochs 100 \
    --boundary-weight 0.5  # 境界損失の重み
```

**境界認識損失の仕組み:**
```
Total Loss = BCE Loss + λ * Boundary Loss

Boundary Loss: マスクの輪郭部分に対する追加ペナルティ
```

#### `train_deeplabv3plus.py`

Atrous Convolutionを使用したDeepLabv3+モデル。

```bash
python train_deeplabv3plus.py \
    --root balloon_dataset/syn1000 \
    --backbone resnet50 \
    --output-stride 16 \
    --epochs 100
```

**DeepLabv3+の特徴:**
- ASPP (Atrous Spatial Pyramid Pooling) による多スケール特徴抽出
- 大きな受容野で広範囲のコンテキストを考慮
- 計算コストはU-Netより高いが、精度向上の可能性

---

### モデル評価

#### `test_unet.py` (他のtest_*.pyも同様)

```bash
python test_unet.py \
    --model models/syn1000-unet.pt \
    --test-dir test_dataset \
    --output-dir results/syn1000-unet
```

**出力内容:**
- `results/`: 予測マスク画像
- `metrics.json`: IoU, Dice, Precision, Recall, F1
- `comparison/`: 入力画像・正解・予測の並び画像

---

## モデル詳細

### 対応モデル一覧

| モデル | 学習スクリプト | テストスクリプト | 特徴 | 推奨用途 |
|--------|---------------|-----------------|------|----------|
| U-Net | `train_unet_split.py` | `test_unet.py` | 軽量・高速 | 初期実験、ベースライン |
| ResNet-UNet | `train_resnet_unet.py` | `test_resnet_unet.py` | ImageNet事前学習 | 汎化性能重視 |
| DeepLabv3+ | `train_deeplabv3plus.py` | `test_deeplabv3plus.py` | 多スケール処理 | 複雑な形状 |
| Boundary U-Net | `train_boundary_unet.py` | `test_boundary_unet.py` | 輪郭強調 | 境界精度重視 |
| Boundary SegFormer | `train_boundary_segformer.py` | `test_boundary_segformer.py` | Transformer | 最新手法の検証 |

### モデル選択の指針

```
初めての実験 → U-Net（ベースライン）
    ↓
精度が不十分 → ResNet-UNet or DeepLabv3+
    ↓
境界がぼやける → Boundary U-Net
    ↓
計算資源が十分 → Boundary SegFormer
```

---

## 実験の進め方

### 典型的な実験フロー

#### Step 1: 小規模データセットで動作確認
```bash
# まず200枚で動作確認
python create_syn_balloon_dataset.py \
    --output-dir balloon_dataset/test200 \
    --target-images 200

# 短いエポックで学習
python train_unet_split.py \
    --root balloon_dataset/test200 \
    --epochs 10 \
    --batch-size 4
```

#### Step 2: 本格的な実験
```bash
# 1000〜2000枚のデータセット
python create_syn_balloon_dataset.py \
    --output-dir balloon_dataset/syn1000 \
    --target-images 1000 \
    --augmentation

# 100エポック学習
python train_unet_split.py \
    --root balloon_dataset/syn1000 \
    --epochs 100 \
    --batch-size 8
```

#### Step 3: 複数条件での比較実験
```bash
# scripts/フォルダのシェルスクリプトを使用
./scripts/run_balloon_experiments.ps1   # Windows
./scripts/run_onomatopoeia_experiments.sh  # macOS/Linux
```

### 実験結果の管理

```bash
# Weights & Biases（推奨）
pip install wandb
wandb login

# 学習スクリプト内でwandb使用可能
python train_unet_split.py --wandb-project balloon-seg
```

---

## 既存データセット一覧

### balloon_dataset/ 内のデータセット

| ディレクトリ名 | 枚数 | 特徴 |
|---------------|------|------|
| `syn200_dataset/` | 200 | 少量データ実験用 |
| `syn500_dataset/` | 500 | 中規模データ |
| `syn1000_dataset/` | 1000 | 標準的な実験用 |
| `syn1500_dataset/` | 1500 | 大規模データ |
| `syn2000_dataset/` | 2000 | 大規模データ |
| `syn1000-corner/` | 1000 | 角配置あり |
| `syn1000-balloon-corner/` | 1000 | パネル角配置 |
| `realballoons_synbacks_200/` | 200 | 実吹き出し+合成背景 |

### ディレクトリ命名規則
- `synXXXX`: 合成データXXXX枚
- `-corner`: パネル角配置を使用
- `-balloon-corner`: 吹き出しをパネル角に優先配置
- `-aug`: データ拡張あり
- `-bXX` / `-bgXX`: 吹き出し/背景の混合比率

---

## 注意事項・Tips

### 絶対に削除してはいけないファイル

```
⚠️ 削除禁止
├── balloon_count_statistics.txt   # 吹き出し個数統計
├── balloon_size_statistics.txt    # 吹き出しサイズ統計
└── onomatopoeia_statistics.txt    # オノマトペ統計

これらはデータセット生成時に参照される重要な統計情報です。
削除すると、実マンガに近い分布のデータが生成できなくなります。
```

### よくある問題と対処法

#### GPUメモリ不足
```bash
# バッチサイズを小さくする
python train_unet_split.py --batch-size 4  # デフォルト8→4

# 画像サイズを小さくする（スクリプト内で変更）
IMAGE_SIZE = 384  # デフォルト512→384
```

#### 学習が進まない
- 学習率を確認: `--lr 1e-4` が一般的
- データセットを確認: `images/` と `masks/` のファイル数が一致しているか
- マスク画像が正しいか目視確認

#### パネル検出が失敗する
- 背景画像の品質を確認（コマ境界が明確か）
- 二値化閾値の調整が必要な場合あり

### 実験時のTips

1. **まず小規模で試す**: 200枚、10エポックで動作確認
2. **ログを確認**: tqdmの進捗やloss値の推移をチェック
3. **中間結果を保存**: `--save-interval 10` などで定期保存
4. **GitでコードをCommit**: 実験条件を再現できるように

---

## トラブルシューティング

### ModuleNotFoundError: No module named 'segmentation_models_pytorch'
```bash
pip install segmentation-models-pytorch
```

### CUDA out of memory
```bash
# バッチサイズを減らす
python train_unet_split.py --batch-size 2

# または環境変数で制限
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### FileNotFoundError: balloon_count_statistics.txt
```
統計ファイルが見つかりません。
以下を確認:
1. カレントディレクトリが正しいか
2. ファイルが存在するか: ls *.txt
3. ファイル名のtypoがないか
```

### OpenAI API関連エラー
```bash
# APIキーを確認
echo $OPENAI_API_KEY

# レート制限の場合は待機
# スクリプト内でsleep追加が必要な場合あり
```

### マスク画像が真っ黒/真っ白
- 合成処理のバグの可能性
- `generate_random_composite.py` でデバッグ確認
- 閾値処理の問題の可能性

---

## 参考文献・リソース

### 使用ライブラリ
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models_pytorch)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

### 関連論文
- U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)
- DeepLab v3+: Encoder-Decoder with Atrous Separable Convolution (2018)
- SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (2021)

### データセット
- [Manga109](http://www.manga109.org/): 日本のマンガデータセット

---

## 連絡先

質問や問題があれば、研究室のSlackまたはメールで連絡してください。

---

最終更新: 2026年3月
