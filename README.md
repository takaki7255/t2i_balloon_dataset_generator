# 🎈 Manga Balloon Generation Project

マンガ吹き出し自動生成・合成システム

## 📁 プロジェクト構造

### 🎯 メインスクリプト

#### データセット作成
- `create_syn_dataset.py` - **メインスクリプト**: 統計情報ベースの高品質データセット生成
- `create_syn_dataset_aug.py` - データ拡張付きデータセット生成
- `create_synreal_dataset.py` - 合成+実データの統合データセット生成
- `create_real_test_datasets.py` - 実データのテストデータセット作成

#### 吹き出し・背景生成
- `balloon_generate.py` - 吹き出し画像生成
- `back_generate.py` - 背景画像生成
- `balloon_mask.py` - マスク生成

#### 合成システム
- `composite_balloons.py` - 基本合成機能
- `composite_balloons_optimized.py` - 最適化版合成
- `composite_balloons_crop_optimized.py` - クロッピング最適化版
- `generate_random_composite.py` - ランダム合成

#### モデル学習・評価
- `train_unet_balloon.py` - U-Net学習（メイン）
- `train_unet_split.py` - U-Netデータ分割学習
- `train_deeplabv3_split.py` - **DeepLab v3+学習**（ASPP搭載）
- `run_deeplabv3_experiments.py` - DeepLab v3+バッチ実験実行
- `finetune_unet.py` - ファインチューニング
- `test_unet.py` - モデル評価

#### マスク生成
- `generate_all_masks.py` - 従来マスク生成
- `generate_all_mask_sam2.py` - SAM2ベースマスク生成

#### その他
- `seg_balloon_dataset_from_manga109seg.py` - Manga109データセット処理
- `split_dataset.py` - データセット分割
- `inpaint_openai.py` - OpenAI inpainting

### 🛠️ ツール（tools/）

#### データ分析ツール
- `dataset_quality_analyzer.py` - **総合品質分析ツール**
- `analyze_balloon_scale.py` - 吹き出しサイズ分析
- `analyze_new_stats.py` - 新設定統計分析
- `analyze_optimization.py` - 最適化効果分析
- `analyze_deeplabv3_architecture.py` - DeepLab v3+アーキテクチャ分析
- `compare_architectures.py` - U-Net vs DeepLab v3+比較
- `explain_deeplabv3_differences.py` - 学習スクリプト差分解説

#### テスト・検証ツール
- `test_statistical_sampling.py` - 統計サンプリングテスト
- `test_crop_comparison.py` - クロッピング効果比較
- `test_scale_sampling.py` - スケールサンプリングテスト
- `check_balloons_size.py` - 吹き出しサイズチェック
- `check_alpha.py` - アルファチャンネル確認

#### 評価・比較ツール
- `performance_comparison.py` - パフォーマンス比較
- `evaluate_current_settings.py` - 設定評価

### 📊 データ・設定ファイル

#### 統計情報
- `balloon_count_statistics.txt` - **重要**: 実データ統計情報
- `balloon_efficiency_analysis.png` - 効率分析結果
- `balloon_size_distribution.png` - サイズ分布グラフ

#### モデル
- `sam2.1_hiera_base_plus.pt` - SAM2モデル

### 📁 データディレクトリ

#### 生成データ
- `generated_balloons/` - 生成された吹き出し画像
- `generated_backs/` - 生成された背景画像
- `generated_double_backs/` - 二重背景画像
- `masks/` - マスクファイル

#### データセット
- `syn_mihiraki300_dataset/` - **最新メインデータセット**（300枚）
- `syn_dataset_aug/` - 拡張データセット
- `syn_mihiraki_dataset_aug/` - mihiraki拡張データセット
- `real_dataset/` - 実データセット
- `synthetic_dataset/` - 旧合成データセット
- `synthetic_real_dataset/` - 合成+実データ統合

#### テスト・結果
- `test_results/` - モデル評価結果
- `test_image/` - テスト画像
- `crop_comparison_test/` - クロッピング比較結果
- `real-unet-03/` - 学習済みモデル

## 🎯 使用方法

### 1. 基本的なデータセット生成
```bash
python create_syn_dataset.py
```

### 2. データセット品質分析
```bash
cd tools
python dataset_quality_analyzer.py
```

### 3. モデル学習

#### U-Net学習
```bash
python train_unet_split.py
```

#### DeepLab v3+学習
```bash
# 単一実験
python train_deeplabv3_split.py

# バッチ実験（複数設定）
python run_deeplabv3_experiments.py
```

### 4. モデル評価
```bash
python test_unet.py
```

## 📈 最新の成果

### データセット品質（syn_mihiraki300_dataset）
- **吹き出し個数精度**: 平均12.40個 vs 実際12.26個（誤差1.1%）
- **画面幅比**: 9.9%（適切なサイズ）
- **クロップ効率**: 51.1%（大幅な余白削減）
- **統計準拠**: 実際のマンガ分布に準拠

### 主要改善点
1. ✅ CFG設定出力の実装
2. ✅ クロッピング最適化（余白51%削減）
3. ✅ 統計情報ベースサンプリング
4. ✅ 詳細ログシステム
5. ✅ サイズ最適化（25%→10%）
6. ✅ 包括的分析ツール

## 🔧 設定

主要な設定は`create_syn_dataset.py`内のCFGで管理：

```python
CFG = {
    "NUM_BALLOONS_RANGE": (7, 17),      # 実際の統計に合わせた範囲
    "SCALE_MODE": "lognormal",           # 統計ベースサンプリング
    "SCALE_MEAN": 0.10,                  # 背景幅の10%
    "SCALE_CLIP": (0.05, 0.18),          # 5%-18%の範囲
    "COUNT_STATS_FILE": "balloon_count_statistics.txt"
}
```

## 📋 注意事項

- `tools/`内のスクリプトは親ディレクトリから実行される前提
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
