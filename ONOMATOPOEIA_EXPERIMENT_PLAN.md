# オノマトペセグメンテーション実験計画

## 目的
漫画ページからオノマトペを自動検出・セグメンテーションするモデルの開発と評価

---

## 1. データセット準備

### 1.1 合成データセット（Synthetic）

#### A. ランダム配置データセット
**スクリプト**: `create_syn_onomatopoeia_dataset.py`

| データセット名 | 画像数 | 特徴 | 用途 |
|--------------|--------|------|------|
| `syn200_onomatopoeia` | 200 | 基本ベースライン | 初期実験 |
| `syn500_onomatopoeia` | 500 | 中規模 | モデル比較 |
| `syn1000_onomatopoeia` | 1000 | 標準規模 | メイン実験 |
| `syn2000_onomatopoeia` | 2000 | 大規模 | データ量依存性検証 |

**設定**:
- 背景: `generated_double_backs_1536x1024/`
- オノマトペ個数範囲: 1-15個/画像
- スケール範囲: 0.0001-0.005（背景面積比）
- データ拡張: なし

#### B. パネル内配置データセット（データ拡張なし）
**スクリプト**: `create_onomatopoeia_panel_dataset_augmented.py --no-augmentation`

| データセット名 | 画像数 | 特徴 | 用途 |
|--------------|--------|------|------|
| `syn500_panel` | 500 | パネル検出ベース | パネル配置効果検証 |
| `syn1000_panel` | 1000 | 標準規模パネル配置 | ランダム配置との比較 |

**設定**:
- パネル検出: 有効（検出失敗時は全体ランダム）
- オノマトペ個数: パネルサイズに応じて1-5個
- データ拡張: なし

#### C. パネル内配置データセット（データ拡張あり）
**スクリプト**: `create_onomatopoeia_panel_dataset_augmented.py`

| データセット名 | 画像数 | 特徴 | 用途 |
|--------------|--------|------|------|
| `syn500_panel_aug` | 500 | パネル配置+拡張 | 拡張効果の初期検証 |
| `syn1000_panel_aug` | 1000 | 標準規模+拡張 | メイン実験（拡張あり） |
| `syn2000_panel_aug` | 2000 | 大規模+拡張 | 最高性能追求 |

**データ拡張内容**:
- 回転: ±30度（確率70%）
- スケール: 0.8-1.2倍（確率50%）
- アスペクト比: 0.9-1.1倍（確率30%）
- せん断: ±15度（確率40%）
- 透明度: 0.7-1.0（確率50%）
- ブラー: カーネル1,3（確率30%）
- ランダム消去: 5-15%（確率30%）

### 1.2 実画像データセット（Real）

#### テストデータセット
**スクリプト**: `create_real_test_datasets.py`

| データセット名 | 画像数 | ソース | 用途 |
|--------------|--------|--------|------|
| `real_onomatopoeia_test100` | 100 | Manga109 | テスト専用 |
| `real_onomatopoeia_test200` | 200 | Manga109 | 詳細評価 |

**注意**: 実画像は**テストのみ**に使用。学習には使わない。

---

## 2. 実験設計

### 実験1: ベースライン確立（データセット規模の影響）
**目的**: 合成データの規模とモデル性能の関係を把握

| 実験ID | データセット | モデル | エポック | 備考 |
|--------|------------|--------|---------|------|
| E1-1 | syn200_onomatopoeia | U-Net | 100 | 最小規模 |
| E1-2 | syn500_onomatopoeia | U-Net | 100 | 中規模 |
| E1-3 | syn1000_onomatopoeia | U-Net | 100 | 標準規模 |
| E1-4 | syn2000_onomatopoeia | U-Net | 100 | 大規模 |

**評価**:
- テストデータ: `real_onomatopoeia_test100`
- メトリクス: IoU, Dice, Precision, Recall

**期待結果**: データ量に応じた性能向上を確認

---

### 実験2: 配置方法の比較
**目的**: ランダム配置 vs パネル内配置の効果検証

| 実験ID | データセット | モデル | エポック | 配置方法 |
|--------|------------|--------|---------|---------|
| E2-1 | syn1000_onomatopoeia | U-Net | 100 | ランダム配置 |
| E2-2 | syn1000_panel | U-Net | 100 | パネル内配置 |

**評価**:
- テストデータ: `real_onomatopoeia_test100`
- 比較項目: IoU, Dice, 検出率、誤検出率

**仮説**: パネル内配置は実際の漫画に近く、性能向上が期待される

---

### 実験3: データ拡張の効果
**目的**: データ拡張による汎化性能向上の検証

| 実験ID | データセット | モデル | エポック | データ拡張 |
|--------|------------|--------|---------|-----------|
| E3-1 | syn1000_panel | U-Net | 100 | なし |
| E3-2 | syn1000_panel_aug | U-Net | 100 | あり（全種） |
| E3-3 | syn500_panel | U-Net | 100 | なし |
| E3-4 | syn500_panel_aug | U-Net | 100 | あり（全種） |

**評価**:
- テストデータ: `real_onomatopoeia_test200`
- 特に注目: 回転・せん断による斜め配置への対応

**仮説**: データ拡張により、少ないデータでも高性能を達成

---

### 実験4: モデルアーキテクチャ比較
**目的**: 最適なモデル構造の選定

| 実験ID | データセット | モデル | エポック | 備考 |
|--------|------------|--------|---------|------|
| E4-1 | syn1000_panel_aug | U-Net | 100 | ベースライン |
| E4-2 | syn1000_panel_aug | U-Net (ResNet backbone) | 100 | より深い特徴抽出 |
| E4-3 | syn1000_panel_aug | DeepLabV3+ | 100 | ASPP機能 |
| E4-4 | syn1000_panel_aug | SegFormer | 100 | Transformer |

**評価**:
- テストデータ: `real_onomatopoeia_test200`
- 比較項目: IoU, 推論速度、メモリ使用量

**期待結果**: 吹き出しセグメンテーションの知見を活用

---

### 実験5: 最良設定での大規模学習
**目的**: 最高性能モデルの構築

| 実験ID | データセット | モデル | エポック | 備考 |
|--------|------------|--------|---------|------|
| E5-1 | syn2000_panel_aug | 最良モデル | 150 | 最大データ量 |
| E5-2 | syn2000_panel_aug | 最良モデル（低メモリ版） | 150 | メモリ制約対応 |

**評価**:
- テストデータ: `real_onomatopoeia_test200`
- 詳細分析: 誤検出パターン、失敗ケース分析

---

### 実験6: ファインチューニング（オプション）
**目的**: 実画像への適応

| 実験ID | ベースモデル | ファインチューニングデータ | エポック | 備考 |
|--------|------------|----------------------|---------|------|
| E6-1 | E5-1 | 実画像50枚（手動作成） | 20 | ドメイン適応 |

**注意**: 実画像アノテーションのコストが高いため、オプション扱い

---

## 3. 評価指標

### 3.1 定量評価

**セグメンテーション精度**:
- IoU (Intersection over Union)
- Dice係数
- Precision（適合率）
- Recall（再現率）
- F1スコア

**検出性能**:
- 検出率（正しく検出したオノマトペ数 / 全オノマトペ数）
- 誤検出率（誤検出数 / 全検出数）

### 3.2 定性評価

**可視化項目**:
- 予測マスクのオーバーレイ表示
- 誤検出・未検出の具体例
- 複雑なレイアウトでの性能
- 斜め配置への対応

**失敗ケース分析**:
- 小さいオノマトペの検出失敗
- 背景との区別困難なケース
- 重なり合うオノマトペ
- 部分的に隠れたオノマトペ

---

## 4. 実験実行手順

### Phase 1: データセット作成（1-2日）

```powershell
# ランダム配置データセット
python create_syn_onomatopoeia_dataset.py --dataset-name syn200 --target-images 200
python create_syn_onomatopoeia_dataset.py --dataset-name syn500 --target-images 500
python create_syn_onomatopoeia_dataset.py --dataset-name syn1000 --target-images 1000
python create_syn_onomatopoeia_dataset.py --dataset-name syn2000 --target-images 2000

# パネル内配置（拡張なし）
python create_onomatopoeia_panel_dataset_augmented.py --dataset-name syn500_panel --target-images 500 --no-augmentation
python create_onomatopoeia_panel_dataset_augmented.py --dataset-name syn1000_panel --target-images 1000 --no-augmentation

# パネル内配置（拡張あり）
python create_onomatopoeia_panel_dataset_augmented.py --dataset-name syn500_panel_aug --target-images 500
python create_onomatopoeia_panel_dataset_augmented.py --dataset-name syn1000_panel_aug --target-images 1000
python create_onomatopoeia_panel_dataset_augmented.py --dataset-name syn2000_panel_aug --target-images 2000

# 実画像テストセット
python create_real_test_datasets.py --target-category onomatopoeia --output-dir real_onomatopoeia_test100 --sample-size 100
python create_real_test_datasets.py --target-category onomatopoeia --output-dir real_onomatopoeia_test200 --sample-size 200
```

### Phase 2: 実験1-2実行（3-4日）

```powershell
# 実験1: データセット規模
python train_unet_split.py --root onomatopoeia_datasets/syn200/syn200 --dataset syn200-ono --epochs 100
python train_unet_split.py --root onomatopoeia_datasets/syn500/syn500 --dataset syn500-ono --epochs 100
python train_unet_split.py --root onomatopoeia_datasets/syn1000/syn1000 --dataset syn1000-ono --epochs 100
python train_unet_split.py --root onomatopoeia_datasets/syn2000/syn2000 --dataset syn2000-ono --epochs 100

# 実験2: 配置方法
python train_unet_split.py --root onomatopoeia_datasets/syn1000/syn1000 --dataset syn1000-ono-random --epochs 100
python train_unet_split.py --root onomatopoeia_datasets/syn1000_panel/syn1000_panel --dataset syn1000-ono-panel --epochs 100

# 評価
python test_unet.py --model-tag syn200-ono-unet-01 --data-root real_onomatopoeia_test100
python test_unet.py --model-tag syn500-ono-unet-01 --data-root real_onomatopoeia_test100
python test_unet.py --model-tag syn1000-ono-unet-01 --data-root real_onomatopoeia_test100
python test_unet.py --model-tag syn2000-ono-unet-01 --data-root real_onomatopoeia_test100
```

### Phase 3: 実験3-4実行（4-5日）

```powershell
# 実験3: データ拡張
python train_unet_split.py --root onomatopoeia_datasets/syn1000_panel/syn1000_panel --dataset syn1000-ono-panel --epochs 100
python train_unet_split.py --root onomatopoeia_datasets/syn1000_panel_aug/syn1000_panel_aug --dataset syn1000-ono-panel-aug --epochs 100
python train_unet_split.py --root onomatopoeia_datasets/syn500_panel/syn500_panel --dataset syn500-ono-panel --epochs 100
python train_unet_split.py --root onomatopoeia_datasets/syn500_panel_aug/syn500_panel_aug --dataset syn500-ono-panel-aug --epochs 100

# 実験4: モデル比較
python train_unet_split.py --root onomatopoeia_datasets/syn1000_panel_aug/syn1000_panel_aug --dataset syn1000-ono-unet --epochs 100
python train_resnet_unet.py --root onomatopoeia_datasets/syn1000_panel_aug/syn1000_panel_aug --dataset syn1000-ono-resnet --epochs 100
python train_deeplabv3plus.py --root onomatopoeia_datasets/syn1000_panel_aug/syn1000_panel_aug --dataset syn1000-ono-deeplabv3 --epochs 100

# 評価
python test_unet.py --model-tag syn1000-ono-panel-aug-unet-01 --data-root real_onomatopoeia_test200
python test_unet.py --model-tag syn1000-ono-resnet-01 --data-root real_onomatopoeia_test200
python test_unet.py --model-tag syn1000-ono-deeplabv3-01 --data-root real_onomatopoeia_test200
```

### Phase 4: 実験5実行（3-4日）

```powershell
# 最良設定での大規模学習
python train_unet_split.py --root onomatopoeia_datasets/syn2000_panel_aug/syn2000_panel_aug --dataset syn2000-ono-best --epochs 150 --batch 8
python train_unet_split_lowmem.py --root onomatopoeia_datasets/syn2000_panel_aug/syn2000_panel_aug --dataset syn2000-ono-best-lowmem --epochs 150 --batch 4

# 最終評価
python test_unet.py --model-tag syn2000-ono-best-unet-01 --data-root real_onomatopoeia_test200 --save-pred-n 50

# オーバーレイ可視化
python overlay_masks.py --model syn2000-ono-best-unet-01
```

### Phase 5: 分析とレポート作成（2-3日）

---

## 5. 期待される結果

### 仮説

1. **データ量**: syn1000以上で性能飽和、syn2000で最良
2. **配置方法**: パネル内配置がランダム配置より5-10% IoU向上
3. **データ拡張**: IoU 3-7%向上、特に回転・せん断が効果的
4. **モデル**: U-NetとDeepLabV3+が同程度、SegFormerがやや劣る（小さい物体）
5. **実画像**: 合成データで学習したモデルがIoU 0.70以上を達成

### 成功基準

- **最低基準**: IoU > 0.65, Dice > 0.75（実画像テストセット）
- **目標基準**: IoU > 0.75, Dice > 0.85（実画像テストセット）
- **優秀基準**: IoU > 0.80, Dice > 0.90（実画像テストセット）

---

## 6. リソース見積もり

### 計算リソース
- **GPU**: RTX 3090 または同等（VRAM 24GB推奨）
- **学習時間**: 1実験あたり2-6時間（データセット規模による）
- **総計算時間**: 約100-150 GPU時間

### ストレージ
- **データセット**: 約30-50GB
- **モデルチェックポイント**: 約20GB
- **実験結果**: 約10-20GB
- **合計**: 約60-90GB

### 作業時間
- **データセット作成**: 1-2日
- **実験実行**: 10-15日
- **分析・可視化**: 2-3日
- **レポート作成**: 2-3日
- **合計**: 約15-23日

---

## 7. 比較: 吹き出し vs オノマトペ

| 項目 | 吹き出し | オノマトペ | 主な違い |
|-----|---------|----------|---------|
| **サイズ** | 大きい（画像の10-30%） | 小さい（画像の0.1-1%） | オノマトペは検出が困難 |
| **形状** | 楕円・矩形が多い | 不規則、文字形状 | オノマトペは多様 |
| **配置** | コマの中心付近 | コマ内のどこでも | オノマトペは位置予測困難 |
| **回転** | ほぼ水平 | 斜め配置が多い | 回転データ拡張が重要 |
| **密度** | 1-3個/コマ | 0-10個/コマ | オノマトペは個数が不定 |
| **重なり** | 少ない | 頻繁 | オノマトペは分離が困難 |
| **期待性能** | IoU 0.85+ | IoU 0.70-0.80 | オノマトペは難易度高 |

### 対応策

**オノマトペ特有の課題への対応**:
1. 小さいオブジェクト → より細かい特徴抽出、アンカーフリー手法
2. 斜め配置 → 回転データ拡張（±30度）
3. 多様な形状 → データ拡張（せん断、アスペクト比）
4. 重なり → NMS (Non-Maximum Suppression) の調整
5. 個数不定 → 個数を予測しない密度マップ方式

---

## 8. 次のステップ（実験完了後）

1. **論文執筆**: 手法と結果をまとめる
2. **モデル公開**: Hugging Face / GitHub
3. **応用研究**:
   - 吹き出し + オノマトペの統合検出
   - テキスト認識との統合（OCR）
   - 漫画翻訳パイプラインへの組み込み
4. **改善方向**:
   - 境界の精度向上（Boundary-aware loss）
   - 小さいオブジェクトの検出率向上
   - 推論速度の最適化

---

## 9. チェックリスト

### データセット作成
- [ ] syn200_onomatopoeia
- [ ] syn500_onomatopoeia
- [ ] syn1000_onomatopoeia
- [ ] syn2000_onomatopoeia
- [ ] syn500_panel
- [ ] syn1000_panel
- [ ] syn500_panel_aug
- [ ] syn1000_panel_aug
- [ ] syn2000_panel_aug
- [ ] real_onomatopoeia_test100
- [ ] real_onomatopoeia_test200

### 実験実行
- [ ] 実験1: データセット規模（E1-1 ~ E1-4）
- [ ] 実験2: 配置方法（E2-1 ~ E2-2）
- [ ] 実験3: データ拡張（E3-1 ~ E3-4）
- [ ] 実験4: モデル比較（E4-1 ~ E4-4）
- [ ] 実験5: 大規模学習（E5-1 ~ E5-2）

### 評価・分析
- [ ] 定量評価（IoU, Dice, F1）
- [ ] 定性評価（可視化、失敗ケース分析）
- [ ] 比較表作成
- [ ] グラフ・図の作成

### ドキュメント
- [ ] 実験ログの整理
- [ ] 結果レポート作成
- [ ] README更新
- [ ] モデルカード作成（最良モデル）

---

## 10. 連絡先・参考資料

**関連ドキュメント**:
- `EXPERIMENT_AUTOMATION.md`: 自動実験実行スクリプト
- `MEMORY_OPTIMIZATION.md`: メモリ最適化手法
- `onomatopoeia_statistics.txt`: オノマトペ統計情報

**参考実験**:
- 吹き出しセグメンテーション実験結果
- balloon_experiment_results/ ディレクトリ

**wandbプロジェクト**:
- プロジェクト名: `balloon-seg-onomatopoeia`（仮）
- 吹き出し実験との比較のため同じプロジェクトを使用可能
