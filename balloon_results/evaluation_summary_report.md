# Balloon Segmentation モデル評価結果レポート

**作成日**: 2026年3月9日  
**テストデータセット**: balloon_dataset/test300_dataset (300枚)

## Global Metrics 比較表

| Model | IoU | F1 Score | Accuracy | Precision | Recall |
|-------|-----|----------|----------|-----------|--------|
| real200_dataset-resnet34-pretrained-unet-01 | **0.8808** | **0.9366** | **0.9850** | 0.9416 | 0.9317 |
| syn1000-corner-unet-02-real200_dataset-finetuned | 0.8762 | 0.9340 | 0.9846 | 0.9486 | 0.9200 |
| real200_dataset-unet-01 | 0.8701 | 0.9305 | 0.9838 | **0.9495** | 0.9123 |
| real200_dataset-deeplabv3plus-resnet50-pretrained-s16-01 | 0.8647 | 0.9274 | 0.9830 | 0.9430 | 0.9124 |
| real200_dataset-segformer-gray-01 | 0.8519 | 0.9200 | 0.9811 | 0.9260 | 0.9141 |
| syn1000-corner-unet-01 | 0.7800 | 0.8764 | 0.9701 | 0.8595 | 0.8939 |
| syn1000-bg75-unet-01 | 0.7631 | 0.8657 | 0.9663 | 0.8213 | 0.9151 |
| syn1000-b75-unet-01 | 0.7627 | 0.8654 | 0.9672 | 0.8432 | 0.8888 |
| syn1000-bg50-unet-01 | 0.7481 | 0.8559 | 0.9635 | 0.8064 | 0.9119 |
| syn1000-random-unet-01 | 0.7440 | 0.8532 | 0.9636 | 0.8178 | 0.8918 |
| syn1000-b50-unet-01 | 0.7220 | 0.8386 | 0.9636 | 0.8848 | 0.7969 |
| realballoons_synbacks_200-unet-01 | 0.7060 | 0.8277 | 0.9551 | 0.7598 | 0.9088 |
| syn200_dataset-unet-01 | 0.7056 | 0.8274 | 0.9588 | 0.8235 | 0.8314 |
| syn1000-bg25-unet-01 | 0.7051 | 0.8271 | 0.9541 | 0.7484 | **0.9242** |
| syn1000-b25-unet-01 | 0.6977 | 0.8219 | 0.9559 | 0.7904 | 0.8561 |
| synballoons_realbacks_200-unet-01 | 0.3740 | 0.5444 | 0.9221 | 0.8915 | 0.3918 |

## カテゴリ別分析

### 1. 実データ学習モデル (Real Data Models)
| Model | IoU | F1 | Accuracy |
|-------|-----|-----|----------|
| real200_dataset-resnet34-pretrained-unet-01 | 0.8808 | 0.9366 | 0.9850 |
| real200_dataset-unet-01 | 0.8701 | 0.9305 | 0.9838 |
| real200_dataset-deeplabv3plus-resnet50-pretrained-s16-01 | 0.8647 | 0.9274 | 0.9830 |
| real200_dataset-segformer-gray-01 | 0.8519 | 0.9200 | 0.9811 |

### 2. 合成データ学習モデル (Synthetic Data Models)
| Model | IoU | F1 | Accuracy |
|-------|-----|-----|----------|
| syn1000-corner-unet-01 | 0.7800 | 0.8764 | 0.9701 |
| syn1000-bg75-unet-01 | 0.7631 | 0.8657 | 0.9663 |
| syn1000-b75-unet-01 | 0.7627 | 0.8654 | 0.9672 |
| syn1000-bg50-unet-01 | 0.7481 | 0.8559 | 0.9635 |
| syn1000-random-unet-01 | 0.7440 | 0.8532 | 0.9636 |
| syn1000-b50-unet-01 | 0.7220 | 0.8386 | 0.9636 |
| syn1000-bg25-unet-01 | 0.7051 | 0.8271 | 0.9541 |
| syn1000-b25-unet-01 | 0.6977 | 0.8219 | 0.9559 |
| syn200_dataset-unet-01 | 0.7056 | 0.8274 | 0.9588 |

### 3. ファインチューニングモデル (Fine-tuned Models)
| Model | IoU | F1 | Accuracy |
|-------|-----|-----|----------|
| syn1000-corner-unet-02-real200_dataset-finetuned | 0.8762 | 0.9340 | 0.9846 |

### 4. 混合データセットモデル (Mixed Dataset Models)
| Model | IoU | F1 | Accuracy |
|-------|-----|-----|----------|
| realballoons_synbacks_200-unet-01 | 0.7060 | 0.8277 | 0.9551 |
| synballoons_realbacks_200-unet-01 | 0.3740 | 0.5444 | 0.9221 |

## 主な知見

### パフォーマンス順位 (IoU基準)
1. **real200_dataset-resnet34-pretrained-unet-01** (IoU: 0.8808) - 最高性能
2. **syn1000-corner-unet-02-real200_dataset-finetuned** (IoU: 0.8762) - ファインチューニング効果
3. **real200_dataset-unet-01** (IoU: 0.8701) - 基本UNet

### 重要な発見
- **実データ学習モデル**が全体的に最も高い性能を示す (IoU: 0.85-0.88)
- **合成データからのファインチューニング**は実データのみの学習に匹敵する性能を達成 (IoU: 0.8762)
- **合成データのみ**のモデルは実データモデルより約10-15%低いIoUを示す
- **corner配置**の合成データ (syn1000-corner) が**random配置** (syn1000-random) より優れた性能
- **背景ブレンド率**が高いほど性能向上 (bg25 < bg50 < bg75)
- **吹き出しブレンド率**も同様の傾向 (b25 < b50 < b75)
- **synballoons_realbacks_200**モデルは最も低い性能 (IoU: 0.3740) - 合成吹き出し+実背景の組み合わせが困難

---
*このレポートは各モデルのevaluation_results.jsonから自動生成されました*
