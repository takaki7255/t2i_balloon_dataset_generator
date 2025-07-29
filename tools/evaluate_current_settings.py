"""
balloon_count_statisticsの検証と現在設定の評価
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# 統計情報
print("実際のマンガデータ統計:")
print("==================================================")
print("Total images analyzed: 10619")
print("Total balloon annotations: 130180")
print("Images with balloons: 9916")
print("Images without balloons: 703")
print()
print("Balloon Count Statistics:")
print("Mean: 12.259158")
print("Median: 12.000000")
print("Standard deviation: 6.980222")
print("Min: 0")
print("Max: 44")
print("25th percentile: 7.00")
print("75th percentile: 17.00")
print()

# 現在の設定
current_cfg = {
    "NUM_BALLOONS_RANGE": (2, 5),       # 実際の分布は0-44個、平均12.26個
    "SCALE_MEAN": 0.10,                  # 背景幅の10%
    "SCALE_STD": 0.03,                   
    "SCALE_CLIP": (0.05, 0.18),         # 5%-18%の範囲
}

print("現在の設定:")
print("==================================================")
print(f"NUM_BALLOONS_RANGE: {current_cfg['NUM_BALLOONS_RANGE']}")
print(f"  → 実際の統計: 平均12.26個、中央値12個、範囲0-44個")
print(f"  → 現在の設定は実際より大幅に少ない")
print()
print(f"SCALE_MEAN: {current_cfg['SCALE_MEAN']} (背景幅の10%)")
print(f"SCALE_CLIP: {current_cfg['SCALE_CLIP']} (背景幅の5%-18%)")
print()

# 改善提案
print("改善提案:")
print("==================================================")
print("1. 吹き出し個数の範囲を統計により近づける:")
print("   現在: (2, 5)")
print("   提案: (5, 15) または (7, 17) # 25th-75th percentile")
print("   または統計分布を直接使用 (COUNT_PROBS)")
print()

print("2. スケール設定の検証:")
print("   現在のスケール平均: 10% (改善済み)")
print("   → 前回の25%から大幅改善されている")
print("   → 7.2%-12.4%の範囲になったのは良い結果")
print()

print("3. 統計分布の活用:")
count_stats_percentages = {
    0: 6.6, 1: 1.3, 2: 1.9, 3: 2.1, 4: 2.7, 5: 3.0, 6: 3.5, 7: 4.4, 8: 4.6, 9: 5.0,
    10: 5.3, 11: 5.6, 12: 6.2, 13: 5.4, 14: 5.3, 15: 5.3, 16: 5.2, 17: 4.3, 18: 3.9, 19: 3.4,
    20: 2.8
}

print("   最頻値の範囲: 10-12個が最も多い (各5-6%)")
print("   7-17個の範囲が全体の約50%を占める")
print("   現在の(2,5)は低い確率帯のみを対象にしている")
