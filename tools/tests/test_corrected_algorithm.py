#!/usr/bin/env python3
"""
修正されたアルゴリズムのテストスクリプト
実データ統計に準拠したサイズ生成を検証する
"""

import sys
import os
sys.path.append('.')

from create_syn_dataset import calculate_area_based_size, sample_scale
import numpy as np
import random

def test_corrected_algorithm():
    print("🔬 修正されたアルゴリズムのテスト")
    print("=" * 60)
    
    # 修正されたCFG設定（実データ統計ベース - 調整版）
    CORRECTED_CFG = {
        "SCALE_MODE": "lognormal",           
        "SCALE_MEAN": 0.009200,              # 実データ平均を少し上げて調整
        "SCALE_STD": 0.008000,               # 標準偏差を増やして分散を実データに近づける
        "SCALE_CLIP": (0.002000, 0.020000), # クリッピング範囲を広げて自然な分散を許容
        "SCALE_RANGE": (0.002000, 0.020000),   # より広い範囲で自然な分散を確保
    }
    
    # 実際の統計値（基準値）
    REAL_STATS = {
        "mean": 0.008769,
        "median": 0.007226,
        "std": 0.006773,
        "p25": 0.004381,
        "p75": 0.011281
    }
    
    # 背景サイズとクロップサイズの設定
    bg_w, bg_h = 1536, 1024
    bg_area = bg_w * bg_h
    crop_w, crop_h = 300, 200  # 典型的なクロップサイズ
    
    print(f"📐 テスト設定:")
    print(f"  背景サイズ: {bg_w}×{bg_h} ({bg_area:,} pixels²)")
    print(f"  クロップサイズ: {crop_w}×{crop_h}")
    print(f"  実データ平均面積比: {REAL_STATS['mean']*100:.3f}%")
    print(f"  実データ範囲: {REAL_STATS['p25']*100:.3f}%-{REAL_STATS['p75']*100:.3f}%")
    print()
    
    # 統計収集
    sampled_scales = []
    final_sizes = []
    area_ratios = []
    
    # サンプリングテスト
    num_samples = 1000
    print(f"🎲 {num_samples}回のサンプリングテスト...")
    
    for i in range(num_samples):
        # スケールサンプリング
        scale = sample_scale(bg_w, crop_w, CORRECTED_CFG)
        sampled_scales.append(scale)
        
        # 面積ベースサイズ計算
        new_w, new_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, scale,
            max_w_ratio=0.20, max_h_ratio=0.30
        )
        
        final_sizes.append((new_w, new_h))
        
        # 実際の面積比計算
        actual_area = new_w * new_h
        area_ratio = actual_area / bg_area
        area_ratios.append(area_ratio)
    
    # 統計分析
    scale_stats = {
        "mean": np.mean(sampled_scales),
        "median": np.median(sampled_scales),
        "std": np.std(sampled_scales),
        "p25": np.percentile(sampled_scales, 25),
        "p75": np.percentile(sampled_scales, 75),
        "min": np.min(sampled_scales),
        "max": np.max(sampled_scales)
    }
    
    area_stats = {
        "mean": np.mean(area_ratios),
        "median": np.median(area_ratios),
        "std": np.std(area_ratios),
        "p25": np.percentile(area_ratios, 25),
        "p75": np.percentile(area_ratios, 75),
        "min": np.min(area_ratios),
        "max": np.max(area_ratios)
    }
    
    # 結果表示
    print("📊 サンプリング結果:")
    print("-" * 40)
    print("スケール値（サンプリング）:")
    print(f"  平均: {scale_stats['mean']:.6f} (設定: {CORRECTED_CFG['SCALE_MEAN']:.6f})")
    print(f"  中央値: {scale_stats['median']:.6f}")
    print(f"  標準偏差: {scale_stats['std']:.6f} (設定: {CORRECTED_CFG['SCALE_STD']:.6f})")
    print(f"  25-75%: {scale_stats['p25']:.6f}-{scale_stats['p75']:.6f}")
    print(f"  範囲: {scale_stats['min']:.6f}-{scale_stats['max']:.6f}")
    print()
    
    print("実際の面積比（最終結果）:")
    print(f"  平均: {area_stats['mean']:.6f} ({area_stats['mean']*100:.3f}%)")
    print(f"  中央値: {area_stats['median']:.6f} ({area_stats['median']*100:.3f}%)")
    print(f"  標準偏差: {area_stats['std']:.6f}")
    print(f"  25-75%: {area_stats['p25']:.6f}-{area_stats['p75']:.6f} ({area_stats['p25']*100:.3f}%-{area_stats['p75']*100:.3f}%)")
    print(f"  範囲: {area_stats['min']:.6f}-{area_stats['max']:.6f}")
    print()
    
    # 実データとの比較
    print("🎯 実データとの比較:")
    print("-" * 40)
    
    def compare_stat(name, actual, expected, tolerance=0.1):
        diff = abs(actual - expected)
        ratio = actual / expected if expected > 0 else float('inf')
        tolerance_range = expected * tolerance
        
        if diff <= tolerance_range:
            status = "✅ 良好"
        elif diff <= tolerance_range * 2:
            status = "⚠️ やや差あり"
        else:
            status = "❌ 大きな差"
            
        print(f"  {name}: {actual:.6f} vs {expected:.6f} (差: {diff:.6f}, 比: {ratio:.2f}) {status}")
    
    compare_stat("平均面積比", area_stats['mean'], REAL_STATS['mean'])
    compare_stat("中央値面積比", area_stats['median'], REAL_STATS['median'])
    compare_stat("標準偏差", area_stats['std'], REAL_STATS['std'])
    compare_stat("25パーセンタイル", area_stats['p25'], REAL_STATS['p25'])
    compare_stat("75パーセンタイル", area_stats['p75'], REAL_STATS['p75'])
    
    print()
    
    # サンプル生成結果
    print("📏 サンプル生成結果（最初の10個）:")
    print("-" * 40)
    for i in range(min(10, len(final_sizes))):
        w, h = final_sizes[i]
        scale = sampled_scales[i]
        area_ratio = area_ratios[i]
        print(f"  {i+1:2d}: {w:3d}×{h:3d}px (面積比: {area_ratio*100:.3f}%, スケール: {scale:.6f})")
    
    print()
    
    # 総合評価
    print("🏆 総合評価:")
    print("-" * 40)
    
    mean_match = abs(area_stats['mean'] - REAL_STATS['mean']) <= REAL_STATS['mean'] * 0.1
    median_match = abs(area_stats['median'] - REAL_STATS['median']) <= REAL_STATS['median'] * 0.1
    std_match = abs(area_stats['std'] - REAL_STATS['std']) <= REAL_STATS['std'] * 0.2
    
    if mean_match and median_match and std_match:
        print("✅ アルゴリズム修正成功！実データ統計と良好な一致")
    elif mean_match and median_match:
        print("⚠️ 中心値は良好、分散の調整が必要")
    else:
        print("❌ さらなる調整が必要")
    
    # 旧設定との比較参考値
    old_mean_area = (0.105 ** 2)  # 旧設定での平均面積比
    improvement = abs(area_stats['mean'] - REAL_STATS['mean']) / abs(old_mean_area - REAL_STATS['mean'])
    print(f"🔄 改善度: {(1-improvement)*100:.1f}% (旧設定比)")

if __name__ == "__main__":
    test_corrected_algorithm()
