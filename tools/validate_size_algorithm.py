#!/usr/bin/env python3
"""
吹き出しサイズ決定アルゴリズムの検証ツール
統計的精度を継続的にモニタリング
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from create_syn_dataset import sample_scale, calculate_area_based_size

def test_size_algorithm(cfg=None, num_tests=1000, bg_size=(1536, 1024), crop_size=(300, 200)):
    """
    サイズ決定アルゴリズムの統計的精度をテスト
    """
    print("🔬 修正されたアルゴリズムのテスト")
    print("=" * 60)
    
    if cfg is None:
        # デフォルト設定（最新版）
        cfg = {
            "SCALE_MODE": "lognormal",
            "SCALE_MEAN": 0.009200,
            "SCALE_STD": 0.012000,
            "SCALE_CLIP": (0.001200, 0.028000),
        }
    
    bg_w, bg_h = bg_size
    crop_w, crop_h = crop_size
    bg_area = bg_w * bg_h
    
    print(f"📐 テスト設定:")
    print(f"  背景サイズ: {bg_w}×{bg_h} ({bg_area:,} pixels²)")
    print(f"  クロップサイズ: {crop_w}×{crop_h}")
    print(f"  実データ平均面積比: 0.877%")
    print(f"  実データ範囲: 0.438%-1.128%")
    
    # サンプリングテスト
    print(f"\n🎲 {num_tests}回のサンプリングテスト...")
    
    sampled_scales = []
    actual_ratios = []
    sample_results = []
    
    for i in range(num_tests):
        # スケールサンプリング
        scale = sample_scale(bg_w, crop_w, cfg)
        sampled_scales.append(scale)
        
        # 実際のサイズ計算
        new_w, new_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, scale,
            max_w_ratio=0.20, max_h_ratio=0.30
        )
        
        # 実際の面積比を計算
        actual_area = new_w * new_h
        actual_ratio = actual_area / bg_area
        actual_ratios.append(actual_ratio)
        
        # サンプル結果を記録（最初の10個）
        if i < 10:
            sample_results.append({
                'index': i + 1,
                'width': new_w,
                'height': new_h,
                'area_ratio': actual_ratio,
                'scale': scale
            })
    
    # 統計計算
    sampled_scales = np.array(sampled_scales)
    actual_ratios = np.array(actual_ratios)
    
    print(f"📊 サンプリング結果:")
    print("-" * 40)
    print(f"スケール値（サンプリング）:")
    print(f"  平均: {sampled_scales.mean():.6f} (設定: {cfg['SCALE_MEAN']:.6f})")
    print(f"  中央値: {np.median(sampled_scales):.6f}")
    print(f"  標準偏差: {sampled_scales.std():.6f} (設定: {cfg['SCALE_STD']:.6f})")
    print(f"  25-75%: {np.percentile(sampled_scales, 25):.6f}-{np.percentile(sampled_scales, 75):.6f}")
    print(f"  範囲: {sampled_scales.min():.6f}-{sampled_scales.max():.6f}")
    
    print(f"\n実際の面積比（最終結果）:")
    print(f"  平均: {actual_ratios.mean():.6f} ({actual_ratios.mean()*100:.3f}%)")
    print(f"  中央値: {np.median(actual_ratios):.6f} ({np.median(actual_ratios)*100:.3f}%)")
    print(f"  標準偏差: {actual_ratios.std():.6f}")
    print(f"  25-75%: {np.percentile(actual_ratios, 25):.6f}-{np.percentile(actual_ratios, 75):.6f} ({np.percentile(actual_ratios, 25)*100:.3f}%-{np.percentile(actual_ratios, 75)*100:.3f}%)")
    print(f"  範囲: {actual_ratios.min():.6f}-{actual_ratios.max():.6f}")
    
    # 実データとの比較
    real_data = {
        'mean': 0.008769,
        'median': 0.007226,
        'std': 0.006773,
        'p25': 0.004381,
        'p75': 0.011281
    }
    
    print(f"\n🎯 実データとの比較:")
    print("-" * 40)
    
    mean_diff = abs(actual_ratios.mean() - real_data['mean'])
    mean_ratio = actual_ratios.mean() / real_data['mean']
    median_diff = abs(np.median(actual_ratios) - real_data['median'])
    median_ratio = np.median(actual_ratios) / real_data['median']
    std_diff = abs(actual_ratios.std() - real_data['std'])
    std_ratio = actual_ratios.std() / real_data['std']
    p25_diff = abs(np.percentile(actual_ratios, 25) - real_data['p25'])
    p25_ratio = np.percentile(actual_ratios, 25) / real_data['p25']
    p75_diff = abs(np.percentile(actual_ratios, 75) - real_data['p75'])
    p75_ratio = np.percentile(actual_ratios, 75) / real_data['p75']
    
    def evaluate_metric(ratio, diff, threshold_ratio=0.05, threshold_diff=0.001):
        if abs(ratio - 1.0) <= threshold_ratio and diff <= threshold_diff:
            return "✅ 良好"
        elif abs(ratio - 1.0) <= 0.15:
            return "⚠️ 改善の余地"
        else:
            return "❌ 大きな差"
    
    print(f"  平均面積比: {actual_ratios.mean():.6f} vs {real_data['mean']:.6f} (差: {mean_diff:.6f}, 比: {mean_ratio:.2f}) {evaluate_metric(mean_ratio, mean_diff)}")
    print(f"  中央値面積比: {np.median(actual_ratios):.6f} vs {real_data['median']:.6f} (差: {median_diff:.6f}, 比: {median_ratio:.2f}) {evaluate_metric(median_ratio, median_diff)}")
    print(f"  標準偏差: {actual_ratios.std():.6f} vs {real_data['std']:.6f} (差: {std_diff:.6f}, 比: {std_ratio:.2f}) {evaluate_metric(std_ratio, std_diff, 0.15, 0.002)}")
    print(f"  25パーセンタイル: {np.percentile(actual_ratios, 25):.6f} vs {real_data['p25']:.6f} (差: {p25_diff:.6f}, 比: {p25_ratio:.2f}) {evaluate_metric(p25_ratio, p25_diff)}")
    print(f"  75パーセンタイル: {np.percentile(actual_ratios, 75):.6f} vs {real_data['p75']:.6f} (差: {p75_diff:.6f}, 比: {p75_ratio:.2f}) {evaluate_metric(p75_ratio, p75_diff)}")
    
    # サンプル結果表示
    print(f"\n📏 サンプル生成結果（最初の10個）:")
    print("-" * 40)
    for result in sample_results:
        print(f"  {result['index']:2d}: {result['width']:3d}×{result['height']:3d}px (面積比: {result['area_ratio']*100:.3f}%, スケール: {result['scale']:.6f})")
    
    # 総合評価
    print(f"\n🏆 総合評価:")
    print("-" * 40)
    
    # 精度スコア計算
    accuracy_scores = []
    accuracy_scores.append(min(1.0, 1.0 - abs(mean_ratio - 1.0)))
    accuracy_scores.append(min(1.0, 1.0 - abs(median_ratio - 1.0)))
    accuracy_scores.append(min(1.0, 1.0 - abs(std_ratio - 1.0)))
    accuracy_scores.append(min(1.0, 1.0 - abs(p25_ratio - 1.0)))
    accuracy_scores.append(min(1.0, 1.0 - abs(p75_ratio - 1.0)))
    
    overall_accuracy = np.mean(accuracy_scores)
    
    if overall_accuracy >= 0.95:
        status = "🎯 優秀"
    elif overall_accuracy >= 0.85:
        status = "✅ 良好"
    elif overall_accuracy >= 0.70:
        status = "⚠️ 改善の余地"
    else:
        status = "❌ 要改善"
    
    print(f"{status}")
    print(f"🔄 全体精度: {overall_accuracy*100:.1f}%")
    
    return {
        'overall_accuracy': overall_accuracy,
        'metrics': {
            'mean_ratio': mean_ratio,
            'median_ratio': median_ratio,
            'std_ratio': std_ratio,
            'p25_ratio': p25_ratio,
            'p75_ratio': p75_ratio
        },
        'actual_stats': {
            'mean': actual_ratios.mean(),
            'median': np.median(actual_ratios),
            'std': actual_ratios.std(),
            'p25': np.percentile(actual_ratios, 25),
            'p75': np.percentile(actual_ratios, 75)
        }
    }

if __name__ == "__main__":
    # 最新設定でテスト実行
    test_size_algorithm()
