#!/usr/bin/env python3
"""
現在のcreate_syn_dataset.pyのサイズ決定方法をテストして検証
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 現在のディレクトリをパスに追加
sys.path.insert(0, '/Users/x20047xx/研究室/manga/t2i_balloon_gen')

def test_current_algorithm():
    """現在のアルゴリズムの動作をテスト"""
    print("🧪 現在のcreate_syn_dataset.pyのサイズ決定アルゴリズムテスト")
    print("=" * 70)
    
    try:
        # モジュールをインポート
        from create_syn_dataset import calculate_area_based_size, sample_scale
        print("✅ モジュールインポート成功")
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return
    
    # 現在のCFG設定を確認
    try:
        from create_syn_dataset import CFG
        print("\n📋 現在のCFG設定:")
        print("-" * 40)
        relevant_keys = ['SCALE_MODE', 'SCALE_MEAN', 'SCALE_STD', 'SCALE_CLIP', 'SCALE_RANGE']
        for key in relevant_keys:
            if key in CFG:
                value = CFG[key]
                if key in ['SCALE_MEAN', 'SCALE_STD']:
                    print(f"  {key}: {value:.6f} ({value*100:.3f}%)")
                else:
                    print(f"  {key}: {value}")
        print()
    except Exception as e:
        print(f"⚠️ CFG読み込みエラー: {e}")
        # デフォルト設定を使用
        CFG = {
            'SCALE_MODE': 'lognormal',
            'SCALE_MEAN': 0.009200,
            'SCALE_STD': 0.008000,
            'SCALE_CLIP': (0.002000, 0.020000),
            'SCALE_RANGE': (0.002000, 0.020000)
        }
        print("デフォルト設定を使用")
    
    # テスト設定
    bg_w, bg_h = 1536, 1024
    bg_area = bg_w * bg_h
    crop_w, crop_h = 300, 200
    
    print(f"📐 テスト環境:")
    print(f"  背景サイズ: {bg_w}×{bg_h} ({bg_area:,} pixels²)")
    print(f"  吹き出しクロップサイズ: {crop_w}×{crop_h}")
    print()
    
    # 実データ統計（参考値）
    REAL_STATS = {
        "mean": 0.008769,     # 0.877%
        "median": 0.007226,   # 0.723%
        "std": 0.006773,      # 0.677%
        "p25": 0.004381,      # 0.438%
        "p75": 0.011281       # 1.128%
    }
    
    print(f"📊 実データ統計（参考値）:")
    print(f"  平均面積比: {REAL_STATS['mean']*100:.3f}%")
    print(f"  中央値面積比: {REAL_STATS['median']*100:.3f}%")
    print(f"  標準偏差: {REAL_STATS['std']*100:.3f}%")
    print(f"  25-75%範囲: {REAL_STATS['p25']*100:.3f}%-{REAL_STATS['p75']*100:.3f}%")
    print()
    
    # 1. スケールサンプリングテスト
    print("🎲 1. スケールサンプリングテスト（1000回）")
    print("-" * 50)
    
    sampled_scales = []
    for _ in range(1000):
        scale = sample_scale(bg_w, crop_w, CFG)
        sampled_scales.append(scale)
    
    # 統計計算
    scale_stats = {
        "mean": np.mean(sampled_scales),
        "median": np.median(sampled_scales),
        "std": np.std(sampled_scales),
        "p25": np.percentile(sampled_scales, 25),
        "p75": np.percentile(sampled_scales, 75),
        "min": np.min(sampled_scales),
        "max": np.max(sampled_scales)
    }
    
    print(f"サンプル結果:")
    print(f"  平均: {scale_stats['mean']:.6f} ({scale_stats['mean']*100:.3f}%)")
    print(f"  中央値: {scale_stats['median']:.6f} ({scale_stats['median']*100:.3f}%)")
    print(f"  標準偏差: {scale_stats['std']:.6f} ({scale_stats['std']*100:.3f}%)")
    print(f"  25-75%範囲: {scale_stats['p25']:.6f}-{scale_stats['p75']:.6f}")
    print(f"  最小-最大: {scale_stats['min']:.6f}-{scale_stats['max']:.6f}")
    print()
    
    # CFG設定との比較
    cfg_mean = CFG.get('SCALE_MEAN', 0)
    cfg_std = CFG.get('SCALE_STD', 0)
    print(f"CFG設定との比較:")
    print(f"  平均 → 設定: {cfg_mean:.6f}, 実測: {scale_stats['mean']:.6f}, 差: {abs(cfg_mean - scale_stats['mean']):.6f}")
    print(f"  標準偏差 → 設定: {cfg_std:.6f}, 実測: {scale_stats['std']:.6f}, 差: {abs(cfg_std - scale_stats['std']):.6f}")
    print()
    
    # 2. 面積ベースサイズ計算テスト
    print("📏 2. 面積ベースサイズ計算テスト")
    print("-" * 50)
    
    test_scales = [0.005, 0.008769, 0.012, 0.015, 0.020]
    final_sizes = []
    area_ratios = []
    
    print("スケール値 | 期待面積比(%) | 生成サイズ    | 実際面積比(%) | 誤差(%)")
    print("-" * 70)
    
    for scale in test_scales:
        # 面積ベースサイズ計算
        new_w, new_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, scale,
            max_w_ratio=0.20, max_h_ratio=0.30
        )
        
        # 実際の面積比計算
        actual_area = new_w * new_h
        actual_ratio = actual_area / bg_area
        error = abs(actual_ratio - scale) / scale * 100 if scale > 0 else 0
        
        final_sizes.append((new_w, new_h))
        area_ratios.append(actual_ratio)
        
        print(f"{scale:8.6f} | {scale*100:11.3f} | {new_w:3d}×{new_h:3d}px | {actual_ratio*100:11.3f} | {error:7.1f}")
    
    print()
    
    # 3. 統計的サンプリングテスト
    print("📈 3. 統計的サンプリングテスト（500回の完全フロー）")
    print("-" * 60)
    
    complete_test_results = []
    for _ in range(500):
        # スケールサンプリング
        scale = sample_scale(bg_w, crop_w, CFG)
        
        # サイズ計算
        new_w, new_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, scale,
            max_w_ratio=0.20, max_h_ratio=0.30
        )
        
        # 実際の面積比
        actual_area = new_w * new_h
        actual_ratio = actual_area / bg_area
        
        complete_test_results.append({
            'sampled_scale': scale,
            'final_width': new_w,
            'final_height': new_h,
            'final_area_ratio': actual_ratio
        })
    
    # 最終結果の統計
    final_area_ratios = [r['final_area_ratio'] for r in complete_test_results]
    final_stats = {
        "mean": np.mean(final_area_ratios),
        "median": np.median(final_area_ratios),
        "std": np.std(final_area_ratios),
        "p25": np.percentile(final_area_ratios, 25),
        "p75": np.percentile(final_area_ratios, 75),
        "min": np.min(final_area_ratios),
        "max": np.max(final_area_ratios)
    }
    
    print(f"最終的な面積比統計:")
    print(f"  平均: {final_stats['mean']:.6f} ({final_stats['mean']*100:.3f}%)")
    print(f"  中央値: {final_stats['median']:.6f} ({final_stats['median']*100:.3f}%)")
    print(f"  標準偏差: {final_stats['std']:.6f} ({final_stats['std']*100:.3f}%)")
    print(f"  25-75%範囲: {final_stats['p25']:.6f}-{final_stats['p75']:.6f}")
    print(f"  最小-最大: {final_stats['min']:.6f}-{final_stats['max']:.6f}")
    print()
    
    # 4. 実データとの適合性評価
    print("🎯 4. 実データとの適合性評価")
    print("-" * 50)
    
    def evaluate_match(actual, expected, tolerance=0.1):
        """統計値の一致度を評価"""
        diff = abs(actual - expected)
        ratio = actual / expected if expected > 0 else float('inf')
        
        if diff <= expected * tolerance:
            return "✅ 良好"
        elif diff <= expected * tolerance * 2:
            return "⚠️ やや差あり"
        else:
            return "❌ 大きな差"
    
    evaluations = []
    metrics = [
        ("平均", final_stats['mean'], REAL_STATS['mean']),
        ("中央値", final_stats['median'], REAL_STATS['median']),
        ("標準偏差", final_stats['std'], REAL_STATS['std']),
        ("25%ile", final_stats['p25'], REAL_STATS['p25']),
        ("75%ile", final_stats['p75'], REAL_STATS['p75'])
    ]
    
    for name, actual, expected in metrics:
        evaluation = evaluate_match(actual, expected)
        ratio = actual / expected if expected > 0 else float('inf')
        diff_percent = abs(actual - expected) / expected * 100
        
        print(f"  {name}: {actual:.6f} vs {expected:.6f} (比: {ratio:.2f}, 差: {diff_percent:.1f}%) {evaluation}")
        evaluations.append(evaluation)
    
    print()
    
    # 5. 総合評価
    print("🏆 5. 総合評価")
    print("-" * 30)
    
    good_count = evaluations.count("✅ 良好")
    ok_count = evaluations.count("⚠️ やや差あり")
    bad_count = evaluations.count("❌ 大きな差")
    
    print(f"良好: {good_count}/5, やや差: {ok_count}/5, 大きな差: {bad_count}/5")
    
    if good_count >= 4:
        print("✅ アルゴリズムは実データ統計と良好に一致しています")
        grade = "A"
    elif good_count >= 3:
        print("⚠️ 概ね良好ですが、一部調整の余地があります")
        grade = "B"
    elif good_count >= 2:
        print("⚠️ 部分的な改善が必要です")
        grade = "C"
    else:
        print("❌ 大幅な調整が必要です")
        grade = "D"
    
    print(f"総合評価: {grade}")
    print()
    
    # 6. サンプル出力例
    print("📋 6. サンプル出力例（最初の10個）")
    print("-" * 50)
    print("No. | サンプルスケール | 生成サイズ    | 面積比(%) | 実データ適合")
    print("-" * 60)
    
    for i in range(min(10, len(complete_test_results))):
        result = complete_test_results[i]
        scale = result['sampled_scale']
        w, h = result['final_width'], result['final_height']
        ratio = result['final_area_ratio']
        
        # 実データ範囲内かチェック
        in_range = REAL_STATS['p25'] <= ratio <= REAL_STATS['p75']
        status = "✅" if in_range else "⚠️"
        
        print(f"{i+1:3d} | {scale:14.6f} | {w:3d}×{h:3d}px | {ratio*100:7.3f} | {status}")
    
    print()
    
    # 7. 修正効果の確認
    print("🔄 7. 修正効果の確認")
    print("-" * 40)
    
    # 旧設定での理論値
    old_scale_mean = 0.105  # 旧設定の幅比平均
    old_area_mean = old_scale_mean ** 2  # 旧方式での面積比
    
    current_area_mean = final_stats['mean']
    real_area_mean = REAL_STATS['mean']
    
    old_error = abs(old_area_mean - real_area_mean) / real_area_mean * 100
    current_error = abs(current_area_mean - real_area_mean) / real_area_mean * 100
    improvement = (old_error - current_error) / old_error * 100 if old_error > 0 else 0
    
    print(f"旧方式での面積比: {old_area_mean*100:.3f}% (誤差: {old_error:.1f}%)")
    print(f"現在の面積比: {current_area_mean*100:.3f}% (誤差: {current_error:.1f}%)")
    print(f"改善度: {improvement:.1f}%")
    
    if improvement > 50:
        print("✅ 大幅な改善が確認されました")
    elif improvement > 20:
        print("⚠️ 改善が確認されましたが、さらなる調整の余地があります")
    else:
        print("❌ 十分な改善が見られません")

if __name__ == "__main__":
    test_current_algorithm()
