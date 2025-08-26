#!/usr/bin/env python3
"""
調整後のアルゴリズムをテスト
"""

import sys
import os
import numpy as np

# 現在のディレクトリをパスに追加
sys.path.insert(0, '/Users/x20047xx/研究室/manga/t2i_balloon_gen')

def test_adjusted_algorithm():
    """調整後のアルゴリズムをテスト"""
    print("🔬 調整後のアルゴリズムテスト")
    print("=" * 50)
    
    try:
        from create_syn_dataset import calculate_area_based_size, sample_scale
        print("✅ モジュールインポート成功")
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return
    
    # 調整されたCFG設定
    CFG = {
        'SCALE_MODE': 'lognormal',
        'SCALE_MEAN': 0.009400,
        'SCALE_STD': 0.009500,
        'SCALE_CLIP': (0.001500, 0.025000),
        'SCALE_RANGE': (0.001500, 0.025000)
    }
    
    print("📋 調整されたCFG設定:")
    print(f"  SCALE_MEAN: {CFG['SCALE_MEAN']:.6f} ({CFG['SCALE_MEAN']*100:.3f}%)")
    print(f"  SCALE_STD: {CFG['SCALE_STD']:.6f} ({CFG['SCALE_STD']*100:.3f}%)")
    print(f"  SCALE_CLIP: {CFG['SCALE_CLIP']}")
    print()
    
    # テスト設定
    bg_w, bg_h = 1536, 1024
    bg_area = bg_w * bg_h
    crop_w, crop_h = 300, 200
    
    # 実データ統計
    REAL_STATS = {
        "mean": 0.008769,
        "median": 0.007226,
        "std": 0.006773,
        "p25": 0.004381,
        "p75": 0.011281
    }
    
    print(f"🎲 500回の調整後テスト...")
    
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
        
        complete_test_results.append(actual_ratio)
    
    # 統計計算
    final_stats = {
        "mean": np.mean(complete_test_results),
        "median": np.median(complete_test_results),
        "std": np.std(complete_test_results),
        "p25": np.percentile(complete_test_results, 25),
        "p75": np.percentile(complete_test_results, 75),
        "min": np.min(complete_test_results),
        "max": np.max(complete_test_results)
    }
    
    print("📊 調整後の結果:")
    print(f"  平均: {final_stats['mean']:.6f} ({final_stats['mean']*100:.3f}%)")
    print(f"  中央値: {final_stats['median']:.6f} ({final_stats['median']*100:.3f}%)")
    print(f"  標準偏差: {final_stats['std']:.6f} ({final_stats['std']*100:.3f}%)")
    print(f"  25-75%範囲: {final_stats['p25']:.6f}-{final_stats['p75']:.6f}")
    print()
    
    # 実データとの比較
    print("🎯 実データとの比較:")
    
    def compare(name, actual, expected):
        diff_percent = abs(actual - expected) / expected * 100
        ratio = actual / expected
        
        if diff_percent <= 10:
            status = "✅ 良好"
        elif diff_percent <= 20:
            status = "⚠️ やや差"
        else:
            status = "❌ 大きな差"
            
        print(f"  {name}: {actual:.6f} vs {expected:.6f} (差: {diff_percent:.1f}%, 比: {ratio:.2f}) {status}")
        return status
    
    evaluations = []
    evaluations.append(compare("平均", final_stats['mean'], REAL_STATS['mean']))
    evaluations.append(compare("中央値", final_stats['median'], REAL_STATS['median']))
    evaluations.append(compare("標準偏差", final_stats['std'], REAL_STATS['std']))
    evaluations.append(compare("25%ile", final_stats['p25'], REAL_STATS['p25']))
    evaluations.append(compare("75%ile", final_stats['p75'], REAL_STATS['p75']))
    
    print()
    
    # 総合評価
    good_count = evaluations.count("✅ 良好")
    ok_count = evaluations.count("⚠️ やや差")
    bad_count = evaluations.count("❌ 大きな差")
    
    print("🏆 調整後の総合評価:")
    print(f"良好: {good_count}/5, やや差: {ok_count}/5, 大きな差: {bad_count}/5")
    
    if good_count >= 4:
        print("✅ 調整成功！実データ統計と良好に一致")
        grade = "A"
    elif good_count >= 3:
        print("⚠️ 概ね良好、さらなる微調整の余地あり")
        grade = "B+"
    else:
        print("⚠️ 追加調整が推奨されます")
        grade = "B"
    
    print(f"調整後評価: {grade}")
    
    # 改善度の計算
    old_mean = 0.008309  # 前回のテスト結果
    current_mean = final_stats['mean']
    real_mean = REAL_STATS['mean']
    
    old_error = abs(old_mean - real_mean) / real_mean * 100
    current_error = abs(current_mean - real_mean) / real_mean * 100
    
    if old_error > 0:
        improvement = (old_error - current_error) / old_error * 100
        print(f"\n📈 改善度: {improvement:.1f}%")
        
        if improvement > 30:
            print("✅ 大幅な改善が確認されました")
        elif improvement > 10:
            print("⚠️ 改善が確認されました")
        elif improvement > -10:
            print("➡️ ほぼ同等の性能")
        else:
            print("⚠️ 性能が低下しました")

if __name__ == "__main__":
    test_adjusted_algorithm()
