"""
CFG設定と実際の統計データの比較・検証
"""

import numpy as np

def analyze_cfg_vs_statistics():
    """CFG設定と実際の統計を比較"""
    
    print("=== CFG設定 vs 実際の統計データ比較 ===")
    print()
    
    # 現在のCFG設定
    current_cfg = {
        "NUM_BALLOONS_RANGE": (7, 17),
        "SCALE_RANGE": (0.1, 0.3),
        "SCALE_MEAN": 0.10,
        "SCALE_STD": 0.03,
        "SCALE_CLIP": (0.05, 0.18),
    }
    
    # 実際の統計データ
    actual_stats = {
        # balloon_count_statistics.txt から
        "count_mean": 13.128278,
        "count_median": 13.0,
        "count_std": 6.384961,
        "count_25th": 8.0,
        "count_75th": 17.0,
        "count_min": 1,
        "count_max": 44,
        
        # balloon_size_statistics.txt から
        "size_ratio_mean": 0.008769,
        "size_ratio_median": 0.007226,
        "size_ratio_std": 0.006773,
        "size_ratio_25th": 0.004381,
        "size_ratio_75th": 0.011281,
        "size_ratio_min": 0.000011,
        "size_ratio_max": 0.283788,
    }
    
    print("🎯 **吹き出し個数の設定**")
    print(f"CFG設定: {current_cfg['NUM_BALLOONS_RANGE']}")
    print(f"実際の統計:")
    print(f"  平均: {actual_stats['count_mean']:.1f}個")
    print(f"  中央値: {actual_stats['count_median']:.1f}個")
    print(f"  25-75%範囲: {actual_stats['count_25th']:.0f}-{actual_stats['count_75th']:.0f}個")
    print(f"  最小-最大: {actual_stats['count_min']}-{actual_stats['count_max']}個")
    print()
    
    # 個数設定の評価
    cfg_min, cfg_max = current_cfg['NUM_BALLOONS_RANGE']
    if cfg_min >= actual_stats['count_25th'] and cfg_max <= actual_stats['count_75th']:
        print("✅ **個数設定は適切**: 25-75%範囲内でバランス良い")
    elif cfg_min < actual_stats['count_25th']:
        print("⚠️ **個数設定低め**: 最小値が25%値より小さい")
    elif cfg_max > actual_stats['count_75th']:
        print("⚠️ **個数設定高め**: 最大値が75%値より大きい")
    else:
        print("❓ **個数設定要確認**: 統計と一致しない")
    
    print()
    print("🎯 **吹き出しサイズの設定**")
    print(f"CFG設定:")
    print(f"  スケール範囲: {current_cfg['SCALE_RANGE']}")
    print(f"  平均: {current_cfg['SCALE_MEAN']}")
    print(f"  標準偏差: {current_cfg['SCALE_STD']}")
    print(f"  クリップ範囲: {current_cfg['SCALE_CLIP']}")
    print()
    print(f"実際の統計 (画面に対する面積比):")
    print(f"  平均: {actual_stats['size_ratio_mean']:.6f} ({actual_stats['size_ratio_mean']*100:.3f}%)")
    print(f"  中央値: {actual_stats['size_ratio_median']:.6f} ({actual_stats['size_ratio_median']*100:.3f}%)")
    print(f"  25-75%範囲: {actual_stats['size_ratio_25th']:.6f}-{actual_stats['size_ratio_75th']:.6f}")
    print(f"  面積比換算: {actual_stats['size_ratio_25th']*100:.3f}%-{actual_stats['size_ratio_75th']*100:.3f}%")
    print()
    
    # 面積比から幅比への換算（目安）
    # 面積比 = (幅比)² として計算
    width_ratio_from_area_mean = np.sqrt(actual_stats['size_ratio_mean'])
    width_ratio_from_area_25th = np.sqrt(actual_stats['size_ratio_25th'])
    width_ratio_from_area_75th = np.sqrt(actual_stats['size_ratio_75th'])
    
    print("📐 **面積比から推定される幅比**")
    print(f"  平均幅比: {width_ratio_from_area_mean:.3f} ({width_ratio_from_area_mean*100:.1f}%)")
    print(f"  25-75%幅比: {width_ratio_from_area_25th:.3f}-{width_ratio_from_area_75th:.3f}")
    print(f"  幅比換算: {width_ratio_from_area_25th*100:.1f}%-{width_ratio_from_area_75th*100:.1f}%")
    print()
    
    # CFG設定の評価
    cfg_mean = current_cfg['SCALE_MEAN']
    cfg_min, cfg_max = current_cfg['SCALE_RANGE']
    
    print("📊 **サイズ設定の評価**")
    if abs(cfg_mean - width_ratio_from_area_mean) < 0.02:
        print("✅ **平均サイズ設定は適切**")
    elif cfg_mean > width_ratio_from_area_mean:
        ratio = cfg_mean / width_ratio_from_area_mean
        print(f"⚠️ **平均サイズ設定大きめ**: 実際の{ratio:.1f}倍")
    else:
        ratio = width_ratio_from_area_mean / cfg_mean
        print(f"⚠️ **平均サイズ設定小さめ**: 実際の{1/ratio:.1f}倍")
    
    if cfg_min >= width_ratio_from_area_25th * 0.8 and cfg_max <= width_ratio_from_area_75th * 1.2:
        print("✅ **範囲設定は適切**: 統計範囲とバランス良い")
    else:
        print("⚠️ **範囲設定要調整**: 統計データと乖離がある")
    
    print()
    print("🔧 **推奨設定**")
    print("個数範囲:")
    print(f"  現在: {current_cfg['NUM_BALLOONS_RANGE']}")
    print(f"  推奨: ({int(actual_stats['count_25th'])}, {int(actual_stats['count_75th'])}) # 25-75%範囲")
    print()
    print("サイズ設定:")
    print(f"  現在の平均: {current_cfg['SCALE_MEAN']}")
    print(f"  推奨平均: {width_ratio_from_area_mean:.3f} # 実際の統計に合わせる")
    print(f"  現在の範囲: {current_cfg['SCALE_RANGE']}")
    print(f"  推奨範囲: ({width_ratio_from_area_25th:.3f}, {width_ratio_from_area_75th:.3f}) # 25-75%範囲")
    print()
    
    # 詳細分析
    print("=" * 60)
    print("📈 **詳細分析**")
    print()
    print("実際のマンガにおける吹き出しサイズ:")
    print(f"• 平均面積比: {actual_stats['size_ratio_mean']*100:.3f}% (画面全体に対して)")
    print(f"• 標準的な範囲: {actual_stats['size_ratio_25th']*100:.3f}%-{actual_stats['size_ratio_75th']*100:.3f}%")
    print(f"• 最大サイズ: {actual_stats['size_ratio_max']*100:.1f}% (極端なケース)")
    print()
    print("CFG設定での想定サイズ:")
    cfg_area_min = current_cfg['SCALE_RANGE'][0] ** 2
    cfg_area_max = current_cfg['SCALE_RANGE'][1] ** 2
    cfg_area_mean = current_cfg['SCALE_MEAN'] ** 2
    print(f"• 設定面積比範囲: {cfg_area_min*100:.1f}%-{cfg_area_max*100:.1f}%")
    print(f"• 設定平均面積比: {cfg_area_mean*100:.1f}%")
    print()
    
    if cfg_area_mean > actual_stats['size_ratio_mean'] * 2:
        print("❌ **設定が大きすぎる**: 実際の2倍以上のサイズ")
    elif cfg_area_mean > actual_stats['size_ratio_mean'] * 1.5:
        print("⚠️ **設定がやや大きい**: 実際の1.5倍以上のサイズ")
    elif cfg_area_mean < actual_stats['size_ratio_mean'] * 0.5:
        print("⚠️ **設定が小さすぎる**: 実際の半分以下のサイズ")
    else:
        print("✅ **設定サイズは許容範囲内**")

if __name__ == "__main__":
    analyze_cfg_vs_statistics()
