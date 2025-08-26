#!/usr/bin/env python3
"""
修正されたcalculate_area_based_size関数の動作テスト
"""

import sys
import os
sys.path.append('.')

from create_syn_dataset import calculate_area_based_size, sample_scale
import numpy as np

def test_function_behavior():
    print("🧪 修正された関数の動作テスト")
    print("=" * 60)
    
    # テスト設定
    bg_w, bg_h = 1536, 1024
    bg_area = bg_w * bg_h
    crop_w, crop_h = 300, 200
    
    # 修正されたCFG設定
    CFG = {
        "SCALE_MODE": "lognormal",
        "SCALE_MEAN": 0.009200,
        "SCALE_STD": 0.008000,
        "SCALE_CLIP": (0.002000, 0.020000),
        "SCALE_RANGE": (0.002000, 0.020000),
    }
    
    print(f"📐 テスト設定:")
    print(f"  背景サイズ: {bg_w}×{bg_h}")
    print(f"  クロップサイズ: {crop_w}×{crop_h}")
    print(f"  設定平均面積比: {CFG['SCALE_MEAN']*100:.3f}%")
    print()
    
    # 複数のスケール値でテスト
    test_scales = [0.005, 0.008769, 0.012, 0.015, 0.020]  # 実データ範囲を含むテスト値
    
    print("🔍 スケール値別の生成サイズテスト:")
    print("-" * 60)
    print("スケール値 | 面積比(%) | 生成サイズ    | 実際面積比(%)")
    print("-" * 60)
    
    for scale in test_scales:
        # 面積ベースサイズ計算
        new_w, new_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, scale,
            max_w_ratio=0.20, max_h_ratio=0.30
        )
        
        # 実際の面積比計算
        actual_area = new_w * new_h
        actual_ratio = actual_area / bg_area
        
        print(f"{scale:8.6f} | {scale*100:7.3f} | {new_w:3d}×{new_h:3d}px | {actual_ratio*100:9.3f}")
    
    print()
    
    # サンプリング関数のテスト
    print("🎲 サンプリング関数テスト（100回）:")
    print("-" * 40)
    
    samples = []
    for _ in range(100):
        scale = sample_scale(bg_w, crop_w, CFG)
        samples.append(scale)
    
    # 統計計算
    mean_sample = np.mean(samples)
    std_sample = np.std(samples)
    min_sample = np.min(samples)
    max_sample = np.max(samples)
    median_sample = np.median(samples)
    
    print(f"平均: {mean_sample:.6f} (設定: {CFG['SCALE_MEAN']:.6f})")
    print(f"標準偏差: {std_sample:.6f} (設定: {CFG['SCALE_STD']:.6f})")
    print(f"中央値: {median_sample:.6f}")
    print(f"範囲: {min_sample:.6f} - {max_sample:.6f}")
    print(f"クリップ範囲: {CFG['SCALE_CLIP'][0]:.6f} - {CFG['SCALE_CLIP'][1]:.6f}")
    
    # クリッピング効果の確認
    clipped_low = sum(1 for s in samples if s <= CFG['SCALE_CLIP'][0] + 1e-6)
    clipped_high = sum(1 for s in samples if s >= CFG['SCALE_CLIP'][1] - 1e-6)
    
    print(f"クリッピング: 下限{clipped_low}%, 上限{clipped_high}%")
    
    print()
    
    # 実データとの比較
    real_mean = 0.008769
    real_std = 0.006773
    
    mean_error = abs(mean_sample - real_mean) / real_mean * 100
    std_error = abs(std_sample - real_std) / real_std * 100
    
    print("📊 実データとの比較:")
    print(f"平均誤差: {mean_error:.1f}%")
    print(f"標準偏差誤差: {std_error:.1f}%")
    
    if mean_error < 15 and std_error < 30:
        print("✅ 設定は実データに近い特性を示しています")
    else:
        print("⚠️ 設定の追加調整が推奨されます")
    
    print()
    
    # 修正前後の比較
    print("🔄 修正前後の比較例:")
    print("-" * 40)
    
    test_scale = 0.105  # 旧設定の典型的な値
    
    # 旧方式での計算（参考値）
    old_target_area = bg_area * (test_scale ** 2)
    old_ideal_w = int(np.sqrt(old_target_area / (crop_h / crop_w)))
    old_ideal_h = int(np.sqrt(old_target_area * (crop_h / crop_w)))
    old_area_ratio = old_target_area / bg_area
    
    # 新方式での計算
    new_w, new_h = calculate_area_based_size(
        crop_w, crop_h, bg_w, bg_h, test_scale,
        max_w_ratio=0.20, max_h_ratio=0.30
    )
    new_area_ratio = (new_w * new_h) / bg_area
    
    print(f"テストスケール値: {test_scale}")
    print(f"旧方式 → 面積比: {old_area_ratio*100:.3f}%, サイズ: {old_ideal_w}×{old_ideal_h}px")
    print(f"新方式 → 面積比: {new_area_ratio*100:.3f}%, サイズ: {new_w}×{new_h}px")
    print(f"面積比差: {abs(old_area_ratio - new_area_ratio)*100:.3f}%")

if __name__ == "__main__":
    test_function_behavior()
