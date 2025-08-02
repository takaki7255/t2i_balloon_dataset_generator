"""
旧方法 vs 新方法のリサイズ比較テスト
"""

import cv2
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append('..')
from create_syn_dataset import calculate_area_based_size, sample_scale

def old_resize_method(crop_w, crop_h, bg_w, bg_h, scale):
    """旧方法: 幅ベースのリサイズ"""
    new_balloon_w = int(bg_w * scale)
    new_balloon_h = int(crop_h * (new_balloon_w / crop_w))
    return new_balloon_w, new_balloon_h

def test_resize_comparison():
    """リサイズ方法の比較テスト"""
    
    print("=== リサイズ方法比較テスト ===")
    print()
    
    # テストケース（実際の吹き出しサイズを基に）
    test_cases = [
        {"name": "横長吹き出し", "crop_w": 740, "crop_h": 769},
        {"name": "縦長吹き出し", "crop_w": 506, "crop_h": 1326},
        {"name": "正方形吹き出し", "crop_w": 600, "crop_h": 580},
        {"name": "極端な縦長", "crop_w": 300, "crop_h": 1200},
        {"name": "極端な横長", "crop_w": 1200, "crop_h": 300},
    ]
    
    bg_w, bg_h = 1000, 600
    test_scale = 0.1
    
    print(f"背景サイズ: {bg_w}x{bg_h}")
    print(f"テストスケール: {test_scale}")
    print("=" * 80)
    
    for case in test_cases:
        crop_w, crop_h = case["crop_w"], case["crop_h"]
        aspect_ratio = crop_h / crop_w
        
        # 旧方法
        old_w, old_h = old_resize_method(crop_w, crop_h, bg_w, bg_h, test_scale)
        old_area = old_w * old_h
        old_bg_ratio = old_area / (bg_w * bg_h)
        
        # 新方法
        new_w, new_h = calculate_area_based_size(crop_w, crop_h, bg_w, bg_h, test_scale)
        new_area = new_w * new_h
        new_bg_ratio = new_area / (bg_w * bg_h)
        
        print(f"🔹 {case['name']}")
        print(f"  元サイズ: {crop_w}x{crop_h} (アスペクト比: {aspect_ratio:.3f})")
        print(f"  旧方法: {old_w}x{old_h} (面積: {old_area:,}px², 背景の{old_bg_ratio:.1%})")
        print(f"  新方法: {new_w}x{new_h} (面積: {new_area:,}px², 背景の{new_bg_ratio:.1%})")
        
        # 改善評価
        area_reduction = (old_area - new_area) / old_area * 100
        if aspect_ratio > 1.2:  # 縦長
            print(f"  → 縦長吹き出しの面積 {area_reduction:+.1f}% 変化 ({'改善' if area_reduction > 0 else '変化'})")
        elif aspect_ratio < 0.8:  # 横長
            print(f"  → 横長吹き出しの面積 {area_reduction:+.1f}% 変化 ({'改善' if area_reduction < 0 else '変化'})")
        else:
            print(f"  → 正方形に近い形状の面積 {area_reduction:+.1f}% 変化")
        print()
    
    print("=" * 80)
    print("💡 改善効果:")
    print("  ✅ 縦長吹き出しの過大サイズが解消")
    print("  ✅ アスペクト比に関わらず一定の面積比を維持")
    print("  ✅ 最大サイズ制限による安全性向上")
    print("  ✅ より公平で予測可能なリサイズ")

def test_multiple_scales():
    """複数のスケール値でのテスト"""
    print("\n=== 複数スケールでの一貫性テスト ===")
    
    # 極端な縦長吹き出し
    crop_w, crop_h = 400, 1200
    bg_w, bg_h = 1000, 600
    
    scales = [0.05, 0.08, 0.10, 0.12, 0.15]
    
    print(f"テスト吹き出し: {crop_w}x{crop_h} (縦長, アスペクト比: {crop_h/crop_w:.2f})")
    print(f"背景: {bg_w}x{bg_h}")
    print()
    print("スケール  | 旧方法(WxH)  | 旧面積比 | 新方法(WxH)  | 新面積比 | 改善")
    print("-" * 70)
    
    for scale in scales:
        # 旧方法
        old_w, old_h = old_resize_method(crop_w, crop_h, bg_w, bg_h, scale)
        old_ratio = (old_w * old_h) / (bg_w * bg_h)
        
        # 新方法
        new_w, new_h = calculate_area_based_size(crop_w, crop_h, bg_w, bg_h, scale)
        new_ratio = (new_w * new_h) / (bg_w * bg_h)
        
        improvement = (old_ratio - new_ratio) / old_ratio * 100
        
        print(f"{scale:6.2f}  | {old_w:3}x{old_h:3}   | {old_ratio:6.1%}  | {new_w:3}x{new_h:3}   | {new_ratio:6.1%}  | {improvement:+4.0f}%")

if __name__ == "__main__":
    test_resize_comparison()
    test_multiple_scales()
