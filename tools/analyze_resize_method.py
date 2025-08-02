"""
現在のリサイズ方法の分析と縦長吹き出しの問題調査
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_current_resize_method():
    """現在のリサイズ方法を分析"""
    
    print("=== 現在のリサイズ方法の分析 ===")
    print()
    
    print("📏 現在の手順:")
    print("1. マスクの境界ボックスでクロップ（余白除去）")
    print("2. crop_h, crop_w = cropped_balloon.shape[:2]")
    print("3. balloon_scale = sample_scale(bg_w, crop_w, cfg)  ← 幅のみ考慮")
    print("4. new_balloon_w = int(bg_w * balloon_scale)")
    print("5. new_balloon_h = int(crop_h * (new_balloon_w / crop_w))  ← アスペクト比維持")
    print("6. if new_balloon_w >= bg_w or new_balloon_h >= bg_h: continue  ← サイズ制限")
    print()
    
    print("⚠️ 問題点:")
    print("1. 【幅のみでスケール決定】")
    print("   - balloon_scale = sample_scale(bg_w, crop_w, cfg)")
    print("   - 縦長吹き出しの高さは考慮されない")
    print()
    print("2. 【縦長吹き出しの問題】")
    print("   - 幅ベースでスケールを決めるため、縦長の場合:")
    print("     crop_w = 100, crop_h = 400 (縦長)")
    print("     bg_w = 1000, balloon_scale = 0.1")
    print("     → new_balloon_w = 100")
    print("     → new_balloon_h = 400 * (100/100) = 400")
    print("     → 結果: 100x400 (かなり大きい)")
    print()
    print("3. 【横長吹き出しとの不平等】")
    print("   - 横長の場合:")
    print("     crop_w = 400, crop_h = 100 (横長)")
    print("     bg_w = 1000, balloon_scale = 0.1")
    print("     → new_balloon_w = 100")
    print("     → new_balloon_h = 100 * (100/400) = 25")
    print("     → 結果: 100x25 (小さい)")
    print()
    
    print("🔍 実際のデータで確認:")
    
    # 実際の吹き出しデータを分析
    balloons_dir = "../generated_balloons"
    masks_dir = "../masks"
    
    if not os.path.exists(balloons_dir):
        print(f"❌ {balloons_dir} が見つかりません")
        return
    
    aspect_ratios = []
    sizes_info = []
    
    for balloon_file in os.listdir(balloons_dir)[:20]:  # 最初の20個で分析
        if not balloon_file.endswith('.png'):
            continue
            
        balloon_path = os.path.join(balloons_dir, balloon_file)
        mask_file = f"{Path(balloon_file).stem}_mask.png"
        mask_path = os.path.join(masks_dir, mask_file)
        
        if not os.path.exists(mask_path):
            continue
            
        balloon = cv2.imread(balloon_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if balloon is None or mask is None:
            continue
        
        # 境界ボックス取得
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            continue
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        crop_w = x_max - x_min + 1
        crop_h = y_max - y_min + 1
        aspect_ratio = crop_h / crop_w
        
        aspect_ratios.append(aspect_ratio)
        sizes_info.append({
            'file': balloon_file,
            'crop_w': crop_w,
            'crop_h': crop_h,
            'aspect_ratio': aspect_ratio
        })
    
    # 統計情報を表示
    if aspect_ratios:
        print(f"\n📊 アスペクト比分析 (H/W):")
        print(f"  平均: {np.mean(aspect_ratios):.3f}")
        print(f"  最小: {np.min(aspect_ratios):.3f} (横長)")
        print(f"  最大: {np.max(aspect_ratios):.3f} (縦長)")
        print(f"  標準偏差: {np.std(aspect_ratios):.3f}")
        
        # 縦長と横長の例を表示
        sorted_by_aspect = sorted(sizes_info, key=lambda x: x['aspect_ratio'])
        
        print(f"\n🔹 最も横長な吹き出し:")
        horizontal = sorted_by_aspect[0]
        print(f"  ファイル: {horizontal['file']}")
        print(f"  サイズ: {horizontal['crop_w']}x{horizontal['crop_h']}")
        print(f"  アスペクト比: {horizontal['aspect_ratio']:.3f}")
        
        print(f"\n🔹 最も縦長な吹き出し:")
        vertical = sorted_by_aspect[-1]
        print(f"  ファイル: {vertical['file']}")
        print(f"  サイズ: {vertical['crop_w']}x{vertical['crop_h']}")
        print(f"  アスペクト比: {vertical['aspect_ratio']:.3f}")
        
        # 背景幅1000px、スケール0.1での結果サイズをシミュレート
        bg_w = 1000
        scale = 0.1
        
        print(f"\n🎯 背景幅{bg_w}px、スケール{scale}での結果予測:")
        
        for example in [horizontal, vertical]:
            new_w = int(bg_w * scale)
            new_h = int(example['crop_h'] * (new_w / example['crop_w']))
            area = new_w * new_h
            bg_area_ratio = area / (bg_w * 600)  # 仮に背景高さ600px
            
            print(f"  {example['file']}:")
            print(f"    元サイズ: {example['crop_w']}x{example['crop_h']}")
            print(f"    結果サイズ: {new_w}x{new_h}")
            print(f"    面積: {area:,}px² (背景の{bg_area_ratio:.1%})")
            print()

def propose_improvements():
    """改善提案"""
    print("=" * 60)
    print("🔧 改善提案")
    print("=" * 60)
    print()
    
    print("💡 提案1: 面積ベースのスケーリング")
    print("現在: new_balloon_w = bg_w * scale (幅のみ)")
    print("改善: target_area = bg_area * scale²")
    print("     aspect_ratio = crop_h / crop_w")
    print("     new_w = sqrt(target_area / aspect_ratio)")
    print("     new_h = sqrt(target_area * aspect_ratio)")
    print()
    
    print("💡 提案2: 最大サイズ制限ベースのスケーリング")
    print("現在: スケールを先に決定 → サイズ制限でカット")
    print("改善: サイズ制限を考慮してスケールを調整")
    print("     max_scale_by_width = max_w / crop_w")
    print("     max_scale_by_height = max_h / crop_h")
    print("     scale = min(target_scale, max_scale_by_width, max_scale_by_height)")
    print()
    
    print("💡 提案3: アスペクト比別のスケール調整")
    print("現在: 全ての吹き出しに同じスケール分布")
    print("改善: アスペクト比に応じてスケールを調整")
    print("     if aspect_ratio > 1.5: scale *= 0.8  # 縦長は小さく")
    print("     elif aspect_ratio < 0.7: scale *= 1.2  # 横長は大きく")
    print()

if __name__ == "__main__":
    analyze_current_resize_method()
    propose_improvements()
