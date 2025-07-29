#!/usr/bin/env python3
"""
背景と吹き出しのサイズ分析スクリプト
"""

import cv2
import numpy as np
from pathlib import Path
import os

def analyze_sizes():
    balloons_dir = "../generated_balloons"
    masks_dir = "../masks"
    backgrounds_dir = "../generated_double_backs"
    
    print("ディレクトリ存在確認:")
    print(f"  balloons_dir: {os.path.exists(balloons_dir)}")
    print(f"  masks_dir: {os.path.exists(masks_dir)}")
    print(f"  backgrounds_dir: {os.path.exists(backgrounds_dir)}")
    
    # 背景画像サイズ分析
    print("\n=== 背景画像サイズ分析 ===")
    bg_sizes = []
    
    if not os.path.exists(backgrounds_dir):
        print(f"エラー: {backgrounds_dir} が見つかりません")
        return
        
    bg_files = [f for f in os.listdir(backgrounds_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"背景画像数: {len(bg_files)}")
    
    for bg_file in bg_files[:5]:  # 最初の5枚を分析
        bg_path = os.path.join(backgrounds_dir, bg_file)
        try:
            bg_img = cv2.imread(bg_path)
            if bg_img is not None:
                h, w = bg_img.shape[:2]
                bg_sizes.append((w, h))
                print(f"{bg_file}: {w}x{h}")
            else:
                print(f"警告: {bg_file} を読み込めません")
        except Exception as e:
            print(f"エラー: {bg_file} - {e}")
    
    # 吹き出し画像サイズ分析
    print("\n=== 吹き出し画像サイズ分析 ===")
    balloon_sizes = []
    balloon_files = [f for f in os.listdir(balloons_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for balloon_file in balloon_files[:10]:  # 最初の10枚を分析
        balloon_path = os.path.join(balloons_dir, balloon_file)
        balloon_img = cv2.imread(balloon_path)
        
        # 対応するマスクファイル
        balloon_stem = Path(balloon_file).stem
        mask_file = f"{balloon_stem}_mask.png"
        mask_path = os.path.join(masks_dir, mask_file)
        
        if balloon_img is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            h, w = balloon_img.shape[:2]
            
            # マスクの境界ボックス計算
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                crop_w, crop_h = x_max - x_min + 1, y_max - y_min + 1
            else:
                crop_w, crop_h = w, h
            
            balloon_sizes.append((w, h, crop_w, crop_h))
            efficiency = (crop_w * crop_h) / (w * h)
            print(f"{balloon_file}: 元{w}x{h} → クロップ{crop_w}x{crop_h} (効率:{efficiency:.3f})")
    
    # 統計情報
    if bg_sizes and balloon_sizes:
        avg_bg_w = sum(w for w, h in bg_sizes) / len(bg_sizes)
        avg_bg_h = sum(h for w, h in bg_sizes) / len(bg_sizes)
        
        avg_balloon_w = sum(crop_w for w, h, crop_w, crop_h in balloon_sizes) / len(balloon_sizes)
        avg_balloon_h = sum(crop_h for w, h, crop_w, crop_h in balloon_sizes) / len(balloon_sizes)
        
        print(f"\n=== 統計情報 ===")
        print(f"平均背景サイズ: {avg_bg_w:.0f}x{avg_bg_h:.0f}")
        print(f"平均吹き出しサイズ（クロップ後）: {avg_balloon_w:.0f}x{avg_balloon_h:.0f}")
        print(f"現在のスケール設定での吹き出し幅:")
        print(f"  最小 (5%): {avg_bg_w * 0.05:.0f}px")
        print(f"  平均 (25%): {avg_bg_w * 0.25:.0f}px")
        print(f"  最大 (40%): {avg_bg_w * 0.40:.0f}px")
        print(f"吹き出し実サイズと推奨スケール:")
        recommended_scale = avg_balloon_w / avg_bg_w
        print(f"  推奨スケール: {recommended_scale:.3f} ({recommended_scale*100:.1f}%)")

if __name__ == "__main__":
    analyze_sizes()
