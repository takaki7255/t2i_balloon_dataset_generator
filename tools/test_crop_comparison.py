"""
クロップ最適化の効果を確認するためのテストスクリプト
余白除去前後での合成結果を比較する
"""

import cv2
import numpy as np
from pathlib import Path
import os
import random
import time

# 改善前の関数（参考用）
def composite_original(background_path, balloon_mask_pairs, scale_range=(0.1, 0.4)):
    """従来版：余白込みでリサイズ"""
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    bg_h, bg_w = background.shape[:2]
    result_img = background.copy()
    
    balloon_path, mask_path = balloon_mask_pairs[0]  # 最初の1個だけテスト
    
    balloon = cv2.imread(balloon_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if balloon is None or mask is None:
        return None, None
    
    # 従来版：余白込み全体をリサイズ
    original_h, original_w = balloon.shape[:2]
    balloon_scale = random.uniform(scale_range[0], scale_range[1])
    new_w = int(bg_w * balloon_scale)
    new_h = int(original_h * (new_w / original_w))
    
    balloon_resized = cv2.resize(balloon, (new_w, new_h))
    mask_resized = cv2.resize(mask, (new_w, new_h))
    
    # 中央配置
    x = (bg_w - new_w) // 2
    y = (bg_h - new_h) // 2
    
    if x >= 0 and y >= 0 and x + new_w <= bg_w and y + new_h <= bg_h:
        # アルファブレンディング
        mask_norm = mask_resized.astype(np.float32) / 255.0
        mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
        
        bg_region = result_img[y:y+new_h, x:x+new_w]
        blended = balloon_resized.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
        result_img[y:y+new_h, x:x+new_w] = blended.astype(np.uint8)
    
    return result_img, (original_w, original_h, new_w, new_h)

# 改善版の関数
def get_mask_bbox(mask):
    """マスクの非ゼロ領域の境界ボックスを取得"""
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1

def crop_balloon_and_mask(balloon, mask):
    """マスクの境界ボックスに基づいて吹き出し画像とマスクをクロップ"""
    x, y, w, h = get_mask_bbox(mask)
    
    cropped_balloon = balloon[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    return cropped_balloon, cropped_mask, (x, y, w, h)

def composite_cropped(background_path, balloon_mask_pairs, scale_range=(0.1, 0.4)):
    """改善版：吹き出し部分のみクロップしてリサイズ"""
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    bg_h, bg_w = background.shape[:2]
    result_img = background.copy()
    
    balloon_path, mask_path = balloon_mask_pairs[0]  # 最初の1個だけテスト
    
    balloon = cv2.imread(balloon_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if balloon is None or mask is None:
        return None, None
    
    # 改善版：吹き出し部分のみクロップ
    cropped_balloon, cropped_mask, bbox = crop_balloon_and_mask(balloon, mask)
    
    if cropped_balloon.size == 0 or cropped_mask.size == 0:
        return None, None
    
    crop_h, crop_w = cropped_balloon.shape[:2]
    balloon_scale = random.uniform(scale_range[0], scale_range[1])
    new_w = int(bg_w * balloon_scale)
    new_h = int(crop_h * (new_w / crop_w))
    
    balloon_resized = cv2.resize(cropped_balloon, (new_w, new_h))
    mask_resized = cv2.resize(cropped_mask, (new_w, new_h))
    
    # 中央配置
    x = (bg_w - new_w) // 2
    y = (bg_h - new_h) // 2
    
    if x >= 0 and y >= 0 and x + new_w <= bg_w and y + new_h <= bg_h:
        # アルファブレンディング
        mask_norm = mask_resized.astype(np.float32) / 255.0
        mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
        
        bg_region = result_img[y:y+new_h, x:x+new_w]
        blended = balloon_resized.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
        result_img[y:y+new_h, x:x+new_w] = blended.astype(np.uint8)
    
    original_h, original_w = balloon.shape[:2]
    return result_img, (original_w, original_h, new_w, new_h, crop_w, crop_h, bbox)

def test_crop_comparison():
    """クロップ最適化の効果比較テスト"""
    
    # パス設定
    balloons_dir = "generated_balloons"
    masks_dir = "masks"
    backgrounds_dir = "generated_backs"
    
    # データ取得
    balloon_mask_pairs = []
    for balloon_file in os.listdir(balloons_dir):
        if balloon_file.endswith(('.png', '.jpg', '.jpeg')):
            balloon_path = os.path.join(balloons_dir, balloon_file)
            balloon_stem = Path(balloon_file).stem
            
            mask_file = f"{balloon_stem}_mask.png"
            mask_path = os.path.join(masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                balloon_mask_pairs.append((balloon_path, mask_path))
    
    background_files = [os.path.join(backgrounds_dir, f) for f in os.listdir(backgrounds_dir) 
                       if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not background_files or not balloon_mask_pairs:
        print("テストデータが見つかりません")
        return
    
    print(f"テストデータ: 背景 {len(background_files)}個, 吹き出し {len(balloon_mask_pairs)}個")
    
    # 出力ディレクトリ作成
    output_dir = "crop_comparison_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # テスト実行
    test_bg = background_files[0]
    test_pairs = balloon_mask_pairs[:5]  # 最初の5個でテスト
    
    print(f"\n=== クロップ最適化比較テスト ===")
    print(f"背景: {Path(test_bg).name}")
    
    random.seed(42)  # 再現性のため
    
    for i, pair in enumerate(test_pairs):
        balloon_name = Path(pair[0]).stem
        
        # 従来版テスト
        random.seed(42 + i)  # 同じランダム値を使用
        result_orig, info_orig = composite_original(test_bg, [pair])
        
        # 改善版テスト
        random.seed(42 + i)  # 同じランダム値を使用
        result_crop, info_crop = composite_cropped(test_bg, [pair])
        
        if result_orig is not None and result_crop is not None:
            # 結果保存
            orig_path = os.path.join(output_dir, f"test_{i+1:02d}_original_{balloon_name}.png")
            crop_path = os.path.join(output_dir, f"test_{i+1:02d}_cropped_{balloon_name}.png")
            
            cv2.imwrite(orig_path, result_orig)
            cv2.imwrite(crop_path, result_crop)
            
            # 情報表示
            orig_w, orig_h, new_w_orig, new_h_orig = info_orig
            if len(info_crop) == 7:
                orig_w2, orig_h2, new_w_crop, new_h_crop, crop_w, crop_h, bbox = info_crop
                
                print(f"\n{i+1}. {balloon_name}")
                print(f"  元画像サイズ: {orig_w}x{orig_h}")
                print(f"  クロップサイズ: {crop_w}x{crop_h} (効率: {(crop_w*crop_h)/(orig_w*orig_h)*100:.1f}%)")
                print(f"  従来版リサイズ: {new_w_orig}x{new_h_orig}")
                print(f"  改善版リサイズ: {new_w_crop}x{new_h_crop}")
                print(f"  余白削減効果: 幅 {((orig_w-crop_w)/orig_w)*100:.1f}%, 高さ {((orig_h-crop_h)/orig_h)*100:.1f}%")
        else:
            print(f"✗ テスト失敗: {balloon_name}")
    
    print(f"\n結果を {output_dir} に保存しました")
    print("従来版(_original)と改善版(_cropped)の比較画像を確認してください")

if __name__ == "__main__":
    test_crop_comparison()
