"""
吹き出し部分のみをクロップしてからリサイズする最適化版の合成関数
余白を除去してより効率的な合成を行う
"""

import cv2
import numpy as np
from pathlib import Path
import os
import random
from typing import Tuple, List
import time

def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    マスクの非ゼロ領域の境界ボックスを取得
    
    Args:
        mask: グレースケールマスク画像
    
    Returns:
        (x, y, width, height) の境界ボックス
    """
    # マスクの非ゼロピクセルの座標を取得
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]  # 全体を返す
    
    # 境界ボックス計算
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1

def crop_balloon_and_mask(balloon: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    マスクの境界ボックスに基づいて吹き出し画像とマスクをクロップ
    
    Args:
        balloon: 吹き出し画像 (BGR)
        mask: マスク画像 (グレースケール)
    
    Returns:
        (クロップされた吹き出し, クロップされたマスク, 境界ボックス)
    """
    x, y, w, h = get_mask_bbox(mask)
    
    # クロップ実行
    cropped_balloon = balloon[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    return cropped_balloon, cropped_mask, (x, y, w, h)

def regions_overlap(region1: tuple, region2: tuple) -> bool:
    """2つの領域が重複するかチェック"""
    x1_min, y1_min, x1_max, y1_max = region1
    x2_min, y2_min, x2_max, y2_max = region2
    
    return not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)

def calculate_overlap_area(region1: tuple, region2: tuple) -> int:
    """2つの領域の重複面積を計算"""
    x1_min, y1_min, x1_max, y1_max = region1
    x2_min, y2_min, x2_max, y2_max = region2
    
    # 重複領域の計算
    overlap_x_min = max(x1_min, x2_min)
    overlap_y_min = max(y1_min, y2_min)
    overlap_x_max = min(x1_max, x2_max)
    overlap_y_max = min(y1_max, y2_max)
    
    if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
        return 0
    
    return (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)

def composite_random_balloons_crop_optimized(
    background_path: str, 
    balloon_mask_pairs: list,
    scale_range: tuple = (0.1, 0.4), 
    num_balloons_range: tuple = (2, 10),
    max_attempts: int = 200
) -> tuple:
    """
    吹き出し部分のみをクロップしてからリサイズする最適化版の合成関数
    
    Args:
        background_path: 背景画像のパス
        balloon_mask_pairs: [(balloon_path, mask_path), ...] のリスト
        scale_range: 吹き出しのスケール範囲（最小, 最大）
        num_balloons_range: 配置する吹き出し数の範囲（最小, 最大）
        max_attempts: 配置試行回数の上限
    
    Returns:
        (合成画像, 合成マスク, 配置された吹き出し名リスト)
    """
    # 背景画像読み込み
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")
    
    bg_h, bg_w = background.shape[:2]
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    # 配置する吹き出し数をランダムに決定
    min_balloons, max_balloons = num_balloons_range
    max_balloons = min(max_balloons, len(balloon_mask_pairs))
    num_balloons = random.randint(min_balloons, max_balloons)
    
    # ランダムに吹き出しを選択
    selected_pairs = random.sample(balloon_mask_pairs, num_balloons)
    
    # 配置済み領域を記録する配列
    occupied_regions = []
    successfully_placed = []
    
    for balloon_path, mask_path in selected_pairs:
        # 吹き出しとマスク読み込み
        balloon = cv2.imread(balloon_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if balloon is None or mask is None:
            print(f"警告: 画像読み込み失敗 ({balloon_path}, {mask_path})")
            continue
        
        # マスクの境界ボックスでクロップ
        cropped_balloon, cropped_mask, bbox = crop_balloon_and_mask(balloon, mask)
        
        if cropped_balloon.size == 0 or cropped_mask.size == 0:
            print(f"警告: クロップ結果が空 ({balloon_path})")
            continue
        
        # クロップされた画像のサイズ
        crop_h, crop_w = cropped_balloon.shape[:2]
        
        # ランダムサイズ（吹き出し部分基準）
        balloon_scale = random.uniform(scale_range[0], scale_range[1])
        new_balloon_w = int(bg_w * balloon_scale)
        new_balloon_h = int(crop_h * (new_balloon_w / crop_w))
        
        # 背景サイズを超える場合はスキップ
        if new_balloon_w >= bg_w or new_balloon_h >= bg_h:
            continue
        
        # クロップされた画像をリサイズ
        balloon_resized = cv2.resize(cropped_balloon, (new_balloon_w, new_balloon_h))
        mask_resized = cv2.resize(cropped_mask, (new_balloon_w, new_balloon_h))
        
        # 重複を避けつつ位置を探す
        placed = False
        best_position = None
        min_overlap_ratio = float('inf')
        
        # まず重複回避を試行
        for attempt in range(max_attempts // 2):
            max_x = max(0, bg_w - new_balloon_w)
            max_y = max(0, bg_h - new_balloon_h)
            
            if max_x <= 0 or max_y <= 0:
                break
                
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # 新しい領域
            new_region = (x, y, x + new_balloon_w, y + new_balloon_h)
            
            # 既存領域との重複チェック
            max_overlap_ratio = 0
            for occupied in occupied_regions:
                if regions_overlap(new_region, occupied):
                    overlap_area = calculate_overlap_area(new_region, occupied)
                    new_area = new_balloon_w * new_balloon_h
                    overlap_ratio = overlap_area / new_area
                    max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)
            
            # 重複が少ない場合は配置
            if max_overlap_ratio <= 0.15:
                best_position = (x, y)
                placed = True
                break
            
            # より良い位置を記録
            if max_overlap_ratio < min_overlap_ratio:
                min_overlap_ratio = max_overlap_ratio
                best_position = (x, y)
        
        # 重複回避に失敗した場合、最も重複の少ない位置に配置
        if not placed and best_position is not None:
            x, y = best_position
            placed = True
        
        # 最終的にランダム位置に配置（フォールバック）
        if not placed:
            max_x = max(0, bg_w - new_balloon_w)
            max_y = max(0, bg_h - new_balloon_h)
            if max_x >= 0 and max_y >= 0:
                x = random.randint(0, max_x) if max_x > 0 else 0
                y = random.randint(0, max_y) if max_y > 0 else 0
                placed = True
        
        if placed:
            # 合成実行
            mask_norm = mask_resized.astype(np.float32) / 255.0
            mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
            
            # 背景画像の該当領域
            bg_region = result_img[y:y+new_balloon_h, x:x+new_balloon_w]
            
            # アルファブレンディング
            blended = balloon_resized.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
            result_img[y:y+new_balloon_h, x:x+new_balloon_w] = blended.astype(np.uint8)
            
            # マスクも合成
            result_mask[y:y+new_balloon_h, x:x+new_balloon_w] = np.maximum(
                result_mask[y:y+new_balloon_h, x:x+new_balloon_w], mask_resized)
            
            # 配置済み領域に追加
            new_region = (x, y, x + new_balloon_w, y + new_balloon_h)
            occupied_regions.append(new_region)
            successfully_placed.append(Path(balloon_path).stem)
        else:
            print(f"配置失敗: {Path(balloon_path).stem}")
    
    return result_img, result_mask, successfully_placed

def test_crop_optimization():
    """クロップ最適化のテストとパフォーマンス比較"""
    
    # パス設定
    balloons_dir = "generated_balloons"
    masks_dir = "masks"
    backgrounds_dir = "generated_backs"
    
    # 吹き出しとマスクの対応を取得
    balloon_mask_pairs = []
    for balloon_file in os.listdir(balloons_dir):
        if balloon_file.endswith(('.png', '.jpg', '.jpeg')):
            balloon_path = os.path.join(balloons_dir, balloon_file)
            balloon_stem = Path(balloon_file).stem
            
            mask_file = f"{balloon_stem}_mask.png"
            mask_path = os.path.join(masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                balloon_mask_pairs.append((balloon_path, mask_path))
    
    # 背景画像を取得
    background_files = []
    for bg_file in os.listdir(backgrounds_dir):
        if bg_file.endswith(('.png', '.jpg', '.jpeg')):
            background_files.append(os.path.join(backgrounds_dir, bg_file))
    
    if not background_files or not balloon_mask_pairs:
        print("背景画像または吹き出しが見つかりません")
        return
    
    print(f"テスト用データ: 背景 {len(background_files)}個, 吹き出し {len(balloon_mask_pairs)}個")
    
    # テスト実行
    test_bg = background_files[0]
    test_pairs = balloon_mask_pairs[:10]  # 最初の10個でテスト
    
    CFG = {
        "SCALE_RANGE": (0.1, 0.4),
        "NUM_BALLOONS_RANGE": (3, 7),
        "MAX_ATTEMPTS": 200,
        "SEED": 42
    }
    
    random.seed(CFG["SEED"])
    
    print("\n=== クロップ最適化版テスト ===")
    start_time = time.time()
    
    try:
        result_img, result_mask, placed_balloons = composite_random_balloons_crop_optimized(
            test_bg, test_pairs,
            scale_range=CFG["SCALE_RANGE"],
            num_balloons_range=CFG["NUM_BALLOONS_RANGE"],
            max_attempts=CFG["MAX_ATTEMPTS"]
        )
        
        elapsed_time = time.time() - start_time
        
        # 結果保存
        output_dir = "results_crop_optimized"
        mask_output_dir = "results_crop_optimized_mask"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(mask_output_dir, exist_ok=True)
        
        output_img_path = os.path.join(output_dir, "test_crop_opt.png")
        output_mask_path = os.path.join(mask_output_dir, "test_crop_opt_mask.png")
        
        cv2.imwrite(output_img_path, result_img)
        cv2.imwrite(output_mask_path, result_mask)
        
        print(f"✓ 合成成功: {len(placed_balloons)}個の吹き出しを配置")
        print(f"  配置された吹き出し: {', '.join(placed_balloons)}")
        print(f"  処理時間: {elapsed_time:.3f}秒")
        print(f"  出力: {output_img_path}")
        print(f"  マスク: {output_mask_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ テスト失敗: {e}")
        return False

if __name__ == "__main__":
    test_crop_optimization()
