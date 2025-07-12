"""
1つの背景画像に対して複数の吹き出しをランダムに合成し、
1背景画像につき5枚の画像を生成するスクリプト
"""

import cv2
import numpy as np
from pathlib import Path
import os
import random
from tqdm import tqdm

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


def composite_random_balloons_enhanced(background_path: str, balloon_mask_pairs: list,
                                     scale_range: tuple = (0.1, 0.4), 
                                     num_balloons_range: tuple = (2, 10),
                                     max_attempts: int = 200) -> tuple:
    """
    1つの背景画像にランダムに選択した複数の吹き出しを重複なしで合成する
    
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
    max_balloons = min(max_balloons, len(balloon_mask_pairs))  # 利用可能な吹き出し数を超えないように
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
        
        original_balloon_h, original_balloon_w = balloon.shape[:2]
        
        # ランダムサイズ
        balloon_scale = random.uniform(scale_range[0], scale_range[1])
        new_balloon_w = int(bg_w * balloon_scale)
        new_balloon_h = int(original_balloon_h * (new_balloon_w / original_balloon_w))
        
        # 背景サイズを超える場合はスキップ
        if new_balloon_w >= bg_w or new_balloon_h >= bg_h:
            continue
        
        # リサイズ
        balloon_resized = cv2.resize(balloon, (new_balloon_w, new_balloon_h))
        mask_resized = cv2.resize(mask, (new_balloon_w, new_balloon_h))
        
        # 重複を避けつつ位置を探す（失敗した場合は重複を許可）
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
            print(f"重複許可で配置: {Path(balloon_path).stem} (重複率: {min_overlap_ratio:.2f})")
        
        # 最終的にランダム位置に配置（フォールバック）
        if not placed:
            max_x = max(0, bg_w - new_balloon_w)
            max_y = max(0, bg_h - new_balloon_h)
            if max_x >= 0 and max_y >= 0:
                x = random.randint(0, max_x) if max_x > 0 else 0
                y = random.randint(0, max_y) if max_y > 0 else 0
                placed = True
                print(f"フォールバック配置: {Path(balloon_path).stem}")
        
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
            print(f"配置失敗: {Path(balloon_path).stem}（サイズが大きすぎます）")
    
    return result_img, result_mask, successfully_placed


def get_next_output_number(output_dir: str) -> int:
    """次の出力番号を取得"""
    os.makedirs(output_dir, exist_ok=True)
    
    existing_numbers = []
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            stem = Path(file).stem
            if stem.isdigit():
                existing_numbers.append(int(stem))
    
    return 1 if not existing_numbers else max(existing_numbers) + 1


def main():
    """メイン処理"""
    
    # パス設定
    balloons_dir = "generated_balloons"
    masks_dir = "masks"
    backgrounds_dir = "generated_backs"
    output_dir = "results"
    mask_output_dir = "results_mask"
    
    # 設定
    CFG = {
        "SCALE_RANGE": (0.1, 0.4),          # 吹き出しのスケール範囲
        "NUM_BALLOONS_RANGE": (2, 10),      # 配置する吹き出し数の範囲
        "IMAGES_PER_BACKGROUND": 50,         # 1背景画像あたりの生成数
        "MAX_ATTEMPTS": 200,                # 配置試行回数の上限
        "OVERLAP_THRESHOLD": 0.15,          # 重複許容率
    }
    
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)
    
    # 吹き出しとマスクの対応を取得
    balloon_mask_pairs = []
    
    print("吹き出し・マスクペアを検索中...")
    for balloon_file in os.listdir(balloons_dir):
        if balloon_file.endswith(('.png', '.jpg', '.jpeg')):
            balloon_path = os.path.join(balloons_dir, balloon_file)
            balloon_stem = Path(balloon_file).stem
            
            # 対応するマスクファイルを検索
            mask_file = f"{balloon_stem}_mask.png"
            mask_path = os.path.join(masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                balloon_mask_pairs.append((balloon_path, mask_path))
            else:
                print(f"警告: {balloon_file} に対応するマスク {mask_file} が見つかりません")
    
    # 背景画像を取得
    background_files = []
    for bg_file in os.listdir(backgrounds_dir):
        if bg_file.endswith(('.png', '.jpg', '.jpeg')):
            background_files.append(os.path.join(backgrounds_dir, bg_file))
    
    print(f"見つかった吹き出し: {len(balloon_mask_pairs)}個")
    print(f"見つかった背景: {len(background_files)}個")
    print(f"生成予定画像数: {len(background_files) * CFG['IMAGES_PER_BACKGROUND']}個")
    
    if len(balloon_mask_pairs) < CFG["NUM_BALLOONS_RANGE"][1]:
        print(f"警告: 利用可能な吹き出しが少ないです。最大配置数を{len(balloon_mask_pairs)}個に調整します。")
        CFG["NUM_BALLOONS_RANGE"] = (
            min(CFG["NUM_BALLOONS_RANGE"][0], len(balloon_mask_pairs)),
            len(balloon_mask_pairs)
        )
    
    # 次の番号を取得
    current_number = get_next_output_number(output_dir)
    success_count = 0
    
    print(f"\n合成開始（開始番号: {current_number:03d}）...")
    
    # 各背景画像について処理
    for bg_idx, bg_path in enumerate(tqdm(background_files, desc="背景処理中")):
        bg_name = Path(bg_path).stem
        
        # 1つの背景画像につき5枚生成
        for img_idx in range(CFG["IMAGES_PER_BACKGROUND"]):
            try:
                # ランダム複数合成実行
                result_img, result_mask, placed_balloons = composite_random_balloons_enhanced(
                    bg_path, 
                    balloon_mask_pairs,
                    scale_range=CFG["SCALE_RANGE"],
                    num_balloons_range=CFG["NUM_BALLOONS_RANGE"],
                    max_attempts=CFG["MAX_ATTEMPTS"]
                )
                
                # ファイル保存
                output_img_path = os.path.join(output_dir, f"{current_number:03d}.png")
                output_mask_path = os.path.join(mask_output_dir, f"{current_number:03d}_mask.png")
                
                cv2.imwrite(output_img_path, result_img)
                cv2.imwrite(output_mask_path, result_mask)
                
                balloon_names = ",".join(placed_balloons) if placed_balloons else "なし"
                print(f"✓ {current_number:03d}.png (背景:{bg_name}, 吹き出し{len(placed_balloons)}個: {balloon_names})")
                success_count += 1
                current_number += 1
                
            except Exception as e:
                print(f"✗ 合成失敗 (背景:{bg_name}, {img_idx+1}枚目): {e}")
    
    print(f"\n合成完了: {success_count}個の画像を生成しました")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"マスク出力ディレクトリ: {mask_output_dir}")


if __name__ == "__main__":
    main()
