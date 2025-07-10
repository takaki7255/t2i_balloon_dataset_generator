"""
吹き出し画像とマスクを背景画像に合成するスクリプト
"""

import cv2
import numpy as np
from pathlib import Path
import os
import random

def composite_balloon(background_path: str, balloon_path: str, mask_path: str, 
                     output_dir: str = "results", mask_output_dir: str = "results_mask",
                     scale_range: tuple = (0.1, 0.4), position: tuple = None) -> tuple:
    """
    背景画像に吹き出しとマスクを合成する
    
    Args:
        background_path: 背景画像のパス
        balloon_path: 吹き出し画像のパス
        mask_path: マスク画像のパス
        output_dir: 合成画像の出力ディレクトリ
        mask_output_dir: 合成マスクの出力ディレクトリ
        scale_range: 吹き出しのスケール範囲（最小, 最大）
        position: 合成位置 (x, y)。Noneの場合はランダムに配置
    
    Returns:
        (合成画像パス, 合成マスクパス)
    """
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)
    
    # 画像読み込み
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    balloon = cv2.imread(balloon_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if background is None or balloon is None or mask is None:
        raise FileNotFoundError("画像の読み込みに失敗しました")
    
    bg_h, bg_w = background.shape[:2]
    original_balloon_h, original_balloon_w = balloon.shape[:2]
    
    # 吹き出しのリサイズ（ランダムなスケール）
    balloon_scale = random.uniform(scale_range[0], scale_range[1])
    new_balloon_w = int(bg_w * balloon_scale)
    new_balloon_h = int(original_balloon_h * (new_balloon_w / original_balloon_w))
    
    # 吹き出しとマスクをリサイズ
    balloon_resized = cv2.resize(balloon, (new_balloon_w, new_balloon_h))
    mask_resized = cv2.resize(mask, (new_balloon_w, new_balloon_h))
    
    # 合成位置の決定（ランダム配置）
    if position is None:
        max_x = max(0, bg_w - new_balloon_w)
        max_y = max(0, bg_h - new_balloon_h)
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
    else:
        x, y = position
    
    # 画像範囲の調整
    x = max(0, min(x, bg_w - new_balloon_w))
    y = max(0, min(y, bg_h - new_balloon_h))
    
    # 合成画像とマスクの作成
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    # マスクを使用して吹き出しを合成
    mask_norm = mask_resized.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
    
    # 背景画像の該当領域を取得
    bg_region = result_img[y:y+new_balloon_h, x:x+new_balloon_w]
    
    # アルファブレンディング
    blended = balloon_resized.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
    result_img[y:y+new_balloon_h, x:x+new_balloon_w] = blended.astype(np.uint8)
    
    # マスクも合成
    result_mask[y:y+new_balloon_h, x:x+new_balloon_w] = mask_resized
    
    # ファイル名生成
    bg_name = Path(background_path).stem
    balloon_name = Path(balloon_path).stem
    
    output_img_path = os.path.join(output_dir, f"{bg_name}_{balloon_name}.png")
    output_mask_path = os.path.join(mask_output_dir, f"{bg_name}_{balloon_name}_mask.png")
    
    # 保存
    cv2.imwrite(output_img_path, result_img)
    cv2.imwrite(output_mask_path, result_mask)
    
    return output_img_path, output_mask_path


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


def composite_random_balloons(background_path: str, balloon_mask_pairs: list,
                              output_dir: str = "results", mask_output_dir: str = "results_mask",
                              scale_range: tuple = (0.15, 0.35), num_balloons: int = None,
                              max_attempts: int = 100) -> tuple:
    """
    1つの背景画像にランダムに選択した複数の吹き出しを重複なしで合成する
    
    Args:
        background_path: 背景画像のパス
        balloon_mask_pairs: [(balloon_path, mask_path), ...] のリスト
        output_dir: 合成画像の出力ディレクトリ
        mask_output_dir: 合成マスクの出力ディレクトリ
        scale_range: 吹き出しのスケール範囲（最小, 最大）
        num_balloons: 配置する吹き出し数（Noneの場合は2-4個からランダム）
        max_attempts: 配置試行回数の上限
    
    Returns:
        (合成画像パス, 合成マスクパス)
    """
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)
    
    # 背景画像読み込み
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")
    
    bg_h, bg_w = background.shape[:2]
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    # 配置する吹き出し数を決定
    if num_balloons is None:
        num_balloons = random.randint(2, min(4, len(balloon_mask_pairs)))
    else:
        num_balloons = min(num_balloons, len(balloon_mask_pairs))
    
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
        
        # リサイズ
        balloon_resized = cv2.resize(balloon, (new_balloon_w, new_balloon_h))
        mask_resized = cv2.resize(mask, (new_balloon_w, new_balloon_h))
        
        # 重複しない位置を探す
        placed = False
        for attempt in range(max_attempts):
            max_x = max(0, bg_w - new_balloon_w)
            max_y = max(0, bg_h - new_balloon_h)
            
            if max_x <= 0 or max_y <= 0:
                break
                
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # 新しい領域
            new_region = (x, y, x + new_balloon_w, y + new_balloon_h)
            
            # 既存領域との重複チェック
            overlap = False
            for occupied in occupied_regions:
                if regions_overlap(new_region, occupied):
                    # 重複率を計算
                    overlap_area = calculate_overlap_area(new_region, occupied)
                    new_area = new_balloon_w * new_balloon_h
                    overlap_ratio = overlap_area / new_area
                    
                    if overlap_ratio > 0.2:  # 20%以上の重複は避ける
                        overlap = True
                        break
            
            if not overlap:
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
                occupied_regions.append(new_region)
                successfully_placed.append(Path(balloon_path).stem)
                placed = True
                break
        
        if not placed:
            print(f"警告: {Path(balloon_path).stem} の配置に失敗しました（重複回避できず）")
    
    return result_img, result_mask, successfully_placed


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


def main():
    """メイン処理：合成モードを選択して実行"""
    
    # パス設定
    balloons_dir = "generated_balloons"
    masks_dir = "masks"
    backgrounds_dir = "generated_backs"
    output_dir = "results"
    mask_output_dir = "results_mask"
    
    # 吹き出しとマスクの対応を取得
    balloon_mask_pairs = []
    
    for balloon_file in os.listdir(balloons_dir):
        if balloon_file.endswith(('.png', '.jpg', '.jpeg')):
            balloon_path = os.path.join(balloons_dir, balloon_file)
            balloon_stem = Path(balloon_file).stem
            
            # 対応するマスクファイルを検索
            mask_file = f"{balloon_stem}_mask.png"
            mask_path = os.path.join(masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                balloon_mask_pairs.append((balloon_path, mask_path))
                print(f"対応ペア見つかりました: {balloon_file} <-> {mask_file}")
            else:
                print(f"警告: {balloon_file} に対応するマスク {mask_file} が見つかりません")
    
    # 背景画像を取得
    background_files = []
    for bg_file in os.listdir(backgrounds_dir):
        if bg_file.endswith(('.png', '.jpg', '.jpeg')):
            background_files.append(os.path.join(backgrounds_dir, bg_file))
    
    print(f"\n見つかった吹き出し: {len(balloon_mask_pairs)}個")
    print(f"見つかった背景: {len(background_files)}個")
    
    # 合成モード選択
    print("\n合成モードを選択してください:")
    print("1. 1対1合成 (全組み合わせ)")
    print("2. ランダム複数合成 (重複なし)")
    
    try:
        mode = input("モードを選択 (1 または 2): ").strip()
    except KeyboardInterrupt:
        print("\n処理を中断しました")
        return
    
    # 次の番号を取得
    next_number = get_next_output_number(output_dir)
    success_count = 0
    current_number = next_number
    
    if mode == "1":
        # 1対1合成モード
        print(f"合計合成数: {len(background_files) * len(balloon_mask_pairs)}個")
        
        for bg_path in background_files:
            for balloon_path, mask_path in balloon_mask_pairs:
                try:
                    # ナンバリング形式のファイル名
                    output_img_path = os.path.join(output_dir, f"{current_number:03d}.png")
                    output_mask_path = os.path.join(mask_output_dir, f"{current_number:03d}_mask.png")
                    
                    # 合成実行
                    img_path, mask_path_out = composite_balloon(
                        bg_path, balloon_path, mask_path, 
                        output_dir, mask_output_dir
                    )
                    
                    # ファイル名をナンバリング形式に変更
                    bg_name = Path(bg_path).stem
                    balloon_name = Path(balloon_path).stem
                    
                    # 生成されたファイルをリネーム
                    old_img_path = os.path.join(output_dir, f"{bg_name}_{balloon_name}.png")
                    old_mask_path = os.path.join(mask_output_dir, f"{bg_name}_{balloon_name}_mask.png")
                    
                    if os.path.exists(old_img_path):
                        os.rename(old_img_path, output_img_path)
                    if os.path.exists(old_mask_path):
                        os.rename(old_mask_path, output_mask_path)
                    
                    print(f"✓ 合成完了: {current_number:03d}.png (背景:{bg_name}, 吹き出し:{balloon_name})")
                    success_count += 1
                    current_number += 1
                    
                except Exception as e:
                    print(f"✗ 合成失敗 (背景:{Path(bg_path).stem}, 吹き出し:{Path(balloon_path).stem}): {e}")
    
    elif mode == "2":
        # ランダム複数合成モード
        try:
            num_images = int(input(f"生成する画像数を入力 (1-{len(background_files)*10}): ").strip())
        except (ValueError, KeyboardInterrupt):
            print("無効な入力です。デフォルトで各背景に1枚ずつ生成します。")
            num_images = len(background_files)
        
        print(f"ランダム複数合成で {num_images} 個の画像を生成します")
        
        for i in range(num_images):
            try:
                # ランダムに背景を選択
                bg_path = random.choice(background_files)
                
                # ランダム複数合成実行
                result_img, result_mask, placed_balloons = composite_random_balloons(
                    bg_path, balloon_mask_pairs
                )
                
                # ファイル保存
                output_img_path = os.path.join(output_dir, f"{current_number:03d}.png")
                output_mask_path = os.path.join(mask_output_dir, f"{current_number:03d}_mask.png")
                
                cv2.imwrite(output_img_path, result_img)
                cv2.imwrite(output_mask_path, result_mask)
                
                bg_name = Path(bg_path).stem
                balloon_names = ",".join(placed_balloons)
                print(f"✓ 合成完了: {current_number:03d}.png (背景:{bg_name}, 吹き出し:{balloon_names})")
                success_count += 1
                current_number += 1
                
            except Exception as e:
                print(f"✗ 合成失敗 (画像{i+1}): {e}")
    
    else:
        print("無効なモードです。1 または 2 を選択してください。")
        return
    
    print(f"\n合成完了: {success_count}個の画像を生成しました")


if __name__ == "__main__":
    main()
