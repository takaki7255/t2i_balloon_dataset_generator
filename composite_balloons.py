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
                     balloon_scale: float = 0.3, position: tuple = None) -> tuple:
    """
    背景画像に吹き出しとマスクを合成する
    
    Args:
        background_path: 背景画像のパス
        balloon_path: 吹き出し画像のパス
        mask_path: マスク画像のパス
        output_dir: 合成画像の出力ディレクトリ
        mask_output_dir: 合成マスクの出力ディレクトリ
        balloon_scale: 吹き出しのスケール（背景画像に対する比率）
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
    
    # 吹き出しのリサイズ（背景画像のサイズに基づいて）
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


def main():
    """メイン処理：全ての組み合わせで合成を実行"""
    
    # パス設定
    balloons_dir = "generated_balloons"
    masks_dir = "masks"
    backgrounds_dir = "generated_backs"
    
    # 背景画像とマスクの対応を取得
    balloon_files = []
    mask_files = []
    
    for balloon_file in os.listdir(balloons_dir):
        if balloon_file.endswith(('.png', '.jpg', '.jpeg')):
            balloon_path = os.path.join(balloons_dir, balloon_file)
            balloon_stem = Path(balloon_file).stem
            
            # 対応するマスクファイルを検索
            mask_file = f"{balloon_stem}_mask.png"
            mask_path = os.path.join(masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                balloon_files.append(balloon_path)
                mask_files.append(mask_path)
                print(f"対応ペア見つかりました: {balloon_file} <-> {mask_file}")
            else:
                print(f"警告: {balloon_file} に対応するマスク {mask_file} が見つかりません")
    
    # 背景画像を取得
    background_files = []
    for bg_file in os.listdir(backgrounds_dir):
        if bg_file.endswith(('.png', '.jpg', '.jpeg')):
            background_files.append(os.path.join(backgrounds_dir, bg_file))
    
    print(f"\n見つかった吹き出し: {len(balloon_files)}個")
    print(f"見つかった背景: {len(background_files)}個")
    print(f"合計合成数: {len(balloon_files) * len(background_files)}個")
    
    # 全ての組み合わせで合成実行
    success_count = 0
    for bg_path in background_files:
        for balloon_path, mask_path in zip(balloon_files, mask_files):
            try:
                img_path, mask_path_out = composite_balloon(bg_path, balloon_path, mask_path)
                print(f"✓ 合成完了: {img_path}")
                success_count += 1
            except Exception as e:
                print(f"✗ 合成失敗 ({bg_path}, {balloon_path}): {e}")
    
    print(f"\n合成完了: {success_count}個の画像を生成しました")


if __name__ == "__main__":
    main()
