"""
generated_balloons内の全ての画像に対してマスクを生成するスクリプト
"""

import os
import glob
import shutil
from pathlib import Path
from balloon_mask import generate_mask


def rename_images(directory):
    """
    指定ディレクトリ内の画像ファイルを通し番号にリネーム
    
    Args:
        directory (str): 画像ファイルがあるディレクトリパス
    
    Returns:
        int: リネームしたファイル数
    """
    if not os.path.exists(directory):
        print(f"ディレクトリ '{directory}' が見つかりません")
        return 0
    
    # 画像ファイルを取得
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    # ソートして順序を固定
    image_files.sort()
    
    # 桁数を計算（最低3桁）
    padding = max(3, len(str(len(image_files))))
    
    # リネーム実行
    for i, file_path in enumerate(image_files, 1):
        old_path = Path(file_path)
        new_name = f"{i:0{padding}d}{old_path.suffix}"
        new_path = old_path.parent / new_name
        
        # 同じファイル名の場合はスキップ
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"リネーム: {old_path.name} → {new_name}")
    
    print(f"リネーム完了: {len(image_files)}個のファイル\n")
    return len(image_files)


def generate_all_masks():
    """画像をリネームしてからマスクを生成"""
    
    balloons_dir = "balloons"
    masks_dir = "balloon_masks"
    
    # 1. まず画像ファイルを通し番号にリネーム
    print("=== 画像ファイルのリネーム ===")
    num_renamed = rename_images(balloons_dir)
    
    if num_renamed == 0:
        print("処理する画像がありません")
        return
    
    # 2. 既存のマスクディレクトリがあれば削除して再作成
    print("=== マスクディレクトリの準備 ===")
    if os.path.exists(masks_dir):
        print(f"既存のマスクディレクトリを削除: {masks_dir}")
        shutil.rmtree(masks_dir)
    os.makedirs(masks_dir, exist_ok=True)
    print(f"マスクディレクトリを作成: {masks_dir}\n")
    
    # 3. マスク生成
    print("=== マスク生成開始 ===")
    processed = 0
    failed = 0
    
    for filename in sorted(os.listdir(balloons_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_path = os.path.join(balloons_dir, filename)
            stem = Path(filename).stem
            mask_filename = f"{stem}_mask.png"
            output_path = os.path.join(masks_dir, mask_filename)
            
            try:
                generate_mask(input_path, output_path)
                print(f"✓ マスク生成完了: {mask_filename}")
                processed += 1
            except Exception as e:
                print(f"✗ マスク生成失敗 ({filename}): {e}")
                failed += 1
    
    print(f"\n処理完了: {processed}個のマスクを生成, {failed}個が失敗")

if __name__ == "__main__":
    generate_all_masks()
