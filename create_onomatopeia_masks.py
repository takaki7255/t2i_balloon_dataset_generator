"""
オノマトペ画像に対してマスクを生成するスクリプト
全ての文字領域を面積フィルタリングでマスク化
"""

import os
import glob
import shutil
import cv2
import numpy as np
from pathlib import Path


def generate_onomatopoeia_mask(input_path: str, output_path: str, min_area: int = 100) -> str:
    """
    オノマトペ画像の全ての輪郭を塗りつぶしたマスクを PNG で保存
    
    Args:
        input_path: 入力画像のパス
        output_path: 出力マスクのパス
        min_area: 残す輪郭の最小面積（デフォルト100ピクセル）
    
    Returns:
        str: 保存したマスクファイルのパス
    """
    # 1) 画像読み込み
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {input_path}")

    # 2) 前処理：グレースケール → ガウシアンぼかし
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) 二値化（背景が白っぽい場合は THRESH_BINARY_INV）
    _, bin_img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4) 輪郭抽出（階層構造を取得）
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("No contours found!")

    # 5) 面積フィルタリング + 穴の除外
    # hierarchy[0][i] = [next, previous, first_child, parent]
    # parent == -1 → 外側の輪郭（文字の外側）
    # parent != -1 → 内側の輪郭（文字の穴）
    filtered_contours = []
    for i, cnt in enumerate(contours):
        # 外側の輪郭のみを対象
        if hierarchy[0][i][3] == -1:  # parent == -1
            area = cv2.contourArea(cnt)
            if area >= min_area:
                filtered_contours.append(cnt)
    
    if not filtered_contours:
        raise RuntimeError(f"No contours with area >= {min_area} found!")
    
    print(f"  検出輪郭数: {len(contours)} → 外側の輪郭: {len([h for h in hierarchy[0] if h[3] == -1])} → フィルタ後: {len(filtered_contours)} (最小面積: {min_area})")

    # 6) マスク生成：全ての輪郭を描画（白:255, 黒:0）
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, filtered_contours, -1, color=255, thickness=cv2.FILLED)

    # 7) 保存
    cv2.imwrite(output_path, mask)

    return output_path


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

    balloons_dir = "onomatopeias"
    masks_dir = "onomatopeia_masks"
    
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
                generate_onomatopoeia_mask(input_path, output_path, min_area=100)
                print(f"✓ マスク生成完了: {mask_filename}")
                processed += 1
            except Exception as e:
                print(f"✗ マスク生成失敗 ({filename}): {e}")
                failed += 1
    
    print(f"\n処理完了: {processed}個のマスクを生成, {failed}個が失敗")

if __name__ == "__main__":
    generate_all_masks()
