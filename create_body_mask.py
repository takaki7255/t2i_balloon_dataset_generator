"""
人物画像に対してマスクを生成するスクリプト
全ての文字領域を面積フィルタリングでマスク化
"""

import os
import glob
import shutil
import cv2
import numpy as np
from pathlib import Path


def generate_body_mask(input_path: str, output_path: str, min_area: int = 10) -> str:
    """
    人物画像の全ての輪郭を塗りつぶしたマスクを PNG で保存
    
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

    # 元の画像サイズを保存
    original_shape = img.shape[:2]

    # 2) パディングを追加（画像端の輪郭を検出するため）
    padding = 10
    img_padded = cv2.copyMakeBorder(
        img, padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)  # 白でパディング
    )

    # 3) 前処理：グレースケール → ガウシアンぼかし
    gray = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4) 二値化（背景が白っぽい場合は THRESH_BINARY_INV）
    _, bin_img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 5) モルフォロジー演算（膨張→収縮）で輪郭を閉じる
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6) 輪郭抽出（輪郭を近似せずに全ての点を保持）
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise RuntimeError("No contours found!")

    # 7) 面積フィルタリング
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            filtered_contours.append(cnt)
    
    if not filtered_contours:
        raise RuntimeError(f"No contours with area >= {min_area} found!")
    
    print(f"  検出輪郭数: {len(contours)} → フィルタ後: {len(filtered_contours)} (最小面積: {min_area})")

    # 8) マスク生成：全ての輪郭を詳細に塗りつぶす（白:255, 黒:0）
    mask_padded = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask_padded, filtered_contours, -1, color=255, thickness=cv2.FILLED)

    # 9) パディングを削除して元のサイズに戻す
    mask = mask_padded[padding:-padding, padding:-padding]

    # 10) 保存
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

    bodies_dir = "bodies"
    masks_dir = "body_masks"
    
    # 1. まず画像ファイルを通し番号にリネーム
    print("=== 画像ファイルのリネーム ===")
    num_renamed = rename_images(bodies_dir)
    
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
    
    for filename in sorted(os.listdir(bodies_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_path = os.path.join(bodies_dir, filename)
            stem = Path(filename).stem
            mask_filename = f"{stem}_mask.png"
            output_path = os.path.join(masks_dir, mask_filename)
            
            try:
                generate_body_mask(input_path, output_path, min_area=10)
                print(f"✓ マスク生成完了: {mask_filename}")
                processed += 1
            except Exception as e:
                print(f"✗ マスク生成失敗 ({filename}): {e}")
                failed += 1
    
    print(f"\n処理完了: {processed}個のマスクを生成, {failed}個が失敗")

if __name__ == "__main__":
    generate_all_masks()
