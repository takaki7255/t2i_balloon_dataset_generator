"""
吹き出し画像からマスクを生成するスクリプト

機能:
1. 単一画像のマスク生成
2. ディレクトリ内の全画像に対する一括マスク生成
3. ファイル名の通し番号リネーム

使用例:
    # 単一ファイル
    python generate_balloon_masks.py --input balloon.png --output mask.png

    # ディレクトリ一括処理（デフォルト: balloons/images/ → balloons/masks/）
    python generate_balloon_masks.py --all

    # カスタムディレクトリ
    python generate_balloon_masks.py --all --input-dir my_balloons/ --output-dir my_masks/
"""

import os
import glob
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_mask(input_path: str, output_path: str | None = None) -> str:
    """
    入力画像の最外輪郭を塗りつぶしたマスクをPNGで保存
    
    背景が無地の吹き出し画像から最外輪郭を抽出し、
    同じサイズの二値マスク画像を生成する。
    
    Args:
        input_path: 入力画像のパス
        output_path: 出力マスクのパス（Noneで自動生成）
    
    Returns:
        保存したマスク画像のパス
    """
    # 1) 画像読み込み
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {input_path}")

    # 2) 前処理：グレースケール → ガウシアンぼかし
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) 二値化（背景が白っぽい場合は THRESH_BINARY_INV が無難）
    _, bin_img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4) 輪郭抽出（外側のみ）
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("No contours found!")

    # 最大面積 = 吹き出し本体
    balloon_cnt = max(contours, key=cv2.contourArea)

    # 5) マスク生成（白:255, 黒:0）
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [balloon_cnt], -1, color=255, thickness=cv2.FILLED)

    # 6) 保存
    if output_path is None:
        stem = Path(input_path).stem
        output_path = f"{stem}_mask.png"
    cv2.imwrite(output_path, mask)

    return output_path


def rename_images(directory: str) -> int:
    """
    指定ディレクトリ内の画像ファイルを通し番号にリネーム
    
    Args:
        directory: 画像ファイルがあるディレクトリパス
    
    Returns:
        リネームしたファイル数
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


def generate_all_masks(
    input_dir: str = "balloons/images/",
    output_dir: str = "balloons/masks/",
    rename: bool = True
):
    """
    ディレクトリ内の全画像に対してマスクを生成
    
    Args:
        input_dir: 入力画像ディレクトリ
        output_dir: 出力マスクディレクトリ
        rename: True の場合、先に画像を通し番号にリネーム
    """
    # 1. 画像ファイルを通し番号にリネーム（オプション）
    if rename:
        print("=== 画像ファイルのリネーム ===")
        num_renamed = rename_images(input_dir)
        
        if num_renamed == 0:
            print("処理する画像がありません")
            return
    
    # 2. 既存のマスクディレクトリがあれば削除して再作成
    print("=== マスクディレクトリの準備 ===")
    if os.path.exists(output_dir):
        print(f"既存のマスクディレクトリを削除: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"マスクディレクトリを作成: {output_dir}\n")
    
    # 3. マスク生成
    print("=== マスク生成開始 ===")
    processed = 0
    failed = 0
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_path = os.path.join(input_dir, filename)
            stem = Path(filename).stem
            mask_filename = f"{stem}_mask.png"
            output_path = os.path.join(output_dir, mask_filename)
            
            try:
                generate_mask(input_path, output_path)
                print(f"✓ マスク生成完了: {mask_filename}")
                processed += 1
            except Exception as e:
                print(f"✗ マスク生成失敗 ({filename}): {e}")
                failed += 1
    
    print(f"\n処理完了: {processed}個のマスクを生成, {failed}個が失敗")


def main():
    parser = argparse.ArgumentParser(
        description="吹き出し画像からマスクを生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一ファイルのマスク生成
  python generate_balloon_masks.py --input balloon.png --output mask.png

  # ディレクトリ一括処理（デフォルトディレクトリ）
  python generate_balloon_masks.py --all

  # カスタムディレクトリで一括処理
  python generate_balloon_masks.py --all --input-dir my_balloons/ --output-dir my_masks/

  # リネームなしで一括処理
  python generate_balloon_masks.py --all --no-rename
        """
    )
    
    # 単一ファイル用引数
    parser.add_argument("--input", "-i", type=str, help="入力画像ファイルパス")
    parser.add_argument("--output", "-o", type=str, help="出力マスクファイルパス")
    
    # 一括処理用引数
    parser.add_argument("--all", "-a", action="store_true", help="ディレクトリ内の全画像を処理")
    parser.add_argument("--input-dir", type=str, default="balloons/images/", 
                        help="入力画像ディレクトリ (default: balloons/images/)")
    parser.add_argument("--output-dir", type=str, default="balloons/masks/",
                        help="出力マスクディレクトリ (default: balloons/masks/)")
    parser.add_argument("--no-rename", action="store_true", 
                        help="ファイル名のリネームをスキップ")
    
    args = parser.parse_args()
    
    if args.all:
        # 一括処理モード
        generate_all_masks(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            rename=not args.no_rename
        )
    elif args.input:
        # 単一ファイルモード
        try:
            out = generate_mask(args.input, args.output)
            print(f"Mask saved to: {out}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        # 引数なしの場合はヘルプを表示
        parser.print_help()


if __name__ == "__main__":
    main()
