"""
背景が無地の吹き出し画像から最外輪郭を抽出し，
同じサイズの二値マスク画像 (PNG) を生成するスクリプト。
"""

import cv2
import numpy as np
from pathlib import Path

def generate_mask(input_path: str, output_path: str | None = None) -> str:
    """入力画像の最外輪郭を塗りつぶしたマスクを PNG で保存し，パスを返す"""
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


if __name__ == "__main__":
    # 入力ファイルと出力ファイルをここで指定
    input_file = "./generated_image/sample.png"  # 処理したい画像ファイルのパスを指定
    output_file = "output_mask.png"  # 出力マスクファイルのパスを指定（Noneで自動生成）
    
    try:
        out = generate_mask(input_file, output_file)
        print(f"Mask saved to: {out}")
    except Exception as e:
        print(f"Error: {e}")
