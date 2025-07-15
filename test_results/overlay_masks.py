import os
import cv2
import numpy as np
from glob import glob

model_name = "synreal-unet-01"
image_dir = os.path.join(model_name, "images")
mask_dir = os.path.join(model_name, "predicts")
output_dir = os.path.join(model_name, "overlay_soft_red")
os.makedirs(output_dir, exist_ok=True)

for img_path in glob(os.path.join(image_dir, "*.png")):
    filename = os.path.basename(img_path)
    name, _ = os.path.splitext(filename)
    mask_path = os.path.join(mask_dir, f"{name}_pred.png")

    if not os.path.exists(mask_path):
        print(f"対応するマスクが見つかりません: {mask_path}")
        continue

    # 元画像・マスク読み込み
    image = cv2.imread(img_path).astype(np.float32)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # マスク領域
    mask_area = mask == 255

    # 赤系に色調変更
    red_tinted = image.copy()
    red_tinted[mask_area, 1] *= 0.5  # Gを50%
    red_tinted[mask_area, 0] *= 0.5  # Bを50%

    # αブレンド（透過率70%）
    alpha = 0.7
    blended = (image * (1 - alpha) + red_tinted * alpha).astype(np.uint8)

    # 保存
    output_path = os.path.join(output_dir, f"{name}_overlay.png")
    cv2.imwrite(output_path, blended)
    print(f"保存しました: {output_path}")
