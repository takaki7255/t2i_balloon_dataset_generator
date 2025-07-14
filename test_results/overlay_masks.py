import os
import cv2
import numpy as np
from glob import glob

model_name = "synreal-unet-01"
image_dir = os.path.join(model_name, "images")
mask_dir = os.path.join(model_name, "predicts")
output_dir = os.path.join(model_name, "overlay")
os.makedirs(output_dir, exist_ok=True)

for img_path in glob(os.path.join(image_dir, "*.png")):
    filename = os.path.basename(img_path)
    name, _ = os.path.splitext(filename)
    mask_path = os.path.join(mask_dir, f"{name}_pred.png")

    if not os.path.exists(mask_path):
        print(f"対応するマスクが見つかりません: {mask_path}")
        continue

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 輪郭を検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # マスクの赤色半透明塗りつぶし
    overlay = image.copy()
    fill_color = (0, 0, 255)  # 赤（BGR）

    for contour in contours:
        # 塗りつぶし（透明度 0.3）
        mask_fill = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(mask_fill, [contour], -1, fill_color, thickness=cv2.FILLED)
        overlay = cv2.addWeighted(overlay, 1.0, mask_fill, 0.5, 0)

        # 輪郭線（赤で強調）
        cv2.drawContours(overlay, [contour], -1, (0, 0, 255), thickness=1)

    out_path = os.path.join(output_dir, f"{name}_overlay.png")
    cv2.imwrite(out_path, overlay)
    print(f"保存しました: {out_path}")
