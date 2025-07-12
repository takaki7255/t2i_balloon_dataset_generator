import os
import random
import shutil
from glob import glob

# ===== 設定 =====
mask_root = "./../Manga109_released_2023_12_07/masks"         # マスクのルート（masks/作品名/カテゴリ名/*.png）
image_root = "./../Manga109_released_2023_12_07/images"       # 画像のルート
output_root = "real_dataset"     # 保存先
target_category = "balloon" # 処理対象カテゴリ
num_samples = 200

# ===== 出力フォルダ作成 =====
os.makedirs(os.path.join(output_root, "images"), exist_ok=True)
os.makedirs(os.path.join(output_root, "masks"), exist_ok=True)

# ===== balloonカテゴリのマスクを収集 =====
mask_paths = glob(os.path.join(mask_root, "*", target_category, "*_mask.png"))

# ===== ランダム抽出 =====
random.seed(42)
selected_masks = random.sample(mask_paths, min(num_samples, len(mask_paths)))

for idx, mask_path in enumerate(selected_masks):
    # パスを安全に分解
    class_dir = os.path.dirname(mask_path)                  # e.g., masks/Title/balloon
    title = os.path.basename(os.path.dirname(class_dir))    # "Title"
    mask_filename = os.path.basename(mask_path)             # "000_mask.png"

    # 対応する画像の推定
    img_filename = mask_filename.replace("_mask.png", ".jpg")
    image_path = os.path.join(image_root, title, img_filename)

    # 出力先ファイル名
    out_img_path = os.path.join(output_root, "images", f"{idx:03}.jpg")
    out_mask_path = os.path.join(output_root, "masks", f"{idx:03}_mask.png")

    # エラーチェック
    if not os.path.isfile(image_path):
        print(f"[スキップ] 元画像が存在しません: {image_path}")
        continue

    shutil.copy(image_path, out_img_path)
    shutil.copy(mask_path, out_mask_path)
    print(f"[{idx+1}] コピー完了: {image_path} + {mask_path}")

print("✅ balloonカテゴリのマスクペア抽出完了")
