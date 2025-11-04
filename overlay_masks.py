#python overlay_masks.py --root onomatopoeia_experiment_results --model ono-model-01
import os
import cv2
import numpy as np
import argparse
from glob import glob

# コマンドライン引数のパース
parser = argparse.ArgumentParser(description='マスクを赤くオーバーレイした画像を生成')
parser.add_argument('--model', type=str, default=None,
                    help='処理するモデル名（例: syn1000-corner-unet-01）。指定しない場合は全モデルを処理')
parser.add_argument('--root', type=str, default='balloon_experiment_results',
                    help='実験結果のルートディレクトリ（デフォルト: balloon_experiment_results）')
args = parser.parse_args()

experiment_root = args.root

# 処理対象のモデルディレクトリを決定
if args.model:
    # 特定のモデルのみ処理
    model_path = os.path.join(experiment_root, args.model)
    if not os.path.isdir(model_path):
        print(f"❌ エラー: モデルディレクトリが見つかりません: {model_path}")
        exit(1)
    model_dirs = [args.model]
    print(f"Processing single model: {args.model}\n")
else:
    # すべてのモデルディレクトリを検索
    model_dirs = [d for d in os.listdir(experiment_root) 
                  if os.path.isdir(os.path.join(experiment_root, d)) 
                  and not d.endswith('.txt')]
    print(f"Found {len(model_dirs)} model directories to process\n")

for model_name in model_dirs:
    model_path = os.path.join(experiment_root, model_name)
    image_dir = os.path.join(model_path, "images")
    mask_dir = os.path.join(model_path, "predicts")
    output_dir = os.path.join(model_path, "overlay_soft_red")
    
    # images と predicts フォルダが存在するか確認
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"⚠️  Skipping {model_name}: missing images or predicts folder")
        continue
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing: {model_name}")
    
    processed_count = 0
    for img_path in glob(os.path.join(image_dir, "*.png")):
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        mask_path = os.path.join(mask_dir, f"{name}_pred.png")

        if not os.path.exists(mask_path):
            print(f"  ⚠️  対応するマスクが見つかりません: {filename}")
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
        processed_count += 1
    
    print(f"  ✅ Processed {processed_count} images -> {output_dir}\n")

print("All models processed!")
