"""
ファインチューニング用の小規模な実漫画画像データセットを作成するスクリプト
Manga109から少数のballoonサンプルを抽出してtrain/val構造で保存
"""

import os
import random
import shutil
from glob import glob
from pathlib import Path

# ===== 設定 =====
mask_root = "./../Manga109/masks"         # マスクのルート（masks/作品名/カテゴリ名/*.png）
image_root = "./../Manga109/images"       # 画像のルート
target_category = "balloon"               # 処理対象カテゴリ

# ファインチューニング用データセット設定
FINETUNE_CONFIG = {
    "output_root": "finetune100_dataset",     # 出力フォルダ名
    "total_samples": 100,                   # 総サンプル数（小規模）
    "train_ratio": 0.8,                    # train:val = 8:2
    "seed": 18,                            # ランダムシード
}

def create_finetune_dataset():
    """
    ファインチューニング用の小規模データセットを作成
    """
    config = FINETUNE_CONFIG
    output_root = config["output_root"]
    total_samples = config["total_samples"]
    train_ratio = config["train_ratio"]
    
    print(f"=== ファインチューニング用データセット作成 ===")
    print(f"出力先: {output_root}")
    print(f"総サンプル数: {total_samples}")
    print(f"train:val = {train_ratio:.0%}:{1-train_ratio:.0%}")
    
    # 出力フォルダ作成
    train_img_dir = os.path.join(output_root, "train", "images")
    train_mask_dir = os.path.join(output_root, "train", "masks")
    val_img_dir = os.path.join(output_root, "val", "images")
    val_mask_dir = os.path.join(output_root, "val", "masks")
    
    for dir_path in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # balloonカテゴリのマスクを収集
    print("\nballoon カテゴリのマスクを検索中...")
    mask_paths = glob(os.path.join(mask_root, "*", target_category, "*_mask.png"))
    
    if not mask_paths:
        print("エラー: balloon カテゴリのマスクが見つかりません")
        print(f"検索パス: {os.path.join(mask_root, '*', target_category, '*_mask.png')}")
        return
    
    print(f"見つかったマスク数: {len(mask_paths)}")
    
    if len(mask_paths) < total_samples:
        print(f"警告: 利用可能なマスク数（{len(mask_paths)}）が必要数（{total_samples}）より少ないです")
        total_samples = len(mask_paths)
    
    # ランダムシード設定
    random.seed(config["seed"])
    
    # ランダムにサンプルを選択
    selected_masks = random.sample(mask_paths, total_samples)
    
    # train/val 分割
    train_count = int(total_samples * train_ratio)
    train_masks = selected_masks[:train_count]
    val_masks = selected_masks[train_count:]
    
    print(f"\ntrain: {len(train_masks)}枚")
    print(f"val: {len(val_masks)}枚")
    
    # データセット作成
    def copy_samples(mask_list, split_name, img_dir, mask_dir):
        """指定されたsplit用のサンプルをコピー"""
        success_count = 0
        
        for idx, mask_path in enumerate(mask_list):
            # パスを分解
            class_dir = os.path.dirname(mask_path)                  # masks/Title/balloon
            title = os.path.basename(os.path.dirname(class_dir))    # Title
            mask_filename = os.path.basename(mask_path)             # 000_mask.png

            # 対応する画像の推定
            img_filename = mask_filename.replace("_mask.png", ".jpg")
            image_path = os.path.join(image_root, title, img_filename)

            # 出力先ファイル名
            out_img_path = os.path.join(img_dir, f"{idx+1:03d}.jpg")
            out_mask_path = os.path.join(mask_dir, f"{idx+1:03d}_mask.png")

            # 画像存在チェック
            if not os.path.isfile(image_path):
                print(f"[{split_name}] スキップ: 元画像が存在しません - {image_path}")
                continue

            try:
                # コピー実行
                shutil.copy(image_path, out_img_path)
                shutil.copy(mask_path, out_mask_path)
                print(f"[{split_name}:{idx+1:03d}] コピー完了: {title}/{img_filename}")
                success_count += 1
            except Exception as e:
                print(f"[{split_name}] エラー: コピー失敗 {image_path}: {e}")
        
        return success_count
    
    # train データ作成
    print(f"\n=== train データ作成中 ===")
    train_success = copy_samples(train_masks, "train", train_img_dir, train_mask_dir)
    
    # val データ作成
    print(f"\n=== val データ作成中 ===")
    val_success = copy_samples(val_masks, "val", val_img_dir, val_mask_dir)
    
    # 最終レポート
    print(f"\n=== ファインチューニング用データセット作成完了 ===")
    print(f"出力先: {output_root}")
    print(f"train: {train_success}枚")
    print(f"val: {val_success}枚")
    print(f"総計: {train_success + val_success}枚")
    
    # フォルダ構造確認
    print(f"\n=== フォルダ構造 ===")
    for split in ["train", "val"]:
        img_count = len(list(Path(output_root).glob(f"{split}/images/*.jpg")))
        mask_count = len(list(Path(output_root).glob(f"{split}/masks/*_mask.png")))
        print(f"{split}/")
        print(f"  images/: {img_count} files")
        print(f"  masks/:  {mask_count} files")
    
    # ファインチューニング用の推奨設定を表示
    print(f"\n=== ファインチューニング推奨設定 ===")
    print(f'CFG = {{')
    print(f'    "ROOT": Path("{output_root}"),')
    print(f'    "BATCH": 2,                    # 小規模なので小さなバッチサイズ')
    print(f'    "EPOCHS": 20,                  # 少ないエポック数')
    print(f'    "LR": 1e-5,                    # 小さな学習率')
    print(f'    "PATIENCE": 5,                 # 早期終了')
    print(f'}}')
    
    return output_root

def main():
    """メイン処理"""
    
    # 既存の使用済みデータをチェック（オプション）
    existing_datasets = ["real_dataset", "test_dataset"]
    used_masks = set()
    
    for dataset_name in existing_datasets:
        if os.path.exists(dataset_name):
            print(f"既存データセット検出: {dataset_name}")
            # 既存データセットで使用されたマスクを特定（重複回避）
            # この部分は必要に応じて実装
    
    # ファインチューニング用データセット作成
    output_path = create_finetune_dataset()
    
    print(f"\n✅ ファインチューニング用データセット作成完了")
    print(f"次のステップ:")
    print(f"1. finetune_unet.py の ROOT を '{output_path}' に設定")
    print(f"2. 学習実行: python finetune_unet.py")

if __name__ == "__main__":
    main()
