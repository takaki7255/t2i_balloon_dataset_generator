"""
real_dataset と syn_dataset のすべての画像・マスクを統合し、
8:2 に分割して synreal_dataset を生成するスクリプト
"""

import random, shutil
from pathlib import Path
import os

# -------------------------------------------------------------
CFG = {
    "REAL_ROOT":   Path("real_dataset"),     # realデータセットのルート
    "SYN_ROOT":    Path("syn_dataset"),      # synデータセットのルート
    "OUT_DIR":     Path("synreal_dataset"),  # 出力先ルート
    "SPLIT":       {"train":0.8, "val":0.2},
    "SEED":        42,
}
# -------------------------------------------------------------

def collect_image_mask_pairs(dataset_root: Path) -> list:
    """
    データセットから画像・マスクペアを収集する
    train/val フォルダが存在する場合は両方から収集
    """
    image_mask_pairs = []
    
    # train フォルダから収集
    train_images_dir = dataset_root / "train" / "images"
    train_masks_dir = dataset_root / "train" / "masks"
    if train_images_dir.exists() and train_masks_dir.exists():
        for img_ext in ["*.png", "*.jpg", "*.jpeg"]:
            for img_path in train_images_dir.glob(img_ext):
                stem = img_path.stem
                mask_path = train_masks_dir / f"{stem}_mask.png"
                if mask_path.exists():
                    image_mask_pairs.append((str(img_path), str(mask_path)))
                else:
                    print(f"[warn] マスクが見つかりません: {mask_path}")
    
    # val フォルダから収集
    val_images_dir = dataset_root / "val" / "images"
    val_masks_dir = dataset_root / "val" / "masks"
    if val_images_dir.exists() and val_masks_dir.exists():
        for img_ext in ["*.png", "*.jpg", "*.jpeg"]:
            for img_path in val_images_dir.glob(img_ext):
                stem = img_path.stem
                mask_path = val_masks_dir / f"{stem}_mask.png"
                if mask_path.exists():
                    image_mask_pairs.append((str(img_path), str(mask_path)))
                else:
                    print(f"[warn] マスクが見つかりません: {mask_path}")
    
    return image_mask_pairs

def get_next_number(existing_pairs: list) -> int:
    """既存のペアから次の番号を取得"""
    if not existing_pairs:
        return 1
    
    max_num = 0
    for img_path, _ in existing_pairs:
        stem = Path(img_path).stem
        if stem.isdigit():
            max_num = max(max_num, int(stem))
    
    return max_num + 1

def main(cfg):
    random.seed(cfg["SEED"])
    
    print("画像・マスクペアを収集中...")
    
    # real_dataset から収集
    real_pairs = collect_image_mask_pairs(cfg["REAL_ROOT"])
    print(f"real_dataset から {len(real_pairs)} ペア収集")
    
    # syn_dataset から収集
    syn_pairs = collect_image_mask_pairs(cfg["SYN_ROOT"])
    print(f"syn_dataset から {len(syn_pairs)} ペア収集")
    
    # 統合
    all_pairs = real_pairs + syn_pairs
    print(f"統合後: {len(all_pairs)} ペア")
    
    if not all_pairs:
        print("エラー: 画像・マスクペアが見つかりません")
        return
    
    assert 0.999 < sum(cfg["SPLIT"].values()) < 1.001, "split の合計は 1 にしてください"
    
    # ---- シャッフルして分割 --------------------------------------------------
    random.shuffle(all_pairs)
    n = len(all_pairs)
    splits = {k: int(v * n) for k, v in cfg["SPLIT"].items()}
    # 端数調整（合計が n になるよう最後に train に寄せる）
    diff = n - sum(splits.values())
    splits["train"] += diff
    
    idx = 0
    buckets = {}
    for split, cnt in splits.items():
        buckets[split] = all_pairs[idx: idx + cnt]
        idx += cnt
    
    # ---- 出力フォルダ作成 ----------------------------------------------------
    for split in cfg["SPLIT"]:
        (cfg["OUT_DIR"] / split / "images").mkdir(parents=True, exist_ok=True)
        (cfg["OUT_DIR"] / split / "masks").mkdir(parents=True, exist_ok=True)
    
    # ---- コピー（連番でリネーム） --------------------------------------------
    current_number = 1
    
    for split, pairs in buckets.items():
        print(f"\n{split} データセットを作成中...")
        
        for img_path, mask_path in pairs:
            # 拡張子を取得
            img_ext = Path(img_path).suffix
            
            # 新しいファイル名（連番）
            new_img_name = f"{current_number:03d}{img_ext}"
            new_mask_name = f"{current_number:03d}_mask.png"
            
            # コピー先パス
            dst_img = cfg["OUT_DIR"] / split / "images" / new_img_name
            dst_mask = cfg["OUT_DIR"] / split / "masks" / new_mask_name
            
            # コピー実行
            try:
                shutil.copy2(img_path, dst_img)
                shutil.copy2(mask_path, dst_mask)
                print(f"✓ {current_number:03d} (元: {Path(img_path).name})")
                current_number += 1
            except Exception as e:
                print(f"✗ コピー失敗 {Path(img_path).name}: {e}")
    
    # ---- レポート ------------------------------------------------------------
    print(f"\n=== 統合データセット作成完了 ===")
    print(f"出力先: {cfg['OUT_DIR']}")
    for split in cfg["SPLIT"]:
        img_count = len(list((cfg["OUT_DIR"] / split / "images").glob("*")))
        mask_count = len(list((cfg["OUT_DIR"] / split / "masks").glob("*")))
        print(f"{split:<5}: {img_count} 画像, {mask_count} マスク")
    
    # 元データセットの詳細
    print(f"\n=== 元データセット内訳 ===")
    print(f"real_dataset: {len(real_pairs)} ペア")
    print(f"syn_dataset:  {len(syn_pairs)} ペア")
    print(f"合計:        {len(all_pairs)} ペア")

if __name__ == "__main__":
    main(CFG)
