"""
real フォルダの画像とマスクを train / val に分割して
real_dataset/ 以下へコピーするスクリプト
"""

import random, shutil
from pathlib import Path

# -------------------------------------------------------------
CFG = {
    "IMG_DIR":   Path("real/images"),      # 元画像ディレクトリ
    "MASK_DIR":  Path("real/masks"),       # マスクディレクトリ
    "OUT_DIR":   Path("real_dataset"),     # 出力先ルート
    "SPLIT":     {"train":0.8, "val":0.2}, # train:val = 8:2
    "SEED":      42,
}
# -------------------------------------------------------------

def main(cfg):
    random.seed(cfg["SEED"])

    # jpg画像を取得
    img_paths = sorted(cfg["IMG_DIR"].glob("*.jpg"))
    print(f"元画像数: {len(img_paths)} 枚")
    
    assert 0.999 < sum(cfg["SPLIT"].values()) < 1.001, "split の合計は 1 にしてください"

    # ---- シャッフルして分割 --------------------------------------------------
    random.shuffle(img_paths)
    n = len(img_paths)
    splits = {k:int(v*n) for k,v in cfg["SPLIT"].items()}
    # 端数調整（合計が n になるよう最後に train に寄せる）
    diff = n - sum(splits.values()); splits["train"] += diff

    idx = 0; buckets = {}
    for split, cnt in splits.items():
        buckets[split] = img_paths[idx: idx+cnt]
        idx += cnt

    # ---- 出力フォルダ作成 ----------------------------------------------------
    for split in cfg["SPLIT"]:
        (cfg["OUT_DIR"]/split/"images").mkdir(parents=True, exist_ok=True)
        (cfg["OUT_DIR"]/split/"masks").mkdir(parents=True, exist_ok=True)

    # ---- コピー --------------------------------------------------------------
    for split, paths in buckets.items():
        for img_path in paths:
            stem = img_path.stem  # 例: "000"
            mask_path = cfg["MASK_DIR"]/f"{stem}_mask.png"
            if not mask_path.exists():
                print(f"[warn] マスクが見つかりません: {mask_path}")
                continue

            dst_img  = cfg["OUT_DIR"]/split/"images"/img_path.name
            dst_mask = cfg["OUT_DIR"]/split/"masks"/mask_path.name
            shutil.copy2(img_path,  dst_img)
            shutil.copy2(mask_path, dst_mask)

    # ---- レポート ------------------------------------------------------------
    for split in cfg["SPLIT"]:
        cnt = len(list((cfg["OUT_DIR"]/split/"images").glob("*.jpg")))
        print(f"{split:<5}: {cnt} ペア")

if __name__ == "__main__":
    main(CFG)
