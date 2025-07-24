"""
generated_balloons 内の全画像に対して SAM/SAM2 で吹き出しマスクを生成
1) Zero‑shot (=事前学習重みのまま) でも動く
2) sam_ckpt に自前 fine‑tune 重みを渡せば高精度化
"""

import os, cv2, torch, numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --------------------------- SAM 初期化 ----------------------------------- #
def build_mask_generator(
        sam_type="vit_h",            # "vit_l"/"vit_b" も可
        sam_ckpt="sam_vit_h_14_sam2.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"):
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device)
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=800  # 小ゴミ除去
    )

# ----------------------- Balloon Mask 選択 Heuristic ---------------------- #
def choose_balloon_mask(masks: list, image: np.ndarray) -> np.ndarray:
    """
    SAM が返した複数マスクから「吹き出しらしい」1枚を返す
    ・面積 / 明度 / 楕円率でスコア化
    """
    best, best_score = None, -1
    for m in masks:
        mask = m["segmentation"]          # bool H×W
        area = mask.sum()
        if area < 800:                    # 極小は除外
            continue
        # 平均明度 (吹き出し=白地を想定)
        mean_v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2][mask].mean()
        # 楕円率 (吹き出し≈楕円) …ラフに外接矩形で近似
        ys, xs = np.where(mask)
        if len(xs) < 10:
            continue
        w, h = xs.max() - xs.min(), ys.max() - ys.min()
        ellipse_ratio = min(w, h) / max(w, h)          # 1 に近いほど正円
        score = 0.6 * (area / image.size) + 0.3 * (mean_v / 255) + 0.1 * ellipse_ratio
        if score > best_score:
            best, best_score = mask, score
    if best is None:
        raise RuntimeError("Balloon mask not found")
    return (best * 255).astype(np.uint8)

# -------------------------- 1 枚処理関数 ---------------------------------- #
def generate_mask_sam(img_path: str, mask_path: str,
                      mask_generator: SamAutomaticMaskGenerator):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    masks = mask_generator.generate(img)
    mask = choose_balloon_mask(masks, img)
    cv2.imwrite(mask_path, mask)

# ------------------------ 全ファイルループ -------------------------------- #
def main():
    balloons_dir = "generated_balloons"
    masks_dir = "masks_sam"
    os.makedirs(masks_dir, exist_ok=True)

    mg = build_mask_generator()         # <- 1 回だけ初期化

    processed = skipped = 0
    for fn in os.listdir(balloons_dir):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        inp = os.path.join(balloons_dir, fn)
        out = os.path.join(masks_dir, f"{Path(fn).stem}_mask.png")
        if os.path.exists(out):
            print(f"スキップ: {fn}")
            skipped += 1
            continue
        try:
            generate_mask_sam(inp, out, mg)
            print(f"✓ {fn}")
            processed += 1
        except Exception as e:
            print(f"✗ {fn}: {e}")

    print(f"\n処理完了: {processed} 生成, {skipped} スキップ")

if __name__ == "__main__":
    main()
