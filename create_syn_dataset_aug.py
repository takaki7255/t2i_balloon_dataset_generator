"""
吹き出し画像+マスクを背景に合成してシンセ／拡張データセットを生成するスクリプト
Augmentation:  rotation / gaussian noise / cut‑out
"""

import cv2
import numpy as np
from pathlib import Path
import os
import random
import shutil
from typing import Tuple, List

# ----------  augmentation -------------------------------------------------- #
def augment_balloon_and_mask(
        balloon: np.ndarray,
        mask: np.ndarray,
        cfg: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    • ランダム回転
    • マスク領域にだけガウスノイズ
    • マスク領域ランダムカットアウト
    """
    h, w = mask.shape
    # --- 1) rotation -------------------------------------------------------- #
    angle = random.uniform(*cfg["ROTATION_RANGE"])
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 回転後の画像サイズを計算
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    balloon = cv2.warpAffine(
        balloon,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255)   # 白背景
    )
    mask = cv2.warpAffine(
        mask,
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )

    # --- 2) gaussian noise -------------------------------------------------- #
    if random.random() < cfg["NOISE_PROB"]:
        sigma = random.uniform(*cfg["NOISE_SIGMA"])
        noise = np.random.normal(0, sigma, balloon.shape).astype(np.float32)
        # マスク外はノイズをゼロ
        noise[mask == 0] = 0
        balloon = np.clip(balloon.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # --- 3) random cut‑out -------------------------------------------------- #
    if random.random() < cfg["CUTOUT_PROB"]:
        # マスク領域がある座標を抽出
        coords = np.column_stack(np.where(mask > 0))
        if len(coords):
            yx = coords[random.randint(0, len(coords) - 1)]
            cut_w = int(random.uniform(*cfg["CUTOUT_RATIO"]) * new_w)
            cut_h = int(random.uniform(*cfg["CUTOUT_RATIO"]) * new_h)
            x0 = max(0, yx[1] - cut_w // 2)
            y0 = max(0, yx[0] - cut_h // 2)
            x1 = min(new_w, x0 + cut_w)
            y1 = min(new_h, y0 + cut_h)
            balloon[y0:y1, x0:x1] = 255   # 白で塗りつぶし
            mask[y0:y1, x0:x1] = 0

    return balloon, mask
# --------------------------------------------------------------------------- #

def regions_overlap(region1: tuple, region2: tuple) -> bool:
    x1_min, y1_min, x1_max, y1_max = region1
    x2_min, y2_min, x2_max, y2_max = region2
    return not (x1_max <= x2_min or x2_max <= x1_min or
                y1_max <= y2_min or y2_max <= y1_min)


def calculate_overlap_area(region1: tuple, region2: tuple) -> int:
    x1_min, y1_min, x1_max, y1_max = region1
    x2_min, y2_min, x2_max, y2_max = region2
    dx = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    dy = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    return dx * dy

# -------------- 合成本体（augmentation呼び出しを追加） -------------------- #
def composite_random_balloons_enhanced(
        background_path: str,
        balloon_mask_pairs: List[Tuple[str, str]],
        scale_range: Tuple[float, float] = (0.1, 0.4),
        num_balloons_range: Tuple[int, int] = (2, 10),
        max_attempts: int = 200,
        aug_cfg: dict = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:

    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")

    bg_h, bg_w = background.shape[:2]
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)

    min_b, max_b = num_balloons_range
    max_b = min(max_b, len(balloon_mask_pairs))
    num_balloons = random.randint(min_b, max_b)

    selected_pairs = random.sample(balloon_mask_pairs, num_balloons)
    occupied_regions, placed_names = [], []

    for balloon_path, mask_path in selected_pairs:
        balloon = cv2.imread(balloon_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if balloon is None or mask is None:
            continue

        # ----------- augmentation here ------------------------------------- #
        balloon, mask = augment_balloon_and_mask(balloon, mask, aug_cfg)
        if mask.sum() == 0:          # マスクが消滅したらスキップ
            continue
        # ------------------------------------------------------------------ #

        # スケール後のサイズ計算
        bh, bw = balloon.shape[:2]
        scale = random.uniform(*scale_range)
        new_w = int(bg_w * scale)
        new_h = int(bh * (new_w / bw))
        if new_w >= bg_w or new_h >= bg_h:
            continue

        balloon_r = cv2.resize(balloon, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # --- 位置探索（既存ロジックそのまま） ----------------------------- #
        placed = False
        best_pos, min_ovr = None, float("inf")
        for _ in range(max_attempts // 2):
            max_x = bg_w - new_w
            max_y = bg_h - new_h
            if max_x <= 0 or max_y <= 0:
                break
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            region = (x, y, x + new_w, y + new_h)
            max_ratio = 0
            for occ in occupied_regions:
                if regions_overlap(region, occ):
                    ratio = calculate_overlap_area(region, occ) / (new_w * new_h)
                    max_ratio = max(max_ratio, ratio)
            if max_ratio <= 0.15:
                best_pos, placed = (x, y), True
                break
            if max_ratio < min_ovr:
                min_ovr, best_pos = max_ratio, (x, y)

        if not placed and best_pos:
            x, y = best_pos
            placed = True

        if not placed:
            continue

        # --- 合成 ---------------------------------------------------------- #
        alpha = (mask_r.astype(np.float32) / 255.0)[:, :, None]
        roi = result_img[y:y+new_h, x:x+new_w]
        result_img[y:y+new_h, x:x+new_w] = (balloon_r * alpha + roi * (1 - alpha)).astype(np.uint8)
        result_mask[y:y+new_h, x:x+new_w] = np.maximum(
            result_mask[y:y+new_h, x:x+new_w], mask_r)

        occupied_regions.append((x, y, x + new_w, y + new_h))
        placed_names.append(Path(balloon_path).stem)

    return result_img, result_mask, placed_names
# --------------------------------------------------------------------------- #

# ---------- train/val split / dataset generator (元コードのまま) ----------- #
def split_balloons(balloon_mask_pairs, train_ratio=0.8, seed=42):
    random.seed(seed)
    pairs = balloon_mask_pairs[:]
    random.shuffle(pairs)
    n = len(pairs)
    t = int(n * train_ratio)
    return pairs[:t], pairs[t:]

def generate_dataset_split(background_files, balloon_pairs,
                           output_dir, mask_output_dir, split_name,
                           target_count, cfg):
    print(f"\n=== {split_name} 生成 ({target_count} 枚目標) ===")
    num = 1
    bg_idx = 0
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)
    while num <= target_count:
        bg_path = background_files[bg_idx % len(background_files)]
        try:
            img, msk, _ = composite_random_balloons_enhanced(
                bg_path, balloon_pairs,
                scale_range=cfg["SCALE_RANGE"],
                num_balloons_range=cfg["NUM_BALLOONS_RANGE"],
                max_attempts=cfg["MAX_ATTEMPTS"],
                aug_cfg=cfg  # 追加
            )
            cv2.imwrite(os.path.join(output_dir, f"{num:04d}.png"), img)
            cv2.imwrite(os.path.join(mask_output_dir, f"{num:04d}_mask.png"), msk)
            if num % 10 == 0:
                print(f"  {num}/{target_count} done")
            num += 1
        except Exception as e:
            print(f"✗ 合成失敗: {e}")
        bg_idx += 1
    print(f"✅ {split_name} 完了\n")

# ------------------------------ main --------------------------------------- #
def main():
    balloons_dir = "generated_balloons"
    masks_dir = "masks"
    backgrounds_dir = "generated_backs"
    out_root = "syn_dataset_aug"

    CFG = {
        # 既存
        "SCALE_RANGE": (0.1, 0.4),
        "NUM_BALLOONS_RANGE": (2, 10),
        "TARGET_TOTAL_IMAGES": 1000,
        "MAX_ATTEMPTS": 200,
        "BALLOON_SPLIT_RATIO": 0.8,
        "SEED": 42,
        # augmentation
        "ROTATION_RANGE": (-20, 20),   # degrees
        "NOISE_PROB": 0.5,
        "NOISE_SIGMA": (5, 15),        # pixel std‑dev
        "CUTOUT_PROB": 0.3,
        "CUTOUT_RATIO": (0.15, 0.35)   # relative width/height
    }

    random.seed(CFG["SEED"])
    np.random.seed(CFG["SEED"])

    # ペア収集
    pairs = []
    for b in os.listdir(balloons_dir):
        if not b.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem = Path(b).stem
        m = f"{stem}_mask.png"
        if os.path.exists(os.path.join(masks_dir, m)):
            pairs.append((os.path.join(balloons_dir, b),
                          os.path.join(masks_dir, m)))
    # 背景
    bgs = [os.path.join(backgrounds_dir, f) for f in os.listdir(backgrounds_dir)
           if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    train_pairs, val_pairs = split_balloons(pairs, CFG["BALLOON_SPLIT_RATIO"], CFG["SEED"])
    random.shuffle(bgs)
    split_pt = int(len(bgs) * CFG["BALLOON_SPLIT_RATIO"])
    train_bgs, val_bgs = bgs[:split_pt], bgs[split_pt:]

    # 生成
    train_target = int(CFG["TARGET_TOTAL_IMAGES"] * CFG["BALLOON_SPLIT_RATIO"])
    val_target = CFG["TARGET_TOTAL_IMAGES"] - train_target

    generate_dataset_split(train_bgs, train_pairs,
                           os.path.join(out_root, "train", "images"),
                           os.path.join(out_root, "train", "masks"),
                           "train", train_target, CFG)

    generate_dataset_split(val_bgs, val_pairs,
                           os.path.join(out_root, "val", "images"),
                           os.path.join(out_root, "val", "masks"),
                           "val", val_target, CFG)

    print(f"★ 出力先: {out_root}")

if __name__ == "__main__":
    main()
