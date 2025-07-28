"""
吹き出し画像+マスクを背景に合成してシンセ／拡張データセットを生成するスクリプト
Augmentation:  rotation / gaussian noise / cut‑out
"""

import cv2
import re
import numpy as np
from pathlib import Path
import os
import random
import shutil
from typing import Tuple, List
import json
from datetime import datetime

# ----------  augmentation -------------------------------------------------- #
def augment_balloon_and_mask(balloon: np.ndarray,
                             mask: np.ndarray,
                             cfg: dict):
    """
    - ランダム回転
    - 水平反転
    - 端を切り取る（ランダムに上下左右の一部を削除）
    - 線の太さを細くする（マスクを侵食して境界を内側へ）
    """
    defaults = dict(
        HFLIP_PROB=0.5,
        ROT_PROB=1.0,
        ROT_RANGE=(-20, 20),
        CUT_EDGE_PROB=0.4,
        CUT_EDGE_RATIO=(0.05, 0.20),
        THIN_LINE_PROB=0.5,
        THIN_PIXELS=(1, 3),
    )
    if cfg is None: cfg = {}
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    # ---------- 1. Horizontal flip ----------
    if random.random() < cfg["HFLIP_PROB"]:
        balloon = cv2.flip(balloon, 1)
        mask    = cv2.flip(mask, 1)

    # ---------- 2. Rotation ----------
    if random.random() < cfg["ROT_PROB"]:
        angle  = random.uniform(*cfg["ROT_RANGE"])
        h, w   = mask.shape[:2]
        center = (w/2, h/2)
        M      = cv2.getRotationMatrix2D(center, angle, 1.0)
        # 回転後キャンバスを拡げる
        cos, sin = abs(M[0,0]), abs(M[0,1])
        new_w = int(h*sin + w*cos)
        new_h = int(h*cos + w*sin)
        M[0,2] += (new_w/2) - center[0]
        M[1,2] += (new_h/2) - center[1]

        balloon = cv2.warpAffine(balloon, M, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderValue=(255,255,255))
        mask    = cv2.warpAffine(mask,    M, (new_w, new_h),
                                 flags=cv2.INTER_NEAREST,
                                 borderValue=0)

    # ---------- 3. Cut edge ----------
    if random.random() < cfg["CUT_EDGE_PROB"]:
        h, w = mask.shape
        side = random.choice(["top", "bottom", "left", "right"])
        ratio = random.uniform(*cfg["CUT_EDGE_RATIO"])
        if side == "top":
            y0, y1 = 0, int(h*ratio)
            x0, x1 = 0, w
        elif side == "bottom":
            y0, y1 = h-int(h*ratio), h
            x0, x1 = 0, w
        elif side == "left":
            x0, x1 = 0, int(w*ratio)
            y0, y1 = 0, h
        else: # right
            x0, x1 = w-int(w*ratio), w
            y0, y1 = 0, h
        balloon[y0:y1, x0:x1] = 255  # 背景色で塗る
        mask[y0:y1, x0:x1]    = 0

    # ---------- 4. Thin line ----------
    if random.random() < cfg["THIN_LINE_PROB"]:
        # マスクを侵食して境界を内側へ押し込む
        k = random.randint(*cfg["THIN_PIXELS"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
        eroded = cv2.erode(mask, kernel, iterations=1)

        # 侵食で消えた境界部分だけ白塗りして線を細く見せる
        diff = (mask > 0) & (eroded == 0)
        balloon[diff] = 255
        mask = eroded

        # 安全策：全消滅したら戻す
        if mask.sum() == 0:
            mask = eroded = cv2.dilate(mask, kernel, iterations=1)

    return balloon, mask

def sample_scale(bg_w: int, bw: int, cfg: dict) -> float:
    mode = cfg.get("SCALE_MODE", "uniform")
    if mode == "lognormal":
        mean = cfg["SCALE_MEAN"]; std = cfg["SCALE_STD"]
        clip_min, clip_max = cfg["SCALE_CLIP"]
        mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
        sigma = np.sqrt(np.log(1 + (std**2)/(mean**2)))
        s = np.random.lognormal(mu, sigma)
        return float(np.clip(s, clip_min, clip_max))
    else:
        return random.uniform(*cfg["SCALE_RANGE"])
    
def sample_num_balloons(cfg: dict, max_available: int) -> int:
    """
    cfg["COUNT_PROBS"] があればその分布から取得。
    無ければ NUM_BALLOONS_RANGE から一様サンプル。
    最終的に [lower, max_available] にクリップ。
    """
    lower, upper = cfg.get("NUM_BALLOONS_RANGE", (2, 10))
    n = None

    probs = cfg.get("COUNT_PROBS", None)
    if probs is not None:
        # 0 個を避けたい場合は確率を再正規化する or 後でmax(1, n)
        idx = np.arange(len(probs))
        n = int(np.random.choice(idx, p=probs))

    if n is None:
        n = random.randint(lower, upper)

    n = max(lower, n)
    n = min(max_available, n)
    if n <= 0:  # 念のため
        n = 1
    return n

def load_count_probs(path: str, drop_zero: bool = True):
    """
    'N balloons: M images' 形式の統計を読み込み、確率分布を返す
    """
    hist = {}
    pat = re.compile(r"^(\d+)\s+balloons:\s+(\d+)\s+images", re.I)
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                n, freq = int(m.group(1)), int(m.group(2))
                hist[n] = freq

    if drop_zero and 0 in hist:
        hist.pop(0)

    max_n = max(hist.keys())
    arr = np.zeros(max_n + 1, dtype=np.float32)
    for k, v in hist.items():
        arr[k] = v
    probs = arr / arr.sum()
    return probs
# --------------------------------------------------------------------------- #


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)  # 最後の保険


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

def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    マスクの非ゼロ領域の境界ボックスを取得
    
    Args:
        mask: グレースケールマスク画像
    
    Returns:
        (x, y, width, height) の境界ボックス
    """
    # マスクの非ゼロピクセルの座標を取得
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]  # 全体を返す
    
    # 境界ボックス計算
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1

def crop_balloon_and_mask(balloon: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    マスクの境界ボックスに基づいて吹き出し画像とマスクをクロップ
    
    Args:
        balloon: 吹き出し画像 (BGR)
        mask: マスク画像 (グレースケール)
    
    Returns:
        (クロップされた吹き出し, クロップされたマスク, 境界ボックス)
    """
    x, y, w, h = get_mask_bbox(mask)
    
    # クロップ実行
    cropped_balloon = balloon[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    return cropped_balloon, cropped_mask, (x, y, w, h)

# -------------- 合成本体（augmentation呼び出しを追加） -------------------- #
def composite_random_balloons_enhanced(
        background_path: str,
        balloon_mask_pairs: List[Tuple[str, str]],
        scale_range: Tuple[float, float] = (0.1, 0.4),
        num_balloons_range: Tuple[int, int] = (2, 10),
        max_attempts: int = 200,
        aug_cfg: dict = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:

    if aug_cfg is None:
        aug_cfg = {}
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")

    bg_h, bg_w = background.shape[:2]
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)

    if aug_cfg is None:
        aug_cfg = {}
    max_b = min(num_balloons_range[1], len(balloon_mask_pairs))
    num_balloons = sample_num_balloons(aug_cfg, max_b)

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

        # マスクの境界ボックスでクロップ（余白除去）
        cropped_balloon, cropped_mask, bbox = crop_balloon_and_mask(balloon, mask)
        
        if cropped_balloon.size == 0 or cropped_mask.size == 0:
            print(f"警告: クロップ結果が空 ({balloon_path})")
            continue

        # クロップされた画像のサイズでスケール計算
        crop_h, crop_w = cropped_balloon.shape[:2]
        try:
            scale = sample_scale(bg_w, crop_w, aug_cfg)  # クロップ後の幅を基準
        except KeyError:
            scale = random.uniform(*scale_range)
        new_w = int(bg_w * scale)
        new_h = int(crop_h * (new_w / crop_w))  # クロップ後のアスペクト比を維持
        if new_w >= bg_w or new_h >= bg_h:
            continue

        balloon_r = cv2.resize(cropped_balloon, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

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
    backgrounds_dir = "generated_double_backs"
    out_root = "syn_mihiraki_dataset_aug"

    CFG = {
    # ===== 基本設定 =====
    "TARGET_TOTAL_IMAGES": 200,       # テスト用に少なく設定
    "MAX_ATTEMPTS": 200,
    "BALLOON_SPLIT_RATIO": 0.8,
    "SEED": 42,

    # ===== 個数（統計利用） =====
    "COUNT_STATS_FILE": "balloon_count_statistics.txt",  # 実測ヒストのファイル
    "COUNT_PROBS": None,          # ← main() で読み込んで埋める
    "NUM_BALLOONS_RANGE": (7, 17),# フォールバック or クリップ用の下限/上限

    # ===== サイズ（統計利用したい場合） =====
    # ログ正規分布でサンプリングする設定
    "SCALE_MODE": "lognormal",    # "uniform" なら従来通り
    "SCALE_MEAN": 0.072,          # 横幅比の平均
    "SCALE_STD":  0.035,          # 横幅比の標準偏差
    "SCALE_CLIP": (0.03, 0.12),   # クリップ範囲
    "SCALE_RANGE": (0.05, 0.09),  # フォールバック用（uniform時にも使う）

    # ===== augmentation =====
    "HFLIP_PROB": 0.5,
    "ROT_PROB": 1.0,
    "ROT_RANGE": (-20, 20),
    "CUT_EDGE_PROB": 0.4,
    "CUT_EDGE_RATIO": (0.05, 0.20),
    "THIN_LINE_PROB": 0.5,
    "THIN_PIXELS": (1, 3),
}
    try:
        CFG["COUNT_PROBS"] = load_count_probs(CFG["COUNT_STATS_FILE"], drop_zero=True)
        print("個数ヒストグラムを読み込みました。")
    except Exception as e:
        print(f"個数ヒスト読込に失敗（フォールバックします）: {e}")

    random.seed(CFG["SEED"])
    np.random.seed(CFG["SEED"])

    # 出力ディレクトリを作成
    os.makedirs(out_root, exist_ok=True)
    
    # CFG設定をファイル出力
    config_output = {
        "timestamp": datetime.now().isoformat(),
        "script_name": "create_syn_dataset_aug.py",
        "dataset_output_path": out_root,
        "config": CFG,
        "input_directories": {
            "balloons_dir": balloons_dir,
            "masks_dir": masks_dir,
            "backgrounds_dir": backgrounds_dir
        },
        "augmentation_info": {
            "horizontal_flip": "水平反転",
            "rotation": "回転（角度範囲指定）",
            "cut_edge": "端の切り取り",
            "thin_line": "線の太さを細くする（侵食処理）"
        }
    }
    
    config_file_path = os.path.join(out_root, "config.json")
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"設定情報を保存: {config_file_path}")

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

    # 統計情報を収集・追加
    total_generated = train_target + val_target
    stats = {
        "total_images": total_generated,
        "train_images": train_target,
        "val_images": val_target,
        "train_balloons_used": len(train_pairs),
        "val_balloons_used": len(val_pairs),
        "train_backgrounds_used": len(train_bgs),
        "val_backgrounds_used": len(val_bgs),
        "total_backgrounds_available": len(bgs),
        "total_balloon_pairs_available": len(pairs)
    }
    
    # config.jsonに統計情報を追加
    config_output["statistics"] = stats
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"\n=== データセット生成完了 ===")
    print(f"★ 出力先: {out_root}")
    print(f"総生成画像数: {total_generated}枚 (train: {train_target}, val: {val_target})")
    print(f"使用吹き出し: train {len(train_pairs)}個, val {len(val_pairs)}個")
    print(f"使用背景: train {len(train_bgs)}個, val {len(val_bgs)}個")
    print(f"設定・統計情報: {config_file_path}")

if __name__ == "__main__":
    main()
