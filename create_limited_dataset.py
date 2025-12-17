"""
吹き出し・背景画像のバリエーションを制限した合成データセット作成スクリプト

吹き出しや背景画像の使用枚数/割合を指定して、
データセットのバリエーションによる精度への影響を調査できます。

使用例:
  # 吹き出し50%、背景50%を使用して1000枚生成
  python create_limited_dataset.py --balloon-ratio 0.5 --background-ratio 0.5 --target-images 1000 --final-output-dir balloon_dataset/syn1000-b50-bg50

  # 吹き出し10枚、背景20枚を使用（枚数指定）
  python create_limited_dataset.py --balloon-count 10 --background-count 20 --target-images 500 --final-output-dir balloon_dataset/syn500-b10-bg20

  # 吹き出し100%、背景25%を使用
  python create_limited_dataset.py --balloon-ratio 1.0 --background-ratio 0.25 --target-images 1000 --final-output-dir balloon_dataset/syn1000-b100-bg25
"""

import cv2
import numpy as np
from pathlib import Path
import os
import random
import shutil
from tqdm import tqdm
import json
from datetime import datetime
import re
import argparse
from typing import List, Tuple, Optional


def regions_overlap(region1: tuple, region2: tuple) -> bool:
    """2つの領域が重複するかチェック"""
    x1_min, y1_min, x1_max, y1_max = region1
    x2_min, y2_min, x2_max, y2_max = region2
    return not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)


def calculate_overlap_area(region1: tuple, region2: tuple) -> int:
    """2つの領域の重複面積を計算"""
    x1_min, y1_min, x1_max, y1_max = region1
    x2_min, y2_min, x2_max, y2_max = region2
    
    overlap_x_min = max(x1_min, x2_min)
    overlap_y_min = max(y1_min, y2_min)
    overlap_x_max = min(x1_max, x2_max)
    overlap_y_max = min(y1_max, y2_max)
    
    if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
        return 0
    return (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)


def get_mask_bbox(mask):
    """マスクの非ゼロ領域の境界ボックスを取得"""
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def crop_balloon_and_mask(balloon, mask):
    """マスクの境界ボックスに基づいて吹き出し画像とマスクをクロップ"""
    x, y, w, h = get_mask_bbox(mask)
    cropped_balloon = balloon[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    return cropped_balloon, cropped_mask, (x, y, w, h)


def sample_scale(bg_w: int, bw: int, cfg: dict) -> float:
    """統計情報に基づいてスケールをサンプリング"""
    mode = cfg.get("SCALE_MODE", "uniform")
    if mode == "lognormal":
        clip_min, clip_max = cfg["SCALE_CLIP"]
        target_median = 0.007226
        mu = np.log(target_median)
        sigma = 0.85
        s = np.random.lognormal(mu, sigma)
        return float(np.clip(s, clip_min, clip_max))
    else:
        return random.uniform(*cfg["SCALE_RANGE"])


def calculate_area_based_size(crop_w: int, crop_h: int, bg_w: int, bg_h: int, 
                             target_scale: float, max_w_ratio: float = 0.3, 
                             max_h_ratio: float = 0.4) -> tuple:
    """面積ベースのリサイズサイズ計算"""
    bg_area = bg_w * bg_h
    target_area = bg_area * target_scale
    aspect_ratio = crop_h / crop_w
    
    ideal_w = int(np.sqrt(target_area / aspect_ratio))
    ideal_h = int(np.sqrt(target_area * aspect_ratio))
    
    max_w = int(bg_w * max_w_ratio)
    max_h = int(bg_h * max_h_ratio)
    
    scale_by_w = max_w / ideal_w if ideal_w > max_w else 1.0
    scale_by_h = max_h / ideal_h if ideal_h > max_h else 1.0
    adjust_scale = min(scale_by_w, scale_by_h)
    
    new_w = max(int(ideal_w * adjust_scale), 20)
    new_h = max(int(ideal_h * adjust_scale), 20)
    return new_w, new_h


def sample_num_balloons(cfg: dict, max_available: int) -> int:
    """統計情報に基づいて吹き出し個数をサンプリング"""
    lower, upper = cfg.get("NUM_BALLOONS_RANGE", (2, 10))
    probs = cfg.get("COUNT_PROBS", None)
    
    if probs is not None:
        idx = np.arange(len(probs))
        n = int(np.random.choice(idx, p=probs))
    else:
        n = random.randint(lower, upper)
    
    n = max(lower, min(max_available, n))
    return max(1, n)


def load_count_probs(path: str, drop_zero: bool = True):
    """'N balloons: M images' 形式の統計を読み込み、確率分布を返す"""
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

    if not hist:
        return None
        
    max_n = max(hist.keys())
    arr = np.zeros(max_n + 1, dtype=np.float32)
    for k, v in hist.items():
        arr[k] = v
    return arr / arr.sum()


def _json_default(o):
    """JSON serialization helper"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def composite_random_balloons(background_path: str, balloon_mask_pairs: list,
                             cfg: dict = None) -> tuple:
    """背景画像に複数の吹き出しをランダムに合成する"""
    if cfg is None:
        cfg = {}
        
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")
    
    bg_h, bg_w = background.shape[:2]
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    max_balloons = min(cfg.get("NUM_BALLOONS_RANGE", (2, 10))[1], len(balloon_mask_pairs))
    num_balloons = sample_num_balloons(cfg, max_balloons)
    num_balloons = min(num_balloons, len(balloon_mask_pairs))
    
    selected_pairs = random.sample(balloon_mask_pairs, num_balloons)
    
    occupied_regions = []
    successfully_placed = []
    balloon_details = []
    max_attempts = cfg.get("MAX_ATTEMPTS", 200)
    
    for balloon_path, mask_path in selected_pairs:
        balloon = cv2.imread(balloon_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if balloon is None or mask is None:
            continue
        
        cropped_balloon, cropped_mask, bbox = crop_balloon_and_mask(balloon, mask)
        if cropped_balloon.size == 0 or cropped_mask.size == 0:
            continue

        crop_h, crop_w = cropped_balloon.shape[:2]
        
        try:
            balloon_scale = sample_scale(bg_w, crop_w, cfg)
        except KeyError:
            balloon_scale = random.uniform(*cfg.get("SCALE_RANGE", (0.07, 0.12)))
        
        new_balloon_w, new_balloon_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, balloon_scale,
            max_w_ratio=cfg.get("MAX_WIDTH_RATIO", 0.3),
            max_h_ratio=cfg.get("MAX_HEIGHT_RATIO", 0.4)
        )
        
        if new_balloon_w >= bg_w or new_balloon_h >= bg_h:
            continue
        
        balloon_resized = cv2.resize(cropped_balloon, (new_balloon_w, new_balloon_h))
        mask_resized = cv2.resize(cropped_mask, (new_balloon_w, new_balloon_h))
        
        # 配置位置を探す
        placed = False
        best_position = None
        min_overlap_ratio = float('inf')
        
        for attempt in range(max_attempts // 2):
            max_x = max(0, bg_w - new_balloon_w)
            max_y = max(0, bg_h - new_balloon_h)
            if max_x <= 0 or max_y <= 0:
                break
                
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            new_region = (x, y, x + new_balloon_w, y + new_balloon_h)
            
            max_overlap_ratio = 0
            for occupied in occupied_regions:
                if regions_overlap(new_region, occupied):
                    overlap_area = calculate_overlap_area(new_region, occupied)
                    new_area = new_balloon_w * new_balloon_h
                    overlap_ratio = overlap_area / new_area
                    max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)
            
            if max_overlap_ratio <= 0.15:
                best_position = (x, y)
                placed = True
                break
            
            if max_overlap_ratio < min_overlap_ratio:
                min_overlap_ratio = max_overlap_ratio
                best_position = (x, y)
        
        if not placed and best_position is not None:
            x, y = best_position
            placed = True
        
        if not placed:
            max_x = max(0, bg_w - new_balloon_w)
            max_y = max(0, bg_h - new_balloon_h)
            if max_x >= 0 and max_y >= 0:
                x = random.randint(0, max_x) if max_x > 0 else 0
                y = random.randint(0, max_y) if max_y > 0 else 0
                placed = True
        
        if placed:
            mask_norm = mask_resized.astype(np.float32) / 255.0
            mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
            
            bg_region = result_img[y:y+new_balloon_h, x:x+new_balloon_w]
            blended = balloon_resized.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
            result_img[y:y+new_balloon_h, x:x+new_balloon_w] = blended.astype(np.uint8)
            
            result_mask[y:y+new_balloon_h, x:x+new_balloon_w] = np.maximum(
                result_mask[y:y+new_balloon_h, x:x+new_balloon_w], mask_resized)
            
            new_region = (x, y, x + new_balloon_w, y + new_balloon_h)
            occupied_regions.append(new_region)
            successfully_placed.append(Path(balloon_path).stem)
            
            balloon_info = {
                "balloon_file": Path(balloon_path).name,
                "original_size": f"{balloon.shape[1]}x{balloon.shape[0]}",
                "cropped_size": f"{crop_w}x{crop_h}",
                "final_size": f"{new_balloon_w}x{new_balloon_h}",
                "position": f"({x},{y})",
                "scale": f"{balloon_scale:.3f}",
            }
            balloon_details.append(balloon_info)
    
    return result_img, result_mask, successfully_placed, balloon_details


def split_balloons(balloon_mask_pairs: list, train_ratio: float = 0.8, seed: int = 42) -> tuple:
    """吹き出しをtrain用とval用に分割する"""
    random.seed(seed)
    shuffled_pairs = balloon_mask_pairs.copy()
    random.shuffle(shuffled_pairs)
    
    n = len(shuffled_pairs)
    train_count = int(n * train_ratio)
    
    return shuffled_pairs[:train_count], shuffled_pairs[train_count:]


def generate_dataset_split(background_files: list, balloon_pairs: list, 
                          output_dir: str, mask_output_dir: str, split_name: str,
                          target_count: int, cfg: dict, final_output_dir: str = None) -> int:
    """指定されたsplit（trainまたはval）のデータセットを生成する"""
    print(f"\n=== {split_name} データセット生成開始 ===")
    print(f"目標画像数: {target_count}")
    print(f"使用背景画像数: {len(background_files)}")
    print(f"使用吹き出し数: {len(balloon_pairs)}")
    
    if final_output_dir:
        log_file_path = os.path.join(final_output_dir, f"{split_name}_composition_log.txt")
    else:
        log_file_path = os.path.join(output_dir, f"{split_name}_composition_log.txt")
    
    current_number = 1
    success_count = 0
    bg_idx = 0
    
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== {split_name.upper()} データセット合成ログ ===\n")
        log_file.write(f"生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"目標画像数: {target_count}\n")
        log_file.write(f"使用背景画像数: {len(background_files)}\n")
        log_file.write(f"使用吹き出し数: {len(balloon_pairs)}\n")
        log_file.write("=" * 80 + "\n\n")
    
    pbar = tqdm(total=target_count, desc=f"{split_name}生成")
    
    while success_count < target_count:
        bg_path = background_files[bg_idx % len(background_files)]
        bg_name = Path(bg_path).stem
        
        try:
            result_img, result_mask, placed_balloons, balloon_details = composite_random_balloons(
                bg_path, balloon_pairs, cfg=cfg
            )
            
            output_img_path = os.path.join(output_dir, f"{current_number:03d}.png")
            output_mask_path = os.path.join(mask_output_dir, f"{current_number:03d}_mask.png")
            
            cv2.imwrite(output_img_path, result_img)
            cv2.imwrite(output_mask_path, result_mask)
            
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"画像 {current_number:03d}.png:\n")
                log_file.write(f"  背景ファイル: {Path(bg_path).name}\n")
                log_file.write(f"  配置した吹き出し数: {len(balloon_details)}\n")
                for i, detail in enumerate(balloon_details, 1):
                    log_file.write(f"    吹き出し{i}: {detail['balloon_file']}\n")
                log_file.write("\n")
            
            success_count += 1
            current_number += 1
            pbar.update(1)
            
        except Exception as e:
            print(f"\n✗ 合成失敗 (背景:{bg_name}): {e}")
        
        bg_idx += 1
    
    pbar.close()
    print(f"✅ {split_name} 完了: {success_count}個の画像を生成")
    return success_count


def select_limited_files(file_list: list, ratio: float = None, count: int = None, 
                        seed: int = 42) -> list:
    """
    ファイルリストから指定した割合または枚数だけ選択する
    
    Args:
        file_list: 元のファイルリスト
        ratio: 使用する割合（0.0-1.0）
        count: 使用する枚数（ratioより優先）
        seed: ランダムシード
    
    Returns:
        選択されたファイルのリスト
    """
    random.seed(seed)
    shuffled = file_list.copy()
    random.shuffle(shuffled)
    
    if count is not None:
        n = min(count, len(shuffled))
    elif ratio is not None:
        n = max(1, int(len(shuffled) * ratio))
    else:
        n = len(shuffled)
    
    return shuffled[:n]


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="吹き出し・背景画像のバリエーションを制限した合成データセット作成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 吹き出し50%、背景50%を使用して1000枚生成
  python create_limited_dataset.py --balloon-ratio 0.5 --background-ratio 0.5 --target-images 1000 --final-output-dir balloon_dataset/syn1000-b50-bg50

  # 吹き出し10枚、背景20枚を使用（枚数指定）
  python create_limited_dataset.py --balloon-count 10 --background-count 20 --target-images 500 --final-output-dir balloon_dataset/syn500-b10-bg20

  # 吹き出し100%、背景25%を使用
  python create_limited_dataset.py --balloon-ratio 1.0 --background-ratio 0.25 --target-images 1000 --final-output-dir balloon_dataset/syn1000-b100-bg25
        """
    )
    
    # 入力ディレクトリ
    parser.add_argument("--balloon-dir", default="balloons/images", 
                        help="吹き出し画像ディレクトリ")
    parser.add_argument("--mask-dir", default="balloons/masks", 
                        help="マスク画像ディレクトリ")
    parser.add_argument("--background-dir", default="generated_double_backs_1536x1024", 
                        help="背景画像ディレクトリ")
    
    # 出力ディレクトリ
    parser.add_argument("--final-output-dir", required=True, 
                        help="最終出力ディレクトリ（train/val分割後）")
    
    # 吹き出し制限（割合または枚数）
    balloon_group = parser.add_mutually_exclusive_group()
    balloon_group.add_argument("--balloon-ratio", type=float, default=None,
                               help="使用する吹き出しの割合（0.0-1.0）")
    balloon_group.add_argument("--balloon-count", type=int, default=None,
                               help="使用する吹き出しの枚数")
    
    # 背景制限（割合または枚数）
    background_group = parser.add_mutually_exclusive_group()
    background_group.add_argument("--background-ratio", type=float, default=None,
                                  help="使用する背景画像の割合（0.0-1.0）")
    background_group.add_argument("--background-count", type=int, default=None,
                                  help="使用する背景画像の枚数")
    
    # 生成設定
    parser.add_argument("--target-images", type=int, required=True,
                        help="生成する画像総数")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="train用の比率（残りがval）")
    parser.add_argument("--seed", type=int, default=42,
                        help="ランダムシード")
    
    args = parser.parse_args()
    
    # 設定
    CFG = {
        "SCALE_RANGE": (0.070, 0.120),
        "NUM_BALLOONS_RANGE": (5, 17),
        "MAX_ATTEMPTS": 200,
        "TRAIN_RATIO": args.train_ratio,
        "BALLOON_SPLIT_SEED": args.seed,
        
        # 統計情報ベースのサンプリング設定
        "SCALE_MODE": "lognormal",
        "SCALE_MEAN": 0.008769,
        "SCALE_STD": 0.006773,
        "SCALE_CLIP": (0.002000, 0.020000),
        "COUNT_PROBS": None,
        "COUNT_STATS_FILE": "balloon_count_statistics.txt",
        
        # 面積ベースリサイズ設定
        "MAX_WIDTH_RATIO": 0.20,
        "MAX_HEIGHT_RATIO": 0.30,
    }
    
    # 一時ディレクトリ
    temp_output_dir = "temp_limited_output"
    temp_mask_dir = "temp_limited_masks"
    
    # ディレクトリ作成
    os.makedirs(temp_output_dir, exist_ok=True)
    os.makedirs(temp_mask_dir, exist_ok=True)
    os.makedirs(args.final_output_dir, exist_ok=True)
    
    # 統計情報ファイルの読み込み
    if CFG.get("COUNT_STATS_FILE") and os.path.exists(CFG["COUNT_STATS_FILE"]):
        try:
            CFG["COUNT_PROBS"] = load_count_probs(CFG["COUNT_STATS_FILE"])
            print(f"統計ベースの吹き出し個数サンプリングを有効化")
        except Exception as e:
            print(f"統計ファイル読み込みエラー: {e}")
    
    print("=" * 60)
    print("  制限付き合成データセット作成")
    print("=" * 60)
    
    # 吹き出しとマスクの対応を取得
    all_balloon_pairs = []
    for balloon_file in os.listdir(args.balloon_dir):
        if balloon_file.endswith(('.png', '.jpg', '.jpeg')):
            balloon_path = os.path.join(args.balloon_dir, balloon_file)
            balloon_stem = Path(balloon_file).stem
            mask_file = f"{balloon_stem}_mask.png"
            mask_path = os.path.join(args.mask_dir, mask_file)
            if os.path.exists(mask_path):
                all_balloon_pairs.append((balloon_path, mask_path))
    
    # 背景画像を取得
    all_background_files = []
    for bg_file in os.listdir(args.background_dir):
        if bg_file.endswith(('.png', '.jpg', '.jpeg')):
            all_background_files.append(os.path.join(args.background_dir, bg_file))
    
    print(f"\n📊 利用可能なデータ:")
    print(f"  吹き出し: {len(all_balloon_pairs)}個")
    print(f"  背景画像: {len(all_background_files)}個")
    
    # 吹き出しを制限
    if args.balloon_ratio is not None or args.balloon_count is not None:
        limited_balloon_pairs = select_limited_files(
            all_balloon_pairs, 
            ratio=args.balloon_ratio, 
            count=args.balloon_count,
            seed=args.seed
        )
        balloon_limit_str = f"{args.balloon_count}枚" if args.balloon_count else f"{args.balloon_ratio*100:.0f}%"
    else:
        limited_balloon_pairs = all_balloon_pairs
        balloon_limit_str = "100%"
    
    # 背景画像を制限
    if args.background_ratio is not None or args.background_count is not None:
        limited_background_files = select_limited_files(
            all_background_files,
            ratio=args.background_ratio,
            count=args.background_count,
            seed=args.seed + 1  # 別のシードを使用
        )
        bg_limit_str = f"{args.background_count}枚" if args.background_count else f"{args.background_ratio*100:.0f}%"
    else:
        limited_background_files = all_background_files
        bg_limit_str = "100%"
    
    print(f"\n🔧 制限設定:")
    print(f"  吹き出し: {balloon_limit_str} → {len(limited_balloon_pairs)}個使用")
    print(f"  背景画像: {bg_limit_str} → {len(limited_background_files)}個使用")
    print(f"  生成画像数: {args.target_images}枚")
    print(f"  train/val比率: {args.train_ratio:.0%}/{1-args.train_ratio:.0%}")
    
    # 吹き出しをtrain/valに分割
    train_balloons, val_balloons = split_balloons(
        limited_balloon_pairs, CFG["TRAIN_RATIO"], CFG["BALLOON_SPLIT_SEED"]
    )
    
    print(f"\n📦 吹き出し分割:")
    print(f"  train用: {len(train_balloons)}個")
    print(f"  val用: {len(val_balloons)}個")
    
    # 目標画像数を計算
    train_target = int(args.target_images * CFG["TRAIN_RATIO"])
    val_target = args.target_images - train_target
    
    # 設定をファイル出力
    config_output = {
        "timestamp": datetime.now().isoformat(),
        "script_name": "create_limited_dataset.py",
        "dataset_output_path": args.final_output_dir,
        "limitation_settings": {
            "balloon_ratio": args.balloon_ratio,
            "balloon_count": args.balloon_count,
            "background_ratio": args.background_ratio,
            "background_count": args.background_count,
            "actual_balloons_used": len(limited_balloon_pairs),
            "actual_backgrounds_used": len(limited_background_files),
            "total_balloons_available": len(all_balloon_pairs),
            "total_backgrounds_available": len(all_background_files),
        },
        "generation_settings": {
            "target_images": args.target_images,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
        },
        "config": CFG,
    }
    
    config_file_path = os.path.join(args.final_output_dir, "config.json")
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"\n設定情報を保存: {config_file_path}")
    
    # trainデータセット生成
    train_img_dir = os.path.join(temp_output_dir, "train")
    train_mask_dir = os.path.join(temp_mask_dir, "train")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    
    train_count = generate_dataset_split(
        limited_background_files, train_balloons,
        train_img_dir, train_mask_dir, "train", train_target, CFG, args.final_output_dir
    )
    
    # valデータセット生成
    val_img_dir = os.path.join(temp_output_dir, "val")
    val_mask_dir = os.path.join(temp_mask_dir, "val")
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    
    val_count = generate_dataset_split(
        limited_background_files, val_balloons,
        val_img_dir, val_mask_dir, "val", val_target, CFG, args.final_output_dir
    )
    
    # 最終ディレクトリにコピー
    print(f"\n=== 最終データセット構造を作成中 ===")
    
    for split in ["train", "val"]:
        (Path(args.final_output_dir) / split / "images").mkdir(parents=True, exist_ok=True)
        (Path(args.final_output_dir) / split / "masks").mkdir(parents=True, exist_ok=True)
        
        src_img_dir = os.path.join(temp_output_dir, split)
        src_mask_dir = os.path.join(temp_mask_dir, split)
        final_img_dir = os.path.join(args.final_output_dir, split, "images")
        final_mask_dir = os.path.join(args.final_output_dir, split, "masks")
        
        for img_file in os.listdir(src_img_dir):
            if img_file.endswith('.png'):
                shutil.copy2(os.path.join(src_img_dir, img_file), os.path.join(final_img_dir, img_file))
        
        for mask_file in os.listdir(src_mask_dir):
            if mask_file.endswith('.png'):
                shutil.copy2(os.path.join(src_mask_dir, mask_file), os.path.join(final_mask_dir, mask_file))
    
    # 一時ディレクトリを削除
    shutil.rmtree(temp_output_dir)
    shutil.rmtree(temp_mask_dir)
    
    # 統計情報を更新
    stats = {
        "total_images": train_count + val_count,
        "train_images": train_count,
        "val_images": val_count,
        "balloons_used": len(limited_balloon_pairs),
        "backgrounds_used": len(limited_background_files),
    }
    config_output["statistics"] = stats
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    
    # 最終レポート
    print(f"\n" + "=" * 60)
    print(f"  データセット作成完了")
    print("=" * 60)
    print(f"📁 出力先: {args.final_output_dir}")
    print(f"📊 生成結果:")
    print(f"  train: {train_count}枚")
    print(f"  val: {val_count}枚")
    print(f"  合計: {train_count + val_count}枚")
    print(f"\n🔧 使用データ:")
    print(f"  吹き出し: {len(limited_balloon_pairs)}個 ({balloon_limit_str})")
    print(f"  背景画像: {len(limited_background_files)}個 ({bg_limit_str})")
    print(f"\n📄 設定ファイル: {config_file_path}")


if __name__ == "__main__":
    main()
