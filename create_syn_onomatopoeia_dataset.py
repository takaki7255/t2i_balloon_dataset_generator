"""
オノマトペを漫画背景画像に合成し、学習データセットを作成するスクリプト
create_syn_dataset.pyのオノマトペ版
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


def crop_onomatopoeia_and_mask(onomatopoeia, mask):
    """マスクの境界ボックスに基づいてオノマトペ画像とマスクをクロップ"""
    x, y, w, h = get_mask_bbox(mask)
    
    cropped_onomatopoeia = onomatopoeia[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    return cropped_onomatopoeia, cropped_mask, (x, y, w, h)


def load_scale_stats(path: str) -> dict:
    """統計ファイルからスケール統計情報を読み込む（バウンディングボックスベース）"""
    stats = {
        "mean": None,
        "median": None,
        "std": None,
        "min": None,
        "max": None,
        "q25": None,
        "q75": None,
        # バウンディングボックスベース
        "bbox_mean": None,
        "bbox_median": None,
        "bbox_std": None,
        "bbox_q25": None,
        "bbox_q75": None,
        "type": "bbox"  # バウンディングボックスベースを使用
    }
    
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        
        # バウンディングボックス統計セクションを抽出
        in_bbox_section = False
        for line in content.split('\n'):
            if "Bounding Box Size Ratio Statistics:" in line:
                in_bbox_section = True
                continue
            if in_bbox_section:
                if "Area Statistics" in line:
                    break
                
                if "Mean:" in line:
                    stats["bbox_mean"] = float(line.split("Mean:")[1].strip())
                elif "Median:" in line:
                    stats["bbox_median"] = float(line.split("Median:")[1].strip())
                elif "Standard deviation:" in line:
                    stats["bbox_std"] = float(line.split("Standard deviation:")[1].strip())
                elif "25th percentile:" in line:
                    stats["bbox_q25"] = float(line.split("25th percentile:")[1].strip())
                elif "75th percentile:" in line:
                    stats["bbox_q75"] = float(line.split("75th percentile:")[1].strip())
        
        print(f"✓ バウンディングボックスベース統計を読み込み: {path}")
        print(f"  中央値: {stats['bbox_median']:.6f}, 平均: {stats['bbox_mean']:.6f}, 標準偏差: {stats['bbox_std']:.6f}")
        return stats
    except Exception as e:
        print(f"統計ファイル読み込みエラー: {e}")
        return stats


def sample_scale(bg_w: int, bw: int, cfg: dict) -> float:
    """統計情報に基づいてスケールをサンプリング（バウンディングボックスベース）"""
    mode = cfg.get("SCALE_MODE", "uniform")
    if mode == "lognormal":
        clip_min, clip_max = cfg["SCALE_CLIP"]
        
        # 統計ファイルから取得した中央値と標準偏差を使用
        # バウンディングボックスベースで計算（セグメンテーションベースより約4倍大きい）
        scale_stats = cfg.get("SCALE_STATS", {})
        # バウンディングボックス中央値: 0.001488（セグメンテーション 0.000371 の約4倍）
        target_median = scale_stats.get("bbox_median", 0.001488)
        sigma = 0.8  # より中央値に近い分布にする
        
        mu = np.log(target_median)
        s = np.random.lognormal(mu, sigma)
        return float(np.clip(s, clip_min, clip_max))
    else:
        return random.uniform(*cfg["SCALE_RANGE"])


def calculate_area_based_size(crop_w: int, crop_h: int, bg_w: int, bg_h: int, 
                             target_scale: float, mask: np.ndarray = None,
                             max_w_ratio: float = 0.15, 
                             max_h_ratio: float = 0.15) -> tuple:
    """面積ベースのリサイズサイズ計算
    
    Args:
        crop_w, crop_h: クロップされた画像の幅・高さ
        bg_w, bg_h: 背景画像の幅・高さ
        target_scale: ターゲットスケール（背景面積比）
        mask: マスク画像（マスク領域のみで面積を計算する場合）
        max_w_ratio, max_h_ratio: 最大サイズ比
    
    Returns:
        (new_w, new_h): リサイズ後のサイズ
    """
    bg_area = bg_w * bg_h
    target_area = bg_area * target_scale
    
    # マスク領域を考慮した場合、アスペクト比とは異なる有効面積比を計算
    if mask is not None:
        # マスク領域のピクセル数
        mask_pixels = np.count_nonzero(mask)
        crop_pixels = crop_w * crop_h
        
        if crop_pixels > 0:
            # 有効面積の比率（マスク領域 / バウンディングボックス）
            mask_ratio = mask_pixels / crop_pixels
        else:
            mask_ratio = 1.0
    else:
        mask_ratio = 1.0
    
    aspect_ratio = crop_h / crop_w
    
    # アスペクト比を維持した理想サイズ
    ideal_w = int(np.sqrt(target_area / aspect_ratio))
    ideal_h = int(np.sqrt(target_area * aspect_ratio))
    
    # 最大サイズ制限（オノマトペは小さめ）
    max_w = int(bg_w * max_w_ratio)
    max_h = int(bg_h * max_h_ratio)
    
    # 制限に合わせて調整
    scale_by_w = max_w / ideal_w if ideal_w > max_w else 1.0
    scale_by_h = max_h / ideal_h if ideal_h > max_h else 1.0
    adjust_scale = min(scale_by_w, scale_by_h)
    
    new_w = int(ideal_w * adjust_scale)
    new_h = int(ideal_h * adjust_scale)
    
    # 最小サイズ確保
    new_w = max(new_w, 10)
    new_h = max(new_h, 10)
    
    return new_w, new_h


def sample_num_onomatopoeia(cfg: dict, max_available: int) -> int:
    """統計情報に基づいてオノマトペ個数をサンプリング"""
    lower, upper = cfg.get("NUM_ONOMATOPOEIA_RANGE", (1, 10))
    n = None

    probs = cfg.get("COUNT_PROBS", None)
    if probs is not None:
        idx = np.arange(len(probs))
        n = int(np.random.choice(idx, p=probs))

    if n is None:
        n = random.randint(lower, upper)

    n = max(lower, n)
    n = min(max_available, n)
    if n <= 0:
        n = 1
    return n


def load_count_probs(path: str, drop_zero: bool = True):
    """統計ファイルから個数の確率分布を読み込む"""
    # オノマトペ統計ファイルは別形式なので、平均・標準偏差から正規分布を生成
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
            
        # 平均と標準偏差を抽出
        mean_match = re.search(r"Mean:\s+([\d.]+)", content)
        std_match = re.search(r"Standard deviation:\s+([\d.]+)", content)
        
        if mean_match and std_match:
            mean = float(mean_match.group(1))
            std = float(std_match.group(1))
            
            # 正規分布に基づく確率分布を生成（0-30の範囲）
            max_n = 30
            arr = np.zeros(max_n + 1, dtype=np.float32)
            
            for n in range(1, max_n + 1):
                # 正規分布の確率密度
                arr[n] = np.exp(-0.5 * ((n - mean) / std) ** 2)
            
            if drop_zero:
                arr[0] = 0
            
            if arr.sum() > 0:
                probs = arr / arr.sum()
                return probs
    except Exception as e:
        print(f"統計ファイル読み込みエラー: {e}")
    
    return None


def _json_default(o):
    """JSON serialization helper"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def composite_random_onomatopoeia(background_path: str, onomatopoeia_mask_pairs: list,
                                 cfg: dict = None) -> tuple:
    """背景画像にランダムに選択した複数のオノマトペを重複なしで合成する"""
    if cfg is None:
        cfg = {}
        
    # 背景画像読み込み
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")
    
    bg_h, bg_w = background.shape[:2]
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    # 配置するオノマトペ数を統計情報に基づいて決定
    max_onomatopoeia = min(cfg.get("NUM_ONOMATOPOEIA_RANGE", (1, 10))[1], len(onomatopoeia_mask_pairs))
    num_onomatopoeia = sample_num_onomatopoeia(cfg, max_onomatopoeia)
    
    # ランダムにオノマトペを選択
    selected_pairs = random.sample(onomatopoeia_mask_pairs, num_onomatopoeia)
    
    # 配置済み領域を記録する配列
    occupied_regions = []
    successfully_placed = []
    onomatopoeia_details = []
    
    # スケール統計情報
    scale_stats = cfg.get("SCALE_STATS", {})
    
    for onomatopoeia_path, mask_path in selected_pairs:
        # オノマトペとマスク読み込み
        onomatopoeia = cv2.imread(onomatopoeia_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if onomatopoeia is None or mask is None:
            continue
        
        # マスクの境界ボックスでクロップ
        cropped_onomatopoeia, cropped_mask, bbox = crop_onomatopoeia_and_mask(onomatopoeia, mask)
        
        if cropped_onomatopoeia.size == 0 or cropped_mask.size == 0:
            continue

        crop_h, crop_w = cropped_onomatopoeia.shape[:2]
        
        # 統計情報に基づくスケールサンプリング
        try:
            onomatopoeia_scale = sample_scale(bg_w, crop_w, cfg)
        except KeyError:
            onomatopoeia_scale = random.uniform(cfg.get("SCALE_RANGE", (0.0001, 0.002))[0], 
                                              cfg.get("SCALE_RANGE", (0.0001, 0.002))[1])
        
        # 面積ベースのサイズ計算（マスク領域を考慮）
        new_onomatopoeia_w, new_onomatopoeia_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, onomatopoeia_scale,
            mask=cropped_mask,  # マスク領域を考慮
            max_w_ratio=cfg.get("MAX_WIDTH_RATIO", 0.15),
            max_h_ratio=cfg.get("MAX_HEIGHT_RATIO", 0.15)
        )
        
        # 背景サイズを超える場合はスキップ
        if new_onomatopoeia_w >= bg_w or new_onomatopoeia_h >= bg_h:
            continue
        
        # クロップされた画像をリサイズ
        onomatopoeia_resized = cv2.resize(cropped_onomatopoeia, (new_onomatopoeia_w, new_onomatopoeia_h))
        mask_resized = cv2.resize(cropped_mask, (new_onomatopoeia_w, new_onomatopoeia_h))
        
        # 重複を避けつつ位置を探す
        placed = False
        best_position = None
        min_overlap_ratio = float('inf')
        
        # まず重複回避を試行
        max_attempts = cfg.get("MAX_ATTEMPTS", 200)
        for attempt in range(max_attempts // 2):
            max_x = max(0, bg_w - new_onomatopoeia_w)
            max_y = max(0, bg_h - new_onomatopoeia_h)
            
            if max_x <= 0 or max_y <= 0:
                break
                
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # 新しい領域
            new_region = (x, y, x + new_onomatopoeia_w, y + new_onomatopoeia_h)
            
            # 既存領域との重複チェック
            max_overlap_ratio = 0
            for occupied in occupied_regions:
                if regions_overlap(new_region, occupied):
                    overlap_area = calculate_overlap_area(new_region, occupied)
                    new_area = new_onomatopoeia_w * new_onomatopoeia_h
                    overlap_ratio = overlap_area / new_area
                    max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)
            
            # 重複が少ない場合は配置
            if max_overlap_ratio <= 0.15:
                best_position = (x, y)
                placed = True
                break
            
            # より良い位置を記録
            if max_overlap_ratio < min_overlap_ratio:
                min_overlap_ratio = max_overlap_ratio
                best_position = (x, y)
        
        # 重複回避に失敗した場合、最も重複の少ない位置に配置
        if not placed and best_position is not None:
            x, y = best_position
            placed = True
        
        # 最終的にランダム位置に配置（フォールバック）
        if not placed:
            max_x = max(0, bg_w - new_onomatopoeia_w)
            max_y = max(0, bg_h - new_onomatopoeia_h)
            if max_x >= 0 and max_y >= 0:
                x = random.randint(0, max_x) if max_x > 0 else 0
                y = random.randint(0, max_y) if max_y > 0 else 0
                placed = True
        
        if placed:
            # 合成実行
            mask_norm = mask_resized.astype(np.float32) / 255.0
            mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
            
            # 背景画像の該当領域
            bg_region = result_img[y:y+new_onomatopoeia_h, x:x+new_onomatopoeia_w]
            
            # アルファブレンディング
            blended = onomatopoeia_resized.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
            result_img[y:y+new_onomatopoeia_h, x:x+new_onomatopoeia_w] = blended.astype(np.uint8)
            
            # マスクも合成
            result_mask[y:y+new_onomatopoeia_h, x:x+new_onomatopoeia_w] = np.maximum(
                result_mask[y:y+new_onomatopoeia_h, x:x+new_onomatopoeia_w], mask_resized)
            
            # 配置済み領域に追加
            new_region = (x, y, x + new_onomatopoeia_w, y + new_onomatopoeia_h)
            occupied_regions.append(new_region)
            successfully_placed.append(Path(onomatopoeia_path).stem)
            
            # 詳細情報を記録
            onomatopoeia_info = {
                "onomatopoeia_file": Path(onomatopoeia_path).name,
                "original_size": f"{onomatopoeia.shape[1]}x{onomatopoeia.shape[0]}",
                "cropped_size": f"{crop_w}x{crop_h}",
                "final_size": f"{new_onomatopoeia_w}x{new_onomatopoeia_h}",
                "position": f"({x},{y})",
                "scale": f"{onomatopoeia_scale:.6f}",
                "scale_ratio": f"{new_onomatopoeia_w/bg_w:.4f}"
            }
            onomatopoeia_details.append(onomatopoeia_info)
    
    return result_img, result_mask, successfully_placed, onomatopoeia_details


def split_onomatopoeia(onomatopoeia_mask_pairs: list, train_ratio: float = 0.8, seed: int = 42) -> tuple:
    """オノマトペをtrain用とval用に分割する"""
    random.seed(seed)
    shuffled_pairs = onomatopoeia_mask_pairs.copy()
    random.shuffle(shuffled_pairs)
    
    n = len(shuffled_pairs)
    train_count = int(n * train_ratio)
    
    train_pairs = shuffled_pairs[:train_count]
    val_pairs = shuffled_pairs[train_count:]
    
    return train_pairs, val_pairs


def generate_dataset_split(background_files: list, onomatopoeia_pairs: list, 
                          output_dir: str, mask_output_dir: str, split_name: str,
                          target_count: int, cfg: dict, final_output_dir: str = None) -> int:
    """指定されたsplit（trainまたはval）のデータセットを生成する"""
    print(f"\n=== {split_name} データセット生成開始 ===")
    print(f"目標画像数: {target_count}")
    print(f"背景画像数: {len(background_files)}")
    print(f"利用可能オノマトペ数: {len(onomatopoeia_pairs)}")
    
    # ログファイルのパス
    if final_output_dir:
        log_file_path = os.path.join(final_output_dir, f"{split_name}_composition_log.txt")
    else:
        log_file_path = os.path.join(output_dir, f"{split_name}_composition_log.txt")
    
    current_number = 1
    success_count = 0
    bg_idx = 0
    
    # ログファイルを初期化
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== {split_name.upper()} オノマトペデータセット合成ログ ===\n")
        log_file.write(f"生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"目標画像数: {target_count}\n")
        log_file.write(f"背景画像数: {len(background_files)}\n")
        log_file.write(f"利用可能オノマトペ数: {len(onomatopoeia_pairs)}\n")
        log_file.write("=" * 80 + "\n\n")
    
    # 目標数に達するまで生成
    while success_count < target_count:
        # 背景画像を循環使用
        bg_path = background_files[bg_idx % len(background_files)]
        bg_name = Path(bg_path).stem
        
        try:
            # オノマトペ合成実行
            result_img, result_mask, placed_onomatopoeia, onomatopoeia_details = composite_random_onomatopoeia(
                bg_path, 
                onomatopoeia_pairs,
                cfg=cfg
            )
            
            # ファイル保存
            output_img_path = os.path.join(output_dir, f"{current_number:03d}.png")
            output_mask_path = os.path.join(mask_output_dir, f"{current_number:03d}_mask.png")
            
            cv2.imwrite(output_img_path, result_img)
            cv2.imwrite(output_mask_path, result_mask)
            
            # ログファイルに詳細情報を記録
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"画像 {current_number:03d}.png:\n")
                log_file.write(f"  背景ファイル: {Path(bg_path).name}\n")
                log_file.write(f"  背景サイズ: {result_img.shape[1]}x{result_img.shape[0]}\n")
                log_file.write(f"  配置したオノマトペ数: {len(onomatopoeia_details)}\n")
                
                for i, detail in enumerate(onomatopoeia_details, 1):
                    log_file.write(f"    オノマトペ{i}: {detail['onomatopoeia_file']}\n")
                    log_file.write(f"      元サイズ: {detail['original_size']}\n")
                    log_file.write(f"      クロップサイズ: {detail['cropped_size']}\n")
                    log_file.write(f"      最終サイズ: {detail['final_size']}\n")
                    log_file.write(f"      配置位置: {detail['position']}\n")
                    log_file.write(f"      スケール: {detail['scale']}\n")
                    log_file.write(f"      スケール比: {detail['scale_ratio']}\n")
                
                log_file.write("\n")
            
            success_count += 1
            current_number += 1
            
            if success_count % 10 == 0:
                print(f"  進捗: {success_count}/{target_count} 完了")
            
        except Exception as e:
            print(f"✗ 合成失敗 (背景:{bg_name}): {e}")
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"❌ 合成失敗: {bg_name} - {str(e)}\n\n")
        
        bg_idx += 1
    
    print(f"✅ {split_name} 完了: {success_count}個の画像を生成")
    print(f"📄 詳細ログ: {log_file_path}")
    return success_count


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="オノマトペ合成データセット作成")
    parser.add_argument("--onomatopoeia-dir", default="onomatopoeias/images", help="オノマトペ画像ディレクトリ")
    parser.add_argument("--mask-dir", default="onomatopoeias/masks", help="マスク画像ディレクトリ")
    parser.add_argument("--background-dir", default="generated_double_backs_1536x1024", help="背景画像ディレクトリ")
    parser.add_argument("--output-dir", default="onomatopoeia_dataset", help="基本出力ディレクトリ")
    parser.add_argument("--dataset-name", type=str, default="test", help="データセット名 (dataset01, train_v1など)")
    parser.add_argument("--target-images", type=int, default=100, help="生成する画像数")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="train用の比率")
    
    args = parser.parse_args()
    
    # 設定
    CFG = {
        "SCALE_RANGE": (0.0001, 0.005),
        "NUM_ONOMATOPOEIA_RANGE": (1, 15),
        "MAX_ATTEMPTS": 200,
        "TRAIN_RATIO": args.train_ratio,
        "ONOMATOPOEIA_SPLIT_SEED": 42,
        
        # 統計情報ベースのサンプリング設定（バウンディングボックスベース）
        "SCALE_MODE": "lognormal",
        "SCALE_MEAN": 0.005623,              # バウンディングボックス平均面積比
        "SCALE_STD": 0.014176,               # バウンディングボックス標準偏差
        "SCALE_CLIP": (0.0005, 0.020),       # バウンディングボックス範囲（Q25-Max付近）
        "COUNT_PROBS": None,                 # オノマトペ個数の確率分布
        "SCALE_STATS": None,                 # スケール統計情報
        "COUNT_STATS_FILE": "onomatopoeia_statistics.txt",
        
        # 面積ベースリサイズ設定
        "MAX_WIDTH_RATIO": 0.25,             # バウンディングボックスベースなので拡大
        "MAX_HEIGHT_RATIO": 0.25,            # バウンディングボックスベースなので拡大
    }
    
    # 統計情報ファイルの読み込み
    if CFG.get("COUNT_STATS_FILE") and os.path.exists(CFG["COUNT_STATS_FILE"]):
        print(f"\n📊 統計情報ファイルを読み込み中: {CFG['COUNT_STATS_FILE']}")
        CFG["SCALE_STATS"] = load_scale_stats(CFG["COUNT_STATS_FILE"])
    
    # 出力ディレクトリ作成（階層構造：onomatopeia_datasets/dataset01/）
    base_output_dir = args.output_dir
    dataset_name = args.dataset_name
    final_output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 設定をファイル出力
    config_output = {
        "timestamp": datetime.now().isoformat(),
        "script_name": "create_syn_onomatopoeia_dataset.py",
        "dataset_name": dataset_name,
        "base_output_path": base_output_dir,
        "dataset_output_path": final_output_dir,
        "target_images": args.target_images,
        "config": CFG,
        "input_directories": {
            "onomatopoeia_dir": args.onomatopoeia_dir,
            "masks_dir": args.mask_dir,
            "backgrounds_dir": args.background_dir
        }
    }
    
    config_file_path = os.path.join(base_output_dir, f"{dataset_name}_config.json")
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"設定情報を保存: {config_file_path}")
    print(f"📁 出力ディレクトリ: {final_output_dir}")
    
    # 個数分布の読み込み
    try:
        CFG["COUNT_PROBS"] = load_count_probs(CFG["COUNT_STATS_FILE"])
        print(f"✓ 統計ベースのオノマトペ個数サンプリングを有効化")
    except Exception as e:
        print(f"個数分布の読み込みエラー: {e}")
        print("一様サンプリングを使用します")
    
    print("\n=== オノマトペデータセット 作成開始 ===")
    
    # オノマトペとマスクの対応を取得
    onomatopoeia_mask_pairs = []
    print("オノマトペ・マスクペアを検索中...")
    for onomatopoeia_file in os.listdir(args.onomatopoeia_dir):
        if onomatopoeia_file.endswith(('.png', '.jpg', '.jpeg')):
            onomatopoeia_path = os.path.join(args.onomatopoeia_dir, onomatopoeia_file)
            onomatopoeia_stem = Path(onomatopoeia_file).stem
            
            # 対応するマスクファイルを検索
            mask_file = f"{onomatopoeia_stem}_mask.png"
            mask_path = os.path.join(args.mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                onomatopoeia_mask_pairs.append((onomatopoeia_path, mask_path))
    
    # 背景画像を取得
    background_files = []
    for bg_file in os.listdir(args.background_dir):
        if bg_file.endswith(('.png', '.jpg', '.jpeg')):
            background_files.append(os.path.join(args.background_dir, bg_file))
    
    print(f"見つかったオノマトペ: {len(onomatopoeia_mask_pairs)}個")
    print(f"見つかった背景: {len(background_files)}個")
    
    # オノマトペをtrain用とval用に分割
    print(f"\nオノマトペを分割中（train:{CFG['TRAIN_RATIO']:.0%}, val:{1-CFG['TRAIN_RATIO']:.0%}）...")
    train_onomatopoeia, val_onomatopoeia = split_onomatopoeia(
        onomatopoeia_mask_pairs, 
        CFG["TRAIN_RATIO"], 
        CFG["ONOMATOPOEIA_SPLIT_SEED"]
    )
    
    print(f"train用オノマトペ: {len(train_onomatopoeia)}個")
    print(f"val用オノマトペ: {len(val_onomatopoeia)}個")
    
    # 目標画像数を計算
    train_target = int(args.target_images * CFG["TRAIN_RATIO"])
    val_target = args.target_images - train_target
    
    print(f"\n目標画像数:")
    print(f"train: {train_target}枚")
    print(f"val: {val_target}枚")
    print(f"合計: {args.target_images}枚")
    
    # train データセット生成
    train_img_dir = os.path.join(final_output_dir, "train", "images")
    train_mask_dir = os.path.join(final_output_dir, "train", "masks")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    
    train_count = generate_dataset_split(
        background_files, train_onomatopoeia,
        train_img_dir, train_mask_dir, "train", train_target, 
        CFG, final_output_dir
    )
    
    # val データセット生成
    val_img_dir = os.path.join(final_output_dir, "val", "images")
    val_mask_dir = os.path.join(final_output_dir, "val", "masks")
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    
    val_count = generate_dataset_split(
        background_files, val_onomatopoeia,
        val_img_dir, val_mask_dir, "val", val_target, 
        CFG, final_output_dir
    )
    
    # 最終レポート
    print(f"\n=== オノマトペデータセット 作成完了 ===")
    print(f"出力先: {final_output_dir}")
    print(f"総生成画像数: {train_count + val_count}枚")
    
    # 統計情報を収集
    stats = {
        "total_images": train_count + val_count,
        "train_images": train_count,
        "val_images": val_count,
        "train_onomatopoeia_used": len(train_onomatopoeia),
        "val_onomatopoeia_used": len(val_onomatopoeia),
        "total_backgrounds_available": len(background_files),
        "total_onomatopoeia_pairs_available": len(onomatopoeia_mask_pairs)
    }
    
    # 統計を表示
    for split in ["train", "val"]:
        img_count = len(list(Path(final_output_dir).glob(f"{split}/images/*.png")))
        mask_count = len(list(Path(final_output_dir).glob(f"{split}/masks/*.png")))
        print(f"{split}: {img_count} 画像, {mask_count} マスク")
    
    # 統計情報をconfig.jsonに追加
    config_output["statistics"] = stats
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    
    print(f"\n=== オノマトペ使用状況 ===")
    print(f"train用オノマトペ: {len(train_onomatopoeia)}個")
    print(f"val用オノマトペ: {len(val_onomatopoeia)}個")
    print(f"重複なし: train と val で異なるオノマトペを使用")
    print(f"設定・統計情報: {config_file_path}")


if __name__ == "__main__":
    main()
