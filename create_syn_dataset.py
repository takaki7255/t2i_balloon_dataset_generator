"""
吹き出しをtrain用とval用に分けて合成し、syn_datasetを作成するスクリプト
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

def regions_overlap(region1: tuple, region2: tuple) -> bool:
    """2つの領域が重複するかチェック"""
    x1_min, y1_min, x1_max, y1_max = region1
    x2_min, y2_min, x2_max, y2_max = region2
    
    return not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)


def calculate_overlap_area(region1: tuple, region2: tuple) -> int:
    """2つの領域の重複面積を計算"""
    x1_min, y1_min, x1_max, y1_max = region1
    x2_min, y2_min, x2_max, y2_max = region2
    
    # 重複領域の計算
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
        mean = cfg["SCALE_MEAN"]
        std = cfg["SCALE_STD"]
        clip_min, clip_max = cfg["SCALE_CLIP"]
        mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
        sigma = np.sqrt(np.log(1 + (std**2)/(mean**2)))
        s = np.random.lognormal(mu, sigma)
        return float(np.clip(s, clip_min, clip_max))
    else:
        return random.uniform(*cfg["SCALE_RANGE"])
    
def sample_num_balloons(cfg: dict, max_available: int) -> int:
    """
    統計情報に基づいて吹き出し個数をサンプリング
    cfg["COUNT_PROBS"] があればその分布から取得。
    無ければ NUM_BALLOONS_RANGE から一様サンプル。
    """
    lower, upper = cfg.get("NUM_BALLOONS_RANGE", (2, 10))
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

    if not hist:
        return None
        
    max_n = max(hist.keys())
    arr = np.zeros(max_n + 1, dtype=np.float32)
    for k, v in hist.items():
        arr[k] = v
    probs = arr / arr.sum()
    return probs


def _json_default(o):
    """JSON serialization helper"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def composite_random_balloons_enhanced(background_path: str, balloon_mask_pairs: list,
                                     scale_range: tuple = (0.1, 0.4), 
                                     num_balloons_range: tuple = (2, 10),
                                     max_attempts: int = 200,
                                     cfg: dict = None) -> tuple:
    """
    1つの背景画像にランダムに選択した複数の吹き出しを重複なしで合成する
    統計情報に基づくサンプリング対応版
    """
    if cfg is None:
        cfg = {}
        
    # 背景画像読み込み
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")
    
    bg_h, bg_w = background.shape[:2]
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    # 配置する吹き出し数を統計情報に基づいて決定
    max_balloons = min(num_balloons_range[1], len(balloon_mask_pairs))
    num_balloons = sample_num_balloons(cfg, max_balloons)
    
    # ランダムに吹き出しを選択
    selected_pairs = random.sample(balloon_mask_pairs, num_balloons)
    
    # 配置済み領域を記録する配列
    occupied_regions = []
    successfully_placed = []
    
    for balloon_path, mask_path in selected_pairs:
        # 吹き出しとマスク読み込み
        balloon = cv2.imread(balloon_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if balloon is None or mask is None:
            continue
        
        # マスクの境界ボックスでクロップ（余白除去）
        cropped_balloon, cropped_mask, bbox = crop_balloon_and_mask(balloon, mask)
        
        if cropped_balloon.size == 0 or cropped_mask.size == 0:
            print(f"警告: クロップ結果が空 ({balloon_path})")
            continue

        # クロップされた画像のサイズでスケール計算
        crop_h, crop_w = cropped_balloon.shape[:2]
        
        # 統計情報に基づくスケールサンプリング
        try:
            balloon_scale = sample_scale(bg_w, crop_w, cfg)
        except KeyError:
            balloon_scale = random.uniform(scale_range[0], scale_range[1])
            
        new_balloon_w = int(bg_w * balloon_scale)
        new_balloon_h = int(crop_h * (new_balloon_w / crop_w))
        
        # 背景サイズを超える場合はスキップ
        if new_balloon_w >= bg_w or new_balloon_h >= bg_h:
            continue
        
        # クロップされた画像をリサイズ
        balloon_resized = cv2.resize(cropped_balloon, (new_balloon_w, new_balloon_h))
        mask_resized = cv2.resize(cropped_mask, (new_balloon_w, new_balloon_h))
        
        # 重複を避けつつ位置を探す
        placed = False
        best_position = None
        min_overlap_ratio = float('inf')
        
        # まず重複回避を試行
        for attempt in range(max_attempts // 2):
            max_x = max(0, bg_w - new_balloon_w)
            max_y = max(0, bg_h - new_balloon_h)
            
            if max_x <= 0 or max_y <= 0:
                break
                
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # 新しい領域
            new_region = (x, y, x + new_balloon_w, y + new_balloon_h)
            
            # 既存領域との重複チェック
            max_overlap_ratio = 0
            for occupied in occupied_regions:
                if regions_overlap(new_region, occupied):
                    overlap_area = calculate_overlap_area(new_region, occupied)
                    new_area = new_balloon_w * new_balloon_h
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
            max_x = max(0, bg_w - new_balloon_w)
            max_y = max(0, bg_h - new_balloon_h)
            if max_x >= 0 and max_y >= 0:
                x = random.randint(0, max_x) if max_x > 0 else 0
                y = random.randint(0, max_y) if max_y > 0 else 0
                placed = True
        
        if placed:
            # 合成実行
            mask_norm = mask_resized.astype(np.float32) / 255.0
            mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
            
            # 背景画像の該当領域
            bg_region = result_img[y:y+new_balloon_h, x:x+new_balloon_w]
            
            # アルファブレンディング
            blended = balloon_resized.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
            result_img[y:y+new_balloon_h, x:x+new_balloon_w] = blended.astype(np.uint8)
            
            # マスクも合成
            result_mask[y:y+new_balloon_h, x:x+new_balloon_w] = np.maximum(
                result_mask[y:y+new_balloon_h, x:x+new_balloon_w], mask_resized)
            
            # 配置済み領域に追加
            new_region = (x, y, x + new_balloon_w, y + new_balloon_h)
            occupied_regions.append(new_region)
            successfully_placed.append(Path(balloon_path).stem)
    
    return result_img, result_mask, successfully_placed


def split_balloons(balloon_mask_pairs: list, train_ratio: float = 0.8, seed: int = 42) -> tuple:
    """
    吹き出しをtrain用とval用に分割する
    
    Args:
        balloon_mask_pairs: 吹き出し・マスクペアのリスト
        train_ratio: train用の比率
        seed: ランダムシード
    
    Returns:
        (train用ペア, val用ペア)
    """
    random.seed(seed)
    shuffled_pairs = balloon_mask_pairs.copy()
    random.shuffle(shuffled_pairs)
    
    n = len(shuffled_pairs)
    train_count = int(n * train_ratio)
    
    train_pairs = shuffled_pairs[:train_count]
    val_pairs = shuffled_pairs[train_count:]
    
    return train_pairs, val_pairs


def generate_dataset_split(background_files: list, balloon_pairs: list, 
                          output_dir: str, mask_output_dir: str, split_name: str,
                          target_count: int, cfg: dict) -> int:
    """
    指定されたsplit（trainまたはval）のデータセットを生成する
    """
    print(f"\n=== {split_name} データセット生成開始 ===")
    print(f"目標画像数: {target_count}")
    print(f"背景画像数: {len(background_files)}")
    print(f"利用可能吹き出し数: {len(balloon_pairs)}")
    
    if len(balloon_pairs) < cfg["NUM_BALLOONS_RANGE"][1]:
        adjusted_max = len(balloon_pairs)
        print(f"吹き出し数を調整: 最大{adjusted_max}個")
        current_range = (min(cfg["NUM_BALLOONS_RANGE"][0], adjusted_max), adjusted_max)
    else:
        current_range = cfg["NUM_BALLOONS_RANGE"]
    
    current_number = 1
    success_count = 0
    bg_idx = 0
    
    # 目標数に達するまで生成
    while success_count < target_count:
        # 背景画像を循環使用
        bg_path = background_files[bg_idx % len(background_files)]
        bg_name = Path(bg_path).stem
        
        try:
            # ランダム複数合成実行（統計情報対応）
            result_img, result_mask, placed_balloons = composite_random_balloons_enhanced(
                bg_path, 
                balloon_pairs,
                scale_range=cfg["SCALE_RANGE"],
                num_balloons_range=current_range,
                max_attempts=cfg["MAX_ATTEMPTS"],
                cfg=cfg  # 統計情報パラメータを渡す
            )
            
            # ファイル保存
            output_img_path = os.path.join(output_dir, f"{current_number:03d}.png")
            output_mask_path = os.path.join(mask_output_dir, f"{current_number:03d}_mask.png")
            
            cv2.imwrite(output_img_path, result_img)
            cv2.imwrite(output_mask_path, result_mask)
            
            success_count += 1
            current_number += 1
            
            if success_count % 10 == 0:  # 10枚ごとに進捗表示
                print(f"  進捗: {success_count}/{target_count} 完了")
            
        except Exception as e:
            print(f"✗ 合成失敗 (背景:{bg_name}): {e}")
        
        bg_idx += 1
    
    print(f"✅ {split_name} 完了: {success_count}個の画像を生成")
    return success_count


def main():
    """メイン処理"""
    
    # パス設定
    balloons_dir = "generated_balloons"
    masks_dir = "masks"
    backgrounds_dir = "generated_double_backs"
    temp_output_dir = "temp_syn_results"
    temp_mask_output_dir = "temp_syn_results_mask"
    final_output_dir = "syn_mihiraki300_dataset"
    
    # 設定
    CFG = {
        "SCALE_RANGE": (0.1, 0.3),          # 吹き出しのスケール範囲
        "NUM_BALLOONS_RANGE": (2, 7),       # 1画像あたりの吹き出し数
        "MAX_ATTEMPTS": 200,                 # 配置試行回数
        "TARGET_TOTAL_IMAGES": 300,          # 総生成画像数
        "TRAIN_RATIO": 0.8,                  # train用の比率
        "BALLOON_SPLIT_SEED": 42,            # 吹き出し分割のランダムシード
        
        # 統計情報ベースのサンプリング設定
        "SCALE_MODE": "lognormal",           # "uniform" or "lognormal" 
        "SCALE_MEAN": 0.25,                  # lognormal分布のmean (SCALE_MODE="lognormal"時のみ)
        "SCALE_STD": 0.08,                   # lognormal分布のstd (SCALE_MODE="lognormal"時のみ)
        "SCALE_CLIP": (0.05, 0.4),           # スケールのクリップ範囲
        "COUNT_PROBS": None,                 # 吹き出し個数の確率分布 (load_count_probs()で設定可能)
        "COUNT_STATS_FILE": "balloon_count_statistics.txt",  # 統計ファイルのパス
    }
    
    # ディレクトリ作成
    os.makedirs(temp_output_dir, exist_ok=True)
    os.makedirs(temp_mask_output_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # CFG設定をファイル出力
    config_output = {
        "timestamp": datetime.now().isoformat(),
        "script_name": "create_syn_dataset.py",
        "dataset_output_path": final_output_dir,
        "config": CFG,
        "input_directories": {
            "balloons_dir": balloons_dir,
            "masks_dir": masks_dir,
            "backgrounds_dir": backgrounds_dir
        }
    }
    
    config_file_path = os.path.join(final_output_dir, "config.json")
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"設定情報を保存: {config_file_path}")
    
    # 統計情報ファイルの読み込み（オプション）
    if CFG.get("COUNT_STATS_FILE") and os.path.exists(CFG["COUNT_STATS_FILE"]):
        print(f"統計情報ファイルを読み込み: {CFG['COUNT_STATS_FILE']}")
        try:
            CFG["COUNT_PROBS"] = load_count_probs(CFG["COUNT_STATS_FILE"])
            print(f"統計ベースの吹き出し個数サンプリングを有効化")
        except Exception as e:
            print(f"統計ファイル読み込みエラー: {e}")
            print("一様サンプリングを使用します")
    
    print("=== syn_dataset 作成開始 ===")
    
    # 吹き出しとマスクの対応を取得
    balloon_mask_pairs = []
    print("吹き出し・マスクペアを検索中...")
    for balloon_file in os.listdir(balloons_dir):
        if balloon_file.endswith(('.png', '.jpg', '.jpeg')):
            balloon_path = os.path.join(balloons_dir, balloon_file)
            balloon_stem = Path(balloon_file).stem
            
            # 対応するマスクファイルを検索
            mask_file = f"{balloon_stem}_mask.png"
            mask_path = os.path.join(masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                balloon_mask_pairs.append((balloon_path, mask_path))
    
    # 背景画像を取得
    background_files = []
    for bg_file in os.listdir(backgrounds_dir):
        if bg_file.endswith(('.png', '.jpg', '.jpeg')):
            background_files.append(os.path.join(backgrounds_dir, bg_file))
    
    print(f"見つかった吹き出し: {len(balloon_mask_pairs)}個")
    print(f"見つかった背景: {len(background_files)}個")
    
    # 吹き出しをtrain用とval用に分割
    print(f"\n吹き出しを分割中（train:{CFG['TRAIN_RATIO']:.0%}, val:{1-CFG['TRAIN_RATIO']:.0%}）...")
    train_balloons, val_balloons = split_balloons(
        balloon_mask_pairs, 
        CFG["TRAIN_RATIO"], 
        CFG["BALLOON_SPLIT_SEED"]
    )
    
    print(f"train用吹き出し: {len(train_balloons)}個")
    print(f"val用吹き出し: {len(val_balloons)}個")
    
    # 目標画像数を計算
    train_target = int(CFG["TARGET_TOTAL_IMAGES"] * CFG["TRAIN_RATIO"])
    val_target = CFG["TARGET_TOTAL_IMAGES"] - train_target
    
    print(f"\n目標画像数:")
    print(f"train: {train_target}枚")
    print(f"val: {val_target}枚")
    print(f"合計: {CFG['TARGET_TOTAL_IMAGES']}枚")
    
    # train データセット生成
    train_temp_img_dir = os.path.join(temp_output_dir, "train")
    train_temp_mask_dir = os.path.join(temp_mask_output_dir, "train")
    os.makedirs(train_temp_img_dir, exist_ok=True)
    os.makedirs(train_temp_mask_dir, exist_ok=True)
    
    train_count = generate_dataset_split(
        background_files, train_balloons,
        train_temp_img_dir, train_temp_mask_dir, "train", train_target, CFG
    )
    
    # val データセット生成
    val_temp_img_dir = os.path.join(temp_output_dir, "val")
    val_temp_mask_dir = os.path.join(temp_mask_output_dir, "val")
    os.makedirs(val_temp_img_dir, exist_ok=True)
    os.makedirs(val_temp_mask_dir, exist_ok=True)
    
    val_count = generate_dataset_split(
        background_files, val_balloons,
        val_temp_img_dir, val_temp_mask_dir, "val", val_target, CFG
    )
    
    # 最終的なデータセット構造を作成
    print(f"\n=== 最終データセット構造を作成中 ===")
    
    for split in ["train", "val"]:
        # ディレクトリ作成
        (Path(final_output_dir) / split / "images").mkdir(parents=True, exist_ok=True)
        (Path(final_output_dir) / split / "masks").mkdir(parents=True, exist_ok=True)
        
        # ファイルをコピー
        temp_img_dir = os.path.join(temp_output_dir, split)
        temp_mask_dir = os.path.join(temp_mask_output_dir, split)
        final_img_dir = os.path.join(final_output_dir, split, "images")
        final_mask_dir = os.path.join(final_output_dir, split, "masks")
        
        # 画像をコピー
        for img_file in os.listdir(temp_img_dir):
            if img_file.endswith('.png'):
                shutil.copy2(
                    os.path.join(temp_img_dir, img_file),
                    os.path.join(final_img_dir, img_file)
                )
        
        # マスクをコピー
        for mask_file in os.listdir(temp_mask_dir):
            if mask_file.endswith('.png'):
                shutil.copy2(
                    os.path.join(temp_mask_dir, mask_file),
                    os.path.join(final_mask_dir, mask_file)
                )
    
    # 一時ディレクトリを削除
    shutil.rmtree(temp_output_dir)
    shutil.rmtree(temp_mask_output_dir)
    
    # 最終レポート
    print(f"\n=== syn_dataset 作成完了 ===")
    print(f"出力先: {final_output_dir}")
    print(f"総生成画像数: {train_count + val_count}枚")
    
    # 統計情報を収集
    stats = {
        "total_images": train_count + val_count,
        "train_images": train_count,
        "val_images": val_count,
        "train_balloons_used": len(train_balloons),
        "val_balloons_used": len(val_balloons),
        "total_backgrounds_available": len(background_files),
        "total_balloon_pairs_available": len(balloon_mask_pairs)
    }
    
    for split in ["train", "val"]:
        img_count = len(list(Path(final_output_dir).glob(f"{split}/images/*.png")))
        mask_count = len(list(Path(final_output_dir).glob(f"{split}/masks/*.png")))
        print(f"{split}: {img_count} 画像, {mask_count} マスク")
    
    # 統計情報をconfig.jsonに追加
    config_output["statistics"] = stats
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    
    print(f"\n=== 吹き出し使用状況 ===")
    print(f"train用吹き出し: {len(train_balloons)}個")
    print(f"val用吹き出し: {len(val_balloons)}個")
    print(f"重複なし: train と val で異なる吹き出しを使用")
    print(f"設定・統計情報: {config_file_path}")


if __name__ == "__main__":
    main()
