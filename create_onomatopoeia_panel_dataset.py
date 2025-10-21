"""
オノマトペをコマ内に配置する合成データセット作成スクリプト

create_corner_aligned_dataset.pyのパネル検出機能を使用して、
オノマトペをコマの内側にランダムに配置する
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
import argparse
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Point:
    """座標を表すクラス"""
    x: int
    y: int
    
    def __iter__(self):
        return iter((self.x, self.y))


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


def detect_panels_simple(image: np.ndarray, 
                        area_ratio_threshold: float = 0.85,
                        min_area: int = 10000) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    シンプルな二値化・輪郭抽出によるコマ検出
    
    Args:
        image: 入力画像（カラー）
        area_ratio_threshold: 輪郭面積/バウンディングボックス面積の閾値
        min_area: 最小コマ面積
    
    Returns:
        List[(panel_mask, bbox)]: パネルマスク、バウンディングボックスのリスト
    """
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二値化（Otsuの自動閾値）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ノイズ除去（モルフォロジー演算）
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    panels = []
    
    for contour in contours:
        # 輪郭の面積を計算
        contour_area = cv2.contourArea(contour)
        
        # 小さすぎる輪郭は無視
        if contour_area < min_area:
            continue
        
        # バウンディングボックスを取得
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        
        # 面積比を計算
        area_ratio = contour_area / bbox_area if bbox_area > 0 else 0
        
        # 面積比が閾値以上の場合、コマとして認識
        if area_ratio >= area_ratio_threshold:
            # パネルマスクを作成
            panel_mask = np.zeros((h, w), dtype=np.uint8)
            
            # 輪郭を相対座標に変換してマスクに描画
            contour_relative = contour - np.array([x, y])
            cv2.drawContours(panel_mask, [contour_relative], -1, 255, -1)
            
            panels.append((panel_mask, (x, y, w, h)))
    
    return panels


def sample_scale(bg_w: int, cfg: dict) -> float:
    """統計情報に基づいてスケールをサンプリング（バウンディングボックスベース）"""
    mode = cfg.get("SCALE_MODE", "uniform")
    if mode == "lognormal":
        clip_min, clip_max = cfg["SCALE_CLIP"]
        
        scale_stats = cfg.get("SCALE_STATS", {})
        target_median = scale_stats.get("bbox_median", 0.001488)
        sigma = 0.8
        
        mu = np.log(target_median)
        s = np.random.lognormal(mu, sigma)
        clipped = float(np.clip(s, clip_min, clip_max))
        return clipped
    else:
        return random.uniform(*cfg["SCALE_RANGE"])


def calculate_area_based_size(crop_w: int, crop_h: int, bg_w: int, bg_h: int, 
                             target_scale: float, mask: np.ndarray = None,
                             max_w_ratio: float = 0.25, 
                             max_h_ratio: float = 0.25) -> tuple:
    """面積ベースのリサイズサイズ計算"""
    bg_area = bg_w * bg_h
    target_area = bg_area * target_scale
    
    # アスペクト比 = 高さ/幅（create_syn_dataset.pyと同じ定義）
    aspect_ratio = crop_h / crop_w
    
    # アスペクト比を維持した理想サイズ
    ideal_w = int(np.sqrt(target_area / aspect_ratio))
    ideal_h = int(np.sqrt(target_area * aspect_ratio))
    
    # 最大サイズ制限
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


def place_onomatopoeia_in_panel(panel_image: np.ndarray, panel_mask: np.ndarray,
                               panel_bbox: Tuple[int, int, int, int],
                               onomatopoeia: np.ndarray, onomatopoeia_mask: np.ndarray,
                               target_scale: float, bg_w: int, bg_h: int, cfg: dict) -> Tuple[bool, Optional[Dict]]:
    """
    オノマトペをコマ内に配置
    
    Args:
        panel_image: パネル画像
        panel_mask: パネルマスク
        panel_bbox: パネルのバウンディングボックス (x, y, w, h)
        onomatopoeia: オノマトペ画像
        onomatopoeia_mask: オノマトペマスク
        target_scale: ターゲットスケール（背景画像全体に対する比率）
        bg_w: 背景画像の幅
        bg_h: 背景画像の高さ
        cfg: 設定
    
    Returns:
        (成功フラグ, 配置情報)
    """
    panel_x, panel_y, panel_w, panel_h = panel_bbox
    
    # クロップ
    cropped_ono, cropped_mask, _ = crop_onomatopoeia_and_mask(onomatopoeia, onomatopoeia_mask)
    
    if cropped_ono.size == 0:
        return False, None
    
    crop_h, crop_w = cropped_ono.shape[:2]
    
    # サイズ計算（背景画像全体のサイズを使用）
    new_w, new_h = calculate_area_based_size(
        crop_w, crop_h, bg_w, bg_h, target_scale,
        mask=cropped_mask,
        max_w_ratio=cfg.get("MAX_WIDTH_RATIO", 0.25),
        max_h_ratio=cfg.get("MAX_HEIGHT_RATIO", 0.25)
    )
    
    # コマより大きくなったらスキップ
    if new_w >= panel_w or new_h >= panel_h:
        return False, None
    
    # リサイズ
    ono_resized = cv2.resize(cropped_ono, (new_w, new_h))
    mask_resized = cv2.resize(cropped_mask, (new_w, new_h))
    
    # コマ内のランダム位置に配置
    max_x = max(0, panel_w - new_w)
    max_y = max(0, panel_h - new_h)
    
    if max_x <= 0 or max_y <= 0:
        return False, None
    
    x_in_panel = random.randint(0, max_x)
    y_in_panel = random.randint(0, max_y)
    
    # グローバル座標に変換
    x_global = panel_x + x_in_panel
    y_global = panel_y + y_in_panel
    
    # コマ内に収まるかチェック（マスク領域）
    mask_region = mask_resized > 0
    if not np.all(mask_region <= panel_mask[y_in_panel:y_in_panel+new_h, x_in_panel:x_in_panel+new_w]):
        return False, None
    
    placement_info = {
        "size": (new_w, new_h),
        "position": (x_global, y_global),
        "position_in_panel": (x_in_panel, y_in_panel),
        "scale": float(target_scale)
    }
    
    return True, placement_info


def composite_onomatopoeia_in_panels(background_path: str, onomatopoeia_mask_pairs: list,
                                    cfg: dict = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    背景画像のコマを検出し、各コマにオノマトペを配置
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
    
    # パネル検出
    panels = detect_panels_simple(background,
                                 area_ratio_threshold=cfg.get("PANEL_AREA_RATIO", 0.70),
                                 min_area=cfg.get("PANEL_MIN_AREA", 5000))
    
    if len(panels) == 0:
        # フォールバック: 背景画像全体をパネルとして扱う
        full_panel_mask = np.ones((bg_h, bg_w), dtype=np.uint8) * 255
        panels = [(full_panel_mask, (0, 0, bg_w, bg_h))]
    
    placements = []
    
    # 各パネルにオノマトペを配置
    for panel_mask, panel_bbox in panels:
        panel_x, panel_y, panel_w, panel_h = panel_bbox
        
        # このパネルに配置するオノマトペ数を決定
        num_onomatopoeia = random.randint(
            cfg.get("ONOMATOPOEIA_PER_PANEL", (1, 3))[0],
            cfg.get("ONOMATOPOEIA_PER_PANEL", (1, 3))[1]
        )
        
        # ランダムに選択
        selected_pairs = random.sample(onomatopoeia_mask_pairs, 
                                      min(num_onomatopoeia, len(onomatopoeia_mask_pairs)))
        
        placed_regions = []  # このパネル内での配置済み領域
        
        for ono_path, mask_path in selected_pairs:
            onomatopoeia = cv2.imread(ono_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if onomatopoeia is None or mask is None:
                continue
            
            # スケール決定
            ono_scale = sample_scale(bg_w, cfg)
            
            # 配置試行
            max_attempts = cfg.get("MAX_ATTEMPTS", 50)
            placed = False
            
            for attempt in range(max_attempts):
                success, placement_info = place_onomatopoeia_in_panel(
                    panel_mask, panel_mask, panel_bbox,
                    onomatopoeia, mask, ono_scale, bg_w, bg_h, cfg
                )
                
                if not success:
                    continue
                
                # 重複チェック
                x, y = placement_info["position"]
                w, h = placement_info["size"]
                new_region = (x, y, x + w, y + h)
                
                # パネル内の他のオノマトペとの重複確認
                has_overlap = False
                for placed_region in placed_regions:
                    if regions_overlap(new_region, placed_region):
                        has_overlap = True
                        break
                
                if not has_overlap:
                    # 配置実行
                    x_in_panel, y_in_panel = placement_info["position_in_panel"]
                    w, h = placement_info["size"]
                    
                    # 画像領域取得
                    ono_resized = cv2.resize(
                        cv2.imread(ono_path, cv2.IMREAD_COLOR),
                        (w, h)
                    )
                    mask_resized = cv2.resize(
                        cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE),
                        (w, h)
                    )
                    
                    # クロップ版を使用
                    ono_cropped, mask_cropped, _ = crop_onomatopoeia_and_mask(ono_resized, mask_resized)
                    
                    # グローバル座標で配置
                    x_global = placement_info["position"][0]
                    y_global = placement_info["position"][1]
                    w_ono, h_ono = ono_cropped.shape[1], ono_cropped.shape[0]
                    
                    # アルファブレンディング
                    mask_norm = mask_cropped.astype(np.float32) / 255.0
                    mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
                    
                    bg_region = result_img[y_global:y_global+h_ono, x_global:x_global+w_ono]
                    blended = ono_cropped.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
                    result_img[y_global:y_global+h_ono, x_global:x_global+w_ono] = blended.astype(np.uint8)
                    
                    # マスク合成
                    result_mask[y_global:y_global+h_ono, x_global:x_global+w_ono] = np.maximum(
                        result_mask[y_global:y_global+h_ono, x_global:x_global+w_ono],
                        mask_cropped
                    )
                    
                    placed_regions.append(new_region)
                    placements.append({
                        "onomatopoeia_file": Path(ono_path).name,
                        "panel_bbox": panel_bbox,
                        "position": placement_info["position"],
                        "size": placement_info["size"],
                        "scale": placement_info["scale"]
                    })
                    
                    placed = True
                    break
    
    return result_img, result_mask, placements


def load_scale_stats(path: str) -> dict:
    """統計ファイルからスケール統計情報を読み込む（バウンディングボックスベース）"""
    stats = {
        "bbox_mean": None,
        "bbox_median": None,
        "bbox_std": None,
        "bbox_q25": None,
        "bbox_q75": None,
        "type": "bbox"
    }
    
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        
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
        print(f"  中央値: {stats['bbox_median']:.6f}")
        return stats
    except Exception as e:
        print(f"統計ファイル読み込みエラー: {e}")
        return stats


def _json_default(o):
    """JSON serialization helper"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def main():
    parser = argparse.ArgumentParser(description="オノマトペをコマ内に配置するデータセット作成")
    parser.add_argument("--onomatopoeia-dir", default="onomatopeias", help="オノマトペ画像ディレクトリ")
    parser.add_argument("--mask-dir", default="onomatopeia_masks", help="マスク画像ディレクトリ")
    parser.add_argument("--background-dir", default="generated_double_backs", help="背景画像ディレクトリ")
    parser.add_argument("--output-dir", default="onomatopeia_datasets", help="基本出力ディレクトリ")
    parser.add_argument("--dataset-name", type=str, default="panel_dataset01", help="データセット名")
    parser.add_argument("--target-images", type=int, default=100, help="生成する画像数")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="train用の比率")
    
    args = parser.parse_args()
    
    # 設定
    CFG = {
        "SCALE_RANGE": (0.0001, 0.005),
        "SCALE_MODE": "lognormal",
        "SCALE_MEAN": 0.005623,
        "SCALE_STD": 0.014176,
        "SCALE_CLIP": (0.001, 0.030),  # より広い範囲に拡大（0.001～0.030 = 0.1%～3%）
        "COUNT_STATS_FILE": "onomatopoeia_statistics.txt",
        "MAX_WIDTH_RATIO": 0.25,
        "MAX_HEIGHT_RATIO": 0.25,
        "MAX_ATTEMPTS": 50,
        "ONOMATOPOEIA_PER_PANEL": (1, 3),  # パネルあたりのオノマトペ数
        "PANEL_AREA_RATIO": 0.70,  # 0.85から0.70に緩和
        "PANEL_MIN_AREA": 5000,    # 10000から5000に緩和
        "SCALE_STATS": None,
    }
    
    # 統計情報の読み込み
    stats_file = Path(CFG["COUNT_STATS_FILE"])
    if stats_file.exists():
        print(f"📊 統計情報ファイルを読み込み中: {stats_file}")
        CFG["SCALE_STATS"] = load_scale_stats(str(stats_file))
    
    # ディレクトリ作成
    base_output_dir = args.output_dir
    dataset_name = args.dataset_name
    final_output_dir = Path(base_output_dir) / dataset_name
    
    os.makedirs(final_output_dir, exist_ok=True)
    
    print(f"\n📁 出力ディレクトリ: {final_output_dir}")
    print(f"🎯 目標画像数: {args.target_images}枚")
    
    # オノマトペとマスクのペア取得
    onomatopoeia_mask_pairs = []
    for ono_file in os.listdir(args.onomatopoeia_dir):
        if ono_file.endswith(('.png', '.jpg', '.jpeg')):
            ono_path = os.path.join(args.onomatopoeia_dir, ono_file)
            ono_stem = Path(ono_file).stem
            mask_file = f"{ono_stem}_mask.png"
            mask_path = os.path.join(args.mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                onomatopoeia_mask_pairs.append((ono_path, mask_path))
    
    print(f"見つかったオノマトペペア: {len(onomatopoeia_mask_pairs)}個")
    
    # 背景画像取得
    background_files = []
    for bg_file in os.listdir(args.background_dir):
        if bg_file.endswith(('.png', '.jpg', '.jpeg')):
            background_files.append(os.path.join(args.background_dir, bg_file))
    
    print(f"見つかった背景画像: {len(background_files)}個")
    
    # train/valディレクトリ作成
    for split in ["train", "val"]:
        (final_output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (final_output_dir / split / "masks").mkdir(parents=True, exist_ok=True)
    
    # 画像生成
    train_count = 0
    val_count = 0
    train_target = int(args.target_images * args.train_ratio)
    val_target = args.target_images - train_target
    
    train_number = 1
    val_number = 1
    bg_idx = 0
    
    print(f"\n🚀 オノマトペ配置開始...")
    print(f"目標: train {train_target}枚, val {val_target}枚")
    
    while train_count < train_target or val_count < val_target:
        if train_count >= train_target and val_count >= val_target:
            break
        
        # 背景画像を循環使用
        bg_path = background_files[bg_idx % len(background_files)]
        bg_idx += 1
        
        try:
            # オノマトペ配置
            result_img, result_mask, placements = composite_onomatopoeia_in_panels(
                bg_path, onomatopoeia_mask_pairs, CFG
            )
            
            if len(placements) == 0:
                continue  # 配置できなかった場合はスキップ
            
            # train/valに振り分け
            if train_count < train_target:
                split = "train"
                out_num = f"{train_number:03d}"
                train_number += 1
                train_count += 1
            elif val_count < val_target:
                split = "val"
                out_num = f"{val_number:03d}"
                val_number += 1
                val_count += 1
            else:
                continue  # 割り当て済みならスキップ
            
            # 保存
            img_path = final_output_dir / split / "images" / f"{out_num}.png"
            mask_path = final_output_dir / split / "masks" / f"{out_num}.png"
            
            cv2.imwrite(str(img_path), result_img)
            cv2.imwrite(str(mask_path), result_mask)
            
        except Exception as e:
            print(f"❌ 合成失敗: {Path(bg_path).name} - {str(e)}")
    
    print(f"\n✅ 生成完了")
    print(f"train: {train_count}枚, val: {val_count}枚")
    print(f"合計: {train_count + val_count}枚")


if __name__ == "__main__":
    main()
