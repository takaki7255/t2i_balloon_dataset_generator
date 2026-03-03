"""
オノマトペ合成データセット作成スクリプト（統合版）

機能:
- マスクベースのクロップ（デフォルト）
- 面積ベースのスケール計算
- パネル内配置（デフォルト）：検出したコマ内のランダム位置に配置
- マンガ特有のデータ拡張（--augmentation オプション）
- 統計情報からのサンプリング

使用例:
  # 基本的な使用方法（デフォルトでパネル内配置を使用）
  python create_syn_onomatopoeia_dataset.py --output-dir onomatopoeia_dataset/syn500 --target-images 500

  # パネル配置を無効化（完全ランダム配置）
  python create_syn_onomatopoeia_dataset.py --output-dir onomatopoeia_dataset/syn500-random --target-images 500 --no-panel-placement

  # データ拡張を有効化（マンガ特有のaugmentation）
  python create_syn_onomatopoeia_dataset.py --output-dir onomatopoeia_dataset/syn500-aug --target-images 500 --augmentation
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
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


# =============================================================================
# 共通ユーティリティ関数
# =============================================================================

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


def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """マスクの非ゼロ領域の境界ボックスを取得"""
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def crop_onomatopoeia_and_mask(onomatopoeia: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """マスクの境界ボックスに基づいてオノマトペ画像とマスクをクロップ"""
    x, y, w, h = get_mask_bbox(mask)
    
    cropped_onomatopoeia = onomatopoeia[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    return cropped_onomatopoeia, cropped_mask, (x, y, w, h)


def load_scale_stats(path: str) -> dict:
    """統計ファイルからスケール統計情報を読み込む"""
    stats = {
        "bbox_mean": None,
        "bbox_median": None,
        "bbox_std": None,
        "bbox_q25": None,
        "bbox_q75": None,
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
        
        print(f"✓ 統計情報を読み込み: {path}")
        if stats["bbox_median"]:
            print(f"  中央値: {stats['bbox_median']:.6f}, 平均: {stats['bbox_mean']:.6f}")
        return stats
    except Exception as e:
        print(f"統計ファイル読み込みエラー: {e}")
        return stats


def sample_scale(bg_w: int, bw: int, cfg: dict) -> float:
    """統計情報に基づいてスケールをサンプリング"""
    mode = cfg.get("SCALE_MODE", "uniform")
    if mode == "lognormal":
        clip_min, clip_max = cfg["SCALE_CLIP"]
        
        scale_stats = cfg.get("SCALE_STATS", {})
        target_median = scale_stats.get("bbox_median", cfg.get("SCALE_MEDIAN", 0.001488))
        sigma = cfg.get("SCALE_SIGMA", 0.8)
        
        mu = np.log(target_median)
        s = np.random.lognormal(mu, sigma)
        return float(np.clip(s, clip_min, clip_max))
    else:
        return random.uniform(*cfg["SCALE_RANGE"])


def calculate_area_based_size(crop_w: int, crop_h: int, bg_w: int, bg_h: int, 
                             target_scale: float, mask: np.ndarray = None,
                             max_w_ratio: float = 0.15, 
                             max_h_ratio: float = 0.15) -> Tuple[int, int]:
    """
    面積ベースのリサイズサイズ計算（マスク領域を考慮）
    """
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
    
    new_w = int(ideal_w * adjust_scale)
    new_h = int(ideal_h * adjust_scale)
    
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
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
            
        mean_match = re.search(r"Mean:\s+([\d.]+)", content)
        std_match = re.search(r"Standard deviation:\s+([\d.]+)", content)
        
        if mean_match and std_match:
            mean = float(mean_match.group(1))
            std = float(std_match.group(1))
            
            max_n = 30
            arr = np.zeros(max_n + 1, dtype=np.float32)
            
            for n in range(1, max_n + 1):
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


# =============================================================================
# データ拡張関数（マンガ特有のaugmentation）
# =============================================================================

def augment_onomatopoeia_and_mask(onomatopoeia: np.ndarray, mask: np.ndarray, cfg: dict):
    """
    マンガ特有のオノマトペ用データ拡張を適用
    - 回転（±30度）
    - 水平反転
    - アスペクト比変更（0.9-1.1）
    - せん断変換（±15度）- 斜体効果
    - 透明度変化（0.7-1.0）- 薄いオノマトペ対応
    - ガウシアンブラー - 動きのブレ表現
    - ランダム消去（5-15%）- 部分的な欠損
    
    ※スケール変動は配置時にすでにランダムサンプリングされるため不要
    """
    defaults = dict(
        HFLIP_PROB=0.5,
        ROT_PROB=0.7,
        ROT_RANGE=(-30, 30),
        ASPECT_RATIO_PROB=0.3,
        ASPECT_RATIO_RANGE=(0.9, 1.1),
        SHEAR_PROB=0.4,
        SHEAR_RANGE=(-0.3, 0.3),  # tan(15度) ≈ 0.27
        ALPHA_PROB=0.5,
        ALPHA_RANGE=(0.7, 1.0),
        BLUR_PROB=0.3,
        BLUR_KERNEL_SIZES=[1, 3],
        RANDOM_ERASING_PROB=0.3,
        ERASING_RATIO_RANGE=(0.05, 0.15),
    )
    if cfg is None:
        cfg = {}
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    
    h, w = mask.shape[:2]
    
    # 1. 水平反転
    if random.random() < cfg["HFLIP_PROB"]:
        onomatopoeia = cv2.flip(onomatopoeia, 1)
        mask = cv2.flip(mask, 1)

    # 2. 回転（±30度）
    if random.random() < cfg["ROT_PROB"]:
        angle = random.uniform(*cfg["ROT_RANGE"])
        h, w = mask.shape[:2]
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = abs(M[0,0]), abs(M[0,1])
        new_w = int(h*sin + w*cos)
        new_h = int(h*cos + w*sin)
        M[0,2] += (new_w/2) - center[0]
        M[1,2] += (new_h/2) - center[1]

        onomatopoeia = cv2.warpAffine(onomatopoeia, M, (new_w, new_h),
                                       flags=cv2.INTER_LINEAR,
                                       borderValue=(255,255,255))
        mask = cv2.warpAffine(mask, M, (new_w, new_h),
                               flags=cv2.INTER_NEAREST,
                               borderValue=0)

    # 3. アスペクト比変更（0.9-1.1）
    if random.random() < cfg["ASPECT_RATIO_PROB"]:
        aspect_scale = random.uniform(*cfg["ASPECT_RATIO_RANGE"])
        h, w = mask.shape[:2]
        new_w = int(w * aspect_scale)
        if new_w > 10:
            onomatopoeia = cv2.resize(onomatopoeia, (new_w, h))
            mask = cv2.resize(mask, (new_w, h))

    # 4. せん断変換（±15度）- 斜体効果
    if random.random() < cfg["SHEAR_PROB"]:
        h, w = mask.shape[:2]
        shear = random.uniform(*cfg["SHEAR_RANGE"])
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        onomatopoeia = cv2.warpAffine(onomatopoeia, M, (w, h), 
                                       flags=cv2.INTER_LINEAR, 
                                       borderMode=cv2.BORDER_CONSTANT, 
                                       borderValue=(255, 255, 255))
        mask = cv2.warpAffine(mask, M, (w, h), 
                               flags=cv2.INTER_NEAREST, 
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=0)

    # 5. 透明度変化（0.7-1.0）- 薄いオノマトペ対応
    if random.random() < cfg["ALPHA_PROB"]:
        alpha = random.uniform(*cfg["ALPHA_RANGE"])
        white_bg = np.ones_like(onomatopoeia) * 255
        onomatopoeia = (onomatopoeia.astype(np.float32) * alpha + 
                        white_bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    # 6. ガウシアンブラー（動きのブレ表現）
    if random.random() < cfg["BLUR_PROB"]:
        kernel_size = random.choice(cfg["BLUR_KERNEL_SIZES"])
        if kernel_size > 1:
            onomatopoeia = cv2.GaussianBlur(onomatopoeia, (kernel_size, kernel_size), 0)

    # 7. ランダム消去（5-15%）- 部分的な欠損
    if random.random() < cfg["RANDOM_ERASING_PROB"]:
        h, w = mask.shape[:2]
        erase_ratio = random.uniform(*cfg["ERASING_RATIO_RANGE"])
        erase_h = int(h * np.sqrt(erase_ratio))
        erase_w = int(w * np.sqrt(erase_ratio))
        
        if erase_h > 0 and erase_w > 0:
            top = random.randint(0, max(0, h - erase_h))
            left = random.randint(0, max(0, w - erase_w))
            onomatopoeia[top:top+erase_h, left:left+erase_w] = 255  # 白で消去
            mask[top:top+erase_h, left:left+erase_w] = 0  # マスクも消去

    return onomatopoeia, mask


# =============================================================================
# コマ角配置関連
# =============================================================================

@dataclass
class Point:
    """座標を表すクラス"""
    x: int
    y: int
    
    def __iter__(self):
        return iter((self.x, self.y))


@dataclass
class PanelQuad:
    """パネルの四隅の座標"""
    lt: Point
    rt: Point
    lb: Point
    rb: Point


@dataclass
class CornerPosition:
    """コマの角の位置情報"""
    position: Point
    corner_type: str
    panel_quad: PanelQuad
    panel_bbox: Tuple[int, int, int, int]
    panel_mask: np.ndarray


def detect_panels_simple(image: np.ndarray, 
                        area_ratio_threshold: float = 0.85,
                        min_area: int = 10000) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]]:
    """シンプルな二値化・輪郭抽出によるコマ検出"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    panels = []
    
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        area_ratio = contour_area / bbox_area if bbox_area > 0 else 0
        
        if area_ratio >= area_ratio_threshold:
            panel_mask = np.zeros((h, w), dtype=np.uint8)
            contour_relative = contour - np.array([x, y])
            cv2.drawContours(panel_mask, [contour_relative], -1, 255, -1)
            panels.append((panel_mask, (x, y, w, h), contour))
    
    return panels


def extract_panel_corners(background_path: str, 
                         area_ratio_threshold: float = 0.85,
                         min_area: int = 10000) -> List[CornerPosition]:
    """背景画像からコマを抽出し、角の位置を取得"""
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")
    
    panels = detect_panels_simple(background, area_ratio_threshold, min_area)
    
    corners = []
    for panel_mask, bbox, contour in panels:
        x, y, w, h = bbox
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= 4:
            points = approx.reshape(-1, 2)
            points_sorted_y = points[points[:, 1].argsort()]
            top_points = points_sorted_y[:len(points_sorted_y)//2]
            bottom_points = points_sorted_y[len(points_sorted_y)//2:]
            
            top_points = top_points[top_points[:, 0].argsort()]
            lt = Point(int(top_points[0][0]), int(top_points[0][1]))
            rt = Point(int(top_points[-1][0]), int(top_points[-1][1]))
            
            bottom_points = bottom_points[bottom_points[:, 0].argsort()]
            lb = Point(int(bottom_points[0][0]), int(bottom_points[0][1]))
            rb = Point(int(bottom_points[-1][0]), int(bottom_points[-1][1]))
        else:
            lt = Point(x, y)
            rt = Point(x + w, y)
            lb = Point(x, y + h)
            rb = Point(x + w, y + h)
        
        quad = PanelQuad(lt=lt, rt=rt, lb=lb, rb=rb)
        
        for corner_type in ['lt', 'rt', 'lb', 'rb']:
            position = getattr(quad, corner_type)
            corners.append(CornerPosition(
                position=position,
                corner_type=corner_type,
                panel_quad=quad,
                panel_bbox=bbox,
                panel_mask=panel_mask
            ))
    
    return corners


def place_onomatopoeia_at_corner(background: np.ndarray, onomatopoeia: np.ndarray, mask: np.ndarray,
                                corner_pos: CornerPosition, scale: float, 
                                corner_ratio: float, cfg: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    """コマの角にオノマトペを配置"""
    bg_h, bg_w = background.shape[:2]
    
    cropped_ono, cropped_mask, bbox = crop_onomatopoeia_and_mask(onomatopoeia, mask)
    
    if cropped_ono.size == 0 or cropped_mask.size == 0:
        return background, np.zeros((bg_h, bg_w), dtype=np.uint8), {}
    
    crop_h, crop_w = cropped_ono.shape[:2]
    
    new_w, new_h = calculate_area_based_size(
        crop_w, crop_h, bg_w, bg_h, scale,
        mask=cropped_mask,
        max_w_ratio=cfg.get("MAX_WIDTH_RATIO", 0.15),
        max_h_ratio=cfg.get("MAX_HEIGHT_RATIO", 0.15)
    )
    
    ono_resized = cv2.resize(cropped_ono, (new_w, new_h))
    mask_resized = cv2.resize(cropped_mask, (new_w, new_h))
    
    mask_coords = cv2.findNonZero(mask_resized)
    if mask_coords is not None:
        mask_x, mask_y, mask_w, mask_h = cv2.boundingRect(mask_coords)
    else:
        mask_x, mask_y, mask_w, mask_h = 0, 0, new_w, new_h
    
    corner_x, corner_y = corner_pos.position
    
    # オノマトペ用のはみ出し設定（より控えめ）
    overhang_ratio = cfg.get("OVERHANG_RATIO", 0.1)
    overhang_w = int(mask_w * overhang_ratio)
    overhang_h = int(mask_h * overhang_ratio)
    border_width = cfg.get("BORDER_WIDTH", 5)
    
    if corner_pos.corner_type == 'lt':
        x, y = corner_x - mask_x - overhang_w + border_width, corner_y - mask_y - overhang_h + border_width
    elif corner_pos.corner_type == 'rt':
        x, y = corner_x - (mask_x + mask_w) + overhang_w - border_width, corner_y - mask_y - overhang_h + border_width
    elif corner_pos.corner_type == 'lb':
        x, y = corner_x - mask_x - overhang_w + border_width, corner_y - (mask_y + mask_h) + overhang_h - border_width
    elif corner_pos.corner_type == 'rb':
        x, y = corner_x - (mask_x + mask_w) + overhang_w - border_width, corner_y - (mask_y + mask_h) + overhang_h - border_width
    else:
        x, y = corner_x, corner_y
    
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    # 画像範囲内にクリップ
    start_x = max(0, x)
    start_y = max(0, y)
    end_x = min(x + new_w, bg_w)
    end_y = min(y + new_h, bg_h)
    
    clip_w = end_x - start_x
    clip_h = end_y - start_y
    
    if clip_w <= 0 or clip_h <= 0:
        return background, result_mask, {}
    
    ono_start_x = start_x - x
    ono_start_y = start_y - y
    
    ono_region = ono_resized[ono_start_y:ono_start_y + clip_h, ono_start_x:ono_start_x + clip_w]
    mask_region = mask_resized[ono_start_y:ono_start_y + clip_h, ono_start_x:ono_start_x + clip_w]
    
    mask_norm = mask_region.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
    
    bg_region = result_img[start_y:end_y, start_x:end_x]
    
    if ono_region.shape == bg_region.shape and mask_3ch.shape == bg_region.shape:
        blended = ono_region.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
        result_img[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)
        result_mask[start_y:end_y, start_x:end_x] = mask_region
    else:
        return background, result_mask, {}
    
    detail_info = {
        "corner_type": corner_pos.corner_type,
        "corner_position": f"({corner_x},{corner_y})",
        "position": f"({x},{y})",
        "size": f"{new_w}x{new_h}",
        "scale": f"{scale:.6f}",
    }
    
    return result_img, result_mask, detail_info


# =============================================================================
# 合成関数
# =============================================================================

def composite_random_onomatopoeia(background_path: str, onomatopoeia_mask_pairs: list,
                                 cfg: dict = None, use_augmentation: bool = False) -> tuple:
    """ランダム配置によるオノマトペ合成"""
    if cfg is None:
        cfg = {}
        
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")
    
    bg_h, bg_w = background.shape[:2]
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    max_onomatopoeia = min(cfg.get("NUM_ONOMATOPOEIA_RANGE", (1, 10))[1], len(onomatopoeia_mask_pairs))
    num_onomatopoeia = sample_num_onomatopoeia(cfg, max_onomatopoeia)
    
    selected_pairs = random.sample(onomatopoeia_mask_pairs, num_onomatopoeia)
    
    occupied_regions = []
    successfully_placed = []
    onomatopoeia_details = []
    
    for ono_path, mask_path in selected_pairs:
        onomatopoeia = cv2.imread(ono_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if onomatopoeia is None or mask is None:
            continue
        
        # データ拡張を適用
        if use_augmentation:
            onomatopoeia, mask = augment_onomatopoeia_and_mask(onomatopoeia, mask, cfg)
        
        cropped_ono, cropped_mask, bbox = crop_onomatopoeia_and_mask(onomatopoeia, mask)
        
        if cropped_ono.size == 0 or cropped_mask.size == 0:
            continue

        crop_h, crop_w = cropped_ono.shape[:2]
        
        try:
            ono_scale = sample_scale(bg_w, crop_w, cfg)
        except KeyError:
            ono_scale = random.uniform(cfg.get("SCALE_RANGE", (0.0001, 0.005))[0], 
                                       cfg.get("SCALE_RANGE", (0.0001, 0.005))[1])
        
        new_w, new_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, ono_scale,
            mask=cropped_mask,
            max_w_ratio=cfg.get("MAX_WIDTH_RATIO", 0.15),
            max_h_ratio=cfg.get("MAX_HEIGHT_RATIO", 0.15)
        )
        
        if new_w >= bg_w or new_h >= bg_h:
            continue
        
        ono_resized = cv2.resize(cropped_ono, (new_w, new_h))
        mask_resized = cv2.resize(cropped_mask, (new_w, new_h))
        
        placed = False
        best_position = None
        min_overlap_ratio = float('inf')
        
        max_attempts = cfg.get("MAX_ATTEMPTS", 200)
        for attempt in range(max_attempts // 2):
            max_x = max(0, bg_w - new_w)
            max_y = max(0, bg_h - new_h)
            
            if max_x <= 0 or max_y <= 0:
                break
                
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            new_region = (x, y, x + new_w, y + new_h)
            
            max_overlap_ratio = 0
            for occupied in occupied_regions:
                if regions_overlap(new_region, occupied):
                    overlap_area = calculate_overlap_area(new_region, occupied)
                    new_area = new_w * new_h
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
            max_x = max(0, bg_w - new_w)
            max_y = max(0, bg_h - new_h)
            if max_x >= 0 and max_y >= 0:
                x = random.randint(0, max_x) if max_x > 0 else 0
                y = random.randint(0, max_y) if max_y > 0 else 0
                placed = True
        
        if placed:
            mask_norm = mask_resized.astype(np.float32) / 255.0
            mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
            
            bg_region = result_img[y:y+new_h, x:x+new_w]
            
            blended = ono_resized.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
            result_img[y:y+new_h, x:x+new_w] = blended.astype(np.uint8)
            
            result_mask[y:y+new_h, x:x+new_w] = np.maximum(
                result_mask[y:y+new_h, x:x+new_w], mask_resized)
            
            new_region = (x, y, x + new_w, y + new_h)
            occupied_regions.append(new_region)
            successfully_placed.append(Path(ono_path).stem)
            
            ono_info = {
                "onomatopoeia_file": Path(ono_path).name,
                "placement_type": "random",
                "original_size": f"{onomatopoeia.shape[1]}x{onomatopoeia.shape[0]}",
                "cropped_size": f"{crop_w}x{crop_h}",
                "final_size": f"{new_w}x{new_h}",
                "position": f"({x},{y})",
                "scale": f"{ono_scale:.6f}",
            }
            onomatopoeia_details.append(ono_info)
    
    return result_img, result_mask, successfully_placed, onomatopoeia_details


def composite_corner_aligned_onomatopoeia(background_path: str, onomatopoeia_mask_pairs: list,
                                         corner_ratio: float = 0.3, cfg: dict = None,
                                         use_augmentation: bool = False) -> tuple:
    """コマの角配置とランダム配置を組み合わせたオノマトペ合成"""
    if cfg is None:
        cfg = {}
    
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")
    
    bg_h, bg_w = background.shape[:2]
    
    try:
        corners = extract_panel_corners(background_path)
    except Exception as e:
        print(f"コマ抽出エラー: {e}")
        corners = []
    
    total_available = len(onomatopoeia_mask_pairs)
    num_total = sample_num_onomatopoeia(cfg, total_available)
    num_total = min(num_total, total_available)
    
    if num_total == 0:
        return background, np.zeros((bg_h, bg_w), dtype=np.uint8), [], []
    
    corner_placement_ratio = cfg.get("CORNER_PLACEMENT_RATIO", 0.3)
    num_corner = int(num_total * corner_placement_ratio)
    num_random = num_total - num_corner
    
    if corners:
        num_corner = min(num_corner, len(corners))
        num_random = num_total - num_corner
    else:
        num_corner = 0
        num_random = num_total
    
    shuffled_pairs = onomatopoeia_mask_pairs.copy()
    random.shuffle(shuffled_pairs)
    corner_pairs = shuffled_pairs[:num_corner]
    random_pairs = shuffled_pairs[num_corner:num_corner + num_random]
    
    selected_corners = random.sample(corners, num_corner) if num_corner > 0 else []
    
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    successfully_placed = []
    onomatopoeia_details = []
    occupied_regions = []
    
    def check_overlap(x, y, w, h, margin=10):
        for ox, oy, ow, oh in occupied_regions:
            if not (x + w + margin < ox or x > ox + ow + margin or
                    y + h + margin < oy or y > oy + oh + margin):
                return True
        return False
    
    # 1. コマの角に配置
    for corner_pos, (ono_path, mask_path) in zip(selected_corners, corner_pairs):
        onomatopoeia = cv2.imread(ono_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if onomatopoeia is None or mask is None:
            continue
        
        if use_augmentation:
            onomatopoeia, mask = augment_onomatopoeia_and_mask(onomatopoeia, mask, cfg)
        
        try:
            ono_scale = sample_scale(bg_w, onomatopoeia.shape[1], cfg)
        except KeyError:
            ono_scale = random.uniform(cfg.get("SCALE_RANGE", (0.0001, 0.005))[0], 
                                       cfg.get("SCALE_RANGE", (0.0001, 0.005))[1])
        
        try:
            result_img, current_mask, detail_info = place_onomatopoeia_at_corner(
                result_img, onomatopoeia, mask, corner_pos, ono_scale, corner_ratio, cfg
            )
            
            if not detail_info:
                continue
            
            result_mask = np.maximum(result_mask, current_mask)
            
            successfully_placed.append(Path(ono_path).stem)
            
            ono_info = {
                "onomatopoeia_file": Path(ono_path).name,
                "placement_type": "corner",
                **detail_info
            }
            onomatopoeia_details.append(ono_info)
            
            pos_str = detail_info.get("position", "(0,0)")
            size_str = detail_info.get("size", "0x0")
            bx, by = map(int, pos_str.strip("()").split(","))
            bw, bh = map(int, size_str.split("x"))
            occupied_regions.append((bx, by, bw, bh))
            
        except Exception as e:
            if cfg.get("DEBUG", False):
                print(f"角配置エラー: {e}")
            continue
    
    # 2. ランダム位置に配置
    for ono_path, mask_path in random_pairs:
        onomatopoeia = cv2.imread(ono_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if onomatopoeia is None or mask is None:
            continue
        
        if use_augmentation:
            onomatopoeia, mask = augment_onomatopoeia_and_mask(onomatopoeia, mask, cfg)
        
        cropped_ono, cropped_mask, bbox = crop_onomatopoeia_and_mask(onomatopoeia, mask)
        if cropped_ono.size == 0:
            continue
        
        crop_h, crop_w = cropped_ono.shape[:2]
        
        try:
            ono_scale = sample_scale(bg_w, crop_w, cfg)
        except KeyError:
            ono_scale = random.uniform(cfg.get("SCALE_RANGE", (0.0001, 0.005))[0], 
                                       cfg.get("SCALE_RANGE", (0.0001, 0.005))[1])
        
        new_w, new_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, ono_scale,
            max_w_ratio=cfg.get("MAX_WIDTH_RATIO", 0.15),
            max_h_ratio=cfg.get("MAX_HEIGHT_RATIO", 0.15)
        )
        
        placed = False
        for attempt in range(10):
            max_x = max(0, bg_w - new_w)
            max_y = max(0, bg_h - new_h)
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0
            
            if not check_overlap(x, y, new_w, new_h):
                placed = True
                break
        
        if not placed:
            continue
        
        try:
            ono_resized = cv2.resize(cropped_ono, (new_w, new_h))
            mask_resized = cv2.resize(cropped_mask, (new_w, new_h))
            
            mask_norm = mask_resized.astype(np.float32) / 255.0
            mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
            
            bg_region = result_img[y:y+new_h, x:x+new_w]
            blended = ono_resized.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
            result_img[y:y+new_h, x:x+new_w] = blended.astype(np.uint8)
            
            result_mask[y:y+new_h, x:x+new_w] = np.maximum(
                result_mask[y:y+new_h, x:x+new_w], mask_resized
            )
            
            successfully_placed.append(Path(ono_path).stem)
            
            ono_info = {
                "onomatopoeia_file": Path(ono_path).name,
                "placement_type": "random",
                "position": f"({x},{y})",
                "size": f"{new_w}x{new_h}",
                "scale": f"{ono_scale:.6f}"
            }
            onomatopoeia_details.append(ono_info)
            
            occupied_regions.append((x, y, new_w, new_h))
            
        except Exception as e:
            if cfg.get("DEBUG", False):
                print(f"ランダム配置エラー: {e}")
            continue
    
    return result_img, result_mask, successfully_placed, onomatopoeia_details


def composite_panel_onomatopoeia(background_path: str, onomatopoeia_mask_pairs: list,
                                cfg: dict = None, use_augmentation: bool = False) -> tuple:
    """
    パネル内配置によるオノマトペ合成（デフォルトモード）
    検出したコマ内のランダムな位置にオノマトペを配置する
    """
    if cfg is None:
        cfg = {}
    
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")
    
    bg_h, bg_w = background.shape[:2]
    
    # パネル検出
    panels = detect_panels_simple(background)
    
    if not panels:
        # パネルが検出できない場合は完全ランダム配置にフォールバック
        return composite_random_onomatopoeia(background_path, onomatopoeia_mask_pairs, 
                                             cfg=cfg, use_augmentation=use_augmentation)
    
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    max_onomatopoeia = min(cfg.get("NUM_ONOMATOPOEIA_RANGE", (1, 10))[1], len(onomatopoeia_mask_pairs))
    num_onomatopoeia = sample_num_onomatopoeia(cfg, max_onomatopoeia)
    
    selected_pairs = random.sample(onomatopoeia_mask_pairs, min(num_onomatopoeia, len(onomatopoeia_mask_pairs)))
    
    occupied_regions = []
    successfully_placed = []
    onomatopoeia_details = []
    
    for ono_path, mask_path in selected_pairs:
        onomatopoeia = cv2.imread(ono_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if onomatopoeia is None or mask is None:
            continue
        
        # データ拡張を適用
        if use_augmentation:
            onomatopoeia, mask = augment_onomatopoeia_and_mask(onomatopoeia, mask, cfg)
        
        cropped_ono, cropped_mask, bbox = crop_onomatopoeia_and_mask(onomatopoeia, mask)
        
        if cropped_ono.size == 0 or cropped_mask.size == 0:
            continue

        crop_h, crop_w = cropped_ono.shape[:2]
        
        # スケール計算（背景画像全体のサイズを使用）
        try:
            ono_scale = sample_scale(bg_w, crop_w, cfg)
        except KeyError:
            ono_scale = random.uniform(cfg.get("SCALE_RANGE", (0.0001, 0.005))[0], 
                                       cfg.get("SCALE_RANGE", (0.0001, 0.005))[1])
        
        new_w, new_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, ono_scale,
            mask=cropped_mask,
            max_w_ratio=cfg.get("MAX_WIDTH_RATIO", 0.25),
            max_h_ratio=cfg.get("MAX_HEIGHT_RATIO", 0.25)
        )
        
        if new_w >= bg_w or new_h >= bg_h:
            continue
        
        # ランダムにパネルを選択してその中に配置を試みる
        placed = False
        random.shuffle(panels)
        
        for panel_mask, panel_bbox, _ in panels:
            panel_x, panel_y, panel_w, panel_h = panel_bbox
            
            # コマより大きくなったらスキップ
            if new_w >= panel_w or new_h >= panel_h:
                continue
            
            # コマ内のランダム位置に配置を試みる
            max_attempts = cfg.get("MAX_PANEL_ATTEMPTS", 10)
            for attempt in range(max_attempts):
                max_x = max(0, panel_w - new_w)
                max_y = max(0, panel_h - new_h)
                
                if max_x <= 0 or max_y <= 0:
                    break
                
                x_in_panel = random.randint(0, max_x)
                y_in_panel = random.randint(0, max_y)
                
                # グローバル座標に変換
                x_global = panel_x + x_in_panel
                y_global = panel_y + y_in_panel
                
                # 重複チェック
                new_region = (x_global, y_global, x_global + new_w, y_global + new_h)
                overlap = False
                for occupied in occupied_regions:
                    if regions_overlap(new_region, occupied):
                        overlap_area = calculate_overlap_area(new_region, occupied)
                        if overlap_area / (new_w * new_h) > 0.15:
                            overlap = True
                            break
                
                if not overlap:
                    placed = True
                    break
            
            if placed:
                break
        
        if not placed:
            # パネル内に配置できなかった場合は全体でランダム配置を試みる
            max_x = max(0, bg_w - new_w)
            max_y = max(0, bg_h - new_h)
            if max_x > 0 and max_y > 0:
                x_global = random.randint(0, max_x)
                y_global = random.randint(0, max_y)
                placed = True
        
        if placed:
            ono_resized = cv2.resize(cropped_ono, (new_w, new_h))
            mask_resized = cv2.resize(cropped_mask, (new_w, new_h))
            
            mask_norm = mask_resized.astype(np.float32) / 255.0
            mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
            
            # 画像範囲内にクリップ
            end_x = min(x_global + new_w, bg_w)
            end_y = min(y_global + new_h, bg_h)
            actual_w = end_x - x_global
            actual_h = end_y - y_global
            
            if actual_w > 0 and actual_h > 0:
                bg_region = result_img[y_global:end_y, x_global:end_x]
                ono_region = ono_resized[:actual_h, :actual_w]
                mask_3ch_region = mask_3ch[:actual_h, :actual_w]
                mask_region = mask_resized[:actual_h, :actual_w]
                
                if bg_region.shape == ono_region.shape:
                    blended = ono_region.astype(np.float32) * mask_3ch_region + \
                              bg_region.astype(np.float32) * (1 - mask_3ch_region)
                    result_img[y_global:end_y, x_global:end_x] = blended.astype(np.uint8)
                    
                    result_mask[y_global:end_y, x_global:end_x] = np.maximum(
                        result_mask[y_global:end_y, x_global:end_x], mask_region)
                    
                    new_region = (x_global, y_global, x_global + new_w, y_global + new_h)
                    occupied_regions.append(new_region)
                    successfully_placed.append(Path(ono_path).stem)
                    
                    ono_info = {
                        "onomatopoeia_file": Path(ono_path).name,
                        "placement_type": "panel",
                        "original_size": f"{onomatopoeia.shape[1]}x{onomatopoeia.shape[0]}",
                        "cropped_size": f"{crop_w}x{crop_h}",
                        "final_size": f"{new_w}x{new_h}",
                        "position": f"({x_global},{y_global})",
                        "scale": f"{ono_scale:.6f}",
                    }
                    onomatopoeia_details.append(ono_info)
    
    return result_img, result_mask, successfully_placed, onomatopoeia_details


# =============================================================================
# データセット生成
# =============================================================================

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
                          target_count: int, cfg: dict, 
                          use_panel_placement: bool = True,
                          use_augmentation: bool = False) -> int:
    """指定されたsplit（trainまたはval）のデータセットを生成する"""
    print(f"\n=== {split_name} データセット生成開始 ===")
    print(f"目標画像数: {target_count}")
    print(f"背景画像数: {len(background_files)}")
    print(f"利用可能オノマトペ数: {len(onomatopoeia_pairs)}")
    print(f"パネル内配置: {'有効' if use_panel_placement else '無効（完全ランダム）'}")
    print(f"データ拡張: {'有効' if use_augmentation else '無効'}")
    
    log_file_path = os.path.join(output_dir, "..", f"{split_name}_composition_log.txt")
    
    current_number = 1
    success_count = 0
    bg_idx = 0
    
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== {split_name.upper()} オノマトペデータセット合成ログ ===\n")
        log_file.write(f"生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"目標画像数: {target_count}\n")
        log_file.write(f"パネル内配置: {'有効' if use_panel_placement else '無効'}\n")
        log_file.write(f"データ拡張: {'有効' if use_augmentation else '無効'}\n")
        log_file.write("=" * 80 + "\n\n")
    
    while success_count < target_count:
        bg_path = background_files[bg_idx % len(background_files)]
        bg_name = Path(bg_path).stem
        
        try:
            if use_panel_placement:
                # パネル内配置（デフォルト）
                result_img, result_mask, placed_items, details = composite_panel_onomatopoeia(
                    bg_path, onomatopoeia_pairs, cfg=cfg,
                    use_augmentation=use_augmentation
                )
            else:
                # 完全ランダム配置
                result_img, result_mask, placed_items, details = composite_random_onomatopoeia(
                    bg_path, onomatopoeia_pairs, cfg=cfg, use_augmentation=use_augmentation
                )
            
            output_img_path = os.path.join(output_dir, f"{current_number:03d}.png")
            output_mask_path = os.path.join(mask_output_dir, f"{current_number:03d}_mask.png")
            
            cv2.imwrite(output_img_path, result_img)
            cv2.imwrite(output_mask_path, result_mask)
            
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"画像 {current_number:03d}.png:\n")
                log_file.write(f"  背景ファイル: {Path(bg_path).name}\n")
                log_file.write(f"  配置したオノマトペ数: {len(details)}\n")
                
                for i, detail in enumerate(details, 1):
                    log_file.write(f"    オノマトペ{i}: {detail.get('onomatopoeia_file', 'N/A')}\n")
                    log_file.write(f"      配置タイプ: {detail.get('placement_type', 'unknown')}\n")
                    log_file.write(f"      位置: {detail.get('position', 'N/A')}\n")
                    log_file.write(f"      サイズ: {detail.get('final_size', detail.get('size', 'N/A'))}\n")
                
                log_file.write("\n")
            
            success_count += 1
            current_number += 1
            
            if success_count % 10 == 0:
                print(f"  進捗: {success_count}/{target_count} 完了")
            
        except Exception as e:
            print(f"✗ 合成失敗 (背景:{bg_name}): {e}")
        
        bg_idx += 1
    
    print(f"✅ {split_name} 完了: {success_count}個の画像を生成")
    return success_count


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="オノマトペ合成データセット作成（統合版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本（デフォルトでパネル内配置を使用）
  python create_syn_onomatopoeia_dataset.py --output-dir onomatopoeia_dataset/syn500 --target-images 500

  # パネル配置を無効化（完全ランダム）
  python create_syn_onomatopoeia_dataset.py --output-dir onomatopoeia_dataset/syn500-random --target-images 500 --no-panel-placement

  # データ拡張を有効化（マンガ特有のaugmentation）
  python create_syn_onomatopoeia_dataset.py --output-dir onomatopoeia_dataset/syn500-aug --target-images 500 --augmentation
        """
    )
    
    # 入力ディレクトリ
    parser.add_argument("--onomatopoeia-dir", default="onomatopoeias/images", help="オノマトペ画像ディレクトリ")
    parser.add_argument("--mask-dir", default="onomatopoeias/masks", help="マスク画像ディレクトリ")
    parser.add_argument("--background-dir", default="generated_double_backs_1536x1024", help="背景画像ディレクトリ")
    
    # 出力
    parser.add_argument("--output-dir", required=True, help="出力ディレクトリ")
    parser.add_argument("--target-images", type=int, required=True, help="生成する画像総数")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="train用の比率")
    
    # 機能オプション
    parser.add_argument("--no-panel-placement", action="store_true", help="パネル配置を無効化（完全ランダム配置）")
    parser.add_argument("--augmentation", action="store_true", help="データ拡張を有効化（マンガ特有のaugmentation）")
    
    # スケール設定
    parser.add_argument("--scale-range", nargs=2, type=float, default=[0.0001, 0.005], help="スケール範囲")
    parser.add_argument("--num-range", nargs=2, type=int, default=[1, 15], help="配置数範囲")
    
    # 統計ファイル
    parser.add_argument("--stats-file", default="onomatopoeia_statistics.txt", help="統計ファイル")
    
    args = parser.parse_args()
    
    # 設定
    CFG = {
        "SCALE_RANGE": tuple(args.scale_range),
        "NUM_ONOMATOPOEIA_RANGE": tuple(args.num_range),
        "MAX_ATTEMPTS": 200,
        "MAX_PANEL_ATTEMPTS": 10,
        "TRAIN_RATIO": args.train_ratio,
        "ONOMATOPOEIA_SPLIT_SEED": 42,
        
        # 統計情報ベースのサンプリング
        "SCALE_MODE": "lognormal",
        "SCALE_MEDIAN": 0.001488,
        "SCALE_SIGMA": 0.8,
        "SCALE_CLIP": (0.0005, 0.02),
        "COUNT_PROBS": None,
        "SCALE_STATS": None,
        
        # 面積ベースリサイズ
        "MAX_WIDTH_RATIO": 0.25,
        "MAX_HEIGHT_RATIO": 0.25,
    }
    
    # 統計ファイル読み込み
    if args.stats_file and os.path.exists(args.stats_file):
        print(f"統計ファイルを読み込み: {args.stats_file}")
        try:
            CFG["SCALE_STATS"] = load_scale_stats(args.stats_file)
            CFG["COUNT_PROBS"] = load_count_probs(args.stats_file)
            print("✓ 統計ベースのサンプリングを有効化")
        except Exception as e:
            print(f"統計ファイル読み込みエラー: {e}")
    
    # ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 設定をファイル出力
    use_panel_placement = not args.no_panel_placement
    config_output = {
        "timestamp": datetime.now().isoformat(),
        "script_name": "create_syn_onomatopoeia_dataset.py",
        "output_path": args.output_dir,
        "target_images": args.target_images,
        "panel_placement": use_panel_placement,
        "augmentation": args.augmentation,
        "config": {k: v for k, v in CFG.items() if k not in ["COUNT_PROBS", "SCALE_STATS"]},
    }
    
    config_file_path = os.path.join(args.output_dir, "config.json")
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"設定情報を保存: {config_file_path}")
    
    print("\n=== オノマトペ合成データセット 作成開始 ===")
    print(f"パネル内配置: {'有効' if use_panel_placement else '無効（完全ランダム）'}")
    print(f"データ拡張: {'有効' if args.augmentation else '無効'}")
    
    # オノマトペとマスクの対応を取得
    onomatopoeia_mask_pairs = []
    print("オノマトペ・マスクペアを検索中...")
    for ono_file in os.listdir(args.onomatopoeia_dir):
        if ono_file.endswith(('.png', '.jpg', '.jpeg')):
            ono_path = os.path.join(args.onomatopoeia_dir, ono_file)
            ono_stem = Path(ono_file).stem
            
            mask_file = f"{ono_stem}_mask.png"
            mask_path = os.path.join(args.mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                onomatopoeia_mask_pairs.append((ono_path, mask_path))
    
    # 背景画像を取得
    background_files = []
    for bg_file in os.listdir(args.background_dir):
        if bg_file.endswith(('.png', '.jpg', '.jpeg')):
            background_files.append(os.path.join(args.background_dir, bg_file))
    
    print(f"見つかったオノマトペ: {len(onomatopoeia_mask_pairs)}個")
    print(f"見つかった背景: {len(background_files)}個")
    
    # オノマトペを分割
    print(f"\nオノマトペを分割中（train:{CFG['TRAIN_RATIO']:.0%}, val:{1-CFG['TRAIN_RATIO']:.0%}）...")
    train_onomatopoeia, val_onomatopoeia = split_onomatopoeia(
        onomatopoeia_mask_pairs, CFG["TRAIN_RATIO"], CFG["ONOMATOPOEIA_SPLIT_SEED"]
    )
    
    print(f"train用オノマトペ: {len(train_onomatopoeia)}個")
    print(f"val用オノマトペ: {len(val_onomatopoeia)}個")
    
    # 目標画像数を計算
    train_target = int(args.target_images * CFG["TRAIN_RATIO"])
    val_target = args.target_images - train_target
    
    print(f"\n目標画像数:")
    print(f"train: {train_target}枚")
    print(f"val: {val_target}枚")
    
    # train データセット生成
    train_img_dir = os.path.join(args.output_dir, "train", "images")
    train_mask_dir = os.path.join(args.output_dir, "train", "masks")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    
    train_count = generate_dataset_split(
        background_files, train_onomatopoeia,
        train_img_dir, train_mask_dir, "train", train_target, CFG,
        use_panel_placement=use_panel_placement,
        use_augmentation=args.augmentation
    )
    
    # val データセット生成
    val_img_dir = os.path.join(args.output_dir, "val", "images")
    val_mask_dir = os.path.join(args.output_dir, "val", "masks")
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    
    val_count = generate_dataset_split(
        background_files, val_onomatopoeia,
        val_img_dir, val_mask_dir, "val", val_target, CFG,
        use_panel_placement=use_panel_placement,
        use_augmentation=args.augmentation
    )
    
    # 最終レポート
    print(f"\n=== オノマトペ合成データセット 作成完了 ===")
    print(f"出力先: {args.output_dir}")
    print(f"総生成画像数: {train_count + val_count}枚")
    
    for split in ["train", "val"]:
        img_count = len(list(Path(args.output_dir).glob(f"{split}/images/*.png")))
        mask_count = len(list(Path(args.output_dir).glob(f"{split}/masks/*.png")))
        print(f"{split}: {img_count} 画像, {mask_count} マスク")
    
    # 統計情報を更新
    config_output["statistics"] = {
        "total_images": train_count + val_count,
        "train_images": train_count,
        "val_images": val_count,
        "train_onomatopoeia_used": len(train_onomatopoeia),
        "val_onomatopoeia_used": len(val_onomatopoeia),
    }
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    
    print(f"\n設定・統計情報: {config_file_path}")


if __name__ == "__main__":
    main()
