"""
吹き出し合成データセット作成スクリプト（統合版）

機能:
- マスクベースのクロップ（デフォルト）
- 面積ベースのスケール計算
- コマ角配置 + ランダム配置（デフォルト）
- データ拡張（--augmentation オプション）
- 統計情報からのサンプリング

使用例:
  # 基本的な使用方法（デフォルトでパネル検出＋角配置を使用）
  python create_syn_balloon_dataset.py --output-dir balloon_dataset/syn500 --target-images 500

  # パネル配置を無効化（完全ランダム配置）
  python create_syn_balloon_dataset.py --output-dir balloon_dataset/syn500-random --target-images 500 --no-panel-placement

  # データ拡張を有効化
  python create_syn_balloon_dataset.py --output-dir balloon_dataset/syn500-aug --target-images 500 --augmentation
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


def crop_balloon_and_mask(balloon: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
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
        
        target_median = cfg.get("SCALE_MEDIAN", 0.007226)
        sigma = cfg.get("SCALE_SIGMA", 0.85)
        
        mu = np.log(target_median)
        s = np.random.lognormal(mu, sigma)
        return float(np.clip(s, clip_min, clip_max))
    else:
        return random.uniform(*cfg["SCALE_RANGE"])


def calculate_area_based_size(crop_w: int, crop_h: int, bg_w: int, bg_h: int, 
                             target_scale: float, mask: np.ndarray = None,
                             max_w_ratio: float = 0.3, 
                             max_h_ratio: float = 0.4) -> Tuple[int, int]:
    """
    面積ベースのリサイズサイズ計算（マスク領域を考慮）
    
    Args:
        crop_w, crop_h: クロップ後のサイズ
        bg_w, bg_h: 背景画像サイズ
        target_scale: 目標スケール値（面積比）
        mask: マスク画像（マスク領域のみで面積を計算する場合）
        max_w_ratio: 最大幅比率
        max_h_ratio: 最大高さ比率
    
    Returns:
        (new_w, new_h): 調整されたサイズ
    """
    bg_area = bg_w * bg_h
    target_area = bg_area * target_scale
    
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
    new_w = max(new_w, 20)
    new_h = max(new_h, 20)
    
    return new_w, new_h


def sample_num_balloons(cfg: dict, max_available: int) -> int:
    """統計情報に基づいて吹き出し個数をサンプリング"""
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


# =============================================================================
# データ拡張関数
# =============================================================================

def augment_balloon_and_mask(balloon: np.ndarray, mask: np.ndarray, cfg: dict):
    """
    データ拡張を適用
    - ランダム回転
    - 水平反転
    - 端を切り取る
    - 線の太さを細くする
    - コマ角四角形化
    - コマ辺直線化
    """
    defaults = dict(
        HFLIP_PROB=0.5,
        ROT_PROB=1.0,
        ROT_RANGE=(-20, 20),
        CUT_EDGE_PROB=0.4,
        CUT_EDGE_RATIO=(0.05, 0.20),
        THIN_LINE_PROB=0.5,
        THIN_PIXELS=(1, 3),
        PANEL_CORNER_PROB=0.3,
        PANEL_CORNER_RATIO=(0.05, 0.15),
        PANEL_EDGE_PROB=0.25,
        PANEL_EDGE_RATIO=(0.30, 0.70),
    )
    if cfg is None:
        cfg = {}
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    
    # 1. Horizontal flip
    if random.random() < cfg["HFLIP_PROB"]:
        balloon = cv2.flip(balloon, 1)
        mask = cv2.flip(mask, 1)

    # 2. Rotation
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

        balloon = cv2.warpAffine(balloon, M, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderValue=(255,255,255))
        mask = cv2.warpAffine(mask, M, (new_w, new_h),
                                 flags=cv2.INTER_NEAREST,
                                 borderValue=0)

    # 3. Cut edge
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
        else:  # right
            x0, x1 = w-int(w*ratio), w
            y0, y1 = 0, h
        balloon[y0:y1, x0:x1] = 255
        mask[y0:y1, x0:x1] = 0

    # 4. Thin line
    if random.random() < cfg["THIN_LINE_PROB"]:
        k = random.randint(*cfg["THIN_PIXELS"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
        eroded = cv2.erode(mask, kernel, iterations=1)
        diff = (mask > 0) & (eroded == 0)
        balloon[diff] = 255
        mask = eroded
        if mask.sum() == 0:
            mask = cv2.dilate(mask, kernel, iterations=1)

    # 5. Panel corner squaring
    if random.random() < cfg["PANEL_CORNER_PROB"]:
        balloon, mask = apply_panel_corner_squaring(balloon, mask, cfg)

    # 6. Panel edge straightening
    if random.random() < cfg["PANEL_EDGE_PROB"]:
        balloon, mask = apply_panel_edge_straightening(balloon, mask, cfg)

    return balloon, mask


def apply_panel_corner_squaring(balloon: np.ndarray, mask: np.ndarray, cfg: dict):
    """コマ角四角形化処理"""
    h, w = mask.shape
    corner = random.choice(["top-left", "top-right", "bottom-left", "bottom-right"])
    cut_ratio = random.uniform(*cfg["PANEL_CORNER_RATIO"])
    
    if corner == "top-left":
        top_cut = int(h * cut_ratio * random.uniform(0.5, 1.5))
        left_cut = int(w * cut_ratio * random.uniform(0.5, 1.5))
        balloon[:top_cut, :] = 255
        mask[:top_cut, :] = 0
        balloon[:, :left_cut] = 255
        mask[:, :left_cut] = 0
    elif corner == "top-right":
        top_cut = int(h * cut_ratio * random.uniform(0.5, 1.5))
        right_cut = int(w * cut_ratio * random.uniform(0.5, 1.5))
        balloon[:top_cut, :] = 255
        mask[:top_cut, :] = 0
        balloon[:, w-right_cut:] = 255
        mask[:, w-right_cut:] = 0
    elif corner == "bottom-left":
        bottom_cut = int(h * cut_ratio * random.uniform(0.5, 1.5))
        left_cut = int(w * cut_ratio * random.uniform(0.5, 1.5))
        balloon[h-bottom_cut:, :] = 255
        mask[h-bottom_cut:, :] = 0
        balloon[:, :left_cut] = 255
        mask[:, :left_cut] = 0
    else:  # bottom-right
        bottom_cut = int(h * cut_ratio * random.uniform(0.5, 1.5))
        right_cut = int(w * cut_ratio * random.uniform(0.5, 1.5))
        balloon[h-bottom_cut:, :] = 255
        mask[h-bottom_cut:, :] = 0
        balloon[:, w-right_cut:] = 255
        mask[:, w-right_cut:] = 0
    
    return balloon, mask


def apply_panel_edge_straightening(balloon: np.ndarray, mask: np.ndarray, cfg: dict):
    """コマ辺直線化処理"""
    h, w = mask.shape
    edge = random.choice(["left", "right", "top", "bottom"])
    straight_ratio = random.uniform(*cfg["PANEL_EDGE_RATIO"])
    cut_depth_ratio = random.uniform(0.03, 0.08)
    
    if edge == "left":
        cut_width = int(w * cut_depth_ratio)
        start_y = int(h * (1 - straight_ratio) / 2)
        end_y = int(h * (1 + straight_ratio) / 2)
        balloon[start_y:end_y, :cut_width] = 255
        mask[start_y:end_y, :cut_width] = 0
    elif edge == "right":
        cut_width = int(w * cut_depth_ratio)
        start_y = int(h * (1 - straight_ratio) / 2)
        end_y = int(h * (1 + straight_ratio) / 2)
        balloon[start_y:end_y, w-cut_width:] = 255
        mask[start_y:end_y, w-cut_width:] = 0
    elif edge == "top":
        cut_height = int(h * cut_depth_ratio)
        start_x = int(w * (1 - straight_ratio) / 2)
        end_x = int(w * (1 + straight_ratio) / 2)
        balloon[:cut_height, start_x:end_x] = 255
        mask[:cut_height, start_x:end_x] = 0
    else:  # bottom
        cut_height = int(h * cut_depth_ratio)
        start_x = int(w * (1 - straight_ratio) / 2)
        end_x = int(w * (1 + straight_ratio) / 2)
        balloon[h-cut_height:, start_x:end_x] = 255
        mask[h-cut_height:, start_x:end_x] = 0
    
    return balloon, mask


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
    lt: Point  # left-top
    rt: Point  # right-top
    lb: Point  # left-bottom
    rb: Point  # right-bottom


@dataclass
class CornerPosition:
    """コマの角の位置情報"""
    position: Point
    corner_type: str  # 'lt', 'rt', 'lb', 'rb'
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


def place_balloon_at_corner(background: np.ndarray, balloon: np.ndarray, mask: np.ndarray,
                           corner_pos: CornerPosition, balloon_scale: float, 
                           corner_ratio: float, cfg: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    """コマの角に吹き出しを配置"""
    bg_h, bg_w = background.shape[:2]
    
    cropped_balloon, cropped_mask, bbox = crop_balloon_and_mask(balloon, mask)
    
    if cropped_balloon.size == 0 or cropped_mask.size == 0:
        return background, np.zeros((bg_h, bg_w), dtype=np.uint8), {}
    
    crop_h, crop_w = cropped_balloon.shape[:2]
    
    new_balloon_w, new_balloon_h = calculate_area_based_size(
        crop_w, crop_h, bg_w, bg_h, balloon_scale,
        max_w_ratio=cfg.get("MAX_WIDTH_RATIO", 0.3),
        max_h_ratio=cfg.get("MAX_HEIGHT_RATIO", 0.4)
    )
    
    balloon_resized = cv2.resize(cropped_balloon, (new_balloon_w, new_balloon_h))
    mask_resized = cv2.resize(cropped_mask, (new_balloon_w, new_balloon_h))
    
    mask_coords = cv2.findNonZero(mask_resized)
    if mask_coords is not None:
        mask_x, mask_y, mask_w, mask_h = cv2.boundingRect(mask_coords)
    else:
        mask_x, mask_y, mask_w, mask_h = 0, 0, new_balloon_w, new_balloon_h
    
    corner_x, corner_y = corner_pos.position
    
    overhang_ratio = cfg.get("OVERHANG_RATIO", 0.15)
    overhang_w = int(mask_w * overhang_ratio)
    overhang_h = int(mask_h * overhang_ratio)
    border_width = cfg.get("BORDER_WIDTH", 10)
    
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
    
    if corner_pos.panel_mask is not None and corner_pos.panel_mask.size > 0:
        bbox_x, bbox_y, bbox_w, bbox_h = corner_pos.panel_bbox
        
        balloon_start_x = max(0, x)
        balloon_start_y = max(0, y)
        balloon_end_x = min(x + new_balloon_w, bg_w)
        balloon_end_y = min(y + new_balloon_h, bg_h)
        
        overlap_start_x = max(balloon_start_x, bbox_x)
        overlap_start_y = max(balloon_start_y, bbox_y)
        overlap_end_x = min(balloon_end_x, bbox_x + bbox_w)
        overlap_end_y = min(balloon_end_y, bbox_y + bbox_h)
        
        overlap_w = overlap_end_x - overlap_start_x
        overlap_h = overlap_end_y - overlap_start_y
        
        if overlap_w <= 0 or overlap_h <= 0:
            return background, result_mask, {}
        
        balloon_offset_x = overlap_start_x - x
        balloon_offset_y = overlap_start_y - y
        panel_mask_offset_x = overlap_start_x - bbox_x
        panel_mask_offset_y = overlap_start_y - bbox_y
        
        balloon_region_temp = balloon_resized[balloon_offset_y:balloon_offset_y + overlap_h,
                                             balloon_offset_x:balloon_offset_x + overlap_w]
        mask_region_temp = mask_resized[balloon_offset_y:balloon_offset_y + overlap_h,
                                       balloon_offset_x:balloon_offset_x + overlap_w]
        panel_mask_region = corner_pos.panel_mask[panel_mask_offset_y:panel_mask_offset_y + overlap_h,
                                                  panel_mask_offset_x:panel_mask_offset_x + overlap_w]
        
        if panel_mask_region.size == 0 or mask_region_temp.size == 0:
            return background, result_mask, {}
        
        panel_mask_h, panel_mask_w = corner_pos.panel_mask.shape[:2]
        panel_rect_mask = np.ones((panel_mask_h, panel_mask_w), dtype=np.uint8) * 255
        
        safe_distance = int(cfg.get("PANEL_SAFE_DISTANCE", 5))
        panel_safe = panel_rect_mask.copy()
        
        if safe_distance > 0:
            panel_safe[:safe_distance, :] = 0
            panel_safe[-safe_distance:, :] = 0
            panel_safe[:, :safe_distance] = 0
            panel_safe[:, -safe_distance:] = 0
        
        panel_safe = cv2.bitwise_and(panel_safe, corner_pos.panel_mask)
        
        panel_safe_region = panel_safe[panel_mask_offset_y:panel_mask_offset_y + overlap_h,
                                       panel_mask_offset_x:panel_mask_offset_x + overlap_w]
        
        final_mask_region = cv2.bitwise_and(mask_region_temp, panel_safe_region)
        
        panel_border = cv2.subtract(corner_pos.panel_mask, panel_safe)
        panel_border_region = panel_border[panel_mask_offset_y:panel_mask_offset_y + overlap_h,
                                          panel_mask_offset_x:panel_mask_offset_x + overlap_w]
        
        mask_norm = final_mask_region.astype(np.float32) / 255.0
        mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
        
        bg_region = result_img[overlap_start_y:overlap_end_y, overlap_start_x:overlap_end_x]
        
        if balloon_region_temp.shape == bg_region.shape and mask_3ch.shape == bg_region.shape:
            blended = balloon_region_temp.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
            result_img[overlap_start_y:overlap_end_y, overlap_start_x:overlap_end_x] = blended.astype(np.uint8)
            result_mask[overlap_start_y:overlap_end_y, overlap_start_x:overlap_end_x] = final_mask_region
            
            orig_bg_region = background[overlap_start_y:overlap_end_y, overlap_start_x:overlap_end_x]
            border_idx = (panel_border_region > 0)
            result_img[overlap_start_y:overlap_end_y, overlap_start_x:overlap_end_x][border_idx] = orig_bg_region[border_idx]
        else:
            return background, result_mask, {}
    else:
        start_x = max(0, x)
        start_y = max(0, y)
        end_x = min(x + new_balloon_w, bg_w)
        end_y = min(y + new_balloon_h, bg_h)
        
        clip_w = end_x - start_x
        clip_h = end_y - start_y
        
        if clip_w <= 0 or clip_h <= 0:
            return background, result_mask, {}
        
        balloon_start_x = start_x - x
        balloon_start_y = start_y - y
        
        balloon_region = balloon_resized[balloon_start_y:balloon_start_y + clip_h,
                                        balloon_start_x:balloon_start_x + clip_w]
        mask_region = mask_resized[balloon_start_y:balloon_start_y + clip_h,
                                   balloon_start_x:balloon_start_x + clip_w]
        
        mask_norm = mask_region.astype(np.float32) / 255.0
        mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
        
        bg_region = result_img[start_y:end_y, start_x:end_x]
        
        if balloon_region.shape == bg_region.shape and mask_3ch.shape == bg_region.shape:
            blended = balloon_region.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
            result_img[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)
            result_mask[start_y:end_y, start_x:end_x] = mask_region
        else:
            return background, result_mask, {}
    
    detail_info = {
        "corner_type": corner_pos.corner_type,
        "corner_position": f"({corner_x},{corner_y})",
        "balloon_position": f"({x},{y})",
        "balloon_size": f"{new_balloon_w}x{new_balloon_h}",
        "scale": f"{balloon_scale:.3f}",
        "corner_ratio": f"{corner_ratio:.3f}",
        "overhang": f"{overhang_w}x{overhang_h}"
    }
    
    return result_img, result_mask, detail_info


# =============================================================================
# 合成関数
# =============================================================================

def composite_random_balloons(background_path: str, balloon_mask_pairs: list,
                             cfg: dict = None, use_augmentation: bool = False) -> tuple:
    """ランダム配置による吹き出し合成"""
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
    
    for balloon_path, mask_path in selected_pairs:
        balloon = cv2.imread(balloon_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if balloon is None or mask is None:
            continue
        
        # データ拡張を適用
        if use_augmentation:
            balloon, mask = augment_balloon_and_mask(balloon, mask, cfg)
        
        cropped_balloon, cropped_mask, bbox = crop_balloon_and_mask(balloon, mask)
        
        if cropped_balloon.size == 0 or cropped_mask.size == 0:
            continue

        crop_h, crop_w = cropped_balloon.shape[:2]
        
        try:
            balloon_scale = sample_scale(bg_w, crop_w, cfg)
        except KeyError:
            balloon_scale = random.uniform(cfg.get("SCALE_RANGE", (0.1, 0.4))[0], 
                                         cfg.get("SCALE_RANGE", (0.1, 0.4))[1])
        
        new_balloon_w, new_balloon_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, balloon_scale,
            mask=cropped_mask,
            max_w_ratio=cfg.get("MAX_WIDTH_RATIO", 0.3),
            max_h_ratio=cfg.get("MAX_HEIGHT_RATIO", 0.4)
        )
        
        if new_balloon_w >= bg_w or new_balloon_h >= bg_h:
            continue
        
        balloon_resized = cv2.resize(cropped_balloon, (new_balloon_w, new_balloon_h))
        mask_resized = cv2.resize(cropped_mask, (new_balloon_w, new_balloon_h))
        
        placed = False
        best_position = None
        min_overlap_ratio = float('inf')
        
        max_attempts = cfg.get("MAX_ATTEMPTS", 200)
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
                "placement_type": "random",
                "original_size": f"{balloon.shape[1]}x{balloon.shape[0]}",
                "cropped_size": f"{crop_w}x{crop_h}",
                "final_size": f"{new_balloon_w}x{new_balloon_h}",
                "position": f"({x},{y})",
                "scale": f"{balloon_scale:.3f}",
            }
            balloon_details.append(balloon_info)
    
    return result_img, result_mask, successfully_placed, balloon_details


def composite_corner_aligned_balloons(background_path: str, balloon_mask_pairs: list,
                                     corner_ratio: float = 0.3, cfg: dict = None,
                                     use_augmentation: bool = False) -> tuple:
    """コマの角配置とランダム配置を組み合わせた吹き出し合成"""
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
    
    total_balloons_available = len(balloon_mask_pairs)
    num_balloons_total = sample_num_balloons(cfg, total_balloons_available)
    num_balloons_total = min(num_balloons_total, total_balloons_available)
    
    if num_balloons_total == 0:
        return background, np.zeros((bg_h, bg_w), dtype=np.uint8), [], []
    
    corner_placement_ratio = cfg.get("CORNER_PLACEMENT_RATIO", 0.4)
    num_corner_balloons = int(num_balloons_total * corner_placement_ratio)
    num_random_balloons = num_balloons_total - num_corner_balloons
    
    if corners:
        num_corner_balloons = min(num_corner_balloons, len(corners))
        num_random_balloons = num_balloons_total - num_corner_balloons
    else:
        num_corner_balloons = 0
        num_random_balloons = num_balloons_total
    
    shuffled_pairs = balloon_mask_pairs.copy()
    random.shuffle(shuffled_pairs)
    corner_pairs = shuffled_pairs[:num_corner_balloons]
    random_pairs = shuffled_pairs[num_corner_balloons:num_corner_balloons + num_random_balloons]
    
    selected_corners = random.sample(corners, num_corner_balloons) if num_corner_balloons > 0 else []
    
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    successfully_placed = []
    balloon_details = []
    occupied_regions = []
    
    def check_overlap(x, y, w, h, margin=20):
        for ox, oy, ow, oh in occupied_regions:
            if not (x + w + margin < ox or x > ox + ow + margin or
                    y + h + margin < oy or y > oy + oh + margin):
                return True
        return False
    
    # 1. コマの角に配置
    for corner_pos, (balloon_path, mask_path) in zip(selected_corners, corner_pairs):
        balloon = cv2.imread(balloon_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if balloon is None or mask is None:
            continue
        
        # データ拡張を適用
        if use_augmentation:
            balloon, mask = augment_balloon_and_mask(balloon, mask, cfg)
        
        try:
            balloon_scale = sample_scale(bg_w, balloon.shape[1], cfg)
        except KeyError:
            balloon_scale = random.uniform(cfg.get("SCALE_RANGE", (0.1, 0.4))[0], 
                                         cfg.get("SCALE_RANGE", (0.1, 0.4))[1])
        
        try:
            result_img, current_mask, detail_info = place_balloon_at_corner(
                result_img, balloon, mask, corner_pos, balloon_scale, corner_ratio, cfg
            )
            
            if not detail_info:
                continue
            
            result_mask = np.maximum(result_mask, current_mask)
            
            successfully_placed.append(Path(balloon_path).stem)
            
            balloon_info = {
                "balloon_file": Path(balloon_path).name,
                "placement_type": "corner",
                **detail_info
            }
            balloon_details.append(balloon_info)
            
            pos_str = detail_info.get("balloon_position", "(0,0)")
            size_str = detail_info.get("balloon_size", "0x0")
            bx, by = map(int, pos_str.strip("()").split(","))
            bw, bh = map(int, size_str.split("x"))
            occupied_regions.append((bx, by, bw, bh))
            
        except Exception as e:
            if cfg.get("DEBUG", False):
                import traceback
                print(f"角配置エラー: {e}")
                traceback.print_exc()
            continue
    
    # 2. ランダム位置に配置
    for balloon_path, mask_path in random_pairs:
        balloon = cv2.imread(balloon_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if balloon is None or mask is None:
            continue
        
        # データ拡張を適用
        if use_augmentation:
            balloon, mask = augment_balloon_and_mask(balloon, mask, cfg)
        
        cropped_balloon, cropped_mask, bbox = crop_balloon_and_mask(balloon, mask)
        if cropped_balloon.size == 0:
            continue
        
        crop_h, crop_w = cropped_balloon.shape[:2]
        
        try:
            balloon_scale = sample_scale(bg_w, crop_w, cfg)
        except KeyError:
            balloon_scale = random.uniform(cfg.get("SCALE_RANGE", (0.1, 0.4))[0], 
                                         cfg.get("SCALE_RANGE", (0.1, 0.4))[1])
        
        new_balloon_w, new_balloon_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, balloon_scale,
            max_w_ratio=cfg.get("MAX_WIDTH_RATIO", 0.3),
            max_h_ratio=cfg.get("MAX_HEIGHT_RATIO", 0.4)
        )
        
        placed = False
        for attempt in range(10):
            max_x = max(0, bg_w - new_balloon_w)
            max_y = max(0, bg_h - new_balloon_h)
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0
            
            if not check_overlap(x, y, new_balloon_w, new_balloon_h):
                placed = True
                break
        
        if not placed:
            continue
        
        try:
            balloon_resized = cv2.resize(cropped_balloon, (new_balloon_w, new_balloon_h))
            mask_resized = cv2.resize(cropped_mask, (new_balloon_w, new_balloon_h))
            
            mask_norm = mask_resized.astype(np.float32) / 255.0
            mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
            
            bg_region = result_img[y:y+new_balloon_h, x:x+new_balloon_w]
            blended = balloon_resized.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
            result_img[y:y+new_balloon_h, x:x+new_balloon_w] = blended.astype(np.uint8)
            
            result_mask[y:y+new_balloon_h, x:x+new_balloon_w] = np.maximum(
                result_mask[y:y+new_balloon_h, x:x+new_balloon_w], mask_resized
            )
            
            successfully_placed.append(Path(balloon_path).stem)
            
            balloon_info = {
                "balloon_file": Path(balloon_path).name,
                "placement_type": "random",
                "position": f"({x},{y})",
                "size": f"{new_balloon_w}x{new_balloon_h}",
                "scale": f"{balloon_scale:.3f}"
            }
            balloon_details.append(balloon_info)
            
            occupied_regions.append((x, y, new_balloon_w, new_balloon_h))
            
        except Exception as e:
            if cfg.get("DEBUG", False):
                print(f"ランダム配置エラー: {e}")
            continue
    
    return result_img, result_mask, successfully_placed, balloon_details


# =============================================================================
# データセット生成
# =============================================================================

def split_balloons(balloon_mask_pairs: list, train_ratio: float = 0.8, seed: int = 42) -> tuple:
    """吹き出しをtrain用とval用に分割する"""
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
                          target_count: int, cfg: dict, 
                          use_corner_placement: bool = True,
                          use_augmentation: bool = False,
                          corner_ratio: float = 0.3) -> int:
    """指定されたsplit（trainまたはval）のデータセットを生成する"""
    print(f"\n=== {split_name} データセット生成開始 ===")
    print(f"目標画像数: {target_count}")
    print(f"背景画像数: {len(background_files)}")
    print(f"利用可能吹き出し数: {len(balloon_pairs)}")
    print(f"パネル配置（角+ランダム）: {'有効' if use_corner_placement else '無効'}")
    print(f"データ拡張: {'有効' if use_augmentation else '無効'}")
    
    log_file_path = os.path.join(output_dir, "..", f"{split_name}_composition_log.txt")
    
    current_number = 1
    success_count = 0
    bg_idx = 0
    
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== {split_name.upper()} データセット合成ログ ===\n")
        log_file.write(f"生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"目標画像数: {target_count}\n")
        log_file.write(f"パネル配置: {'有効' if use_corner_placement else '無効'}\n")
        log_file.write(f"データ拡張: {'有効' if use_augmentation else '無効'}\n")
        log_file.write("=" * 80 + "\n\n")
    
    while success_count < target_count:
        bg_path = background_files[bg_idx % len(background_files)]
        bg_name = Path(bg_path).stem
        
        try:
            if use_corner_placement:
                result_img, result_mask, placed_balloons, balloon_details = composite_corner_aligned_balloons(
                    bg_path, balloon_pairs, corner_ratio=corner_ratio, cfg=cfg,
                    use_augmentation=use_augmentation
                )
            else:
                result_img, result_mask, placed_balloons, balloon_details = composite_random_balloons(
                    bg_path, balloon_pairs, cfg=cfg, use_augmentation=use_augmentation
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
                    log_file.write(f"    吹き出し{i}: {detail.get('balloon_file', 'N/A')}\n")
                    log_file.write(f"      配置タイプ: {detail.get('placement_type', 'unknown')}\n")
                    if 'corner_type' in detail:
                        log_file.write(f"      角タイプ: {detail['corner_type']}\n")
                    log_file.write(f"      位置: {detail.get('position', detail.get('balloon_position', 'N/A'))}\n")
                    log_file.write(f"      サイズ: {detail.get('final_size', detail.get('balloon_size', detail.get('size', 'N/A')))}\n")
                
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
        description="吹き出し合成データセット作成（統合版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本（デフォルトでパネル検出＋角配置を使用）
  python create_syn_balloon_dataset.py --output-dir balloon_dataset/syn500 --target-images 500

  # パネル配置を無効化（完全ランダム）
  python create_syn_balloon_dataset.py --output-dir balloon_dataset/syn500-random --target-images 500 --no-panel-placement

  # データ拡張を有効化
  python create_syn_balloon_dataset.py --output-dir balloon_dataset/syn500-aug --target-images 500 --augmentation
        """
    )
    
    # 入力ディレクトリ
    parser.add_argument("--balloon-dir", default="balloons/images", help="吹き出し画像ディレクトリ")
    parser.add_argument("--mask-dir", default="balloons/masks", help="マスク画像ディレクトリ")
    parser.add_argument("--background-dir", default="generated_double_backs_1536x1024", help="背景画像ディレクトリ")
    
    # 出力
    parser.add_argument("--output-dir", required=True, help="出力ディレクトリ")
    parser.add_argument("--target-images", type=int, required=True, help="生成する画像総数")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="train用の比率")
    
    # 機能オプション
    parser.add_argument("--no-panel-placement", action="store_true", help="パネル配置を無効化（完全ランダム配置）")
    parser.add_argument("--augmentation", action="store_true", help="データ拡張を有効化")
    parser.add_argument("--corner-ratio", type=float, default=0.3, help="コマ角配置時の角比率")
    parser.add_argument("--corner-placement-ratio", type=float, default=0.4, help="角配置とランダム配置の比率")
    
    # スケール設定
    parser.add_argument("--scale-range", nargs=2, type=float, default=[0.07, 0.12], help="スケール範囲")
    parser.add_argument("--num-balloons-range", nargs=2, type=int, default=[5, 17], help="配置数範囲")
    
    # 統計ファイル
    parser.add_argument("--stats-file", default="balloon_count_statistics.txt", help="統計ファイル")
    
    args = parser.parse_args()
    
    # 設定
    CFG = {
        "SCALE_RANGE": tuple(args.scale_range),
        "NUM_BALLOONS_RANGE": tuple(args.num_balloons_range),
        "MAX_ATTEMPTS": 200,
        "TRAIN_RATIO": args.train_ratio,
        "BALLOON_SPLIT_SEED": 42,
        
        # 統計情報ベースのサンプリング
        "SCALE_MODE": "lognormal",
        "SCALE_MEDIAN": 0.007226,
        "SCALE_SIGMA": 0.85,
        "SCALE_CLIP": (0.002, 0.02),
        "COUNT_PROBS": None,
        
        # 面積ベースリサイズ
        "MAX_WIDTH_RATIO": 0.3,
        "MAX_HEIGHT_RATIO": 0.4,
        
        # コーナー配置
        "CORNER_PLACEMENT_RATIO": args.corner_placement_ratio,
        "OVERHANG_RATIO": 0.25,
        "BORDER_WIDTH": 10,
        "PANEL_SAFE_DISTANCE": 5,
    }
    
    # 統計ファイル読み込み
    if args.stats_file and os.path.exists(args.stats_file):
        print(f"統計ファイルを読み込み: {args.stats_file}")
        try:
            CFG["COUNT_PROBS"] = load_count_probs(args.stats_file)
            print("✓ 統計ベースの吹き出し個数サンプリングを有効化")
        except Exception as e:
            print(f"統計ファイル読み込みエラー: {e}")
    
    # ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 設定をファイル出力
    use_panel_placement = not args.no_panel_placement
    config_output = {
        "timestamp": datetime.now().isoformat(),
        "script_name": "create_syn_balloon_dataset.py",
        "output_path": args.output_dir,
        "target_images": args.target_images,
        "panel_placement": use_panel_placement,
        "augmentation": args.augmentation,
        "config": {k: v for k, v in CFG.items() if k != "COUNT_PROBS"},
    }
    
    config_file_path = os.path.join(args.output_dir, "config.json")
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"設定情報を保存: {config_file_path}")
    
    print("\n=== 吹き出し合成データセット 作成開始 ===")
    print(f"パネル配置（角+ランダム）: {'有効' if use_panel_placement else '無効（完全ランダム）'}")
    print(f"データ拡張: {'有効' if args.augmentation else '無効'}")
    
    # 吹き出しとマスクの対応を取得
    balloon_mask_pairs = []
    print("吹き出し・マスクペアを検索中...")
    for balloon_file in os.listdir(args.balloon_dir):
        if balloon_file.endswith(('.png', '.jpg', '.jpeg')):
            balloon_path = os.path.join(args.balloon_dir, balloon_file)
            balloon_stem = Path(balloon_file).stem
            
            mask_file = f"{balloon_stem}_mask.png"
            mask_path = os.path.join(args.mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                balloon_mask_pairs.append((balloon_path, mask_path))
    
    # 背景画像を取得
    background_files = []
    for bg_file in os.listdir(args.background_dir):
        if bg_file.endswith(('.png', '.jpg', '.jpeg')):
            background_files.append(os.path.join(args.background_dir, bg_file))
    
    print(f"見つかった吹き出し: {len(balloon_mask_pairs)}個")
    print(f"見つかった背景: {len(background_files)}個")
    
    # 吹き出しを分割
    print(f"\n吹き出しを分割中（train:{CFG['TRAIN_RATIO']:.0%}, val:{1-CFG['TRAIN_RATIO']:.0%}）...")
    train_balloons, val_balloons = split_balloons(
        balloon_mask_pairs, CFG["TRAIN_RATIO"], CFG["BALLOON_SPLIT_SEED"]
    )
    
    print(f"train用吹き出し: {len(train_balloons)}個")
    print(f"val用吹き出し: {len(val_balloons)}個")
    
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
        background_files, train_balloons,
        train_img_dir, train_mask_dir, "train", train_target, CFG,
        use_corner_placement=use_panel_placement,
        use_augmentation=args.augmentation,
        corner_ratio=args.corner_ratio
    )
    
    # val データセット生成
    val_img_dir = os.path.join(args.output_dir, "val", "images")
    val_mask_dir = os.path.join(args.output_dir, "val", "masks")
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    
    val_count = generate_dataset_split(
        background_files, val_balloons,
        val_img_dir, val_mask_dir, "val", val_target, CFG,
        use_corner_placement=use_panel_placement,
        use_augmentation=args.augmentation,
        corner_ratio=args.corner_ratio
    )
    
    # 最終レポート
    print(f"\n=== 吹き出し合成データセット 作成完了 ===")
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
        "train_balloons_used": len(train_balloons),
        "val_balloons_used": len(val_balloons),
    }
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    
    print(f"\n設定・統計情報: {config_file_path}")


if __name__ == "__main__":
    main()
