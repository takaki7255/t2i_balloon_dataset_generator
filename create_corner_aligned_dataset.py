"""
コマの角に合わせて吹き出しを配置する合成データセット作成スクリプト

frame_separation.pyのコマ抽出機能と、create_syn_dataset.pyの合成方法を組み合わせて、
コマの角に吹き出しを配置し、角の形状に合わせて吹き出しを切り取る機能を提供します。
rm -rf corner_erode_test && python create_corner_aligned_dataset.py --background-dir generated_double_backs --balloon-dir generated_balloons --mask-dir masks --output-dir temp_corner_erode_test --mask-output-dir temp_corner_erode_test_masks --final-output-dir corner_erode_test --corner-ratio 0.4 --target-images 100
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
        
        # 実データの分位点を正確に再現するための調整済みパラメータ
        target_median = 0.007226
        mu = np.log(target_median)
        sigma = 0.85  # 実データの分布形状に最適化された値
        
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


@dataclass
class CornerPosition:
    """コマの角の位置情報"""
    position: Point  # 角の座標
    corner_type: str  # 'lt', 'rt', 'lb', 'rb'
    panel_quad: PanelQuad  # パネルの四角形情報
    panel_bbox: Tuple[int, int, int, int]  # パネルのバウンディングボックス
    panel_mask: np.ndarray  # パネルのマスク


def detect_panels_simple(image: np.ndarray, 
                        area_ratio_threshold: float = 0.85,
                        min_area: int = 10000) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]]:
    """
    シンプルな二値化・輪郭抽出によるコマ検出
    
    Args:
        image: 入力画像（カラー）
        area_ratio_threshold: 輪郭面積/バウンディングボックス面積の閾値（デフォルト0.85）
        min_area: 最小コマ面積（デフォルト10000ピクセル）
    
    Returns:
        List[(panel_mask, bbox, contour)]: パネルマスク、バウンディングボックス、輪郭のリスト
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
    img_h, img_w = image.shape[:2]
    
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
            
            # 輪郭も返す（元の座標系）
            panels.append((panel_mask, (x, y, w, h), contour))
    
    return panels


def extract_panel_corners(background_path: str, 
                         area_ratio_threshold: float = 0.85,
                         min_area: int = 10000) -> List[CornerPosition]:
    """
    背景画像からコマを抽出し、角の位置を取得
    
    Args:
        background_path: 背景画像のパス
        area_ratio_threshold: 輪郭面積/バウンディングボックス面積の閾値
        min_area: 最小コマ面積
    """
    # 背景画像読み込み
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")
    
    # コマ検出
    panels = detect_panels_simple(background, area_ratio_threshold, min_area)
    
    corners = []
    for panel_mask, bbox, contour in panels:
        x, y, w, h = bbox
        
        # 輪郭を多角形近似して角を抽出
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 近似した輪郭から4つの角を抽出
        # 輪郭点が4点未満の場合はバウンディングボックスの角を使用
        if len(approx) >= 4:
            # 輪郭点をリストに変換
            points = approx.reshape(-1, 2)
            
            # 左上・右上・左下・右下の角を見つける
            # y座標でソート
            points_sorted_y = points[points[:, 1].argsort()]
            top_points = points_sorted_y[:len(points_sorted_y)//2]  # 上半分
            bottom_points = points_sorted_y[len(points_sorted_y)//2:]  # 下半分
            
            # 上半分をx座標でソート
            top_points = top_points[top_points[:, 0].argsort()]
            lt = Point(int(top_points[0][0]), int(top_points[0][1]))  # 左上
            rt = Point(int(top_points[-1][0]), int(top_points[-1][1]))  # 右上
            
            # 下半分をx座標でソート
            bottom_points = bottom_points[bottom_points[:, 0].argsort()]
            lb = Point(int(bottom_points[0][0]), int(bottom_points[0][1]))  # 左下
            rb = Point(int(bottom_points[-1][0]), int(bottom_points[-1][1]))  # 右下
        else:
            # 輪郭点が少ない場合はバウンディングボックスを使用
            lt = Point(x, y)
            rt = Point(x + w, y)
            lb = Point(x, y + h)
            rb = Point(x + w, y + h)
        
        # 四隅の座標を定義
        quad = PanelQuad(lt=lt, rt=rt, lb=lb, rb=rb)
        
        # 各角の位置を記録
        corner_types = ['lt', 'rt', 'lb', 'rb']
        for corner_type in corner_types:
            position = getattr(quad, corner_type)
            corners.append(CornerPosition(
                position=position,
                corner_type=corner_type,
                panel_quad=quad,
                panel_bbox=bbox,
                panel_mask=panel_mask
            ))
    
    return corners


def resize_balloon_mask(balloon_mask: np.ndarray, balloon_size: Tuple[int, int]) -> np.ndarray:
    """吹き出しマスクをリサイズ（角の切り取りなし）"""
    balloon_w, balloon_h = balloon_size  # (width, height)の順序
    mask_resized = cv2.resize(balloon_mask, (balloon_w, balloon_h))
    return mask_resized


def place_balloon_at_corner(background: np.ndarray, balloon: np.ndarray, mask: np.ndarray,
                           corner_pos: CornerPosition, balloon_scale: float, 
                           corner_ratio: float, cfg: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    """コマの角に吹き出しを配置（はみ出し配置＋パネルマスクでの論理積）"""
    bg_h, bg_w = background.shape[:2]
    
    # 吹き出しをクロップ
    cropped_balloon, cropped_mask, bbox = crop_balloon_and_mask(balloon, mask)
    
    if cropped_balloon.size == 0 or cropped_mask.size == 0:
        return background, np.zeros((bg_h, bg_w), dtype=np.uint8), {}
    
    crop_h, crop_w = cropped_balloon.shape[:2]
    
    # 面積ベースのサイズ計算
    new_balloon_w, new_balloon_h = calculate_area_based_size(
        crop_w, crop_h, bg_w, bg_h, balloon_scale,
        max_w_ratio=cfg.get("MAX_WIDTH_RATIO", 0.3),
        max_h_ratio=cfg.get("MAX_HEIGHT_RATIO", 0.4)
    )
    
    # 吹き出しをリサイズ
    balloon_resized = cv2.resize(cropped_balloon, (new_balloon_w, new_balloon_h))
    
    # マスクもリサイズ（角の切り取りは行わない）
    mask_resized = resize_balloon_mask(cropped_mask, (new_balloon_w, new_balloon_h))
    
    # マスクの実際の内容領域（非ゼロ領域）を取得
    # これにより余白を除いた実際の吹き出し領域がわかる
    mask_coords = cv2.findNonZero(mask_resized)
    if mask_coords is not None:
        mask_x, mask_y, mask_w, mask_h = cv2.boundingRect(mask_coords)
    else:
        # マスクが空の場合は全体を使用
        mask_x, mask_y, mask_w, mask_h = 0, 0, new_balloon_w, new_balloon_h
    
    # コマの角から外側にはみ出すように配置
    corner_x, corner_y = corner_pos.position
    
    # はみ出し量を計算（マスクの実際のサイズに基づいて10-15%程度）
    overhang_ratio = cfg.get("OVERHANG_RATIO", 0.15)  # デフォルト15%
    overhang_w = int(mask_w * overhang_ratio)
    overhang_h = int(mask_h * overhang_ratio)
    
    # 輪郭線保護のための内側へのオフセット（デフォルト5ピクセル）
    border_width = cfg.get("BORDER_WIDTH", 10)
    
    # 角のタイプに応じて配置位置を計算（マスクの実際の領域を考慮）
    # 輪郭線を保護するため、border_widthピクセル分内側に配置
    if corner_pos.corner_type == 'lt':
        # 左上角：マスクの左上端を基準にはみ出させ、右下方向に内側へオフセット
        x, y = corner_x - mask_x - overhang_w + border_width, corner_y - mask_y - overhang_h + border_width
    elif corner_pos.corner_type == 'rt':
        # 右上角：マスクの右上端を基準にはみ出させ、左下方向に内側へオフセット
        x, y = corner_x - (mask_x + mask_w) + overhang_w - border_width, corner_y - mask_y - overhang_h + border_width
    elif corner_pos.corner_type == 'lb':
        # 左下角：マスクの左下端を基準にはみ出させ、右上方向に内側へオフセット
        x, y = corner_x - mask_x - overhang_w + border_width, corner_y - (mask_y + mask_h) + overhang_h - border_width
    elif corner_pos.corner_type == 'rb':
        # 右下角：マスクの右下端を基準にはみ出させ、左上方向に内側へオフセット
        x, y = corner_x - (mask_x + mask_w) + overhang_w - border_width, corner_y - (mask_y + mask_h) + overhang_h - border_width
    else:
        x, y = corner_x, corner_y
    
    # 合成実行
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    # まず、パネルマスクとの論理積を取る（はみ出し部分を切り取る）
    if corner_pos.panel_mask is not None and corner_pos.panel_mask.size > 0:
        # panel_maskはbbox領域のマスク（背景画像全体ではない）
        panel_mask_h, panel_mask_w = corner_pos.panel_mask.shape[:2]
        bbox_x, bbox_y, bbox_w, bbox_h = corner_pos.panel_bbox
        
        # デバッグ: パネルマスクのサイズと背景画像のサイズを確認
        if cfg.get("DEBUG", False):
            print(f"背景サイズ: {bg_w}x{bg_h}, パネルマスクサイズ: {panel_mask_w}x{panel_mask_h}")
            print(f"bbox: ({bbox_x},{bbox_y},{bbox_w},{bbox_h})")
            print(f"配置位置: ({x},{y}), 吹き出しサイズ: {new_balloon_w}x{new_balloon_h}")
        
        # 吹き出しとbbox領域の重なりを計算
        # 背景画像の座標系で計算
        balloon_start_x = max(0, x)
        balloon_start_y = max(0, y)
        balloon_end_x = min(x + new_balloon_w, bg_w)
        balloon_end_y = min(y + new_balloon_h, bg_h)
        
        # bbox領域との交差を計算
        overlap_start_x = max(balloon_start_x, bbox_x)
        overlap_start_y = max(balloon_start_y, bbox_y)
        overlap_end_x = min(balloon_end_x, bbox_x + bbox_w)
        overlap_end_y = min(balloon_end_y, bbox_y + bbox_h)
        
        # 重なり領域のサイズ
        overlap_w = overlap_end_x - overlap_start_x
        overlap_h = overlap_end_y - overlap_start_y
        
        if overlap_w <= 0 or overlap_h <= 0:
            # bbox領域との重なりがない場合はスキップ
            return background, result_mask, {}
        
        # 吹き出し内の対応する領域（吹き出し座標系）
        balloon_offset_x = overlap_start_x - x
        balloon_offset_y = overlap_start_y - y
        
        # パネルマスク内の対応する領域（bbox座標系）
        panel_mask_offset_x = overlap_start_x - bbox_x
        panel_mask_offset_y = overlap_start_y - bbox_y
        
        # 各領域を取得
        balloon_region_temp = balloon_resized[balloon_offset_y:balloon_offset_y + overlap_h,
                                             balloon_offset_x:balloon_offset_x + overlap_w]
        mask_region_temp = mask_resized[balloon_offset_y:balloon_offset_y + overlap_h,
                                       balloon_offset_x:balloon_offset_x + overlap_w]
        panel_mask_region = corner_pos.panel_mask[panel_mask_offset_y:panel_mask_offset_y + overlap_h,
                                                  panel_mask_offset_x:panel_mask_offset_x + overlap_w]
        
        # サイズチェック
        if panel_mask_region.size == 0 or mask_region_temp.size == 0:
            return background, result_mask, {}
        
        # 1) パネルマスクを四角形に近似して、内側領域を抽出
        # まず、パネルのバウンディングボックスから四角形マスクを作成
        panel_mask_h, panel_mask_w = corner_pos.panel_mask.shape[:2]
        panel_rect_mask = np.ones((panel_mask_h, panel_mask_w), dtype=np.uint8) * 255
        
        # 枠線保護のため、矩形を内側に収縮
        safe_distance = int(cfg.get("PANEL_SAFE_DISTANCE", 2))
        panel_safe = panel_rect_mask.copy()
        
        # 上下左右からsafe_distanceピクセル分内側の矩形マスクを作成
        if safe_distance > 0:
            panel_safe[:safe_distance, :] = 0  # 上端
            panel_safe[-safe_distance:, :] = 0  # 下端
            panel_safe[:, :safe_distance] = 0  # 左端
            panel_safe[:, -safe_distance:] = 0  # 右端
        
        # 元のパネルマスクとの論理積を取る（パネルの形状を保持）
        panel_safe = cv2.bitwise_and(panel_safe, corner_pos.panel_mask)
        
        # ROIへ切り出し
        panel_safe_region = panel_safe[panel_mask_offset_y:panel_mask_offset_y + overlap_h,
                                       panel_mask_offset_x:panel_mask_offset_x + overlap_w]
        
        # 吹き出しマスクと "内側だけ" の論理積
        final_mask_region = cv2.bitwise_and(mask_region_temp, panel_safe_region)
        
        # 2) "枠線そのもの" を特定（元マスク−内側マスク）。あとで再描画に使う
        panel_border = cv2.subtract(corner_pos.panel_mask, panel_safe)
        panel_border_region = panel_border[panel_mask_offset_y:panel_mask_offset_y + overlap_h,
                                          panel_mask_offset_x:panel_mask_offset_x + overlap_w]
        
        # 通常ブレンド
        mask_norm = final_mask_region.astype(np.float32) / 255.0
        mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
        
        bg_region = result_img[overlap_start_y:overlap_end_y, overlap_start_x:overlap_end_x]
        
        if balloon_region_temp.shape == bg_region.shape and mask_3ch.shape == bg_region.shape:
            # 吹き出しを合成
            blended = balloon_region_temp.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
            result_img[overlap_start_y:overlap_end_y, overlap_start_x:overlap_end_x] = blended.astype(np.uint8)
            result_mask[overlap_start_y:overlap_end_y, overlap_start_x:overlap_end_x] = final_mask_region
            
            # 3) ブレンド後、"枠線ピクセル" は元の背景で上書きして完全復元
            orig_bg_region = background[overlap_start_y:overlap_end_y, overlap_start_x:overlap_end_x]
            border_idx = (panel_border_region > 0)
            result_img[overlap_start_y:overlap_end_y, overlap_start_x:overlap_end_x][border_idx] = orig_bg_region[border_idx]
        else:
            return background, result_mask, {}
    else:
        # パネルマスクがない場合は通常の配置
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
            print(f"形状不一致: balloon={balloon_region.shape}, bg={bg_region.shape}, mask={mask_3ch.shape}")
            return background, result_mask, {}
    
    # 詳細情報
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


def composite_corner_aligned_balloons(background_path: str, balloon_mask_pairs: list,
                                     corner_ratio: float = 0.3, cfg: dict = None) -> tuple:
    """コマの角配置とランダム配置を組み合わせた吹き出し合成"""
    if cfg is None:
        cfg = {}
    
    # 背景画像読み込み
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"背景画像の読み込みに失敗: {background_path}")
    
    bg_h, bg_w = background.shape[:2]
    
    # コマの角を抽出
    try:
        corners = extract_panel_corners(background_path)
    except Exception as e:
        print(f"コマ抽出エラー: {e}")
        corners = []
    
    # 吹き出しの総数を統計に基づいて決定
    total_balloons_available = len(balloon_mask_pairs)
    num_balloons_total = sample_num_balloons(cfg, total_balloons_available)
    num_balloons_total = min(num_balloons_total, total_balloons_available)
    
    if num_balloons_total == 0:
        return background, np.zeros((bg_h, bg_w), dtype=np.uint8), [], []
    
    # 角配置とランダム配置の比率を決定（デフォルト: 角30%, ランダム70%）
    corner_placement_ratio = cfg.get("CORNER_PLACEMENT_RATIO", 0.3)
    num_corner_balloons = int(num_balloons_total * corner_placement_ratio)
    num_random_balloons = num_balloons_total - num_corner_balloons
    
    # 角配置の数を実際に配置可能な数に制限
    if corners:
        num_corner_balloons = min(num_corner_balloons, len(corners))
        num_random_balloons = num_balloons_total - num_corner_balloons
    else:
        num_corner_balloons = 0
        num_random_balloons = num_balloons_total
    
    # 吹き出しをシャッフルして分配
    shuffled_pairs = balloon_mask_pairs.copy()
    random.shuffle(shuffled_pairs)
    corner_pairs = shuffled_pairs[:num_corner_balloons]
    random_pairs = shuffled_pairs[num_corner_balloons:num_corner_balloons + num_random_balloons]
    
    # 角をランダムに選択
    selected_corners = random.sample(corners, num_corner_balloons) if num_corner_balloons > 0 else []
    
    result_img = background.copy()
    result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    successfully_placed = []
    balloon_details = []
    occupied_regions = []  # 配置済み領域を記録（重複回避用）
    
    def check_overlap(x, y, w, h, margin=20):
        """既存の配置領域との重複をチェック"""
        for ox, oy, ow, oh in occupied_regions:
            # マージンを含めた領域で重複判定
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
        
        # スケールサンプリング
        try:
            balloon_scale = sample_scale(bg_w, balloon.shape[1], cfg)
        except KeyError:
            balloon_scale = random.uniform(cfg.get("SCALE_RANGE", (0.1, 0.4))[0], 
                                         cfg.get("SCALE_RANGE", (0.1, 0.4))[1])
        
        # 角に配置
        try:
            result_img, current_mask, detail_info = place_balloon_at_corner(
                result_img, balloon, mask, corner_pos, balloon_scale, corner_ratio, cfg
            )
            
            if not detail_info:
                continue
            
            # マスクを合成
            result_mask = np.maximum(result_mask, current_mask)
            
            successfully_placed.append(Path(balloon_path).stem)
            
            # 詳細情報を記録
            balloon_info = {
                "balloon_file": Path(balloon_path).name,
                "placement_type": "corner",
                **detail_info
            }
            balloon_details.append(balloon_info)
            
            # 配置領域を記録（重複回避用）
            # detail_infoから位置とサイズを抽出
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
        
        # 吹き出しをクロップ
        cropped_balloon, cropped_mask, bbox = crop_balloon_and_mask(balloon, mask)
        if cropped_balloon.size == 0:
            continue
        
        crop_h, crop_w = cropped_balloon.shape[:2]
        
        # スケールサンプリング
        try:
            balloon_scale = sample_scale(bg_w, crop_w, cfg)
        except KeyError:
            balloon_scale = random.uniform(cfg.get("SCALE_RANGE", (0.1, 0.4))[0], 
                                         cfg.get("SCALE_RANGE", (0.1, 0.4))[1])
        
        # サイズ計算
        new_balloon_w, new_balloon_h = calculate_area_based_size(
            crop_w, crop_h, bg_w, bg_h, balloon_scale,
            max_w_ratio=cfg.get("MAX_WIDTH_RATIO", 0.3),
            max_h_ratio=cfg.get("MAX_HEIGHT_RATIO", 0.4)
        )
        
        # 重複しない位置を探す（最大10回試行）
        placed = False
        for attempt in range(10):
            max_x = max(0, bg_w - new_balloon_w)
            max_y = max(0, bg_h - new_balloon_h)
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0
            
            # 重複チェック
            if not check_overlap(x, y, new_balloon_w, new_balloon_h):
                placed = True
                break
        
        if not placed:
            continue
        
        # 吹き出しをリサイズして配置
        try:
            balloon_resized = cv2.resize(cropped_balloon, (new_balloon_w, new_balloon_h))
            mask_resized = cv2.resize(cropped_mask, (new_balloon_w, new_balloon_h))
            
            # アルファブレンディング
            mask_norm = mask_resized.astype(np.float32) / 255.0
            mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
            
            bg_region = result_img[y:y+new_balloon_h, x:x+new_balloon_w]
            blended = balloon_resized.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
            result_img[y:y+new_balloon_h, x:x+new_balloon_w] = blended.astype(np.uint8)
            
            # マスクも合成
            result_mask[y:y+new_balloon_h, x:x+new_balloon_w] = np.maximum(
                result_mask[y:y+new_balloon_h, x:x+new_balloon_w], mask_resized
            )
            
            successfully_placed.append(Path(balloon_path).stem)
            
            # 詳細情報を記録
            balloon_info = {
                "balloon_file": Path(balloon_path).name,
                "placement_type": "random",
                "position": f"({x},{y})",
                "size": f"{new_balloon_w}x{new_balloon_h}",
                "scale": f"{balloon_scale:.3f}"
            }
            balloon_details.append(balloon_info)
            
            # 配置領域を記録
            occupied_regions.append((x, y, new_balloon_w, new_balloon_h))
            
        except Exception as e:
            if cfg.get("DEBUG", False):
                print(f"ランダム配置エラー: {e}")
            continue
    
    return result_img, result_mask, successfully_placed, balloon_details


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
                          target_count: int, corner_ratio: float, cfg: dict, 
                          final_output_dir: str = None) -> int:
    """指定されたsplit（trainまたはval）のデータセットを生成する"""
    print(f"\n=== {split_name} データセット生成開始 ===")
    print(f"目標画像数: {target_count}")
    print(f"背景画像数: {len(background_files)}")
    print(f"利用可能吹き出し数: {len(balloon_pairs)}")
    print(f"角切り取り比率: {corner_ratio}")
    
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
        log_file.write(f"=== {split_name.upper()} コーナーアライメントデータセット合成ログ ===\n")
        log_file.write(f"生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"目標画像数: {target_count}\n")
        log_file.write(f"背景画像数: {len(background_files)}\n")
        log_file.write(f"利用可能吹き出し数: {len(balloon_pairs)}\n")
        log_file.write(f"角切り取り比率: {corner_ratio}\n")
        log_file.write("=" * 80 + "\n\n")
    
    # 目標数に達するまで生成
    while success_count < target_count:
        # 背景画像を循環使用
        bg_path = background_files[bg_idx % len(background_files)]
        bg_name = Path(bg_path).stem
        
        try:
            # コーナーアライメント合成実行
            result_img, result_mask, placed_balloons, balloon_details = composite_corner_aligned_balloons(
                bg_path, 
                balloon_pairs,
                corner_ratio=corner_ratio,
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
                log_file.write(f"  配置した吹き出し数: {len(balloon_details)}\n")
                
                for i, detail in enumerate(balloon_details, 1):
                    log_file.write(f"    吹き出し{i}: {detail['balloon_file']}\n")
                    log_file.write(f"      配置タイプ: {detail.get('placement_type', 'unknown')}\n")
                    
                    # 角配置の場合の詳細情報
                    if 'corner_type' in detail:
                        log_file.write(f"      角タイプ: {detail['corner_type']}\n")
                        log_file.write(f"      角位置: {detail['corner_position']}\n")
                        log_file.write(f"      配置位置: {detail['balloon_position']}\n")
                        log_file.write(f"      サイズ: {detail['balloon_size']}\n")
                        log_file.write(f"      スケール: {detail['scale']}\n")
                        log_file.write(f"      角比率: {detail['corner_ratio']}\n")
                        log_file.write(f"      はみ出し量: {detail['overhang']}\n")
                    # ランダム配置の場合の詳細情報
                    else:
                        log_file.write(f"      配置位置: {detail.get('position', 'N/A')}\n")
                        log_file.write(f"      サイズ: {detail.get('size', 'N/A')}\n")
                        log_file.write(f"      スケール: {detail.get('scale', 'N/A')}\n")
                
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
    parser = argparse.ArgumentParser(description="コマの角に合わせた吹き出し合成データセット作成")
    parser.add_argument("--balloon-dir", required=True, help="吹き出し画像ディレクトリ")
    parser.add_argument("--mask-dir", required=True, help="マスク画像ディレクトリ")
    parser.add_argument("--background-dir", required=True, help="背景画像ディレクトリ")
    parser.add_argument("--output-dir", required=True, help="出力画像ディレクトリ")
    parser.add_argument("--mask-output-dir", required=True, help="出力マスクディレクトリ")
    parser.add_argument("--final-output-dir", default="corner_aligned_dataset", help="最終出力ディレクトリ")
    parser.add_argument("--corner-ratio", type=float, default=0.3, help="角の切り取り比率")
    parser.add_argument("--target-images", type=int, default=100, help="生成する画像数")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="train用の比率")
    
    args = parser.parse_args()
    
    # final_output_dirが指定されていない場合はエラー
    if not args.final_output_dir:
        parser.error("--final-output-dir は必須です")
    
    # 設定
    CFG = {
        "SCALE_RANGE": (0.070, 0.120),
        "NUM_BALLOONS_RANGE": (1, 5),  # コーナーアライメント用に調整
        "MAX_ATTEMPTS": 100,
        "TRAIN_RATIO": args.train_ratio,
        "BALLOON_SPLIT_SEED": 39,
        
        # 統計情報ベースのサンプリング設定（create_syn_dataset.pyと同じ）
        "SCALE_MODE": "lognormal",
        "SCALE_MEAN": 0.008769,              # 実データ平均面積比
        "SCALE_STD": 0.006773,               # 実データ標準偏差
        "SCALE_CLIP": (0.002000, 0.020000),  # 実データ範囲
        "COUNT_PROBS": None,                 # 吹き出し個数の確率分布
        "COUNT_STATS_FILE": "balloon_count_statistics.txt",  # 統計ファイルのパス
        
        # 面積ベースリサイズ設定
        "MAX_WIDTH_RATIO": 0.20,
        "MAX_HEIGHT_RATIO": 0.30,
        
        # コーナー配置設定
        "OVERHANG_RATIO": 0.25,  # はみ出し比率（デフォルト0.15→0.25に変更で約67%増加）
    }
    
    # ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.mask_output_dir, exist_ok=True)
    os.makedirs(args.final_output_dir, exist_ok=True)
    
    # 設定をファイル出力
    config_output = {
        "timestamp": datetime.now().isoformat(),
        "script_name": "create_corner_aligned_dataset.py",
        "dataset_output_path": args.final_output_dir,
        "corner_ratio": args.corner_ratio,
        "target_images": args.target_images,
        "config": CFG,
        "input_directories": {
            "balloons_dir": args.balloon_dir,
            "masks_dir": args.mask_dir,
            "backgrounds_dir": args.background_dir
        }
    }
    
    config_file_path = os.path.join(args.final_output_dir, "config.json")
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"設定情報を保存: {config_file_path}")
    
    # 統計情報ファイルの読み込み（create_syn_dataset.pyと同じ）
    if CFG.get("COUNT_STATS_FILE") and os.path.exists(CFG["COUNT_STATS_FILE"]):
        print(f"統計情報ファイルを読み込み: {CFG['COUNT_STATS_FILE']}")
        try:
            CFG["COUNT_PROBS"] = load_count_probs(CFG["COUNT_STATS_FILE"])
            print(f"統計ベースの吹き出し個数サンプリングを有効化")
        except Exception as e:
            print(f"統計ファイル読み込みエラー: {e}")
            print("一様サンプリングを使用します")
    
    print("=== コーナーアライメントデータセット 作成開始 ===")
    
    # 吹き出しとマスクの対応を取得
    balloon_mask_pairs = []
    print("吹き出し・マスクペアを検索中...")
    for balloon_file in os.listdir(args.balloon_dir):
        if balloon_file.endswith(('.png', '.jpg', '.jpeg')):
            balloon_path = os.path.join(args.balloon_dir, balloon_file)
            balloon_stem = Path(balloon_file).stem
            
            # 対応するマスクファイルを検索
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
    train_target = int(args.target_images * CFG["TRAIN_RATIO"])
    val_target = args.target_images - train_target
    
    print(f"\n目標画像数:")
    print(f"train: {train_target}枚")
    print(f"val: {val_target}枚")
    print(f"合計: {args.target_images}枚")
    
    # train データセット生成
    train_img_dir = os.path.join(args.output_dir, "train")
    train_mask_dir = os.path.join(args.mask_output_dir, "train")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    
    train_count = generate_dataset_split(
        background_files, train_balloons,
        train_img_dir, train_mask_dir, "train", train_target, 
        args.corner_ratio, CFG, args.final_output_dir
    )
    
    # val データセット生成
    val_img_dir = os.path.join(args.output_dir, "val")
    val_mask_dir = os.path.join(args.mask_output_dir, "val")
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    
    val_count = generate_dataset_split(
        background_files, val_balloons,
        val_img_dir, val_mask_dir, "val", val_target, 
        args.corner_ratio, CFG, args.final_output_dir
    )
    
    # 最終的なデータセット構造を作成（final_output_dirが指定されている場合のみ）
    # output_dirとfinal_output_dirが異なる場合のみコピー処理を実行
    should_copy_to_final = (args.final_output_dir and 
                           os.path.abspath(args.output_dir) != os.path.abspath(args.final_output_dir))
    
    if should_copy_to_final:
        print(f"\n=== 最終データセット構造を作成中 ===")
        
        for split in ["train", "val"]:
            # ディレクトリ作成
            (Path(args.final_output_dir) / split / "images").mkdir(parents=True, exist_ok=True)
            (Path(args.final_output_dir) / split / "masks").mkdir(parents=True, exist_ok=True)
            
            # ファイルをコピー
            src_img_dir = os.path.join(args.output_dir, split)
            src_mask_dir = os.path.join(args.mask_output_dir, split)
            final_img_dir = os.path.join(args.final_output_dir, split, "images")
            final_mask_dir = os.path.join(args.final_output_dir, split, "masks")
            
            # 画像をコピー
            for img_file in os.listdir(src_img_dir):
                if img_file.endswith('.png'):
                    shutil.copy2(
                        os.path.join(src_img_dir, img_file),
                        os.path.join(final_img_dir, img_file)
                    )
            
            # マスクをコピー
            for mask_file in os.listdir(src_mask_dir):
                if mask_file.endswith('.png'):
                    shutil.copy2(
                        os.path.join(src_mask_dir, mask_file),
                        os.path.join(final_mask_dir, mask_file)
                    )
        
        # 一時ディレクトリを削除
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        if os.path.exists(args.mask_output_dir):
            shutil.rmtree(args.mask_output_dir)
    
    # 最終レポート
    final_report_dir = args.final_output_dir if should_copy_to_final else args.output_dir
    print(f"\n=== コーナーアライメントデータセット 作成完了 ===")
    print(f"出力先: {final_report_dir}")
    print(f"総生成画像数: {train_count + val_count}枚")
    print(f"角切り取り比率: {args.corner_ratio}")
    
    # 統計情報を収集
    stats = {
        "total_images": train_count + val_count,
        "train_images": train_count,
        "val_images": val_count,
        "train_balloons_used": len(train_balloons),
        "val_balloons_used": len(val_balloons),
        "total_backgrounds_available": len(background_files),
        "total_balloon_pairs_available": len(balloon_mask_pairs),
        "corner_ratio": args.corner_ratio
    }
    
    # 統計を表示（実際のディレクトリから確認）
    check_dir = args.final_output_dir if should_copy_to_final else args.output_dir
    for split in ["train", "val"]:
        if should_copy_to_final:
            img_count = len(list(Path(check_dir).glob(f"{split}/images/*.png")))
            mask_count = len(list(Path(check_dir).glob(f"{split}/masks/*.png")))
        else:
            # final_output_dirにコピーしない場合は、output_dir配下に直接trainとvalがある
            img_count = len(list(Path(check_dir).glob(f"{split}/*.png")))
            mask_count = len(list(Path(args.mask_output_dir).glob(f"{split}/*.png")))
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
