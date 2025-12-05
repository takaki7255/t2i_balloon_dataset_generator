"""
オノマトペをコマ内に配置する合成データセット作成スクリプト（データ拡張版）

create_syn_onomatopoeia_dataset.pyのデータ拡張機能を統合し、
オノマトペをコマの内側にランダムに配置する

漫画オノマトペに有効なデータ拡張:
- 回転: ±30度（漫画特有の斜め配置に対応）
- スケール変動: 0.8～1.2倍（サイズのバリエーション）
- アスペクト比変更: 0.9～1.1倍（縦横の伸縮）
- せん断変換: ±15度（斜体効果）
- 透明度変化: 0.7～1.0（薄いオノマトペに対応）
- ガウシアンブラー: カーネル(1,3)（動きのブレ表現）
- ランダム消去: 5～15%（部分的な欠損に対応）
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


def apply_augmentation(image: np.ndarray, mask: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    漫画オノマトペに特化したデータ拡張を適用
    
    Args:
        image: オノマトペ画像
        mask: マスク画像
        cfg: 設定辞書
    
    Returns:
        (augmented_image, augmented_mask): 拡張後の画像とマスク
    """
    aug_cfg = cfg.get("AUGMENTATION", {})
    
    if not aug_cfg.get("ENABLED", False):
        return image, mask
    
    h, w = image.shape[:2]
    
    # 1. 回転（±30度）- 漫画では斜めに配置されることが多い
    if aug_cfg.get("ROTATION", True) and random.random() < aug_cfg.get("ROTATION_PROB", 0.7):
        angle = random.uniform(-30, 30)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 2. スケール変動（0.8～1.2倍）
    if aug_cfg.get("SCALE", True) and random.random() < aug_cfg.get("SCALE_PROB", 0.5):
        scale = random.uniform(0.8, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 元のサイズに合わせるため、パディングまたはクロップ
        if new_w > w or new_h > h:
            # クロップ
            start_x = max(0, (new_w - w) // 2)
            start_y = max(0, (new_h - h) // 2)
            image = image[start_y:start_y+h, start_x:start_x+w]
            mask = mask[start_y:start_y+h, start_x:start_x+w]
        else:
            # パディング
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            image = cv2.copyMakeBorder(image, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, 
                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))
            mask = cv2.copyMakeBorder(mask, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x,
                                     cv2.BORDER_CONSTANT, value=0)
    
    # 3. アスペクト比変更（0.9～1.1倍）- 縦横の伸縮
    if aug_cfg.get("ASPECT_RATIO", True) and random.random() < aug_cfg.get("ASPECT_PROB", 0.3):
        aspect_x = random.uniform(0.9, 1.1)
        aspect_y = random.uniform(0.9, 1.1)
        new_w, new_h = int(w * aspect_x), int(h * aspect_y)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 4. せん断変換（±15度）- 斜体効果
    if aug_cfg.get("SHEAR", True) and random.random() < aug_cfg.get("SHEAR_PROB", 0.4):
        shear = random.uniform(-0.3, 0.3)  # tan(15度) ≈ 0.27
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 5. 透明度変化（0.7～1.0）- 薄いオノマトペに対応
    if aug_cfg.get("ALPHA", True) and random.random() < aug_cfg.get("ALPHA_PROB", 0.5):
        alpha = random.uniform(0.7, 1.0)
        # マスクは変更せず、画像のみ透明度を変更（白背景に近づける）
        white_bg = np.ones_like(image) * 255
        image = (image * alpha + white_bg * (1 - alpha)).astype(np.uint8)
    
    # 6. ガウシアンブラー（動きのブレ表現）
    if aug_cfg.get("BLUR", True) and random.random() < aug_cfg.get("BLUR_PROB", 0.3):
        kernel_size = random.choice([1, 3])
        if kernel_size > 1:
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # 7. ランダム消去（5～15%）- 部分的な欠損
    if aug_cfg.get("RANDOM_ERASING", True) and random.random() < aug_cfg.get("ERASING_PROB", 0.3):
        erase_ratio = random.uniform(0.05, 0.15)
        erase_h = int(h * np.sqrt(erase_ratio))
        erase_w = int(w * np.sqrt(erase_ratio))
        
        if erase_h > 0 and erase_w > 0:
            top = random.randint(0, max(0, h - erase_h))
            left = random.randint(0, max(0, w - erase_w))
            image[top:top+erase_h, left:left+erase_w] = 255  # 白で消去
            mask[top:top+erase_h, left:left+erase_w] = 0     # マスクも消去
    
    return image, mask


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
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        # バウンディングボックス
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        
        # 面積比チェック（矩形に近い形状のみ）
        if area / bbox_area < area_ratio_threshold:
            continue
        
        # パネルマスクを作成
        panel_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(panel_mask, [contour], -1, 255, -1)
        
        panels.append((panel_mask, (x, y, w, h)))
    
    return panels


def sample_scale(page_w: int, page_h: int, cfg: dict) -> float:
    """背景画像（ページ全体）サイズに応じたスケールをサンプリング"""
    # 背景画像面積に対する比率として計算
    page_area = page_w * page_h
    scale_range = cfg.get("SCALE_RANGE", (0.005, 0.03))
    return random.uniform(*scale_range)


def calculate_onomatopoeia_size(crop_w: int, crop_h: int, page_w: int, page_h: int,
                                target_scale: float, panel_w: int = None, panel_h: int = None,
                                mask: np.ndarray = None) -> tuple:
    """オノマトペのリサイズサイズを計算
    
    Args:
        crop_w, crop_h: クロップ後のオノマトペサイズ
        page_w, page_h: 背景画像（ページ全体）のサイズ
        target_scale: 背景画像面積に対する目標スケール
        panel_w, panel_h: パネルサイズ（配置先の制限用、Noneならページサイズを使用）
        mask: マスク画像
    
    Returns:
        (new_w, new_h): リサイズ後のサイズ
    """
    # 背景画像面積を基準にサイズを計算
    page_area = page_w * page_h
    target_area = page_area * target_scale
    
    # マスク領域を考慮
    if mask is not None:
        mask_pixels = np.count_nonzero(mask)
        crop_pixels = crop_w * crop_h
        mask_ratio = mask_pixels / crop_pixels if crop_pixels > 0 else 1.0
    else:
        mask_ratio = 1.0
    
    aspect_ratio = crop_h / crop_w
    
    # アスペクト比を維持した理想サイズ
    ideal_w = int(np.sqrt(target_area / aspect_ratio))
    ideal_h = int(np.sqrt(target_area * aspect_ratio))
    
    # パネルサイズの制限（パネルが指定されていればその50%以内、なければページの30%以内）
    limit_w = panel_w if panel_w else page_w
    limit_h = panel_h if panel_h else page_h
    max_w = int(limit_w * 0.5)
    max_h = int(limit_h * 0.5)
    
    scale_by_w = max_w / ideal_w if ideal_w > max_w else 1.0
    scale_by_h = max_h / ideal_h if ideal_h > max_h else 1.0
    adjust_scale = min(scale_by_w, scale_by_h)
    
    new_w = int(ideal_w * adjust_scale)
    new_h = int(ideal_h * adjust_scale)
    
    # 最小サイズ確保
    new_w = max(new_w, 20)
    new_h = max(new_h, 20)
    
    return new_w, new_h


def sample_num_onomatopoeia_for_page(cfg: dict, max_available: int) -> int:
    """ページ全体に配置するオノマトペ個数を統計からサンプリング
    
    統計データ:
        平均: 7.2個, 中央値: 6個, 標準偏差: 6.1
        25パーセンタイル: 3個, 75パーセンタイル: 10個
        最小: 1個, 最大: 82個
    
    Args:
        cfg: 設定辞書
        max_available: 利用可能なオノマトペの最大数
    
    Returns:
        配置するオノマトペの個数
    """
    # 統計に基づく正規分布でサンプリング
    mean = cfg.get("NUM_ONOMATOPOEIA_MEAN", 7.0)
    std = cfg.get("NUM_ONOMATOPOEIA_STD", 6.0)
    min_count = cfg.get("NUM_ONOMATOPOEIA_MIN", 1)
    max_count = cfg.get("NUM_ONOMATOPOEIA_MAX", 20)
    
    # 正規分布からサンプリングして整数化
    n = int(round(random.gauss(mean, std)))
    
    # 範囲内にクリップ
    n = max(min_count, min(max_count, n))
    n = min(max_available, n)
    
    if n <= 0:
        n = 1
    return n


def place_single_onomatopoeia(result_img: np.ndarray, result_mask: np.ndarray,
                              onomatopoeia_path: str, mask_path: str,
                              panel_mask: np.ndarray, panel_bbox: tuple,
                              occupied_regions: list, cfg: dict,
                              page_size: tuple = None) -> Optional[dict]:
    """1つのオノマトペをパネル内に配置
    
    Args:
        result_img: 結果画像（in-place更新）
        result_mask: 結果マスク（in-place更新）
        onomatopoeia_path: オノマトペ画像パス
        mask_path: マスク画像パス
        panel_mask: パネルマスク
        panel_bbox: パネルのバウンディングボックス (x, y, w, h)
        occupied_regions: 既に配置済みの領域リスト（in-place更新）
        cfg: 設定辞書
        page_size: 背景画像（ページ全体）のサイズ (page_w, page_h)
    
    Returns:
        配置成功時はオノマトペ情報の辞書、失敗時はNone
    """
    x, y, w, h = panel_bbox
    
    # 背景画像サイズ（指定がなければresult_imgのサイズを使用）
    if page_size is None:
        page_h, page_w = result_img.shape[:2]
    else:
        page_w, page_h = page_size
    
    # オノマトペとマスク読み込み
    onomatopoeia = cv2.imread(onomatopoeia_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if onomatopoeia is None or mask is None:
        return None
    
    # データ拡張を適用
    onomatopoeia, mask = apply_augmentation(onomatopoeia, mask, cfg)
    
    # マスクの境界ボックスでクロップ
    cropped_onomatopoeia, cropped_mask, bbox = crop_onomatopoeia_and_mask(onomatopoeia, mask)
    
    if cropped_onomatopoeia.size == 0 or cropped_mask.size == 0:
        return None
    
    crop_h, crop_w = cropped_onomatopoeia.shape[:2]
    
    # スケールサンプリング（背景画像サイズを基準）
    onomatopoeia_scale = sample_scale(page_w, page_h, cfg)
    
    # サイズ計算（背景画像サイズを基準、パネルサイズで制限）
    new_w, new_h = calculate_onomatopoeia_size(
        crop_w, crop_h, page_w, page_h, onomatopoeia_scale,
        panel_w=w, panel_h=h,
        mask=cropped_mask
    )
    
    # パネルサイズを超える場合はスキップ
    if new_w >= w or new_h >= h:
        return None
    
    # リサイズ
    onomatopoeia_resized = cv2.resize(cropped_onomatopoeia, (new_w, new_h))
    mask_resized = cv2.resize(cropped_mask, (new_w, new_h))
    
    # パネル内での配置位置を探す
    best_position = None
    min_overlap_ratio = float('inf')
    
    max_attempts = cfg.get("MAX_ATTEMPTS", 100)
    for attempt in range(max_attempts):
        # パネル内のランダム位置
        if w - new_w <= 0 or h - new_h <= 0:
            break
        
        local_x = random.randint(0, w - new_w)
        local_y = random.randint(0, h - new_h)
        
        # グローバル座標
        global_x = x + local_x
        global_y = y + local_y
        
        # パネルマスク内かチェック
        center_x = global_x + new_w // 2
        center_y = global_y + new_h // 2
        
        if center_y >= panel_mask.shape[0] or center_x >= panel_mask.shape[1]:
            continue
        
        if panel_mask[center_y, center_x] == 0:
            continue
        
        # 新しい領域
        new_region = (global_x, global_y, global_x + new_w, global_y + new_h)
        
        # 重複チェック
        max_overlap_ratio = 0
        for occupied in occupied_regions:
            if regions_overlap(new_region, occupied):
                overlap_area = calculate_overlap_area(new_region, occupied)
                new_area = new_w * new_h
                overlap_ratio = overlap_area / new_area
                max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)
        
        # 重複が少ない場合は配置
        if max_overlap_ratio <= 0.15:
            best_position = (global_x, global_y)
            break
        
        if max_overlap_ratio < min_overlap_ratio:
            min_overlap_ratio = max_overlap_ratio
            best_position = (global_x, global_y)
    
    # 配置実行
    if best_position is None:
        return None
    
    global_x, global_y = best_position
    
    # 合成
    mask_norm = mask_resized.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
    
    # 背景画像の該当領域
    bg_region = result_img[global_y:global_y+new_h, global_x:global_x+new_w]
    
    # アルファブレンディング
    blended = onomatopoeia_resized.astype(np.float32) * mask_3ch + bg_region.astype(np.float32) * (1 - mask_3ch)
    result_img[global_y:global_y+new_h, global_x:global_x+new_w] = blended.astype(np.uint8)
    
    # マスク合成
    result_mask[global_y:global_y+new_h, global_x:global_x+new_w] = np.maximum(
        result_mask[global_y:global_y+new_h, global_x:global_x+new_w], mask_resized)
    
    # 配置済み領域に追加
    new_region = (global_x, global_y, global_x + new_w, global_y + new_h)
    occupied_regions.append(new_region)
    
    return {
        "onomatopoeia_file": Path(onomatopoeia_path).name,
        "final_size": f"{new_w}x{new_h}",
        "position": f"({global_x},{global_y})",
        "scale": f"{onomatopoeia_scale:.6f}",
    }


def composite_onomatopoeia_on_page(page_img: np.ndarray, panels: list,
                                   onomatopoeia_pairs: list, cfg: dict) -> tuple:
    """ページ全体にオノマトペを配置
    
    1. 最初にページ全体に配置するオノマトペ個数を統計からサンプリング
    2. パネル検出時：検出されたパネルに分散して配置
    3. パネル未検出時：ページ全体にランダム配置
    
    Args:
        page_img: ページ画像
        panels: 検出されたパネルのリスト [(panel_mask, bbox), ...]
        onomatopoeia_pairs: オノマトペとマスクのペアのリスト
        cfg: 設定辞書
    
    Returns:
        (result_img, result_mask, onomatopoeia_details, placement_info)
    """
    h, w = page_img.shape[:2]
    page_size = (w, h)  # 背景画像サイズ
    result_img = page_img.copy()
    result_mask = np.zeros((h, w), dtype=np.uint8)
    
    # ページ全体に配置するオノマトペ個数をサンプリング
    target_count = sample_num_onomatopoeia_for_page(cfg, len(onomatopoeia_pairs))
    
    # オノマトペをシャッフルして選択
    selected_pairs = random.sample(onomatopoeia_pairs, min(target_count * 2, len(onomatopoeia_pairs)))
    
    occupied_regions = []
    onomatopoeia_details = []
    placed_count = 0
    pair_idx = 0
    
    # ページ全体用のマスクとbbox（ランダム配置用）
    full_mask = np.ones((h, w), dtype=np.uint8) * 255
    full_bbox = (0, 0, w, h)
    
    if len(panels) == 0:
        # パネル未検出：ページ全体にランダム配置
        for onomatopoeia_path, mask_path in selected_pairs:
            if placed_count >= target_count:
                break
            
            detail = place_single_onomatopoeia(
                result_img, result_mask,
                onomatopoeia_path, mask_path,
                full_mask, full_bbox,
                occupied_regions, cfg,
                page_size=page_size
            )
            
            if detail is not None:
                detail["panel"] = "全体"
                onomatopoeia_details.append(detail)
                placed_count += 1
        
        placement_info = {
            "mode": "full_page",
            "panels_detected": 0,
            "target_count": target_count,
            "placed_count": placed_count
        }
    else:
        # パネル検出：各パネルに1〜2個ずつ配置、残りはランダム配置
        max_per_panel = cfg.get("MAX_ONOMATOPOEIA_PER_PANEL", 2)
        panel_placement_count = {}  # 各パネルへの配置数を記録
        
        # Phase 1: 各パネルに最大max_per_panel個まで配置
        for panel_idx, (panel_mask, bbox) in enumerate(panels):
            if placed_count >= target_count:
                break
            if pair_idx >= len(selected_pairs):
                break
            
            panel_placement_count[panel_idx] = 0
            
            # このパネルに最大max_per_panel個配置を試みる
            attempts_for_panel = 0
            while (panel_placement_count[panel_idx] < max_per_panel 
                   and placed_count < target_count 
                   and pair_idx < len(selected_pairs)
                   and attempts_for_panel < max_per_panel * 2):
                
                onomatopoeia_path, mask_path = selected_pairs[pair_idx]
                pair_idx += 1
                attempts_for_panel += 1
                
                detail = place_single_onomatopoeia(
                    result_img, result_mask,
                    onomatopoeia_path, mask_path,
                    panel_mask, bbox,
                    occupied_regions, cfg,
                    page_size=page_size
                )
                
                if detail is not None:
                    detail["panel"] = f"panel_{panel_idx}"
                    onomatopoeia_details.append(detail)
                    placed_count += 1
                    panel_placement_count[panel_idx] += 1
        
        # Phase 2: 目標数に達していなければ、ページ全体にランダム配置
        while placed_count < target_count and pair_idx < len(selected_pairs):
            onomatopoeia_path, mask_path = selected_pairs[pair_idx]
            pair_idx += 1
            
            detail = place_single_onomatopoeia(
                result_img, result_mask,
                onomatopoeia_path, mask_path,
                full_mask, full_bbox,
                occupied_regions, cfg,
                page_size=page_size
            )
            
            if detail is not None:
                detail["panel"] = "random"
                onomatopoeia_details.append(detail)
                placed_count += 1
        
        placement_info = {
            "mode": "panels_and_random",
            "panels_detected": len(panels),
            "target_count": target_count,
            "placed_count": placed_count,
            "placed_in_panels": sum(panel_placement_count.values()),
            "placed_random": placed_count - sum(panel_placement_count.values())
        }
    
    return result_img, result_mask, onomatopoeia_details, placement_info

def split_onomatopoeia(onomatopoeia_mask_pairs: list, train_ratio: float = 0.8, seed: int = 42) -> tuple:
    """オノマトペをtrain用とval用に分割"""
    random.seed(seed)
    shuffled_pairs = onomatopoeia_mask_pairs.copy()
    random.shuffle(shuffled_pairs)
    
    n = len(shuffled_pairs)
    train_count = int(n * train_ratio)
    
    train_pairs = shuffled_pairs[:train_count]
    val_pairs = shuffled_pairs[train_count:]
    
    return train_pairs, val_pairs


def generate_dataset_split(page_files: list, onomatopoeia_pairs: list,
                          output_dir: str, mask_output_dir: str, split_name: str,
                          target_count: int, cfg: dict, final_output_dir: str = None) -> int:
    """指定されたsplit（trainまたはval）のデータセットを生成
    
    1ページから1枚の画像を生成し、統計に応じたオノマトペ個数を配置する
    """
    print(f"\n=== {split_name} データセット生成開始 ===")
    print(f"目標画像数: {target_count}")
    print(f"ページ画像数: {len(page_files)}")
    print(f"利用可能オノマトペ数: {len(onomatopoeia_pairs)}")
    
    if final_output_dir:
        log_file_path = os.path.join(final_output_dir, f"{split_name}_composition_log.txt")
    else:
        log_file_path = os.path.join(output_dir, f"{split_name}_composition_log.txt")
    
    current_number = 1
    success_count = 0
    page_idx = 0
    
    # ログファイル初期化
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== {split_name.upper()} パネル内オノマトペデータセット合成ログ ===\n")
        log_file.write(f"生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"目標画像数: {target_count}\n")
        log_file.write(f"ページ画像数: {len(page_files)}\n")
        log_file.write(f"利用可能オノマトペ数: {len(onomatopoeia_pairs)}\n")
        log_file.write("=" * 80 + "\n\n")
    
    # プログレスバー
    pbar = tqdm(total=target_count, desc=f"{split_name} 生成中", unit="img")
    
    while success_count < target_count:
        page_path = page_files[page_idx % len(page_files)]
        page_name = Path(page_path).stem
        
        try:
            # ページ画像読み込み
            page_img = cv2.imread(page_path, cv2.IMREAD_COLOR)
            if page_img is None:
                raise FileNotFoundError(f"ページ画像読み込み失敗: {page_path}")
            
            h, w = page_img.shape[:2]
            
            # パネル検出
            panels = detect_panels_simple(page_img, 
                                        area_ratio_threshold=cfg.get("PANEL_AREA_RATIO_THRESHOLD", 0.85),
                                        min_area=cfg.get("PANEL_MIN_AREA", 10000))
            
            # ページ全体にオノマトペを配置（1ページ=1画像）
            result_img, result_mask, onomatopoeia_details, placement_info = composite_onomatopoeia_on_page(
                page_img, panels, onomatopoeia_pairs, cfg
            )
            
            if len(onomatopoeia_details) > 0:
                # ファイル保存
                output_img_path = os.path.join(output_dir, f"{current_number:03d}.png")
                output_mask_path = os.path.join(mask_output_dir, f"{current_number:03d}_mask.png")
                
                cv2.imwrite(output_img_path, result_img)
                cv2.imwrite(output_mask_path, result_mask)
                
                # ログ記録
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"画像 {current_number:03d}.png:\n")
                    log_file.write(f"  ページファイル: {Path(page_path).name}\n")
                    log_file.write(f"  画像サイズ: {w}x{h}\n")
                    log_file.write(f"  配置モード: {placement_info['mode']}\n")
                    log_file.write(f"  検出パネル数: {placement_info['panels_detected']}\n")
                    log_file.write(f"  目標オノマトペ数: {placement_info['target_count']}\n")
                    log_file.write(f"  配置オノマトペ数: {placement_info['placed_count']}\n")
                    
                    for i, detail in enumerate(onomatopoeia_details, 1):
                        log_file.write(f"    オノマトペ{i}: {detail['onomatopoeia_file']}\n")
                        log_file.write(f"      最終サイズ: {detail['final_size']}\n")
                        log_file.write(f"      配置位置: {detail['position']}\n")
                        log_file.write(f"      パネル: {detail.get('panel', 'N/A')}\n")
                        log_file.write(f"      スケール: {detail['scale']}\n")
                    
                    log_file.write("\n")
                
                success_count += 1
                current_number += 1
                pbar.update(1)
            else:
                # 配置失敗
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"⚠️ オノマトペ配置失敗: {page_name}\n\n")
            
            # 次のページへ
            page_idx += 1
        
        except Exception as e:
            # エラーはログのみに記録
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"❌ 処理失敗: {page_name} - {str(e)}\n\n")
            # エラー時も次のページへ
            page_idx += 1
    
    pbar.close()
    print(f"✅ {split_name} 完了: {success_count}個の画像を生成")
    print(f"📄 詳細ログ: {log_file_path}")
    return success_count


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="パネル内オノマトペ合成データセット作成（データ拡張版）")
    parser.add_argument("--onomatopoeia-dir", default="onomatopeias", help="オノマトペ画像ディレクトリ")
    parser.add_argument("--mask-dir", default="onomatopeia_masks", help="マスク画像ディレクトリ")
    parser.add_argument("--page-dir", default="generated_double_backs_1536x1024", help="ページ画像ディレクトリ")
    parser.add_argument("--output-dir", default="onomatopoeia_dataset", help="基本出力ディレクトリ")
    parser.add_argument("--dataset-name", type=str, default="1000-panel-aug", help="データセット名")
    parser.add_argument("--target-images", type=int, default=1000, help="生成する画像数")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="train用の比率")
    parser.add_argument("--no-augmentation", action="store_true", help="データ拡張を無効化")
    
    args = parser.parse_args()
    
    # 設定
    CFG = {
        # 背景画像面積に対するオノマトペ面積の比率 (0.5%〜3%)
        # 例: 1536x1024 = 1,572,864 ピクセルの場合
        #   0.005 → 約7,864 px² → 約 89x89
        #   0.03  → 約47,186 px² → 約 217x217
        "SCALE_RANGE": (0.005, 0.03),
        
        # オノマトペ個数（統計: 平均7.2, 中央値6, 標準偏差6.1, 25%ile=3, 75%ile=10）
        "NUM_ONOMATOPOEIA_MEAN": 7.0,   # 平均
        "NUM_ONOMATOPOEIA_STD": 6.0,    # 標準偏差
        "NUM_ONOMATOPOEIA_MIN": 1,      # 最小
        "NUM_ONOMATOPOEIA_MAX": 20,     # 最大（現実的な上限）
        
        "MAX_ATTEMPTS": 100,
        "MAX_ONOMATOPOEIA_PER_PANEL": 2,  # 1パネルあたりの最大オノマトペ数
        "TRAIN_RATIO": args.train_ratio,
        "ONOMATOPOEIA_SPLIT_SEED": 42,
        "PANEL_AREA_RATIO_THRESHOLD": 0.85,
        "PANEL_MIN_AREA": 10000,
        
        # データ拡張設定
        "AUGMENTATION": {
            "ENABLED": not args.no_augmentation,
            "ROTATION": True,
            "ROTATION_PROB": 0.7,
            "SCALE": True,
            "SCALE_PROB": 0.5,
            "ASPECT_RATIO": True,
            "ASPECT_PROB": 0.3,
            "SHEAR": True,
            "SHEAR_PROB": 0.4,
            "ALPHA": True,
            "ALPHA_PROB": 0.5,
            "BLUR": True,
            "BLUR_PROB": 0.3,
            "RANDOM_ERASING": False,  # 無効化（オノマトペに矩形消去は不自然）
            "ERASING_PROB": 0.0,
        }
    }
    
    print("\n=== パネル内オノマトペデータセット 作成開始（データ拡張版） ===")
    print(f"データ拡張: {'有効' if CFG['AUGMENTATION']['ENABLED'] else '無効'}")
    
    # 出力ディレクトリ作成
    base_output_dir = args.output_dir
    dataset_name = args.dataset_name
    final_output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 設定を保存
    config_output = {
        "timestamp": datetime.now().isoformat(),
        "script_name": "create_onomatopoeia_panel_dataset_augmented.py",
        "dataset_name": dataset_name,
        "base_output_path": base_output_dir,
        "dataset_output_path": final_output_dir,
        "target_images": args.target_images,
        "config": CFG,
        "input_directories": {
            "onomatopoeia_dir": args.onomatopoeia_dir,
            "masks_dir": args.mask_dir,
            "pages_dir": args.page_dir
        }
    }
    
    config_file_path = os.path.join(base_output_dir, f"{dataset_name}_config.json")
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2)
    print(f"設定情報を保存: {config_file_path}")
    print(f"📁 出力ディレクトリ: {final_output_dir}")
    
    # オノマトペとマスクの対応を取得
    onomatopoeia_mask_pairs = []
    print("\nオノマトペ・マスクペアを検索中...")
    for onomatopoeia_file in os.listdir(args.onomatopoeia_dir):
        if onomatopoeia_file.endswith(('.png', '.jpg', '.jpeg')):
            onomatopoeia_path = os.path.join(args.onomatopoeia_dir, onomatopoeia_file)
            onomatopoeia_stem = Path(onomatopoeia_file).stem
            
            mask_file = f"{onomatopoeia_stem}_mask.png"
            mask_path = os.path.join(args.mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                onomatopoeia_mask_pairs.append((onomatopoeia_path, mask_path))
    
    # ページ画像を取得
    page_files = []
    for page_file in os.listdir(args.page_dir):
        if page_file.endswith(('.png', '.jpg', '.jpeg')):
            page_files.append(os.path.join(args.page_dir, page_file))
    
    print(f"見つかったオノマトペ: {len(onomatopoeia_mask_pairs)}個")
    print(f"見つかったページ: {len(page_files)}個")
    
    # オノマトペが0個の場合はエラー終了
    if len(onomatopoeia_mask_pairs) == 0:
        print("\n❌ エラー: オノマトペ・マスクペアが見つかりませんでした")
        print(f"   オノマトペディレクトリ: {args.onomatopoeia_dir}")
        print(f"   マスクディレクトリ: {args.mask_dir}")
        print("\n確認事項:")
        print("  1. ディレクトリが存在するか")
        print("  2. オノマトペ画像（.png, .jpg, .jpeg）が存在するか")
        print("  3. 対応するマスク（<name>_mask.png）が存在するか")
        return
    
    # ページが0個の場合もエラー終了
    if len(page_files) == 0:
        print("\n❌ エラー: ページ画像が見つかりませんでした")
        print(f"   ページディレクトリ: {args.page_dir}")
        return
    
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
        page_files, train_onomatopoeia,
        train_img_dir, train_mask_dir, "train", train_target,
        CFG, final_output_dir
    )
    
    # val データセット生成
    val_img_dir = os.path.join(final_output_dir, "val", "images")
    val_mask_dir = os.path.join(final_output_dir, "val", "masks")
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    
    val_count = generate_dataset_split(
        page_files, val_onomatopoeia,
        val_img_dir, val_mask_dir, "val", val_target,
        CFG, final_output_dir
    )
    
    # 最終レポート
    print(f"\n=== パネル内オノマトペデータセット 作成完了 ===")
    print(f"出力先: {final_output_dir}")
    print(f"総生成画像数: {train_count + val_count}枚")
    
    # 統計情報
    stats = {
        "total_images": train_count + val_count,
        "train_images": train_count,
        "val_images": val_count,
        "train_onomatopoeia_used": len(train_onomatopoeia),
        "val_onomatopoeia_used": len(val_onomatopoeia),
        "total_pages_available": len(page_files),
        "total_onomatopoeia_pairs_available": len(onomatopoeia_mask_pairs),
        "augmentation_enabled": CFG['AUGMENTATION']['ENABLED']
    }
    
    for split in ["train", "val"]:
        img_count = len(list(Path(final_output_dir).glob(f"{split}/images/*.png")))
        mask_count = len(list(Path(final_output_dir).glob(f"{split}/masks/*.png")))
        print(f"{split}: {img_count} 画像, {mask_count} マスク")
    
    # 統計情報を保存
    config_output["statistics"] = stats
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2)
    
    print(f"\n設定・統計情報: {config_file_path}")


if __name__ == "__main__":
    main()
