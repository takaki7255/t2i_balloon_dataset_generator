#!/usr/bin/env python3
"""
コマ角・辺形状変更機能の拡張版
より複雑で現実的なマンガコマ効果を実装
"""

import cv2
import numpy as np
import random
from typing import Tuple

def advanced_panel_augmentation(balloon: np.ndarray, mask: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    高度なコマ角・辺形状変更
    
    新機能:
    1. 斜め切り取り（コマの斜め境界）
    2. 複数角同時切り取り
    3. 波状切り取り（不規則なコマ境界）
    4. グラデーション切り取り（ぼかし境界）
    """
    
    defaults = dict(
        # 既存機能
        PANEL_CLIP_PROB=0.3,
        PANEL_CLIP_RATIO=(0.10, 0.40),
        PANEL_EDGE_PROB=0.25,
        PANEL_EDGE_RATIO=(0.20, 0.60),
        
        # 新機能
        DIAGONAL_CUT_PROB=0.2,        # 斜め切り取り
        DIAGONAL_ANGLE_RANGE=(15, 45), # 斜め角度範囲
        MULTI_CORNER_PROB=0.15,       # 複数角同時切り取り
        WAVY_EDGE_PROB=0.1,          # 波状境界
        WAVE_AMPLITUDE_RATIO=(0.02, 0.08), # 波の振幅
        GRADIENT_CUT_PROB=0.1,        # グラデーション切り取り
        GRADIENT_WIDTH_RATIO=(0.05, 0.15), # グラデーション幅
    )
    
    if cfg is None: 
        cfg = {}
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    
    h, w = mask.shape
    
    # ---------- 1. 斜め切り取り ----------
    if random.random() < cfg["DIAGONAL_CUT_PROB"]:
        corner = random.choice(["top-left", "top-right", "bottom-left", "bottom-right"])
        angle = random.uniform(*cfg["DIAGONAL_ANGLE_RANGE"])
        size_ratio = random.uniform(*cfg["PANEL_CLIP_RATIO"])
        
        apply_diagonal_cut(balloon, mask, corner, angle, size_ratio)
    
    # ---------- 2. 複数角同時切り取り ----------
    elif random.random() < cfg["MULTI_CORNER_PROB"]:
        num_corners = random.choice([2, 3])  # 2-3角を同時切り取り
        corners = random.sample(["top-left", "top-right", "bottom-left", "bottom-right"], num_corners)
        
        for corner in corners:
            size_ratio = random.uniform(0.05, 0.20)  # 小さめに切り取り
            apply_corner_cut(balloon, mask, corner, size_ratio)
    
    # ---------- 3. 波状境界切り取り ----------
    elif random.random() < cfg["WAVY_EDGE_PROB"]:
        edge = random.choice(["left", "right", "top", "bottom"])
        amplitude_ratio = random.uniform(*cfg["WAVE_AMPLITUDE_RATIO"])
        frequency = random.uniform(2, 6)  # 波の頻度
        
        apply_wavy_cut(balloon, mask, edge, amplitude_ratio, frequency)
    
    # ---------- 4. グラデーション切り取り ----------
    elif random.random() < cfg["GRADIENT_CUT_PROB"]:
        edge = random.choice(["left", "right", "top", "bottom"])
        cut_ratio = random.uniform(*cfg["PANEL_EDGE_RATIO"])
        gradient_width_ratio = random.uniform(*cfg["GRADIENT_WIDTH_RATIO"])
        
        apply_gradient_cut(balloon, mask, edge, cut_ratio, gradient_width_ratio)
    
    return balloon, mask

def apply_diagonal_cut(balloon: np.ndarray, mask: np.ndarray, corner: str, angle: float, size_ratio: float):
    """斜め切り取りを適用"""
    h, w = mask.shape
    cut_size = int(min(h, w) * size_ratio)
    
    # 斜めライン作成
    if corner == "top-left":
        for y in range(cut_size):
            x_end = int(y / np.tan(np.radians(angle)))
            if x_end < w:
                mask[y, :x_end] = 0
                balloon[y, :x_end] = 255
    elif corner == "top-right":
        for y in range(cut_size):
            x_start = w - int(y / np.tan(np.radians(angle)))
            if x_start > 0:
                mask[y, x_start:] = 0
                balloon[y, x_start:] = 255
    # 他の角も同様に実装...

def apply_corner_cut(balloon: np.ndarray, mask: np.ndarray, corner: str, size_ratio: float):
    """通常の角切り取り（既存機能の簡略版）"""
    h, w = mask.shape
    cut_size = int(min(h, w) * size_ratio)
    
    if corner == "top-left":
        mask[:cut_size, :cut_size] = 0
        balloon[:cut_size, :cut_size] = 255
    elif corner == "top-right":
        mask[:cut_size, w-cut_size:] = 0
        balloon[:cut_size, w-cut_size:] = 255
    # 他の角も同様...

def apply_wavy_cut(balloon: np.ndarray, mask: np.ndarray, edge: str, amplitude_ratio: float, frequency: float):
    """波状境界切り取りを適用"""
    h, w = mask.shape
    amplitude = int(min(h, w) * amplitude_ratio)
    
    if edge == "left":
        for y in range(h):
            wave_offset = int(amplitude * np.sin(2 * np.pi * frequency * y / h))
            cut_width = max(5, 10 + wave_offset)  # 最小5px
            if cut_width < w:
                mask[y, :cut_width] = 0
                balloon[y, :cut_width] = 255
    elif edge == "right":
        for y in range(h):
            wave_offset = int(amplitude * np.sin(2 * np.pi * frequency * y / h))
            cut_width = max(5, 10 + wave_offset)
            if cut_width < w:
                mask[y, w-cut_width:] = 0
                balloon[y, w-cut_width:] = 255
    # 他の辺も同様...

def apply_gradient_cut(balloon: np.ndarray, mask: np.ndarray, edge: str, cut_ratio: float, gradient_width_ratio: float):
    """グラデーション切り取りを適用（ぼかし境界）"""
    h, w = mask.shape
    
    if edge == "left":
        cut_width = int(w * 0.05)  # 基本切り取り幅
        gradient_width = int(w * gradient_width_ratio)
        
        # 段階的に透明度を変える
        for x in range(cut_width + gradient_width):
            if x < cut_width:
                alpha = 0.0  # 完全に切り取り
            else:
                alpha = (x - cut_width) / gradient_width  # グラデーション
            
            for y in range(int(h * (1 - cut_ratio) / 2), int(h * (1 + cut_ratio) / 2)):
                if alpha == 0.0:
                    mask[y, x] = 0
                    balloon[y, x] = 255
                else:
                    # グラデーション効果
                    mask[y, x] = int(mask[y, x] * alpha)
                    balloon[y, x] = balloon[y, x] * alpha + 255 * (1 - alpha)
    # 他の辺も同様...

if __name__ == "__main__":
    print("高度なコマ角・辺形状変更機能の拡張版")
    print("この機能は将来のバージョンで実装予定です")
