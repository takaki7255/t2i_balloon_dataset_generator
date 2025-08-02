#!/usr/bin/env python3
"""
コマ角・辺形状変更データ拡張の詳細分析スクリプト
生成されたデータセットでの拡張適用状況を分析
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import defaultdict, Counter

def analyze_panel_augmentation_dataset(dataset_path: str):
    """
    パネル形状変更が適用されたデータセットを分析
    """
    print(f"=== パネル形状変更データセット分析 ===")
    print(f"対象: {dataset_path}")
    
    # 設定ファイル読み込み
    config_path = os.path.join(dataset_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"\n📊 設定情報:")
        aug_probs = config.get("augmentation_info", {}).get("augmentation_probabilities", {})
        for aug_type, prob in aug_probs.items():
            print(f"  {aug_type}: {prob}")
    
    # 画像分析
    results = {
        "normal_images": [],
        "augmented_images": [],
        "shape_analysis": defaultdict(list)
    }
    
    for split in ["train", "val"]:
        images_dir = os.path.join(dataset_path, split, "images")
        masks_dir = os.path.join(dataset_path, split, "masks")
        
        if not os.path.exists(images_dir):
            continue
            
        for img_file in os.listdir(images_dir):
            if not img_file.endswith('.png'):
                continue
                
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file.replace('.png', '_mask.png'))
            
            if not os.path.exists(mask_path):
                continue
            
            # 画像とマスク読み込み
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                continue
            
            # 形状分析
            shape_features = analyze_balloon_shapes(mask)
            
            if "_normal" in img_file:
                results["normal_images"].append({
                    "file": img_file,
                    "split": split,
                    "features": shape_features
                })
            elif "_aug" in img_file:
                results["augmented_images"].append({
                    "file": img_file,
                    "split": split,
                    "features": shape_features
                })
    
    # 統計分析
    print(f"\n📈 データセット統計:")
    print(f"通常画像: {len(results['normal_images'])}枚")
    print(f"拡張画像: {len(results['augmented_images'])}枚")
    
    # 形状特徴比較
    compare_shape_features(results, dataset_path)
    
    return results

def analyze_balloon_shapes(mask: np.ndarray) -> dict:
    """
    吹き出しマスクの形状特徴を分析
    """
    if mask.sum() == 0:
        return {"error": "empty_mask"}
    
    # 輪郭検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"error": "no_contours"}
    
    features = {}
    
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 100:  # 小さすぎる輪郭は無視
            continue
            
        # 基本形状特徴
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 境界ボックス
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0
        
        # 凸包との比較（形状の複雑さ）
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 円形度（コマ角切り取りで変化）
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 角の検出（approxPolyDP使用）
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        corner_count = len(approx)
        
        # 境界の直線性検証（コマ辺直線化検出）
        edge_linearity = analyze_edge_linearity(contour, x, y, w, h)
        
        features[f"balloon_{i}"] = {
            "area": area,
            "perimeter": perimeter,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "circularity": circularity,
            "corner_count": corner_count,
            "edge_linearity": edge_linearity,
            "bbox_area_ratio": area / (w * h) if w * h > 0 else 0
        }
    
    return features

def analyze_edge_linearity(contour, x, y, w, h) -> dict:
    """
    輪郭の辺が直線的かどうかを分析（コマ辺直線化の検出）
    """
    linearity = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    
    # 各辺の直線性を計算
    for point in contour:
        px, py = point[0]
        
        # 相対位置計算
        rel_x = (px - x) / w if w > 0 else 0
        rel_y = (py - y) / h if h > 0 else 0
        
        # 境界近くの点の直線性をチェック
        if rel_y < 0.1:  # 上辺
            linearity["top"] += 1
        elif rel_y > 0.9:  # 下辺
            linearity["bottom"] += 1
        elif rel_x < 0.1:  # 左辺
            linearity["left"] += 1
        elif rel_x > 0.9:  # 右辺
            linearity["right"] += 1
    
    return linearity

def compare_shape_features(results: dict, output_dir: str):
    """
    通常画像と拡張画像の形状特徴を比較
    """
    normal_features = []
    aug_features = []
    
    # 特徴量収集
    for item in results["normal_images"]:
        for balloon_key, features in item["features"].items():
            if balloon_key.startswith("balloon_"):
                normal_features.append(features)
    
    for item in results["augmented_images"]:
        for balloon_key, features in item["features"].items():
            if balloon_key.startswith("balloon_"):
                aug_features.append(features)
    
    if not normal_features or not aug_features:
        print("特徴量が不足しているため比較をスキップします")
        return
    
    # 比較グラフ作成
    create_comparison_plots(normal_features, aug_features, output_dir)
    
    # 統計比較
    print(f"\n📊 形状特徴比較:")
    feature_names = ["circularity", "solidity", "aspect_ratio", "corner_count"]
    
    for feature in feature_names:
        normal_vals = [f[feature] for f in normal_features if feature in f]
        aug_vals = [f[feature] for f in aug_features if feature in f]
        
        if normal_vals and aug_vals:
            normal_mean = np.mean(normal_vals)
            aug_mean = np.mean(aug_vals)
            change_pct = ((aug_mean - normal_mean) / normal_mean * 100) if normal_mean != 0 else 0
            
            print(f"  {feature}:")
            print(f"    通常: {normal_mean:.3f} → 拡張: {aug_mean:.3f} ({change_pct:+.1f}%)")

def create_comparison_plots(normal_features: list, aug_features: list, output_dir: str):
    """
    比較グラフを作成
    """
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('コマ角・辺形状変更データ拡張の効果分析', fontsize=16)
    
    feature_configs = [
        ("circularity", "円形度", "形状の規則性"),
        ("solidity", "充実度", "凸性の程度"), 
        ("aspect_ratio", "アスペクト比", "縦横比"),
        ("corner_count", "角の数", "形状の複雑さ"),
        ("bbox_area_ratio", "境界ボックス面積比", "形状効率"),
    ]
    
    for i, (feature, title, desc) in enumerate(feature_configs):
        if i >= 6:  # グラフ数制限
            break
            
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        normal_vals = [f[feature] for f in normal_features if feature in f]
        aug_vals = [f[feature] for f in aug_features if feature in f]
        
        if normal_vals and aug_vals:
            ax.hist(normal_vals, alpha=0.7, label='通常画像', bins=20, color='blue')
            ax.hist(aug_vals, alpha=0.7, label='拡張画像', bins=20, color='red')
            ax.set_title(f'{title}\n({desc})')
            ax.set_xlabel('値')
            ax.set_ylabel('頻度')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 余ったサブプロットを非表示
    for i in range(len(feature_configs), 6):
        row, col = i // 3, i % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'panel_augmentation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 分析グラフを保存: {output_dir}/panel_augmentation_analysis.png")

if __name__ == "__main__":
    # テストデータセットを分析
    dataset_path = "../panel_corner_test_dataset"
    if os.path.exists(dataset_path):
        results = analyze_panel_augmentation_dataset(dataset_path)
    else:
        print(f"データセットが見つかりません: {dataset_path}")
        print("先に以下のコマンドでデータセットを生成してください:")
        print("python create_mixed_dataset.py --normal 20 --augmented 20 --output panel_corner_test_dataset")
