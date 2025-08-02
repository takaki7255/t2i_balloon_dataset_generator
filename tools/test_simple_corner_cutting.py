#!/usr/bin/env python3
"""
シンプルなコマ角切り落とし機能のテストスクリプト
新しい実装の効果を分かりやすく確認
"""

import cv2
import numpy as np
import os
import random
import sys
sys.path.append('..')
from create_syn_dataset_aug import apply_panel_corner_squaring, apply_panel_edge_straightening

def test_simple_corner_cutting():
    """シンプルなコマ角切り落としのテスト"""
    print("=== シンプルなコマ角切り落としテスト ===")
    
    # テスト用データのパス
    balloons_dir = "../generated_balloons"
    masks_dir = "../masks"
    output_dir = "simple_corner_test_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 吹き出しファイルを取得
    balloon_files = [f for f in os.listdir(balloons_dir) if f.endswith('.png')]
    
    if not balloon_files:
        print("吹き出し画像が見つかりません")
        return
    
    # テスト設定
    cfg = {
        "PANEL_CORNER_PROB": 1.0,
        "PANEL_CORNER_RATIO": (0.05, 0.15),  # 5-15%切り落とし
        "PANEL_EDGE_PROB": 1.0,
        "PANEL_EDGE_RATIO": (0.30, 0.70),   # 30-70%の範囲で辺を直線化
    }
    
    test_count = min(6, len(balloon_files))
    random.seed(42)
    
    print(f"テスト対象: {test_count}個の吹き出し")
    
    for i in range(test_count):
        balloon_file = balloon_files[i]
        balloon_stem = os.path.splitext(balloon_file)[0]
        mask_file = f"{balloon_stem}_mask.png"
        
        balloon_path = os.path.join(balloons_dir, balloon_file)
        mask_path = os.path.join(masks_dir, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"マスクが見つかりません: {mask_file}")
            continue
        
        # 画像とマスク読み込み
        balloon = cv2.imread(balloon_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if balloon is None or mask is None:
            print(f"読み込み失敗: {balloon_file}")
            continue
        
        print(f"処理中: {balloon_file}")
        
        # 各角での効果をテスト
        corners = ["top-left", "top-right", "bottom-left", "bottom-right"]
        
        results = []
        
        # オリジナル
        results.append((balloon.copy(), mask.copy(), "Original"))
        
        # 各角での切り落とし効果
        for corner in corners:
            test_balloon = balloon.copy()
            test_mask = mask.copy()
            
            # 特定の角を強制的に指定してテスト
            h, w = test_mask.shape
            cut_ratio = 0.10  # 10%固定
            
            if corner == "top-left":
                top_cut = int(h * cut_ratio)
                left_cut = int(w * cut_ratio)
                test_balloon[:top_cut, :] = 255
                test_mask[:top_cut, :] = 0
                test_balloon[:, :left_cut] = 255
                test_mask[:, :left_cut] = 0
                
            elif corner == "top-right":
                top_cut = int(h * cut_ratio)
                right_cut = int(w * cut_ratio)
                test_balloon[:top_cut, :] = 255
                test_mask[:top_cut, :] = 0
                test_balloon[:, w-right_cut:] = 255
                test_mask[:, w-right_cut:] = 0
                
            elif corner == "bottom-left":
                bottom_cut = int(h * cut_ratio)
                left_cut = int(w * cut_ratio)
                test_balloon[h-bottom_cut:, :] = 255
                test_mask[h-bottom_cut:, :] = 0
                test_balloon[:, :left_cut] = 255
                test_mask[:, :left_cut] = 0
                
            else:  # bottom-right
                bottom_cut = int(h * cut_ratio)
                right_cut = int(w * cut_ratio)
                test_balloon[h-bottom_cut:, :] = 255
                test_mask[h-bottom_cut:, :] = 0
                test_balloon[:, w-right_cut:] = 255
                test_mask[:, w-right_cut:] = 0
            
            results.append((test_balloon, test_mask, f"{corner.replace('-', ' ').title()} Cut"))
        
        # 比較画像作成
        comparison = create_corner_comparison_grid(results)
        cv2.imwrite(os.path.join(output_dir, f"{balloon_stem}_corner_comparison.png"), comparison)
    
    print(f"\n✅ テスト完了")
    print(f"結果は {output_dir}/ に保存されました")

def create_corner_comparison_grid(results):
    """コーナー比較グリッド画像を作成"""
    target_h, target_w = 200, 200
    
    # 画像を準備
    processed_images = []
    for balloon, mask, title in results:
        # マスクオーバーレイを作成
        overlay = create_mask_overlay(balloon, mask, title)
        resized = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)
        processed_images.append(resized)
    
    # グリッドレイアウト：上段にオリジナル、下段に4つの角
    if len(processed_images) >= 5:
        # 1行目：オリジナル（中央配置）
        top_row = np.hstack([
            np.ones((target_h, target_w, 3), dtype=np.uint8) * 255,  # 空白
            processed_images[0],  # オリジナル
            np.ones((target_h, target_w, 3), dtype=np.uint8) * 255   # 空白
        ])
        
        # 2行目：4つの角
        bottom_row = np.hstack([
            processed_images[1],  # top-left
            processed_images[2],  # top-right
            processed_images[3]   # bottom-left
        ])
        
        # 3行目：残りの角
        third_row = np.hstack([
            np.ones((target_h, target_w, 3), dtype=np.uint8) * 255,  # 空白
            processed_images[4],  # bottom-right
            np.ones((target_h, target_w, 3), dtype=np.uint8) * 255   # 空白
        ])
        
        grid = np.vstack([top_row, bottom_row, third_row])
    else:
        # フォールバック：横一列
        grid = np.hstack(processed_images)
    
    return grid

def create_mask_overlay(balloon: np.ndarray, mask: np.ndarray, title: str) -> np.ndarray:
    """マスクオーバーレイ付きの画像を作成"""
    # マスクを3チャンネルに変換
    overlay = balloon.copy()
    
    # マスク部分を薄い緑でハイライト
    mask_area = mask > 0
    overlay[mask_area] = overlay[mask_area] * 0.7 + np.array([0, 180, 0]) * 0.3
    
    # タイトルを追加
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)  # 赤色
    thickness = 1
    
    text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    text_x = (overlay.shape[1] - text_size[0]) // 2
    text_y = 25
    
    # 白い背景を追加してテキストを見やすくする
    cv2.rectangle(overlay, (text_x-5, text_y-15), (text_x+text_size[0]+5, text_y+5), (255, 255, 255), -1)
    cv2.putText(overlay, title, (text_x, text_y), font, font_scale, color, thickness)
    
    return overlay

if __name__ == "__main__":
    test_simple_corner_cutting()
