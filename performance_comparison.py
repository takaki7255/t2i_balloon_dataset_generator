"""
最適化版と従来版の詳細比較テスト
"""

import cv2
import numpy as np
from pathlib import Path
import os
import time
import random
from composite_balloons_optimized import composite_random_balloons_optimized
from generate_random_composite import composite_random_balloons_enhanced

def detailed_comparison_test():
    """詳細な性能比較テスト"""
    balloons_dir = "generated_balloons"
    masks_dir = "masks"
    backgrounds_dir = "generated_backs"
    
    # テスト設定
    test_config = {
        "scale_range": (0.1, 0.4),
        "num_balloons_range": (3, 8),
        "max_attempts": 200,
        "test_iterations": 5  # 各背景での試行回数
    }
    
    # テスト用ファイルを選択
    balloon_files = [f for f in os.listdir(balloons_dir) if f.endswith('.png')][:20]
    bg_files = [f for f in os.listdir(backgrounds_dir) if f.endswith('.png')][:5]
    
    print("=== 詳細性能比較テスト ===")
    print(f"テスト背景: {len(bg_files)}個")
    print(f"テスト吹き出し: {len(balloon_files)}個")
    print(f"各背景での試行回数: {test_config['test_iterations']}回")
    
    # ペアを構築
    test_pairs = []
    for balloon_file in balloon_files:
        balloon_path = os.path.join(balloons_dir, balloon_file)
        mask_path = os.path.join(masks_dir, f"{Path(balloon_file).stem}_mask.png")
        if os.path.exists(mask_path):
            test_pairs.append((balloon_path, mask_path))
    
    if not test_pairs:
        print("テスト用のペアが見つかりませんでした")
        return
    
    print(f"有効なペア: {len(test_pairs)}個")
    
    # 比較結果を格納
    comparison_results = {
        "original": {"times": [], "success_counts": [], "total_balloons": []},
        "optimized_crop": {"times": [], "success_counts": [], "total_balloons": []},
        "optimized_no_crop": {"times": [], "success_counts": [], "total_balloons": []}
    }
    
    for bg_file in bg_files:
        bg_path = os.path.join(backgrounds_dir, bg_file)
        bg_name = Path(bg_file).stem
        
        print(f"\n=== 背景: {bg_name} ===")
        
        for iteration in range(test_config["test_iterations"]):
            print(f"  試行 {iteration + 1}/{test_config['test_iterations']}")
            
            # 固定シードで公平比較
            test_seed = 42 + iteration
            
            # 1. 従来手法
            random.seed(test_seed)
            start_time = time.time()
            try:
                orig_img, orig_mask, orig_placed = composite_random_balloons_enhanced(
                    bg_path, test_pairs,
                    scale_range=test_config["scale_range"],
                    num_balloons_range=test_config["num_balloons_range"],
                    max_attempts=test_config["max_attempts"]
                )
                orig_time = time.time() - start_time
                orig_success = len(orig_placed)
                comparison_results["original"]["times"].append(orig_time)
                comparison_results["original"]["success_counts"].append(orig_success)
                comparison_results["original"]["total_balloons"].append(test_config["num_balloons_range"][1])
                
                print(f"    従来手法: {orig_success}個配置, {orig_time:.3f}秒")
            except Exception as e:
                print(f"    従来手法エラー: {e}")
            
            # 2. 最適化手法（クロッピングあり）
            random.seed(test_seed)
            start_time = time.time()
            try:
                opt_crop_img, opt_crop_mask, opt_crop_placed = composite_random_balloons_optimized(
                    bg_path, test_pairs,
                    scale_range=test_config["scale_range"],
                    num_balloons_range=test_config["num_balloons_range"],
                    max_attempts=test_config["max_attempts"],
                    use_cropping=True
                )
                opt_crop_time = time.time() - start_time
                opt_crop_success = len(opt_crop_placed)
                comparison_results["optimized_crop"]["times"].append(opt_crop_time)
                comparison_results["optimized_crop"]["success_counts"].append(opt_crop_success)
                comparison_results["optimized_crop"]["total_balloons"].append(test_config["num_balloons_range"][1])
                
                print(f"    最適化（crop）: {opt_crop_success}個配置, {opt_crop_time:.3f}秒")
            except Exception as e:
                print(f"    最適化（crop）エラー: {e}")
            
            # 3. 最適化手法（クロッピングなし）
            random.seed(test_seed)
            start_time = time.time()
            try:
                opt_no_crop_img, opt_no_crop_mask, opt_no_crop_placed = composite_random_balloons_optimized(
                    bg_path, test_pairs,
                    scale_range=test_config["scale_range"],
                    num_balloons_range=test_config["num_balloons_range"],
                    max_attempts=test_config["max_attempts"],
                    use_cropping=False
                )
                opt_no_crop_time = time.time() - start_time
                opt_no_crop_success = len(opt_no_crop_placed)
                comparison_results["optimized_no_crop"]["times"].append(opt_no_crop_time)
                comparison_results["optimized_no_crop"]["success_counts"].append(opt_no_crop_success)
                comparison_results["optimized_no_crop"]["total_balloons"].append(test_config["num_balloons_range"][1])
                
                print(f"    最適化（no crop）: {opt_no_crop_success}個配置, {opt_no_crop_time:.3f}秒")
            except Exception as e:
                print(f"    最適化（no crop）エラー: {e}")
            
            # 結果画像保存（最初の試行のみ）
            if iteration == 0:
                if 'orig_img' in locals():
                    cv2.imwrite(f"test_orig_{bg_name}.png", orig_img)
                if 'opt_crop_img' in locals():
                    cv2.imwrite(f"test_opt_crop_{bg_name}.png", opt_crop_img)
                if 'opt_no_crop_img' in locals():
                    cv2.imwrite(f"test_opt_no_crop_{bg_name}.png", opt_no_crop_img)
    
    # 統計分析
    print(f"\n=== 総合統計結果 ===")
    
    for method_name, data in comparison_results.items():
        if data["times"]:
            avg_time = np.mean(data["times"])
            avg_success = np.mean(data["success_counts"])
            avg_success_rate = np.mean([s/t for s, t in zip(data["success_counts"], data["total_balloons"])]) * 100
            
            print(f"\n{method_name}:")
            print(f"  平均実行時間: {avg_time:.3f}秒")
            print(f"  平均配置数: {avg_success:.1f}個")
            print(f"  平均成功率: {avg_success_rate:.1f}%")
            print(f"  標準偏差時間: {np.std(data['times']):.3f}秒")
    
    # 改善率計算
    if comparison_results["original"]["times"] and comparison_results["optimized_crop"]["times"]:
        time_improvement_crop = ((np.mean(comparison_results["original"]["times"]) - 
                                np.mean(comparison_results["optimized_crop"]["times"])) / 
                               np.mean(comparison_results["original"]["times"])) * 100
        
        success_improvement_crop = ((np.mean(comparison_results["optimized_crop"]["success_counts"]) - 
                                   np.mean(comparison_results["original"]["success_counts"])) / 
                                  np.mean(comparison_results["original"]["success_counts"])) * 100
        
        print(f"\n=== 改善率（クロッピング版） ===")
        print(f"時間短縮: {time_improvement_crop:+.1f}%")
        print(f"配置成功率改善: {success_improvement_crop:+.1f}%")
    
    if comparison_results["original"]["times"] and comparison_results["optimized_no_crop"]["times"]:
        time_improvement_no_crop = ((np.mean(comparison_results["original"]["times"]) - 
                                   np.mean(comparison_results["optimized_no_crop"]["times"])) / 
                                  np.mean(comparison_results["original"]["times"])) * 100
        
        success_improvement_no_crop = ((np.mean(comparison_results["optimized_no_crop"]["success_counts"]) - 
                                      np.mean(comparison_results["original"]["success_counts"])) / 
                                     np.mean(comparison_results["original"]["success_counts"])) * 100
        
        print(f"\n=== 改善率（非クロッピング版） ===")
        print(f"時間短縮: {time_improvement_no_crop:+.1f}%")
        print(f"配置成功率改善: {success_improvement_no_crop:+.1f}%")
    
    return comparison_results


if __name__ == "__main__":
    detailed_comparison_test()
