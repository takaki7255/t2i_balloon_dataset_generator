"""
最適化前後の合成結果を比較するスクリプト
"""

import cv2
import numpy as np
from pathlib import Path
import os
import time
import matplotlib.pyplot as plt
from composite_balloons_optimized import composite_random_balloons_optimized, get_mask_bounding_box, crop_balloon_and_mask
from generate_random_composite import composite_random_balloons_enhanced

def analyze_balloon_efficiency():
    """吹き出し画像の余白効率を分析"""
    balloons_dir = "generated_balloons"
    masks_dir = "masks"
    
    print("=== 吹き出し余白効率分析 ===")
    
    efficiency_data = []
    
    for balloon_file in os.listdir(balloons_dir):
        if not balloon_file.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        balloon_path = os.path.join(balloons_dir, balloon_file)
        mask_path = os.path.join(masks_dir, f"{Path(balloon_file).stem}_mask.png")
        
        if not os.path.exists(mask_path):
            continue
        
        # 画像とマスクを読み込み
        balloon = cv2.imread(balloon_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if balloon is None or mask is None:
            continue
        
        # 元のサイズ
        original_area = balloon.shape[0] * balloon.shape[1]
        
        # クロップ後のサイズ
        cropped_balloon, cropped_mask, bbox = crop_balloon_and_mask(balloon, mask)
        
        if cropped_balloon is not None:
            cropped_area = cropped_balloon.shape[0] * cropped_balloon.shape[1]
            efficiency = (cropped_area / original_area) * 100
            
            efficiency_data.append({
                'name': Path(balloon_file).stem,
                'original_size': f"{balloon.shape[1]}x{balloon.shape[0]}",
                'cropped_size': f"{cropped_balloon.shape[1]}x{cropped_balloon.shape[0]}",
                'efficiency': efficiency,
                'reduction': 100 - efficiency
            })
    
    # 統計情報を表示
    if efficiency_data:
        efficiencies = [data['efficiency'] for data in efficiency_data]
        reductions = [data['reduction'] for data in efficiency_data]
        
        print(f"分析対象: {len(efficiency_data)}個の吹き出し")
        print(f"平均効率: {np.mean(efficiencies):.1f}% (余白除去率: {np.mean(reductions):.1f}%)")
        print(f"最高効率: {max(efficiencies):.1f}%")
        print(f"最低効率: {min(efficiencies):.1f}%")
        print(f"中央値: {np.median(efficiencies):.1f}%")
        
        # 効率の低い順にTop5を表示
        sorted_data = sorted(efficiency_data, key=lambda x: x['efficiency'])
        print(f"\n余白が多い吹き出し Top5:")
        for i, data in enumerate(sorted_data[:5]):
            print(f"  {i+1}. {data['name']}: {data['efficiency']:.1f}% ({data['original_size']} → {data['cropped_size']})")
        
        # グラフ作成
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(efficiencies, bins=20, alpha=0.7, color='blue')
        plt.title('Efficiency Distribution')
        plt.xlabel('Efficiency (%)')
        plt.ylabel('Count')
        
        plt.subplot(2, 2, 2)
        plt.hist(reductions, bins=20, alpha=0.7, color='red')
        plt.title('Whitespace Reduction Distribution')
        plt.xlabel('Reduction (%)')
        plt.ylabel('Count')
        
        plt.subplot(2, 2, 3)
        names = [data['name'] for data in sorted_data[:10]]
        effs = [data['efficiency'] for data in sorted_data[:10]]
        plt.barh(range(len(names)), effs)
        plt.yticks(range(len(names)), names)
        plt.xlabel('Efficiency (%)')
        plt.title('Bottom 10 Efficiency')
        
        plt.subplot(2, 2, 4)
        plt.boxplot(efficiencies)
        plt.ylabel('Efficiency (%)')
        plt.title('Efficiency Box Plot')
        
        plt.tight_layout()
        plt.savefig('balloon_efficiency_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\nグラフを保存: balloon_efficiency_analysis.png")
        
        return efficiency_data
    else:
        print("分析可能な吹き出しが見つかりませんでした")
        return []


def compare_methods():
    """従来手法と最適化手法を比較"""
    balloons_dir = "generated_balloons"
    masks_dir = "masks"
    backgrounds_dir = "generated_backs"
    
    # テスト用のファイルを選択
    balloon_files = [f for f in os.listdir(balloons_dir) if f.endswith('.png')][:10]
    bg_files = [f for f in os.listdir(backgrounds_dir) if f.endswith('.png')][:3]
    
    print("=== 手法比較テスト ===")
    
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
    
    results = []
    
    for bg_file in bg_files:
        bg_path = os.path.join(backgrounds_dir, bg_file)
        bg_name = Path(bg_file).stem
        
        print(f"\n背景: {bg_name}")
        
        # 従来手法
        print("  従来手法実行中...")
        start_time = time.time()
        try:
            orig_img, orig_mask, orig_placed = composite_random_balloons_enhanced(
                bg_path, test_pairs,
                scale_range=(0.1, 0.4),
                num_balloons_range=(3, 6),
                max_attempts=100
            )
            orig_time = time.time() - start_time
            orig_success = len(orig_placed)
        except Exception as e:
            print(f"    従来手法エラー: {e}")
            orig_time = 0
            orig_success = 0
        
        # 最適化手法
        print("  最適化手法実行中...")
        start_time = time.time()
        try:
            opt_img, opt_mask, opt_placed = composite_random_balloons_optimized(
                bg_path, test_pairs,
                scale_range=(0.1, 0.4),
                num_balloons_range=(3, 6),
                max_attempts=100
            )
            opt_time = time.time() - start_time
            opt_success = len(opt_placed)
        except Exception as e:
            print(f"    最適化手法エラー: {e}")
            opt_time = 0
            opt_success = 0
        
        # 結果保存
        if orig_success > 0:
            cv2.imwrite(f"compare_orig_{bg_name}.png", orig_img)
            cv2.imwrite(f"compare_orig_{bg_name}_mask.png", orig_mask)
        
        if opt_success > 0:
            cv2.imwrite(f"compare_opt_{bg_name}.png", opt_img)
            cv2.imwrite(f"compare_opt_{bg_name}_mask.png", opt_mask)
        
        results.append({
            'background': bg_name,
            'orig_time': orig_time,
            'orig_success': orig_success,
            'opt_time': opt_time,
            'opt_success': opt_success
        })
        
        print(f"    従来手法: {orig_success}個配置, {orig_time:.2f}秒")
        print(f"    最適化手法: {opt_success}個配置, {opt_time:.2f}秒")
    
    # 総合結果
    print(f"\n=== 総合結果 ===")
    total_orig_time = sum(r['orig_time'] for r in results)
    total_opt_time = sum(r['opt_time'] for r in results)
    total_orig_success = sum(r['orig_success'] for r in results)
    total_opt_success = sum(r['opt_success'] for r in results)
    
    print(f"従来手法: 総配置数 {total_orig_success}, 総時間 {total_orig_time:.2f}秒")
    print(f"最適化手法: 総配置数 {total_opt_success}, 総時間 {total_opt_time:.2f}秒")
    
    if total_orig_time > 0:
        time_improvement = ((total_orig_time - total_opt_time) / total_orig_time) * 100
        print(f"時間短縮: {time_improvement:.1f}%")
    
    if total_orig_success > 0:
        success_improvement = ((total_opt_success - total_orig_success) / total_orig_success) * 100
        print(f"配置成功率改善: {success_improvement:.1f}%")
    
    return results


def main():
    """メイン処理"""
    print("吹き出し合成最適化分析ツール")
    print("1. 余白効率分析")
    print("2. 手法比較テスト")
    print("3. 両方実行")
    
    choice = input("選択 (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        efficiency_data = analyze_balloon_efficiency()
    
    if choice in ['2', '3']:
        comparison_results = compare_methods()
    
    print("\n分析完了！")


if __name__ == "__main__":
    main()
