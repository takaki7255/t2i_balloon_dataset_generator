#!/usr/bin/env python3
"""
Augmentation適用状況の詳細分析
各吹き出しに対してどのaugmentationが適用されるかを確認
"""

import random
import numpy as np

def analyze_augmentation_coverage(num_balloons=1000):
    """
    指定された数の吹き出しに対してaugmentationの適用状況を分析
    """
    # 現在の設定
    cfg = {
        "HFLIP_PROB": 0.5,
        "ROT_PROB": 1.0,
        "ROT_RANGE": (-20, 20),
        "CUT_EDGE_PROB": 0.4,
        "CUT_EDGE_RATIO": (0.05, 0.20),
        "THIN_LINE_PROB": 0.5,
        "THIN_PIXELS": (1, 3),
    }
    
    results = {
        'no_augmentation': 0,
        'only_rotation': 0,
        'rotation_and_flip': 0,
        'rotation_and_cut': 0,
        'rotation_and_thin': 0,
        'multiple_augmentations': 0,
        'all_augmentations': 0
    }
    
    augmentation_counts = {
        'rotation': 0,
        'flip': 0,
        'cut_edge': 0,
        'thin_line': 0
    }
    
    for i in range(num_balloons):
        applied = []
        
        # 各augmentationの適用判定
        if random.random() < cfg["ROT_PROB"]:
            applied.append('rotation')
            augmentation_counts['rotation'] += 1
            
        if random.random() < cfg["HFLIP_PROB"]:
            applied.append('flip')
            augmentation_counts['flip'] += 1
            
        if random.random() < cfg["CUT_EDGE_PROB"]:
            applied.append('cut_edge')
            augmentation_counts['cut_edge'] += 1
            
        if random.random() < cfg["THIN_LINE_PROB"]:
            applied.append('thin_line')
            augmentation_counts['thin_line'] += 1
        
        # 結果分類
        num_applied = len(applied)
        
        if num_applied == 0:
            results['no_augmentation'] += 1
        elif num_applied == 1 and 'rotation' in applied:
            results['only_rotation'] += 1
        elif num_applied == 2:
            if 'rotation' in applied and 'flip' in applied:
                results['rotation_and_flip'] += 1
            elif 'rotation' in applied and 'cut_edge' in applied:
                results['rotation_and_cut'] += 1
            elif 'rotation' in applied and 'thin_line' in applied:
                results['rotation_and_thin'] += 1
        elif num_applied >= 3:
            results['multiple_augmentations'] += 1
            if num_applied == 4:
                results['all_augmentations'] += 1
    
    return results, augmentation_counts

def print_analysis_results(results, augmentation_counts, num_balloons):
    """分析結果を表示"""
    print("=== Augmentation適用状況分析 ===\n")
    print(f"分析対象: {num_balloons}個の吹き出し\n")
    
    print("=== 各Augmentationの適用率 ===")
    for aug_type, count in augmentation_counts.items():
        rate = (count / num_balloons) * 100
        print(f"{aug_type:12}: {count:4}/{num_balloons} ({rate:5.1f}%)")
    
    print("\n=== 組み合わせパターン ===")
    for pattern, count in results.items():
        rate = (count / num_balloons) * 100
        print(f"{pattern:25}: {count:4}/{num_balloons} ({rate:5.1f}%)")
    
    print("\n=== 重要な統計 ===")
    no_aug = results['no_augmentation']
    at_least_one = num_balloons - no_aug
    print(f"拡張なし                   : {no_aug:4}/{num_balloons} ({(no_aug/num_balloons)*100:5.1f}%)")
    print(f"少なくとも1つの拡張適用      : {at_least_one:4}/{num_balloons} ({(at_least_one/num_balloons)*100:5.1f}%)")
    
    multiple_aug = results['multiple_augmentations'] + results['all_augmentations']
    print(f"複数拡張の同時適用          : {multiple_aug:4}/{num_balloons} ({(multiple_aug/num_balloons)*100:5.1f}%)")

def suggest_alternative_configs():
    """代替設定を提案"""
    print("\n=== 代替設定の提案 ===\n")
    
    configs = [
        {
            "name": "保守的設定（少ない拡張）",
            "config": {
                "HFLIP_PROB": 0.3,
                "ROT_PROB": 0.6,
                "CUT_EDGE_PROB": 0.2,
                "THIN_LINE_PROB": 0.3
            },
            "description": "拡張を控えめにして、元の吹き出しの特徴を保持"
        },
        {
            "name": "バランス設定",
            "config": {
                "HFLIP_PROB": 0.5,
                "ROT_PROB": 0.8,
                "CUT_EDGE_PROB": 0.4,
                "THIN_LINE_PROB": 0.5
            },
            "description": "適度な多様性と元特徴の保持のバランス"
        },
        {
            "name": "現在の設定（積極的）",
            "config": {
                "HFLIP_PROB": 0.5,
                "ROT_PROB": 1.0,
                "CUT_EDGE_PROB": 0.4,
                "THIN_LINE_PROB": 0.5
            },
            "description": "すべてに回転適用、高い多様性を重視"
        }
    ]
    
    for i, config_data in enumerate(configs, 1):
        print(f"{i}. {config_data['name']}")
        print(f"   説明: {config_data['description']}")
        print("   設定:")
        for key, value in config_data['config'].items():
            print(f"     {key}: {value}")
        
        # 簡易計算
        probs = list(config_data['config'].values())
        no_aug_prob = 1.0
        for p in probs:
            no_aug_prob *= (1 - p)
        at_least_one_prob = 1 - no_aug_prob
        
        print(f"   → 少なくとも1つ適用される確率: {at_least_one_prob*100:.1f}%")
        print()

if __name__ == "__main__":
    # 分析実行
    random.seed(42)
    results, aug_counts = analyze_augmentation_coverage(1000)
    
    # 結果表示
    print_analysis_results(results, aug_counts, 1000)
    
    # 代替設定提案
    suggest_alternative_configs()
