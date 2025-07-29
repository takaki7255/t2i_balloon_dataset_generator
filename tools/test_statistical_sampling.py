#!/usr/bin/env python3
"""
統計情報ベースサンプリング機能のテストスクリプト
"""

import json
import os
import sys
from pathlib import Path
import tempfile

# create_syn_dataset.pyから関数をインポート
sys.path.append('..')
from create_syn_dataset import load_count_probs, sample_num_balloons, sample_scale

def test_count_probs_loading():
    """統計ファイル読み込みテスト"""
    print("=== 統計ファイル読み込みテスト ===")
    
    stats_file = "../balloon_count_statistics.txt"
    if not os.path.exists(stats_file):
        print(f"統計ファイルが見つかりません: {stats_file}")
        return None
        
    try:
        probs = load_count_probs(stats_file)
        if probs is not None:
            print(f"✅ 統計分布読み込み成功")
            print(f"確率配列サイズ: {len(probs)}")
            print(f"確率の合計: {probs.sum():.4f}")
            
            # 上位確率の表示
            print("\n上位確率:")
            for i, p in enumerate(probs):
                if p > 0.01:  # 1%以上の確率
                    print(f"  {i}個: {p:.3f} ({p*100:.1f}%)")
            return probs
        else:
            print("❌ 統計分布読み込み失敗")
            return None
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def test_statistical_sampling(probs, num_tests=100):
    """統計ベースサンプリングテスト"""
    print(f"\n=== 統計ベースサンプリングテスト (n={num_tests}) ===")
    
    cfg_stats = {
        "COUNT_PROBS": probs,
        "NUM_BALLOONS_RANGE": (1, 20)
    }
    
    cfg_uniform = {
        "NUM_BALLOONS_RANGE": (1, 20)
    }
    
    # 統計ベースサンプリング結果
    stats_results = []
    for _ in range(num_tests):
        n = sample_num_balloons(cfg_stats, max_available=50)
        stats_results.append(n)
    
    # 一様サンプリング結果  
    uniform_results = []
    for _ in range(num_tests):
        n = sample_num_balloons(cfg_uniform, max_available=50)
        uniform_results.append(n)
    
    # 結果比較
    print("統計ベース分布:")
    count_distribution(stats_results, "  ")
    
    print("\n一様分布:")
    count_distribution(uniform_results, "  ")

def test_scale_sampling():
    """スケールサンプリングテスト"""
    print("\n=== スケールサンプリングテスト ===")
    
    cfg_uniform = {
        "SCALE_MODE": "uniform",
        "SCALE_RANGE": (0.1, 0.4)
    }
    
    cfg_lognormal = {
        "SCALE_MODE": "lognormal", 
        "SCALE_MEAN": 0.2,
        "SCALE_STD": 0.05,
        "SCALE_CLIP": (0.05, 0.4)
    }
    
    bg_w = 800
    bw = 200
    
    # テストサンプリング
    uniform_scales = [sample_scale(bg_w, bw, cfg_uniform) for _ in range(100)]
    lognormal_scales = [sample_scale(bg_w, bw, cfg_lognormal) for _ in range(100)]
    
    print(f"一様分布スケール: mean={sum(uniform_scales)/len(uniform_scales):.3f}, range=[{min(uniform_scales):.3f}, {max(uniform_scales):.3f}]")
    print(f"対数正規分布スケール: mean={sum(lognormal_scales)/len(lognormal_scales):.3f}, range=[{min(lognormal_scales):.3f}, {max(lognormal_scales):.3f}]")

def count_distribution(values, prefix=""):
    """数値分布を表示"""
    from collections import Counter
    counts = Counter(values)
    total = len(values)
    
    for val in sorted(counts.keys())[:10]:  # 上位10個まで表示
        freq = counts[val]
        pct = freq / total * 100
        print(f"{prefix}{val}個: {freq}回 ({pct:.1f}%)")

def test_config_generation():
    """設定ファイル生成テスト"""
    print("\n=== 設定ファイル生成テスト ===")
    
    # 統計情報を読み込み
    probs = load_count_probs("../balloon_count_statistics.txt")
    
    test_cfg = {
        "SCALE_RANGE": (0.1, 0.3),
        "NUM_BALLOONS_RANGE": (2, 10),
        "MAX_ATTEMPTS": 200,
        "TARGET_TOTAL_IMAGES": 50,  # テスト用に少数
        "TRAIN_RATIO": 0.8,
        "BALLOON_SPLIT_SEED": 42,
        
        # 統計情報ベースサンプリング
        "SCALE_MODE": "lognormal",
        "SCALE_MEAN": 0.25,
        "SCALE_STD": 0.08,
        "SCALE_CLIP": (0.05, 0.5),
        "COUNT_PROBS": probs,
        "COUNT_STATS_FILE": "balloon_count_statistics.txt",
    }
    
    # 一時ファイルで設定出力テスト
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        # JSON serialization helper
        def _json_default(o):
            import numpy as np
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            if isinstance(o, Path):
                return str(o)
            return str(o)
        
        json.dump(test_cfg, f, ensure_ascii=False, indent=2, default=_json_default)
        temp_path = f.name
    
    print(f"✅ 設定ファイル生成成功: {temp_path}")
    
    # ファイル読み込み確認
    with open(temp_path, 'r') as f:
        loaded_cfg = json.load(f)
    
    print(f"✅ 設定ファイル読み込み成功")
    print(f"統計確率配列サイズ: {len(loaded_cfg['COUNT_PROBS']) if loaded_cfg['COUNT_PROBS'] else 'None'}")
    
    # クリーンアップ
    os.unlink(temp_path)

if __name__ == "__main__":
    print("統計情報ベースサンプリング機能テスト")
    print("=" * 50)
    
    # 1. 統計ファイル読み込みテスト
    probs = test_count_probs_loading()
    
    if probs is not None:
        # 2. サンプリングテスト
        test_statistical_sampling(probs, 200)
        
        # 3. スケールサンプリングテスト
        test_scale_sampling()
        
        # 4. 設定ファイル生成テスト
        test_config_generation()
        
        print("\n✅ 全テスト完了")
    else:
        print("\n❌ 統計ファイル読み込みに失敗したため、テストを中断")
