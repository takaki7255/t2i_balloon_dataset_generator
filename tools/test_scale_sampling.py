#!/usr/bin/env python3
"""
スケールサンプリングのテスト
"""

import numpy as np
import random
import matplotlib.pyplot as plt

def sample_scale_test(bg_w: int, bw: int, cfg: dict) -> float:
    """統計情報に基づくスケールサンプリングのテスト版"""
    mode = cfg.get("SCALE_MODE", "uniform")
    if mode == "lognormal":
        mean = cfg["SCALE_MEAN"]
        std = cfg["SCALE_STD"]
        clip_min, clip_max = cfg["SCALE_CLIP"]
        
        # 対数正規分布のパラメータ計算
        mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
        sigma = np.sqrt(np.log(1 + (std**2)/(mean**2)))
        
        s = np.random.lognormal(mu, sigma)
        clipped = np.clip(s, clip_min, clip_max)
        
        print(f"  lognormal生成値: {s:.3f} → クリップ後: {clipped:.3f}")
        return float(clipped)
    else:
        s = random.uniform(*cfg["SCALE_RANGE"])
        print(f"  uniform生成値: {s:.3f}")
        return s

def test_scale_sampling():
    cfg = {
        "SCALE_MODE": "lognormal",
        "SCALE_MEAN": 0.25,
        "SCALE_STD": 0.08,
        "SCALE_CLIP": (0.05, 0.4),
        "SCALE_RANGE": (0.1, 0.3)
    }
    
    bg_w = 1331  # 平均背景幅
    bw = 583     # 平均吹き出し幅
    
    print("=== スケールサンプリングテスト ===")
    print(f"背景幅: {bg_w}px, 吹き出し幅: {bw}px")
    print(f"設定: mean={cfg['SCALE_MEAN']}, std={cfg['SCALE_STD']}")
    print(f"クリップ範囲: {cfg['SCALE_CLIP']}")
    
    # サンプリングテスト
    scales = []
    sizes = []
    
    for i in range(20):
        scale = sample_scale_test(bg_w, bw, cfg)
        final_w = int(bg_w * scale)
        scales.append(scale)
        sizes.append(final_w)
        print(f"テスト{i+1:2d}: スケール={scale:.3f} → 最終幅={final_w}px")
    
    print(f"\n=== 統計情報 ===")
    print(f"平均スケール: {np.mean(scales):.3f}")
    print(f"最小スケール: {np.min(scales):.3f}")
    print(f"最大スケール: {np.max(scales):.3f}")
    print(f"平均最終幅: {np.mean(sizes):.0f}px")
    print(f"最小最終幅: {np.min(sizes):.0f}px")
    print(f"最大最終幅: {np.max(sizes):.0f}px")
    
    # 対数正規分布の理論値確認
    mu = np.log(cfg['SCALE_MEAN']**2 / np.sqrt(cfg['SCALE_STD']**2 + cfg['SCALE_MEAN']**2))
    sigma = np.sqrt(np.log(1 + (cfg['SCALE_STD']**2)/(cfg['SCALE_MEAN']**2)))
    
    print(f"\n=== 対数正規分布パラメータ ===")
    print(f"mu: {mu:.4f}")
    print(f"sigma: {sigma:.4f}")
    print(f"理論平均: {np.exp(mu + sigma**2/2):.3f}")

if __name__ == "__main__":
    test_scale_sampling()
