#!/usr/bin/env python3
"""
修正されたアルゴリズムで小規模テストデータセットを生成
"""

import sys
import os
sys.path.append('.')

from create_syn_dataset import main

def generate_test_dataset():
    print("🚀 修正されたアルゴリズムでテストデータセット生成開始")
    print("=" * 60)
    
    # テスト用の設定を一時的に適用
    import create_syn_dataset
    
    # 小規模テスト用の設定を上書き
    original_cfg = create_syn_dataset.CFG.copy()
    
    test_cfg = {
        **original_cfg,
        "TARGET_TOTAL_IMAGES": 20,        # テスト用に20枚のみ
        "NUM_BALLOONS_RANGE": (5, 12),    # 吹き出し個数を適度に
        "SCALE_MEAN": 0.009200,           # 修正された設定
        "SCALE_STD": 0.008000,
        "SCALE_CLIP": (0.002000, 0.020000),
        "SCALE_RANGE": (0.002000, 0.020000),
    }
    
    # 設定を適用
    create_syn_dataset.CFG.update(test_cfg)
    
    # 出力ディレクトリを変更
    original_main = create_syn_dataset.main
    
    def test_main():
        # main関数内の設定を変更
        temp_output_dir = "test_corrected_results"
        temp_mask_output_dir = "test_corrected_results_mask"
        final_output_dir = "corrected_algorithm_test_dataset"
        
        # ディレクトリ作成
        os.makedirs(temp_output_dir, exist_ok=True)
        os.makedirs(temp_mask_output_dir, exist_ok=True)
        os.makedirs(final_output_dir, exist_ok=True)
        
        print(f"📁 出力先: {final_output_dir}")
        print(f"🎯 生成枚数: {test_cfg['TARGET_TOTAL_IMAGES']}")
        print(f"🎈 吹き出し個数: {test_cfg['NUM_BALLOONS_RANGE']}")
        print(f"📊 面積比平均: {test_cfg['SCALE_MEAN']*100:.3f}%")
        
        # 元の main 関数を呼び出し（設定は既に変更済み）
        return original_main()
    
    try:
        result = test_main()
        print("✅ テストデータセット生成完了")
        
        # 生成されたデータセットの簡単な分析
        dataset_path = "corrected_algorithm_test_dataset"
        if os.path.exists(dataset_path):
            train_images = len([f for f in os.listdir(os.path.join(dataset_path, "train", "images")) if f.endswith('.png')])
            val_images = len([f for f in os.listdir(os.path.join(dataset_path, "val", "images")) if f.endswith('.png')])
            print(f"📈 生成結果: train={train_images}枚, val={val_images}枚")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
    finally:
        # 設定を復元
        create_syn_dataset.CFG.update(original_cfg)

if __name__ == "__main__":
    generate_test_dataset()
