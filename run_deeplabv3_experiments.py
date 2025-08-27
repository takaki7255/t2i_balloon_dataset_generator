"""
DeepLab v3+ 学習スクリプト実行例
異なる設定でのテスト用
"""

import os
import sys
from pathlib import Path

# 元のスクリプトから設定とmain関数をインポート
sys.path.append(str(Path(__file__).parent))

def run_deeplabv3_experiment(dataset_name, backbone="resnet50", output_stride=16, batch_size=8, epochs=50):
    """
    DeepLab v3+ 実験実行
    """
    
    # 設定を動的に変更
    from train_deeplabv3_split import CFG, main
    
    # 設定上書き
    CFG["ROOT"] = Path(dataset_name)
    CFG["BACKBONE"] = backbone
    CFG["OUTPUT_STRIDE"] = output_stride
    CFG["BATCH"] = batch_size
    CFG["EPOCHS"] = epochs
    CFG["DATASET"] = dataset_name.replace("_dataset", "").replace("syn", "")
    CFG["RUN_NAME"] = f"deeplabv3_{backbone}_os{output_stride}_{CFG['DATASET']}"
    
    print(f"🚀 DeepLab v3+ 実験開始")
    print(f"  データセット: {dataset_name}")
    print(f"  バックボーン: {backbone}")
    print(f"  Output Stride: {output_stride}")
    print(f"  バッチサイズ: {batch_size}")
    print(f"  エポック数: {epochs}")
    
    # 学習実行
    main()

if __name__ == "__main__":
    # 複数の実験を順次実行
    experiments = [
        # (dataset, backbone, output_stride, batch_size, epochs)
        ("syn2000_dataset01", "resnet50", 16, 8, 50),
        ("syn2000_dataset01", "resnet50", 8, 6, 50),   # 高解像度、バッチサイズ減
        ("syn2000_dataset01", "resnet101", 16, 6, 50), # 大きなモデル
    ]
    
    for dataset, backbone, output_stride, batch_size, epochs in experiments:
        if Path(dataset).exists():
            try:
                run_deeplabv3_experiment(dataset, backbone, output_stride, batch_size, epochs)
                print(f"✅ 実験完了: {dataset} - {backbone} - OS{output_stride}")
            except Exception as e:
                print(f"❌ 実験失敗: {dataset} - {backbone} - OS{output_stride}: {e}")
        else:
            print(f"⚠️ データセットが見つかりません: {dataset}")
