"""
generated_balloons内の全ての画像に対してマスクを生成するスクリプト
"""

import os
from pathlib import Path
from balloon_mask import generate_mask

def generate_all_masks():
    """generated_balloons内の全画像にマスクを生成"""
    
    balloons_dir = "generated_balloons"
    masks_dir = "masks"
    
    # masksディレクトリ作成
    os.makedirs(masks_dir, exist_ok=True)
    
    # 処理済みファイル数
    processed = 0
    skipped = 0
    
    for filename in os.listdir(balloons_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(balloons_dir, filename)
            stem = Path(filename).stem
            mask_filename = f"{stem}_mask.png"
            output_path = os.path.join(masks_dir, mask_filename)
            
            # 既にマスクが存在する場合はスキップ
            if os.path.exists(output_path):
                print(f"スキップ: {mask_filename} は既に存在します")
                skipped += 1
                continue
            
            try:
                generate_mask(input_path, output_path)
                print(f"✓ マスク生成完了: {mask_filename}")
                processed += 1
            except Exception as e:
                print(f"✗ マスク生成失敗 ({filename}): {e}")
    
    print(f"\n処理完了: {processed}個のマスクを生成, {skipped}個をスキップ")

if __name__ == "__main__":
    generate_all_masks()
