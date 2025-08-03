#!/usr/bin/env python3
"""
generated_double_backs内の画像をグレースケール変換して
generated_double_backs_grayに保存するスクリプト
"""

import cv2
import os
from pathlib import Path
import glob

def convert_images_to_grayscale():
    """画像をグレースケール変換して保存"""
    
    # ディレクトリパス設定
    input_dir = "./../generated_double_backs"
    output_dir = "./../generated_double_backs_gray"

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 対応する画像形式
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    
    # すべての画像ファイルを取得
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"❌ {input_dir}に画像ファイルが見つかりません")
        return
    
    print(f"📁 入力ディレクトリ: {input_dir}")
    print(f"📁 出力ディレクトリ: {output_dir}")
    print(f"🖼️  処理対象: {len(image_files)}枚の画像")
    print("-" * 50)
    
    success_count = 0
    error_count = 0
    
    for i, image_path in enumerate(image_files, 1):
        try:
            # ファイル名を取得
            filename = os.path.basename(image_path)
            
            # 画像を読み込み
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 読み込み失敗: {filename}")
                error_count += 1
                continue
            
            # グレースケール変換
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 出力パスを設定
            output_path = os.path.join(output_dir, filename)
            
            # グレースケール画像を保存
            cv2.imwrite(output_path, gray_image)
            
            print(f"✅ [{i:3d}/{len(image_files)}] {filename} → グレースケール変換完了")
            success_count += 1
            
        except Exception as e:
            print(f"❌ エラー: {filename} - {str(e)}")
            error_count += 1
    
    print("-" * 50)
    print(f"🎉 変換完了!")
    print(f"✅ 成功: {success_count}枚")
    if error_count > 0:
        print(f"❌ エラー: {error_count}枚")
    print(f"📁 保存先: {output_dir}/")

def verify_conversion():
    """変換結果を確認"""
    input_dir = "generated_double_backs"
    output_dir = "generated_double_backs_gray"
    
    if not os.path.exists(output_dir):
        print(f"❌ 出力ディレクトリが存在しません: {output_dir}")
        return
    
    # ファイル数を比較
    input_files = []
    output_files = []
    
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
        input_files.extend(glob.glob(os.path.join(input_dir, ext)))
        output_files.extend(glob.glob(os.path.join(output_dir, ext)))
    
    print(f"📊 変換結果確認:")
    print(f"   入力ファイル数: {len(input_files)}")
    print(f"   出力ファイル数: {len(output_files)}")
    
    if len(input_files) == len(output_files):
        print("✅ すべてのファイルが正常に変換されました")
    else:
        print("⚠️  ファイル数が一致しません")
    
    # サンプル画像の情報を表示
    if output_files:
        sample_path = output_files[0]
        sample_image = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        if sample_image is not None:
            print(f"📸 サンプル画像情報:")
            print(f"   ファイル: {os.path.basename(sample_path)}")
            print(f"   サイズ: {sample_image.shape[1]}x{sample_image.shape[0]}")
            print(f"   チャンネル数: 1 (グレースケール)")

if __name__ == "__main__":
    print("🖼️  画像グレースケール変換ツール")
    print("=" * 50)
    
    # グレースケール変換実行
    convert_images_to_grayscale()
    
    print()
    
    # 変換結果確認
    verify_conversion()
