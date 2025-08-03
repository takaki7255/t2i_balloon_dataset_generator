#!/usr/bin/env python3
"""
generated_double_backs内の画像をサイズ別に仕分けするスクリプト
1536x1024の画像とそれ以外を別々のフォルダに分けて保存
"""

import cv2
import os
import shutil
from pathlib import Path
import glob

def sort_images_by_size():
    """画像をサイズ別に仕分け"""
    
    # ディレクトリパス設定
    input_dir = "./generated_double_backs"
    output_1536x1024_dir = "./generated_double_backs_1536x1024"
    output_other_sizes_dir = "./generated_double_backs_other_sizes"

    # 出力ディレクトリを作成
    os.makedirs(output_1536x1024_dir, exist_ok=True)
    os.makedirs(output_other_sizes_dir, exist_ok=True)
    
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
    print(f"📁 1536x1024画像の保存先: {output_1536x1024_dir}")
    print(f"📁 その他サイズ画像の保存先: {output_other_sizes_dir}")
    print(f"🖼️  処理対象: {len(image_files)}枚の画像")
    print("-" * 60)
    
    # 統計用カウンター
    count_1536x1024 = 0
    count_other_sizes = 0
    error_count = 0
    size_stats = {}
    
    for i, image_path in enumerate(image_files, 1):
        try:
            # ファイル名を取得
            filename = os.path.basename(image_path)
            
            # 画像を読み込み（ヘッダー情報のみでサイズを取得）
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 読み込み失敗: {filename}")
                error_count += 1
                continue
            
            # 画像サイズを取得 (height, width, channels)
            height, width = image.shape[:2]
            size_key = f"{width}x{height}"
            
            # サイズ統計を更新
            if size_key in size_stats:
                size_stats[size_key] += 1
            else:
                size_stats[size_key] = 1
            
            # サイズに応じて仕分け
            if width == 1536 and height == 1024:
                # 1536x1024の画像
                output_path = os.path.join(output_1536x1024_dir, filename)
                shutil.copy2(image_path, output_path)
                count_1536x1024 += 1
                status = "✅ 1536x1024"
            else:
                # その他のサイズ
                output_path = os.path.join(output_other_sizes_dir, filename)
                shutil.copy2(image_path, output_path)
                count_other_sizes += 1
                status = f"📏 {width}x{height}"
            
            print(f"[{i:3d}/{len(image_files)}] {filename:15s} → {status}")
            
        except Exception as e:
            print(f"❌ エラー: {filename} - {str(e)}")
            error_count += 1
    
    print("-" * 60)
    print(f"🎉 仕分け完了!")
    print(f"✅ 1536x1024画像: {count_1536x1024}枚")
    print(f"📏 その他サイズ画像: {count_other_sizes}枚")
    if error_count > 0:
        print(f"❌ エラー: {error_count}枚")
    
    print("\n📊 画像サイズ別統計:")
    for size, count in sorted(size_stats.items()):
        print(f"   {size:15s}: {count:3d}枚")
    
    print(f"\n📁 保存先:")
    print(f"   1536x1024: {output_1536x1024_dir}/")
    print(f"   その他: {output_other_sizes_dir}/")

def verify_sorting():
    """仕分け結果を確認"""
    input_dir = "../generated_double_backs"
    output_1536x1024_dir = "../generated_double_backs_1536x1024"
    output_other_sizes_dir = "../generated_double_backs_other_sizes"
    
    print("\n🔍 仕分け結果の確認:")
    print("-" * 40)
    
    # ファイル数を確認
    input_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
        input_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    output_1536x1024_files = []
    output_other_files = []
    
    if os.path.exists(output_1536x1024_dir):
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
            output_1536x1024_files.extend(glob.glob(os.path.join(output_1536x1024_dir, ext)))
    
    if os.path.exists(output_other_sizes_dir):
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
            output_other_files.extend(glob.glob(os.path.join(output_other_sizes_dir, ext)))
    
    total_output = len(output_1536x1024_files) + len(output_other_files)
    
    print(f"📊 ファイル数確認:")
    print(f"   入力ファイル数: {len(input_files)}")
    print(f"   1536x1024出力: {len(output_1536x1024_files)}")
    print(f"   その他出力: {len(output_other_files)}")
    print(f"   出力合計: {total_output}")
    
    if len(input_files) == total_output:
        print("✅ すべてのファイルが正常に仕分けされました")
    else:
        print("⚠️  ファイル数が一致しません")
    
    # サンプル画像の詳細確認
    if output_1536x1024_files:
        sample_path = output_1536x1024_files[0]
        sample_image = cv2.imread(sample_path)
        if sample_image is not None:
            h, w = sample_image.shape[:2]
            print(f"\n📸 1536x1024フォルダのサンプル:")
            print(f"   ファイル: {os.path.basename(sample_path)}")
            print(f"   サイズ: {w}x{h}")
    
    if output_other_files:
        sample_path = output_other_files[0]
        sample_image = cv2.imread(sample_path)
        if sample_image is not None:
            h, w = sample_image.shape[:2]
            print(f"\n📸 その他サイズフォルダのサンプル:")
            print(f"   ファイル: {os.path.basename(sample_path)}")
            print(f"   サイズ: {w}x{h}")

def show_size_distribution():
    """元の画像のサイズ分布を事前確認"""
    input_dir = "../generated_double_backs"
    
    print("🔍 元画像のサイズ分布を確認中...")
    print("-" * 40)
    
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"❌ {input_dir}に画像ファイルが見つかりません")
        return
    
    size_distribution = {}
    
    for image_path in image_files[:10]:  # 最初の10枚をサンプル確認
        try:
            image = cv2.imread(image_path)
            if image is not None:
                height, width = image.shape[:2]
                size_key = f"{width}x{height}"
                filename = os.path.basename(image_path)
                
                if size_key in size_distribution:
                    size_distribution[size_key].append(filename)
                else:
                    size_distribution[size_key] = [filename]
        except:
            continue
    
    print("📊 サンプル画像のサイズ分布（最初の10枚）:")
    for size, files in sorted(size_distribution.items()):
        print(f"   {size:15s}: {len(files)}枚 例: {files[0]}")
    print()

if __name__ == "__main__":
    print("🖼️  画像サイズ別仕分けツール")
    print("=" * 60)
    
    # 事前確認
    show_size_distribution()
    
    # 仕分け実行
    sort_images_by_size()
    
    # 結果確認
    verify_sorting()
