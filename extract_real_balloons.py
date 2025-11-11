"""
Manga109のセグメンテーションアノテーションから吹き出しを抽出

category_id=5の吹き出しをランダムに200個抽出し、
real_balloons/images と real_balloons/masks に保存する
"""

import json
import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
from pycocotools import mask as maskUtils
import os


def rle_to_mask(rle_dict):
    """
    RLE形式のセグメンテーションをバイナリマスクに変換
    
    Args:
        rle_dict: {'size': [height, width], 'counts': ...} 形式のRLE
    
    Returns:
        numpy.ndarray: バイナリマスク (height, width)
    """
    # pycocotoolsのRLE形式に変換
    rle = {
        'size': rle_dict['size'],
        'counts': rle_dict['counts'].encode('utf-8') if isinstance(rle_dict['counts'], str) else rle_dict['counts']
    }
    mask = maskUtils.decode(rle)
    return mask


def extract_balloons_from_json(json_path, manga_name, images_base_dir):
    """
    1つのJSONファイルから吹き出しアノテーションを抽出
    
    Args:
        json_path: JSONファイルのパス
        manga_name: 漫画タイトル
        images_base_dir: 画像ベースディレクトリ
    
    Returns:
        list: 吹き出し情報のリスト
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 画像情報をIDでマッピング
    images_dict = {img['id']: img for img in data['images']}
    
    # category_id=5 (balloon) のアノテーションを抽出
    balloons = []
    for ann in data['annotations']:
        if ann['category_id'] == 5:  # balloon
            image_info = images_dict.get(ann['image_id'])
            if image_info is None:
                continue
            
            # 画像パス構築
            image_path = Path(images_base_dir) / manga_name / image_info['file_name'].split('/')[-1]
            
            if not image_path.exists():
                print(f"警告: 画像が見つかりません: {image_path}")
                continue
            
            balloons.append({
                'manga': manga_name,
                'image_path': str(image_path),
                'annotation': ann,
                'image_info': image_info
            })
    
    return balloons


def save_balloon_with_mask(balloon_info, output_images_dir, output_masks_dir, index):
    """
    吹き出し画像とマスクを保存
    
    Args:
        balloon_info: 吹き出し情報辞書
        output_images_dir: 画像出力ディレクトリ
        output_masks_dir: マスク出力ディレクトリ
        index: 連番
    
    Returns:
        bool: 成功したかどうか
    """
    try:
        # 画像読み込み
        image = cv2.imread(balloon_info['image_path'])
        if image is None:
            print(f"エラー: 画像読み込み失敗: {balloon_info['image_path']}")
            return False
        
        # RLEマスクをデコード
        ann = balloon_info['annotation']
        mask = rle_to_mask(ann['segmentation'])
        
        # バウンディングボックスで切り出し
        x, y, w, h = ann['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # 画像サイズチェック
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            print(f"警告: 無効なバウンディングボックス: {ann['bbox']}")
            return False
        
        # 切り出し
        cropped_image = image[y:y+h, x:x+w].copy()
        cropped_mask = mask[y:y+h, x:x+w].copy()
        
        # マスクを8bitに変換
        cropped_mask_8bit = (cropped_mask * 255).astype(np.uint8)
        
        # 保存
        output_image_path = output_images_dir / f"{index:04d}.png"
        output_mask_path = output_masks_dir / f"{index:04d}.png"
        
        cv2.imwrite(str(output_image_path), cropped_image)
        cv2.imwrite(str(output_mask_path), cropped_mask_8bit)
        
        return True
        
    except Exception as e:
        print(f"エラー: {balloon_info['manga']} - {e}")
        return False


def main():
    # パス設定
    manga_seg_jsons_dir = Path("../Manga109_released_2023_12_07/manga_seg_jsons")
    manga_images_dir = Path("../Manga109_released_2023_12_07/images")
    output_base_dir = Path("real_balloons")
    output_images_dir = output_base_dir / "images"
    output_masks_dir = output_base_dir / "masks"
    
    # 出力ディレクトリ作成
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Manga109 吹き出し抽出スクリプト")
    print("=" * 60)
    print(f"JSONディレクトリ: {manga_seg_jsons_dir}")
    print(f"画像ディレクトリ: {manga_images_dir}")
    print(f"出力先: {output_base_dir}")
    print("=" * 60)
    
    # すべてのJSONファイルを処理
    all_balloons = []
    json_files = sorted(manga_seg_jsons_dir.glob("*.json"))
    
    print(f"\n📂 {len(json_files)}個のJSONファイルを処理中...")
    
    for json_path in tqdm(json_files, desc="JSONファイル処理"):
        manga_name = json_path.stem
        balloons = extract_balloons_from_json(json_path, manga_name, manga_images_dir)
        all_balloons.extend(balloons)
    
    print(f"\n✓ 合計 {len(all_balloons)} 個の吹き出しを検出しました")
    
    # ランダムに200個サンプリング
    target_count = 200
    if len(all_balloons) < target_count:
        print(f"警告: 吹き出しが{len(all_balloons)}個しかありません。全て保存します。")
        sampled_balloons = all_balloons
    else:
        sampled_balloons = random.sample(all_balloons, target_count)
    
    print(f"\n💾 {len(sampled_balloons)}個の吹き出しを保存中...")
    
    # 保存
    success_count = 0
    for idx, balloon_info in enumerate(tqdm(sampled_balloons, desc="保存中"), start=1):
        if save_balloon_with_mask(balloon_info, output_images_dir, output_masks_dir, idx):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ 完了: {success_count}/{len(sampled_balloons)} 個の吹き出しを保存しました")
    print(f"画像: {output_images_dir}")
    print(f"マスク: {output_masks_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
