"""
generated_body内のキャラクター画像からマスクを生成するスクリプト

2つのキャラクター抽出アルゴリズムを提供：
1. 輪郭ベース (extract_by_contour): 大津の二値化 + 輪郭抽出 [推奨]
   - より安定した結果
   - 自動閾値調整
   - ノイズに強い

2. flood-fillベース (extract_character_alpha): 線画検出 + 塗りつぶし
   - 複雑な形状に対応
   - 閾値調整が必要
   - 線画に依存

使用例:
    # 輪郭ベースで実行（推奨）
    python generate_body_mask.py
    
    # 両方の方法で比較実行
    from generate_body_mask import main_compare
    main_compare()
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_by_contour(path,
                       morph_kernel=3,    # モルフォロジー処理のカーネルサイズ
                       min_area=1000,     # 最小輪郭面積（ノイズ除去）
                       keep_largest=True  # 最大輪郭のみを保持
                       ):
    """輪郭ベースのキャラクター抽出（より安定）"""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1) 大津の二値化で自動閾値調整
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 二値化結果を反転（白背景→黒、キャラクター→白）
    binary_inv = cv2.bitwise_not(binary)
    
    # 2) モルフォロジー処理でノイズ除去
    if morph_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)
        binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
    
    # 3) 輪郭抽出
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4) 面積でフィルタリング
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not valid_contours:
        print(f"Warning: No valid contours found in {path}")
        return np.zeros((h, w), dtype=np.uint8)
    
    # 5) 最大面積の輪郭を選択（またはすべての有効輪郭）
    if keep_largest:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        contours_to_draw = [largest_contour]
    else:
        contours_to_draw = valid_contours
    
    # 6) マスク作成
    char_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(char_mask, contours_to_draw, 255)
    
    return char_mask

def extract_character_alpha(path,
                            white_thresh=245,  # 「ほぼ白」の判定閾値
                            dilate_iter=2,     # 線の隙間を塞ぐ膨張回数
                            close_kernel=5,    # エッジを整えるクロージング核
                            keep_largest=True  # 最大連結成分のみ採用
                            ):
    """キャラクター画像からアルファマスクを抽出（flood-fillベース）"""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1) 非白=線画領域を取得（黒〜グレー）
    nonwhite = (gray < white_thresh).astype(np.uint8) * 255

    # 2) 膨張で線の隙間を少し塞いでバリアを作る
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink_barrier = cv2.dilate(nonwhite, k3, iterations=dilate_iter)

    # 3) バリア以外（白キャンバス）を外周から flood fill して背景ラベル化
    white_canvas = (ink_barrier == 0).astype(np.uint8) * 255
    flood = white_canvas.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # 複数の外周種点から安全に塗る
    seeds = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
    step = max(1, min(h, w)//50)
    for x in range(0, w, step): 
        seeds += [(x, 0), (x, h-1)]
    for y in range(0, h, step): 
        seeds += [(0, y), (w-1, y)]
    
    for sx, sy in seeds:
        if flood[sy, sx] == 255:
            cv2.floodFill(flood, mask, (sx, sy), 128)

    bg_mask = (flood == 128).astype(np.uint8)      # 背景
    char_mask = (1 - bg_mask) * 255                # キャラ=非背景

    # 4) 連結成分の最大のみ残す（ゴミ除去）
    if keep_largest:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(char_mask, 8)
        if len(stats) > 1:
            max_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            char_mask = (labels == max_idx).astype(np.uint8) * 255

    # 5) 形を整える
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    char_mask = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, k_close, iterations=1)

    return char_mask

def generate_character_mask(
    input_path: str, 
    output_path: str | None = None, 
    method: str = "contour",  # "contour" または "floodfill"
    **kwargs
) -> str:
    """キャラクター画像からマスクを生成して保存
    
    Args:
        input_path: 入力画像パス
        output_path: 出力マスクパス（Noneの場合は自動生成）
        method: 抽出方法 ("contour" または "floodfill")
        **kwargs: 各抽出関数への追加パラメータ
    """
    # 抽出方法の選択
    if method == "contour":
        char_mask = extract_by_contour(input_path, **kwargs)
    elif method == "floodfill":
        char_mask = extract_character_alpha(input_path, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'contour' or 'floodfill'")
    
    # 出力パスの決定
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = f"{input_stem}_mask.png"
    
    # マスクを保存
    cv2.imwrite(output_path, char_mask)
    
    return output_path

def generate_all_body_masks(
    input_dir="generated_body", 
    output_dir="body_masks", 
    overwrite=False,
    method="contour",  # "contour" または "floodfill"
    **extract_kwargs
):
    """generated_body内の全キャラクター画像にマスクを生成
    
    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        overwrite: 既存ファイルを上書きするか
        method: 抽出方法 ("contour" または "floodfill")
        **extract_kwargs: 抽出関数への追加パラメータ
    """
    
    # ディレクトリ存在チェック
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 対象ファイル一覧取得
    image_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(filename)
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} character images in {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # 処理統計
    processed = 0
    skipped = 0
    errors = 0
    
    # 各画像を処理
    for filename in tqdm(image_files, desc="Generating character masks"):
        input_path = os.path.join(input_dir, filename)
        stem = Path(filename).stem
        mask_filename = f"{stem}_mask.png"
        output_path = os.path.join(output_dir, mask_filename)
        
        # 既存ファイルのスキップ判定
        if os.path.exists(output_path) and not overwrite:
            print(f"スキップ: {mask_filename} は既に存在します")
            skipped += 1
            continue
        
        try:
            # マスク生成（選択された方法で）
            if method == "contour":
                char_mask = extract_by_contour(input_path, **extract_kwargs)
            elif method == "floodfill":
                char_mask = extract_character_alpha(input_path, **extract_kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # マスクの品質チェック
            mask_area = np.sum(char_mask > 0)
            total_area = char_mask.shape[0] * char_mask.shape[1]
            coverage = mask_area / total_area
            
            if coverage < 0.01:  # マスクが小さすぎる場合
                print(f"警告: {filename} のマスクが小さすぎます (カバー率: {coverage:.3f})")
            elif coverage > 0.95:  # マスクが大きすぎる場合
                print(f"警告: {filename} のマスクが大きすぎます (カバー率: {coverage:.3f})")
            
            # マスク保存
            cv2.imwrite(output_path, char_mask)
            processed += 1
            
        except Exception as e:
            print(f"エラー: {filename} の処理に失敗しました - {str(e)}")
            errors += 1
    
    # 結果報告
    print(f"\n=== 処理結果 ===")
    print(f"処理完了: {processed} ファイル")
    print(f"スキップ: {skipped} ファイル")
    print(f"エラー: {errors} ファイル")
    print(f"総ファイル数: {len(image_files)} ファイル")
    
    if processed > 0:
        print(f"\nマスクファイルは {output_dir} に保存されました")

def main():
    """メイン実行関数"""
    print("=== キャラクター画像マスク生成ツール ===")
    print("使用方法: contour（輪郭ベース）またはfloodfill（塗りつぶしベース）")
    
    # 輪郭ベースの方法をデフォルトで使用
    print("\n輪郭ベースの方法でマスクを生成中...")
    generate_all_body_masks(
        input_dir="generated_body",
        output_dir="body_masks",
        overwrite=False,
        method="contour",
        morph_kernel=3,
        min_area=1000,
        keep_largest=True
    )
    
    print("\nマスク生成が完了しました！")
    print("輪郭ベースの方法を使用しました")
    print("設定: morph_kernel=3, min_area=1000, keep_largest=True")

def main_compare():
    """両方の方法で比較実行する関数"""
    print("=== マスク生成方法の比較実行 ===")
    
    # 輪郭ベース
    print("\n1. 輪郭ベースでマスク生成...")
    generate_all_body_masks(
        input_dir="generated_body",
        output_dir="body_masks_contour",
        overwrite=True,
        method="contour",
        morph_kernel=3,
        min_area=1000,
        keep_largest=True
    )
    
    # flood-fillベース
    print("\n2. flood-fillベースでマスク生成...")
    generate_all_body_masks(
        input_dir="generated_body",
        output_dir="body_masks_floodfill",
        overwrite=True,
        method="floodfill",
        white_thresh=200,
        dilate_iter=2,
        close_kernel=5,
        keep_largest=True
    )
    
    print("\n両方の方法でマスクを生成しました！")
    print("結果を比較してください:")
    print("- body_masks_contour/ (輪郭ベース)")
    print("- body_masks_floodfill/ (flood-fillベース)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        main_compare()
    else:
        main()