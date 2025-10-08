"""
シンプルなコマ検出のテストスクリプト
"""
import cv2
import numpy as np
import sys

def detect_panels_simple(image, area_ratio_threshold=0.85, min_area=10000):
    """
    シンプルな二値化・輪郭抽出によるコマ検出
    """
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二値化（Otsuの自動閾値）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ノイズ除去（モルフォロジー演算）
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    panels = []
    img_h, img_w = image.shape[:2]
    
    print(f"検出された輪郭数: {len(contours)}")
    
    for i, contour in enumerate(contours):
        # 輪郭の面積を計算
        contour_area = cv2.contourArea(contour)
        
        # 小さすぎる輪郭は無視
        if contour_area < min_area:
            continue
        
        # バウンディングボックスを取得
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        
        # 面積比を計算
        area_ratio = contour_area / bbox_area if bbox_area > 0 else 0
        
        print(f"輪郭 {i}: 面積={contour_area:.0f}, bbox面積={bbox_area}, 面積比={area_ratio:.3f}")
        
        # 面積比が閾値以上の場合、コマとして認識
        if area_ratio >= area_ratio_threshold:
            # パネルマスクを作成
            panel_mask = np.zeros((h, w), dtype=np.uint8)
            
            # 輪郭を相対座標に変換してマスクに描画
            contour_relative = contour - np.array([x, y])
            cv2.drawContours(panel_mask, [contour_relative], -1, 255, -1)
            
            panels.append((panel_mask, (x, y, w, h)))
            print(f"  → コマとして認識")
    
    return panels, binary


def visualize_panels(image, panels):
    """検出されたパネルを可視化"""
    result = image.copy()
    
    for i, (panel_mask, bbox) in enumerate(panels):
        x, y, w, h = bbox
        # バウンディングボックスを描画
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # ラベルを表示
        cv2.putText(result, f"Panel {i+1}", (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用法: python test_panel_detection.py <画像パス>")
        print("例: python test_panel_detection.py generated_double_backs/001.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"エラー: 画像の読み込みに失敗 - {image_path}")
        sys.exit(1)
    
    print(f"\n画像サイズ: {image.shape[1]}x{image.shape[0]}")
    print("=" * 60)
    
    # コマ検出（デフォルトパラメータ）
    print("\n[デフォルトパラメータ: 面積比>=0.85, 最小面積=10000]")
    panels_default, binary = detect_panels_simple(image)
    print(f"\n検出されたコマ数: {len(panels_default)}\n")
    
    # 結果を可視化
    result_default = visualize_panels(image, panels_default)
    
    # 保存
    output_path = "test_panel_detection_result.png"
    cv2.imwrite(output_path, result_default)
    print(f"結果画像を保存: {output_path}")
    
    binary_path = "test_panel_detection_binary.png"
    cv2.imwrite(binary_path, binary)
    print(f"二値化画像を保存: {binary_path}")
    
    # より緩い条件でもテスト
    print("\n" + "=" * 60)
    print("[緩いパラメータ: 面積比>=0.75, 最小面積=5000]")
    panels_loose, _ = detect_panels_simple(image, area_ratio_threshold=0.75, min_area=5000)
    print(f"\n検出されたコマ数: {len(panels_loose)}\n")
    
    result_loose = visualize_panels(image, panels_loose)
    output_loose = "test_panel_detection_result_loose.png"
    cv2.imwrite(output_loose, result_loose)
    print(f"結果画像を保存（緩い条件）: {output_loose}")
