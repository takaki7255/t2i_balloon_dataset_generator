import os, cv2, torch, numpy as np
from pathlib import Path
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from typing import List
from tqdm import tqdm

# --------------------------- デバイス判定 -------------------------------- #
def get_device():
    if torch.version.cuda is not None:
        return "cuda"
    return "cpu"

# ------------------------ SAM2 MaskGenerator 初期化 ----------------------- #
def build_mask_generator(points_per_side: int = 16,
                         min_area: int = 800):
    device = get_device()  # "cpu", "mps", or "cuda"
    mg = SAM2AutomaticMaskGenerator.from_pretrained(
        "facebook/sam2-hiera-base-plus",
        device=device,                    # ← ここが必須
        points_per_side=points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=min_area,
    )
    mg.device_type = device              # 後段で参照するため覚えておく
    print(f"[SAM2] device = {device}")
    return mg

# ----------------------- Balloon Mask 選択 Heuristic ---------------------- #
def select_balloon_mask(masks: List[dict], image: np.ndarray) -> np.ndarray:
    hsv_v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    best, best_score = None, -1
    img_area = image.shape[0] * image.shape[1]
    for m in masks:
        mask = m["segmentation"]
        area = mask.sum()
        if area < 800:
            continue
        mean_v = hsv_v[mask].mean() / 255.0
        ys, xs = np.where(mask)
        if len(xs) < 10:
            continue
        w, h = xs.max() - xs.min(), ys.max() - ys.min()
        ellipse_ratio = min(w, h) / max(w, h)
        score = 0.6 * (area / img_area) + 0.3 * mean_v + 0.1 * ellipse_ratio
        if score > best_score:
            best, best_score = mask, score
    if best is None:
        raise RuntimeError("Balloon mask not found")
    return (best.astype(np.uint8) * 255)

# ----------------------- 後処理: 白黒反転→輪郭検出→塗りつぶし ------------------- #
def post_process_mask(mask: np.ndarray) -> np.ndarray:
    """
    SAM2マスクに対して以下の処理を実行:
    1. 白黒反転
    2. 輪郭検出
    3. 輪郭内側を全て白で塗りつぶし
    """
    # 入力マスクの型と形状を確認・修正
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # uint8型に変換
    mask = mask.astype(np.uint8)
    
    # 1. 白黒反転
    inverted_mask = 255 - mask
    
    # 2. 輪郭検出（外側輪郭のみ）
    contours, _ = cv2.findContours(
        inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        # 輪郭が見つからない場合は元のマスクを返す
        return mask
    
    # 3. 最大面積の輪郭を選択
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 4. 新しいマスクを作成し、輪郭内側を白で塗りつぶし
    # 明示的にuint8のグレースケール配列を作成
    processed_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.drawContours(processed_mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)
    
    return processed_mask

# -------------------------- 1 枚処理関数 ---------------------------------- #
def generate_mask_sam2(img_path: str, mask_path: str, processed_path: str,
                       mask_generator: SAM2AutomaticMaskGenerator):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    dev = mask_generator.device_type
    # autocast は GPU/MPS 時のみ
    autocast_ctx = (torch.cuda.amp.autocast if dev == "cuda"
                    else (torch.autocast if dev == "mps" else None))

    if autocast_ctx is None:
        masks = mask_generator.generate(img)
    else:
        with torch.inference_mode(), autocast_ctx(device_type=dev, dtype=torch.bfloat16):
            masks = mask_generator.generate(img)

    mask = select_balloon_mask(masks, img)
    cv2.imwrite(mask_path, mask)
    
    # 追加処理: 白黒反転 → 輪郭検出 → 内側塗りつぶし
    processed_mask = post_process_mask(mask)
    cv2.imwrite(processed_path, processed_mask)

# ------------------------ 全ファイルループ -------------------------------- #
def main():
    balloons_dir = Path("bodies")
    masks_dir    = Path("masks_sam2")
    processed_dir = Path("body_masks")  # 処理済みマスク用ディレクトリ
    masks_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)

    gen = build_mask_generator()     # 1 回だけ初期化

    imgs = [p for p in balloons_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    for img_path in tqdm(imgs, desc="SAM2 mask gen"):
        out_path = masks_dir / f"{img_path.stem}_mask.png"
        processed_path = processed_dir / f"{img_path.stem}_processed_mask.png"
        
        if out_path.exists() and processed_path.exists():
            continue
        try:
            generate_mask_sam2(str(img_path), str(out_path), str(processed_path), gen)
        except Exception as e:
            print(f"✗ {img_path.name}: {e}")

if __name__ == "__main__":
    main()
