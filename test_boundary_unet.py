"""
Boundary-Aware U-Net テストスクリプト

学習済みモデルを使用して推論を行い、結果を可視化・評価する。

Usage:
    # テスト実行
    python test_boundary_unet.py --checkpoint models/best_model_gray.pth
    
    # カスタムパラメータ
    python test_boundary_unet.py --checkpoint models/best_model_lsd_sdf.pth --input_type lsd_sdf
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.boundary_aware_unet import (
    BoundaryAwareUNet,
    prepare_input_gray,
    prepare_input_lsd_sdf
)
from train_boundary_unet import BalloonDataset, calculate_iou


def visualize_result(image, mask_true, mask_pred, save_path):
    """
    結果を可視化して保存
    
    Args:
        image: 入力画像 (H, W) or (H, W, 3)
        mask_true: 正解マスク (H, W)
        mask_pred: 予測マスク (H, W)
        save_path: 保存先パス
    """
    # グレースケールをRGBに変換
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_rgb = image.copy()
    
    # マスクをカラーマップに変換
    mask_true_colored = cv2.applyColorMap((mask_true * 255).astype(np.uint8), cv2.COLORMAP_GREEN)
    mask_pred_colored = cv2.applyColorMap((mask_pred * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # オーバーレイ
    overlay_true = cv2.addWeighted(image_rgb, 0.6, mask_true_colored, 0.4, 0)
    overlay_pred = cv2.addWeighted(image_rgb, 0.6, mask_pred_colored, 0.4, 0)
    
    # 差分（TP: 緑, FP: 赤, FN: 青）
    tp = (mask_true > 0.5) & (mask_pred > 0.5)
    fp = (mask_true <= 0.5) & (mask_pred > 0.5)
    fn = (mask_true > 0.5) & (mask_pred <= 0.5)
    
    diff = np.zeros_like(image_rgb)
    diff[tp] = [0, 255, 0]  # 緑: True Positive
    diff[fp] = [0, 0, 255]  # 赤: False Positive
    diff[fn] = [255, 0, 0]  # 青: False Negative
    
    # 結合
    h, w = image.shape[:2]
    result = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
    
    # 1行目
    result[0:h, 0:w] = image_rgb
    result[0:h, w:w*2] = overlay_true
    result[0:h, w*2:w*3] = overlay_pred
    
    # 2行目
    result[h:h*2, 0:w] = (mask_true[:, :, np.newaxis] * 255).astype(np.uint8).repeat(3, axis=2)
    result[h:h*2, w:w*2] = (mask_pred[:, :, np.newaxis] * 255).astype(np.uint8).repeat(3, axis=2)
    result[h:h*2, w*2:w*3] = diff
    
    # テキスト追加
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Input', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(result, 'GT Overlay', (w+10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Pred Overlay', (w*2+10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(result, 'GT Mask', (10, h+30), font, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Pred Mask', (w+10, h+30), font, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Diff (TP/FP/FN)', (w*2+10, h+30), font, 1, (255, 255, 255), 2)
    
    cv2.imwrite(str(save_path), result)


def test_model(model, dataloader, device, output_dir, save_vis=True):
    """
    モデルをテスト
    """
    model.eval()
    
    all_ious = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = output_dir / 'visualizations'
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for idx, (images, masks, filenames) in enumerate(tqdm(dataloader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device)
            
            # 推論
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            # バッチ内の各画像を処理
            for i in range(images.size(0)):
                pred = preds[i, 0].cpu().numpy()
                mask = masks[i, 0].cpu().numpy()
                filename = filenames[i]
                
                # 2値化
                pred_binary = (pred > 0.5).astype(np.float32)
                
                # メトリクス計算
                iou = calculate_iou(
                    torch.from_numpy(pred).unsqueeze(0),
                    torch.from_numpy(mask).unsqueeze(0)
                )
                
                # Precision, Recall, F1
                tp = ((pred_binary == 1) & (mask == 1)).sum()
                fp = ((pred_binary == 1) & (mask == 0)).sum()
                fn = ((pred_binary == 0) & (mask == 1)).sum()
                
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
                
                all_ious.append(iou)
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1s.append(f1)
                
                # 可視化
                if save_vis:
                    # 入力画像を復元
                    if images.size(1) == 1:
                        img = images[i, 0].cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = images[i, 0].cpu().numpy()  # グレースケールチャンネル
                        img = (img * 255).astype(np.uint8)
                    
                    save_path = vis_dir / f"{Path(filename).stem}_result.png"
                    visualize_result(img, mask, pred_binary, save_path)
    
    # 統計
    metrics = {
        'mean_iou': np.mean(all_ious),
        'std_iou': np.std(all_ious),
        'mean_precision': np.mean(all_precisions),
        'mean_recall': np.mean(all_recalls),
        'mean_f1': np.mean(all_f1s),
    }
    
    return metrics, all_ious


def main(args):
    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # チェックポイント読み込み
    print(f"\n=== Loading Checkpoint ===")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 引数復元
    if 'args' in checkpoint:
        train_args = checkpoint['args']
        input_type = train_args.get('input_type', args.input_type)
        base_channels = train_args.get('base_channels', 64)
    else:
        input_type = args.input_type
        base_channels = 64
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input type: {input_type}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'val_iou' in checkpoint:
        print(f"Validation IoU: {checkpoint['val_iou']:.4f}")
    
    # モデル構築
    in_channels = 1 if input_type == 'gray' else 3
    model = BoundaryAwareUNet(
        in_channels=in_channels,
        n_classes=1,
        base_channels=base_channels
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # データセット
    print(f"\n=== Loading Test Dataset ===")
    test_dataset = BalloonDataset(
        args.test_images,
        args.test_masks,
        input_type=input_type
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # テスト実行
    print(f"\n=== Testing ===")
    metrics, all_ious = test_model(
        model, test_loader, device, 
        args.output_dir, 
        save_vis=args.save_vis
    )
    
    # 結果表示
    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"Mean IoU:       {metrics['mean_iou']:.4f} ± {metrics['std_iou']:.4f}")
    print(f"Mean Precision: {metrics['mean_precision']:.4f}")
    print(f"Mean Recall:    {metrics['mean_recall']:.4f}")
    print(f"Mean F1:        {metrics['mean_f1']:.4f}")
    print(f"{'='*60}")
    
    # 結果保存
    output_dir = Path(args.output_dir)
    with open(output_dir / 'test_results.txt', 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Input type: {input_type}\n")
        f.write(f"Test images: {args.test_images}\n")
        f.write(f"\n")
        f.write(f"Mean IoU:       {metrics['mean_iou']:.4f} ± {metrics['std_iou']:.4f}\n")
        f.write(f"Mean Precision: {metrics['mean_precision']:.4f}\n")
        f.write(f"Mean Recall:    {metrics['mean_recall']:.4f}\n")
        f.write(f"Mean F1:        {metrics['mean_f1']:.4f}\n")
        f.write(f"\n")
        f.write(f"Per-image IoU:\n")
        for i, iou in enumerate(all_ious):
            f.write(f"  {i+1}: {iou:.4f}\n")
    
    print(f"\nResults saved to {output_dir / 'test_results.txt'}")
    
    if args.save_vis:
        print(f"Visualizations saved to {output_dir / 'visualizations'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Boundary-Aware U-Net')
    
    # モデル
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input_type', type=str, default='gray', choices=['gray', 'lsd_sdf'],
                        help='Input type (used if not in checkpoint)')
    
    # データセット
    parser.add_argument('--test_images', type=str, default='balloon_dataset/syn200_dataset/images',
                        help='Test images directory')
    parser.add_argument('--test_masks', type=str, default='balloon_dataset/syn200_dataset/masks',
                        help='Test masks directory')
    
    # テストパラメータ
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # 出力
    parser.add_argument('--output_dir', type=str, default='test_results/boundary_unet',
                        help='Output directory')
    parser.add_argument('--save_vis', action='store_true', default=True,
                        help='Save visualization images')
    parser.add_argument('--no_vis', action='store_false', dest='save_vis',
                        help='Do not save visualization images')
    
    args = parser.parse_args()
    
    main(args)
