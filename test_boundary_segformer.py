"""
Boundary-Aware SegFormer テストスクリプト

学習済みモデルを使用して推論を行い、結果を可視化・評価する。

Usage:
    # テスト実行
    python test_boundary_segformer.py --checkpoint models/best_segformer_gray.pth
    
    # カスタムパラメータ
    python test_boundary_segformer.py --checkpoint models/best_segformer_lsd_sdf.pth --input_type lsd_sdf
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.boundary_aware_segformer import (
    BoundaryAwareSegFormer,
    prepare_input_gray,
    prepare_input_lsd_sdf
)
from train_boundary_segformer import BalloonDataset, calculate_iou


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
    
    # マスクをカラーマップに変換 (COLORMAP_SUMMER は緑系)
    mask_true_colored = cv2.applyColorMap((mask_true * 255).astype(np.uint8), cv2.COLORMAP_SUMMER)
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
    モデルをテスト（test_unet.pyと同様のメトリクス出力）
    """
    model.eval()
    
    all_dices = []
    all_ious = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    # Global metrics用
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    
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
                
                # TP, FP, FN, TN
                tp = ((pred_binary == 1) & (mask > 0.5)).sum()
                fp = ((pred_binary == 1) & (mask <= 0.5)).sum()
                fn = ((pred_binary == 0) & (mask > 0.5)).sum()
                tn = ((pred_binary == 0) & (mask <= 0.5)).sum()
                
                # Global用に累積
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn
                
                # Per-image metrics
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
                iou = tp / (tp + fp + fn + 1e-6)
                dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
                
                all_dices.append(float(dice))
                all_ious.append(float(iou))
                all_precisions.append(float(precision))
                all_recalls.append(float(recall))
                all_f1s.append(float(f1))
                
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
    
    # Global metrics計算
    global_precision = total_tp / (total_tp + total_fp + 1e-6)
    global_recall = total_tp / (total_tp + total_fn + 1e-6)
    global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall + 1e-6)
    global_iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    global_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-6)
    global_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-6)
    
    # 結果をまとめる（test_unet.pyと同形式）
    metrics = {
        # Average metrics
        'avg_dice': float(np.mean(all_dices)),
        'avg_iou': float(np.mean(all_ious)),
        'avg_precision': float(np.mean(all_precisions)),
        'avg_recall': float(np.mean(all_recalls)),
        'avg_f1': float(np.mean(all_f1s)),
        # Global metrics
        'global_dice': float(global_dice),
        'global_iou': float(global_iou),
        'global_precision': float(global_precision),
        'global_recall': float(global_recall),
        'global_f1': float(global_f1),
        'accuracy': float(global_accuracy),
        # Individual metrics for std
        'individual_dice': all_dices,
        'individual_iou': all_ious,
        'individual_precision': all_precisions,
        'individual_recall': all_recalls,
        'individual_f1': all_f1s
    }
    
    return metrics


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
        backbone = train_args.get('backbone', 'nvidia/mit-b1')
        height = train_args.get('height', 384)
        width = train_args.get('width', 512)
    else:
        input_type = args.input_type
        backbone = 'nvidia/mit-b1'
        height = 384
        width = 512
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backbone: {backbone}")
    print(f"Input type: {input_type}")
    print(f"Input size: {height}x{width}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'val_iou' in checkpoint:
        print(f"Validation IoU: {checkpoint['val_iou']:.4f}")
    
    # モデル構築
    in_channels = 1 if input_type == 'gray' else 3
    model = BoundaryAwareSegFormer(
        in_channels=in_channels,
        num_classes=1,
        backbone=backbone,
        pretrained=False  # テスト時は事前学習モデル不要
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # データセット
    print(f"\n=== Loading Test Dataset ===")
    test_dataset = BalloonDataset(
        args.test_images,
        args.test_masks,
        input_type=input_type,
        target_size=(height, width)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # テスト実行
    print(f"\n=== Testing ===")
    metrics = test_model(
        model, test_loader, device, 
        args.output_dir, 
        save_vis=args.save_vis
    )
    
    # 結果表示（test_unet.pyと同形式）
    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"Average Metrics (per image):")
    print(f"  Dice Score: {metrics['avg_dice']:.4f} ± {np.std(metrics['individual_dice']):.4f}")
    print(f"  IoU:        {metrics['avg_iou']:.4f} ± {np.std(metrics['individual_iou']):.4f}")
    print(f"  Precision:  {metrics['avg_precision']:.4f} ± {np.std(metrics['individual_precision']):.4f}")
    print(f"  Recall:     {metrics['avg_recall']:.4f} ± {np.std(metrics['individual_recall']):.4f}")
    print(f"  F1 Score:   {metrics['avg_f1']:.4f} ± {np.std(metrics['individual_f1']):.4f}")
    print(f"\nGlobal Metrics (all pixels):")
    print(f"  Dice Score: {metrics['global_dice']:.4f}")
    print(f"  IoU:        {metrics['global_iou']:.4f}")
    print(f"  Precision:  {metrics['global_precision']:.4f}")
    print(f"  Recall:     {metrics['global_recall']:.4f}")
    print(f"  F1 Score:   {metrics['global_f1']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"{'='*60}")
    
    # モデルタグを生成
    checkpoint_path = Path(args.checkpoint)
    model_tag = checkpoint_path.parent.name
    
    # 結果保存（JSON形式 - test_unet.pyと同形式）
    output_dir = Path(args.output_dir)
    results_json = {
        "model_tag": model_tag,
        "data_root": str(args.test_images),
        "img_size": [height, width],
        "batch_size": args.batch_size,
        "evaluation_time": datetime.now().isoformat(),
        "metrics": {
            "average_metrics": {
                "dice": metrics["avg_dice"],
                "iou": metrics["avg_iou"],
                "precision": metrics["avg_precision"],
                "recall": metrics["avg_recall"],
                "f1_score": metrics["avg_f1"]
            },
            "global_metrics": {
                "dice": metrics["global_dice"],
                "iou": metrics["global_iou"],
                "precision": metrics["global_precision"],
                "recall": metrics["global_recall"],
                "f1_score": metrics["global_f1"],
                "accuracy": metrics["accuracy"]
            }
        },
        "statistics": {
            "total_images": len(metrics["individual_dice"]),
            "dice_std": float(np.std(metrics["individual_dice"])),
            "iou_std": float(np.std(metrics["individual_iou"])),
            "precision_std": float(np.std(metrics["individual_precision"])),
            "recall_std": float(np.std(metrics["individual_recall"])),
            "f1_std": float(np.std(metrics["individual_f1"]))
        }
    }
    
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    # テキスト形式でも保存
    txt_path = output_dir / "evaluation_summary.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Model Evaluation Results\n")
        f.write(f"={'='*50}\n\n")
        f.write(f"Model: {model_tag}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Backbone: {backbone}\n")
        f.write(f"Input type: {input_type}\n")
        f.write(f"Dataset: {args.test_images}\n")
        f.write(f"Image Size: {height}x{width}\n")
        f.write(f"Total Images: {len(metrics['individual_dice'])}\n")
        f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Average Metrics (per image):\n")
        f.write(f"  Dice Score: {metrics['avg_dice']:.4f} ± {np.std(metrics['individual_dice']):.4f}\n")
        f.write(f"  IoU:        {metrics['avg_iou']:.4f} ± {np.std(metrics['individual_iou']):.4f}\n")
        f.write(f"  Precision:  {metrics['avg_precision']:.4f} ± {np.std(metrics['individual_precision']):.4f}\n")
        f.write(f"  Recall:     {metrics['avg_recall']:.4f} ± {np.std(metrics['individual_recall']):.4f}\n")
        f.write(f"  F1 Score:   {metrics['avg_f1']:.4f} ± {np.std(metrics['individual_f1']):.4f}\n\n")
        
        f.write(f"Global Metrics (all pixels):\n")
        f.write(f"  Dice Score: {metrics['global_dice']:.4f}\n")
        f.write(f"  IoU:        {metrics['global_iou']:.4f}\n")
        f.write(f"  Precision:  {metrics['global_precision']:.4f}\n")
        f.write(f"  Recall:     {metrics['global_recall']:.4f}\n")
        f.write(f"  F1 Score:   {metrics['global_f1']:.4f}\n")
        f.write(f"  Accuracy:   {metrics['accuracy']:.4f}\n")
    
    print(f"\n評価結果保存完了:")
    print(f"  JSON: {json_path}")
    print(f"  テキスト: {txt_path}")
    
    if args.save_vis:
        print(f"  Visualizations: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Boundary-Aware SegFormer')
    
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
    parser.add_argument('--output_dir', type=str, default='test_results/boundary_segformer',
                        help='Output directory')
    parser.add_argument('--save_vis', action='store_true', default=True,
                        help='Save visualization images')
    parser.add_argument('--no_vis', action='store_false', dest='save_vis',
                        help='Do not save visualization images')
    
    args = parser.parse_args()
    
    main(args)
