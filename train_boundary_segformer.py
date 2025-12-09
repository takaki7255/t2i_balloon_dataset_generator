"""
Boundary-Aware SegFormer 学習スクリプト

グレースケール1ch入力、またはLSD+SDF+Gray 3ch入力に対応した学習コード。
BCE + Dice + Boundary Aware Lossを使用。

Usage:
    # グレースケール入力で学習（内部で3chに複製される）
    python train_boundary_segformer.py --input_type gray
    
    # LSD+SDF+Gray入力で学習
    python train_boundary_segformer.py --input_type lsd_sdf
    
    # カスタムパラメータ
    python train_boundary_segformer.py --input_type gray --epochs 100 --batch_size 4 --lr 0.00006
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm

from models.boundary_aware_segformer import (
    BoundaryAwareSegFormer, 
    CombinedLoss,
    prepare_input_gray,
    prepare_input_lsd_sdf
)


class BalloonDataset(Dataset):
    """
    吹き出しセグメンテーションデータセット
    """
    def __init__(self, image_dir, mask_dir, input_type='gray', target_size=(384, 512), transform=None):
        """
        Args:
            image_dir: 画像ディレクトリ
            mask_dir: マスクディレクトリ
            input_type: 'gray' or 'lsd_sdf'
            target_size: (H, W) リサイズ後のサイズ
            transform: データ拡張（未実装）
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.input_type = input_type
        self.target_size = target_size
        self.transform = transform
        
        # 画像ファイルリスト
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + 
                                  list(self.image_dir.glob('*.jpg')))
        
        print(f"Dataset: {len(self.image_files)} images found in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 画像読み込み
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # マスク読み込み
        mask_filename = img_path.stem + '_mask.png'
        mask_path = self.mask_dir / mask_filename
        
        if not mask_path.exists():
            # _mask.pngがない場合、同名ファイルを探す
            mask_path = self.mask_dir / img_path.name
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # リサイズ
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]))
        
        # マスクを0/1に正規化
        mask = (mask > 127).astype(np.float32)
        
        # 入力準備
        if self.input_type == 'gray':
            input_tensor = prepare_input_gray(image)
        elif self.input_type == 'lsd_sdf':
            input_tensor = prepare_input_lsd_sdf(image, mask)
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")
        
        # マスクをテンソルに変換
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return input_tensor, mask_tensor, str(img_path.name)


def calculate_iou(pred, target, threshold=0.5):
    """
    IoU (Intersection over Union) を計算
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    1エポック学習
    """
    model.train()
    
    epoch_loss = 0.0
    epoch_bce = 0.0
    epoch_dice = 0.0
    epoch_boundary = 0.0
    epoch_iou = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, masks, _) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        
        # Loss計算
        loss, loss_dict = criterion(outputs, masks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # メトリクス計算
        with torch.no_grad():
            pred = torch.sigmoid(outputs)
            batch_iou = calculate_iou(pred, masks)
        
        # 累積
        epoch_loss += loss_dict['total']
        epoch_bce += loss_dict['bce']
        epoch_dice += loss_dict['dice']
        epoch_boundary += loss_dict['boundary']
        epoch_iou += batch_iou
        
        # プログレスバー更新
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'iou': f"{batch_iou:.4f}"
        })
    
    # 平均値
    num_batches = len(dataloader)
    metrics = {
        'loss': epoch_loss / num_batches,
        'bce': epoch_bce / num_batches,
        'dice': epoch_dice / num_batches,
        'boundary': epoch_boundary / num_batches,
        'iou': epoch_iou / num_batches
    }
    
    return metrics


def validate(model, dataloader, criterion, device):
    """
    検証
    """
    model.eval()
    
    val_loss = 0.0
    val_iou = 0.0
    
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            
            # Loss計算
            loss, loss_dict = criterion(outputs, masks)
            
            # IoU計算
            pred = torch.sigmoid(outputs)
            batch_iou = calculate_iou(pred, masks)
            
            val_loss += loss_dict['total']
            val_iou += batch_iou
    
    num_batches = len(dataloader)
    metrics = {
        'loss': val_loss / num_batches,
        'iou': val_iou / num_batches
    }
    
    return metrics


def main(args):
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # データセット
    print("\n=== Loading Dataset ===")
    train_dataset = BalloonDataset(
        args.train_images,
        args.train_masks,
        input_type=args.input_type,
        target_size=(args.height, args.width)
    )
    
    val_dataset = BalloonDataset(
        args.val_images,
        args.val_masks,
        input_type=args.input_type,
        target_size=(args.height, args.width)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # モデル
    print("\n=== Building Model ===")
    in_channels = 1 if args.input_type == 'gray' else 3
    model = BoundaryAwareSegFormer(
        in_channels=in_channels,
        num_classes=1,
        backbone=args.backbone,
        pretrained=args.pretrained
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Boundary-Aware SegFormer ({args.backbone})")
    print(f"Input type: {args.input_type} ({in_channels} channels)")
    print(f"Input size: {args.height}x{args.width}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 損失関数
    criterion = CombinedLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        boundary_weight=args.boundary_weight
    )
    
    # オプティマイザ
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # スケジューラ (PyTorch 2.2+ では verbose パラメータは非推奨)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 出力ディレクトリ
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 学習ループ
    print("\n=== Training Start ===")
    best_val_iou = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # 学習
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | "
              f"BCE: {train_metrics['bce']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f} | "
              f"Boundary: {train_metrics['boundary']:.4f} | "
              f"IoU: {train_metrics['iou']:.4f}")
        
        # 検証
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"[Val]   Loss: {val_metrics['loss']:.4f} | "
              f"IoU: {val_metrics['iou']:.4f}")
        
        # スケジューラ更新
        scheduler.step(val_metrics['loss'])
        
        # ベストモデル保存
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_metrics['iou'],
                'val_loss': val_metrics['loss'],
                'args': vars(args)
            }, output_dir / f'best_segformer_{args.input_type}.pth')
            print(f"✓ Best model saved! (IoU: {best_val_iou:.4f})")
        
        # 定期保存
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)
            }, output_dir / f'segformer_epoch_{epoch}_{args.input_type}.pth')
    
    print("\n=== Training Complete ===")
    print(f"Best validation IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Boundary-Aware SegFormer')
    
    # データセット
    parser.add_argument('--train_images', type=str, default='balloon_dataset/syn1000_dataset/images',
                        help='Training images directory')
    parser.add_argument('--train_masks', type=str, default='balloon_dataset/syn1000_dataset/masks',
                        help='Training masks directory')
    parser.add_argument('--val_images', type=str, default='balloon_dataset/syn200_dataset/images',
                        help='Validation images directory')
    parser.add_argument('--val_masks', type=str, default='balloon_dataset/syn200_dataset/masks',
                        help='Validation masks directory')
    
    # モデル
    parser.add_argument('--input_type', type=str, default='gray', choices=['gray', 'lsd_sdf'],
                        help='Input type: gray (1ch) or lsd_sdf (3ch)')
    parser.add_argument('--backbone', type=str, default='nvidia/mit-b1',
                        help='SegFormer backbone (nvidia/mit-b0, nvidia/mit-b1, etc.)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet pretrained model')
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained',
                        help='Do not use pretrained model')
    
    # 入力サイズ
    parser.add_argument('--height', type=int, default=384,
                        help='Input height')
    parser.add_argument('--width', type=int, default=512,
                        help='Input width')
    
    # 損失関数
    parser.add_argument('--bce_weight', type=float, default=1.0,
                        help='BCE loss weight')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='Dice loss weight')
    parser.add_argument('--boundary_weight', type=float, default=1.0,
                        help='Boundary loss weight')
    
    # 学習パラメータ
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00006,
                        help='Learning rate (SegFormer推奨: 6e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    
    # その他
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    main(args)
