"""
Boundary-Aware SegFormer implementation for semantic segmentation

境界を意識した損失関数（BCE + Dice + Boundary Aware）を使用するSegFormer。
グレースケール1チャンネル入力、または線分検出(LSD) + SDF + グレースケール 3チャンネル入力に対応。

Architecture:
    - Encoder: MIT-B1 (Hierarchical Transformer)
    - Decoder: Lightweight All-MLP decoder
    - Pretrained: ImageNet-1k
    - Output: Single channel segmentation mask

Loss Functions:
    - BCE (Binary Cross Entropy)
    - Dice Loss
    - Boundary Loss (境界領域を強調)

Usage:
    # グレースケール1チャンネル入力
    model = BoundaryAwareSegFormer(in_channels=1, num_classes=1, backbone='mit_b1')
    
    # LSD + SDF + Gray (3チャンネル入力)
    model = BoundaryAwareSegFormer(in_channels=3, num_classes=1, backbone='mit_b1')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class BoundaryAwareSegFormer(nn.Module):
    """
    Boundary-Aware SegFormer
    
    Hugging Face transformersライブラリのSegFormerをベースに、
    カスタム入力チャンネル数と損失関数を使用。
    """
    def __init__(self, in_channels=3, num_classes=1, backbone='nvidia/mit-b1', pretrained=True):
        """
        Args:
            in_channels: 入力チャンネル数 (常に3を使用。gray入力の場合はprepare_input_grayで3chに複製)
            num_classes: 出力クラス数 (バイナリセグメンテーションなら1)
            backbone: SegFormerバックボーン ('nvidia/mit-b1', 'nvidia/mit-b0', etc.)
            pretrained: ImageNet事前学習モデルを使用するか
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        if pretrained:
            # 事前学習モデルをロード
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                backbone,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # ランダム初期化
            config = SegformerConfig.from_pretrained(backbone)
            config.num_labels = num_classes
            self.segformer = SegformerForSemanticSegmentation(config)
        
        # Note: in_channelsは常に3を想定。gray入力はprepare_input_grayで3chに複製される
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (B, C, H, W) 入力画像
        
        Returns:
            logits: (B, num_classes, H, W) 出力ロジット
        """
        outputs = self.segformer(pixel_values=x)
        logits = outputs.logits
        
        # 入力サイズにアップサンプリング
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return logits


class CombinedLoss(nn.Module):
    """
    Combined Loss: BCE + Dice + Boundary Loss
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, boundary_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target, smooth=1.0):
        """
        Dice Loss
        """
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    def boundary_loss(self, pred, target, boundary_width=3):
        """
        Boundary Loss: 境界領域を強調
        """
        pred = torch.sigmoid(pred)
        
        # 境界マスクを生成
        boundary_mask = self.get_boundary_mask(target, boundary_width)
        boundary_mask = boundary_mask.to(pred.device)
        
        # 境界領域でのBCE
        bce_boundary = F.binary_cross_entropy(pred, target, reduction='none')
        weighted_bce = bce_boundary * (1 + boundary_mask * 9)  # 境界を10倍重視
        
        return weighted_bce.mean()
    
    def get_boundary_mask(self, target, width=3):
        """
        境界マスクを生成（CPU上で処理）
        """
        target_np = target.cpu().numpy()
        batch_size = target_np.shape[0]
        boundary_masks = []
        
        for i in range(batch_size):
            mask = target_np[i, 0]  # (H, W)
            
            # エッジ検出
            kernel = np.ones((width, width), np.uint8)
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
            boundary = dilated - eroded
            
            boundary_masks.append(boundary)
        
        boundary_tensor = torch.from_numpy(np.array(boundary_masks)).unsqueeze(1).float()
        return boundary_tensor
    
    def forward(self, pred, target):
        """
        Combined loss calculation
        """
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        total_loss = (self.bce_weight * bce + 
                     self.dice_weight * dice + 
                     self.boundary_weight * boundary)
        
        return total_loss, {
            'bce': bce.item(),
            'dice': dice.item(),
            'boundary': boundary.item(),
            'total': total_loss.item()
        }


def extract_lsd_features(image):
    """
    線分検出（LSD）を使用して特徴抽出
    
    Args:
        image: numpy array (H, W) グレースケール画像
    
    Returns:
        numpy array (H, W) 線分密度マップ
    """
    # LSDで線分検出
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(image)[0]
    
    # 線分密度マップを作成
    line_map = np.zeros_like(image, dtype=np.float32)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            cv2.line(line_map, (x1, y1), (x2, y2), 255, 2)
    
    return line_map / 255.0


def compute_sdf(mask):
    """
    符号付き距離関数（SDF）を計算
    
    Args:
        mask: numpy array (H, W) バイナリマスク (0 or 1)
    
    Returns:
        numpy array (H, W) 正規化されたSDF
    """
    # 内側と外側の距離変換
    dist_inside = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(1 - mask)
    
    # SDF: 内側が正、外側が負
    sdf = dist_inside - dist_outside
    
    # 正規化 [-1, 1]
    max_dist = max(np.abs(sdf.min()), np.abs(sdf.max()))
    if max_dist > 0:
        sdf = sdf / max_dist
    
    return sdf


def prepare_input_gray(image):
    """
    グレースケール1チャンネル入力を準備
    グレースケール値を3チャンネルに複製してSegFormerの入力形式に合わせる
    
    Args:
        image: numpy array (H, W, 3) or (H, W)
    
    Returns:
        torch.Tensor (3, H, W)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 正規化 [0, 1]
    gray = gray.astype(np.float32) / 255.0
    
    # 3チャンネルに複製
    gray_3ch = np.stack([gray, gray, gray], axis=0)
    
    return torch.from_numpy(gray_3ch).float()


def prepare_input_lsd_sdf(image, mask=None):
    """
    LSD + SDF + グレースケールの3チャンネル入力を準備
    
    Args:
        image: numpy array (H, W, 3) or (H, W)
        mask: numpy array (H, W) バイナリマスク（学習時のみ使用）
    
    Returns:
        torch.Tensor (3, H, W)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # グレースケール
    gray_norm = gray.astype(np.float32) / 255.0
    
    # LSD特徴
    lsd_features = extract_lsd_features(gray)
    
    # SDF（マスクがある場合のみ）
    if mask is not None:
        sdf = compute_sdf(mask)
    else:
        sdf = np.zeros_like(gray, dtype=np.float32)
    
    # 3チャンネルに結合
    features = np.stack([gray_norm, lsd_features, sdf], axis=0)
    
    return torch.from_numpy(features).float()


if __name__ == "__main__":
    # テスト
    print("=== Boundary-Aware SegFormer Test ===")
    
    # グレースケールモデル
    print("\n[1/2] Testing gray (1ch) model...")
    model_gray = BoundaryAwareSegFormer(in_channels=1, num_classes=1, backbone='nvidia/mit-b1', pretrained=True)
    x_gray = torch.randn(2, 1, 384, 512)
    output = model_gray(x_gray)
    print(f"Gray Input: {x_gray.shape} -> Output: {output.shape}")
    
    # LSD+SDF+Grayモデル
    print("\n[2/2] Testing LSD+SDF+Gray (3ch) model...")
    model_lsd = BoundaryAwareSegFormer(in_channels=3, num_classes=1, backbone='nvidia/mit-b1', pretrained=True)
    x_lsd = torch.randn(2, 3, 384, 512)
    output = model_lsd(x_lsd)
    print(f"LSD+SDF+Gray Input: {x_lsd.shape} -> Output: {output.shape}")
    
    # 損失関数テスト
    print("\n[Loss] Testing combined loss...")
    criterion = CombinedLoss()
    target = torch.randint(0, 2, (2, 1, 384, 512)).float()
    loss, loss_dict = criterion(output, target)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    # パラメータ数
    total_params = sum(p.numel() for p in model_gray.parameters())
    trainable_params = sum(p.numel() for p in model_gray.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ All tests passed!")
