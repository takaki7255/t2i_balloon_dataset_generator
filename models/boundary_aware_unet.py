"""
Boundary-Aware U-Net implementation for semantic segmentation

境界を意識した損失関数（BCE + Dice + Boundary Aware）を使用するU-Net。
グレースケール1チャンネル入力、または線分検出(LSD) + SDF入力に対応。

Architecture:
    - Encoder: Standard U-Net encoder with max pooling
    - Decoder: U-Net style upsampling with skip connections
    - Output: Single channel segmentation mask

Loss Functions:
    - BCE (Binary Cross Entropy)
    - Dice Loss
    - Boundary Loss (境界領域を強調)

Usage:
    # グレースケール1チャンネル入力
    model = BoundaryAwareUNet(in_channels=1, n_classes=1)
    
    # LSD + SDF (3チャンネル入力)
    model = BoundaryAwareUNet(in_channels=3, n_classes=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


class DoubleConv(nn.Module):
    """
    畳み込みブロック: (Conv -> BN -> ReLU) x 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    ダウンサンプリングブロック: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    アップサンプリングブロック: UpConv -> Concat -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # パディング調整（入力サイズが2のべき乗でない場合）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # スキップ接続
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BoundaryAwareUNet(nn.Module):
    """
    Boundary-Aware U-Net
    """
    def __init__(self, in_channels=1, n_classes=1, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        
        # Output
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
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
    
    Args:
        image: numpy array (H, W, 3) or (H, W)
    
    Returns:
        torch.Tensor (1, H, W)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 正規化 [0, 1]
    gray = gray.astype(np.float32) / 255.0
    
    return torch.from_numpy(gray).unsqueeze(0)


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
    print("=== Boundary-Aware U-Net Test ===")
    
    # グレースケールモデル
    model_gray = BoundaryAwareUNet(in_channels=1, n_classes=1)
    x_gray = torch.randn(2, 1, 256, 256)
    output = model_gray(x_gray)
    print(f"Gray Input: {x_gray.shape} -> Output: {output.shape}")
    
    # LSD+SDFモデル
    model_lsd = BoundaryAwareUNet(in_channels=3, n_classes=1)
    x_lsd = torch.randn(2, 3, 256, 256)
    output = model_lsd(x_lsd)
    print(f"LSD+SDF Input: {x_lsd.shape} -> Output: {output.shape}")
    
    # 損失関数テスト
    criterion = CombinedLoss()
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    loss, loss_dict = criterion(output, target)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    # パラメータ数
    total_params = sum(p.numel() for p in model_gray.parameters())
    print(f"\nTotal parameters: {total_params:,}")
