"""
DeepLabv3+ implementation for semantic segmentation

DeepLabv3+は以下の特徴を持つセマンティックセグメンテーションモデル：
- Atrous Spatial Pyramid Pooling (ASPP)
- エンコーダー・デコーダー構造
- 低レベル特徴との結合によるリファインメント

Architecture:
    - Encoder: ResNet50/ResNet101 with atrous convolutions
    - ASPP: Multiple dilated convolutions for multi-scale context
    - Decoder: Lightweight decoder with skip connections
    - Output: Single channel segmentation mask

Usage:
    # ResNet50 with pretrained weights, output_stride=16
    model = DeepLabv3Plus(backbone='resnet50', pretrained=True, output_stride=16, n_classes=1)
    
    # ResNet101 without pretrained, output_stride=8
    model = DeepLabv3Plus(backbone='resnet101', pretrained=False, output_stride=8, n_classes=1)

References:
    - Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
      Chen et al., ECCV 2018
    - https://arxiv.org/abs/1802.02611
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module
    
    複数のdilation率を持つ畳み込みでマルチスケールな文脈情報を取得
    
    Args:
        in_channels: 入力チャネル数
        out_channels: 出力チャネル数
        atrous_rates: dilation率のリスト (default: [6, 12, 18])
    """
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super().__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions with different rates
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        
        # Projection layer (concatenated features -> output_channels)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        
        # Global pooling branchを元のサイズに戻す
        res[-1] = F.interpolate(res[-1], size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate all branches
        res = torch.cat(res, dim=1)
        
        return self.project(res)


class DeepLabv3PlusDecoder(nn.Module):
    """
    DeepLabv3+ Decoder
    
    ASPP出力と低レベル特徴を結合してリファインメント
    
    Args:
        low_level_channels: 低レベル特徴のチャネル数
        aspp_channels: ASPP出力のチャネル数
        out_channels: 出力チャネル数
        n_classes: 分類クラス数
    """
    def __init__(self, low_level_channels, aspp_channels=256, out_channels=256, n_classes=1):
        super().__init__()
        
        # 低レベル特徴の次元削減
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder convolutions
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Output layer
        self.out_conv = nn.Conv2d(out_channels, n_classes, 1)
    
    def forward(self, x, low_level_feat):
        # 低レベル特徴を処理
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # ASPP出力を低レベル特徴のサイズにアップサンプル
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate
        x = torch.cat([x, low_level_feat], dim=1)
        
        # Decoder
        x = self.decoder(x)
        
        # Output
        x = self.out_conv(x)
        
        return x


class DeepLabv3Plus(nn.Module):
    """
    DeepLabv3+ for semantic segmentation
    
    Args:
        backbone (str): バックボーン ('resnet50' or 'resnet101')
        pretrained (bool): ImageNet pretrainedを使用するか
        output_stride (int): 出力ストライド (8 or 16)
        n_classes (int): 出力クラス数（デフォルト1: バイナリセグメンテーション）
        freeze_backbone (bool): バックボーンを固定するか（転移学習用）
        aspp_rates (list): ASPP のdilation率リスト
    """
    def __init__(self, backbone='resnet50', pretrained=True, output_stride=16, 
                 n_classes=1, freeze_backbone=False, aspp_rates=[6, 12, 18]):
        super().__init__()
        
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.output_stride = output_stride
        self.n_classes = n_classes
        
        # output_strideの検証
        if output_stride not in [8, 16]:
            raise ValueError(f"output_stride must be 8 or 16, got {output_stride}")
        
        # ResNetバックボーンを読み込み
        if backbone == 'resnet50':
            if pretrained:
                try:
                    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                except:
                    resnet = models.resnet50(pretrained=True)
            else:
                resnet = models.resnet50(pretrained=False)
            low_level_channels = 256
            high_level_channels = 2048
            
        elif backbone == 'resnet101':
            if pretrained:
                try:
                    resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
                except:
                    resnet = models.resnet101(pretrained=True)
            else:
                resnet = models.resnet101(pretrained=False)
            low_level_channels = 256
            high_level_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet50' or 'resnet101'")
        
        # エンコーダー（ResNetの各ステージ）
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 低レベル特徴（1/4 resolution）
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        
        # Atrous convolutionの適用（output_strideに応じて）
        if output_stride == 16:
            # layer4にdilation=2を適用
            self._modify_layer_dilation(self.layer4, dilation=2)
        elif output_stride == 8:
            # layer3にdilation=2、layer4にdilation=4を適用
            self._modify_layer_dilation(self.layer3, dilation=2)
            self._modify_layer_dilation(self.layer4, dilation=4)
        
        # バックボーンを固定する場合
        if freeze_backbone:
            for param in [self.conv1, self.bn1, self.layer1, 
                         self.layer2, self.layer3, self.layer4]:
                for p in param.parameters():
                    p.requires_grad = False
        
        # ASPP
        self.aspp = ASPP(high_level_channels, out_channels=256, atrous_rates=aspp_rates)
        
        # Decoder
        self.decoder = DeepLabv3PlusDecoder(
            low_level_channels=low_level_channels,
            aspp_channels=256,
            out_channels=256,
            n_classes=n_classes
        )
        
        # モデル情報を表示
        print(f"DeepLabv3+ initialized:")
        print(f"  - Backbone: {backbone}")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Output stride: {output_stride}")
        print(f"  - Output classes: {n_classes}")
        print(f"  - Freeze backbone: {freeze_backbone}")
        print(f"  - ASPP rates: {aspp_rates}")
    
    def _modify_layer_dilation(self, layer, dilation):
        """
        ResNetのlayerにatrous convolutionを適用
        
        Args:
            layer: ResNetのlayer (layer3 or layer4)
            dilation: dilation率
        """
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                # stride=2の畳み込みをstride=1に変更
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                # dilationを設定
                if module.kernel_size == (3, 3):
                    module.dilation = (dilation, dilation)
                    module.padding = (dilation, dilation)
    
    def forward(self, x):
        input_shape = x.shape[2:]  # 元の入力サイズを保存
        
        # Encoder (ResNet backbone)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 低レベル特徴を保存 (1/4 resolution)
        low_level_feat = self.layer1(x)
        
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        x = self.decoder(x, low_level_feat)
        
        # 元の入力サイズにアップサンプル
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x
    
    def get_model_info(self):
        """モデル情報を返す"""
        return {
            'backbone': self.backbone_name,
            'pretrained': self.pretrained,
            'output_stride': self.output_stride,
            'n_classes': self.n_classes,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# テスト用コード
if __name__ == "__main__":
    print("="*60)
    print("DeepLabv3+ Model Test")
    print("="*60)
    
    # ResNet50 with pretrained, output_stride=16
    print("\n1. ResNet50 with pretrained weights (output_stride=16):")
    model_r50_s16 = DeepLabv3Plus(backbone='resnet50', pretrained=True, 
                                   output_stride=16, n_classes=1)
    info = model_r50_s16.get_model_info()
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    
    # ResNet50 without pretrained, output_stride=8
    print("\n2. ResNet50 without pretrained (output_stride=8):")
    model_r50_s8 = DeepLabv3Plus(backbone='resnet50', pretrained=False, 
                                  output_stride=8, n_classes=1)
    info = model_r50_s8.get_model_info()
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    
    # ResNet101 with pretrained
    print("\n3. ResNet101 with pretrained weights (output_stride=16):")
    model_r101 = DeepLabv3Plus(backbone='resnet101', pretrained=True, 
                                output_stride=16, n_classes=1)
    info = model_r101.get_model_info()
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    
    # Forward pass test
    print("\n4. Forward pass test:")
    x = torch.randn(2, 3, 384, 512)  # (B, C, H, W)
    with torch.no_grad():
        out = model_r50_s16(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Memory estimation
    print("\n5. Memory estimation (approximate):")
    print(f"ResNet50 (stride=16): ~{info['total_params'] * 4 / 1e6:.1f} MB (FP32)")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)
