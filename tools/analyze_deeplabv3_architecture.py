"""
DeepLab v3+ データフロー解析ツール
モデルの各段階での特徴マップサイズを表示
"""

import torch
import torch.nn as nn
from train_deeplabv3_split import DeepLabV3Plus

def analyze_deeplabv3_dataflow():
    """DeepLab v3+のデータフローを詳細に解析"""
    
    print("🔍 DeepLab v3+ データフロー解析")
    print("=" * 60)
    
    # モデル作成
    model = DeepLabV3Plus(num_classes=1, backbone='resnet50', output_stride=16)
    model.eval()
    
    # 入力データ (バッチサイズ1, RGB, 512x512)
    input_tensor = torch.randn(1, 3, 512, 512)
    print(f"📥 入力サイズ: {list(input_tensor.shape)}")
    
    print("\n🏗️ Encoder (ResNet Backbone)")
    print("-" * 40)
    
    # Backbone through layers
    x = input_tensor
    
    # Initial convolution + pooling
    x = model.backbone.conv1(x)
    x = model.backbone.bn1(x)
    x = model.backbone.relu(x)
    print(f"  Conv1 + BN + ReLU: {list(x.shape)}")
    
    x = model.backbone.maxpool(x)
    print(f"  MaxPool: {list(x.shape)}")
    
    # ResNet layers
    low_level_feat = model.backbone.layer1(x)
    print(f"  Layer1 (low-level): {list(low_level_feat.shape)}")
    
    x = model.backbone.layer2(low_level_feat)
    print(f"  Layer2: {list(x.shape)}")
    
    x = model.backbone.layer3(x)
    print(f"  Layer3: {list(x.shape)}")
    
    features = model.backbone.layer4(x)
    print(f"  Layer4 (high-level): {list(features.shape)}")
    
    print(f"\n🔥 ASPP (Atrous Spatial Pyramid Pooling)")
    print("-" * 40)
    
    # ASPP processing
    aspp_output = model.aspp(features)
    print(f"  ASPP出力: {list(aspp_output.shape)}")
    
    print(f"\n🔄 Decoder")
    print("-" * 40)
    
    # Low-level feature projection
    low_level_projected = model.decoder.low_level_project(low_level_feat)
    print(f"  Low-level投影: {list(low_level_projected.shape)}")
    
    # Upsample ASPP output
    import torch.nn.functional as F
    aspp_upsampled = F.interpolate(aspp_output, size=low_level_projected.shape[2:], 
                                  mode='bilinear', align_corners=False)
    print(f"  ASPPアップサンプル: {list(aspp_upsampled.shape)}")
    
    # Concatenate
    concatenated = torch.cat((aspp_upsampled, low_level_projected), dim=1)
    print(f"  結合後: {list(concatenated.shape)}")
    
    # Decoder convolutions
    x = model.decoder.conv1(concatenated)
    print(f"  Decoder Conv1: {list(x.shape)}")
    
    x = model.decoder.conv2(x)
    print(f"  Decoder Conv2: {list(x.shape)}")
    
    x = model.decoder.classifier(x)
    print(f"  分類層: {list(x.shape)}")
    
    # Final upsampling
    final_output = F.interpolate(x, size=input_tensor.shape[2:], 
                                mode='bilinear', align_corners=False)
    print(f"  最終出力: {list(final_output.shape)}")
    
    print(f"\n📊 解像度変化サマリー")
    print("-" * 40)
    print(f"  入力:        512×512")
    print(f"  Conv1後:     256×256 (stride=2)")
    print(f"  MaxPool後:   128×128 (stride=2)")
    print(f"  Layer1後:    128×128 (stride=1)")
    print(f"  Layer2後:     64×64  (stride=2)")
    print(f"  Layer3後:     32×32  (stride=2)")
    print(f"  Layer4後:     32×32  (stride=1, dilation=2)")
    print(f"  ASPP後:       32×32  (同じ)")
    print(f"  Decoder後:   128×128 (4倍アップサンプル)")
    print(f"  最終出力:    512×512 (4倍アップサンプル)")
    
    print(f"\n🎯 モデル特徴")
    print("-" * 40)
    print("✅ エンコーダー: ResNet50によるマルチスケール特徴抽出")
    print("✅ ASPP: 異なるdilation rate [6,12,18] でマルチスケール統合")
    print("✅ デコーダー: Low-level特徴との融合で境界精度向上")
    print("✅ Skip Connection: U-Netライクな詳細情報保持")
    print("✅ Atrous Convolution: パラメータ効率的な受容野拡大")

def compare_output_strides():
    """Output Stride 8 vs 16 の比較"""
    
    print(f"\n⚖️ Output Stride 比較")
    print("=" * 60)
    
    models = {
        "OS=16": DeepLabV3Plus(backbone='resnet50', output_stride=16),
        "OS=8":  DeepLabV3Plus(backbone='resnet50', output_stride=8)
    }
    
    input_tensor = torch.randn(1, 3, 512, 512)
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            features, low_level = model.backbone(input_tensor)
            print(f"{name}: Encoder出力 = {list(features.shape)}, Low-level = {list(low_level.shape)}")
    
    print(f"\n📈 Output Stride の影響:")
    print("  OS=16: 計算効率重視、メモリ使用量少")
    print("  OS=8:  高精度重視、計算量・メモリ使用量多")
    print("  推奨: 通常はOS=16、高精度が必要な場合OS=8")

def analyze_aspp_receptive_field():
    """ASPPの受容野解析"""
    
    print(f"\n🔬 ASPP受容野解析")
    print("=" * 60)
    
    # Dilation rates
    rates_os16 = [6, 12, 18]
    rates_os8 = [12, 24, 36]
    
    print("Output Stride = 16の場合:")
    for rate in rates_os16:
        receptive_field = 3 + 2 * (rate - 1)  # 3x3 conv with dilation
        print(f"  Dilation {rate:2d}: 受容野 {receptive_field:2d}×{receptive_field}")
    
    print("\nOutput Stride = 8の場合:")
    for rate in rates_os8:
        receptive_field = 3 + 2 * (rate - 1)
        print(f"  Dilation {rate:2d}: 受容野 {receptive_field:2d}×{receptive_field}")
    
    print("\n🎯 ASPPの効果:")
    print("  • マルチスケール情報を並列処理")
    print("  • 小さな物体から大きな物体まで対応")
    print("  • Global Average Poolingで画像全体の文脈も取得")

if __name__ == "__main__":
    analyze_deeplabv3_dataflow()
    compare_output_strides()
    analyze_aspp_receptive_field()
