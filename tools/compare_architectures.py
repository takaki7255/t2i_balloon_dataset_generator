"""
U-Net vs DeepLab v3+ アーキテクチャ比較図
"""

def print_architecture_comparison():
    print("🏗️ U-Net vs DeepLab v3+ アーキテクチャ比較")
    print("=" * 80)
    
    print("\n📐 U-Net構造:")
    print("""
    入力(512x512)
        ↓ Conv+Pool
    256x256 ←─────────────────────┐ Skip Connection
        ↓ Conv+Pool              │
    128x128 ←─────────────────┐   │
        ↓ Conv+Pool          │   │
     64x64 ←─────────────┐   │   │
        ↓ Conv+Pool      │   │   │
     32x32              │   │   │
        ↓ Bottleneck    │   │   │
     32x32              │   │   │
        ↓ UpConv+Skip───┘   │   │
     64x64                  │   │
        ↓ UpConv+Skip───────┘   │
    128x128                     │
        ↓ UpConv+Skip───────────┘
    256x256
        ↓ UpConv
    出力(512x512)
    """)
    
    print("\n🔬 DeepLab v3+ 構造:")
    print("""
    入力(512x512)
        ↓ ResNet Encoder
    32x32 (high-level) ──→ ASPP ──→ Decoder ──┐
        │                    ↓              │
        │               マルチスケール         │
        │               [6,12,18]          │
        │                    ↓              │
    128x128 (low-level) ─────────────────→ 融合 ─→ 出力(512x512)
    """)
    
    print("\n📊 詳細比較:")
    print("-" * 80)
    print("| 特徴                | U-Net              | DeepLab v3+           |")
    print("|--------------------|--------------------|----------------------|")
    print("| Skip Connection    | 対称的な多層接続      | 1つのlow-level融合    |")
    print("| マルチスケール処理    | Encoder-Decoder     | ASPP                 |")
    print("| 受容野拡大          | プーリング          | Atrous Convolution    |")
    print("| パラメータ数        | 中程度             | やや多い              |")
    print("| 計算量             | 軽量               | 中程度                |")
    print("| 境界精度           | 良好               | より良好              |")
    print("| 小物体検出         | 普通               | 優秀                 |")
    print("| メモリ使用量        | 少ない             | 中程度                |")
    
    print("\n🎯 吹き出し検出での利点:")
    print("-" * 50)
    print("DeepLab v3+の優位性:")
    print("✅ ASPPによる様々なサイズの吹き出しへの対応")
    print("✅ Atrous convolutionによる細かい境界の保持")
    print("✅ Global Average Poolingによる文脈理解")
    print("✅ Low-level特徴融合による詳細な境界検出")
    
    print("\nU-Netの優位性:")
    print("✅ 軽量で高速な推論")
    print("✅ シンプルなアーキテクチャ")
    print("✅ 少ないメモリ使用量")
    print("✅ 実装が容易")

if __name__ == "__main__":
    print_architecture_comparison()
