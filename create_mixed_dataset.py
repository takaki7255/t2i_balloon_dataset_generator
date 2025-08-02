#!/usr/bin/env python3
"""
通常合成＋データ拡張の混合データセット生成
コマンドライン引数で枚数を指定可能

使用例:
  python create_mixed_dataset.py --normal 200 --augmented 300
  python create_mixed_dataset.py --normal 150 --augmented 150 --output my_dataset
"""

import argparse
import sys
import os

# 元のスクリプトから必要な部分をインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from create_syn_dataset_aug import *

def _json_default(o):
    """JSON serialization helper for numpy and Path objects"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)  # 最後の保険

def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='通常合成＋データ拡張の混合データセット生成')
    
    parser.add_argument('--normal', type=int, default=100,
                        help='通常合成（拡張なし）の枚数 (default: 100)')
    parser.add_argument('--augmented', type=int, default=100,
                        help='データ拡張適用の枚数 (default: 100)')
    parser.add_argument('--output', type=str, default='mixed_dataset',
                        help='出力ディレクトリ名 (default: mixed_dataset)')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='train/valの分割比率 (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='ランダムシード (default: 42)')
    
    # Augmentation設定
    parser.add_argument('--hflip-prob', type=float, default=0.5,
                        help='水平反転確率 (default: 0.5)')
    parser.add_argument('--rot-prob', type=float, default=1.0,
                        help='回転確率 (default: 1.0)')
    parser.add_argument('--cut-edge-prob', type=float, default=0.4,
                        help='端切り取り確率 (default: 0.4)')
    parser.add_argument('--thin-line-prob', type=float, default=0.5,
                        help='線細化確率 (default: 0.5)')
    
    # コマ角・辺の形状変更設定（新機能）
    parser.add_argument('--panel-corner-prob', type=float, default=0.3,
                        help='コマ角四角形化確率 (default: 0.3)')
    parser.add_argument('--panel-edge-prob', type=float, default=0.25,
                        help='コマ辺直線化確率 (default: 0.25)')
    
    # ディレクトリ設定
    parser.add_argument('--balloons-dir', type=str, default='generated_balloons',
                        help='吹き出し画像ディレクトリ (default: generated_balloons)')
    parser.add_argument('--masks-dir', type=str, default='masks',
                        help='マスクディレクトリ (default: masks)')
    parser.add_argument('--backgrounds-dir', type=str, default='generated_double_backs',
                        help='背景画像ディレクトリ (default: generated_double_backs)')
    
    return parser.parse_args()

def create_custom_config(args):
    """引数に基づいてCFG設定を作成"""
    total_images = args.normal + args.augmented
    
    CFG = {
        # ===== 基本設定 =====
        "TARGET_NORMAL_IMAGES": args.normal,
        "TARGET_AUGMENTED_IMAGES": args.augmented,
        "TARGET_TOTAL_IMAGES": total_images,
        "MAX_ATTEMPTS": 200,
        "BALLOON_SPLIT_RATIO": args.split_ratio,
        "SEED": args.seed,

        # ===== 個数（統計データに基づいて最適化） =====
        "COUNT_STATS_FILE": "balloon_count_statistics.txt",
        "COUNT_PROBS": None,
        "NUM_BALLOONS_RANGE": (9, 17),

        # ===== サイズ（実際の統計データに基づいて精密調整） =====
        "SCALE_MODE": "lognormal",
        "SCALE_MEAN": 0.105,
        "SCALE_STD": 0.025,
        "SCALE_CLIP": (0.065, 0.130),
        "SCALE_RANGE": (0.070, 0.120),
        
        # ===== 面積ベースリサイズ設定 =====
        "MAX_WIDTH_RATIO": 0.20,
        "MAX_HEIGHT_RATIO": 0.30,

        # ===== augmentation（コマンドライン引数から設定） =====
        "HFLIP_PROB": args.hflip_prob,
        "ROT_PROB": args.rot_prob,
        "ROT_RANGE": (-20, 20),
        "CUT_EDGE_PROB": args.cut_edge_prob,
        "CUT_EDGE_RATIO": (0.05, 0.20),
        "THIN_LINE_PROB": args.thin_line_prob,
        "THIN_PIXELS": (1, 3),
        
        # ===== コマ角・辺の形状変更（新機能） =====
        "PANEL_CORNER_PROB": args.panel_corner_prob,
        "PANEL_CORNER_RATIO": (0.05, 0.15),  # 角切り落としサイズ比率（5-15%）
        "PANEL_EDGE_PROB": args.panel_edge_prob,
        "PANEL_EDGE_RATIO": (0.30, 0.70),    # 辺直線化範囲比率（30-70%）
    }
    
    return CFG

def main_custom():
    """カスタマイズ可能なメイン関数"""
    args = parse_arguments()
    
    print("=== 混合データセット生成 ===")
    print(f"通常合成: {args.normal}枚")
    print(f"拡張適用: {args.augmented}枚")
    print(f"合計: {args.normal + args.augmented}枚")
    print(f"出力先: {args.output}")
    print(f"Train/Val比率: {args.split_ratio}")
    print()
    
    # 設定作成
    CFG = create_custom_config(args)
    
    # ディレクトリ設定
    balloons_dir = args.balloons_dir
    masks_dir = args.masks_dir
    backgrounds_dir = args.backgrounds_dir
    out_root = args.output
    
    # 整合性チェック
    expected_total = CFG["TARGET_NORMAL_IMAGES"] + CFG["TARGET_AUGMENTED_IMAGES"]
    if CFG["TARGET_TOTAL_IMAGES"] != expected_total:
        print(f"警告: TARGET_TOTAL_IMAGES ({CFG['TARGET_TOTAL_IMAGES']}) が通常+拡張の合計 ({expected_total}) と一致しません")
        print(f"TARGET_TOTAL_IMAGESを {expected_total} に自動調整します")
        CFG["TARGET_TOTAL_IMAGES"] = expected_total
    
    # 統計データ読み込み
    try:
        CFG["COUNT_PROBS"] = load_count_probs(CFG["COUNT_STATS_FILE"], drop_zero=True)
        print("個数ヒストグラムを読み込みました。")
    except Exception as e:
        print(f"個数ヒスト読込に失敗（フォールバックします）: {e}")

    random.seed(CFG["SEED"])
    np.random.seed(CFG["SEED"])

    # 出力ディレクトリを作成
    os.makedirs(out_root, exist_ok=True)
    
    # CFG設定をファイル出力
    config_output = {
        "timestamp": datetime.now().isoformat(),
        "script_name": "create_mixed_dataset.py",
        "command_line_args": vars(args),
        "dataset_output_path": out_root,
        "config": CFG,
        "input_directories": {
            "balloons_dir": balloons_dir,
            "masks_dir": masks_dir,
            "backgrounds_dir": backgrounds_dir
        },
        "augmentation_info": {
            "description": "通常合成とデータ拡張を組み合わせたデータセット",
            "normal_augmentation": "なし（オリジナルの面積ベースリサイズのみ）",
            "augmentation_probabilities": {
                "horizontal_flip": args.hflip_prob,
                "rotation": args.rot_prob,
                "cut_edge": args.cut_edge_prob,
                "thin_line": args.thin_line_prob,
                "panel_corner_square": args.panel_corner_prob,
                "panel_edge_straight": args.panel_edge_prob
            },
            "file_naming": {
                "normal": "XXXX_normal.png (拡張なし)",
                "augmented": "XXXX_aug.png (拡張適用)"
            }
        }
    }
    
    config_file_path = os.path.join(out_root, "config.json")
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"設定情報を保存: {config_file_path}")

    # ペア収集
    pairs = []
    for b in os.listdir(balloons_dir):
        if not b.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem = Path(b).stem
        m = f"{stem}_mask.png"
        if os.path.exists(os.path.join(masks_dir, m)):
            pairs.append((os.path.join(balloons_dir, b),
                          os.path.join(masks_dir, m)))
    
    if not pairs:
        print(f"エラー: 吹き出し画像とマスクのペアが見つかりません")
        print(f"  吹き出しディレクトリ: {balloons_dir}")
        print(f"  マスクディレクトリ: {masks_dir}")
        return
    
    # 背景画像収集
    bgs = [os.path.join(backgrounds_dir, f) for f in os.listdir(backgrounds_dir)
           if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    if not bgs:
        print(f"エラー: 背景画像が見つかりません")
        print(f"  背景ディレクトリ: {backgrounds_dir}")
        return

    print(f"吹き出しペア: {len(pairs)}個")
    print(f"背景画像: {len(bgs)}枚")

    train_pairs, val_pairs = split_balloons(pairs, CFG["BALLOON_SPLIT_RATIO"], CFG["SEED"])
    random.shuffle(bgs)
    split_pt = int(len(bgs) * CFG["BALLOON_SPLIT_RATIO"])
    train_bgs, val_bgs = bgs[:split_pt], bgs[split_pt:]

    # 生成
    train_normal = int(CFG["TARGET_NORMAL_IMAGES"] * CFG["BALLOON_SPLIT_RATIO"])
    train_augmented = int(CFG["TARGET_AUGMENTED_IMAGES"] * CFG["BALLOON_SPLIT_RATIO"])
    val_normal = CFG["TARGET_NORMAL_IMAGES"] - train_normal
    val_augmented = CFG["TARGET_AUGMENTED_IMAGES"] - train_augmented

    generate_dataset_split(train_bgs, train_pairs,
                           os.path.join(out_root, "train", "images"),
                           os.path.join(out_root, "train", "masks"),
                           "train", train_normal, train_augmented, CFG)

    generate_dataset_split(val_bgs, val_pairs,
                           os.path.join(out_root, "val", "images"),
                           os.path.join(out_root, "val", "masks"),
                           "val", val_normal, val_augmented, CFG)

    # 統計情報を収集・追加
    total_train = train_normal + train_augmented
    total_val = val_normal + val_augmented
    total_generated = total_train + total_val
    
    stats = {
        "total_images": total_generated,
        "normal_images": CFG["TARGET_NORMAL_IMAGES"],
        "augmented_images": CFG["TARGET_AUGMENTED_IMAGES"],
        "train_images": total_train,
        "train_normal": train_normal,
        "train_augmented": train_augmented,
        "val_images": total_val,
        "val_normal": val_normal,
        "val_augmented": val_augmented,
        "train_balloons_used": len(train_pairs),
        "val_balloons_used": len(val_pairs),
        "train_backgrounds_used": len(train_bgs),
        "val_backgrounds_used": len(val_bgs),
        "total_backgrounds_available": len(bgs),
        "total_balloon_pairs_available": len(pairs)
    }
    
    # config.jsonに統計情報を追加
    config_output["statistics"] = stats
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_output, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"\n=== データセット生成完了 ===")
    print(f"★ 出力先: {out_root}")
    print(f"総生成画像数: {total_generated}枚")
    print(f"  - 通常合成: {CFG['TARGET_NORMAL_IMAGES']}枚 (train: {train_normal}, val: {val_normal})")
    print(f"  - 拡張適用: {CFG['TARGET_AUGMENTED_IMAGES']}枚 (train: {train_augmented}, val: {val_augmented})")
    print(f"使用吹き出し: train {len(train_pairs)}個, val {len(val_pairs)}個")
    print(f"使用背景: train {len(train_bgs)}個, val {len(val_bgs)}個")
    print(f"設定・統計情報: {config_file_path}")

if __name__ == "__main__":
    main_custom()
