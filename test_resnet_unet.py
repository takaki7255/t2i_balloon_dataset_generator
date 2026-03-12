#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResNet-UNet inference and evaluation script
Loads a trained ResNet-UNet model and evaluates on test dataset
Calculates Dice/IoU metrics and saves prediction masks

Command Line Usage Examples:
----------------------------
# Basic evaluation with model tag
python test_resnet_unet.py --model-tag real200-resnet34-pretrained-unet-01

# Specify custom paths
python test_resnet_unet.py \
    --model-tag syn1000-resnet34-pretrained-unet-01 \
    --data-root balloon_dataset/test100_dataset \
    --models-dir ./balloon_models

# Specify backbone explicitly
python test_resnet_unet.py \
    --model-tag syn500-resnet50-scratch-unet-01 \
    --backbone resnet50 \
    --data-root ./test_dataset

# Full configuration example
python test_resnet_unet.py \
    --model-tag syn1000-resnet34-pretrained-unet-01 \
    --backbone resnet34 \
    --models-dir ./balloon_models \
    --data-root ./test_dataset \
    --result-dir ./experiment_results \
    --batch 16 \
    --save-pred-n 10 \
    --wandb-proj balloon-seg-resnet-experiments \
    --run-name syn1000-resnet34-test-01

# Directly specify model path
python test_resnet_unet.py \
    --model-path ./balloon_models/real200-resnet34-pretrained-unet-01.pt \
    --backbone resnet34 \
    --data-root ./test_dataset

# Disable wandb logging
python test_resnet_unet.py \
    --model-tag real200-resnet34-pretrained-unet-01 \
    --backbone resnet34

Available Arguments:
-------------------
--model-tag     : Model tag (e.g., real200-resnet34-pretrained-unet-01)
--model-path    : Direct path to model .pt file (overrides model-tag)
--backbone      : ResNet backbone ('resnet34' or 'resnet50') - REQUIRED
--models-dir    : Directory containing model files (default: ./models)
--data-root     : Test dataset root directory (contains images/ and masks/)
--batch         : Batch size for inference
--result-dir    : Directory to save evaluation results
--save-pred-n   : Number of prediction images to save (0=save all)
--wandb-proj    : Wandb project name
--run-name      : Wandb run name
--use-wandb     : Enable wandb logging (disabled by default)

Output:
-------
Results are saved to: {result-dir}/{model-tag}/
  - evaluation_results.json    : Metrics in JSON format
  - evaluation_summary.txt     : Human-readable summary
  - images/                    : Original test images
  - masks/                     : Ground truth masks
  - predicts/                  : Predicted masks
  - comparisons/               : Side-by-side comparisons
"""

import glob, os, random, time
from pathlib import Path
import argparse

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb

# モデルをインポート
from models import ResNetUNet

# --------------------------------------------------------------------
CFG = {
    # ------- 必ず書き換える -------------
    "MODEL_TAG":   "real200-resnet34-pretrained-unet-01",
    "BACKBONE":    "resnet34",  # "resnet34" or "resnet50"
    "DATA_ROOT":   Path("test_dataset"),
    # ----------------------------------
    # "IMG_SIZE":    (384, 512),
    "IMG_SIZE":    (576, 768),
    "IMAGENET_NORM": True,  # ImageNet正規化 (pretrained推奨)
    "BATCH":       8,

    # wandb
    "WANDB_PROJ":  "balloon-seg-resnet",
    "RUN_NAME":    "",
    "SAVE_PRED_N": 20,
    "PRED_DIR":    "test_predictions",
    "SEED":        42,
}
# --------------------------------------------------------------------


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True


# ---------------- Dataset ----------------
class BalloonDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size, imagenet_norm=True):
        self.img_paths = sorted(glob.glob(str(img_dir/"*.png")))
        if not self.img_paths:
            self.img_paths = sorted(glob.glob(str(img_dir/"*.jpg")))
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        
        self.mask_dir  = mask_dir
        
        if isinstance(img_size, tuple):
            resize_size = img_size
        else:
            resize_size = (img_size, img_size)
        
        # ImageNet正規化パラメータ
        if imagenet_norm:
            self.img_tf = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_tf = transforms.Compose([
                transforms.Resize(resize_size), transforms.ToTensor()
            ])
        
        self.mask_tf = transforms.Compose([
            transforms.Resize(resize_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_p = self.img_paths[idx]
        stem   = Path(img_p).stem
        mask_p = self.mask_dir / f"{stem}_mask.png"
        img  = self.img_tf(Image.open(img_p).convert("RGB"))
        mask = self.mask_tf(Image.open(mask_p).convert("L"))
        mask = (mask > .5).float()
        return img, mask, stem


# ---------------- Detailed Metrics -----------------
@torch.no_grad()
def evaluate_detailed(model, loader, device, thr=0.5):
    """詳細な評価メトリクスを計算"""
    model.eval()
    
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    all_f1 = []
    
    total_tp = total_fp = total_fn = total_tn = 0
    
    for x, y, _ in tqdm(loader, desc="評価中", leave=False):
        x, y = x.to(device), y.to(device)
        p = torch.sigmoid(model(x))
        pb = (p > thr).float()
        
        for i in range(x.size(0)):
            pred_flat = pb[i].flatten()
            gt_flat = y[i].flatten()
            
            tp = (pred_flat * gt_flat).sum().item()
            fp = (pred_flat * (1 - gt_flat)).sum().item()
            fn = ((1 - pred_flat) * gt_flat).sum().item()
            tn = ((1 - pred_flat) * (1 - gt_flat)).sum().item()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
            iou = tp / (tp + fp + fn + 1e-7)
            
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            all_dice.append(dice)
            all_iou.append(iou)
    
    avg_dice = np.mean(all_dice)
    avg_iou = np.mean(all_iou)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    
    global_precision = total_tp / (total_tp + total_fp + 1e-7)
    global_recall = total_tp / (total_tp + total_fn + 1e-7)
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall + 1e-7)
    global_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-7)
    global_iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)
    
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-7)
    
    return {
        "avg_dice": avg_dice,
        "avg_iou": avg_iou, 
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "global_precision": global_precision,
        "global_recall": global_recall,
        "global_f1": global_f1,
        "global_dice": global_dice,
        "global_iou": global_iou,
        "accuracy": accuracy,
        "individual_dice": all_dice,
        "individual_iou": all_iou,
        "individual_precision": all_precision,
        "individual_recall": all_recall,
        "individual_f1": all_f1
    }


# ---------------- Save All Predictions and Images -------
@torch.no_grad()
def save_all_predictions(model, loader, cfg, device, result_dir, thr=0.5):
    """すべての画像と予測結果を保存"""
    model.eval()
    
    images_dir = result_dir / "images"
    predicts_dir = result_dir / "predicts"
    comparisons_dir = result_dir / "comparisons"
    images_dir.mkdir(exist_ok=True)
    predicts_dir.mkdir(exist_ok=True)
    comparisons_dir.mkdir(exist_ok=True)
    
    img_logs = []
    saved = 0

    # 逆正規化パラメータ（ImageNet）
    if cfg.get("IMAGENET_NORM", False):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    else:
        mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    for x, y, stem in tqdm(loader, desc="予測結果保存中"):
        for i in range(len(x)):
            img = x[i:i+1].to(device)
            gt = y[i]
            pred = torch.sigmoid(model(img))[0, 0]

            # 予測・GTをsingle-channel uint8に変換
            pred_bin = (pred > thr).cpu().numpy().astype(np.uint8) * 255
            gt_np = (gt[0].cpu().numpy().astype(np.uint8)) * 255

            # 元画像を逆正規化して uint8 RGB に変換
            img_np = x[i].cpu().permute(1, 2, 0).numpy().astype(np.float32)  # H,W,C, normalized (or 0-1)
            img_unnorm = (img_np * std.reshape(1, 1, 3)) + mean.reshape(1, 1, 3)
            orig_img = np.clip(img_unnorm * 255.0, 0, 255).astype(np.uint8)

            # 保存
            Image.fromarray(orig_img).save(images_dir / f"{stem[i]}.png")
            Image.fromarray(pred_bin.astype(np.uint8)).save(predicts_dir / f"{stem[i]}_pred.png")

            # 比較画像: 元画像と予測をRGBで横に連結
            pred_rgb = np.stack([pred_bin] * 3, axis=2).astype(np.uint8)
            comparison = np.concatenate([orig_img, pred_rgb], axis=1)
            Image.fromarray(comparison).save(comparisons_dir / f"{stem[i]}_comparison.png")

            # wandb 用の画像（最初のN枚のみ） - wandb が有効な場合のみ作成
            if cfg.get("USE_WANDB", False) and saved < cfg.get("SAVE_PRED_N", 0):
                gt_rgb = np.stack([gt_np] * 3, axis=2).astype(np.uint8)
                trio = np.concatenate([orig_img, gt_rgb, pred_rgb], axis=1)
                img_logs.append(wandb.Image(trio, caption=stem[i] + f" (thr={thr:.2f})"))
                saved += 1
    
    # wandb ログは有効時のみ
    if cfg.get("USE_WANDB", False) and img_logs:
        wandb.log({"test_samples": img_logs})
    elif not cfg.get("USE_WANDB", False):
        print(f"[Info] wandb disabled: saved {saved} sample images to {result_dir}")

    print(f"画像保存完了: {images_dir}")
    print(f"予測結果保存完了: {predicts_dir}")
    print(f"比較画像保存完了: {comparisons_dir}")


# ---------------- Save Results -------
def save_results_to_file(metrics, cfg, result_dir):
    """評価結果をファイルに保存"""
    import json
    from datetime import datetime
    
    results = {
        "model_tag": cfg["MODEL_TAG"],
        "backbone": cfg["BACKBONE"],
        "data_root": str(cfg["DATA_ROOT"]),
        "img_size": cfg["IMG_SIZE"],
        "batch_size": cfg["BATCH"],
        "evaluation_time": datetime.now().isoformat(),
        "metrics": {
            "average_metrics": {
                "dice": float(metrics["avg_dice"]),
                "iou": float(metrics["avg_iou"]),
                "precision": float(metrics["avg_precision"]),
                "recall": float(metrics["avg_recall"]),
                "f1_score": float(metrics["avg_f1"])
            },
            "global_metrics": {
                "dice": float(metrics["global_dice"]),
                "iou": float(metrics["global_iou"]),
                "precision": float(metrics["global_precision"]),
                "recall": float(metrics["global_recall"]),
                "f1_score": float(metrics["global_f1"]),
                "accuracy": float(metrics["accuracy"])
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
    
    json_path = result_dir / "evaluation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    txt_path = result_dir / "evaluation_summary.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"ResNet-UNet Model Evaluation Results\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Model: {cfg['MODEL_TAG']}\n")
        f.write(f"Backbone: {cfg['BACKBONE']}\n")
        f.write(f"Dataset: {cfg['DATA_ROOT']}\n")
        f.write(f"Image Size: {cfg['IMG_SIZE']}\n")
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
    
    print(f"評価結果保存完了:")
    print(f"  JSON: {json_path}")
    print(f"  テキスト: {txt_path}")


# ---------------- Main -------------------
def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='ResNet-UNet Test/Evaluation Script')
    
    # モデル関連
    parser.add_argument('--model-tag', type=str, default=None,
                        help='Model tag name (default: use CFG["MODEL_TAG"])')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Direct path to model checkpoint (overrides --model-tag)')
    parser.add_argument('--backbone', type=str, default=None, choices=['resnet34', 'resnet50'],
                        help='ResNet backbone (REQUIRED if not in CFG)')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Models directory (default: "models")')
    
    # データセット関連
    parser.add_argument('--data-root', type=str, default=None,
                        help='Test dataset root directory (default: use CFG["DATA_ROOT"])')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size (default: use CFG["BATCH"])')
    
    # 出力関連
    parser.add_argument('--result-dir', type=str, default='test_results',
                        help='Results output directory (default: "test_results")')
    parser.add_argument('--save-pred-n', type=int, default=None,
                        help='Number of predictions to save (default: use CFG["SAVE_PRED_N"])')
    
    # wandb関連
    parser.add_argument('--wandb-proj', type=str, default=None,
                        help='Wandb project name (default: use CFG["WANDB_PROJ"])')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable wandb logging (disabled by default)')
    # 新規: 学習時に保存された最適閾値を利用するオプション
    parser.add_argument('--use-best-threshold', action='store_true',
                        help='If available, load best_threshold.txt from runs/<model_tag>/ and use it for binarization')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Fallback threshold for binarization if best threshold not used/found (default:0.5)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    cfg = CFG.copy()
    
    if args.model_tag:
        cfg["MODEL_TAG"] = args.model_tag
        print(f"🏷️  Model tag: {cfg['MODEL_TAG']}")
    
    if args.backbone:
        cfg["BACKBONE"] = args.backbone
        print(f"🏗️  Backbone: {cfg['BACKBONE']}")
    
    if args.data_root:
        cfg["DATA_ROOT"] = Path(args.data_root)
        print(f"📁 Test data root: {cfg['DATA_ROOT']}")
    
    if args.batch:
        cfg["BATCH"] = args.batch
        print(f"📦 Batch size: {cfg['BATCH']}")
    
    if args.save_pred_n:
        cfg["SAVE_PRED_N"] = args.save_pred_n
        print(f"💾 Save predictions: {cfg['SAVE_PRED_N']}")
    
    if args.wandb_proj:
        cfg["WANDB_PROJ"] = args.wandb_proj
        print(f"📊 Wandb project: {cfg['WANDB_PROJ']}")
    
    if args.run_name:
        cfg["RUN_NAME"] = args.run_name
    elif not cfg["RUN_NAME"]:
        cfg["RUN_NAME"] = cfg["MODEL_TAG"] + "-test"
    
    seed_everything(cfg["SEED"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 新規: wandb を使うかどうかの設定
    cfg["USE_WANDB"] = getattr(args, "use_wandb", False)
    if not cfg["USE_WANDB"]:
        print("🚫 wandb logging is DISABLED. All wandb operations will be skipped.")

    # 閾値オプション
    cfg["USE_BEST_THRESHOLD"] = getattr(args, 'use_best_threshold', False)
    cfg["THRESHOLD"] = float(getattr(args, 'threshold', 0.5))
    if cfg["USE_BEST_THRESHOLD"]:
        # try to locate runs/<MODEL_TAG>/best_threshold.txt or result_dir/best_threshold.txt later
        print("🔎 Will attempt to load best_threshold.txt from runs/<model_tag>/ if present")

    # ---------- wandb ----------
    if cfg.get("USE_WANDB", False):
        wandb.init(project=cfg["WANDB_PROJ"], name=cfg["RUN_NAME"], config=cfg)
        run_dir = Path(wandb.run.dir)
    else:
        print("⚠️  Wandb logging disabled")
        run_dir = Path("runs") / cfg["RUN_NAME"]
        run_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 結果保存ディレクトリ作成 ----------
    # NOTE: result_dir is created after resolving the checkpoint path below
    # to ensure the result folder matches the actual tested model filename.
 
    # ---------- data ----------
    test_ds = BalloonDataset(cfg["DATA_ROOT"] / "images",
                            cfg["DATA_ROOT"] / "masks",
                            cfg["IMG_SIZE"],
                            cfg["IMAGENET_NORM"])
    dl = DataLoader(test_ds, batch_size=cfg["BATCH"], shuffle=False,
                    num_workers=4, pin_memory=True)

    print(f"テストデータ数: {len(test_ds)}枚")

    # ---------- model ----------
    # バックボーンが指定されていない場合はエラー
    if not cfg["BACKBONE"]:
        raise ValueError("Backbone must be specified via --backbone or CFG['BACKBONE']")
    
    model = ResNetUNet(
        backbone=cfg["BACKBONE"],
        pretrained=False,  # テスト時はpretrainedは不要
        n_classes=1
    ).to(dev)
    
    # モデルパスの決定
    if args.model_path:
        ckpt = Path(args.model_path)
    else:
        ckpt = Path(args.models_dir) / f"{cfg['MODEL_TAG']}.pt"
    
    # If a checkpoint path is provided (or resolved), prefer its stem as MODEL_TAG
    if ckpt is not None and ckpt.exists():
        cfg["MODEL_TAG"] = ckpt.stem
    
    # Now create the result directory named after the actual tested model
    result_dir = Path(args.result_dir) / cfg["MODEL_TAG"]
    result_dir.mkdir(parents=True, exist_ok=True)

    # If requested, try to read best_threshold.txt
    thr = cfg.get("THRESHOLD", 0.5)
    if cfg.get("USE_BEST_THRESHOLD", False):
        cand = Path("runs") / cfg["MODEL_TAG"] / "best_threshold.txt"
        if (cand.exists()):
            try:
                thr = float(cand.read_text().strip())
                print(f"Loaded best threshold from: {cand} -> {thr:.4f}")
                cfg['BEST_THR'] = thr
            except Exception as e:
                print(f"[Warning] failed to read {cand}: {e} (falling back to {thr})")
        else:
            # also check result_dir
            cand2 = result_dir / 'best_threshold.txt'
            if cand2.exists():
                try:
                    thr = float(cand2.read_text().strip())
                    print(f"Loaded best threshold from: {cand2} -> {thr:.4f}")
                    cfg['BEST_THR'] = thr
                except Exception as e:
                    print(f"[Warning] failed to read {cand2}: {e} (falling back to {thr})")
            else:
                print(f"[Info] best_threshold.txt not found in runs/{cfg['MODEL_TAG']} or {result_dir}; using threshold={thr}")

    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    print(f"モデル読み込み完了: {ckpt}")
    
    # モデル情報を表示
    model_info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Backbone: {model_info['backbone']}")
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Trainable parameters: {model_info['trainable_params']:,}")

    # ---------- 詳細評価 ----------
    print("\n詳細評価を実行中...")
    metrics = evaluate_detailed(model, dl, dev, thr=thr)
    
    print(f"\n=== 評価結果 ===")
    print(f"Average Dice: {metrics['avg_dice']:.4f}")
    print(f"Average IoU:  {metrics['avg_iou']:.4f}")
    print(f"Average F1:   {metrics['avg_f1']:.4f}")
    print(f"Global Dice:  {metrics['global_dice']:.4f}")
    print(f"Global IoU:   {metrics['global_iou']:.4f}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")

    # wandb にログ
    if cfg.get("USE_WANDB", False):
        wandb.log({
            "test_avg_dice": metrics["avg_dice"],
            "test_avg_iou": metrics["avg_iou"],
            "test_avg_f1": metrics["avg_f1"],
            "test_global_dice": metrics["global_dice"],
            "test_global_iou": metrics["global_iou"],
            "test_accuracy": metrics["accuracy"]
        })
    else:
        print(f"[Info] wandb disabled: test metrics printed to console only.")

    # ---------- 全画像と予測結果を保存 ----------
    print("\n予測結果を保存中...")
    save_all_predictions(model, dl, cfg, dev, result_dir, thr=thr)

    # ---------- 評価結果をファイルに保存 ----------
    print("\n評価結果をファイルに保存中...")
    save_results_to_file(metrics, cfg, result_dir)

    print(f"\n=== 完了 ===")
    print(f"結果保存先: {result_dir}")
    
    if cfg.get("USE_WANDB", False):
        wandb.finish()

if __name__=="__main__":
    main()
