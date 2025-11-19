#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepLabv3+ inference and evaluation script
Loads a trained DeepLabv3+ model and evaluates on test dataset
Calculates Dice/IoU metrics and saves prediction masks

Command Line Usage Examples:
----------------------------
# Basic evaluation with model tag
python test_deeplabv3plus.py \
    --model-tag real200-deeplabv3plus-resnet50-pretrained-s16-01 \
    --backbone resnet50 \
    --output-stride 16

# Specify custom paths
python test_deeplabv3plus.py \
    --model-tag syn1000-deeplabv3plus-resnet101-pretrained-s8-01 \
    --backbone resnet101 \
    --output-stride 8 \
    --data-root balloon_dataset/test100_dataset \
    --models-dir ./balloon_models

# Full configuration example
python test_deeplabv3plus.py \
    --model-tag syn1000-deeplabv3plus-resnet50-pretrained-s16-01 \
    --backbone resnet50 \
    --output-stride 16 \
    --models-dir ./balloon_models \
    --data-root ./test_dataset \
    --result-dir ./experiment_results \
    --batch 16 \
    --save-pred-n 10 \
    --wandb-proj balloon-seg-deeplabv3plus-experiments \
    --run-name syn1000-deeplabv3plus-test-01

# Directly specify model path
python test_deeplabv3plus.py \
    --model-path ./balloon_models/real200-deeplabv3plus-resnet50-pretrained-s16-01.pt \
    --backbone resnet50 \
    --output-stride 16 \
    --data-root ./test_dataset

# Disable wandb logging
python test_deeplabv3plus.py \
    --model-tag real200-deeplabv3plus-resnet50-pretrained-s16-01 \
    --backbone resnet50 \
    --output-stride 16 \
    --no-wandb

Available Arguments:
-------------------
--model-tag     : Model tag (e.g., real200-deeplabv3plus-resnet50-pretrained-s16-01)
--model-path    : Direct path to model .pt file (overrides model-tag)
--backbone      : ResNet backbone ('resnet50' or 'resnet101') - REQUIRED
--output-stride : Output stride (8 or 16) - REQUIRED
--models-dir    : Directory containing model files (default: ./models)
--data-root     : Test dataset root directory (contains images/ and masks/)
--batch         : Batch size for inference
--result-dir    : Directory to save evaluation results
--save-pred-n   : Number of prediction images to save (0=save all)
--wandb-proj    : Wandb project name
--run-name      : Wandb run name
--no-wandb      : Disable wandb logging

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
from models import DeepLabv3Plus

# --------------------------------------------------------------------
CFG = {
    # ------- 必ず書き換える -------------
    "MODEL_TAG":   "real200-deeplabv3plus-resnet50-pretrained-s16-01",
    "BACKBONE":    "resnet50",  # "resnet50" or "resnet101"
    "OUTPUT_STRIDE": 16,        # 8 or 16
    "DATA_ROOT":   Path("test_dataset"),
    "IMAGENET_NORM": True,      # ImageNet正規化（pretrainedモデル使用時は推奨）
    # ----------------------------------
    "IMG_SIZE":    (384, 512),
    "BATCH":       8,

    # wandb
    "WANDB_PROJ":  "balloon-seg-deeplabv3plus",
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
                transforms.Resize(resize_size),
                transforms.ToTensor()
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
def evaluate_detailed(model, loader, device):
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
        pb = (p > 0.5).float()
        
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
def save_all_predictions(model, loader, cfg, device, result_dir):
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
    
    for x, y, stem in tqdm(loader, desc="予測結果保存中"):
        for i in range(len(x)):
            img = x[i:i+1].to(device)
            gt = y[i]
            pred = torch.sigmoid(model(img))[0, 0]
            pred_bin = (pred > 0.5).cpu().numpy() * 255
            gt_np = gt[0].cpu().numpy() * 255
            
            orig_img = (x[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(orig_img).save(images_dir / f"{stem[i]}.png")
            
            Image.fromarray(pred_bin.astype(np.uint8)).save(predicts_dir / f"{stem[i]}_pred.png")
            
            comparison = np.concatenate([
                orig_img,
                np.stack([pred_bin] * 3, 2).astype(np.uint8)
            ], axis=1)
            Image.fromarray(comparison).save(comparisons_dir / f"{stem[i]}_comparison.png")
            
            if saved < cfg["SAVE_PRED_N"]:
                trio = np.concatenate([
                    orig_img,
                    np.stack([gt_np] * 3, 2),
                    np.stack([pred_bin] * 3, 2)
                ], 1)
                img_logs.append(wandb.Image(trio, caption=stem[i]))
                saved += 1
    
    if img_logs:
        wandb.log({"test_samples": img_logs})
    
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
        "output_stride": cfg["OUTPUT_STRIDE"],
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
        f.write(f"DeepLabv3+ Model Evaluation Results\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Model: {cfg['MODEL_TAG']}\n")
        f.write(f"Backbone: {cfg['BACKBONE']}\n")
        f.write(f"Output Stride: {cfg['OUTPUT_STRIDE']}\n")
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
    parser = argparse.ArgumentParser(description='DeepLabv3+ Test/Evaluation Script')
    
    # モデル関連
    parser.add_argument('--model-tag', type=str, default=None,
                        help='Model tag name (default: use CFG["MODEL_TAG"])')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Direct path to model checkpoint (overrides --model-tag)')
    parser.add_argument('--backbone', type=str, default=None, choices=['resnet50', 'resnet101'],
                        help='ResNet backbone (REQUIRED if not in CFG)')
    parser.add_argument('--output-stride', type=int, default=None, choices=[8, 16],
                        help='Output stride (REQUIRED if not in CFG)')
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
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    
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
    
    if args.output_stride:
        cfg["OUTPUT_STRIDE"] = args.output_stride
        print(f"📏 Output stride: {cfg['OUTPUT_STRIDE']}")
    
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

    # ---------- wandb ----------
    if not args.no_wandb:
        wandb.init(project=cfg["WANDB_PROJ"], name=cfg["RUN_NAME"], config=cfg)
        run_dir = Path(wandb.run.dir)
    else:
        print("⚠️  Wandb logging disabled")

    # ---------- 結果保存ディレクトリ作成 ----------
    result_dir = Path(args.result_dir) / cfg["MODEL_TAG"]
    result_dir.mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    test_ds = BalloonDataset(cfg["DATA_ROOT"] / "images",
                            cfg["DATA_ROOT"] / "masks",
                            cfg["IMG_SIZE"],
                            cfg["IMAGENET_NORM"])
    dl = DataLoader(test_ds, batch_size=cfg["BATCH"], shuffle=False,
                    num_workers=4, pin_memory=True)

    print(f"テストデータ数: {len(test_ds)}枚")

    # ---------- model ----------
    # バックボーンとoutput_strideが指定されていない場合はエラー
    if not cfg["BACKBONE"]:
        raise ValueError("Backbone must be specified via --backbone or CFG['BACKBONE']")
    if not cfg["OUTPUT_STRIDE"]:
        raise ValueError("Output stride must be specified via --output-stride or CFG['OUTPUT_STRIDE']")
    
    model = DeepLabv3Plus(
        backbone=cfg["BACKBONE"],
        pretrained=False,  # テスト時はpretrainedは不要
        output_stride=cfg["OUTPUT_STRIDE"],
        n_classes=1
    ).to(dev)
    
    # モデルパスの決定
    if args.model_path:
        ckpt = Path(args.model_path)
    else:
        ckpt = Path(args.models_dir) / f"{cfg['MODEL_TAG']}.pt"
    
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    print(f"モデル読み込み完了: {ckpt}")
    
    # モデル情報を表示
    model_info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Backbone: {model_info['backbone']}")
    print(f"  Output Stride: {model_info['output_stride']}")
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Trainable parameters: {model_info['trainable_params']:,}")

    # ---------- 詳細評価 ----------
    print("\n詳細評価を実行中...")
    metrics = evaluate_detailed(model, dl, dev)
    
    print(f"\n=== 評価結果 ===")
    print(f"Average Dice: {metrics['avg_dice']:.4f}")
    print(f"Average IoU:  {metrics['avg_iou']:.4f}")
    print(f"Average F1:   {metrics['avg_f1']:.4f}")
    print(f"Global Dice:  {metrics['global_dice']:.4f}")
    print(f"Global IoU:   {metrics['global_iou']:.4f}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")

    if not args.no_wandb:
        wandb.log({
            "test_avg_dice": metrics["avg_dice"],
            "test_avg_iou": metrics["avg_iou"],
            "test_avg_f1": metrics["avg_f1"],
            "test_global_dice": metrics["global_dice"],
            "test_global_iou": metrics["global_iou"],
            "test_accuracy": metrics["accuracy"]
        })

    # ---------- 全画像と予測結果を保存 ----------
    print("\n予測結果を保存中...")
    save_all_predictions(model, dl, cfg, dev, result_dir)

    # ---------- 評価結果をファイルに保存 ----------
    print("\n評価結果をファイルに保存中...")
    save_results_to_file(metrics, cfg, result_dir)

    print(f"\n=== 完了 ===")
    print(f"結果保存先: {result_dir}")
    
    if not args.no_wandb:
        wandb.finish()

if __name__=="__main__":
    main()
