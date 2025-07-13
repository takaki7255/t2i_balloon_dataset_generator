#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
U-Net 推論専用 : models/<tag>.pt + real_dataset/test を使い
Dice / IoU を計算し、予測マスク PNG を保存＆wandb にアップロード
"""

import glob, os, random, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb

# --------------------------------------------------------------------
CFG = {
    # ------- 必ず書き換える -------------
    "MODEL_TAG":   "synreal-unet-01",     # 例： models/real-unet-01.pt
    "DATA_ROOT":   Path("test_dataset"),  # images, masks が入ったテストデータセット
    # ----------------------------------
    "IMG_SIZE":    512,
    "BATCH":       8,

    # wandb
    "WANDB_PROJ":  "balloon-seg",
    "RUN_NAME":    "",         # 空なら MODEL_TAG を使用
    "SAVE_PRED_N": 20,         # 保存＆ログする枚数
    "PRED_DIR":    "test_predictions",
    "SEED":        42,
}
# --------------------------------------------------------------------


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True


# ---------------- Dataset ----------------
class BalloonDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size):
        # jpg と png 両方に対応
        self.img_paths = sorted(glob.glob(str(img_dir/"*.png")))
        if not self.img_paths:
            self.img_paths = sorted(glob.glob(str(img_dir/"*.jpg")))
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        
        self.mask_dir  = mask_dir
        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor()
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
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


# ---------------- U-Net (同じ定義) ---------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, n_classes=1, chs=(64,128,256,512,1024)):
        super().__init__()
        self.downs = nn.ModuleList()
        in_c = 3
        for c in chs[:-1]:
            self.downs.append(DoubleConv(in_c, c)); in_c = c
        self.bottleneck = DoubleConv(chs[-2], chs[-1])
        self.ups_tr = nn.ModuleList([
            nn.ConvTranspose2d(chs[i], chs[i-1], 2, 2) for i in range(len(chs)-1,0,-1)])
        self.up_convs = nn.ModuleList([
            DoubleConv(chs[i], chs[i-1]) for i in range(len(chs)-1,0,-1)])
        self.out_conv = nn.Conv2d(chs[0], n_classes, 1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, sk in zip(self.ups_tr, self.up_convs, skips[::-1]):
            x = up(x); x = torch.cat([x, sk], 1); x = conv(x)
        return self.out_conv(x)


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
        
        # バッチ内の各画像について計算
        for i in range(x.size(0)):
            pred_flat = pb[i].flatten()
            gt_flat = y[i].flatten()
            
            # TP, FP, FN, TN
            tp = (pred_flat * gt_flat).sum().item()
            fp = (pred_flat * (1 - gt_flat)).sum().item()
            fn = ((1 - pred_flat) * gt_flat).sum().item()
            tn = ((1 - pred_flat) * (1 - gt_flat)).sum().item()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            
            # 個別画像のメトリクス
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
    
    # 全体の平均メトリクス
    avg_dice = np.mean(all_dice)
    avg_iou = np.mean(all_iou)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    
    # 全体のピクセル単位メトリクス
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
    images_dir.mkdir(exist_ok=True)
    predicts_dir.mkdir(exist_ok=True)
    
    img_logs = []
    saved = 0
    
    for x, y, stem in tqdm(loader, desc="予測結果保存中"):
        for i in range(len(x)):
            img = x[i:i+1].to(device)
            gt = y[i]
            pred = torch.sigmoid(model(img))[0, 0]
            pred_bin = (pred > 0.5).cpu().numpy() * 255
            gt_np = gt[0].cpu().numpy() * 255
            
            # 元画像を保存
            orig_img = (x[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(orig_img).save(images_dir / f"{stem[i]}.png")
            
            # 予測マスクを保存
            Image.fromarray(pred_bin.astype(np.uint8)).save(predicts_dir / f"{stem[i]}_pred.png")
            
            # wandb用の画像（最初のN枚のみ）
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


# ---------------- Save Results -------
def save_results_to_file(metrics, cfg, result_dir):
    """評価結果をファイルに保存"""
    import json
    from datetime import datetime
    
    # JSON形式で詳細結果を保存
    results = {
        "model_tag": cfg["MODEL_TAG"],
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
    
    # JSON保存
    json_path = result_dir / "evaluation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 読みやすいテキスト形式でも保存
    txt_path = result_dir / "evaluation_summary.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Model Evaluation Results\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Model: {cfg['MODEL_TAG']}\n")
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
def main():
    cfg = CFG
    if not cfg["RUN_NAME"]:
        cfg["RUN_NAME"] = cfg["MODEL_TAG"] + "-test"
    seed_everything(cfg["SEED"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- wandb ----------
    wandb.init(project=cfg["WANDB_PROJ"], name=cfg["RUN_NAME"], config=cfg)
    run_dir = Path(wandb.run.dir)

    # ---------- 結果保存ディレクトリ作成 ----------
    result_dir = Path("test_results") / cfg["MODEL_TAG"]
    result_dir.mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    test_ds = BalloonDataset(cfg["DATA_ROOT"] / "images",
                            cfg["DATA_ROOT"] / "masks",
                            cfg["IMG_SIZE"])
    dl = DataLoader(test_ds, batch_size=cfg["BATCH"], shuffle=False,
                    num_workers=4, pin_memory=True)

    print(f"テストデータ数: {len(test_ds)}枚")

    # ---------- model ----------
    model = UNet().to(dev)
    ckpt = Path("models") / f"{cfg['MODEL_TAG']}.pt"
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    print(f"モデル読み込み完了: {ckpt}")

    # ---------- 詳細評価 ----------
    print("詳細評価を実行中...")
    metrics = evaluate_detailed(model, dl, dev)
    
    # 基本メトリクス表示
    print(f"\n=== 評価結果 ===")
    print(f"Average Dice: {metrics['avg_dice']:.4f}")
    print(f"Average IoU:  {metrics['avg_iou']:.4f}")
    print(f"Average F1:   {metrics['avg_f1']:.4f}")
    print(f"Global Dice:  {metrics['global_dice']:.4f}")
    print(f"Global IoU:   {metrics['global_iou']:.4f}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")

    # wandb にログ
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
    
    wandb.finish()

if __name__=="__main__":
    main()
