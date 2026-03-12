"""
ResNet-UNet training script with pretrained/from-scratch option

Command Line Usage Examples:
----------------------------
# Basic training with pretrained ResNet34
python train_resnet_unet.py --root ./balloon_dataset/real200_dataset --dataset real200

# Training from scratch (no pretrained)
python train_resnet_unet.py --root ./balloon_dataset/real200_dataset --dataset real200 --no-pretrained

# ResNet50 with pretrained
python train_resnet_unet.py --root ./balloon_dataset/syn1000_dataset --dataset syn1000 --backbone resnet50 --pretrained

# Custom training parameters
python train_resnet_unet.py \
    --root ./balloon_dataset/syn500_dataset \
    --dataset syn500 \
    --backbone resnet34 \
    --pretrained \
    --epochs 150 \
    --batch 16 \
    --lr 1e-4

# Resume training from checkpoint
python train_resnet_unet.py --resume ./balloon_models/real200-resnet34-unet-01.pt

# Full configuration example
python train_resnet_unet.py \
    --root ./balloon_dataset/syn1000_dataset \
    --dataset syn1000 \
    --backbone resnet34 \
    --pretrained \
    --models-dir ./balloon_models \
    --epochs 100 \
    --batch 8 \
    --lr 1e-4 \
    --patience 20 \
    --wandb-proj balloon-seg-resnet \
    --run-name syn1000-resnet34-pretrained-01

Available Arguments:
-------------------
--root          : Dataset root directory (contains train/val/test folders)
--dataset       : Dataset name for model naming and wandb
--backbone      : ResNet backbone ('resnet34' or 'resnet50')
--pretrained    : Use ImageNet pretrained weights
--no-pretrained : Don't use pretrained weights (train from scratch)
--freeze-epochs : Number of epochs to freeze backbone (transfer learning)
--models-dir    : Directory to save trained models
--resume        : Path to checkpoint to resume training
--batch         : Batch size
--epochs        : Number of training epochs
--lr            : Learning rate
--patience      : Early stopping patience (epochs without improvement)
--wandb-proj    : Wandb project name
--run-name      : Wandb run name
--use-wandb     : Enable wandb logging (disabled by default)
--optimize-threshold : Enable post-hoc threshold optimization (compute best threshold on val set)
--thr-metric    : Metric to maximize when optimizing threshold (iou or dice)
--thr-steps     : Number of threshold steps to try between 0.0 and 1.0
"""

import glob, random, time, os
from pathlib import Path
import psutil
import gc
import sys
import argparse

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import wandb

# モデルをインポート
from models import ResNetUNet

# --------------------------------------------------------------------
CFG = {
    # データセット
    "ROOT":        Path("./balloon_dataset/real200_dataset"),
    # "IMG_SIZE":    (384, 512),  # (height, width)
    "IMG_SIZE": (576, 768),  # (height, width) - 512x512 より大きいサイズで試す
    "IMAGENET_NORM": True,      # ImageNet正規化 (pretrained推奨)

    # モデル
    "BACKBONE":    "resnet34",  # "resnet34" or "resnet50"
    "PRETRAINED":  True,        # ImageNet pretrained weights
    "FREEZE_EPOCHS": 0,         # バックボーン固定エポック数（0=固定しない）

    # 学習
    "BATCH":       8,
    "EPOCHS":      100,
    "LR":          1e-4,
    "PATIENCE":    10,
    "SEED":        42,

    # メモリ監視・緊急停止
    "ENABLE_EMERGENCY_STOP": True,
    "EMERGENCY_GPU_THRESHOLD": 0.95,
    "EMERGENCY_RAM_THRESHOLD": 0.90,

    # wandb
    "WANDB_PROJ":  "balloon-seg-resnet",
    "DATASET":     "real200",
    "RUN_NAME":    "",
    "USE_WANDB":   False,

    "MODELS_DIR":  Path("balloon_models"),

    # 予測マスク出力
    "SAVE_PRED_EVERY": 10,
    "PRED_SAMPLE_N":   2,
    "PRED_DIR":        "predictions",

    # 再開 ckpt
    "RESUME":      "",
}
# --------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True

def check_memory_safety(threshold_gpu=0.90, threshold_ram=0.85):
    """メモリ使用量をチェックして危険レベルなら警告"""
    warnings = []
    
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_usage = mem_reserved / total_memory
        
        if gpu_usage > threshold_gpu:
            warnings.append(f"⚠️ GPU使用率が危険: {gpu_usage*100:.1f}% ({mem_reserved:.2f}GB/{total_memory:.2f}GB)")
    
    ram = psutil.virtual_memory()
    ram_usage = ram.percent / 100.0
    
    if ram_usage > threshold_ram:
        warnings.append(f"⚠️ RAM使用率が危険: {ram_usage*100:.1f}% ({ram.used/1e9:.2f}GB/{ram.total/1e9:.2f}GB)")
    
    is_safe = len(warnings) == 0
    warning_message = "\n".join(warnings) if warnings else ""
    
    return is_safe, warning_message

def emergency_save_and_exit(model, cfg, epoch, reason):
    """緊急時にモデルを保存して安全に終了"""
    print("\n" + "="*60)
    print("🚨 緊急停止モード")
    print("="*60)
    print(f"理由: {reason}")
    print(f"現在のエポック: {epoch}")
    
    emergency_path = f"emergency_epoch{epoch}.pth"
    try:
        torch.save(model.state_dict(), emergency_path)
        print(f"✅ 緊急保存完了: {emergency_path}")
    except Exception as e:
        print(f"❌ 緊急保存失敗: {e}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print("\n安全に終了します...")
    print("="*60)
    sys.exit(0)

def print_system_info():
    """システムリソース情報を表示"""
    print("\n" + "="*60)
    print("System Resource Check")
    print("="*60)
    
    try:
        ram = psutil.virtual_memory()
        print(f"RAM: {ram.used/1e9:.1f}GB / {ram.total/1e9:.1f}GB ({ram.percent}%)")
        if ram.percent > 85:
            print("⚠️  WARNING: RAM usage is high!")
    except ImportError:
        print("RAM: (psutil not installed)")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
        print(f"GPU Memory: {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Total: {total_mem:.1f}GB")
    else:
        print("GPU: Not available (using CPU)")
    
    print("="*60 + "\n")

# ---------------- Dataset ----------------
class BalloonDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size, imagenet_norm=True):
        self.img_paths = sorted(glob.glob(str(img_dir / "*.png")))
        if not self.img_paths:
            self.img_paths = sorted(glob.glob(str(img_dir / "*.jpg")))
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
                transforms.ToTensor(),
            ])
        
        self.mask_tf = transforms.Compose([
            transforms.Resize(resize_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
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

# --- 連番を自動で振る -------------------------------------------------
def next_version(models_dir, prefix):
    models_dir.mkdir(exist_ok=True)
    exist = sorted(models_dir.glob(f"{prefix}-*.pt"))
    if not exist: return "01"
    last = int(exist[-1].stem.split("-")[-1])
    return f"{last+1:02d}"

def collect_probs(model, loader, device):
    model.eval()
    probs, gts = [], []
    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(loader):
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()
            probs.append(p.reshape(-1))
            gts.append(y.numpy().reshape(-1))
            
            del x, y, p
            if batch_idx % 3 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
    probs = np.concatenate(probs)
    gts   = np.concatenate(gts)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
        
    return probs, gts

# --- 新規: 閾値最適化ユーティリティ ---------------------------------
def find_best_threshold(probs, gts, metric='iou', n_steps=101):
    """最適な閾値を検索する（IoU/Dice 最大化）

    Args:
        probs: 1D numpy array of predicted probabilities
        gts:   1D numpy array of ground-truth binary labels (0/1)
        metric: 'iou' or 'dice'
        n_steps: number of thresholds to try between 0.0 and 1.0

    Returns:
        best_thr (float), best_score (float)
    """
    assert metric in ('iou', 'dice')
    thresholds = np.linspace(0.0, 1.0, n_steps)
    best_thr = 0.5
    best_score = -1.0

    for thr in thresholds:
        pb = (probs >= thr).astype(np.uint8)
        tp = np.logical_and(pb == 1, gts == 1).sum()
        fp = np.logical_and(pb == 1, gts == 0).sum()
        fn = np.logical_and(pb == 0, gts == 1).sum()

        if metric == 'iou':
            denom = tp + fp + fn
            score = tp / denom if denom > 0 else 1.0
        else:
            denom = 2 * tp + fp + fn
            score = (2 * tp) / denom if denom > 0 else 1.0

        if score > best_score:
            best_score = score
            best_thr = thr
    return float(best_thr), float(best_score)

# -------------- Loss --------------------
class DiceLoss(nn.Module):
    def forward(self,y_p,y_t,eps=1e-7):
        y_p=torch.sigmoid(y_p)
        inter=2*(y_p*y_t).sum((2,3))
        union=(y_p+y_t).sum((2,3))+eps
        return 1-(inter/union).mean()

class ComboLoss(nn.Module):
    def __init__(self,a=.5):
        super().__init__()
        self.bce=nn.BCEWithLogitsLoss()
        self.dice=DiceLoss()
        self.a=a
    def forward(self,y_p,y_t):
        return self.a*self.bce(y_p,y_t)+(1-self.a)*self.dice(y_p,y_t)

# -------------- Train / Eval ------------
def train_epoch(model, loader, lossf, opt, dev, cfg=None, current_epoch=None):
    model.train()
    run = 0
    
    for batch_idx, (x, y, _) in enumerate(tqdm(loader, desc="train", leave=False)):
        if batch_idx % 10 == 0:
            is_safe, warning = check_memory_safety(
                threshold_gpu=cfg.get("EMERGENCY_GPU_THRESHOLD", 0.95) if cfg else 0.95,
                threshold_ram=cfg.get("EMERGENCY_RAM_THRESHOLD", 0.90) if cfg else 0.90
            )
            if not is_safe:
                print(f"\n[Batch {batch_idx}] {warning}")
                if cfg and cfg.get("ENABLE_EMERGENCY_STOP", True) and current_epoch:
                    emergency_save_and_exit(model, cfg, current_epoch, warning)
                else:
                    print("  → 緊急停止は無効化されています（学習継続）")
        
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = lossf(logits, y)
        loss.backward()
        opt.step()
        
        run += loss.item() * x.size(0)
        
        del x, y, logits, loss
        
        if batch_idx % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return run / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, dev, cfg=None, current_epoch=None):
    model.eval()
    dice = iou = 0
    
    for batch_idx, (x, y, _) in enumerate(tqdm(loader, desc="eval ", leave=False)):
        if batch_idx % 15 == 0:
            is_safe, warning = check_memory_safety(
                threshold_gpu=cfg.get("EMERGENCY_GPU_THRESHOLD", 0.95) if cfg else 0.95,
                threshold_ram=cfg.get("EMERGENCY_RAM_THRESHOLD", 0.90) if cfg else 0.90
            )
            if not is_safe:
                print(f"\n[Eval Batch {batch_idx}] {warning}")
                if cfg and cfg.get("ENABLE_EMERGENCY_STOP", True) and current_epoch:
                    emergency_save_and_exit(model, cfg, current_epoch, warning)
                else:
                    print("  → 緊急停止は無効化されています（評価継続）")
        
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        p = torch.sigmoid(model(x))
        pb = (p > .5).float()
        inter = (pb * y).sum((2,3))
        union = (pb + y - pb * y).sum((2,3))
        
        dice_den = (pb + y).sum((2,3)) + 1e-7
        dice += torch.where(dice_den > 1e-6, 2*inter/dice_den, torch.ones_like(dice_den)).mean().item() * x.size(0)
        
        iou_den = union + 1e-7
        iou += torch.where(iou_den > 1e-6, inter/iou_den, torch.ones_like(iou_den)).mean().item() * x.size(0)
        
        del x, y, p, pb, inter, union, dice_den, iou_den
        
        if batch_idx % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
        
    n = len(loader.dataset)
    return dice/n, iou/n

# -------------- Prediction dump ----------
@torch.no_grad()
def save_predictions(model, loader, cfg, epoch, run_dir, device):
    """val loader 先頭から PRED_SAMPLE_N 枚推論し PNG 保存 & wandb へログ"""
    if epoch % cfg["SAVE_PRED_EVERY"] != 0: return
    model.eval()

    pred_dir = run_dir / cfg["PRED_DIR"]
    pred_dir.mkdir(exist_ok=True, parents=True)
    images_for_wandb = []

    # 使用する閾値（事前に計算されていればそれを使う。なければ 0.5）
    thr = float(cfg.get("BEST_THR", 0.5))

    cnt = 0
    for x, y, stem in loader:
        for i in range(len(x)):
            if cnt >= cfg["PRED_SAMPLE_N"]: break
            img = x[i:i+1].to(device)
            gt  = y[i]
            
            pred = torch.sigmoid(model(img))[0,0]
            # 二値化に最適閾値を使う
            pred_bin = (pred > thr).cpu().numpy()*255
            gt_np    = gt[0].cpu().numpy()*255

            # Save PNG
            out_path = pred_dir / f"pred_{epoch:03}_{stem[i]}.png"
            Image.fromarray(pred_bin.astype(np.uint8)).save(out_path)

            # wandb 用の画像は wandb が有効な場合のみ作成
            if cfg.get("USE_WANDB", False):
                orig_np = (img[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
                
                # 画像をリサイズしてメモリ削減（表示用なので品質低下OK）
                h, w = orig_np.shape[:2]
                display_h, display_w = h // 2, w // 2  # 半分のサイズに
                orig_small = Image.fromarray(orig_np).resize((display_w, display_h))
                gt_small = Image.fromarray(gt_np.astype(np.uint8)).resize((display_w, display_h))
                pred_small = Image.fromarray(pred_bin.astype(np.uint8)).resize((display_w, display_h))
                
                trio = np.concatenate([
                    np.array(orig_small),
                    np.stack([np.array(gt_small)]*3, 2),
                    np.stack([np.array(pred_small)]*3, 2)
                ], axis=1)
                images_for_wandb.append(wandb.Image(trio, caption=f"ep{epoch:03}-{stem[i]} (thr={thr:.2f})"))

            cnt += 1
            
            # メモリ解放
            del img, gt, pred, pred_bin, gt_np
            
        if cnt >= cfg["PRED_SAMPLE_N"]: break

    if cfg.get("USE_WANDB", False) and images_for_wandb:
        wandb.log({f"pred_samples_epoch_{epoch}": images_for_wandb})
    elif not cfg.get("USE_WANDB", False):
        print(f"[Info] wandb disabled: saved {cnt} prediction images to {pred_dir}")
    
    # 予測保存後にメモリクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------- Main ---------------------
def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='ResNet-UNet Training Script')
    
    # データセット関連
    parser.add_argument('--root', type=str, default=None,
                        help='Dataset root directory (default: use CFG["ROOT"])')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name for wandb/model tag (default: use CFG["DATASET"])')
    
    # モデル関連
    parser.add_argument('--backbone', type=str, default=None, choices=['resnet34', 'resnet50'],
                        help='ResNet backbone (default: use CFG["BACKBONE"])')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet pretrained weights')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Don\'t use pretrained weights (train from scratch)')
    parser.add_argument('--freeze-epochs', type=int, default=None,
                        help='Number of epochs to freeze backbone (default: use CFG["FREEZE_EPOCHS"])')
    parser.add_argument('--imagenet-norm', action='store_true',
                        help='Use ImageNet normalization (recommended with pretrained)')
    parser.add_argument('--no-imagenet-norm', action='store_true',
                        help='Don\'t use ImageNet normalization')
    parser.add_argument('--models-dir', type=str, default=None,
                        help='Models directory (default: use CFG["MODELS_DIR"])')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path (default: use CFG["RESUME"])')
    
    # 学習パラメータ
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size (default: use CFG["BATCH"])')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: use CFG["EPOCHS"])')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: use CFG["LR"])')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience (default: use CFG["PATIENCE"])')
    
    # wandb関連
    parser.add_argument('--wandb-proj', type=str, default=None,
                        help='Wandb project name (default: use CFG["WANDB_PROJ"])')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    # 新規: wandb を有効化するフラグ（デフォルトは無効）
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable wandb logging (disabled by default)')
    # 閾値最適化オプション（任意）
    parser.add_argument('--optimize-threshold', action='store_true',
                        help='Enable post-hoc threshold optimization (compute best threshold on val set)')
    parser.add_argument('--thr-metric', type=str, default='iou', choices=['iou','dice'],
                        help='Metric to maximize when optimizing threshold (iou or dice)')
    parser.add_argument('--thr-steps', type=int, default=101,
                        help='Number of threshold steps to try between 0.0 and 1.0')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    cfg=CFG.copy()
    
    if args.root:
        cfg["ROOT"] = Path(args.root)
        print(f"📁 Dataset root: {cfg['ROOT']}")
    
    if args.dataset:
        cfg["DATASET"] = args.dataset
        print(f"📊 Dataset name: {cfg['DATASET']}")
    
    if args.backbone:
        cfg["BACKBONE"] = args.backbone
        print(f"🏗️  Backbone: {cfg['BACKBONE']}")
    
    # pretrained フラグの処理
    if args.pretrained:
        cfg["PRETRAINED"] = True
        print(f"✅ Using pretrained weights")
    elif args.no_pretrained:
        cfg["PRETRAINED"] = False
        print(f"🔧 Training from scratch (no pretrained)")
    
    if args.freeze_epochs is not None:
        cfg["FREEZE_EPOCHS"] = args.freeze_epochs
        print(f"❄️  Freeze backbone for {cfg['FREEZE_EPOCHS']} epochs")
    
    # ImageNet正規化フラグの処理
    if args.imagenet_norm:
        cfg["IMAGENET_NORM"] = True
        print(f"✅ Using ImageNet normalization")
    elif args.no_imagenet_norm:
        cfg["IMAGENET_NORM"] = False
        print(f"🔧 Not using ImageNet normalization")
    
    if args.models_dir:
        cfg["MODELS_DIR"] = Path(args.models_dir)
        print(f"💾 Models directory: {cfg['MODELS_DIR']}")
    
    if args.resume:
        cfg["RESUME"] = args.resume
        print(f"🔄 Resume from: {cfg['RESUME']}")
    
    if args.batch:
        cfg["BATCH"] = args.batch
        print(f"📦 Batch size: {cfg['BATCH']}")
    
    if args.epochs:
        cfg["EPOCHS"] = args.epochs
        print(f"🔁 Epochs: {cfg['EPOCHS']}")
    
    if args.lr:
        cfg["LR"] = args.lr
        print(f"📈 Learning rate: {cfg['LR']}")
    
    if args.patience:
        cfg["PATIENCE"] = args.patience
        print(f"⏱️  Patience: {cfg['PATIENCE']}")
    
    if args.wandb_proj:
        cfg["WANDB_PROJ"] = args.wandb_proj
        print(f"📊 Wandb project: {cfg['WANDB_PROJ']}")
    
    if args.run_name:
        cfg["RUN_NAME"] = args.run_name
        print(f"🏷️  Run name: {cfg['RUN_NAME']}")

    # 新規: wandb を使うかどうかの設定（デフォルト: 無効）
    cfg["USE_WANDB"] = getattr(args, "use_wandb", False)
    if cfg["USE_WANDB"]:
        wandb.init(project=cfg["WANDB_PROJ"], name=cfg["RUN_NAME"], config=cfg)
        run_dir = Path(wandb.run.dir)
    else:
        print("🚫 wandb disabled: running without wandb logging")
        run_dir = Path("runs") / cfg["RUN_NAME"]
        run_dir.mkdir(parents=True, exist_ok=True)

    # 閾値最適化設定
    cfg["OPTIMIZE_THRESHOLD"] = getattr(args, 'optimize_threshold', False)
    cfg["THR_METRIC"] = getattr(args, 'thr_metric', 'iou')
    cfg["THR_STEPS"] = getattr(args, 'thr_steps', 101)
    if cfg["OPTIMIZE_THRESHOLD"]:
        print(f"🔎 Threshold optimization ENABLED (metric={cfg['THR_METRIC']}, steps={cfg['THR_STEPS']})")
    else:
        print("🔎 Threshold optimization DISABLED (default threshold=0.5)")

    seed_everything(cfg["SEED"])
    dev="cuda" if torch.cuda.is_available() else "cpu"
    
    print_system_info()
    
    # メモリ監視設定を表示
    print("\n" + "="*60)
    print("🛡️  自動メモリ監視機能 有効")
    print("="*60)
    if cfg.get("ENABLE_EMERGENCY_STOP", True):
        print("緊急停止: ✅ 有効")
        print(f"  - GPU使用率: {cfg.get('EMERGENCY_GPU_THRESHOLD', 0.95)*100:.0f}% で緊急停止")
        print(f"  - RAM使用率: {cfg.get('EMERGENCY_RAM_THRESHOLD', 0.90)*100:.0f}% で緊急停止")
    else:
        print("緊急停止: ⚠️  無効（警告のみ表示）")
    print("="*60 + "\n")
    
    # データセット名からファイル名用のプレフィックスを作成
    dataset_name = cfg["DATASET"].replace("/", "-").replace("\\", "-")
    pretrained_suffix = "pretrained" if cfg["PRETRAINED"] else "scratch"
    prefix = f"{dataset_name}-{cfg['BACKBONE']}-{pretrained_suffix}-unet"
    version = next_version(cfg["MODELS_DIR"], prefix)
    model_tag = f"{prefix}-{version}"

    if not cfg["RUN_NAME"]:
        cfg["RUN_NAME"] = model_tag

    if cfg["USE_WANDB"]:
        wandb.init(project=cfg["WANDB_PROJ"], name=cfg["RUN_NAME"], config=cfg)
        run_dir = Path(wandb.run.dir)
    else:
        print("🚫 wandb disabled: running without wandb logging")
        run_dir = Path("runs") / cfg["RUN_NAME"]
        run_dir.mkdir(parents=True, exist_ok=True)

    root=cfg["ROOT"]
    train_ds=BalloonDataset(root/"train/images",root/"train/masks",cfg["IMG_SIZE"],cfg["IMAGENET_NORM"])
    val_ds  =BalloonDataset(root/"val/images"  ,root/"val/masks"  ,cfg["IMG_SIZE"],cfg["IMAGENET_NORM"])

    dl_tr=DataLoader(train_ds,batch_size=cfg["BATCH"],shuffle=True ,num_workers=0,pin_memory=False)
    dl_va=DataLoader(val_ds  ,batch_size=cfg["BATCH"],shuffle=False,num_workers=0,pin_memory=False)

    # モデル作成
    model = ResNetUNet(
        backbone=cfg["BACKBONE"],
        pretrained=cfg["PRETRAINED"],
        n_classes=1,
        freeze_backbone=(cfg["FREEZE_EPOCHS"] > 0)
    ).to(dev)
    
    # モデル情報をログ
    model_info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Trainable parameters: {model_info['trainable_params']:,}")
    if cfg["USE_WANDB"]:
        wandb.config.update(model_info)
    
    if cfg["RESUME"]:
        model.load_state_dict(torch.load(cfg["RESUME"],map_location=dev))
        print("Resumed from checkpoint")

    opt=torch.optim.AdamW(model.parameters(),lr=cfg["LR"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["EPOCHS"])
    lossf=ComboLoss()

    best_iou=0; patience=0
    for ep in range(1,cfg["EPOCHS"]+1):
        t=time.time()
        
        # バックボーン固定解除
        if ep == cfg["FREEZE_EPOCHS"] + 1 and cfg["FREEZE_EPOCHS"] > 0:
            print(f"\n🔓 Unfreezing backbone at epoch {ep}")
            for param in model.parameters():
                param.requires_grad = True
            # オプティマイザーを再作成（全パラメータを対象に）
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["EPOCHS"]-ep)
        
        # エポック開始時のメモリ安全チェック
        is_safe, warning = check_memory_safety(
            threshold_gpu=cfg.get("EMERGENCY_GPU_THRESHOLD", 0.95) - 0.02,
            threshold_ram=cfg.get("EMERGENCY_RAM_THRESHOLD", 0.90) - 0.02
        )
        if not is_safe:
            print(f"\n[Epoch {ep} Start] {warning}")
            if cfg.get("ENABLE_EMERGENCY_STOP", True):
                emergency_save_and_exit(model, cfg, ep, f"Epoch開始時のメモリ不足\n{warning}")
            else:
                print("  → 緊急停止は無効化されています（学習継続）")
        
        if ep % 10 == 0 and torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n[Epoch {ep}] GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        
        tr_loss=train_epoch(model, dl_tr, lossf, opt, dev, cfg, ep)
        va_dice,va_iou=eval_epoch(model, dl_va, dev, cfg, ep)
        sched.step()

        if ep % 5 == 0:
            probs, gts = collect_probs(model, dl_va, dev)
            prec, rec, _ = precision_recall_curve(gts, probs)
            pr_auc = auc(rec, prec)
            
            sample_indices = np.linspace(0, len(rec)-1, min(200, len(rec)), dtype=int)
            rec_sampled = rec[sample_indices]
            prec_sampled = prec[sample_indices]
            
            if cfg.get("USE_WANDB", False):
                wandb.log({"epoch": ep,
                        "val/pr_auc": pr_auc,
                        "val/pr_curve": wandb.plot.line(
                                wandb.Table(data=np.column_stack([rec_sampled, prec_sampled]),
                                            columns=["recall","precision"]),
                                "recall", "precision",
                                title=f"PR Curve ep{ep} (AUC={pr_auc:.3f})")})
            else:
                print(f"[Info] PR AUC (ep {ep}): {pr_auc:.4f} (wandb disabled)")

            # オプション: 閾値最適化
            if cfg.get("OPTIMIZE_THRESHOLD", False):
                try:
                    best_thr, best_score = find_best_threshold(probs, gts, metric=cfg.get("THR_METRIC", 'iou'), n_steps=cfg.get("THR_STEPS", 101))
                    cfg['BEST_THR'] = best_thr
                    thr_path = run_dir / 'best_threshold.txt'
                    with open(thr_path, 'w', encoding='utf-8') as f:
                        f.write(f"{best_thr:.4f}\n")
                    print(f"[Info] Best threshold ({cfg.get('THR_METRIC','iou')}) = {best_thr:.3f} (score={best_score:.4f}) -> saved: {thr_path}")
                    if cfg.get("USE_WANDB", False):
                        wandb.log({"val/best_threshold": best_thr, "val/best_threshold_score": best_score})
                except Exception as e:
                    print(f"[Warning] threshold optimization failed: {e}")
            
            del probs, gts, prec, rec, sample_indices, rec_sampled, prec_sampled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        save_predictions(model, dl_va, cfg, ep, run_dir, dev)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if cfg.get("USE_WANDB", False):
            wandb.log({"epoch":ep,"loss":tr_loss,
                       "val_dice":va_dice,"val_iou":va_iou,
                       "lr":sched.get_last_lr()[0]})
        else:
            print(f"[Epoch {ep}] loss={tr_loss:.4f} dice={va_dice:.4f} iou={va_iou:.4f}  {time.time()-t:.1f}s")

        if va_iou>best_iou:
            best_iou,patience=va_iou,0
            ckpt_wandb = run_dir / f"best_ep{ep:03}_iou{va_iou:.4f}.pt"
            torch.save(model.state_dict(), ckpt_wandb)

            cfg["MODELS_DIR"].mkdir(parents=True, exist_ok=True)
            ckpt_models = cfg["MODELS_DIR"] / f"{model_tag}.pt"
            torch.save(model.state_dict(), ckpt_models)
            print(f"✓ Best model saved! IoU: {va_iou:.4f} -> {ckpt_models}")
        else:
            patience+=1
            if patience>=cfg["PATIENCE"]:
                print("Early stopping."); break

if __name__=="__main__":
    main()
