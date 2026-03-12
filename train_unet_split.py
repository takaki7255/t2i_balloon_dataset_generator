"""
U-Net training script with train/val split and memory monitoring

Command Line Usage Examples:
----------------------------
# Basic training with default settings
python train_unet_split.py

# Specify dataset and model directory
python train_unet_split.py --root ./balloon_dataset/syn500_dataset --dataset syn500-corner

# Custom training parameters
python train_unet_split.py --root ./balloon_dataset/syn500_dataset --dataset syn500-corner --epochs 150 --batch 16 --lr 5e-5

# Resume training from checkpoint
python train_unet_split.py --resume ./balloon_models/syn500-corner-unet-01.pt

# Full configuration example
python train_unet_split.py \
    --root ./balloon_dataset/syn1000_dataset \
    --dataset syn1000-corner \
    --models-dir ./balloon_models \
    --epochs 100 \
    --batch 8 \
    --lr 1e-4 \
    --patience 20 \
    --wandb-proj balloon-seg-experiments \
    --run-name syn1000-corner-experiment-01

# Disable early stopping (large patience)
python train_unet_split.py --root ./balloon_dataset/syn500_dataset --patience 999

Available Arguments:
-------------------
--root          : Dataset root directory (contains train/val/test folders)
--dataset       : Dataset name for model naming and wandb
--models-dir    : Directory to save trained models
--resume        : Path to checkpoint to resume training
--batch         : Batch size
--epochs        : Number of training epochs
--lr            : Learning rate
--patience      : Early stopping patience (epochs without improvement)
--wandb-proj    : Wandb project name
--run-name      : Wandb run name

Configuration (CFG):
-------------------
    # メモリ監視・緊急停止
    "ENABLE_EMERGENCY_STOP": True,   # False にすると緊急停止を無効化（警告のみ）
    "EMERGENCY_GPU_THRESHOLD": 0.95,  # GPU使用率の緊急停止閾値 (0.0-1.0)
    "EMERGENCY_RAM_THRESHOLD": 0.90,  # RAM使用率の緊急停止閾値 (0.0-1.0)

    # メモリ最適化
    "USE_AMP": True,              # 混合精度学習（-35〜50% VRAM）
    "USE_GRAD_CHECKPOINT": True,  # 勾配チェックポイント（-20〜40% VRAM）
    "USE_CHANNELS_LAST": True,    # channels-last メモリレイアウト（高速化）
"""

import glob, random, time, os
from pathlib import Path
import psutil  # システムリソース監視用
import gc  # ガベージコレクション
import sys  # 緊急終了用
import argparse  # コマンドライン引数

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import wandb

# --------------------------------------------------------------------
CFG = {
    # データセット
    "ROOT":        Path("./balloon_dataset/real200_dataset"),  # train/val フォルダのルート
    "IMG_SIZE":    (384, 512),  # (height, width) = 縦384 × 横512 (メモリ削減 & 縦長対応)

    # 学習
    "BATCH":       8,           # バッチサイズ8に戻す（画像サイズ削減でメモリ節約）
    "EPOCHS":      100,         # バッチ8なら100エポック
    "LR":          1e-4,        # バッチサイズ8の標準学習率
    "PATIENCE":    30,
    "SEED":        42,

    # メモリ監視・緊急停止
    "ENABLE_EMERGENCY_STOP": True,   # False にすると緊急停止を無効化（警告のみ）
    "EMERGENCY_GPU_THRESHOLD": 0.95,  # GPU使用率の緊急停止閾値 (0.0-1.0)
    "EMERGENCY_RAM_THRESHOLD": 0.90,  # RAM使用率の緊急停止閾値 (0.0-1.0)

    # wandb
    "WANDB_PROJ":  "balloon-seg",
    "DATASET":     "balloon_dataset/real200", # または "real" / "synreal" データセットによって書き換える
    "RUN_NAME":    "",

    "MODELS_DIR":  Path("balloon_models"),

    # 予測マスク出力
    "SAVE_PRED_EVERY": 10,     # 5 → 10 に変更（頻度を半減）
    "PRED_SAMPLE_N":   2,      # 3 → 2 に削減（画像数削減）
    "PRED_DIR":        "predictions",

    # 再開 ckpt
    "RESUME":      "",
}
# --------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True

def check_memory_safety(threshold_gpu=0.90, threshold_ram=0.85):
    """
    メモリ使用量をチェックして危険レベルなら警告
    
    Args:
        threshold_gpu: GPU使用率の危険閾値 (0.0-1.0)
        threshold_ram: RAM使用率の危険閾値 (0.0-1.0)
    
    Returns:
        (is_safe, warning_message)
    """
    warnings = []
    
    # GPU メモリチェック
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_usage = mem_reserved / total_memory
        
        if gpu_usage > threshold_gpu:
            warnings.append(f"⚠️ GPU使用率が危険: {gpu_usage*100:.1f}% ({mem_reserved:.2f}GB/{total_memory:.2f}GB)")
    
    # RAM チェック
    ram = psutil.virtual_memory()
    ram_usage = ram.percent / 100.0
    
    if ram_usage > threshold_ram:
        warnings.append(f"⚠️ RAM使用率が危険: {ram_usage*100:.1f}% ({ram.used/1e9:.2f}GB/{ram.total/1e9:.2f}GB)")
    
    is_safe = len(warnings) == 0
    warning_message = "\n".join(warnings) if warnings else ""
    
    return is_safe, warning_message

def emergency_save_and_exit(model, cfg, epoch, reason):
    """
    緊急時にモデルを保存して安全に終了
    
    Args:
        model: 保存するモデル
        cfg: 設定辞書
        epoch: 現在のエポック
        reason: 終了理由
    """
    print("\n" + "="*60)
    print("🚨 緊急停止モード")
    print("="*60)
    print(f"理由: {reason}")
    print(f"現在のエポック: {epoch}")
    
    # 緊急保存
    emergency_path = f"emergency_epoch{epoch}.pth"
    try:
        torch.save(model.state_dict(), emergency_path)
        print(f"✅ 緊急保存完了: {emergency_path}")
    except Exception as e:
        print(f"❌ 緊急保存失敗: {e}")
    
    # メモリクリア
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
    
    # CPU/RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"RAM: {ram.used/1e9:.1f}GB / {ram.total/1e9:.1f}GB ({ram.percent}%)")
        if ram.percent > 85:
            print("⚠️  WARNING: RAM usage is high!")
    except ImportError:
        print("RAM: (psutil not installed)")
    
    # GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
        print(f"GPU Memory: {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Total: {total_mem:.1f}GB")
        
        # GPU温度（試行）
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            print(f"GPU Temperature: {temp}°C")
            if temp > 80:
                print("⚠️  WARNING: GPU temperature is high!")
            pynvml.nvmlShutdown()
        except:
            print("GPU Temperature: (unable to read)")
    else:
        print("GPU: Not available (using CPU)")
    
    print("="*60 + "\n")

# ---------------- Dataset ----------------
class BalloonDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size):
        self.img_paths = sorted(glob.glob(str(img_dir / "*.png")))
        #jpgに対応
        if not self.img_paths:
            self.img_paths = sorted(glob.glob(str(img_dir / "*.jpg")))
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        self.mask_dir  = mask_dir
        
        # img_size がタプルの場合とintの場合に対応
        if isinstance(img_size, tuple):
            resize_size = img_size  # (H, W)
        else:
            resize_size = (img_size, img_size)  # 正方形
        
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
        
        # channels-lastはバッチテンソルに適用するため、ここでは不要
        # （DataLoaderがバッチを作った後、モデルに渡す前に適用）
        
        return img, mask, stem               # stem を返してファイル名に利用

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
            
            p = torch.sigmoid(model(x)).cpu().numpy()      # (B,1,H,W)
            
            probs.append(p.reshape(-1))                    # flatten
            gts.append(y.numpy().reshape(-1))
            
            # メモリ解放
            del x, y, p
            if batch_idx % 3 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
    probs = np.concatenate(probs)   # shape: [N_pixels]
    gts   = np.concatenate(gts)
    
    # 処理後にメモリクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
        
    return probs, gts

# --- 新規: 閾値最適化 ------------------------------------------
def find_best_threshold(probs, gts, metric='iou', n_steps=101):
    """最適な閾値を検索する（デフォルトは IoU 最大化）

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

    # avoid expensive operations inside python loop if arrays are huge,
    # but a simple loop is fine for typical sizes
    for thr in thresholds:
        pb = (probs >= thr).astype(np.uint8)
        tp = np.logical_and(pb == 1, gts == 1).sum()
        fp = np.logical_and(pb == 1, gts == 0).sum()
        fn = np.logical_and(pb == 0, gts == 1).sum()

        if metric == 'iou':
            denom = tp + fp + fn
            score = tp / denom if denom > 0 else 1.0
        else:  # dice
            denom = 2 * tp + fp + fn
            score = (2 * tp) / denom if denom > 0 else 1.0

        if score > best_score:
            best_score = score
            best_thr = thr
    return float(best_thr), float(best_score)

# -------------- U-Net --------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=1, chs=(64,128,256,512,1024)):
        super().__init__()
        self.downs, in_c = nn.ModuleList(), 3
        for c in chs[:-1]:
            self.downs.append(DoubleConv(in_c, c)); in_c=c
        self.bottleneck = DoubleConv(chs[-2], chs[-1])
        self.ups_tr  = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i-1], 2,2)
                         for i in range(len(chs)-1,0,-1)])
        self.up_convs= nn.ModuleList([DoubleConv(chs[i], chs[i-1])
                         for i in range(len(chs)-1,0,-1)])
        self.out_conv= nn.Conv2d(chs[0], n_classes, 1)
        self.pool = nn.MaxPool2d(2)
    def forward(self,x):
        skips=[]
        for l in self.downs:
            x=l(x); skips.append(x); x=self.pool(x)
        x=self.bottleneck(x)
        for up,conv,sk in zip(self.ups_tr,self.up_convs,skips[::-1]):
            x=up(x); x=torch.cat([x,sk],1); x=conv(x)
        return self.out_conv(x)

# -------------- Loss --------------------
class DiceLoss(nn.Module):
    def forward(self,y_p,y_t,eps=1e-7):
        y_p=torch.sigmoid(y_p)
        inter=2*(y_p*y_t).sum((2,3))
        union=(y_p+y_t).sum((2,3))+eps
        return 1-(inter/union).mean()
class ComboLoss(nn.Module):
    def __init__(self,a=.5):
        super().__init__(); self.bce=nn.BCEWithLogitsLoss(); self.dice=DiceLoss(); self.a=a
    def forward(self,y_p,y_t):
        return self.a*self.bce(y_p,y_t)+(1-self.a)*self.dice(y_p,y_t)

# -------------- Train / Eval ------------
def train_epoch(model, loader, lossf, opt, dev, cfg=None, current_epoch=None):
    model.train()
    run = 0
    
    for batch_idx, (x, y, _) in enumerate(tqdm(loader, desc="train", leave=False)):
        # メモリ安全チェック（10バッチごと）
        if batch_idx % 10 == 0:
            is_safe, warning = check_memory_safety(
                threshold_gpu=cfg.get("EMERGENCY_GPU_THRESHOLD", 0.95) if cfg else 0.95,
                threshold_ram=cfg.get("EMERGENCY_RAM_THRESHOLD", 0.90) if cfg else 0.90
            )
            if not is_safe:
                print(f"\n[Batch {batch_idx}] {warning}")
                # 緊急停止が有効な場合のみ停止
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
        
        # バッチごとにメモリ解放（より頻繁に）
        del x, y, logits, loss
        
        # 5バッチごとにキャッシュクリア
        if batch_idx % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()  # Pythonのガベージコレクション
    
    # エポック終了時にメモリ完全クリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return run / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, dev, cfg=None, current_epoch=None):
    model.eval()
    dice = iou = 0
    
    for batch_idx, (x, y, _) in enumerate(tqdm(loader, desc="eval ", leave=False)):
        # メモリ安全チェック（15バッチごと）
        if batch_idx % 15 == 0:
            is_safe, warning = check_memory_safety(
                threshold_gpu=cfg.get("EMERGENCY_GPU_THRESHOLD", 0.95) if cfg else 0.95,
                threshold_ram=cfg.get("EMERGENCY_RAM_THRESHOLD", 0.90) if cfg else 0.90
            )
            if not is_safe:
                print(f"\n[Eval Batch {batch_idx}] {warning}")
                # 緊急停止が有効な場合のみ停止
                if cfg and cfg.get("ENABLE_EMERGENCY_STOP", True) and current_epoch:
                    emergency_save_and_exit(model, cfg, current_epoch, warning)
                else:
                    print("  → 緊急停止は無効化されています（評価継続）")
        
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        
        p = torch.sigmoid(model(x))
        
        pb = (p > .5).float()
        inter = (pb * y).sum((2,3))
        union = (pb + y - pb * y).sum((2,3))
        
        # Dice計算
        dice_den = (pb + y).sum((2,3)) + 1e-7
        dice += torch.where(dice_den > 1e-6, 2*inter/dice_den, torch.ones_like(dice_den)).mean().item() * x.size(0)
        
        # IoU計算
        iou_den = union + 1e-7
        iou += torch.where(iou_den > 1e-6, inter/iou_den, torch.ones_like(iou_den)).mean().item() * x.size(0)
        
        # バッチごとにメモリ解放
        del x, y, p, pb, inter, union, dice_den, iou_den
        
        # 5バッチごとにキャッシュクリア
        if batch_idx % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # 評価終了時にメモリ完全クリア
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

    # 使う閾値（事前に計算されていればそれを使う。なければ 0.5）
    thr = float(cfg.get("BEST_THR", 0.5))

    cnt = 0
    for x, y, stem in loader:
        for i in range(len(x)):
            if cnt >= cfg["PRED_SAMPLE_N"]: break
            img = x[i:i+1].to(device)
            gt  = y[i]
            
            pred = torch.sigmoid(model(img))[0,0]          # (H,W)
            
            # 二値化に最適閾値を使う
            pred_bin = (pred > thr).cpu().numpy()*255
            gt_np    = gt[0].cpu().numpy()*255

            # Save PNG
            out_path = pred_dir / f"pred_{epoch:03}_{stem[i]}.png"
            Image.fromarray(pred_bin.astype(np.uint8)).save(out_path)

            # wandb が有効な場合のみ wandb 用の画像を作成してログ
            if cfg.get("USE_WANDB", True):
                # wandb image (stack original, GT, pred)
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
                
                # メモリ解放（wandb 用のバッファ）
                del orig_np, orig_small, gt_small, pred_small, trio

            cnt += 1
            
            # メモリ解放
            del img, gt, pred, pred_bin, gt_np
            
        if cnt >= cfg["PRED_SAMPLE_N"]: break

    if cfg.get("USE_WANDB", True) and images_for_wandb:
        wandb.log({f"pred_samples_epoch_{epoch}": images_for_wandb})
    elif not cfg.get("USE_WANDB", True):
        # 簡単な通知
        print(f"[Info] wandb disabled: saved {cnt} prediction images to {pred_dir}")
    
    # 予測保存後にメモリクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------- Main ---------------------
def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='U-Net Training Script')
    
    # データセット関連
    parser.add_argument('--root', type=str, default=None,
                        help='Dataset root directory (default: use CFG["ROOT"])')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name for wandb/model tag (default: use CFG["DATASET"])')
    
    # モデル関連
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
    # 新規オプション: wandb を無効化
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging and initialization')
    
    return parser.parse_args()

def main():
    # コマンドライン引数を解析
    args = parse_args()
    
    # CFGを上書き
    cfg=CFG.copy()
    
    if args.root:
        cfg["ROOT"] = Path(args.root)
        print(f"📁 Dataset root: {cfg['ROOT']}")
    
    if args.dataset:
        cfg["DATASET"] = args.dataset
        print(f"📊 Dataset name: {cfg['DATASET']}")
    
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

    # 新規: wandb を使うかどうかの設定
    cfg["USE_WANDB"] = not getattr(args, "no_wandb", False)
    if not cfg["USE_WANDB"]:
        print("🚫 wandb logging is DISABLED. All wandb operations will be skipped.")

    seed_everything(cfg["SEED"])
    dev="cuda" if torch.cuda.is_available() else "cpu"
    
    # システムリソース確認
    print_system_info()
    
    # メモリ監視設定を表示
    print("\n" + "="*60)
    print("🛡️  自動メモリ監視機能 有効")
    print("="*60)
    if CFG.get("ENABLE_EMERGENCY_STOP", True):
        print("緊急停止: ✅ 有効")
        print("監視閾値: (512×512, バッチ8用)")
        print(f"  - GPU使用率: {CFG.get('EMERGENCY_GPU_THRESHOLD', 0.95)*100:.0f}% で緊急停止")
        print(f"  - RAM使用率: {CFG.get('EMERGENCY_RAM_THRESHOLD', 0.90)*100:.0f}% で緊急停止")
    else:
        print("緊急停止: ⚠️  無効（警告のみ表示）")
        print("  ※ メモリ不足でもPCクラッシュするまで学習継続します")
    print("チェック頻度:")
    print("  - Train: 10バッチごと")
    print("  - Eval:  15バッチごと")
    print("  - エポック開始時: 毎回")
    print("緊急停止時の動作:")
    print("  - 現在のモデルを emergency_epochN.pth として保存")
    print("  - メモリをクリアして安全に終了")
    print("="*60 + "\n")
    
    # データセット名からファイル名用のプレフィックスを作成（パス区切り文字を除去）
    dataset_name = cfg["DATASET"].replace("/", "-").replace("\\", "-")
    prefix = f"{dataset_name}-unet"
    version = next_version(cfg["MODELS_DIR"], prefix)
    model_tag = f"{prefix}-{version}"          # syn200_allsize_dataset-unet-01 など

    # wandb の run 名が空ならここで入れる
    if not cfg["RUN_NAME"]:
        cfg["RUN_NAME"] = model_tag

    # wandb の初期化はオプション化
    if cfg.get("USE_WANDB", True):
        wandb.init(project=CFG["WANDB_PROJ"], name=CFG["RUN_NAME"], config=CFG)
        run_dir = Path(wandb.run.dir)
    else:
        # wandb を使わない場合はローカルの run ディレクトリを作成
        run_dir = Path("runs") / cfg["RUN_NAME"]
        run_dir.mkdir(parents=True, exist_ok=True)

    root=cfg["ROOT"]
    train_ds=BalloonDataset(root/"train/images",root/"train/masks",cfg["IMG_SIZE"])
    val_ds  =BalloonDataset(root/"val/images"  ,root/"val/masks"  ,cfg["IMG_SIZE"])
    #test_ds =BalloonDataset(root/"test/images" ,root/"test/masks" ,cfg["IMG_SIZE"])

    dl_tr=DataLoader(train_ds,batch_size=cfg["BATCH"],shuffle=True ,num_workers=0,pin_memory=False)
    dl_va=DataLoader(val_ds  ,batch_size=cfg["BATCH"],shuffle=False,num_workers=0,pin_memory=False)
    #dl_te=DataLoader(test_ds ,batch_size=cfg["BATCH"],shuffle=False,num_workers=4,pin_memory=True)

    # モデル作成
    model = UNet().to(dev)
    
    if cfg["RESUME"]:
        model.load_state_dict(torch.load(cfg["RESUME"],map_location=dev)); print("Resumed")

    opt=torch.optim.AdamW(model.parameters(),lr=cfg["LR"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["EPOCHS"])
    lossf=ComboLoss()

    best_iou=0; patience=0
    for ep in range(1,cfg["EPOCHS"]+1):
        t=time.time()
        
        # エポック開始時のメモリ安全チェック
        is_safe, warning = check_memory_safety(
            threshold_gpu=cfg.get("EMERGENCY_GPU_THRESHOLD", 0.95) - 0.02,  # やや緩めに
            threshold_ram=cfg.get("EMERGENCY_RAM_THRESHOLD", 0.90) - 0.02
        )
        if not is_safe:
            print(f"\n[Epoch {ep} Start] {warning}")
            # 緊急停止が有効な場合のみ停止
            if cfg.get("ENABLE_EMERGENCY_STOP", True):
                emergency_save_and_exit(model, cfg, ep, f"Epoch開始時のメモリ不足\n{warning}")
            else:
                print("  → 緊急停止は無効化されています（学習継続）")
        
        # エポック開始時のメモリ状態（10エポックごと）
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
            
            # PR Curveをダウンサンプリング（メモリ削減）
            sample_indices = np.linspace(0, len(rec)-1, min(200, len(rec)), dtype=int)
            rec_sampled = rec[sample_indices]
            prec_sampled = prec[sample_indices]
            
            if cfg.get("USE_WANDB", True):
                wandb.log({"epoch": ep,
                        "val/pr_auc": pr_auc,
                        "val/pr_curve": wandb.plot.line(
                                wandb.Table(data=np.column_stack([rec_sampled, prec_sampled]),
                                            columns=["recall","precision"]),
                                "recall", "precision",
                                title=f"PR Curve ep{ep} (AUC={pr_auc:.3f})")})
            else:
                print(f"[Info] PR AUC (ep {ep}): {pr_auc:.4f} (wandb disabled)")

            # ここで閾値最適化を行い、cfg と run_dir に保存
            try:
                best_thr, best_score = find_best_threshold(probs, gts, metric='iou')
                cfg['BEST_THR'] = best_thr
                thr_path = run_dir / 'best_threshold.txt'
                with open(thr_path, 'w', encoding='utf-8') as f:
                    f.write(f"{best_thr:.4f}\n")
                print(f"[Info] Best threshold (IoU) = {best_thr:.3f} (IoU={best_score:.4f}) -> saved: {thr_path}")
                if cfg.get("USE_WANDB", True):
                    wandb.log({"val/best_threshold": best_thr, "val/best_threshold_iou": best_score})
            except Exception as e:
                print(f"[Warning] threshold optimization failed: {e}")
            
            # PR Curve後にメモリクリア
            del probs, gts, prec, rec, sample_indices, rec_sampled, prec_sampled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # --- 予測 PNG 保存 & wandb 画像ログ ---
        save_predictions(model, dl_va, cfg, ep, run_dir, dev)
        
        # 予測保存後もメモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # wandb にログ（オプション）
        if cfg.get("USE_WANDB", True):
            wandb.log({"epoch":ep,"loss":tr_loss,
                       "val_dice":va_dice,"val_iou":va_iou,
                       "lr":sched.get_last_lr()[0]})
        else:
            print(f"[Epoch {ep}] loss={tr_loss:.4f} dice={va_dice:.4f} iou={va_iou:.4f}  {time.time()-t:.1f}s")

        if va_iou>best_iou:
            best_iou,patience=va_iou,0
            ckpt_wandb = run_dir / f"best_ep{ep:03}_iou{va_iou:.4f}.pt"
            # torch.save(model.state_dict(), ckpt_wandb)
            # wandb.save(str(ckpt_wandb))
            torch.save(model.state_dict(), ckpt_wandb)

            # models/ にもコピー（固定ファイル名）
            cfg["MODELS_DIR"].mkdir(parents=True, exist_ok=True)
            ckpt_models = cfg["MODELS_DIR"] / f"{model_tag}.pt"
            torch.save(model.state_dict(), ckpt_models)
            print(f"✓ Best model saved! IoU: {va_iou:.4f} -> {ckpt_models}")
        else:
            patience+=1
            if patience>=cfg["PATIENCE"]:
                print("Early stopping."); break

    # # 1) ベスト重みをロード
    # best_model_path = CFG["MODELS_DIR"] / f"{model_tag}.pt"
    # model.load_state_dict(torch.load(best_model_path, map_location=dev))

    # # 2) test セットを読み込み
    # test_ds = BalloonDataset(root/"test/images", root/"test/masks", cfg["IMG_SIZE"])
    # dl_te   = DataLoader(test_ds, batch_size=cfg["BATCH"], shuffle=False,
    #                      num_workers=4, pin_memory=True)

    # # 3) 推論＆ログ
    # test_dice, test_iou = eval_epoch(model, dl_te, dev)
    # print(f"[TEST] Dice={test_dice:.4f}  IoU={test_iou:.4f}")
    # wandb.log({"test_dice": test_dice, "test_iou": test_iou})

if __name__=="__main__":
    main()
