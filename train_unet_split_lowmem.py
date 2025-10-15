"""
U-Net training (train/val/test    # メモリ監視・緊急停止
    "ENABLE_EMERGENCY_STOP": True,   # False にすると緊急停止を無効化（警告のみ）
    "EMERGENCY_GPU_THRESHOLD": 0.95,  # GPU使用率の緊急停止閾値 (0.0-1.0)
    "EMERGENCY_RAM_THRESHOLD": 0.90,  # RAM使用率の緊急停止閾値 (0.0-1.0)

    # メモリ最適化
    "USE_AMP": True,              # 混合精度学習（-35〜50% VRAM）
    "USE_GRAD_CHECKPOINT": True,  # 勾配チェックポイント（-20〜40% VRAM）
    "USE_CHANNELS_LAST": True,    # channels-last メモリレイアウト（高速化）

    # wandbrs) + periodic prediction dump
"""

import glob, random, time, os
from pathlib import Path
import psutil  # システムリソース監視用
import gc  # ガベージコレクション
import sys  # 緊急終了用

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import wandb

# --------------------------------------------------------------------
CFG = {
    # データセット
    "ROOT":        Path("syn500-balloon-corner"),  # train/val フォルダのルート
    "IMG_SIZE":    (512, 768),  # (height, width) = 縦512 × 横768 (メモリ削減 & 縦長対応)

    # 学習
    "BATCH":       8,           # バッチサイズ8に戻す（画像サイズ削減でメモリ節約）
    "EPOCHS":      100,         # バッチ8なら100エポック
    "LR":          1e-4,        # バッチサイズ8の標準学習率
    "PATIENCE":    10,
    "SEED":        42,

    # メモリ監視・緊急停止
    "ENABLE_EMERGENCY_STOP": True,   # False にすると緊急停止を無効化（警告のみ）
    "EMERGENCY_GPU_THRESHOLD": 0.95,  # GPU使用率の緊急停止閾値 (0.0-1.0)
    "EMERGENCY_RAM_THRESHOLD": 0.90,  # RAM使用率の緊急停止閾値 (0.0-1.0)

    # メモリ最適化
    "USE_AMP": True,              # 混合精度学習（-35〜50% VRAM）
    "USE_GRAD_CHECKPOINT": False,  # 勾配チェックポイント（-20〜40% VRAM、学習時間+20〜50%）
    "USE_CHANNELS_LAST": True,    # channels-last メモリレイアウト（高速化＋省メモリ）

    # wandb
    "WANDB_PROJ":  "balloon-seg",
    "DATASET":     "syn500-corner", # または "real" / "synreal" データセットによって書き換える
    "RUN_NAME":    "",

    "MODELS_DIR":  Path("models"),

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

def collect_probs(model, loader, device, use_amp=False, use_channels_last=False):
    model.eval()
    probs, gts = [], []
    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(loader):
            x = x.to(device)
            
            # channels-last 適用
            if use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            
            # AMP対応の推論
            if use_amp:
                with autocast(dtype=torch.float16):
                    p = torch.sigmoid(model(x)).cpu().numpy()      # (B,1,H,W)
            else:
                p = torch.sigmoid(model(x)).cpu().numpy()
            
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

# -------------- U-Net --------------
# -------------- U-Net --------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            # 勾配チェックポイント使用（学習時のみ）
            def f1(x):
                return self.relu1(self.bn1(self.conv1(x)))
            def f2(x):
                return self.relu2(self.bn2(self.conv2(x)))
            x = checkpoint(f1, x, use_reentrant=False)
            x = checkpoint(f2, x, use_reentrant=False)
            return x
        else:
            # 通常の forward（推論時または無効時）
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            return x

class UNet(nn.Module):
    def __init__(self, n_classes=1, chs=(64,128,256,512,1024), use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.downs, in_c = nn.ModuleList(), 3
        for c in chs[:-1]:
            self.downs.append(DoubleConv(in_c, c, use_checkpoint)); in_c=c
        self.bottleneck = DoubleConv(chs[-2], chs[-1], use_checkpoint)
        self.ups_tr  = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i-1], 2,2)
                         for i in range(len(chs)-1,0,-1)])
        self.up_convs= nn.ModuleList([DoubleConv(chs[i], chs[i-1], use_checkpoint)
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
def train_epoch(model, loader, lossf, opt, dev, scaler, cfg=None, current_epoch=None):
    model.train()
    run = 0
    use_amp = cfg.get("USE_AMP", False) if cfg else False
    use_channels_last = cfg.get("USE_CHANNELS_LAST", False) if cfg else False
    
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
        
        # channels-last 適用（バッチテンソルに対して）
        if use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
            y = y.contiguous(memory_format=torch.channels_last)
        
        opt.zero_grad(set_to_none=True)
        
        # 混合精度学習
        if use_amp:
            with autocast(dtype=torch.float16):
                logits = model(x)
                loss = lossf(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
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
    use_amp = cfg.get("USE_AMP", False) if cfg else False
    use_channels_last = cfg.get("USE_CHANNELS_LAST", False) if cfg else False
    
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
        
        # channels-last 適用（バッチテンソルに対して）
        if use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
            y = y.contiguous(memory_format=torch.channels_last)
        
        # 混合精度推論
        if use_amp:
            with autocast(dtype=torch.float16):
                p = torch.sigmoid(model(x))
        else:
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
    use_amp = cfg.get("USE_AMP", False)
    use_channels_last = cfg.get("USE_CHANNELS_LAST", False)

    pred_dir = run_dir / cfg["PRED_DIR"]
    pred_dir.mkdir(exist_ok=True)
    images_for_wandb = []

    cnt = 0
    for x, y, stem in loader:
        for i in range(len(x)):
            if cnt >= cfg["PRED_SAMPLE_N"]: break
            img = x[i:i+1].to(device)
            gt  = y[i]
            
            # channels-last 適用
            if use_channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
            
            # AMP対応の推論
            if use_amp:
                with autocast(dtype=torch.float16):
                    pred = torch.sigmoid(model(img))[0,0]          # (H,W)
            else:
                pred = torch.sigmoid(model(img))[0,0]
            
            pred_bin = (pred > 0.5).cpu().numpy()*255
            gt_np    = gt[0].cpu().numpy()*255

            # Save PNG
            out_path = pred_dir / f"pred_{epoch:03}_{stem[i]}.png"
            Image.fromarray(pred_bin.astype(np.uint8)).save(out_path)

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
            
            images_for_wandb.append(wandb.Image(trio,
                                    caption=f"ep{epoch:03}-{stem[i]}"))
            cnt += 1
            
            # メモリ解放
            del img, gt, pred, pred_bin, gt_np, orig_np
            del orig_small, gt_small, pred_small, trio
            
        if cnt >= cfg["PRED_SAMPLE_N"]: break

    if images_for_wandb:
        wandb.log({f"pred_samples_epoch_{epoch}": images_for_wandb})
    
    # 予測保存後にメモリクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------- Main ---------------------
def main():
    cfg=CFG; seed_everything(cfg["SEED"])
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
    
    prefix = f"{CFG['DATASET']}-unet"
    version = next_version(CFG["MODELS_DIR"], prefix)
    model_tag = f"{prefix}-{version}"          # synthetic-unet-01 など

    # wandb の run 名が空ならここで入れる
    if not CFG["RUN_NAME"]:
        CFG["RUN_NAME"] = model_tag

    wandb.init(project=CFG["WANDB_PROJ"], name=CFG["RUN_NAME"], config=CFG)
    run_dir = Path(wandb.run.dir)

    root=cfg["ROOT"]
    train_ds=BalloonDataset(root/"train/images",root/"train/masks",cfg["IMG_SIZE"])
    val_ds  =BalloonDataset(root/"val/images"  ,root/"val/masks"  ,cfg["IMG_SIZE"])
    #test_ds =BalloonDataset(root/"test/images" ,root/"test/masks" ,cfg["IMG_SIZE"])

    dl_tr=DataLoader(train_ds,batch_size=cfg["BATCH"],shuffle=True ,num_workers=0,pin_memory=False)
    dl_va=DataLoader(val_ds  ,batch_size=cfg["BATCH"],shuffle=False,num_workers=0,pin_memory=False)
    #dl_te=DataLoader(test_ds ,batch_size=cfg["BATCH"],shuffle=False,num_workers=4,pin_memory=True)

    # モデル作成（勾配チェックポイント対応）
    model = UNet(use_checkpoint=cfg.get("USE_GRAD_CHECKPOINT", False)).to(dev)
    
    # channels-last メモリレイアウト
    if torch.cuda.is_available() and cfg.get("USE_CHANNELS_LAST", False):
        model = model.to(memory_format=torch.channels_last)
        print("✅ Channels-last メモリレイアウトを適用")
    
    if cfg["RESUME"]:
        model.load_state_dict(torch.load(cfg["RESUME"],map_location=dev)); print("Resumed")

    opt=torch.optim.AdamW(model.parameters(),lr=cfg["LR"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["EPOCHS"])
    lossf=ComboLoss()
    
    # 混合精度学習用のGradScaler（新しいAPI）
    if cfg.get("USE_AMP", False):
        try:
            # PyTorch 2.0以降の推奨API
            from torch.amp import GradScaler as NewGradScaler
            scaler = NewGradScaler('cuda')
        except ImportError:
            # 古いバージョンへのフォールバック
            scaler = GradScaler()
    else:
        scaler = None
    
    # メモリ最適化設定の表示
    print("\n" + "="*60)
    print("⚡ メモリ最適化設定")
    print("="*60)
    print(f"混合精度（AMP）:         {'✅ 有効' if cfg.get('USE_AMP', False) else '❌ 無効'}")
    print(f"勾配チェックポイント:     {'✅ 有効' if cfg.get('USE_GRAD_CHECKPOINT', False) else '❌ 無効'}")
    print(f"Channels-last:          {'✅ 有効' if cfg.get('USE_CHANNELS_LAST', False) else '❌ 無効'}")
    print("="*60 + "\n")

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
        
        tr_loss=train_epoch(model, dl_tr, lossf, opt, dev, scaler, cfg, ep)
        va_dice,va_iou=eval_epoch(model, dl_va, dev, cfg, ep)
        sched.step()

        if ep % 5 == 0:
            probs, gts = collect_probs(model, dl_va, dev, cfg.get("USE_AMP", False), cfg.get("USE_CHANNELS_LAST", False))
            prec, rec, _ = precision_recall_curve(gts, probs)
            pr_auc = auc(rec, prec)
            
            # PR Curveをダウンサンプリング（メモリ削減）
            sample_indices = np.linspace(0, len(rec)-1, min(200, len(rec)), dtype=int)
            rec_sampled = rec[sample_indices]
            prec_sampled = prec[sample_indices]
            
            wandb.log({"epoch": ep,
                    "val/pr_auc": pr_auc,
                    "val/pr_curve": wandb.plot.line(
                            wandb.Table(data=np.column_stack([rec_sampled, prec_sampled]),
                                        columns=["recall","precision"]),
                            "recall", "precision",
                            title=f"PR Curve ep{ep} (AUC={pr_auc:.3f})")})
            
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

        wandb.log({"epoch":ep,"loss":tr_loss,
                   "val_dice":va_dice,"val_iou":va_iou,
                   "lr":sched.get_last_lr()[0]})
        print(f"[{ep:03}] loss={tr_loss:.4f} dice={va_dice:.4f} iou={va_iou:.4f}  {time.time()-t:.1f}s")

        if va_iou>best_iou:
            best_iou,patience=va_iou,0
            ckpt_wandb = run_dir / f"best_ep{ep:03}_iou{va_iou:.4f}.pt"
            # torch.save(model.state_dict(), ckpt_wandb)
            # wandb.save(str(ckpt_wandb))
            torch.save(model.state_dict(), ckpt_wandb)

            # models/ にもコピー（固定ファイル名）
            ckpt_models = CFG["MODELS_DIR"] / f"{model_tag}.pt"
            torch.save(model.state_dict(), ckpt_models)
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
