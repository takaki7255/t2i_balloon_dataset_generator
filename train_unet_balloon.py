"""
U-Net で漫画吹き出しマスクを学習
  * wandb 監視
  * tqdm 進捗バー
  * EarlyStopping / CosineLR / 再開対応
  *
  * 設定は下の CFG 辞書を編集して下さい
"""

import glob, random, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb

# ---------------------------------------------------------------------
CFG = {
    # データ関連
    "IMG_DIR":      Path("results"),
    "MASK_DIR":     Path("results_mask"),
    "IMG_SIZE":     512,
    "VAL_SPLIT":    0.1,       # train:val = 9:1
    # 学習設定
    "BATCH":        8,
    "EPOCHS":       50,
    "LR":           1e-4,
    "PATIENCE":     10,        # early stop
    "SEED":         42,
    # wandb
    "WANDB_PROJ":   "balloon-seg",
    "RUN_NAME":     "unet-baseline",
    # 再開チェックポイント (空文字なら無視)
    "RESUME":       "",        # e.g. "runs/….pt"
}
# ---------------------------------------------------------------------


# -------------------- ユーティリティ --------------------
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True

# -------------------- Dataset ---------------------------
class BalloonDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size):
        self.img_paths = sorted(glob.glob(str(img_dir / "*.png")))
        self.mask_dir  = mask_dir
        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        stem      = Path(img_path).stem
        mask_path = self.mask_dir / f"{stem}_mask.png"
        img  = self.img_tf(Image.open(img_path).convert("RGB"))
        mask = self.mask_tf(Image.open(mask_path).convert("L"))
        mask = (mask > 0.5).float()
        return img, mask

# -------------------- モデル ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, n_classes=1, chs=(64,128,256,512,1024)):
        super().__init__()
        self.downs = nn.ModuleList()
        in_c = 3
        for c in chs[:-1]:
            self.downs.append(DoubleConv(in_c, c)); in_c = c
        self.bottleneck = DoubleConv(chs[-2], chs[-1])
        self.ups_tr  = nn.ModuleList([
            nn.ConvTranspose2d(chs[i], chs[i-1], 2, stride=2)
            for i in range(len(chs)-1, 0, -1)
        ])
        self.up_convs = nn.ModuleList([
            DoubleConv(chs[i], chs[i-1])
            for i in range(len(chs)-1, 0, -1)
        ])
        self.out_conv = nn.Conv2d(chs[0], n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups_tr, self.up_convs, skips[::-1]):
            x = up(x); x = torch.cat([x, skip], 1); x = conv(x)
        return self.out_conv(x)

# -------------------- 損失 ------------------------------
class DiceLoss(nn.Module):
    def forward(self, y_pred, y_true, eps=1e-7):
        y_pred = torch.sigmoid(y_pred)
        inter  = 2*(y_pred*y_true).sum((2,3))
        union  = (y_pred + y_true).sum((2,3)) + eps
        return 1 - (inter/union).mean()

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(); self.dice = DiceLoss(); self.a = alpha
    def forward(self, y_pred, y_true):
        return self.a*self.bce(y_pred, y_true) + (1-self.a)*self.dice(y_pred, y_true)

# -------------------- 1 エポック学習 ---------------------
def train_epoch(model, loader, loss_fn, opt, device):
    model.train(); running = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward(); opt.step()
        running += loss.item()*x.size(0)
    return running/len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval(); dice=iou=0
    for x, y in tqdm(loader, desc="val  ", leave=False):
        x, y = x.to(device), y.to(device)
        p = torch.sigmoid(model(x)); p_bin = (p>0.5).float()
        inter = (p_bin*y).sum((2,3)); union = (p_bin + y - p_bin*y).sum((2,3))
        dice += (2*inter/(p_bin.sum((2,3))+y.sum((2,3))+1e-7)).mean().item()*x.size(0)
        iou  += (inter/(union+1e-7)).mean().item()*x.size(0)
    n = len(loader.dataset); return dice/n, iou/n

# -------------------- メイン ----------------------------
def main():
    cfg = CFG
    seed_everything(cfg["SEED"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # wandb
    wandb.init(project=cfg["WANDB_PROJ"], name=cfg["RUN_NAME"], config=cfg)
    run_dir = Path(wandb.run.dir)

    ds = BalloonDataset(cfg["IMG_DIR"], cfg["MASK_DIR"], cfg["IMG_SIZE"])
    val_sz = int(len(ds)*cfg["VAL_SPLIT"])
    tr_ds, va_ds = random_split(ds, [len(ds)-val_sz, val_sz],
                                generator=torch.Generator().manual_seed(cfg["SEED"]))
    dl_tr = DataLoader(tr_ds, batch_size=cfg["BATCH"], shuffle=True,  num_workers=4, pin_memory=True)
    dl_va = DataLoader(va_ds, batch_size=cfg["BATCH"], shuffle=False, num_workers=4, pin_memory=True)

    model = UNet().to(device)
    if cfg["RESUME"]:
        model.load_state_dict(torch.load(cfg["RESUME"], map_location=device))
        print("Resumed:", cfg["RESUME"])

    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["LR"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["EPOCHS"])
    lossf = ComboLoss()

    best_iou=0; patience=0
    for ep in range(1, cfg["EPOCHS"]+1):
        t0=time.time()
        tr_loss = train_epoch(model, dl_tr, lossf, opt, device)
        va_dice, va_iou = eval_epoch(model, dl_va, device)
        sched.step()

        wandb.log({"epoch":ep,"loss":tr_loss,"val_dice":va_dice,"val_iou":va_iou,
                   "lr":sched.get_last_lr()[0]})
        print(f"[{ep:03}] loss={tr_loss:.4f} dice={va_dice:.4f} iou={va_iou:.4f}"
              f"  {time.time()-t0:.1f}s")

        if va_iou>best_iou:
            best_iou=va_iou; patience=0
            ckpt=run_dir/f"best_ep{ep:03}_iou{va_iou:.4f}.pt"
            torch.save(model.state_dict(), ckpt); wandb.save(str(ckpt))
        else:
            patience+=1
            if patience>=cfg["PATIENCE"]:
                print("Early stopping."); break

    wandb.finish()

if __name__=="__main__":
    main()
