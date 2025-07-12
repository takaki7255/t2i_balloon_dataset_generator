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
    "MODEL_TAG":   "real-unet-01",     # 例： models/real-unet-01.pt
    "DATA_ROOT":   Path("real_dataset"),  # test/images, test/masks が入った実写専用 DS
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
        self.img_paths = sorted(glob.glob(str(img_dir/"*.png")))
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


# ---------------- Metric -----------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); dice=iou=0
    for x, y, _ in tqdm(loader, desc="test", leave=False):
        x, y = x.to(device), y.to(device)
        p = torch.sigmoid(model(x)); pb = (p>.5).float()
        inter=(pb*y).sum((2,3)); union=(pb+y-pb*y).sum((2,3))
        dice+=(2*inter/(pb.sum((2,3))+y.sum((2,3))+1e-7)).mean().item()*x.size(0)
        iou +=(inter/(union+1e-7)).mean().item()*x.size(0)
    n=len(loader.dataset); return dice/n, iou/n


# ---------------- Save Predictions -------
@torch.no_grad()
def save_predictions(model, loader, cfg, run_dir, device):
    pred_dir = run_dir/cfg["PRED_DIR"]; pred_dir.mkdir(exist_ok=True)
    img_logs=[]; saved=0
    for x, y, stem in loader:
        for i in range(len(x)):
            if saved >= cfg["SAVE_PRED_N"]: break
            img = x[i:i+1].to(device)
            gt  = y[i]
            pr  = torch.sigmoid(model(img))[0,0]
            pr_bin = (pr>0.5).cpu().numpy()*255
            gt_np  = gt[0].cpu().numpy()*255
            Image.fromarray(pr_bin.astype(np.uint8)).save(pred_dir/f"{stem[i]}_pred.png")
            orig = (img[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
            trio = np.concatenate([orig, np.stack([gt_np]*3,2), np.stack([pr_bin]*3,2)],1)
            img_logs.append(wandb.Image(trio, caption=stem[i]))
            saved+=1
        if saved>=cfg["SAVE_PRED_N"]: break
    wandb.log({"test_samples": img_logs})


# ---------------- Main -------------------
def main():
    cfg=CFG
    if not cfg["RUN_NAME"]:
        cfg["RUN_NAME"]=cfg["MODEL_TAG"]+"-test"
    seed_everything(cfg["SEED"])
    dev="cuda" if torch.cuda.is_available() else "cpu"

    # ---------- wandb ----------
    wandb.init(project=cfg["WANDB_PROJ"], name=cfg["RUN_NAME"], config=cfg)
    run_dir=Path(wandb.run.dir)

    # ---------- data ----------
    test_ds = BalloonDataset(cfg["DATA_ROOT"]/ "test/images",
                             cfg["DATA_ROOT"]/ "test/masks",
                             cfg["IMG_SIZE"])
    dl = DataLoader(test_ds, batch_size=cfg["BATCH"], shuffle=False,
                    num_workers=4, pin_memory=True)

    # ---------- model ----------
    model = UNet().to(dev)
    ckpt = Path("models")/f"{cfg['MODEL_TAG']}.pt"
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
    model.load_state_dict(torch.load(ckpt, map_location=dev))

    # ---------- evaluate ----------
    dice,iou = evaluate(model, dl, dev)
    print(f"[TEST] Dice={dice:.4f}  IoU={iou:.4f}")
    wandb.log({"test_dice":dice, "test_iou":iou})

    # ---------- save samples ----------
    save_predictions(model, dl, cfg, run_dir, dev)
    wandb.finish()

if __name__=="__main__":
    main()
