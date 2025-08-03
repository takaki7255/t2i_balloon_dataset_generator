"""
U-Net training (train/val/test folders) + periodic prediction dump
"""

import glob, random, time, os
from pathlib import Path

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
    "ROOT":        Path("syn500-aug_dataset01"),  # train/val フォルダのルート
    "IMG_SIZE":    512,

    # 学習
    "BATCH":       8,
    "EPOCHS":      100,
    "LR":          1e-4,
    "PATIENCE":    10,
    "SEED":        42,

    # wandb
    "WANDB_PROJ":  "balloon-seg",
    "DATASET":     "syn500-aug", # または "real" / "synreal" データセットによって書き換える
    "RUN_NAME":    "",

    "MODELS_DIR":  Path("models"),

    # 予測マスク出力
    "SAVE_PRED_EVERY": 5,      # エポック間隔
    "PRED_SAMPLE_N":   3,      # 保存枚数
    "PRED_DIR":        "predictions",

    # 再開 ckpt
    "RESUME":      "",
}
# --------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True

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
        img_p = self.img_paths[idx]
        stem   = Path(img_p).stem
        mask_p = self.mask_dir / f"{stem}_mask.png"
        img  = self.img_tf(Image.open(img_p).convert("RGB"))
        mask = self.mask_tf(Image.open(mask_p).convert("L"))
        mask = (mask > .5).float()
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
        for x, y, _ in loader:
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()      # (B,1,H,W)
            probs.append(p.reshape(-1))                    # flatten
            gts.append(y.numpy().reshape(-1))
    probs = np.concatenate(probs)   # shape: [N_pixels]
    gts   = np.concatenate(gts)
    return probs, gts

# -------------- U-Net --------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

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
def train_epoch(model,loader,lossf,opt,dev):
    model.train(); run=0
    for x,y,_ in tqdm(loader,desc="train",leave=False):
        x,y=x.to(dev),y.to(dev); opt.zero_grad()
        loss=lossf(model(x),y); loss.backward(); opt.step()
        run+=loss.item()*x.size(0)
    return run/len(loader.dataset)

@torch.no_grad()
def eval_epoch(model,loader,dev):
    model.eval(); dice=iou=0
    for x,y,_ in tqdm(loader,desc="eval ",leave=False):
        x,y=x.to(dev),y.to(dev)
        p=torch.sigmoid(model(x)); pb=(p>.5).float()
        inter=(pb*y).sum((2,3)); union=(pb+y-pb*y).sum((2,3))
        dice+=(2*inter/(pb.sum((2,3))+y.sum((2,3))+1e-7)).mean().item()*x.size(0)
        iou +=(inter/(union+1e-7)).mean().item()*x.size(0)
    n=len(loader.dataset); return dice/n,iou/n

# -------------- Prediction dump ----------
@torch.no_grad()
def save_predictions(model, loader, cfg, epoch, run_dir, device):
    """val loader 先頭から PRED_SAMPLE_N 枚推論し PNG 保存 & wandb へログ"""
    if epoch % cfg["SAVE_PRED_EVERY"] != 0: return
    model.eval()

    pred_dir = run_dir / cfg["PRED_DIR"]
    pred_dir.mkdir(exist_ok=True)
    images_for_wandb = []

    cnt = 0
    for x, y, stem in loader:
        for i in range(len(x)):
            if cnt >= cfg["PRED_SAMPLE_N"]: break
            img = x[i:i+1].to(device)
            gt  = y[i]
            pred = torch.sigmoid(model(img))[0,0]          # (H,W)
            pred_bin = (pred > 0.5).cpu().numpy()*255
            gt_np    = gt[0].cpu().numpy()*255

            # Save PNG
            out_path = pred_dir / f"pred_{epoch:03}_{stem[i]}.png"
            Image.fromarray(pred_bin.astype(np.uint8)).save(out_path)

            # wandb image (stack original, GT, pred)
            orig_np = (img[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
            trio = np.concatenate([orig_np,
                                   np.stack([gt_np]*3,2),
                                   np.stack([pred_bin]*3,2)], axis=1)
            images_for_wandb.append(wandb.Image(trio,
                                    caption=f"ep{epoch:03}-{stem[i]}"))
            cnt += 1
        if cnt >= cfg["PRED_SAMPLE_N"]: break

    if images_for_wandb:
        wandb.log({f"pred_samples_epoch_{epoch}": images_for_wandb})

# -------------- Main ---------------------
def main():
    cfg=CFG; seed_everything(cfg["SEED"])
    dev="cuda" if torch.cuda.is_available() else "cpu"
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

    dl_tr=DataLoader(train_ds,batch_size=cfg["BATCH"],shuffle=True ,num_workers=4,pin_memory=True)
    dl_va=DataLoader(val_ds  ,batch_size=cfg["BATCH"],shuffle=False,num_workers=4,pin_memory=True)
    #dl_te=DataLoader(test_ds ,batch_size=cfg["BATCH"],shuffle=False,num_workers=4,pin_memory=True)

    model=UNet().to(dev)
    if cfg["RESUME"]:
        model.load_state_dict(torch.load(cfg["RESUME"],map_location=dev)); print("Resumed")

    opt=torch.optim.AdamW(model.parameters(),lr=cfg["LR"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["EPOCHS"])
    lossf=ComboLoss()

    best_iou=0; patience=0
    for ep in range(1,cfg["EPOCHS"]+1):
        t=time.time()
        tr_loss=train_epoch(model,dl_tr,lossf,opt,dev)
        va_dice,va_iou=eval_epoch(model,dl_va,dev)
        sched.step()

        if ep % 5 == 0:
            probs, gts = collect_probs(model, dl_va, dev)
            prec, rec, _ = precision_recall_curve(gts, probs)
            pr_auc = auc(rec, prec)
            wandb.log({"epoch": ep,
                    "val/pr_auc": pr_auc,
                    "val/pr_curve": wandb.plot.line(
                            wandb.Table(data=np.column_stack([rec, prec]),
                                        columns=["recall","precision"]),
                            "recall", "precision",
                            title=f"PR Curve ep{ep} (AUC={pr_auc:.3f})")})

        # --- 予測 PNG 保存 & wandb 画像ログ ---
        save_predictions(model, dl_va, cfg, ep, run_dir, dev)

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
