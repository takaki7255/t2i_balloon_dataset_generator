"""
DeepLab v3+ training (train/val/test folders) + periodic prediction dump
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
import torch.nn.functional as F

# --------------------------------------------------------------------
CFG = {
    # データセット
    "ROOT":        Path("syn2000_dataset01"),  # train/val フォルダのルート
    "IMG_SIZE":    512,

    # 学習
    "BATCH":       8,
    "EPOCHS":      100,
    "LR":          1e-4,
    "PATIENCE":    10,
    "SEED":        42,

    # wandb
    "WANDB_PROJ":  "balloon-seg-deeplabv3",
    "DATASET":     "syn2000", # または "real" / "synreal" データセットによって書き換える
    "RUN_NAME":    "",

    "MODELS_DIR":  Path("models"),

    # 予測マスク出力
    "SAVE_PRED_EVERY": 5,      # エポック間隔
    "PRED_SAMPLE_N":   3,      # 保存枚数
    "PRED_DIR":        "predictions",

    # 再開 ckpt
    "RESUME":      "",
    
    # DeepLab v3+ specific
    "BACKBONE":    "resnet50",  # resnet50, resnet101
    "OUTPUT_STRIDE": 16,        # 8 or 16
    "ASPP_DILATE": [12, 24, 36],  # ASPP dilation rates
}
# --------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True

class BalloonDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.img_size = img_size
        
        self.mask_files = []
        for img_path in self.img_files:
            fn = Path(img_path).stem
            mask_path = os.path.join(mask_dir, f"{fn}_mask.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, f"{fn}.png")
            self.mask_files.append(mask_path)
    
    def __len__(self): 
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert("RGB")
        msk = Image.open(self.mask_files[idx]).convert("L")
        
        # リサイズ
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        msk = msk.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
        
        # tensor
        img = transforms.ToTensor()(img)
        msk = transforms.ToTensor()(msk)
        
        return img, msk, Path(self.img_files[idx]).stem

def get_pr_auc(probs, gts):
    probs_flat = probs.flatten()
    gts_flat = gts.flatten()
    p,r,_ = precision_recall_curve(gts_flat, probs_flat)
    return auc(r, p), probs, gts

# -------------- DeepLab v3+ Components --------------

class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        size = x.shape[-2:]
        pool = self.gap(x)
        out = self.relu(self.bn(self.conv(pool)))
        return F.interpolate(out, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        modules = []
        
        # 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        
        # atrous convs
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # image pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, low_level_channels):
        super().__init__()
        self.low_level_project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU())
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),  # 48 + 256 = 304
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        self.classifier = nn.Conv2d(256, num_classes, 1)
        
    def forward(self, x, low_level_feat):
        low_level_feat = self.low_level_project(low_level_feat)
        
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feat), dim=1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)
        
        return x

# Simplified ResNet backbone
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, output_stride=16):
        super().__init__()
        self.inplanes = 64
        
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError
        
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])
    
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x, low_level_feat

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet50', output_stride=16):
        super().__init__()
        
        if backbone == 'resnet50':
            self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], output_stride)
            backbone_channels = 2048
            low_level_channels = 256
        elif backbone == 'resnet101':
            self.backbone = ResNet(Bottleneck, [3, 4, 23, 3], output_stride)
            backbone_channels = 2048
            low_level_channels = 256
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")
        
        if output_stride == 16:
            aspp_dilate = [6, 12, 18]
        elif output_stride == 8:
            aspp_dilate = [12, 24, 36]
        else:
            aspp_dilate = [6, 12, 18]
        
        self.aspp = ASPP(backbone_channels, aspp_dilate)
        self.decoder = Decoder(num_classes, backbone, low_level_channels)
    
    def forward(self, x):
        size = x.shape[-2:]
        features, low_level_feat = self.backbone(x)
        x = self.aspp(features)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x

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
        loss=lossf(model(x),y); loss.backward(); opt.step(); run+=loss.item()
    return run/len(loader)

def eval_epoch(model,loader,dev):
    model.eval(); dice,iou,cnt=0,0,0
    all_probs, all_gts = [], []
    with torch.no_grad():
        for x,y,_ in tqdm(loader,desc="eval",leave=False):
            x,y=x.to(dev),y.to(dev); pred=torch.sigmoid(model(x))
            all_probs.append(pred.cpu().numpy())
            all_gts.append(y.cpu().numpy())
            
            p,g=(pred>.5).float(),y
            inter=(p*g).sum((2,3)); union=(p+g).sum((2,3))
            dice+=(2*inter/(union+1e-7)).sum(); iou+=(inter/(union-inter+1e-7)).sum(); cnt+=p.numel()//p.shape[-1]//p.shape[-2]
    all_probs = np.concatenate(all_probs)
    all_gts = np.concatenate(all_gts)
    return dice/cnt, iou/cnt, all_probs, all_gts

def save_pred_samples(model, loader, save_dir, epoch, n_samples=3, dev="cuda"):
    """予測結果をサンプル保存"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    saved = 0
    
    with torch.no_grad():
        for x, y, names in loader:
            if saved >= n_samples:
                break
            
            x, y = x.to(dev), y.to(dev)
            pred = torch.sigmoid(model(x))
            
            for i in range(x.size(0)):
                if saved >= n_samples:
                    break
                
                # 画像を [0,1] → [0,255] に変換
                img = x[i].cpu().permute(1,2,0).numpy()
                img = (img * 255).astype(np.uint8)
                
                # マスクを [0,1] → [0,255] に変換
                gt = (y[i,0].cpu().numpy() * 255).astype(np.uint8)
                pr = (pred[i,0].cpu().numpy() * 255).astype(np.uint8)
                
                # 3つを並べて保存
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img); axes[0].set_title("Input"); axes[0].axis("off")
                axes[1].imshow(gt, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
                axes[2].imshow(pr, cmap="gray"); axes[2].set_title("Prediction"); axes[2].axis("off")
                
                plt.tight_layout()
                plt.savefig(f"{save_dir}/epoch_{epoch:03d}_sample_{saved:02d}_{names[i]}.png", 
                           bbox_inches="tight", dpi=100)
                plt.close()
                
                saved += 1

def main():
    seed_everything(CFG["SEED"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    
    # データ読み込み
    root = CFG["ROOT"]
    ds_tr = BalloonDataset(root/"train/images", root/"train/masks", CFG["IMG_SIZE"])
    ds_va = BalloonDataset(root/"val/images",   root/"val/masks",   CFG["IMG_SIZE"])
    
    dl_tr = DataLoader(ds_tr, batch_size=CFG["BATCH"], shuffle=True,  num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=CFG["BATCH"], shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train: {len(ds_tr)}, Val: {len(ds_va)}")
    
    # モデル
    model = DeepLabV3Plus(num_classes=1, backbone=CFG["BACKBONE"], output_stride=CFG["OUTPUT_STRIDE"]).to(dev)
    print(f"Model: DeepLab v3+ with {CFG['BACKBONE']} backbone")
    
    loss_fn = ComboLoss()
    opt = torch.optim.Adam(model.parameters(), lr=CFG["LR"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    
    # モデル保存ディレクトリ
    CFG["MODELS_DIR"].mkdir(exist_ok=True)
    pred_dir = Path(CFG["PRED_DIR"]) / "deeplabv3"
    
    # wandb
    dataset = CFG["DATASET"]
    run_name = CFG["RUN_NAME"] or f"deeplabv3plus_{CFG['BACKBONE']}_os{CFG['OUTPUT_STRIDE']}_{dataset}_{int(time.time())}"
    wandb.init(project=CFG["WANDB_PROJ"], name=run_name, config=CFG)
    
    # 学習ループ
    best_dice, wait = 0, 0
    model_tag = f"deeplabv3plus_{CFG['BACKBONE']}_os{CFG['OUTPUT_STRIDE']}_{dataset}"
    
    for ep in range(CFG["EPOCHS"]):
        tr_loss = train_epoch(model, dl_tr, loss_fn, opt, dev)
        va_dice, va_iou, va_probs, va_gts = eval_epoch(model, dl_va, dev)
        
        # PR-AUC計算
        va_prauc, _, _ = get_pr_auc(va_probs, va_gts)
        
        scheduler.step(va_dice)
        
        print(f"Ep{ep:2d} | Loss:{tr_loss:.4f} | Val Dice:{va_dice:.4f} IoU:{va_iou:.4f} PR-AUC:{va_prauc:.4f}")
        
        wandb.log({
            "epoch": ep,
            "train_loss": tr_loss,
            "val_dice": va_dice,
            "val_iou": va_iou,
            "val_pr_auc": va_prauc,
            "lr": opt.param_groups[0]["lr"]
        })
        
        # ベストモデル保存
        if va_dice > best_dice:
            best_dice = va_dice
            wait = 0
            torch.save(model.state_dict(), CFG["MODELS_DIR"] / f"{model_tag}.pt")
            print(f"  → New best model saved! Dice: {best_dice:.4f}")
        else:
            wait += 1
        
        # 予測サンプル保存
        if ep % CFG["SAVE_PRED_EVERY"] == 0:
            save_pred_samples(model, dl_va, pred_dir, ep, CFG["PRED_SAMPLE_N"], dev)
        
        # Early stopping
        if wait >= CFG["PATIENCE"]:
            print(f"Early stopping at epoch {ep}")
            break
    
    wandb.finish()
    print(f"Training finished. Best Dice: {best_dice:.4f}")

if __name__=="__main__":
    main()
