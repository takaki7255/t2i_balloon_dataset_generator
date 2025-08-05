"""
Pre‑trained U‑Net を別データセットで微調整するスクリプト
元の train_unet_split.py と同じフォルダに置けば依存を流用できます
"""

import torch, os, time, random, glob
from pathlib import Path
from tqdm import tqdm
import wandb
from train_unet_split import (  # 既存実装を再利用
    BalloonDataset, UNet, ComboLoss, train_epoch, eval_epoch,
    seed_everything, next_version, save_predictions
)

# --------------------- 設定 ---------------------- #
CFG = {
    # ★ ① 新しいデータセットルート
    "ROOT":        Path("finetune100_dataset"),   # train/val 構造は同じにする
    "IMG_SIZE":    512,

    # ★ ② ハイパーパラメータ（微調整用に小さめ）
    "BATCH":       4,
    "EPOCHS":      50,
    "LR":          1e-5,
    "PATIENCE":    5,
    "SEED":        42,

    # wandb
    "WANDB_PROJ":  "balloon-seg",
    "DATASET":     "finetune100_dataset",
    "RUN_NAME":    "",

    # ★ ③ 事前学習済み ckpt を指定
    "RESUME":      "models/syn750_dataset02-unet-01.pt",

    "MODELS_DIR":  Path("models"),

    # 予測可視化
    "SAVE_PRED_EVERY": 5,
    "PRED_SAMPLE_N":   3,
    "PRED_DIR":        "predictions",
}
# ------------------------------------------------- #

def main():
    cfg = CFG
    seed_everything(cfg["SEED"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # wandb run 名を自動生成
    if not cfg["RUN_NAME"]:
        cfg["RUN_NAME"] = f"{cfg['DATASET']}-finetune"

    wandb.init(project=cfg["WANDB_PROJ"], name=cfg["RUN_NAME"], config=cfg)
    run_dir = Path(wandb.run.dir)

    # ---------- データローダ ----------
    root = cfg["ROOT"]
    tr_ds = BalloonDataset(root / "train/images", root / "train/masks", cfg["IMG_SIZE"])
    va_ds = BalloonDataset(root / "val/images",   root / "val/masks",   cfg["IMG_SIZE"])

    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=cfg["BATCH"],
                                        shuffle=True, num_workers=4, pin_memory=True)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=cfg["BATCH"],
                                        shuffle=False, num_workers=4, pin_memory=True)

    # ---------- モデル ----------
    model = UNet().to(dev)
    if cfg["RESUME"]:
        state = torch.load(cfg["RESUME"], map_location=dev)
        model.load_state_dict(state)
        print(f"✅ Pre‑trained weights loaded from {cfg['RESUME']}")

    # (任意) 下流層のみ更新したい場合は以下をアンコメント
    # for p in model.downs.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg["LR"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["EPOCHS"])
    lossf = ComboLoss()

    # 元モデル名を取得（拡張子を除く）
    base_model_name = Path(cfg["RESUME"]).stem if cfg["RESUME"] else "scratch"
    
    best_iou, patience = 0, 0
    for ep in range(1, cfg["EPOCHS"] + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, tr_dl, lossf, opt, dev)
        va_dice, va_iou = eval_epoch(model, va_dl, dev)
        sched.step()

        save_predictions(model, va_dl, cfg, ep, run_dir, dev)

        wandb.log({"epoch": ep, "loss": tr_loss,
                   "val_dice": va_dice, "val_iou": va_iou,
                   "lr": sched.get_last_lr()[0]})
        print(f"[{ep:03}] loss={tr_loss:.4f}  dice={va_dice:.4f}  iou={va_iou:.4f}  {time.time()-t0:.1f}s")

        if va_iou > best_iou:
            best_iou, patience = va_iou, 0
            ckpt = cfg["MODELS_DIR"] / f"{base_model_name}-{cfg['DATASET']}-finetuned.pt"
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= cfg["PATIENCE"]:
                print("Early stopping"); break

if __name__ == "__main__":
    main()
