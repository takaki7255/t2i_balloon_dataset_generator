"""
Pre‑trained U‑Net を別データセットで微調整するスクリプト
元の train_unet_split.py と同じフォルダに置けば依存を流用できます

使用例:
  python finetune_unet.py --root ./balloon_dataset/syn1000-balloon-corner --resume ./balloon_models/real200_dataset-unet-01.pt
  python finetune_unet.py --root ./balloon_dataset/real200_dataset --dataset real200 --resume ./balloon_models/syn1000-unet-01.pt --epochs 30 --lr 5e-5
"""

import torch, os, time, random, glob, argparse
from pathlib import Path
from tqdm import tqdm
import wandb
from train_unet_split import (  # 既存実装を再利用
    BalloonDataset, UNet, ComboLoss, train_epoch, eval_epoch,
    seed_everything, next_version, save_predictions
)

# --------------------- デフォルト設定 ---------------------- #
CFG = {
    # データセット
    "ROOT":        Path("balloon_dataset/syn1000-balloon-corner"),
    "IMG_SIZE":    (384, 512),  # (height, width)

    # ハイパーパラメータ（微調整用に小さめ）
    "BATCH":       4,
    "EPOCHS":      50,
    "LR":          1e-5,
    "PATIENCE":    5,
    "SEED":        42,

    # wandb
    "WANDB_PROJ":  "balloon-seg",
    "DATASET":     "",  # 空の場合はROOTから自動生成
    "RUN_NAME":    "",

    # 事前学習済み ckpt
    "RESUME":      "",

    "MODELS_DIR":  Path("balloon_models"),

    # 予測可視化
    "SAVE_PRED_EVERY": 5,
    "PRED_SAMPLE_N":   3,
    "PRED_DIR":        "predictions",
}
# ------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description='Finetune U-Net on a new dataset using pretrained weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # syn1000データセットでファインチューニング
  python finetune_unet.py --root ./balloon_dataset/syn1000-balloon-corner --resume ./balloon_models/real200_dataset-unet-01.pt

  # カスタム設定で実行
  python finetune_unet.py --root ./balloon_dataset/real200_dataset --dataset real200-finetune --resume ./balloon_models/syn1000-unet-01.pt --epochs 30 --lr 5e-5 --batch 8
        """
    )
    
    # データセット関連
    parser.add_argument('--root', type=str, default=None,
                        help='Dataset root directory (must contain train/ and val/ subdirs)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name for wandb/model tag (default: derived from root)')
    
    # 事前学習モデル
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to pretrained checkpoint (.pt file) - REQUIRED')
    
    # 学習パラメータ
    parser.add_argument('--batch', type=int, default=None,
                        help=f'Batch size (default: {CFG["BATCH"]})')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Number of epochs (default: {CFG["EPOCHS"]})')
    parser.add_argument('--lr', type=float, default=None,
                        help=f'Learning rate (default: {CFG["LR"]})')
    parser.add_argument('--patience', type=int, default=None,
                        help=f'Early stopping patience (default: {CFG["PATIENCE"]})')
    
    # 保存先
    parser.add_argument('--models-dir', type=str, default=None,
                        help=f'Models directory (default: {CFG["MODELS_DIR"]})')
    
    # wandb関連
    parser.add_argument('--wandb-proj', type=str, default=None,
                        help=f'Wandb project name (default: {CFG["WANDB_PROJ"]})')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    
    # その他
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze encoder (downsampling) layers during finetuning')
    parser.add_argument('--seed', type=int, default=None,
                        help=f'Random seed (default: {CFG["SEED"]})')
    
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = CFG.copy()
    
    # コマンドライン引数で設定を上書き
    if args.root:
        cfg["ROOT"] = Path(args.root)
    
    if args.dataset:
        cfg["DATASET"] = args.dataset
    elif cfg["DATASET"] == "":
        # ROOTから自動生成
        cfg["DATASET"] = cfg["ROOT"].name
    
    if args.resume:
        cfg["RESUME"] = args.resume
    
    if args.batch:
        cfg["BATCH"] = args.batch
    
    if args.epochs:
        cfg["EPOCHS"] = args.epochs
    
    if args.lr:
        cfg["LR"] = args.lr
    
    if args.patience:
        cfg["PATIENCE"] = args.patience
    
    if args.models_dir:
        cfg["MODELS_DIR"] = Path(args.models_dir)
    
    if args.wandb_proj:
        cfg["WANDB_PROJ"] = args.wandb_proj
    
    if args.run_name:
        cfg["RUN_NAME"] = args.run_name
    
    if args.seed:
        cfg["SEED"] = args.seed
    
    # 必須パラメータのチェック
    if not cfg["RESUME"]:
        print("ERROR: --resume is required. Please specify a pretrained checkpoint.")
        print("Example: python finetune_unet.py --resume ./balloon_models/real200_dataset-unet-01.pt --root ./balloon_dataset/syn1000-balloon-corner")
        return
    
    if not Path(cfg["RESUME"]).exists():
        print(f"ERROR: Checkpoint not found: {cfg['RESUME']}")
        return
    
    if not cfg["ROOT"].exists():
        print(f"ERROR: Dataset root not found: {cfg['ROOT']}")
        return
    
    # 設定表示
    print("=" * 60)
    print("  U-Net Finetuning")
    print("=" * 60)
    print(f"📁 Dataset root: {cfg['ROOT']}")
    print(f"📊 Dataset name: {cfg['DATASET']}")
    print(f"🔄 Resume from: {cfg['RESUME']}")
    print(f"📦 Batch size: {cfg['BATCH']}")
    print(f"🔢 Epochs: {cfg['EPOCHS']}")
    print(f"📈 Learning rate: {cfg['LR']}")
    print(f"⏱️  Patience: {cfg['PATIENCE']}")
    print(f"💾 Models dir: {cfg['MODELS_DIR']}")
    if args.freeze_encoder:
        print(f"❄️  Encoder: FROZEN")
    print("=" * 60 + "\n")
    
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
                                        shuffle=True, num_workers=0, pin_memory=False)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=cfg["BATCH"],
                                        shuffle=False, num_workers=0, pin_memory=False)

    # ---------- モデル ----------
    model = UNet().to(dev)
    if cfg["RESUME"]:
        state = torch.load(cfg["RESUME"], map_location=dev)
        model.load_state_dict(state)
        print(f"✅ Pre‑trained weights loaded from {cfg['RESUME']}")

    # エンコーダを固定する場合
    if args.freeze_encoder:
        for p in model.downs.parameters():
            p.requires_grad = False
        print("❄️  Encoder layers frozen")

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
            print(f"✓ Best model saved! IoU: {va_iou:.4f} -> {ckpt}")
        else:
            patience += 1
            if patience >= cfg["PATIENCE"]:
                print("Early stopping"); break

if __name__ == "__main__":
    main()
