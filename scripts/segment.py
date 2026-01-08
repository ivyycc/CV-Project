
"""
lab3.py — UNet training adapted to your project layout (dataset in ./dataset)

Improvements vs previous version:
 - Added gradient clipping (with AMP-safe unscale)
 - Added --clip-norm, --no-amp flags
 - Lower default LR (1e-4) but user-overridable
 - Detects NaNs in loss/parameters, restores last-known good checkpoint and stops
 - Colab-aware defaults for dataset location and num_workers
"""
from __future__ import annotations
import os
import sys
import random
import argparse
from pathlib import Path
import pickle
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
import cv2

def in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

DEFAULT_IMAGES = "/content/dataset/puzzle1000/images" if in_colab() else "./data/images"
DEFAULT_MASKS  = "/content/dataset/puzzle1000/masks"  if in_colab() else "./data/masks"
DEFAULT_FEATURES = "/content/CVProject/../outputs/features.pkl" if in_colab() else "./outputs/features.pkl"
DEFAULT_OUT_DIR = "./models"

# Albumentations: install if missing
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "albumentations", "opencv-python"])
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

# ---------------- UNet ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, up_mode='transpose'):
        super().__init__()
        assert up_mode in ['transpose', 'bilinear']
        self.up_mode = up_mode
        self.d1 = DoubleConv(in_ch, 64); self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128); self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(128, 256); self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(256, 512); self.p4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        if up_mode == 'transpose':
            self.up1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
            self.up2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
            self.up3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
            self.up4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        else:
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up_conv1 = nn.Conv2d(1024,512,kernel_size=1)
            self.up_conv2 = nn.Conv2d(512,256,kernel_size=1)
            self.up_conv3 = nn.Conv2d(256,128,kernel_size=1)
            self.up_conv4 = nn.Conv2d(128,64,kernel_size=1)
        self.u1 = DoubleConv(1024,512)
        self.u2 = DoubleConv(512,256)
        self.u3 = DoubleConv(256,128)
        self.u4 = DoubleConv(128,64)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        x4 = self.d4(self.p3(x3))
        x5 = self.bottleneck(self.p4(x4))
        if self.up_mode == 'transpose':
            u1 = self.up1(x5); u1 = torch.cat([u1, x4], dim=1); u1 = self.u1(u1)
            u2 = self.up2(u1); u2 = torch.cat([u2, x3], dim=1); u2 = self.u2(u2)
            u3 = self.up3(u2); u3 = torch.cat([u3, x2], dim=1); u3 = self.u3(u3)
            u4 = self.up4(u3); u4 = torch.cat([u4, x1], dim=1); u4 = self.u4(u4)
        else:
            u1 = self.up1(x5); u1 = self.up_conv1(u1); u1 = torch.cat([u1, x4], dim=1); u1 = self.u1(u1)
            u2 = self.up2(u1); u2 = self.up_conv2(u2); u2 = torch.cat([u2, x3], dim=1); u2 = self.u2(u2)
            u3 = self.up3(u2); u3 = self.up_conv3(u3); u3 = torch.cat([u3, x2], dim=1); u3 = self.u3(u3)
            u4 = self.up4(u3); u4 = self.up_conv4(u4); u4 = torch.cat([u4, x1], dim=1); u4 = self.u4(u4)
        return self.outc(u4)

# ---------------- Dataset ----------------
def canonical(stem: str) -> str:
    s = stem.lower()
    prefixes = ['image-', 'image_', 'img-', 'img_', 'mask-', 'mask_', 'm-', 'm_']
    suffixes = ['-mask', '_mask']
    changed = True
    while changed:
        changed = False
        for pre in prefixes:
            if s.startswith(pre):
                s = s[len(pre):]; changed = True
        for suf in suffixes:
            if s.endswith(suf):
                s = s[:-len(suf)]; changed = True
    return s

class ProjectSegDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, img_size: int, split: str = 'train'):
        self.pairs = pairs
        self.img_size = img_size
        self.split = split
        self.tfms = self._get_transforms(split)

    def _get_transforms(self, split):
        if split == 'train':
            return A.Compose([
                A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(translate_percent={'x':(-0.05,0.05),'y':(-0.05,0.05)}, scale=(0.9,1.1), rotate=(-15,15), p=0.5),
                A.ColorJitter(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, msk_p = self.pairs[idx]
        img = cv2.imread(str(img_p))[:, :, ::-1]
        mask = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 127).astype(np.uint8)
        transformed = self.tfms(image=img, mask=mask_bin)
        x = transformed['image']
        y = transformed['mask'].long()
        return x, y

# ---------------- Metrics & loops ----------------
@torch.no_grad()
def compute_iou_from_labels(pred_lbl, targets, class_idx=1):
    ious = []
    for b in range(pred_lbl.size(0)):
        p = pred_lbl[b] == class_idx
        t = targets[b] == class_idx
        inter = torch.logical_and(p, t).sum().item()
        union = torch.logical_or(p, t).sum().item()
        iou = inter / union if union > 0 else 1.0
        ious.append(iou)
    return float(np.mean(ious))

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, use_amp, clip_norm):
    model.train()
    total_loss = 0.0; total_iou = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                logits = model(x)
                loss = loss_fn(logits, y)
            # detect NaN loss
            if not torch.isfinite(loss):
                print("Encountered non-finite loss during training (AMP). Aborting batch.")
                return float('nan'), 0.0, True
            scaler.scale(loss).backward()
            # unscale + clip
            if clip_norm and clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                print("Encountered non-finite loss during training. Aborting batch.")
                return float('nan'), 0.0, True
            loss.backward()
            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            pred_lbl = probs.argmax(dim=1)
            total_iou += compute_iou_from_labels(pred_lbl, y) * x.size(0)
    n = len(loader.dataset)
    return total_loss / max(1, n), total_iou / max(1, n), False

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0; total_iou = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        probs = torch.softmax(logits, dim=1)
        pred_lbl = probs.argmax(dim=1)
        total_iou += compute_iou_from_labels(pred_lbl, y) * x.size(0)
    n = len(loader.dataset)
    return total_loss / max(1, n), total_iou / max(1, n)

def save_checkpoint(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))

# ---------------- Helpers ----------------
def build_image_mask_pairs(images_dir: Path, masks_dir: Path):
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')]
    masks  = [p for p in masks_dir.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')]
    img_map = {canonical(p.stem): p for p in images}
    msk_map = {canonical(p.stem): p for p in masks}
    common = sorted(set(img_map.keys()) & set(msk_map.keys()))
    pairs = [(img_map[k], msk_map[k]) for k in common]
    if not pairs:
        raise RuntimeError(f"No matched image/mask pairs found in {images_dir} and {masks_dir}")
    print(f"Found {len(pairs)} matched image-mask pairs.")
    return pairs

def split_pairs(pairs, val_fraction=0.15, seed=42):
    random.seed(seed)
    n = len(pairs); idxs = list(range(n)); random.shuffle(idxs)
    n_val = max(1, int(round(n * val_fraction))); val_idxs = set(idxs[:n_val])
    train_pairs = [pairs[i] for i in idxs if i not in val_idxs]
    val_pairs   = [pairs[i] for i in idxs if i in val_idxs]
    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} val")
    return train_pairs, val_pairs

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Train UNet for puzzle segmentation (dataset/puzzle1000 defaults)")
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES)
    parser.add_argument("--masks_dir",  type=str, default=DEFAULT_MASKS)
    parser.add_argument("--features",   type=str, default=DEFAULT_FEATURES, help="Optional features.pkl (logged only)")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (try smaller to avoid NaNs)")
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--up_mode", type=str, default="transpose", choices=["transpose","bilinear"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip_norm", type=float, default=1.0, help="Gradient clipping norm (0 to disable)")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision (AMP)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement)")

    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    images_dir = Path(args.images_dir)
    masks_dir  = Path(args.masks_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.features and Path(args.features).exists():
        try:
            with open(args.features, "rb") as f:
                feats = pickle.load(f)
            print(f"Loaded features.pkl with {len(feats)} entries (features are NOT used for UNet training).")
        except Exception as e:
            print("Could not load features.pkl:", e)

    if not images_dir.exists() or not masks_dir.exists():
        print("Error: images_dir or masks_dir not found.")
        print("images_dir:", images_dir)
        print("masks_dir:", masks_dir)
        sys.exit(1)

    pairs = build_image_mask_pairs(images_dir, masks_dir)
    train_pairs, val_pairs = split_pairs(pairs, val_fraction=args.val_frac, seed=args.seed)

    train_ds = ProjectSegDataset(train_pairs, img_size=args.img_size, split='train')
    val_ds   = ProjectSegDataset(val_pairs,   img_size=args.img_size, split='val')

    num_workers = 2 if in_colab() else 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=max(1,args.batch_size//2), shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = UNet(in_ch=3, num_classes=2, up_mode=args.up_mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(not args.no_amp and device.type=='cuda'))
    use_amp = (not args.no_amp) and (device.type=='cuda')

    best_val_iou = -1.0
    best_path = out_dir / "unet_best.pth"
    print(f"Start training for {args.epochs} epochs. Best checkpoint -> {best_path}")
    last_good_path = best_path

    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_iou, aborted = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, use_amp, args.clip_norm)
        if aborted or (not np.isfinite(tr_loss)):
            print(f"Training aborted at epoch {epoch} due to non-finite loss. Restoring last good checkpoint: {last_good_path}")
            if last_good_path.exists():
                model.load_state_dict(torch.load(last_good_path, map_location=device))
            break

        va_loss, va_iou = evaluate(model, val_loader, loss_fn, device)
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} train_iou={tr_iou:.4f} | val_loss={va_loss:.4f} val_iou={va_iou:.4f}")

        if va_iou > best_val_iou:
            best_val_iou = va_iou
            save_checkpoint(model, best_path)
            last_good_path = best_path
            epochs_no_improve = 0
            print(f" 🟢 New best val_iou={best_val_iou:.4f} -> saved checkpoint")
        else:
            epochs_no_improve += 1
            print(f" ⚠️ No improvement for {epochs_no_improve}/{args.patience} epochs.")
            if epochs_no_improve >= args.patience:
                print("⏹️ Early stopping triggered. Restoring best checkpoint.")
                model.load_state_dict(torch.load(best_path, map_location=device))
                break

        # NaN/Inf parameter check
        found_nan = False
        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p).any() or torch.isinf(p).any()):
                print(f"Param {name} contains NaN/Inf after epoch {epoch}")
                found_nan = True
                break
        if found_nan:
            print("Detected NaNs in model parameters, restoring last good checkpoint and stopping.")
            model.load_state_dict(torch.load(last_good_path, map_location=device))
            break

        # After each epoch check for NaNs in parameters
        found_nan = False
        for name, p in model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p).any() or torch.isinf(p).any():
                    print(f"Param {name} contains NaN/Inf after epoch {epoch}")
                    found_nan = True
                    break
        if found_nan:
            print("Detected NaNs in model parameters, restoring last good checkpoint and stopping.")
            if last_good_path.exists():
                model.load_state_dict(torch.load(last_good_path, map_location=device))
            break

    print("Training finished. Best val IoU: {:.4f}".format(best_val_iou))
    print("Best checkpoint saved to:", best_path)
    print("Copy/rename it to ./models/unet_best.pth for inference (unet_infer.py / run.sh).")

if __name__ == "__main__":
    main()