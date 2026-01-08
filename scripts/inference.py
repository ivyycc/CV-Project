#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

def in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

DEFAULT_IMAGES = "dataset/puzzle1000/images" if in_colab() else "./data/images"
DEFAULT_MASKS  = "dataset/puzzle1000/masks"  if in_colab() else "./data/masks"
DEFAULT_CHECKPOINT = "./models/unet_best.pth"

# Prefer UNet from segment.py (user note), fallback to lab3, else local impl
_HAS_UNET = False
try:
    from segment import UNet
    _HAS_UNET = True; _UNET_SOURCE = "segment.py"
except Exception:
    try:
        from lab3 import UNet
        _HAS_UNET = True; _UNET_SOURCE = "lab3.py"
    except Exception:
        _HAS_UNET = False; _UNET_SOURCE = None

if not _HAS_UNET:
    import torch.nn as nn
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
                self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
                self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
                self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            else:
                self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.up_conv1 = nn.Conv2d(1024, 512, kernel_size=1)
                self.up_conv2 = nn.Conv2d(512, 256, kernel_size=1)
                self.up_conv3 = nn.Conv2d(256, 128, kernel_size=1)
                self.up_conv4 = nn.Conv2d(128, 64, kernel_size=1)
            self.u1 = DoubleConv(1024, 512)
            self.u2 = DoubleConv(512, 256)
            self.u3 = DoubleConv(256, 128)
            self.u4 = DoubleConv(128, 64)
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

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_mask_zero_one(path: Path, mask01: np.ndarray):
    ensure_dir(str(path.parent))
    if mask01 is None:
        raise ValueError("mask is None")
    m = (mask01 > 0).astype(np.uint8) * 255   # 0 or 255
    cv2.imwrite(str(path), m)
    return path


def save_mask_vis(path: Path, mask01: np.ndarray):
    """
    Save a visualization copy scaled to 0..255 so viewers can see the mask.
    """
    ensure_dir(str(path.parent))
    if mask01 is None:
        raise ValueError("mask is None")
    mvis = (mask01 > 0).astype(np.uint8) * 255
    cv2.imwrite(str(path), mvis)
    return path

def preprocess_img_resize(img_bgr: np.ndarray, target_size: int):
    # Resize then convert to RGB and normalize (must match training)
    img_rs = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = img_rs[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2,0,1)).float()
    return tensor

def postprocess_pred(prob_map: np.ndarray, orig_size: tuple, threshold: float = 0.5):
    target_h, target_w = orig_size
    # cv2.resize expects (width, height)
    prob_resized = cv2.resize(prob_map.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask01 = (prob_resized >= threshold).astype(np.uint8)
    return mask01, prob_resized

def clean_state_dict(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        out[nk] = v
    return out

def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device, strict: bool = True):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict):
        if 'state_dict' in state:
            sd = state['state_dict']
        elif 'model_state' in state:
            sd = state['model_state']
        elif 'model' in state and isinstance(state['model'], dict):
            sd = state['model']
        else:
            sd = state
    else:
        sd = state
    sd = clean_state_dict(sd)
    # debug info: print some keys and model param count
    try:
        sample_keys = list(sd.keys())[:12]
        print("Checkpoint keys (sample):", sample_keys)
    except Exception:
        pass
    total_params = sum(p.numel() for p in model.parameters())
    print("Model total params:", total_params)
    # attempt to load state dict with optionally non-strict loading
    model.load_state_dict(sd, strict=strict)
    return model

def infer(args, orig_masks_dir: Path):
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    ensure_dir(str(masks_dir))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Using device:", device)
    if _UNET_SOURCE:
        print("UNet imported from", _UNET_SOURCE)
    else:
        print("UNet fallback in use (lab code not importable)")

    model = UNet(in_ch=3, num_classes=2, up_mode=args.up_mode)
    model.to(device)

    try:
        print("Loading checkpoint:", args.checkpoint)
        # load non-strict if user asked; non-strict helpful for minor naming mismatches
        model = load_checkpoint(model, args.checkpoint, device, strict=not args.loose_load)
    except Exception as e:
        print("Failed to load checkpoint:", e)
        sys.exit(1)

    model.eval()

    # --- FILTER OUT IMAGES THAT ALREADY HAVE ORIGINAL GT MASKS ---
    all_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    files_to_predict = []
    for p in all_files:
        stem = p.stem
        gt_mask_path = orig_masks_dir / f"{stem}_mask.png"  # ORIGINAL ground-truth masks
        if not gt_mask_path.exists():  # only predict if mask doesn't already exist
            files_to_predict.append(p)

    total = len(files_to_predict)
    predicted = 0
    failed = 0
    processed_list = []

    # directory to write a visualization copy (if debug)
    vis_dir = Path(str(masks_dir)) if args.debug else None

    for idx, p in enumerate(tqdm(files_to_predict, desc="Predicting")):
        stem = p.stem
        out_mask_path = masks_dir / f"{stem}_mask.png"        # submission-style 0/1
        out_mask_vis = None
        if args.debug:
            out_mask_vis = masks_dir / f"{stem}_mask_vis.png"  # 0/255 for viewing
            prob_vis_path = masks_dir / f"{stem}_prob_vis.png" # grayscale prob map

        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print("Failed to read:", p)
            failed += 1
            continue
        h0, w0 = img_bgr.shape[:2]
        tensor = preprocess_img_resize(img_bgr, args.img_size)
        batch_tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            prob = torch.softmax(logits, dim=1)[0,1,:,:].cpu().numpy()

        # postprocess and optionally save debug viz
        try:
            mask01, prob_resized = postprocess_pred(prob, orig_size=(h0, w0), threshold=args.threshold)
            # save submission mask (0/1)
            save_mask_zero_one(out_mask_path, mask01)
            # debug saves
            if args.debug:
                # print prob stats for a few examples
                if idx < args.debug_n:
                    print(f"[DEBUG] {stem} prob stats: min={float(np.min(prob_resized)):.6f}, mean={float(np.mean(prob_resized)):.6f}, max={float(np.max(prob_resized)):.6f}")
                # save prob visualization (0..255)
                try:
                    pv = np.clip(prob_resized, 0.0, 1.0)
                    pv_img = (pv * 255.0).astype(np.uint8)
                    cv2.imwrite(str(prob_vis_path), pv_img)
                except Exception as e:
                    print("[DEBUG] failed to save prob_vis:", e)
                # save mask visualization (0..255)
                try:
                    if out_mask_vis is not None:
                        save_mask_vis(out_mask_vis, mask01)
                except Exception as e:
                    print("[DEBUG] failed to save mask_vis:", e)

            predicted += 1
            processed_list.append(stem)
        except Exception as e:
            print("Failed to process/save for", stem, e)
            failed += 1

    print("\nInference summary:")
    print(f"  Total images to predict: {total}")
    print(f"  Predicted and wrote masks: {predicted}")
    print(f"  Failed reads/saves: {failed}")
    if processed_list:
        print("  Example predicted masks:", processed_list[:10])

def main():
    p = argparse.ArgumentParser(description="UNet inference (Colab-friendly). Defaults to ./models/unet_best.pth")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to UNet checkpoint (state_dict).")
    p.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES, help="Input images directory")
    p.add_argument("--masks_dir", type=str, default=DEFAULT_MASKS, help="Ground-truth masks directory (skipped)")
    p.add_argument("--img_size", type=int, default=512, help="Resize images to this size before feeding model")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size for inference (not used in single-image mode here)")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold on foreground probability to produce binary mask")
    p.add_argument("--up_mode", type=str, default="transpose", choices=["transpose","bilinear"], help="UNet upsampling variant used at training")
    p.add_argument("--force_cpu", action="store_true", help="Force CPU even if GPU available")
    p.add_argument("--debug", action="store_true", help="Enable debug outputs (prob map stats and visualizations)")
    p.add_argument("--debug_n", type=int, default=8, help="Number of debug images to print/save")
    p.add_argument("--loose_load", action="store_true", help="Load checkpoint with strict=False (useful for name-prefix mismatches)")
    args = p.parse_args()

    if not Path(args.images_dir).exists():
        print("Images dir not found:", args.images_dir)
        sys.exit(1)

    # Store original ground-truth masks folder
    orig_masks_dir = Path(args.masks_dir)

    # Ensure predictions go to a separate folder
    pred_masks_dir = orig_masks_dir.parent / "pred_masks"
    os.makedirs(pred_masks_dir, exist_ok=True)
    args.masks_dir = str(pred_masks_dir)  # overwrite for inference output

    # Pass the original masks dir as well
    infer(args, orig_masks_dir)

if __name__ == "__main__":
    main()