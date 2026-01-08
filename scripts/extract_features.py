from __future__ import annotations
import os
import sys
import re
import cv2
import math
import pickle
import argparse
from glob import glob
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from scipy.signal import savgol_filter
from scipy import ndimage
from PIL import Image

# --------- Configurable params ----------
EDGE_SAMPLE_POINTS = 64   # points per resampled edge
EDGE_COLOR_WIDTH = 5      # pixels to sample on each side of edge
MR8_PATCH_PAD = 8         # padding around side bbox for MR8 patch
PROGRESS_PRINT_EVERY = 50  # print progress every N pieces

# --------- Data classes ----------
@dataclass
class EdgeFeat:
    points: np.ndarray            # (N,2) resampled coords (float32)
    type: str                     # 'flat'|'tab'|'blank'
    curvature: np.ndarray         # (N,)
    fourier: np.ndarray           # (n_coeffs,)
    color_hist_left: np.ndarray   # (B,)
    color_hist_right: np.ndarray  # (B,)
    mean_left: np.ndarray         # (3,)
    mean_right: np.ndarray        # (3,)
    length: float
    mr8: np.ndarray = None        # MR8 / texture descriptor (optional)
    embed: Optional[np.ndarray] = None  # placeholder for downstream embeddings

# --------- Small helpers adapted from labs ----------
def read_image(path: str):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    # ensure 3-channel RGB
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    elif im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def extract_largest_contour(mask: np.ndarray) -> np.ndarray | None:
    """Return Nx1x2 contour (int32) of largest contour in binary mask (mask can be 0/255 or bool)."""
    if mask.dtype != np.uint8:
        mask_u = (mask > 0).astype(np.uint8) * 255
    else:
        mask_u = mask
    contours, _ = cv2.findContours(mask_u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    maxc = max(contours, key=cv2.contourArea)
    return maxc.astype(np.int32)

def clockwise_contour(contour: np.ndarray) -> np.ndarray:
    """Ensure contour points are clockwise (Nx1x2)."""
    # OpenCV oriented area positive => clockwise or counterclockwise depending on convention; keep your check
    area = cv2.contourArea(contour, oriented=True)
    if area < 0:
        return contour[::-1]
    return contour

def smooth_contour(contour: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """Savgol smoothing on x,y separately. contour shape Nx1x2 -> returns same shape"""
    pts = contour.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < window:
        return contour
    pad = window // 2
    xs = np.concatenate([pts[-pad:,0], pts[:,0], pts[:pad,0]])
    ys = np.concatenate([pts[-pad:,1], pts[:,1], pts[:pad,1]])
    xs_s = savgol_filter(xs, window, poly)
    ys_s = savgol_filter(ys, window, poly)
    xs_c = xs_s[pad:pad+pts.shape[0]]
    ys_c = ys_s[pad:pad+pts.shape[0]]
    out = np.stack([xs_c, ys_c], axis=1).astype(np.int32)
    return out.reshape(-1,1,2)

def even_spaced_contour(contour: np.ndarray, num_points: int = EDGE_SAMPLE_POINTS) -> np.ndarray:
    pts = contour.reshape(-1,2).astype(np.float32)
    # compute distances between consecutive points (not closed)
    d = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
    cum = np.concatenate(([0.0], np.cumsum(d)))
    total = cum[-1]
    if total == 0:
        return np.tile(pts[0], (num_points,1)).astype(np.float32)
    t = np.linspace(0, total, num_points)
    xs = np.interp(t, cum, pts[:,0])
    ys = np.interp(t, cum, pts[:,1])
    return np.stack([xs, ys], axis=1).astype(np.float32)

def compute_curvature(pts: np.ndarray) -> np.ndarray:
    x = pts[:,0]; y = pts[:,1]
    dx = np.gradient(x); dy = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    denom = (dx*dx + dy*dy)**1.5 + 1e-8
    k = np.abs(dx * ddy - dy * ddx) / denom
    if k.size >= 9:
        k = savgol_filter(k, 9, 3)
    return k

def compute_fourier_descriptor(pts: np.ndarray, n_coeffs: int = 20) -> np.ndarray:
    complex_coords = pts[:,0].astype(np.float32) + 1j * pts[:,1].astype(np.float32)
    F = np.fft.fft(complex_coords)
    mags = np.abs(F[:n_coeffs])
    # always normalize relative to DC component with small epsilon to avoid division by zero
    eps = 1e-9
    mags = mags / (mags[0] + eps)
    return mags.astype(np.float32)

# --------- MR8 / filter helpers ----------
def gaussian_kernel(size: int, sigma: float):
    if size % 2 == 0: size += 1
    k = size // 2
    x,y = np.meshgrid(np.arange(-k,k+1), np.arange(-k,k+1))
    g = np.exp(-(x*x+y*y)/(2*sigma*sigma))
    return g / (g.sum()+1e-12)

def log_kernel(size: int, sigma: float):
    if size % 2 == 0: size += 1
    k = size // 2
    x,y = np.meshgrid(np.arange(-k,k+1), np.arange(-k,k+1))
    r2 = x*x + y*y
    kf = (1/(math.pi*sigma**4))*(1 - r2/(2*sigma*sigma))*np.exp(-r2/(2*sigma*sigma))
    return kf - kf.mean()

def build_rfs_bank_simple():
    filters = []
    filters.append(gaussian_kernel(31, 3.0))
    filters.append(log_kernel(31, 3.0))
    filters.append(gaussian_kernel(31, 6.0))
    filters.append(log_kernel(31, 6.0))
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
    ky = kx.T
    filters.append(kx); filters.append(ky)
    return filters

def compute_mr8_patch(gray_patch: np.ndarray) -> np.ndarray:
    bank = build_rfs_bank_simple()
    feats = []
    for f in bank:
        try:
            r = cv2.filter2D(gray_patch.astype(np.float32), -1, f)
            feats.append(np.max(r))
            feats.append(np.mean(r))
        except Exception:
            feats.append(0.0); feats.append(0.0)
    return np.array(feats, dtype=np.float32)

def color_histogram(samples: List[Tuple[int,int,int]], bins_per_channel: int = 4) -> np.ndarray:
    if len(samples) == 0:
        return np.zeros((bins_per_channel**3,), dtype=np.float32)
    arr = np.asarray(samples, dtype=np.float32)
    # use range up to 256 so 255 is binned correctly
    H, edges = np.histogramdd(arr, bins=(bins_per_channel,)*3, range=((0,256),)*3)
    flat = H.flatten().astype(np.float32)
    s = flat.sum() + 1e-9
    return flat / s

# --------- Core extraction logic ----------
def classify_edge(edge_pts: np.ndarray, flat_thresh: float = 3.0) -> str:
    pts = edge_pts.astype(np.float32)
    if pts.shape[0] < 3:
        return 'flat'
    [vx, vy, x0, y0] = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    d = (pts[:,1] - y0) * vx - (pts[:,0] - x0) * vy
    maxdev = np.max(np.abs(d))
    if maxdev < flat_thresh:
        return 'flat'
    mean_d = np.mean(d)
    return 'tab' if mean_d > 0 else 'blank'

def sample_colors_around_edge(img_rgb: np.ndarray, edge_pts: np.ndarray, width: int = EDGE_COLOR_WIDTH):
    H,W,_ = img_rgb.shape
    left_colors = []
    right_colors = []
    pts = edge_pts.astype(np.float32)
    L = pts.shape[0]
    step = max(1, L // 40)
    for i in range(0, L, step):
        p = pts[i]
        if i < L-1:
            t = pts[i+1] - p
        else:
            t = p - pts[i-1]
        n = np.array([-t[1], t[0]], dtype=np.float32)
        norm = np.linalg.norm(n) + 1e-9
        n = n / norm
        left = p - n * width
        right = p + n * width
        lx, ly = int(round(left[0])), int(round(left[1]))
        rx, ry = int(round(right[0])), int(round(right[1]))
        if 0 <= lx < W and 0 <= ly < H:
            left_colors.append(img_rgb[ly, lx].tolist())
        if 0 <= rx < W and 0 <= ry < H:
            right_colors.append(img_rgb[ry, rx].tolist())
    return left_colors, right_colors

def extract_piece_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any] | None:
    if image is None or mask is None:
        return None
    if mask.dtype != np.uint8:
        mask_u = (mask > 127).astype(np.uint8) * 255
    else:
        mask_u = mask

    # ensure mask and image have same dimensions
    if image.shape[0:2] != mask_u.shape[0:2]:
        # caller should handle/log this; return None to mark failure
        return None

    contour = extract_largest_contour(mask_u)
    if contour is None or len(contour) < 20:
        return None
    contour = clockwise_contour(contour)
    contour = smooth_contour(contour)
    contour_xy = contour.reshape(-1,2).astype(np.int32)

    eps = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) == 4:
        corners = [c[0].astype(np.int32) for c in approx]
    else:
        hull = cv2.convexHull(contour)
        if len(hull) >= 4:
            idxs = np.linspace(0, len(hull)-1, 4, dtype=int)
            corners = [hull[i][0].astype(np.int32) for i in idxs]
        else:
            n = len(contour_xy)
            corners = [contour_xy[0], contour_xy[n//4], contour_xy[n//2], contour_xy[(3*n)//4]]

    sides = []
    n = len(contour_xy)
    corner_idxs = [int(np.argmin(np.linalg.norm(contour_xy - c, axis=1))) for c in corners]
    corner_idxs = sorted(corner_idxs)
    for i in range(4):
        a = corner_idxs[i]
        b = corner_idxs[(i+1)%4]
        if a <= b:
            seg = contour_xy[a:b+1]
        else:
            seg = np.vstack([contour_xy[a:], contour_xy[:b+1]])
        sides.append(seg)

    edge_dicts = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for seg in sides:
        if seg.shape[0] < 6:
            continue
        res = even_spaced_contour(seg, num_points=EDGE_SAMPLE_POINTS)
        curv = compute_curvature(res)
        fourier = compute_fourier_descriptor(res, n_coeffs=20)
        etype = classify_edge(res)
        left_samples, right_samples = sample_colors_around_edge(image, res, width=EDGE_COLOR_WIDTH)
        hist_left = color_histogram(left_samples, bins_per_channel=4)
        hist_right = color_histogram(right_samples, bins_per_channel=4)
        mean_left = np.mean(np.array(left_samples), axis=0) if left_samples else np.zeros(3, dtype=np.float32)
        mean_right = np.mean(np.array(right_samples), axis=0) if right_samples else np.zeros(3, dtype=np.float32)
        xs = res[:,0]; ys = res[:,1]
        x0 = max(0, int(math.floor(xs.min()) - MR8_PATCH_PAD))
        x1 = min(image.shape[1]-1, int(math.ceil(xs.max()) + MR8_PATCH_PAD))
        y0 = max(0, int(math.floor(ys.min()) - MR8_PATCH_PAD))
        y1 = min(image.shape[0]-1, int(math.ceil(ys.max()) + MR8_PATCH_PAD))
        patch = gray[y0:y1+1, x0:x1+1] if (y1>=y0 and x1>=x0) else np.zeros((1,1), dtype=np.uint8)
        mr8 = compute_mr8_patch(patch)
        length = float(np.sum(np.sqrt(((res[1:] - res[:-1])**2).sum(axis=1))))
        ed = EdgeFeat(
            points=res.astype(np.float32),
            type=etype,
            curvature=curv.astype(np.float32),
            fourier=fourier.astype(np.float32),
            color_hist_left=hist_left.astype(np.float32),
            color_hist_right=hist_right.astype(np.float32),
            mean_left=np.array(mean_left, dtype=np.float32),
            mean_right=np.array(mean_right, dtype=np.float32),
            length=length,
            mr8=mr8.astype(np.float32) if mr8 is not None else None,
            embed=None
        )
        edge_dicts.append(ed)

    # if fewer than 2 edges extracted, likely segmentation issue
    if len(edge_dicts) < 2:
        return None

    area = float(cv2.contourArea(contour))
    perim = float(cv2.arcLength(contour, True))
    x,y,w,h = cv2.boundingRect(contour)
    M = cv2.moments(contour)
    if M.get('m00',0) != 0:
        cx = float(M['m10'] / M['m00']); cy = float(M['m01'] / M['m00'])
    else:
        cx, cy = float(x + w/2), float(y + h/2)

    result = {
        'contour': contour.reshape(-1,2).astype(np.int32),
        'edges': edge_dicts,
        'area': area,
        'perimeter': perim,
        'bbox': (int(x), int(y), int(w), int(h)),
        'center': (cx, cy),
        'aspect_ratio': float(w) / (h + 1e-9)
    }
    return result

# --------- Batch processing ----------
def _canonical(s: str):
    """
    Return canonical stem for matching image<->mask. Strips common mask suffixes and trailing separators.
    Examples:
      IMG_001_mask -> img_001
      IMG-001-mask -> img-001
      img_002 -> img_002
    """
    s2 = s.lower()
    # remove common mask suffix patterns (at end)
    s2 = re.sub(r'(?:(?:_|\-)?mask(?:_|\-)?$)', '', s2)
    # strip trailing separators left by removal
    s2 = s2.rstrip('_-')
    return s2

def _mask_label_from_dir(mask_dir: str) -> str:
    basename = os.path.basename(mask_dir.rstrip("/\\"))
    if basename.lower() in ('pred_masks','preds','pred'):
        return 'pred'
    if basename.lower() in ('masks','gt','labels','ground_truth'):
        return 'gt'
    # fallback to sanitized basename
    return basename.replace(' ','_')

def _sample_n_sorted(it, n=5):
    it = sorted(it)
    return it[:n]

def process_all(images_dir: str, masks_dirs: List[str], out_file: str):
    """
    images_dir: folder with original images (IMG_*.jpg)
    masks_dirs: list of mask directories to process (each may contain *_mask.png)
    If masks_dirs length == 1 we keep original keys (stem).
    If >1 we append a suffix to keys derived from mask dir name to avoid collisions.
    """
    # build image file map (stem -> path)
    img_files = {os.path.splitext(os.path.basename(p))[0]: p for p in glob(os.path.join(images_dir, "*")) if p.lower().endswith(('.png','.jpg','.jpeg'))}
    print(f"Found {len(img_files)} image files in '{images_dir}'")
    if len(img_files) > 0:
        print("  sample images:", _sample_n_sorted(list(img_files.keys()), 6))

    all_feats: Dict[str, Any] = {}
    failed: List[Tuple[str, str, str]] = []  # (mdir, stem, reason)

    single_mask_mode = (len(masks_dirs) == 1)

    for mdir in masks_dirs:
        if not os.path.isdir(mdir):
            print(f"Warning: mask dir not found, skipping: {mdir}")
            continue
        msk_files = {os.path.splitext(os.path.basename(p))[0]: p for p in glob(os.path.join(mdir, "*")) if p.lower().endswith(('.png','.jpg','.jpeg'))}
        print(f"[{mdir}] Found {len(msk_files)} mask files")
        if len(msk_files) > 0:
            print("  sample masks:", _sample_n_sorted(list(msk_files.keys()), 6))
        # canonicalize stems
        img_map = { _canonical(k): v for k,v in img_files.items() }
        msk_map = { _canonical(k): v for k,v in msk_files.items() }
        common = sorted(set(img_map.keys()) & set(msk_map.keys()))
        print(f"[{mdir}] Found {len(common)} image-mask pairs to process (common stems)")
        if len(common) == 0:
            # help debugging: show examples of image-only / mask-only stems (small lists)
            img_only = sorted(set(img_map.keys()) - set(msk_map.keys()))
            msk_only = sorted(set(msk_map.keys()) - set(img_map.keys()))
            print(f"  No common stems for masks dir '{mdir}'. Examples:")
            print("   images without mask (sample):", _sample_n_sorted(img_only, 6))
            print("   masks without image (sample):", _sample_n_sorted(msk_only, 6))
            continue

        label = _mask_label_from_dir(mdir)

        for idx, stem in enumerate(common):
            img_path = img_map.get(stem)
            msk_path = msk_map.get(stem)
            if img_path is None or msk_path is None:
                failed.append((mdir, stem, "missing path"))
                continue
            img = read_image(img_path)
            msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                failed.append((mdir, stem, f"read_image returned None for {img_path}")); continue
            if msk is None:
                failed.append((mdir, stem, f"cv2.imread returned None for {msk_path}")); continue

            # ensure mask and image have same dimensions
            if img.shape[0:2] != msk.shape[0:2]:
                failed.append((mdir, stem, f"image/mask size mismatch {img.shape[0:2]} vs {msk.shape[0:2]}"))
                continue

            try:
                feats = extract_piece_features(img, msk)
            except Exception as ex:
                failed.append((mdir, stem, f"exception in extract_piece_features: {ex}"))
                continue

            if feats is None:
                failed.append((mdir, stem, "extract_piece_features returned None (likely small/invalid contour or segmentation issue)"))
                continue

            feats_serializable = feats.copy()
            feats_serializable['edges'] = []
            for ed in feats['edges']:
                feats_serializable['edges'].append({
                    'points': ed.points,
                    'type': ed.type,
                    'curvature': ed.curvature,
                    'fourier': ed.fourier,
                    'color_hist_left': ed.color_hist_left,
                    'color_hist_right': ed.color_hist_right,
                    'mean_left': ed.mean_left,
                    'mean_right': ed.mean_right,
                    'length': ed.length,
                    'mr8': ed.mr8,
                    'embed': None
                })

            # save optional metadata for traceability
            feats_serializable['source_image_path'] = img_path
            feats_serializable['source_mask_path'] = msk_path
            feats_serializable['mask_label'] = label

            # determine feature key
            if single_mask_mode:
                key_name = stem
            else:
                key_name = f"{stem}_{label}"

            # store in global features dict (key_name unique)
            if key_name in all_feats:
                # collision unlikely due to suffixing logic, but avoid overwrite by adding incremental suffix
                i = 1
                base = key_name
                while key_name in all_feats:
                    key_name = f"{base}_{i}"
                    i += 1
            all_feats[key_name] = feats_serializable

            if (idx + 1) % PROGRESS_PRINT_EVERY == 0:
                print(f"[{mdir}] processed {idx+1}/{len(common)} pairs")

    # save features.pkl
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(all_feats, f)
    print("Saved features to:", out_file)
    if failed:
        print("Failed pieces (examples):")
        for item in failed[:20]:
            print(" ", item)
    else:
        print("No failures recorded.")
    print(f"Total features: {len(all_feats)}. Total failures: {len(failed)}")

    # If nothing processed, exit with helpful message
    if len(all_feats) == 0:
        print("ERROR: No features extracted. Please check images_dir and masks_dir paths and filenames (stems must match).")
        raise SystemExit(2)

    return all_feats, failed

# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="Extract piece features (no longer saves cropped RGBA pieces)")
    parser.add_argument("--images_dir", type=str, default="./data/images", help="Directory with original images")
    parser.add_argument("--masks_dir",  type=str, default="./data/pred_masks", help="Single mask dir or comma-separated list. If omitted, the script will auto-detect 'masks' and 'pred_masks' next to images_dir.")
    parser.add_argument("--output",     type=str, default="outputs/test_features.pkl", help="Output features pickle")
    parser.add_argument("--mask_filter", type=str, choices=["all","pred","gt"], default="pred",
                        help="If not 'all', only keep features whose mask_label equals this value ('pred' or 'gt') when saving the final features pickle.")
    args = parser.parse_args()

    IMAGES_DIR = args.images_dir
    OUTPUT_FILE = args.output

    # determine masks directories (same logic as before)
    masks_dirs: List[str] = []
    if args.masks_dir:
        for part in args.masks_dir.split(","):
            part = part.strip()
            if part:
                masks_dirs.append(os.path.abspath(part))
    else:
        parent = os.path.abspath(os.path.join(IMAGES_DIR, os.pardir))
        cand1 = os.path.join(parent, "masks")
        cand2 = os.path.join(parent, "pred_masks")
        if os.path.isdir(cand1):
            masks_dirs.append(cand1)
        if os.path.isdir(cand2):
            masks_dirs.append(cand2)
        if not masks_dirs:
            default = os.path.join(parent, "masks")
            masks_dirs.append(default)

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    print("Starting feature extraction (no crops)...")
    print("Images dir:", IMAGES_DIR)
    print("Mask dirs:", masks_dirs)
    print("Output features file:", OUTPUT_FILE)
    print("Mask filter:", args.mask_filter)

    all_feats, failed = process_all(IMAGES_DIR, masks_dirs, OUTPUT_FILE)

    # Optionally filter by mask_label and overwrite the output pickle with filtered set
    if args.mask_filter != "all":
        filtered = {k: v for k, v in all_feats.items() if v.get("mask_label") == args.mask_filter}
        print(f"Filtered features: kept {len(filtered)} / {len(all_feats)} entries with mask_label='{args.mask_filter}'")
        # overwrite output file with filtered features
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(filtered, f)
        all_feats = filtered

    print(f"Done! Processed {len(all_feats)} feature entries, {len(failed)} failed entries.")

if __name__ == "__main__":
    main()