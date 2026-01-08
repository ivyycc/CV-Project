#!/usr/bin/env python3
"""
Robust puzzle assembler (updated to better support cropped RGBA piece images).

Changes vs original:
 - Adds optional --crops_dir argument (default auto-detects dataset/.../cropped_images)
 - Image cache now indexes multiple lookup keys (stem, stem lower, filename) and
   also tries common suffix-stripping (e.g. remove "_pred", "_gt", "_masks") when
   looking up images so graph node ids can match cropped filenames flexibly.
 - find_image_fast now attempts a small set of normalized candidates before giving up,
   and prints a helpful hint if many nodes are missing.
 - No change to assembly layout/pasting: RGBA crops (with alpha) are pasted as before.

Usage:
  # If you saved crops to dataset/puzzle1000/cropped_images:
  python assemble.py --graph outputs/graph.json --images_dir dataset/puzzle1000/cropped_images --output outputs/final.png

  # Or point to the folder used previously; assembler will attempt flexible matching:
  python assemble.py --graph outputs/graph.json --images_dir /content/dataset/puzzle1000/images --crops_dir dataset/puzzle1000/cropped_images --output outputs/final.png
"""
from __future__ import annotations
import os
import json
from math import ceil, sqrt
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
from tqdm import tqdm
import traceback
import sys
import tempfile

# ------------------ Helpers for flexible name matching ------------------ #
_STRIP_SUFFIXES = ("_pred", "_gt", "_masks", "_mask", "-pred", "-gt")

def _generate_candidate_keys(piece_id: str) -> List[str]:
    """
    Produce candidate cache keys for a given node id.
    Examples:
      "IMG_1234_pred" -> ["IMG_1234_pred","IMG_1234","img_1234_pred","img_1234"]
      "IMG_1234" -> ["IMG_1234","img_1234"]
    """
    stem = Path(piece_id).stem
    candidates = [stem, stem.lower()]
    # If there is a suffix, also include stripped version
    for suf in _STRIP_SUFFIXES:
        if stem.endswith(suf):
            base = stem[: -len(suf)]
            candidates.append(base)
            candidates.append(base.lower())
    # also include replacing spaces with underscore and vice versa
    if " " in stem:
        candidates.append(stem.replace(" ", "_"))
        candidates.append(stem.replace(" ", "_").lower())
    if "_" in stem:
        candidates.append(stem.replace("_", " "))
        candidates.append(stem.replace("_", " ").lower())
    # ensure unique, preserve order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

# ------------------ Image cache ------------------ #
def build_image_cache(images_dir: Path) -> Dict[str, Path]:
    """Pre-build a lookup cache of all images (much faster).

    Cache keys include:
      - file stem (with case)
      - file stem lowercased
      - filename (with extension)
    """
    cache: Dict[str, Path] = {}
    exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    print(f"Building image cache from {images_dir}...")
    if not images_dir.exists():
        return cache
    for f in images_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix not in exts:
            continue
        stem = f.stem
        cache[stem] = f
        cache[stem.lower()] = f
        cache[f.name] = f
    # show unique files count (some files create multiple cache keys)
    unique_files = len({p for p in cache.values()})
    print(f"Found {unique_files:,} image files (cache size {len(cache):,})")
    return cache



def find_image_fast(cache: Dict[str, Path], piece_id: str) -> Path | None:
    """Fast lookup using pre-built cache with multiple candidate keys and suffix-stripping."""
    # Try exact keys first
    if piece_id in cache:
        return cache[piece_id]
    stem = Path(piece_id).stem
    if stem in cache:
        return cache[stem]
    if stem.lower() in cache:
        return cache[stem.lower()]

    # Try generated candidate keys
    for cand in _generate_candidate_keys(piece_id):
        if cand in cache:
            return cache[cand]
    # As a last resort, try removing common image suffixes appended by extract_features2
    for suf in _STRIP_SUFFIXES:
        if stem.endswith(suf):
            base = stem[: -len(suf)]
            for cand in (base, base.lower()):
                if cand in cache:
                    return cache[cand]

    return None

# ------------------ Graph processing utilities ------------------ #
def build_adjacency_from_graph(graph: Dict) -> Dict[str, List[str]]:
    """Convert graph to adjacency list"""
    adj: Dict[str, List[str]] = {}
    nodes = graph.get("nodes", [])
    if isinstance(nodes, dict):
        for pid in nodes.keys():
            adj[str(pid)] = []
    else:
        for n in nodes:
            if isinstance(n, dict):
                pid = str(n.get("id"))
            else:
                pid = str(n)
            adj[pid] = []
    for e in graph.get("edges", []):
        a = str(e.get("a"))
        b = str(e.get("b"))
        if a not in adj: adj[a] = []
        if b not in adj: adj[b] = []
        if b not in adj[a]: adj[a].append(b)
        if a not in adj[b]: adj[b].append(a)
    return adj

def find_connected_components(adj: Dict[str, List[str]]) -> List[List[str]]:
    visited = set()
    comps: List[List[str]] = []
    for node in adj.keys():
        if node in visited:
            continue
        stack = [node]
        comp = []
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.append(n)
            for nb in adj.get(n, []):
                if nb not in visited:
                    stack.append(nb)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps

def downscale_image_if_needed(im: Image.Image, max_dim: int) -> Image.Image:
    if max_dim <= 0:
        return im
    w, h = im.size
    mx = max(w, h)
    if mx <= max_dim:
        return im
    scale = max_dim / float(mx)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return im.resize((new_w, new_h), Image.LANCZOS)

# ------------------ Assemble main ------------------ #
# change in assemble signature defaults:
def assemble(graph_path: str = "outputs/graph.json",
             images_dir: str = "/content/dataset/puzzle1000/images",
             crops_dir: str | None = "/content/dataset/puzzle1000/cropped_images",
             output_path: str = "outputs/final.png",
             padding: int = 10,
             max_components: int = 20,
             max_image_dim: int = 2048,
             max_total_pixels: int = 200_000_000,
             save_tiles_dir: str | None = None):
    images_dir = Path(images_dir)
    if crops_dir:
        crops_dir = Path(crops_dir)
    graph_path = Path(graph_path)
    output_path = Path(output_path)

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    # If a crops_dir is provided prefer it for lookup; otherwise use images_dir
    lookup_dir = crops_dir if crops_dir and crops_dir.exists() else images_dir
    if not lookup_dir.exists():
        raise FileNotFoundError(f"Images/crops directory not found: {lookup_dir}")

    # Load graph
    print("Loading graph...")
    with graph_path.open('r', encoding='utf-8') as f:
        graph = json.load(f)

    adj = build_adjacency_from_graph(graph)
    comps = find_connected_components(adj)

    print(f"\nGraph statistics:")
    print(f"  Nodes: {len(adj)}")
    print(f"  Components: {len(comps)}")
    print(f"  Top 5 component sizes: {[len(c) for c in comps[:5]]}")
    print(f"  Isolated pieces: {sum(1 for c in comps if len(c) == 1)}")

    # Build image cache from lookup_dir (prefer cropped images if available)
    img_cache = build_image_cache(lookup_dir)
    print(f"Using lookup directory: {lookup_dir} (cache entries: {len(img_cache):,})")
    comps_to_process = comps[:max_components]
    print(f"\nProcessing top {len(comps_to_process)} components...")

    comp_images: List[List[Tuple[str, Image.Image]]] = []
    max_w = max_h = 0
    total_loaded = 0
    missing_nodes = []

    for comp_idx, comp in enumerate(tqdm(comps_to_process, desc="Loading images")):
        imgs: List[Tuple[str, Image.Image]] = []
        for pid in comp:
            img_path = find_image_fast(img_cache, pid)
            if img_path is None:
                # if crops_dir was provided and failed, try fallback to images_dir
                if crops_dir and images_dir.exists():
                    fallback_cache = build_image_cache(images_dir)
                    img_path = find_image_fast(fallback_cache, pid)
                if img_path is None:
                    missing_nodes.append(pid)
                    continue
            try:
                im = Image.open(img_path).convert('RGBA')
                if max_image_dim and max(im.size):
                    im = downscale_image_if_needed(im, max_image_dim)
                imgs.append((pid, im))
                w, h = im.size
                max_w = max(max_w, w)
                max_h = max(max_h, h)
                total_loaded += 1
            except Exception as e:
                print(f"Warning: failed to open {img_path}: {e}")
                continue
        if imgs:
            comp_images.append(imgs)

    if missing_nodes:
        print(f"\nWarning: {len(missing_nodes)} graph nodes had no matching image in {lookup_dir}. Examples:")
        for m in missing_nodes[:20]:
            print(" ", m)
        print("If you expected cropped piece images, ensure crop filenames match node ids (or run with --strip_label_suffix at graph build time).")
        print("You can also provide --crops_dir to assemble pointing at your cropped images folder.")

    if max_w == 0 or max_h == 0:
        raise RuntimeError("No images were loaded (check that crop filenames match graph node ids)")

    print(f"\nLoaded {total_loaded} images across {len(comp_images)} components")
    print(f"Max tile size (post-downscale): {max_w}x{max_h}")

    # Create component canvases
    print("\nTiling components...")
    comp_canvases: List[Image.Image] = []
    saved_tile_paths: List[Path] = []
    tmp_dir = None
    if save_tiles_dir:
        tmp_dir = Path(save_tiles_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="assembler_tiles_"))
        print(f"(Temporary component canvas dir: {tmp_dir})")

    try:
        for comp_idx, imgs in enumerate(tqdm(comp_images, desc="Creating tiles")):
            if not imgs:
                continue
            n = len(imgs)
            cols = max(1, int(ceil(sqrt(n))))
            rows = int(ceil(n / cols))
            canvas_w = cols * (max_w + padding) + padding
            canvas_h = rows * (max_h + padding) + padding
            canvas = Image.new('RGBA', (canvas_w, canvas_h), (255, 255, 255, 255))
            for idx, (pid, img) in enumerate(imgs):
                col = idx % cols
                row = idx // cols
                x = padding + col * (max_w + padding)
                y = padding + row * (max_h + padding)
                w, h = img.size
                ox = x + (max_w - w) // 2
                oy = y + (max_h - h) // 2
                canvas.paste(img, (ox, oy), img)
            tile_path = tmp_dir / f"component_{comp_idx:03d}.png"
            try:
                canvas.save(tile_path)
                saved_tile_paths.append(tile_path)
            except Exception as e:
                print(f"Warning: failed to save tile {tile_path}: {e}")
            comp_canvases.append(canvas)
    except Exception as e:
        print("Exception while creating component canvases:")
        traceback.print_exc()
        if saved_tile_paths:
            print("Saved component canvases to:", tmp_dir)
        raise

    if not comp_canvases:
        raise RuntimeError("No component canvases created")

    # Estimate final size and downscale if too large
    total_w = max(c.width for c in comp_canvases)
    total_h = sum(c.height for c in comp_canvases) + padding * (len(comp_canvases) + 1)
    estimated_pixels = total_w * total_h
    print(f"\nEstimated final dimensions: {total_w} x {total_h} = {estimated_pixels:,} pixels")

    if estimated_pixels > max_total_pixels:
        scale = (max_total_pixels / float(estimated_pixels)) ** 0.5
        scale = min(1.0, max(1e-4, scale))
        print(f"Estimated final image too large (> {max_total_pixels:,} px). Downscaling component canvases by factor {scale:.3f} to fit memory.")
        new_canvases = []
        for idx, c in enumerate(comp_canvases):
            new_w = max(1, int(c.width * scale))
            new_h = max(1, int(c.height * scale))
            try:
                c_small = c.resize((new_w, new_h), Image.LANCZOS)
                try:
                    scaled_path = tmp_dir / f"component_{idx:03d}_scaled.png"
                    c_small.save(scaled_path)
                except Exception:
                    pass
                new_canvases.append(c_small)
            except Exception as e:
                print(f"Warning: failed to downscale canvas {idx}: {e}")
                new_canvases.append(c)
        comp_canvases = new_canvases
        total_w = max(c.width for c in comp_canvases)
        total_h = sum(c.height for c in comp_canvases) + padding * (len(comp_canvases) + 1)
        estimated_pixels = total_w * total_h
        print(f"New estimated dimensions: {total_w} x {total_h} = {estimated_pixels:,} pixels")

    # Stack vertically to produce final
    print("\nStacking components...")
    try:
        final = Image.new('RGBA', (total_w, total_h), (255, 255, 255, 255))
        y = padding
        for c in tqdm(comp_canvases, desc="Pasting components"):
            final.paste(c, (0, y), c)
            y += c.height + padding
        final_rgb = final.convert('RGB')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_rgb.save(output_path)
        print(f"\n✅ Saved assembled image to {output_path}")
        print(f"   Dimensions: {final_rgb.size[0]}x{final_rgb.size[1]}")
    except KeyboardInterrupt:
        print("\nInterrupted by user (KeyboardInterrupt). Attempting to save partial output...")
        try:
            if 'final' in locals():
                partial_path = output_path.parent / (output_path.stem + "_partial.png")
                final_rgb = final.convert('RGB')
                final_rgb.save(partial_path)
                print("Saved partial assembled image to", partial_path)
            else:
                print("No final image in memory to save.")
            print("Component canvases are saved in:", tmp_dir)
        except Exception as e:
            print("Failed to save partial image:", e)
        raise
    except MemoryError:
        print("\nMemoryError during final assembly. Attempting to downscale canvases and retry once...")
        try:
            scale = 0.5
            new_canvases = [c.resize((max(1, int(c.width*scale)), max(1, int(c.height*scale))), Image.LANCZOS) for c in comp_canvases]
            total_w = max(c.width for c in new_canvases)
            total_h = sum(c.height for c in new_canvases) + padding * (len(new_canvases) + 1)
            final = Image.new('RGBA', (total_w, total_h), (255,255,255,255))
            y = padding
            for c in tqdm(new_canvases, desc="Pasting components (retry)"):
                final.paste(c, (0, y), c)
                y += c.height + padding
            final_rgb = final.convert('RGB')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_rgb.save(output_path)
            print(f"\n✅ Saved assembled image to {output_path} (after downscale retry)")
            print(f"   Dimensions: {final_rgb.size[0]}x{final_rgb.size[1]}")
        except Exception as e:
            print("Retry failed. Components saved in:", tmp_dir)
            traceback.print_exc()
            raise
    except Exception as e:
        print("Unhandled exception while stacking components:")
        traceback.print_exc()
        print("Saved component canvases (if any) in:", tmp_dir)
        raise

# ------------------ CLI ------------------ #
def main():
    import argparse
    p = argparse.ArgumentParser(description="Robust puzzle assembler (supports cropped piece images)")
    p.add_argument('--graph', type=str, default="outputs/graph.json", help="Graph JSON")
    p.add_argument('--images_dir', type=str, default="/content/dataset/puzzle1000/images", help="Fallback images folder (scene photos)")
    p.add_argument('--crops_dir', type=str, default="dataset/puzzle1000/cropped_images", help="Folder with cropped piece PNGs (preferred)")
    p.add_argument('--output', type=str, default="outputs/final.png")
    p.add_argument('--max-components', type=int, default=20, help="Limit number of components to process")
    p.add_argument('--padding', type=int, default=10, help="Padding between tiles")
    p.add_argument('--max-image-dim', type=int, default=2048, help="Downscale input images to this max side (0=disable)")
    p.add_argument('--max-total-pixels', type=int, default=200_000_000, help="Max total pixels for final image before downscaling")
    p.add_argument('--save-tiles-dir', type=str, default=None, help="If set, saves component canvases to this directory (for debugging)")
    args = p.parse_args()

    assemble(graph_path=args.graph,
             images_dir=args.images_dir,
             crops_dir=args.crops_dir,
             output_path=args.output,
             padding=args.padding,
             max_components=args.max_components,
             max_image_dim=args.max_image_dim,
             max_total_pixels=args.max_total_pixels,
             save_tiles_dir=args.save_tiles_dir)

if __name__ == '__main__':
    main()