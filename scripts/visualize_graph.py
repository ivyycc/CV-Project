#!/usr/bin/env python3
"""
visualize_graph.py (patched draw_match_overlay)

This version crops each source image to the piece bbox (with padding),
scales crops to a common height, reprojects edge points into the crop
coordinates, draws thick/high-contrast edge highlights, and adds a small
zoom-in inset for the matched edges so overlays are easy to inspect.

Usage (same as before):
  python visualize_graph.py --graph outputs/graph.json --features outputs/test_features.pkl --out_dir outputs/visuals --overlay_matches 50
"""
from __future__ import annotations
import os
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont

# ---------- Helpers ----------
def load_graph_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def load_features_pkl(path: Path) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)

def build_nx_graph(graph_dict: dict) -> nx.Graph:
    """Convert graph dict {nodes: [...], edges: [...]} into networkx Graph with attributes."""
    G = nx.Graph()
    for n in graph_dict.get("nodes", []):
        nid = n["id"]
        G.add_node(nid, type=n.get("type", "unknown"))
    for e in graph_dict.get("edges", []):
        a = e["a"]; b = e["b"]
        score = e.get("score", 0.0)
        a_edge = e.get("a_edge")
        b_edge = e.get("b_edge")
        G.add_edge(a, b, score=score, a_edge=a_edge, b_edge=b_edge, a_score=e.get("a_score"), b_score=e.get("b_score"))
    return G

def plot_overview_graph(G: nx.Graph, out_path: Path, title: str = "Adjacency graph", layout: str = "kamada"):
    """Plot overview graph colored by node type and sized by degree. Save PNG."""
    plt.figure(figsize=(12, 10))
    # choose layout
    if layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.circular_layout(G)

    # node attributes
    types = nx.get_node_attributes(G, "type")
    unique_types = sorted(set(types.values()))
    cmap = plt.get_cmap("tab10")
    type2color = {t: cmap(i % 10) for i, t in enumerate(unique_types)}
    node_colors = [type2color.get(types.get(n, "unknown"), (0.6,0.6,0.6)) for n in G.nodes()]

    degrees = dict(G.degree())
    min_sz, max_sz = 80, 800
    if degrees:
        deg_vals = np.array(list(degrees.values()), dtype=float)
        if deg_vals.size == 0:
            node_sizes = [min_sz for _ in degrees]
        elif np.max(deg_vals) == np.min(deg_vals):
            node_sizes = [min_sz + 20 for _ in degrees]
        else:
            node_sizes = list(min_sz + (deg_vals - deg_vals.min()) / (np.ptp(deg_vals) + 1e-12) * (max_sz - min_sz))
    else:
        node_sizes = [min_sz for _ in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="#444444")
    nx.draw_networkx_labels(G, pos, font_size=8)

    legend_elems = [Line2D([0], [0], marker='o', color='w', label=t, markerfacecolor=type2color[t], markersize=10) for t in unique_types]
    if legend_elems:
        plt.legend(handles=legend_elems, title="piece type", loc="upper right")

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved overview graph PNG to: {out_path}")

def export_graph_formats(G: nx.Graph, out_prefix: Path):
    """Save GraphML and GEXF for interactive exploration (Gephi etc)."""
    graphml_path = out_prefix.with_suffix(".graphml")
    gexf_path = out_prefix.with_suffix(".gexf")
    try:
        nx.write_graphml(G, graphml_path)
        print(f"Saved GraphML to: {graphml_path}")
    except Exception as ex:
        print(f"Warning: failed to save GraphML: {ex}")
    try:
        nx.write_gexf(G, gexf_path)
        print(f"Saved GEXF to: {gexf_path}")
    except Exception as ex:
        print(f"Warning: failed to save GEXF: {ex}")

# ---------- Improved overlay renderer ----------
def draw_match_overlay(features: dict, edge_record: dict, out_path: Path,
                       max_side_points: int = 300, pad: int = 12, common_height: int = 512):
    """
    Crop each source image to the piece bbox (+pad), scale both crops to a common height,
    reproject edge points into the cropped/scaled coordinates, and draw clear overlays.

    Returns True if overlay saved; False otherwise.
    """
    a_id = edge_record['a']
    b_id = edge_record['b']
    a_eidx = int(edge_record['a_edge'])
    b_eidx = int(edge_record['b_edge'])

    fa = features.get(a_id)
    fb = features.get(b_id)
    if fa is None or fb is None:
        print(f"Missing features for {a_id} or {b_id}, skipping overlay.")
        return False

    ia = fa.get('source_image_path')
    ib = fb.get('source_image_path')
    if ia is None or ib is None:
        print(f"No source image path for {a_id} or {b_id}, skipping overlay.")
        return False

    # bbox from features: (x,y,w,h)
    bx_a = fa.get('bbox', None)
    bx_b = fb.get('bbox', None)
    if bx_a is None or bx_b is None:
        print(f"No bbox in features for {a_id} or {b_id}, skipping overlay.")
        return False

    try:
        img_a_full = Image.open(ia).convert("RGBA")
        img_b_full = Image.open(ib).convert("RGBA")
    except Exception as ex:
        print(f"Failed to open images for overlay: {ex}")
        return False

    # crop helper with pad, clamp to image
    def crop_with_pad(img, bbox, pad):
        x, y, w, h = bbox
        x0 = max(0, int(x - pad))
        y0 = max(0, int(y - pad))
        x1 = min(img.width, int(x + w + pad))
        y1 = min(img.height, int(y + h + pad))
        return img.crop((x0, y0, x1, y1)), (x0, y0)

    crop_a, off_a = crop_with_pad(img_a_full, bx_a, pad)
    crop_b, off_b = crop_with_pad(img_b_full, bx_b, pad)

    # scale both crops to a common height for side-by-side display
    def scale_to_height(img, target_h):
        w,h = img.size
        if h == 0:
            return img, 1.0
        scale = float(target_h) / float(h)
        new_w = int(round(w * scale))
        return img.resize((new_w, target_h), resample=Image.LANCZOS), scale

    # choose target height = min(common_height, max available height)
    # but limit to avoid extreme upscaling
    target_h = common_height
    crop_a_h = min(crop_a.height, common_height)
    crop_b_h = min(crop_b.height, common_height)
    # scale both to common height equal to min(crop heights, common_height)
    chosen_h = min(crop_a_h, crop_b_h, common_height)
    crop_a_s, s_a = scale_to_height(crop_a, chosen_h)
    crop_b_s, s_b = scale_to_height(crop_b, chosen_h)

    # create canvas side-by-side
    margin = 16
    canvas_w = crop_a_s.width + crop_b_s.width + margin
    canvas_h = chosen_h
    canvas = Image.new("RGBA", (canvas_w, canvas_h + 120), (255,255,255,255))  # extra bottom space for inset/labels
    canvas.paste(crop_a_s, (0, 0))
    canvas.paste(crop_b_s, (crop_a_s.width + margin, 0))

    draw = ImageDraw.Draw(canvas, 'RGBA')

    # map original edge points into crop+scale coordinates
    def map_points_to_crop(edge_points, crop_off, scale):
        # edge_points: Nx2 in original image coords (x,y)
        pts = np.asarray(edge_points, dtype=np.float32)
        # subtract crop offset, then scale
        pts[:,0] = (pts[:,0] - crop_off[0]) * scale
        pts[:,1] = (pts[:,1] - crop_off[1]) * scale
        return pts

    # get edges (safety checks)
    edges_a = fa.get('edges', [])
    edges_b = fb.get('edges', [])
    if a_eidx < 0 or a_eidx >= len(edges_a) or b_eidx < 0 or b_eidx >= len(edges_b):
        print(f"Edge indices out of range for {a_id} or {b_id}, skipping overlay.")
        return False

    pts_a = map_points_to_crop(edges_a[a_eidx]['points'], off_a, s_a)
    pts_b = map_points_to_crop(edges_b[b_eidx]['points'], off_b, s_b)

    # downsample if too many points
    def downsample(pts, maxp):
        if pts.shape[0] <= maxp:
            return pts
        idxs = np.linspace(0, pts.shape[0]-1, maxp).astype(int)
        return pts[idxs]

    pts_a = downsample(pts_a, max_side_points)
    pts_b = downsample(pts_b, max_side_points)

    # compute midpoint of each edge for connector
    mid_a = (tuple(map(int, pts_a.mean(axis=0))))
    mid_b = (tuple(map(int, pts_b.mean(axis=0))))
    # adjust mid_b x for its pasted offset
    mid_b = (mid_b[0] + crop_a_s.width + margin, mid_b[1])

    # draw full contours lightly (if available)
    def draw_contour_on_crop(f, crop_off, scale, offset_x):
        contour = np.asarray(f.get('contour', []), dtype=np.int32)
        if contour.size == 0:
            return
        pts = ((contour - np.array(crop_off)) * scale).tolist()
        pts_t = [(int(round(x))+offset_x, int(round(y))) for x,y in pts]
        if len(pts_t) > 1:
            draw.line(pts_t + [pts_t[0]], fill=(200,200,200,160), width=2)

    draw_contour_on_crop(fa, off_a, s_a, 0)
    draw_contour_on_crop(fb, off_b, s_b, crop_a_s.width + margin)

    # draw matched edges thick and high-contrast
    pts_a_t = [(int(round(x)), int(round(y))) for x,y in pts_a.tolist()]
    pts_b_t = [(int(round(x) + crop_a_s.width + margin), int(round(y))) for x,y in pts_b.tolist()]

    if len(pts_a_t) >= 2:
        draw.line(pts_a_t, fill=(220, 30, 30, 220), width=6)
        for p in pts_a_t:
            draw.ellipse((p[0]-5, p[1]-5, p[0]+5, p[1]+5), fill=(255,255,255,220), outline=(120,0,0,220))
    if len(pts_b_t) >= 2:
        draw.line(pts_b_t, fill=(30, 80, 220, 220), width=6)
        for p in pts_b_t:
            draw.ellipse((p[0]-5, p[1]-5, p[0]+5, p[1]+5), fill=(255,255,255,220), outline=(0,0,120,220))

    # draw connector between edge midpoints
    draw.line([mid_a, mid_b], fill=(40, 160, 60, 200), width=4)

    # write score label near middle
    score = edge_record.get('score', None)
    label = f"{a_id} e{a_eidx}  ↔  {b_id} e{b_eidx}"
    # choose a font if available; PIL default otherwise
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    txt_xy = (10, canvas_h + 6)
    draw.text(txt_xy, label, fill=(0,0,0,220), font=font)
    if score is not None:
        draw.text((10, canvas_h + 28), f"score: {score:.3f}", fill=(0,0,0,200), font=font)

    # inset zoom: crop small regions around the matched edge midpoints and paste below
    try:
        inset_w = 220
        inset_h = 120
        # compute source boxes around midpoints in original crops (before offsets)
        # Use a box size relative to chosen_h
        box_size = max(80, int(chosen_h * 0.25))
        # box for A (in crop_a coords)
        ma_x = int(round(pts_a.mean(axis=0)[0]))
        ma_y = int(round(pts_a.mean(axis=0)[1]))
        box_a = (max(0, ma_x - box_size//2), max(0, ma_y - box_size//2),
                 min(crop_a_s.width, ma_x + box_size//2), min(crop_a_s.height, ma_y + box_size//2))
        mb_x = int(round(pts_b.mean(axis=0)[0]))
        mb_y = int(round(pts_b.mean(axis=0)[1]))
        # adjust for pasted offset when sampling from crop_b_s (we use its local coords)
        box_b = (max(0, mb_x - box_size//2), max(0, mb_y - box_size//2),
                 min(crop_b_s.width, mb_x + box_size//2), min(crop_b_s.height, mb_y + box_size//2))

        inset_a = crop_a_s.crop(box_a).resize((inset_w//2, inset_h), resample=Image.LANCZOS)
        inset_b = crop_b_s.crop(box_b).resize((inset_w//2, inset_h), resample=Image.LANCZOS)
        inset_canvas = Image.new("RGBA", (inset_w, inset_h), (255,255,255,255))
        inset_canvas.paste(inset_a, (0,0))
        inset_canvas.paste(inset_b, (inset_w//2,0))
        canvas.paste(inset_canvas, (10, canvas_h + 48))
    except Exception:
        # if any issue, skip inset
        pass

    # save result
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(str(out_path), quality=92)
    return True

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser(description="Visualize graph and matched pairs from features + graph json (improved overlays)")
    p.add_argument("--graph", type=str, required=True, help="Detailed graph JSON (nodes+edges) output by build_graph.py")
    p.add_argument("--features", type=str, required=True, help="Features pickle (outputs/features_*.pkl)")
    p.add_argument("--out_dir", type=str, default="outputs/visuals", help="Where to save visual outputs")
    p.add_argument("--layout", type=str, default="kamada", choices=("kamada","spring","circular","spectral"), help="Layout for overview graph")
    p.add_argument("--overlay_matches", type=int, default=0, help="If >0, render this many matched-edge overlays (saves images)")
    p.add_argument("--max_overlay", type=int, default=500, help="Max overlays to render (safety)")
    args = p.parse_args()

    graph_path = Path(args.graph)
    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading graph:", graph_path)
    graph_dict = load_graph_json(graph_path)
    print("Loading features:", features_path)
    features = load_features_pkl(features_path)

    print("Building NetworkX graph...")
    G = build_nx_graph(graph_dict)

    # overview plot
    png_path = out_dir / "graph_overview.png"
    plot_overview_graph(G, png_path, title="Puzzle pieces adjacency graph", layout=args.layout)

    # export formats
    export_prefix = out_dir / "graph_export"
    export_graph_formats(G, export_prefix)

    # Save simple adjacency list (if not already saved)
    adj_path = out_dir / "adjacency_list.json"
    adjacency = defaultdict(list)
    for u, v, d in G.edges(data=True):
        adjacency[u].append(v)
        adjacency[v].append(u)
    with open(adj_path, "w") as f:
        json.dump(adjacency, f, indent=2)
    print("Saved adjacency_list.json to", adj_path)

    # Optional: overlay matched pairs (uses original images referenced in features)
    n_over = int(args.overlay_matches)
    if n_over > 0:
        print(f"Rendering up to {n_over} matched overlays (this may create many images)...")
        edges = graph_dict.get("edges", [])
        # sort by score desc to show best matches first
        edges_sorted = sorted(edges, key=lambda e: e.get("score", 0.0), reverse=True)
        n = min(n_over, len(edges_sorted), args.max_overlay)
        for i, edge in enumerate(edges_sorted[:n]):
            out_image = out_dir / f"match_{i:04d}_{edge['a']}_e{edge['a_edge']}_vs_{edge['b']}_e{edge['b_edge']}.jpg"
            ok = draw_match_overlay(features, edge, out_image)
            if ok:
                print(f" Saved overlay {i+1}/{n} -> {out_image}")
            else:
                print(f" Skipped overlay {i+1}/{n} (missing data)")

    print("All done. Visuals saved to:", out_dir)

if __name__ == "__main__":
    main()