#!/usr/bin/env python3
"""
Compute compact per-edge embeddings from features.pkl and save them back.

Usage:
  python compute_edge_embeddings.py --input outputs/test_features.pkl --output outputs/features_emb.pkl --n_components 16

What it does:
 - For each edge it builds a vector from:
     - curvature summary stats (mean, std, min, max, 25/50/75 percentiles)
     - Fourier magnitudes (first n_fourier e.g. 10)
     - color hist left/right (flattened)
     - mean colors left/right (3 + 3)
     - MR8 texture if present (12)
 - Runs StandardScaler then PCA (n_components) and writes 'embed' into each edge dict.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def safe_array(x, dtype=np.float32):
    if x is None:
        return np.array([], dtype=dtype)
    a = np.asarray(x, dtype=dtype).ravel()
    return a

def curvature_stats(curv):
    curv = safe_array(curv)
    if curv.size == 0:
        return np.zeros(7, dtype=np.float32)
    p25 = np.percentile(curv, 25)
    p50 = np.percentile(curv, 50)
    p75 = np.percentile(curv, 75)
    return np.array([curv.mean(), curv.std(), curv.min(), curv.max(), p25, p50, p75], dtype=np.float32)

def build_edge_vector(edge, n_fourier=10, bins_hist=None):
    # curvature stats
    cstats = curvature_stats(edge.get('curvature', None))
    # fourier mags
    four = safe_array(edge.get('fourier', None))
    if four.size < n_fourier:
        four_padded = np.zeros(n_fourier, dtype=np.float32)
        four_padded[:four.size] = four[:]
    else:
        four_padded = four[:n_fourier]
    # color hist left/right (already normalized)
    hist_l = safe_array(edge.get('color_hist_left', None))
    hist_r = safe_array(edge.get('color_hist_right', None))
    if hist_l.size == 0 and hist_r.size == 0:
        # fallback zeros of default 64 bins (4^3) if nothing provided
        hist_l = np.zeros(64, dtype=np.float32)
        hist_r = np.zeros(64, dtype=np.float32)
    # mean colors
    mean_l = safe_array(edge.get('mean_left', np.zeros(3)), dtype=np.float32)
    mean_r = safe_array(edge.get('mean_right', np.zeros(3)), dtype=np.float32)
    # mr8 if present
    mr8 = safe_array(edge.get('mr8', None))
    # length normalized (optional)
    length = np.array([edge.get('length', 0.0)], dtype=np.float32)
    vec = np.concatenate([cstats, four_padded, hist_l, hist_r, mean_l/255.0, mean_r/255.0, mr8, length])
    return vec

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input features pickle")
    p.add_argument("--output", required=True, help="Output features pickle (with edge['embed'])")
    p.add_argument("--n_components", type=int, default=16, help="PCA components for embeddings")
    p.add_argument("--n_fourier", type=int, default=10, help="How many fourier magnitudes to include")
    args = p.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)

    with open(inp, "rb") as f:
        feats = pickle.load(f)

    # collect all per-edge vectors
    edge_keys = []  # tuples (piece_id, edge_idx)
    vectors = []
    for pid, pfeats in feats.items():
        for eidx, ed in enumerate(pfeats.get('edges', [])):
            vec = build_edge_vector(ed, n_fourier=args.n_fourier)
            edge_keys.append((pid, eidx))
            vectors.append(vec)

    if not vectors:
        raise RuntimeError("No edges found in features file.")

    X = np.vstack(vectors).astype(np.float32)
    # impute any NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize then PCA
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=min(args.n_components, Xs.shape[1]), svd_solver='auto', random_state=0)
    Xp = pca.fit_transform(Xs).astype(np.float32)

    # write back embeddings
    assert Xp.shape[0] == len(edge_keys)
    for (pid, eidx), emb in zip(edge_keys, Xp):
        # store into feats under edges[eidx]['embed']
        try:
            feats[pid]['edges'][eidx]['embed'] = emb
        except Exception:
            pass

    # Also save PCA/scaler models for later use (optional)
    meta = {
        'pca_components': pca.n_components_,
        'pca_explained_ratio': pca.explained_variance_ratio_.tolist(),
    }

    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "wb") as f:
        pickle.dump(feats, f)
    print(f"Saved features with embeddings to: {outp}")
    print("Meta:", meta)

if __name__ == "__main__":
    main()