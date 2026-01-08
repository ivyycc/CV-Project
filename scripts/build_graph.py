from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List
import pickle
import sys
from collections import defaultdict
import time

import networkx as nx
import numpy as np

# Colab detection to set sensible defaults
def in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

DEFAULT_FEATURES = "outputs/features_emb.pkl" if in_colab() else "outputs/features_emb.pkl"
DEFAULT_OUT = "outputs/graph.json" if in_colab() else "outputs/graph.json"


# Try to import matcher from local file (edge_matching.py or matching.py)
# Prefer edge_matching.py (common filename in this repo)
try:
    from matching import EdgeMatcher, PieceTypClassifier
except Exception:
    try:
        from matching import EdgeMatcher, PieceTypClassifier  # fallback name
    except Exception as e:
        print("ERROR: Could not import EdgeMatcher and PieceTypClassifier.", file=sys.stderr)
        print("Make sure your matching code is saved as edge_matching.py (or matching.py) in the repo root and defines EdgeMatcher and PieceTypClassifier.", file=sys.stderr)
        raise


def load_features(path: Path) -> Dict[str, Dict]:
    with open(path, 'rb') as f:
        features = pickle.load(f)
    if not isinstance(features, dict):
        raise RuntimeError("features.pkl must contain a dict keyed by piece id")
    return features


def build_graph_from_features(features: Dict[str, Dict],
                              top_k: int = 10,
                              threshold: float = 0.3,
                              matcher: EdgeMatcher = None,
                              use_adaptive: bool = True,
                              min_balance_ratio: float = 0.4) -> Dict:
    """
    Return graph dict with nodes and edges (reciprocal matches only).

    Improvements:
    - Use mutual inclusion in the top-K candidate lists to detect reciprocity
      and use the actual reciprocal edge index reported by the other piece.
    - Add min_balance_ratio to filter out highly asymmetric pairings.
    """
    if matcher is None:
        matcher = EdgeMatcher()

    piece_ids = list(features.keys())
    print(f"Computing matches for {len(piece_ids)} pieces (top_k={top_k}, threshold={threshold}, use_adaptive={use_adaptive})")
    t0 = time.time()

    # 1) For each piece, compute top_k candidate matches per edge
    all_matches: Dict[str, Dict[int, List[tuple]]] = {}
    for i, pid in enumerate(piece_ids):
        pf = features[pid]
        matches = matcher.find_best_matches(pid, pf, features, top_k=top_k, threshold=threshold, use_adaptive=use_adaptive)
        # matches is expected: Dict[edge_idx, List[(other_pid, other_edge_idx, score)]]
        all_matches[pid] = matches
        if (i + 1) % 50 == 0 or (i + 1) == len(piece_ids):
            print(f"  matched {i+1}/{len(piece_ids)} pieces")

    # Count candidate matches
    n_cands = sum(len(cands) for m in all_matches.values() for cands in m.values())
    print(f"Total candidate matches before reciprocity filtering: {n_cands}")

    # 2) Find reciprocal pairs by mutual inclusion in top-k lists
    edges = []
    visited_pairs = set()  # avoid dupes; store frozenset(((a_pid,a_eidx),(b_pid,b_eidx)))
    for a_pid, edge_map in all_matches.items():
        for a_eidx, cand_list in edge_map.items():
            if not cand_list:
                continue
            for (b_pid, b_eidx_from_a, a_score) in cand_list:
                # skip self
                if b_pid == a_pid:
                    continue

                # Look for reciprocal mention: does b list a in any of its edge lists?
                if b_pid not in all_matches:
                    continue
                reciprocal_found = False
                b_score = None
                b_eidx_from_b = None
                for be_idx, b_cands in all_matches[b_pid].items():
                    if not b_cands:
                        continue
                    for (opp_pid, opp_eidx, sc) in b_cands:
                        if opp_pid == a_pid:
                            reciprocal_found = True
                            b_score = float(sc)
                            b_eidx_from_b = int(opp_eidx)
                            break
                    if reciprocal_found:
                        break

                if not reciprocal_found:
                    continue

                # Use the reciprocal edge index reported by B (b_eidx_from_b). If not present, fall back to b_eidx_from_a
                b_eidx = b_eidx_from_b if b_eidx_from_b is not None else int(b_eidx_from_a)

                # avoid duplicates (unordered pair of specific edges)
                pair_key = frozenset(((a_pid, int(a_eidx)), (b_pid, int(b_eidx))))
                if pair_key in visited_pairs:
                    continue

                # require scores not too asymmetric and above threshold
                if b_score is None:
                    continue
                max_score = max(float(a_score), float(b_score))
                min_score = min(float(a_score), float(b_score))
                if max_score < threshold:
                    # skip pairs whose best side does not meet base threshold
                    continue
                if (min_score / (max_score + 1e-12)) < min_balance_ratio:
                    # extremely one-sided: skip unless you want more recall at cost of precision
                    continue

                # Accept the reciprocal match
                sym_score = float((float(a_score) + float(b_score)) / 2.0)
                edge_record = {
                    "a": a_pid,
                    "a_edge": int(a_eidx),
                    "b": b_pid,
                    "b_edge": int(b_eidx),
                    "score": sym_score,
                    "a_score": float(a_score),
                    "b_score": float(b_score)
                }
                edges.append(edge_record)
                visited_pairs.add(pair_key)

    t1 = time.time()
    print(f"Matching completed in {t1 - t0:.1f}s. Found {len(edges)} reciprocal matches (edges).")

    # 3) Build nodes with piece type using PieceTypClassifier
    nodes = []
    for pid in piece_ids:
        try:
            ptype = PieceTypClassifier.classify_piece(features[pid])
        except Exception:
            ptype = "unknown"
        nodes.append({"id": pid, "type": ptype})

    graph = {"nodes": nodes, "edges": edges}
    return graph


def analyze_graph(graph: Dict):
    # basic networkx analysis
    G = nx.Graph()
    for node in graph["nodes"]:
        G.add_node(node["id"], type=node.get("type", "unknown"))
    for e in graph["edges"]:
        a = e["a"]; b = e["b"]
        G.add_edge(a, b, score=e.get("score", 0.0))
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    comps = list(nx.connected_components(G))
    comp_sizes = sorted([len(c) for c in comps], reverse=True)
    print(f"Graph: {n_nodes} nodes, {n_edges} undirected edges, {len(comps)} connected components")
    print("Top component sizes:", comp_sizes[:5])
    degs = [d for _, d in G.degree()]
    if degs:
        print("Degree: min", min(degs), "max", max(degs), "mean", sum(degs)/len(degs))
    return {"n_nodes": n_nodes, "n_edges": n_edges, "n_components": len(comps), "component_sizes": comp_sizes, "degrees": degs}


def save_graph(graph: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(graph, f, indent=2)
    print("Saved graph (nodes+edges) to", out_path)


def main():
    p = argparse.ArgumentParser(description="Build adjacency graph from features.pkl (Colab-friendly defaults)")
    p.add_argument("--features", type=str, default=DEFAULT_FEATURES, help="Path to features.pkl")
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output graph JSON file (nodes+edges)")
    p.add_argument("--top_k", type=int, default=10, help="Top-K candidates per edge")
    p.add_argument("--threshold", type=float, default=0.3, help="Minimum similarity to consider candidate")
    p.add_argument("--min_balance_ratio", type=float, default=0.4, help="Minimum ratio min(score)/max(score) to accept reciprocal pair (0..1). Lower => more recall")
    p.add_argument("--use_adaptive", action="store_true", help="Enable matcher adaptive thresholding (default off in this script). Use --use_adaptive to enable.")
    args = p.parse_args()

    features_path = Path(args.features)
    if not features_path.exists():
        print("Features file not found:", features_path, file=sys.stderr)
        raise SystemExit(1)

    print("Loading features from:", features_path)
    features = load_features(features_path)
    print(f"Loaded features for {len(features)} pieces")

    matcher = EdgeMatcher()  # default weights; tune if needed
    graph = build_graph_from_features(features, top_k=args.top_k, threshold=args.threshold, matcher=matcher, use_adaptive=args.use_adaptive, min_balance_ratio=args.min_balance_ratio)
    
    stats = analyze_graph(graph)

    # Build adjacency list (undirected)
    adjacency = {}
    for e in graph["edges"]:
        a, b = e["a"], e["b"]
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    # Save adjacency list in a simple JSON (submission format)
    out_path = Path(args.out)
    adj_out = out_path.parent / "graph.json"
    adj_out.parent.mkdir(parents=True, exist_ok=True)
    with open(adj_out, "w") as f:
        json.dump(adjacency, f, indent=2)
    print("✅ Saved adjacency list to", adj_out)

    # Save the detailed graph (nodes+edges) to --out
    save_graph(graph, out_path)

    print("Graph summary:", stats)
    print("Done.")


if __name__ == "__main__":
    main()