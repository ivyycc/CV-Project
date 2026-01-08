# Complete EdgeMatcher with embedding support (replace previous incomplete file)
# Saves: edge_matching.py
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any


class EdgeMatcher:
    """Edge matcher with curvature/fourier/color/embed signals and adaptive/top-k candidate trimming."""

    def __init__(
        self,
        shape_weight: float = 0.50,
        color_weight: float = 0.25,
        fourier_weight: float = 0.10,
        embed_weight: float = 0.15,
    ):
        total = float(shape_weight + color_weight + fourier_weight + embed_weight)
        if total <= 0:
            total = 1.0
        self.shape_weight = shape_weight / total
        self.color_weight = color_weight / total
        self.fourier_weight = fourier_weight / total
        self.embed_weight = embed_weight / total

    # ---- compatibility / helpers ----
    def are_compatible_types(self, type1: str, type2: str) -> bool:
        if type1 == "flat" and type2 == "flat":
            return False
        if type1 == "flat" or type2 == "flat":
            return True
        if (type1 == "tab" and type2 == "blank") or (type1 == "blank" and type2 == "tab"):
            return True
        return False

    def _safe_array(self, x):
        if x is None:
            return np.array([], dtype=np.float32)
        return np.asarray(x, dtype=np.float32).ravel()

    # ---- shape / curvature ----
    def compute_curvature_similarity(self, curv1, curv2, invert: bool = False) -> float:
        c1 = self._safe_array(curv1)
        c2 = self._safe_array(curv2)
        if c1.size == 0 or c2.size == 0:
            return 0.0
        # normalize
        c1 = (c1 - c1.mean()) / (c1.std() + 1e-9)
        c2 = (c2 - c2.mean()) / (c2.std() + 1e-9)
        if invert:
            c2 = -c2
        # try forward and reversed
        try:
            corr_f = np.corrcoef(c1, c2)[0, 1]
        except Exception:
            corr_f = 0.0
        try:
            corr_r = np.corrcoef(c1, c2[::-1])[0, 1]
        except Exception:
            corr_r = 0.0
        best = max(np.nan_to_num(corr_f), np.nan_to_num(corr_r))
        sim = (best + 1.0) / 2.0
        return float(np.clip(sim, 0.0, 1.0))

    # ---- fourier ----
    def compute_fourier_similarity(self, f1, f2) -> float:
        a = self._safe_array(f1)
        b = self._safe_array(f2)
        if a.size == 0 or b.size == 0:
            return 0.0
        m = min(len(a), len(b))
        if m == 0:
            return 0.0
        dist = float(np.linalg.norm(a[:m] - b[:m]))
        return float(1.0 / (1.0 + dist))

    # ---- color / histogram ----
    def compute_color_similarity(self, e1: Dict[str, Any], e2: Dict[str, Any]) -> float:
        # left of e1 should match right of e2 and vice versa
        hist1_l = self._safe_array(e1.get("color_hist_left", None))
        hist1_r = self._safe_array(e1.get("color_hist_right", None))
        hist2_l = self._safe_array(e2.get("color_hist_left", None))
        hist2_r = self._safe_array(e2.get("color_hist_right", None))
        # ensure same dims (fallback to zeros)
        size = max(hist1_l.size, hist1_r.size, hist2_l.size, hist2_r.size, 1)
        if hist1_l.size != size:
            h = np.zeros(size, dtype=np.float32); h[:hist1_l.size] = hist1_l; hist1_l = h
        if hist1_r.size != size:
            h = np.zeros(size, dtype=np.float32); h[:hist1_r.size] = hist1_r; hist1_r = h
        if hist2_l.size != size:
            h = np.zeros(size, dtype=np.float32); h[:hist2_l.size] = hist2_l; hist2_l = h
        if hist2_r.size != size:
            h = np.zeros(size, dtype=np.float32); h[:hist2_r.size] = hist2_r; hist2_r = h

        # chi-square (fall back to L2)
        try:
            chi1 = cv2.compareHist(hist1_l.astype(np.float32), hist2_r.astype(np.float32), cv2.HISTCMP_CHISQR)
            chi2 = cv2.compareHist(hist1_r.astype(np.float32), hist2_l.astype(np.float32), cv2.HISTCMP_CHISQR)
        except Exception:
            chi1 = float(np.sum((hist1_l - hist2_r) ** 2))
            chi2 = float(np.sum((hist1_r - hist2_l) ** 2))

        avg_chi = 0.5 * (chi1 + chi2)
        hist_sim = 1.0 / (1.0 + avg_chi)

        # mean colors
        m1l = self._safe_array(e1.get("mean_left", np.zeros(3)))
        m1r = self._safe_array(e1.get("mean_right", np.zeros(3)))
        m2l = self._safe_array(e2.get("mean_left", np.zeros(3)))
        m2r = self._safe_array(e2.get("mean_right", np.zeros(3)))
        # ensure length 3
        if m1l.size != 3: m1l = np.zeros(3)
        if m1r.size != 3: m1r = np.zeros(3)
        if m2l.size != 3: m2l = np.zeros(3)
        if m2r.size != 3: m2r = np.zeros(3)
        d1 = float(np.linalg.norm(m1l - m2r)) / 255.0
        d2 = float(np.linalg.norm(m1r - m2l)) / 255.0
        col_sim = 1.0 - float(min((d1 + d2) / 2.0, 1.0))

        combined = 0.7 * float(hist_sim) + 0.3 * float(col_sim)
        return float(np.clip(combined, 0.0, 1.0))

    # ---- embedding ----
    def compute_embedding_similarity(self, e1: Dict[str, Any], e2: Dict[str, Any]) -> float:
        v1 = e1.get("embed", None)
        v2 = e2.get("embed", None)
        if v1 is None or v2 is None:
            return 0.0
        v1 = np.asarray(v1, dtype=np.float32).ravel()
        v2 = np.asarray(v2, dtype=np.float32).ravel()
        if v1.size == 0 or v2.size == 0:
            return 0.0
        n1 = np.linalg.norm(v1) + 1e-9
        n2 = np.linalg.norm(v2) + 1e-9
        cos = float(np.dot(v1, v2) / (n1 * n2))
        return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))

    # ---- combined edge similarity ----
    def compute_edge_similarity(self, edge1: Dict[str, Any], edge2: Dict[str, Any]) -> float:
        t1 = edge1.get("type", "unknown")
        t2 = edge2.get("type", "unknown")
        if not self.are_compatible_types(t1, t2):
            return 0.0

        # flat edge: rely mostly on color + optional embed
        if t1 == "flat" or t2 == "flat":
            color_sim = self.compute_color_similarity(edge1, edge2)
            emb_sim = self.compute_embedding_similarity(edge1, edge2)
            combined = 0.85 * color_sim + 0.15 * emb_sim
            # limit flat confidence
            return float(min(combined * 0.85, 0.75))

        invert = (t1 == "tab" and t2 == "blank") or (t1 == "blank" and t2 == "tab")
        shape_sim = self.compute_curvature_similarity(edge1.get("curvature", np.array([])),
                                                      edge2.get("curvature", np.array([])),
                                                      invert=invert)
        fourier_sim = self.compute_fourier_similarity(edge1.get("fourier", np.array([])),
                                                      edge2.get("fourier", np.array([])))
        color_sim = self.compute_color_similarity(edge1, edge2)
        embed_sim = self.compute_embedding_similarity(edge1, edge2)

        total = (self.shape_weight * shape_sim +
                 self.fourier_weight * fourier_sim +
                 self.color_weight * color_sim +
                 self.embed_weight * embed_sim)

        # penalize length mismatch
        len1 = float(edge1.get("length", 0.0))
        len2 = float(edge2.get("length", 0.0))
        if len1 > 0 and len2 > 0:
            ratio = min(len1, len2) / (max(len1, len2) + 1e-9)
            if ratio < 0.80:
                penalty = 0.5 + 0.5 * ratio
                total *= penalty

        return float(np.clip(total, 0.0, 1.0))

    # ---- piece-level matching helpers ----
    def match_pieces(self, piece1_features: Dict[str, Any], piece2_features: Dict[str, Any],
                     threshold: float = 0.20) -> List[Tuple[int, int, float]]:
        matches: List[Tuple[int, int, float]] = []
        edges1 = piece1_features.get("edges", [])
        edges2 = piece2_features.get("edges", [])
        if len(edges1) == 0 or len(edges2) == 0:
            return matches
        for i, e1 in enumerate(edges1):
            for j, e2 in enumerate(edges2):
                s = self.compute_edge_similarity(e1, e2)
                if s >= threshold:
                    matches.append((i, j, float(s)))
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

    def find_best_matches(self,
                          piece_id: str,
                          piece_features: Dict[str, Any],
                          all_pieces: Dict[str, Dict[str, Any]],
                          top_k: int = 15,
                          threshold: float = 0.20,
                          use_adaptive: bool = True) -> Dict[int, List[Tuple[str, int, float]]]:
        """
        For each edge index of piece_id, return a list of candidate (other_piece_id, other_edge_idx, score)
        trimmed to top_k (and optionally adaptive thresholding).
        """
        num_edges = max(1, len(piece_features.get("edges", [])))
        edge_matches: Dict[int, List[Tuple[str, int, float]]] = {i: [] for i in range(num_edges)}

        # brute-force over pieces (can be replaced by ANN candidate proposal later)
        for other_id, other_feats in all_pieces.items():
            if other_id == piece_id:
                continue
            # get matches between the two pieces
            mp = self.match_pieces(piece_features, other_feats, threshold=threshold)
            for e1_idx, e2_idx, score in mp:
                if e1_idx not in edge_matches:
                    edge_matches[e1_idx] = []
                edge_matches[e1_idx].append((other_id, int(e2_idx), float(score)))

        # trim per-edge candidate lists
        for e_idx, lst in list(edge_matches.items()):
            if not lst:
                continue
            lst.sort(key=lambda x: x[2], reverse=True)
            if use_adaptive and len(lst) > top_k:
                top_score = lst[0][2]
                adaptive_threshold = max(threshold, top_score * 0.85)
                # look at a slightly larger window then filter by adaptive threhsold
                window = lst[: max(top_k * 2, top_k)]
                filtered = [m for m in window if m[2] >= adaptive_threshold]
                edge_matches[e_idx] = filtered[:top_k]
            else:
                edge_matches[e_idx] = lst[:top_k]

        return edge_matches


class PieceTypClassifier:
    """Classify pieces as corners, borders, or interior"""

    @staticmethod
    def classify_piece(piece_features: Dict[str, Any]) -> str:
        edges = piece_features.get("edges", [])
        flat_count = sum(1 for edge in edges if edge.get("type") == "flat")
        if flat_count >= 2:
            return "corner"
        elif flat_count == 1:
            return "border"
        else:
            return "interior"

    @staticmethod
    def get_flat_edge_indices(piece_features: Dict[str, Any]) -> List[int]:
        return [i for i, edge in enumerate(piece_features.get("edges", [])) if edge.get("type") == "flat"]