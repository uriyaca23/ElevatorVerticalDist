"""Segmentation evaluation metrics.

:class:`SegmentationMetrics` bundles the metrics used across training and
evaluation so call sites don't reimplement them:

- IoU matching between predicted and GT segments (with detection F1 @ IoU >= t)
- Expected Calibration Error (Guo et al. 2017)
- Brier score (Brier 1950)
- Empirical coverage of probability-CI and edge-time-CI intervals
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def ci_center(ci) -> float:
    """Midpoint of a ``(low, high)`` tuple."""
    lo, hi = ci
    return 0.5 * (float(lo) + float(hi))


def _intervals(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Extract (start_center, end_center) per row from CI-based schema."""
    if len(df) == 0:
        return []
    return [
        (ci_center(s), ci_center(e))
        for s, e in zip(df["start_ci"].to_list(), df["end_ci"].to_list())
    ]


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


@dataclass
class DetectionResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ious: list[float] = field(default_factory=list)
    matched_pred_idx: list[int] = field(default_factory=list)
    matched_gt_idx: list[int] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


class SegmentationMetrics:
    """Composite metrics over a set of predicted and ground-truth segments.

    Each method is a pure function on arrays/DataFrames; the class exists
    mainly to group them and to compute a single ``summary`` dict.
    """

    # ---- detection (IoU) ----

    @staticmethod
    def match_segments(
        pred: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float = 0.5,
    ) -> DetectionResult:
        """Greedy one-to-one IoU matching. Each GT matches at most one pred."""
        p_iv = _intervals(pred)
        g_iv = _intervals(gt)
        used_gt: set[int] = set()
        res = DetectionResult(fn=len(g_iv))
        pairs: list[tuple[float, int, int]] = []
        for i, p in enumerate(p_iv):
            for j, g in enumerate(g_iv):
                v = iou(p, g)
                if v > 0:
                    pairs.append((v, i, j))
        pairs.sort(reverse=True)
        used_pred: set[int] = set()
        for v, i, j in pairs:
            if i in used_pred or j in used_gt:
                continue
            if v >= iou_threshold:
                res.tp += 1
                res.ious.append(v)
                res.matched_pred_idx.append(i)
                res.matched_gt_idx.append(j)
                used_pred.add(i)
                used_gt.add(j)
        res.fp = len(p_iv) - len(used_pred)
        res.fn = len(g_iv) - len(used_gt)
        return res

    # ---- calibration ----

    @staticmethod
    def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if probs.size == 0:
            return 0.0
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        total = 0.0
        n = probs.size
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (probs >= lo) & ((probs < hi) if i < n_bins - 1 else (probs <= hi))
            if mask.sum() == 0:
                continue
            acc = labels[mask].mean()
            conf = probs[mask].mean()
            total += (mask.sum() / n) * abs(acc - conf)
        return float(total)

    @staticmethod
    def brier(probs: np.ndarray, labels: np.ndarray) -> float:
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if probs.size == 0:
            return 0.0
        return float(np.mean((probs - labels) ** 2))

    @staticmethod
    def reliability_bins(
        probs: np.ndarray, labels: np.ndarray, n_bins: int = 10,
    ) -> list[dict]:
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        out = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (probs >= lo) & ((probs < hi) if i < n_bins - 1 else (probs <= hi))
            out.append({
                "bin_low": float(lo), "bin_high": float(hi),
                "count": int(mask.sum()),
                "mean_conf": float(probs[mask].mean()) if mask.any() else float("nan"),
                "empirical_acc": float(labels[mask].mean()) if mask.any() else float("nan"),
            })
        return out

    # ---- interval coverage ----

    @staticmethod
    def prob_ci_coverage(
        labels: np.ndarray, p_lo: np.ndarray, p_hi: np.ndarray,
    ) -> float:
        labels = np.asarray(labels, dtype=float)
        p_lo = np.asarray(p_lo, dtype=float)
        p_hi = np.asarray(p_hi, dtype=float)
        if labels.size == 0:
            return float("nan")
        return float(((labels >= p_lo) & (labels <= p_hi)).mean())

    @staticmethod
    def time_ci_coverage(
        residuals_sec: np.ndarray, half_width_sec: float,
    ) -> float:
        r = np.abs(np.asarray(residuals_sec, dtype=float))
        if r.size == 0:
            return float("nan")
        return float((r <= half_width_sec).mean())

    # ---- summary ----

    @classmethod
    def summary(
        cls,
        pred: pd.DataFrame,
        gt: pd.DataFrame,
        iou_threshold: float = 0.5,
        probs: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        p_lo: np.ndarray | None = None,
        p_hi: np.ndarray | None = None,
        start_residuals_sec: np.ndarray | None = None,
        end_residuals_sec: np.ndarray | None = None,
        start_q_sec: float | None = None,
        end_q_sec: float | None = None,
    ) -> dict:
        det = cls.match_segments(pred, gt, iou_threshold=iou_threshold)
        out: dict = {
            "detection": {
                "tp": det.tp, "fp": det.fp, "fn": det.fn,
                "precision": det.precision, "recall": det.recall, "f1": det.f1,
                "mean_iou": float(np.mean(det.ious)) if det.ious else 0.0,
                "iou_threshold": iou_threshold,
            },
        }
        if probs is not None and labels is not None:
            out["calibration"] = {
                "ece": cls.ece(probs, labels),
                "brier": cls.brier(probs, labels),
                "n": int(np.asarray(probs).size),
            }
            if p_lo is not None and p_hi is not None:
                out["calibration"]["prob_ci_coverage"] = cls.prob_ci_coverage(labels, p_lo, p_hi)
        if start_residuals_sec is not None and start_q_sec is not None:
            out["edge_start_ci_coverage"] = cls.time_ci_coverage(start_residuals_sec, start_q_sec)
        if end_residuals_sec is not None and end_q_sec is not None:
            out["edge_end_ci_coverage"] = cls.time_ci_coverage(end_residuals_sec, end_q_sec)
        return out
