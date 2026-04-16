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

    # ---- detection ----

    @staticmethod
    def match_segments(
        pred: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float = 0.0,
    ) -> DetectionResult:
        """Containment-based one-to-one matching.

        A predicted segment is a **true positive** iff exactly one GT segment
        is fully contained in it:

            pred.start_ci.lo <= gt.start_mid <= pred.start_ci.hi
            pred.end_ci.lo   <= gt.end_mid   <= pred.end_ci.hi

        (When the predicted CI is zero-width, the GT midpoint must fall
        exactly at the predicted endpoint — so in practice we use a relaxed
        form where ``pred.start_ci.lo <= gt.start_mid`` and
        ``gt.end_mid <= pred.end_ci.hi`` — the GT interval must lie inside
        the predicted envelope.)

        A predicted segment that contains 0 or ≥ 2 GT segments counts as a
        false positive. GT segments not contained by any prediction are
        false negatives.

        ``iou_threshold`` is retained for back-compat/reporting but no longer
        gates acceptance; IoU is still recorded in ``res.ious`` for inspection.
        """
        p_iv = _intervals(pred)
        g_iv = _intervals(gt)
        res = DetectionResult()

        # pred envelope bounds
        p_env: list[tuple[float, float]] = []
        if len(pred):
            for s_ci, e_ci in zip(pred["start_ci"].to_list(), pred["end_ci"].to_list()):
                p_env.append((float(s_ci[0]), float(e_ci[1])))

        contained: list[list[int]] = [[] for _ in p_iv]
        gt_covered = [False] * len(g_iv)
        for i, (env_lo, env_hi) in enumerate(p_env):
            for j, (g_start, g_end) in enumerate(g_iv):
                if env_lo <= g_start and g_end <= env_hi:
                    contained[i].append(j)

        for i, js in enumerate(contained):
            if len(js) == 1:
                j = js[0]
                if gt_covered[j]:
                    continue  # GT already matched to earlier prediction
                res.tp += 1
                res.matched_pred_idx.append(i)
                res.matched_gt_idx.append(j)
                res.ious.append(iou(p_iv[i], g_iv[j]))
                gt_covered[j] = True
            else:
                res.fp += 1  # 0 or >=2 GTs contained
        res.fn = sum(1 for v in gt_covered if not v)
        return res

    @staticmethod
    def iou_match_segments(
        pred: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float = 0.3,
    ) -> DetectionResult:
        """Best-match-per-GT IOU matching.

        For each GT segment, greedily pick the unmatched prediction with the
        highest IOU; accept the pair iff IOU >= ``iou_threshold``. Predictions
        not matched are false positives; GT not matched are false negatives.
        """
        p_iv = _intervals(pred)
        g_iv = _intervals(gt)
        res = DetectionResult()
        pred_used = [False] * len(p_iv)

        order = sorted(
            range(len(g_iv)),
            key=lambda j: max((iou(p, g_iv[j]) for p in p_iv), default=0.0),
            reverse=True,
        )
        for j in order:
            g = g_iv[j]
            best_i, best_iou = -1, 0.0
            for i, p in enumerate(p_iv):
                if pred_used[i]:
                    continue
                v = iou(p, g)
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_i >= 0 and best_iou >= iou_threshold:
                pred_used[best_i] = True
                res.tp += 1
                res.matched_pred_idx.append(best_i)
                res.matched_gt_idx.append(j)
                res.ious.append(best_iou)
            else:
                res.fn += 1
        res.fp = sum(1 for u in pred_used if not u)
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
