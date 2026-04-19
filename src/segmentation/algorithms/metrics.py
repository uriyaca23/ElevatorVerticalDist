"""Metrics for segmentation / detection outputs.

The headline class here is :class:`IntervalPredictionMetrics` — the
evaluator for the trapezoid detector's interval predictions against GT
rides. It distinguishes four failure modes:

* ``missed``       — GT with no overlapping prediction.
* ``fp``           — prediction with no overlapping GT (landed on
                     "outside").
* ``pred_merged``  — prediction whose interval swallows >1 GT rides.
                     Example: GT = (1,2), (2.5,3.5), (4,5) and pred =
                     (1,5) produces ``pred_merged=1`` and
                     ``gt_merged=3`` (all three GTs were merged into one
                     pred).
* ``gt_split``     — GT covered by >1 predictions.

And the happy path:

* ``clean`` — exactly-one-to-exactly-one matches.

Composite score via :meth:`score`:

    2 * clean / (2 * clean + bad_gt + bad_pred)

which is F1-like over "how many GTs and preds ended up in a clean
pairing" — every error type counts equally. 1.0 = perfect, 0.0 = no
clean match.

The module also exports small placeholders (``SegmentationMetrics``,
``DetectionResult``, ``iou``, ``ci_center``) that the package
``__init__.py`` re-exports. They exist so the package import chain
resolves; they're intentionally thin.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Default matching thresholds — two intervals count as "overlapping" if
# either the absolute overlap is ≥ MIN_OVERLAP_S, or the overlap covers
# ≥ MIN_OVERLAP_FRAC of the shorter interval. The fractional rule catches
# short GT rides fully contained in a long merged prediction.
# --------------------------------------------------------------------------
DEFAULT_MIN_OVERLAP_S = 1.0
DEFAULT_MIN_OVERLAP_FRAC = 0.3


def _overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _intervals_match(
    g0: float, g1: float, p0: float, p1: float,
    min_overlap_s: float, min_overlap_frac: float,
) -> bool:
    ov = _overlap_s(g0, g1, p0, p1)
    if ov <= 0.0:
        return False
    shortest = max(1e-3, min(g1 - g0, p1 - p0))
    return ov >= min_overlap_s or ov / shortest >= min_overlap_frac


@dataclass
class IntervalPredictionMetrics:
    """Per-experiment match counts between GT rides and predictions.

    Build via :meth:`from_intervals`. Sum across experiments with ``+``.
    Derived rates via :meth:`rates`; composite via :meth:`score`.
    """

    n_gt: int = 0
    n_pred: int = 0
    clean: int = 0
    missed: int = 0
    gt_merged: int = 0
    gt_split: int = 0
    fp: int = 0
    pred_merged: int = 0
    pred_split_part: int = 0

    # ---- construction ----

    @classmethod
    def from_intervals(
        cls,
        gt_rides: list[dict],
        predictions: list[dict],
        min_overlap_s: float = DEFAULT_MIN_OVERLAP_S,
        min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
    ) -> "IntervalPredictionMetrics":
        """``gt_rides`` and ``predictions`` are lists of dicts with at
        least ``t_start_s`` and ``t_end_s``."""
        n_g = len(gt_rides)
        n_p = len(predictions)
        gt_to_preds: list[list[int]] = [[] for _ in range(n_g)]
        pred_to_gts: list[list[int]] = [[] for _ in range(n_p)]
        for i, g in enumerate(gt_rides):
            g0, g1 = g["t_start_s"], g["t_end_s"]
            for j, p in enumerate(predictions):
                if _intervals_match(
                    g0, g1, p["t_start_s"], p["t_end_s"],
                    min_overlap_s, min_overlap_frac,
                ):
                    gt_to_preds[i].append(j)
                    pred_to_gts[j].append(i)

        m = cls(n_gt=n_g, n_pred=n_p)
        for i, ps in enumerate(gt_to_preds):
            if len(ps) == 0:
                m.missed += 1
            elif len(ps) == 1:
                if len(pred_to_gts[ps[0]]) == 1:
                    m.clean += 1
                else:
                    m.gt_merged += 1
            else:
                m.gt_split += 1
        for j, gs in enumerate(pred_to_gts):
            if len(gs) == 0:
                m.fp += 1
            elif len(gs) == 1:
                if len(gt_to_preds[gs[0]]) != 1:
                    m.pred_split_part += 1
            else:
                m.pred_merged += 1
        return m

    # ---- aggregation ----

    def __add__(self, other: "IntervalPredictionMetrics") -> "IntervalPredictionMetrics":
        return IntervalPredictionMetrics(
            n_gt=self.n_gt + other.n_gt,
            n_pred=self.n_pred + other.n_pred,
            clean=self.clean + other.clean,
            missed=self.missed + other.missed,
            gt_merged=self.gt_merged + other.gt_merged,
            gt_split=self.gt_split + other.gt_split,
            fp=self.fp + other.fp,
            pred_merged=self.pred_merged + other.pred_merged,
            pred_split_part=self.pred_split_part + other.pred_split_part,
        )

    @classmethod
    def sum(cls, items) -> "IntervalPredictionMetrics":
        out = cls()
        for m in items:
            out = out + m
        return out

    # ---- scoring ----

    def score(self) -> float:
        bad_gt = self.missed + self.gt_merged + self.gt_split
        bad_pred = self.fp + self.pred_merged + self.pred_split_part
        denom = 2 * self.clean + bad_gt + bad_pred
        return 0.0 if denom == 0 else (2.0 * self.clean) / denom

    def rates(self) -> dict[str, float]:
        return {
            "f1_like": self.score(),
            "recall":     self.clean / self.n_gt   if self.n_gt   else 0.0,
            "precision":  self.clean / self.n_pred if self.n_pred else 0.0,
            "miss_rate":  self.missed / self.n_gt  if self.n_gt   else 0.0,
            "merge_rate": self.pred_merged / self.n_pred if self.n_pred else 0.0,
            "fp_rate":    self.fp / self.n_pred    if self.n_pred else 0.0,
        }

    def as_dict(self) -> dict:
        """Flat dict of counts + derived rates — ready for a DataFrame row."""
        out = {
            "n_gt": self.n_gt, "n_pred": self.n_pred,
            "clean": self.clean, "missed": self.missed,
            "gt_merged": self.gt_merged, "gt_split": self.gt_split,
            "fp": self.fp, "pred_merged": self.pred_merged,
            "pred_split_part": self.pred_split_part,
        }
        out.update(self.rates())
        return out


# --------------------------------------------------------------------------
# Minimal placeholders re-exported by ``algorithms/__init__.py``. They
# exist so the import chain resolves and aren't meant to be used directly
# — replace with real implementations when the segmenter pipeline is
# revived.
# --------------------------------------------------------------------------
@dataclass
class SegmentationMetrics:
    """Placeholder."""
    values: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Placeholder."""
    values: dict = field(default_factory=dict)


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    a0, a1 = a
    b0, b1 = b
    ov = _overlap_s(a0, a1, b0, b1)
    un = max(a1, b1) - min(a0, b0)
    return 0.0 if un <= 0 else ov / un


def ci_center(ci: tuple[float, float]) -> float:
    return 0.5 * (ci[0] + ci[1])
