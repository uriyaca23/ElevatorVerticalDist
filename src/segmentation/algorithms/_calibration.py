"""Probability calibration + edge-time conformal quantile.

- :class:`IVAP` implements Inductive Venn-Abers (Vovk & Petej 2012; Vovk,
  Petej & Fedorova 2014): two isotonic regressions bound the calibrated
  probability from above and below, giving a distribution-free interval
  ``[p_lo, p_hi]`` with finite-sample validity.

- :class:`EdgeTimeConformal` holds the 1-alpha empirical quantile of the
  absolute residuals ``|t_pred - t_true|`` on a held-out calibration set
  (split conformal; Vovk et al. 2005). Use it to emit symmetric CIs on
  predicted segment start/end times.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IVAP:
    """Inductive Venn-Abers calibrator.

    ``fit(scores, labels)`` stores the calibration pairs.
    ``predict(score)`` returns ``(p, p_lo, p_hi)``.
    """

    def __init__(self, scores: np.ndarray | None = None, labels: np.ndarray | None = None):
        self.scores: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        if scores is not None and labels is not None:
            self.fit(scores, labels)

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IVAP":
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        if s.shape != y.shape:
            raise ValueError("scores and labels shape mismatch")
        self.scores = s
        self.labels = y
        return self

    def _predict_one(self, score: float) -> tuple[float, float, float]:
        assert self.scores is not None and self.labels is not None
        s_ext = np.concatenate([self.scores, [score]])
        # p0: augment with label 0; p1: augment with label 1
        p_pair = []
        for y_hyp in (0.0, 1.0):
            y_ext = np.concatenate([self.labels, [y_hyp]])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(s_ext, y_ext)
            p_pair.append(float(iso.predict([score])[0]))
        p0, p1 = p_pair
        p_lo = min(p0, p1)
        p_hi = max(p0, p1)
        # Standard IVAP point prediction: p1 / (1 - p0 + p1)
        denom = (1.0 - p0) + p1
        p = p1 / denom if denom > 0 else 0.5 * (p_lo + p_hi)
        p = float(np.clip(p, p_lo, p_hi))
        return p, p_lo, p_hi

    def predict(self, score) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.isscalar(score):
            return self._predict_one(float(score))
        arr = np.asarray(score, dtype=float)
        p = np.empty_like(arr)
        lo = np.empty_like(arr)
        hi = np.empty_like(arr)
        for i, s in enumerate(arr.ravel()):
            p.ravel()[i], lo.ravel()[i], hi.ravel()[i] = self._predict_one(float(s))
        return p, lo, hi

    def save(self, path: Path | str) -> None:
        assert self.scores is not None and self.labels is not None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"scores": self.scores.tolist(), "labels": self.labels.tolist()}
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: Path | str) -> "IVAP":
        with open(path, "r") as f:
            payload = json.load(f)
        return cls(np.asarray(payload["scores"]), np.asarray(payload["labels"]))


class EdgeTimeConformal:
    """Split-conformal quantile on absolute residuals of a scalar prediction."""

    def __init__(self, quantile_sec: float | None = None, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile_sec = quantile_sec

    def fit(self, residuals_sec: np.ndarray, alpha: float | None = None) -> "EdgeTimeConformal":
        if alpha is not None:
            self.alpha = alpha
        r = np.abs(np.asarray(residuals_sec, dtype=float))
        n = len(r)
        if n == 0:
            self.quantile_sec = 0.0
            return self
        # Finite-sample corrected quantile index (Vovk et al. 2005).
        k = int(np.ceil((n + 1) * (1.0 - self.alpha))) - 1
        k = min(max(k, 0), n - 1)
        self.quantile_sec = float(np.sort(r)[k])
        return self

    def quantile(self) -> float:
        if self.quantile_sec is None:
            raise RuntimeError("EdgeTimeConformal not fit")
        return self.quantile_sec


def save_edge_conformal(start: EdgeTimeConformal, end: EdgeTimeConformal, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "alpha": start.alpha,
            "start_q_sec": start.quantile(),
            "end_q_sec": end.quantile(),
        }, f, indent=2)


def load_edge_conformal(path: Path | str) -> tuple[float, float, float]:
    with open(path, "r") as f:
        d = json.load(f)
    return float(d["alpha"]), float(d["start_q_sec"]), float(d["end_q_sec"])
