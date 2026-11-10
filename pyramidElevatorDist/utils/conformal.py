"""Split-conformal calibration of a theoretical-sigma CI.

Given a set of (prediction, truth, theoretical_sigma) triples, we fit
a scalar multiplier ``k`` such that ``|error| ≤ k · σ_theoretical``
holds for at least (1−α) of the triples. This is the classic
non-conformity-score conformal approach: the score is ``|err|/σ`` and
the calibrated multiplier is the (1−α)(n+1)/n empirical quantile of
those scores.

The algorithm-specific "theoretical" σ can be the ZUPT noise σ, the
Fisher-information CRB, or any other per-segment error-scale estimate
that is monotone with the actual error magnitude. The conformal layer
then corrects the overall scale and (optionally) adds a small additive
margin for robustness at long tails.

Group-conditional (Mondrian) calibration
-----------------------------------------
A single scalar ``k`` is calibrated marginally over the whole pool, so
it certifies *marginal* coverage ``Pr(|err| ≤ kσ) ≥ 1−α`` but not
*conditional* coverage within sub-populations. In practice the
trapezoid CI is over-covered on short/low rides and under-covered on
long/tall rides, because the empirical (1−α) quantile of ``|err|/σ``
differs from one ride-size regime to the next (and is not even
monotone in ride size).

When :meth:`fit` is given a per-sample conditioning ``features`` array
(the predicted ride magnitude ``|Δh|``) plus ``bin_edges``, we run
**Mondrian split conformal**: the calibration pool is partitioned into
bins by that feature and a *separate* conformal multiplier ``k_b`` is
fit inside each bin,

    k_b = max( floor, (1−α)(n_b+1)/n_b empirical quantile of {|err_i|/σ_i : i∈bin b} ).

The reported half-width for a new ride is ``k_{b(|Δh|)} · σ``. By the
standard split-conformal argument applied *within* each bin this gives
≈(1−α) coverage in every bin, and because ``k_b`` is the in-bin
quantile it is the *smallest* multiplier achieving that coverage — the
tightest interval consistent with the per-bin target. Bins with fewer
than ``bin_min_count`` calibration samples fall back to the global
multiplier so a sparse bin cannot produce a wild ``k_b``.

Passing no ``features`` (the default, and what ZUPT does) leaves
``bin_edges``/``bin_multipliers`` empty and recovers the original
single-scalar behaviour exactly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass
class ConformalCalibrator:
    """Multiplicative split-conformal calibrator on ``|err|/σ`` scores.

    After :meth:`fit`, :meth:`half_width(sigma, feature)` returns
    ``k · σ + margin`` where ``k`` is the global multiplier, or the
    per-bin multiplier when the calibrator was fit with conditioning
    features and ``feature`` is supplied.
    """

    alpha: float = 0.10                       # miscoverage target (→ 90% CI)
    floor_multiplier: float = 1.645           # minimum k (Z₀.₉₅)
    extra_margin_m: float = 0.0               # constant additive safety margin
    bin_min_count: int = 20                   # below this, a bin falls back to global k
    finite_sample_correction: bool = True     # (1−α)(n+1)/n inflation. False ⇒ plain
                                              # empirical (1−α) quantile = tightest CI
                                              # that still reaches ≈(1−α) coverage.

    # Fitted fields
    n_calibration: int = 0
    multiplier: float = 1.645                 # global (marginal) multiplier / fallback
    p95_score: float = 1.645
    # Mondrian (group-conditional) fields. Empty ⇒ pure marginal calibration.
    bin_edges: list = field(default_factory=list)         # length B+1 conditioning edges
    bin_multipliers: list = field(default_factory=list)   # length B per-bin multipliers
    bin_counts: list = field(default_factory=list)        # length B calibration counts

    # ---- calibration ----
    def fit(
        self,
        abs_errors: Iterable[float],
        theoretical_sigmas: Iterable[float],
        features: Iterable[float] | None = None,
        bin_edges: Sequence[float] | None = None,
    ) -> "ConformalCalibrator":
        errs = np.asarray(list(abs_errors), dtype=float)
        sigs = np.asarray(list(theoretical_sigmas), dtype=float)
        if errs.size == 0:
            # Leave defaults in place; nothing to learn from.
            self.n_calibration = 0
            return self

        eps = 1e-6
        scores = errs / np.clip(sigs, eps, None)
        n = scores.size

        # Global (marginal) multiplier — always fit; doubles as the
        # fallback for sparse Mondrian bins.
        self.multiplier = self._quantile_multiplier(scores)
        self.p95_score = float(np.quantile(scores, min(0.95, self._q_level(n))))
        self.n_calibration = int(n)

        # Optional Mondrian (per-bin) calibration.
        if features is not None and bin_edges is not None and len(bin_edges) >= 2:
            feats = np.asarray(list(features), dtype=float)
            edges = [float(x) for x in bin_edges]
            self.bin_edges = edges
            self.bin_multipliers = []
            self.bin_counts = []
            idx = self._bin_indices(feats, edges)
            for b in range(len(edges) - 1):
                m = idx == b
                nb = int(m.sum())
                self.bin_counts.append(nb)
                if nb >= self.bin_min_count:
                    self.bin_multipliers.append(self._quantile_multiplier(scores[m]))
                else:
                    # Too few samples to trust an in-bin quantile.
                    self.bin_multipliers.append(self.multiplier)
        else:
            self.bin_edges = []
            self.bin_multipliers = []
            self.bin_counts = []
        return self

    def _q_level(self, n: int) -> float:
        if not self.finite_sample_correction:
            return 1.0 - self.alpha
        return min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)

    def _quantile_multiplier(self, scores: np.ndarray) -> float:
        if scores.size == 0:
            return self.floor_multiplier
        k_hat = float(np.quantile(scores, self._q_level(scores.size)))
        return max(self.floor_multiplier, k_hat)

    @staticmethod
    def _bin_indices(feats: np.ndarray, edges: list) -> np.ndarray:
        """Map feature values to bin index 0..B-1 (clamped at the ends)."""
        interior = np.asarray(edges[1:-1], dtype=float)
        idx = np.digitize(np.abs(feats), interior, right=False)
        return np.clip(idx, 0, len(edges) - 2).astype(int)

    # ---- application ----
    def multiplier_for(self, feature: float | None = None) -> float:
        if (
            not self.bin_edges
            or feature is None
            or not math.isfinite(feature)
        ):
            return self.multiplier
        b = int(self._bin_indices(np.asarray([feature], dtype=float), self.bin_edges)[0])
        return float(self.bin_multipliers[b])

    def half_width(self, sigma: float, feature: float | None = None) -> float:
        if not math.isfinite(sigma) or sigma <= 0:
            return math.inf
        return self.multiplier_for(feature) * sigma + self.extra_margin_m

    # ---- checkpoint IO (plain JSON, one file per algorithm) ----
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: Path | str) -> "ConformalCalibrator":
        with open(path, "r") as f:
            d = json.load(f)
        # Tolerate checkpoints written by older/newer schemas: keep only
        # fields this dataclass declares.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
