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
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class ConformalCalibrator:
    """Multiplicative split-conformal calibrator on ``|err|/σ`` scores.

    After :meth:`fit`, :meth:`half_width(sigma)` returns ``k · σ + margin``
    which, by construction, covers the true Δh with probability ≥ 1−α
    on exchangeable data.
    """

    alpha: float = 0.10                       # miscoverage target (→ 90% CI)
    floor_multiplier: float = 1.645           # minimum k (Z₀.₉₅)
    extra_margin_m: float = 0.0               # constant additive safety margin

    # Fitted fields
    n_calibration: int = 0
    multiplier: float = 1.645
    p95_score: float = 1.645

    def fit(
        self,
        abs_errors: Iterable[float],
        theoretical_sigmas: Iterable[float],
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

        q = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        k_hat = float(np.quantile(scores, q))
        p95 = float(np.quantile(scores, min(0.95, q)))

        self.n_calibration = int(n)
        self.multiplier = max(self.floor_multiplier, k_hat)
        self.p95_score = p95
        return self

    def half_width(self, sigma: float) -> float:
        if not math.isfinite(sigma) or sigma <= 0:
            return math.inf
        return self.multiplier * sigma + self.extra_margin_m

    # ---- checkpoint IO (plain JSON, one file per algorithm) ----
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: Path | str) -> "ConformalCalibrator":
        with open(path, "r") as f:
            d = json.load(f)
        return cls(**d)
