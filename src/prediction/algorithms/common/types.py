"""Shared dataclasses for prediction outputs and calibration samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionOutput:
    """Full result of a single-segment height-difference prediction.

    Fields:
        height_diff:        signed predicted Δh in meters (up=+, down=-).
        ci_half_width:      90% CI half-width (m). ``math.inf`` when
                            rejected or theoretical σ is unavailable.
        theoretical_sigma:  per-segment theoretical Gaussian σ (m) of
                            the distance-error, before conformal scaling.
        accepted:           True iff the quality filter considers this
                            estimate trustworthy. When False, downstream
                            inference code should either skip the segment
                            or defer to a fallback.
        quality_score:      0 = excellent, higher = worse.
        reject_reason:      empty when accepted.
        meta:               algorithm-specific extras (fitted parameters,
                            residuals, profile type, etc).
    """
    height_diff: float
    ci_half_width: float
    theoretical_sigma: float
    accepted: bool
    quality_score: float
    reject_reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationSample:
    """One (prediction, truth) pair used to fit the conformal multiplier."""
    predicted_dh: float
    true_dh: float
    theoretical_sigma: float
    accepted: bool
    quality_score: float
    signal_clear: bool
    exp_name: str = ""
    segment_idx: int = -1

    @property
    def abs_error(self) -> float:
        return abs(self.predicted_dh - self.true_dh)
