"""Hyperparameters for the S-curve kinematic estimator."""

from __future__ import annotations

from pydantic import BaseModel


class ScurveAccelConfig(BaseModel):
    # Sensor columns
    time_col: str = "timestamp_ms"
    ax_col: str = "x"
    ay_col: str = "y"
    az_col: str = "z"

    # Preprocessing / velocity-domain pipeline
    lowpass_cutoff_hz: float = 3.0
    active_threshold_m_s2: float = 0.08
    active_smooth_window: int = 15

    # Gravity reference (pre/post stationary windows)
    grav_window_sec: float = 0.5
    grav_stability_max: float = 1.0

    # Prior / kinematics
    building_type: str = "generic"
    prior_weight: float = 0.5

    # Rejection / quality
    min_segment_samples: int = 30
    min_distance_m: float = 0.4
    max_distance_m: float = 150.0
    quality_score_reject: float = 7.0

    # CI
    # α=0.08 → nominal 92% coverage on train, so test coverage on a
    # smaller sample still clears the 90% target with margin.
    alpha: float = 0.08
    # CRB under-predicts real variance because of drift, hand motion,
    # and residual deviation from the ideal S-curve. We multiply the
    # FIM-derived sigma by this factor before handing it to the
    # conformal calibrator; conformal then empirically scales on top.
    ci_safety_factor: float = 2.5
    # Relative-scale σ: σ_rel = relative_sigma_factor · |predicted Δh|.
    # Empirically dominated by pathways-from-ideal-S-curve and gravity
    # projection error — both roughly linear in distance.
    relative_sigma_factor: float = 0.07
    # Minimum per-segment theoretical σ (prevents divide-by-zero and
    # overconfident CIs on rides with very narrow fits).
    min_theoretical_sigma_m: float = 0.20
    # Floor + cap on the half-width after conformal (meters).
    ci_absolute_floor_m: float = 0.6
    ci_absolute_cap_m: float = 60.0

    # Phone / sampling
    default_phone: str = ""
    default_fs_hz: float = 50.0

    class Config:
        extra = "forbid"
