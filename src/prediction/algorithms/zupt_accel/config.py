"""Hyperparameters for the ZUPT accelerometer-only estimator."""

from __future__ import annotations

from pydantic import BaseModel


class ZuptAccelConfig(BaseModel):
    # Sensor columns (in the DataFrame the caller hands us)
    time_col: str = "timestamp_ms"
    ax_col: str = "x"
    ay_col: str = "y"
    az_col: str = "z"

    # Motion-window detection
    lowpass_cutoff_hz: float = 3.0
    active_threshold_m_s2: float = 0.10
    active_smooth_window: int = 15
    active_margin_sec: float = 0.5

    # Gravity projection (pre/post stationary windows)
    grav_window_sec: float = 0.5
    grav_stability_max: float = 1.0          # accept pre/post gravity if std < this
    grav_pre_post_angle_deg: float = 25.0    # reject if frame rotates more than this

    # Quality filter thresholds
    min_segment_samples: int = 30
    min_displacement_m: float = 0.4
    max_distance_m: float = 100.0
    max_ride_drift_deg: float = 15.0
    max_peak_m_s2: float = 8.0
    min_active_fraction: float = 0.04
    quality_score_reject: float = 6.0

    # Conformal / CI
    # We calibrate at α=0.08 (→ nominal 92% coverage) rather than
    # α=0.10 so the blind-test coverage (on a smaller sample) has a
    # small buffer above the 90% target. Train-time coverage of
    # ~92% on the clean pool therefore implies ~90% test coverage
    # with high confidence.
    alpha: float = 0.08
    min_theoretical_sigma_m: float = 0.15    # σ floor: ≈0.15 m is
    # the noise a calibrated Pixel+hand can plausibly beat on a 1-floor
    # ride (white-noise term alone).  Below that we're overconfident and
    # conformal just swallows the slack with a huge multiplier.
    # Absolute CI floor — short elevator rides can't realistically be
    # bounded tighter than ~0.5 m given our sensor grade.
    ci_absolute_floor_m: float = 0.5
    # Soft CI cap as a safety valve. Raised well above a building's
    # full height (20 floors ≈ 60 m) so only truly pathological σ are
    # clipped; the typical CI stays tight because σ varies per segment.
    ci_absolute_cap_m: float = 60.0

    # Phone / sampling defaults (used if context doesn't provide them)
    default_phone: str = ""
    default_fs_hz: float = 50.0

    class Config:
        extra = "forbid"
