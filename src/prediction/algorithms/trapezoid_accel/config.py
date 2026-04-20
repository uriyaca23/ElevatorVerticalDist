"""Hyperparameters for the trapezoid pulse-pair accelerometer estimator."""

from __future__ import annotations

from pydantic import BaseModel


class TrapezoidAccelConfig(BaseModel):
    # Sensor columns
    time_col: str = "timestamp_ms"
    ax_col: str = "x"
    ay_col: str = "y"
    az_col: str = "z"

    # Preprocessing
    # Rolling-mean smoothing window in seconds. 0.4 s is Eyal's
    # calibrated default — suppresses 50 Hz hand-tremor leakage while
    # preserving jerk-phase ramps of a 1 s lobe at 50 Hz.
    smooth_sec: float = 0.4
    # Rolling detrend window (s). Subtracts a slow mean to absorb residual
    # gravity-projection bias on rides where the phone orientation drifts
    # from the calibration window. Must be substantially longer than
    # any real ride (≥ 90 s) or it eats into the acceleration lobes
    # of long rides. Set to 0.0 to disable detrending entirely —
    # typically what you want when pre+post gravity are both stable.
    detrend_sec: float = 0.0

    # Duration-adaptive W floor. The matched-filter R² is biased toward
    # narrow templates on long rides because a narrow sharp spike can
    # outperform the wide-trapezoid true-lobe fit on local R². We
    # therefore floor the searched W at ``max(W_floor_min,
    # W_floor_alpha · ride_duration)``, reflecting the physical fact
    # that long rides have proportionally longer acceleration phases.
    W_floor_min_sec: float = 0.30
    W_floor_alpha: float = 0.04

    # Velocity-anchored amplitude correction. After the (W, f, t_c1,
    # t_c2) shape fit, rescale the shared amplitude |A| so that the
    # implied cruise velocity ``A · W · (1+f)`` matches the cruise
    # velocity measured directly from the ZUPT-integrated signal
    # between the two lobes. This makes Δh robust to narrow-W bias:
    # the shape fit sets Δt_c (which dominates for long rides), the
    # integrated velocity sets v_peak (which dominates the magnitude).
    # Disable with ``False`` to fall back to the pure matched-filter A.
    velocity_anchor_A: bool = True
    # Minimum cruise-window width (s) required before we trust the
    # v_peak measurement. Triangular / very short rides have no true
    # cruise, in which case we keep the fitted |A|.
    velocity_anchor_min_cruise_sec: float = 0.5

    # Gravity reference (pre/post stationary windows)
    grav_window_sec: float = 0.5
    grav_stability_max: float = 1.0

    # Quality filter
    min_segment_samples: int = 30
    min_distance_m: float = 0.4
    max_distance_m: float = 120.0
    # R² on the shared-shape fit below this → reject as a bad fit.
    # 0.5 is a permissive floor; good clean rides sit at 0.85-0.97.
    min_joint_r2: float = 0.45
    # Lobe-pair spacing floor (s). Δt_c < this means the two lobes
    # are too close to represent a physical ride.
    min_delta_tc_sec: float = 0.8
    # How much of the ride window must be covered by the active lobes.
    # Very short lobes in a long window likely indicate missegmentation.
    min_active_fraction: float = 0.05
    quality_score_reject: float = 6.0

    # Per-algorithm theoretical-σ shape:
    #   σ_total² = σ_white² + (k_drift · ride_drift)² · σ_white²
    #            + (k_rel   · |Δh|)²
    #            + σ_grid²   (already in the delta method)
    #
    # The white-noise term comes directly from the matched-filter CRB
    # and tends to under-predict when the phone drifts mid-ride. The
    # drift + relative terms absorb the structural deviations we see
    # in the field data. Both are fit on the train set.
    drift_scale: float = 0.06          # σ_scale per deg of gravity drift
    relative_sigma_factor: float = 0.04  # σ_rel / |predicted Δh|
    min_theoretical_sigma_m: float = 0.15

    # CI floor + cap (absolute meters, applied after conformal multiplier)
    ci_absolute_floor_m: float = 0.5
    ci_absolute_cap_m: float = 40.0

    # Conformal
    # α = 0.08 gives a 92% nominal coverage, so test-set coverage on a
    # smaller sample still clears the 90% target with a buffer.
    alpha: float = 0.08

    # Phone / sampling defaults
    default_phone: str = ""
    default_fs_hz: float = 50.0

    class Config:
        extra = "forbid"
