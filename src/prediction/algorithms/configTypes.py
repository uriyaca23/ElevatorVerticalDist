"""Pydantic config models for the prediction algorithm dispatcher.

Three algorithms are registered:
  * ``BAROMETER_HEIGHT_DIFF`` — ISA-inversion Δh from pressure.
  * ``ZUPT_ACCEL`` — Zero-Velocity Update double integration.
  * ``TRAPEZOID_ACCEL`` — shared-shape trapezoid pulse-pair matched-filter
    fit in the acceleration domain. Replaces the old 7-step S-curve
    velocity-domain NLS fitter — the new method is physically
    equivalent (the trapezoid is the jerk-limited acceleration profile
    with phases 1-3 collapsed to a single symmetric pulse), simpler
    (2 shape params + closed-form amplitude), and more accurate on
    our dataset.

Each algorithm has its own Pydantic config subclass. The top-level
``PREDICT_ALGORITHM_CONFIG`` selects one of them and (optionally)
overrides fields from ``config.json``.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


class PredictAlgorithm(str, Enum):
    BAROMETER_HEIGHT_DIFF = "barometer_height_diff"
    ZUPT_ACCEL = "zupt_accel"
    TRAPEZOID_ACCEL = "trapezoid_accel"


class BarometerHeightDiffConfig(BaseModel):
    time_col: str = "timestamp_ms"
    pressure_col: str = "pressure"
    p0_hpa: float = 1013.25
    edge_avg_samples: int = 1


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
    grav_stability_max: float = 1.0
    grav_pre_post_angle_deg: float = 25.0

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
    # small buffer above the 90% target.
    alpha: float = 0.08
    min_theoretical_sigma_m: float = 0.15
    ci_absolute_floor_m: float = 0.5
    ci_absolute_cap_m: float = 60.0

    # Phone / sampling defaults (used if context doesn't provide them)
    default_phone: str = ""
    default_fs_hz: float = 50.0

    class Config:
        extra = "forbid"


class TrapezoidAccelConfig(BaseModel):
    # Sensor columns
    time_col: str = "timestamp_ms"
    ax_col: str = "x"
    ay_col: str = "y"
    az_col: str = "z"

    # Preprocessing
    smooth_sec: float = 0.4
    detrend_sec: float = 0.0

    # Duration-adaptive W floor
    W_floor_min_sec: float = 0.30
    W_floor_alpha: float = 0.04

    # Velocity-anchored amplitude correction
    velocity_anchor_A: bool = True
    velocity_anchor_min_cruise_sec: float = 0.5

    # Gravity reference (pre/post stationary windows)
    grav_window_sec: float = 0.5
    grav_stability_max: float = 1.0

    # Quality filter — core thresholds
    min_segment_samples: int = 30
    min_distance_m: float = 0.4
    max_distance_m: float = 120.0
    min_active_fraction: float = 0.05
    quality_score_reject: float = 6.0

    # Quality filter — graded minimum joint R² (bin-dependent).
    # A short ride has inherently lower SNR, so we allow a lower R²
    # threshold there; long rides must fit well.
    min_r2_short: float = 0.35   # applied when predicted |Δh| < 3 m
    min_r2_mid: float = 0.50     # applied for 3 ≤ |Δh| < 6 m
    min_r2_long: float = 0.60    # applied for |Δh| ≥ 6 m

    # Lobe-overlap inflation factor applied inside the σ calculator for
    # genuinely unphysical Δt_c < 2W configurations. In the physically
    # valid touching regime (Δt_c = 2W) the off-diagonal Fisher block
    # is zero and no inflation is applied — the joined-pulse fit
    # handles that case directly (see fields below).
    overlap_delta: float = 0.05
    # Absolute hard floor on Δt_c. We no longer apply a W-relative
    # reject rule in the quality filter; the estimator's joined-pulse
    # branch handles the overlap regime.
    min_delta_tc_sec: float = 0.2

    # --- Joined-pulse hybrid ---
    # Prefer the joined-pulse fit (Δt_c = 2W constrained) over the
    # unconstrained pair fit whenever either:
    #   1. joined_r2 > pair_r2 + joined_r2_advantage, OR
    #   2. pair Δt_c / 2W < pair_overlap_handoff AND joined R² is
    #      within joined_r2_advantage of the pair R².
    # These thresholds express Occam's razor (a tie in fit quality
    # favours the model with fewer free parameters) and respect the
    # physics of short rides (overlap regime naturally fits joined).
    joined_r2_advantage: float = 0.02
    pair_overlap_handoff: float = 1.15
    # ZUPT-fallback threshold. When both the pair and joined fits
    # have joint R² below this value the matched-filter framework is
    # judged unreliable and the estimator returns an inline ZUPT
    # displacement flagged as ``zupt_fallback_both_fits_failed``.
    zupt_fallback_r2: float = 0.20

    # Quality filter — new outlier catchers (pull out Yitzchaki-style
    # 13-22 m MAE rides before they pollute the CI coverage pool).
    # - A_anchor_ratio: A_used / A_fit; should be close to 1.
    # - out_of_lobe_residual_frac: ratio of residual-energy *density*
    #   outside the two lobe support windows to the density inside.
    #   A healthy fit has uniform residuals (ratio ≈ 1); a
    #   mid-ride disturbance pushes the ratio >> 1.
    # - cruise_v_cv: coefficient of variation of ZUPT-integrated
    #   velocity across the cruise window; > 0.6 means the cabin is
    #   not really at steady cruise.
    # - pre_post_angle_deg: promoted from score-only to hard reject.
    max_A_anchor_ratio: float = 2.0
    min_A_anchor_ratio: float = 0.5
    # Density-ratio threshold: >4x means outside-lobe residuals are
    # dramatically noisier than inside-lobe residuals — a clear
    # sign of a mid-ride disturbance.
    max_out_of_lobe_frac: float = 4.0
    max_cruise_v_cv: float = 0.6
    max_pre_post_angle_deg: float = 25.0

    # Theoretical-σ knobs.
    # The new σ is a pure delta-method propagation on
    # Δh = s·A·W(1+f)·Δt_c with two physics-grounded modifiers:
    #   1. effective sensor noise is scaled by 1/max(R², r2_epsilon) —
    #      the Wald post-regression form that widens the CI for
    #      poor-fit segments in every parameter simultaneously;
    #   2. σ_Δt_c² is inflated by the off-diagonal-Fisher factor
    #      (2W / (Δt_c - 2W))² in the soft overlap regime
    #      (Δt_c < 4W), to capture the loss of identifiability as
    #      the lobes approach each other.
    # The old ``drift_scale`` and ``relative_sigma_factor`` knobs have
    # been removed — the drift term was inherited from ZUPT (cumulative
    # double-integration drift, which doesn't apply to a matched-filter
    # closed-form), and the relative term double-counted the |Δh|
    # scaling already present through ∂Δh/∂A · Δt_c.
    r2_epsilon: float = 1e-3
    min_theoretical_sigma_m: float = 0.15

    # CI floor + cap (absolute meters, applied after conformal multiplier)
    ci_absolute_floor_m: float = 0.5
    ci_absolute_cap_m: float = 40.0

    # Conformal
    alpha: float = 0.08

    # Phone / sampling defaults
    default_phone: str = ""
    default_fs_hz: float = 50.0

    class Config:
        extra = "forbid"


class PREDICT_ALGORITHM_CONFIG(BaseModel):
    algorithm: PredictAlgorithm = PredictAlgorithm.BAROMETER_HEIGHT_DIFF
    config_path: Path = DEFAULT_CONFIG_PATH
    overrides: dict[str, Any] = Field(default_factory=dict)

    def load_params(self) -> dict[str, Any]:
        with open(self.config_path, "r") as f:
            all_params = json.load(f) or {}
        params = dict(all_params.get(self.algorithm.value, {}))
        params.update(self.overrides)
        return params
