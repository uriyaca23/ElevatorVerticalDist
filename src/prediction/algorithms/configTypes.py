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

    # Quality filter
    min_segment_samples: int = 30
    min_distance_m: float = 0.4
    max_distance_m: float = 120.0
    min_joint_r2: float = 0.45
    min_delta_tc_sec: float = 0.8
    min_active_fraction: float = 0.05
    quality_score_reject: float = 6.0

    # Per-algorithm theoretical-σ shape
    drift_scale: float = 0.06
    relative_sigma_factor: float = 0.04
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
