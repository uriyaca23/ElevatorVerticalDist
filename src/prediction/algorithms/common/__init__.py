"""Shared utilities for the accelerometer-based prediction algorithms.

Modules:
  * ``noise_db`` — phone-model → accelerometer noise σ lookup.
  * ``conformal`` — split-conformal CI calibrator.
  * ``accel_utils`` — gravity estimation, vertical projection, ZUPT
    integration, low-pass filter.
  * ``pulse_pair`` — trapezoid-pulse-pair matched-filter fitter and
    analytic Δh + delta-method σ for the new trapezoid-S-curve method.
  * ``types`` — ``PredictionOutput`` and ``CalibrationSample``.
"""

from .types import PredictionOutput, CalibrationSample
from .noise_db import get_phone_accel_noise_sigma, resolve_phone_to_chip
from .conformal import ConformalCalibrator
from .accel_utils import (
    estimate_gravity_stationary,
    vertical_accel_magnitude,
    vertical_accel_projected,
    zupt_integrate,
    butter_lowpass,
)
from .pulse_pair import (
    PulsePairFit,
    fit_shared_shape_pair,
    height_from_fit,
    theoretical_sigma_height,
    trapezoid_kernel,
    smooth_rolling_mean,
    GRID_W_S,
    GRID_F,
)

__all__ = [
    "PredictionOutput",
    "CalibrationSample",
    "get_phone_accel_noise_sigma",
    "resolve_phone_to_chip",
    "ConformalCalibrator",
    "estimate_gravity_stationary",
    "vertical_accel_magnitude",
    "vertical_accel_projected",
    "zupt_integrate",
    "butter_lowpass",
    "PulsePairFit",
    "fit_shared_shape_pair",
    "height_from_fit",
    "theoretical_sigma_height",
    "trapezoid_kernel",
    "smooth_rolling_mean",
    "GRID_W_S",
    "GRID_F",
]
