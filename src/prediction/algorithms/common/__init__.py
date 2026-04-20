"""Shared utilities for the accelerometer-based prediction algorithms.

The two algorithms (ZUPT and 7-step S-curve) both need:
  * a sensor-noise lookup table indexed by phone model (``noise_db``),
  * a conformal-prediction calibrator (``conformal``),
  * lightweight accelerometer helpers: gravity estimation, vertical
    projection, ZUPT integration (``accel_utils``),
  * a shared ``PredictionOutput`` dataclass (``types``).
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
]
