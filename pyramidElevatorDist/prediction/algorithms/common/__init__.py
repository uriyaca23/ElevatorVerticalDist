"""Prediction-specific shared types.

General-purpose utilities (accelerometer preprocessing, conformal
calibration, sensor-noise DB) that used to live here have been moved to
``src/utils/`` since they aren't prediction-specific. Only
:class:`PredictionOutput` and :class:`CalibrationSample` remain because
their field schema is tied to this stage's API contract.
"""

from .types import PredictionOutput, CalibrationSample

__all__ = ["PredictionOutput", "CalibrationSample"]
