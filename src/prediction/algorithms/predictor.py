"""Prediction algorithm dispatch.

:class:`Predictor` is configured via :class:`PREDICT_ALGORITHM_CONFIG`
(see ``class.py``) which selects one of three algorithms and loads
its hyperparameters from ``config.json``:

  * :class:`~.barometer_only.predict_height_difference_from_barometer`
    — a pure function, float-in-float-out.
  * :class:`~.zupt_accel.ZuptAccelEstimator`
  * :class:`~.trapezoid_accel.TrapezoidAccelEstimator`

Both accelerometer-based algorithms are classes that carry state (the
calibrated conformal multiplier), so :class:`Predictor` instantiates
the correct class at construction time.

Public API:

  * :meth:`Predictor.forward(data)` — returns a ``float`` Δh
    (meters). Backwards-compatible with the barometer baseline.
  * :meth:`Predictor.predict(data, phone_model, pre, post)` — returns
    the full :class:`PredictionOutput` (Δh, CI, accept flag, quality,
    metadata). Available on all algorithms; for the barometer baseline
    we synthesise a permissive PredictionOutput so downstream code can
    treat all three uniformly.
  * :meth:`Predictor.calibrate(samples)` — no-op for the barometer;
    fits the conformal multiplier for ZUPT / trapezoid.
  * :meth:`Predictor.save_calibration(path)` / :meth:`load_calibration(path)`.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Optional

import pandas as pd

from .barometer_only import predict_height_difference_from_barometer
from .common import CalibrationSample, PredictionOutput
from .zupt_accel import ZuptAccelConfig, ZuptAccelEstimator
from .trapezoid_accel import TrapezoidAccelConfig, TrapezoidAccelEstimator

_config_mod = importlib.import_module(__package__ + ".class")
PREDICT_ALGORITHM_CONFIG = _config_mod.PREDICT_ALGORITHM_CONFIG
PredictAlgorithm = _config_mod.PredictAlgorithm
BarometerHeightDiffConfig = _config_mod.BarometerHeightDiffConfig


class Predictor:
    """Unified front-end for all height-difference prediction algorithms."""

    def __init__(self, config: PREDICT_ALGORITHM_CONFIG):
        self.config = config
        self.params = config.load_params()
        self._algo_impl = self._build_algo()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_algo(self):
        algo = self.config.algorithm
        if algo is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
            return BarometerHeightDiffConfig(**self.params)
        if algo is PredictAlgorithm.ZUPT_ACCEL:
            return ZuptAccelEstimator(ZuptAccelConfig(**self.params))
        if algo is PredictAlgorithm.TRAPEZOID_ACCEL:
            return TrapezoidAccelEstimator(TrapezoidAccelConfig(**self.params))
        raise ValueError(f"Unsupported algorithm: {algo}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def forward(self, data: pd.DataFrame, **kwargs) -> float:
        """Point estimate of the height difference (meters)."""
        return self.predict(data, **kwargs).height_diff

    def predict(
        self,
        data: pd.DataFrame,
        phone_model: str = "",
        pre: Optional[pd.DataFrame] = None,
        post: Optional[pd.DataFrame] = None,
    ) -> PredictionOutput:
        algo = self.config.algorithm
        if algo is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
            cfg: BarometerHeightDiffConfig = self._algo_impl
            dh = predict_height_difference_from_barometer(data, cfg)
            # The barometer baseline does not model quality or CI; we
            # surface the estimate with a permissive PredictionOutput so
            # downstream code can treat all three algorithms uniformly.
            return PredictionOutput(
                height_diff=float(dh),
                ci_half_width=math.inf,
                theoretical_sigma=math.inf,
                accepted=True, quality_score=0.0, reject_reason="",
                meta={"method": "barometer"},
            )

        # Accelerometer algorithms share a common per-segment method.
        return self._algo_impl.predict_segment(
            data, phone_model=phone_model, pre=pre, post=post,
        )

    # ------------------------------------------------------------------
    # Calibration + checkpoint IO
    # ------------------------------------------------------------------
    def calibrate(self, samples: list[CalibrationSample]) -> dict:
        algo = self.config.algorithm
        if algo is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
            return {"note": "barometer_does_not_require_calibration"}
        return self._algo_impl.calibrate(samples)

    def save_calibration(self, path: Path | str) -> None:
        if self.config.algorithm is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
            return
        self._algo_impl.save(path)

    def load_calibration(self, path: Path | str) -> None:
        if self.config.algorithm is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
            return
        self._algo_impl.load(path)
