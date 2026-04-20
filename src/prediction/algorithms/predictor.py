"""Prediction algorithm dispatch.

:class:`Predictor` is configured via :class:`PREDICT_ALGORITHM_CONFIG`
(see ``configTypes.py``) which selects one of three algorithms and
loads its hyperparameters from ``config.json``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd

from .barometer_only import predict_height_difference_from_barometer
from .common import CalibrationSample, PredictionOutput
from .accelerometer_only.zupt_accel import ZuptAccelConfig, ZuptAccelEstimator
from .accelerometer_only.trapezoid_accel import (
    TrapezoidAccelConfig, TrapezoidAccelEstimator,
)
from .configTypes import (
    PREDICT_ALGORITHM_CONFIG,
    PredictAlgorithm,
    BarometerHeightDiffConfig,
)


class Predictor:
    def __init__(self, config: PREDICT_ALGORITHM_CONFIG):
        self.config = config
        self.params = config.load_params()
        self._algo_impl = self._build_algo()

    def _build_algo(self):
        algo = self.config.algorithm
        if algo is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
            return BarometerHeightDiffConfig(**self.params)
        if algo is PredictAlgorithm.ZUPT_ACCEL:
            return ZuptAccelEstimator(ZuptAccelConfig(**self.params))
        if algo is PredictAlgorithm.TRAPEZOID_ACCEL:
            return TrapezoidAccelEstimator(TrapezoidAccelConfig(**self.params))
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Dispatch a prediction algorithm on a single elevator ride segment.
    #
    # Input (`data`): a pandas DataFrame of raw sensor samples covering
    # exactly one ride, whose required columns depend on the selected
    # algorithm:
    #   - BAROMETER_HEIGHT_DIFF → column `pressure` (hPa) by default;
    #                             column name configurable via
    #                             `BarometerHeightDiffConfig.pressure_col`.
    #   - ZUPT_ACCEL            → columns `timestamp_ms`, `x`, `y`, `z`
    #                             (raw accelerometer, m/s^2) by default;
    #                             column names configurable on the config.
    #   - TRAPEZOID_ACCEL       → same raw accelerometer schema as ZUPT.
    #
    # The two accelerometer algorithms also accept optional stationary
    # pre/post windows and a `phone_model` string; these feed the gravity-
    # projection and phone-specific noise-DB lookups.
    #
    # Output: a :class:`PredictionOutput` bundling the predicted
    # height-difference (meters), conformal CI half-width, theoretical σ,
    # accept/reject verdict, quality score, and a free-form meta dict.
    # The barometer baseline does not model quality or CI, so it returns
    # a permissive PredictionOutput (accepted=True, ci=inf) so downstream
    # code can treat all algorithms uniformly.
    def predict(
        self,
        data: pd.DataFrame,
        phone_model: str = "",
        pre: Optional[pd.DataFrame] = None,
        post: Optional[pd.DataFrame] = None,
    ) -> PredictionOutput:
        algo = self.config.algorithm
        if algo is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
            dh = predict_height_difference_from_barometer(data, self._algo_impl)
            return PredictionOutput(
                height_diff=float(dh),
                ci_half_width=math.inf,
                theoretical_sigma=math.inf,
                accepted=True, quality_score=0.0, reject_reason="",
                meta={"method": "barometer"},
            )
        return self._algo_impl.predict_segment(
            data, phone_model=phone_model, pre=pre, post=post,
        )

    def calibrate(self, samples: list[CalibrationSample]) -> dict:
        if self.config.algorithm is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
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
