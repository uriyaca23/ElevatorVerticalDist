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

from pyramidElevatorDist.physics.reconstruct_az import reconstruct_az

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


def _resample_to(frame: Optional[pd.DataFrame], hz: int) -> Optional[pd.DataFrame]:
    """Normalize a timestamped window onto a uniform ``hz`` grid (gap-aware,
    time-correct). Returns the input unchanged when it is unusable
    (``None`` / <2 rows / no ``timestamp_ms``)."""
    if (frame is None or "timestamp_ms" not in getattr(frame, "columns", [])
            or len(frame) < 2):
        return frame
    from pyramidElevatorDist.utils.resampling import resample_sensor_with_gaps
    resampled, _intervals = resample_sensor_with_gaps(frame, target_hz=hz)
    return resampled


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
        trapezoid_override: Optional[dict] = None,
        gyro: Optional[pd.DataFrame] = None,
        pre_gyro: Optional[pd.DataFrame] = None,
        post_gyro: Optional[pd.DataFrame] = None,
    ) -> PredictionOutput:
        # `gyro` / `pre_gyro` / `post_gyro` are the matching gyroscope slices
        # for `data` / `pre` / `post`. When `config.reconstruct != "none"` and
        # gyro is present, the accelerometer algorithms below first replace
        # each window with a virtually-flat-phone accel (orientation tracked,
        # gravity kept on +z). No-op otherwise, so existing callers are
        # unaffected.
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
        # First step for the accelerometer algorithms: optional input-cadence
        # normalization (config.resample_hz) so a window may arrive at any
        # sample rate, then optional orientation reconstruction. Each window
        # (ride, pre, post) is normalized independently — self-contained.
        hz = self.config.resample_hz
        if hz:
            data = _resample_to(data, hz)
            pre = _resample_to(pre, hz)
            post = _resample_to(post, hz)
        # Reconstruction of the ride + pre/post gravity-calibration windows.
        # Each window is reconstructed independently — vertically consistent
        # since every seed maps measured gravity onto world +z.
        r = self.config.reconstruct
        data = reconstruct_az(r, data, gyro)
        if pre is not None:
            pre = reconstruct_az(r, pre, pre_gyro)
        if post is not None:
            post = reconstruct_az(r, post, post_gyro)
        if algo is PredictAlgorithm.TRAPEZOID_ACCEL:
            return self._algo_impl.predict_segment(
                data, phone_model=phone_model, pre=pre, post=post,
                trapezoid_override=trapezoid_override,
            )
        # ZUPT and other accel estimators don't consume the trapezoid
        # override — call them with their existing signature.
        return self._algo_impl.predict_segment(
            data, phone_model=phone_model, pre=pre, post=post,
        )

    def calibrate(self, samples: list[CalibrationSample]) -> dict:
        if self.config.algorithm is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
            return {"note": "barometer_does_not_require_calibration"}
        return self._algo_impl.calibrate(samples)

    def save_calibration(self, path: Path | str | None = None) -> None:
        # ``path=None`` writes to the algorithm's own committed
        # calibration.json (the auto-loaded default).
        if self.config.algorithm is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
            return
        self._algo_impl.save(path)

    def load_calibration(self, path: Path | str | None = None) -> None:
        # ``path=None`` reloads the algorithm's own committed
        # calibration.json (the auto-loaded default).
        if self.config.algorithm is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
            return
        self._algo_impl.load(path)
