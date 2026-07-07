"""Segmentation algorithm dispatch.

:class:`Segmenter` is configured via :class:`SEGMENT_ALGORITHM_CONFIG`
(see ``configTypes.py``) which selects an algorithm and loads its
hyperparameters from ``config.json``.
"""

from __future__ import annotations

import pandas as pd

from src.physics.reconstruct_az import reconstruct_az

from .barometer_only import HeightSegmenter
from .accelerometer_only.template_match.check_grid_across_signal.detect import (
    DetectConfig,
    predict_intervals,
)
from .configTypes import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
    PressureFilterConfig,
    TemplateMatchConfig,
)


_SEGMENT_OUTPUT_COLUMNS = [
    "start_ci", "end_ci", "duration", "type", "probability_ci",
]


def _template_match_to_segment_df(predictions: list[dict]) -> pd.DataFrame:
    """Adapt the active detector's list-of-dicts output to the Segmenter's
    CI-valued DataFrame schema.

    The grid detector is deterministic and returns point estimates, so
    ``start_ci`` / ``end_ci`` collapse to zero-width tuples.
    ``probability_ci`` is filled with the pair's joint R² (mean across
    both lobes) as a scalar confidence — identical low/high because the
    detector does not quantify uncertainty.
    """
    rows: list[dict] = []
    for p in predictions:
        t_s = float(p["t_start_s"])
        t_e = float(p["t_end_s"])
        r2 = float(p.get("joint_r2_mean", 1.0))
        rows.append({
            "start_ci": (t_s, t_s),
            "end_ci":   (t_e, t_e),
            "duration": float(p.get("duration_s", t_e - t_s)),
            "type":     str(p.get("ride_type", "ride")),
            "probability_ci": (r2, r2),
        })
    return pd.DataFrame(rows, columns=_SEGMENT_OUTPUT_COLUMNS)


def _resample_to(frame: pd.DataFrame, hz: int) -> pd.DataFrame:
    """Normalize a timestamped sensor frame onto a uniform ``hz`` grid
    (gap-aware, time-correct). Returns the input unchanged when it is unusable
    (``None`` / <2 rows / no ``timestamp_ms``). Lazy import breaks the
    loader<->segmenter module cycle."""
    if (frame is None or "timestamp_ms" not in getattr(frame, "columns", [])
            or len(frame) < 2):
        return frame
    from src.data.loader import resample_sensor_with_gaps
    resampled, _intervals = resample_sensor_with_gaps(frame, target_hz=hz)
    return resampled


class Segmenter:
    def __init__(self, config: SEGMENT_ALGORITHM_CONFIG):
        self.config = config
        self.params = config.load_params()

    # Dispatch a segmentation algorithm on a sensor time series.
    #
    # Input (`data`): a pandas DataFrame of raw sensor samples whose required
    # columns depend on the selected algorithm:
    #   - PRESSURE_FILTER    → columns `time` (sec) and `height` (m) by default;
    #                          column names are configurable via
    #                          `PressureFilterConfig.time_col` / `height_col`.
    #   - ACC_TEMPLATE_MATCH → columns `timestamp_ms` (Unix-epoch ms),
    #                          `x`, `y`, `z` (raw accelerometer, m/s^2).
    #                          The grid detector reads those names directly
    #                          and derives fs / relative time from them.
    #
    # `phone_model` (optional): when non-empty, the ACC_TEMPLATE_MATCH path
    # tightens its amplitude floors using the phone's accelerometer noise σ
    # (see `src.utils.sensor_noise` and the `noise_sigma_multiplier` knob on
    # `TemplateMatchConfig`). Ignored by the pressure filter.
    #
    # Output: a pandas DataFrame of detected elevator segments with columns:
    #   - `start_ci`       : (low, high) confidence interval for ride start time
    #   - `end_ci`         : (low, high) confidence interval for ride end time
    #   - `duration`       : end - start, seconds
    #   - `type`           : ride label ("up"/"down" for pressure,
    #                        "up"/"down" for template match too)
    #   - `probability_ci` : (low, high) CI on ride probability — joint R²
    #                        for template match, (1, 1) for pressure
    # An empty DataFrame with the same columns is returned when no segments
    # are detected (or the input is too short).
    #
    # `gyro` (optional): raw gyroscope (`timestamp_ms, x, y, z`, rad/s). When
    # `config.reconstruct != "none"` and gyro is present, the ACC_TEMPLATE_MATCH
    # path first replaces `data` with a virtually-flat-phone accelerometer (see
    # `src.physics.reconstruct_az.reconstruct_az`); otherwise it is ignored.
    def detect_raw(
        self,
        data: pd.DataFrame,
        phone_model: str = "",
        gyro: pd.DataFrame | None = None,
    ) -> tuple[list[dict], dict]:
        """Low-level ACC_TEMPLATE_MATCH detection through the dispatcher.

        Returns the detector's own ``(predictions, state)`` — the rich
        per-ride dicts (lobes, joint R², heatmap energy) plus the ``state``
        (gridded scores, correlation maps, ``a_vert``/``a_smooth``, the live
        ``DetectConfig``). :meth:`detect` adapts this to the CI DataFrame; the
        in-process boutique pipeline (``src.pipelines.inprocess``) and the
        interactive editor consume the raw form directly, so they no longer
        need to reach past the dispatcher into the low-level detector.

        The orientation-reconstruction first step is applied here (no-op when
        ``config.reconstruct == "none"`` or gyro is absent), so every caller —
        CI-DataFrame or raw — gets it uniformly.
        """
        if self.config.algorithm is not SegmentAlgorithm.ACC_TEMPLATE_MATCH:
            raise ValueError(
                "detect_raw is only defined for ACC_TEMPLATE_MATCH; "
                f"got {self.config.algorithm}"
            )
        # Optional input-cadence normalization: accept any/variable sample rate
        # and resample onto the tuned grid before detecting (no-op when
        # config.resample_hz is None). Gyro is left native — reconstruct_az
        # aligns it to the (resampled) accel timeline itself.
        if self.config.resample_hz:
            data = _resample_to(data, self.config.resample_hz)
        data = reconstruct_az(self.config.reconstruct, data, gyro)
        algo_config = TemplateMatchConfig(**self.params)
        detect_cfg = DetectConfig(**algo_config.model_dump())
        return predict_intervals(data, detect_cfg, phone_model=phone_model)

    def detect(
        self,
        data: pd.DataFrame,
        phone_model: str = "",
        gyro: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self.config.algorithm is SegmentAlgorithm.PRESSURE_FILTER:
            algo_config = PressureFilterConfig(**self.params)
            return HeightSegmenter(algo_config).segment(data)
        if self.config.algorithm is SegmentAlgorithm.ACC_TEMPLATE_MATCH:
            predictions, _state = self.detect_raw(
                data, phone_model=phone_model, gyro=gyro,
            )
            return _template_match_to_segment_df(predictions)
        raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
