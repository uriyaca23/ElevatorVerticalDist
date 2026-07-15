"""Public prediction API.

Two functions:

* :func:`predictSegment` — predict the signed Δh (with confidence interval)
  for a single ride segment using the detector's own trapezoid fit.
* :func:`predictByParameters` — same, but with a manually edited trapezoid
  (the user overrides the pulse shape W / f / |A|).

Both run the boutique pipeline's two estimators — the trapezoid pulse-pair
fit (primary, on the rotation-invariant |a|-g signal) and ZUPT
double-integration (on the gravity-projected a_vert) — with pre/post
stationary-window gravity calibration. Configuration is loaded internally;
the caller never passes hyperparameters.

Inputs are validated on entry (see :mod:`pyramidElevatorDist.schemas` /
:mod:`pyramidElevatorDist.exceptions`); the ``segment`` /
``trapezoid_params`` arguments accept either the typed models
(:class:`~pyramidElevatorDist.types.SegmentSpec`,
:class:`~pyramidElevatorDist.types.TrapezoidParams`) or an equivalent
mapping, which is validated and coerced. Results are typed
:class:`~pyramidElevatorDist.types.PredictionResult` models, never dicts.

The algorithm-dispatcher layer lives in
:mod:`pyramidElevatorDist.prediction.algorithms` (``Predictor`` et al.).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

# Thin façade: it only picks the cadence policy (external data → normalize to
# 50 Hz) and forwards. The Predictor dispatcher normalizes each window's sample
# rate itself; the shared core just slices ride/pre/post and loops.
from pyramidElevatorDist._orchestration import (
    RESAMPLE_TARGET_HZ,
    predict as _predict_segments,
)
from pyramidElevatorDist._validation import (
    check_algorithms,
    check_reconstruct,
    coerce_segment,
    coerce_trapezoid,
    segment_to_core_dict,
    validate_acc,
    validate_gyro,
    validate_prs,
)
from pyramidElevatorDist.types.prediction import (
    PredictionResult,
    PredictionRow,
    SegmentSpec,
    TrapezoidOverride,
    TrapezoidParams,
)

__all__ = ["predictSegment", "predictByParameters"]


def _rows_to_result(rows_by_algo: dict[str, list[dict]],
                    primary: str) -> PredictionResult:
    """Flatten the single-segment ``_predict_segments`` output to one
    typed :class:`PredictionResult`."""
    result: dict[str, Any] = {"primary": primary}
    for aid, rows in rows_by_algo.items():
        result[aid] = (
            PredictionRow.model_validate(rows[0]) if rows else None
        )
    return PredictionResult.model_validate(result)


def predictSegment(
    acc: pd.DataFrame,
    segment: SegmentSpec | Mapping[str, Any],
    phone_model: str = "",
    algorithms: Sequence[str] | None = None,
    resample: bool = True,
    gyro: pd.DataFrame | None = None,
    reconstruct: str = "none",
    prs: pd.DataFrame | None = None,
) -> PredictionResult:
    """Predict Δh for one ride segment.

    Parameters
    ----------
    acc:
        Raw accelerometer samples — columns ``timestamp_ms``, ``x``, ``y``,
        ``z``.
    segment:
        Ride interval — a :class:`SegmentSpec` or a mapping
        ``{"type": "up"|"down", "start_s": float, "end_s": float}``. May
        also carry a ``trapezoid_override`` (see
        :func:`predictByParameters`).
    phone_model:
        Optional phone identifier for the accelerometer noise model.
    algorithms:
        Subset of ``["trap", "zupt"]`` to run. ``None`` runs both.
    prs:
        Optional pressure frame (columns ``timestamp_ms``, ``pressure``) for
        the whole session. When given, the result gains a ``baro`` row — the
        barometer (ground-truth) Δh for the segment, sign as measured.
    resample:
        When ``True`` (default), ``acc`` is first normalized onto the uniform
        50 Hz grid (gap-aware, time-correct), matching ``findSegments``. The
        ``segment`` ``start_s`` / ``end_s`` are interpreted on that grid, so
        pass the same ``acc`` and default ``resample`` you used for detection.
        Set ``False`` only when ``acc`` is already at the canonical cadence.

    Returns
    -------
    PredictionResult
        Typed per-algorithm rows: ``result.trap`` / ``result.zupt`` (and
        ``result.baro`` when ``prs`` was given), each a
        :class:`~pyramidElevatorDist.types.PredictionRow` or ``None``.
    """
    acc = validate_acc(acc, func="predictSegment")
    gyro = validate_gyro(gyro, func="predictSegment")
    prs = validate_prs(prs, func="predictSegment", allow_empty=True)
    reconstruct = check_reconstruct(reconstruct, func="predictSegment")
    algorithms = check_algorithms(algorithms, func="predictSegment")
    spec = coerce_segment(segment, func="predictSegment")

    rows_by_algo, primary = _predict_segments(
        acc, [segment_to_core_dict(spec)],
        phone_model=phone_model, algorithms=algorithms,
        resample_hz=(RESAMPLE_TARGET_HZ if resample else None),
        gyro=gyro, reconstruct=reconstruct, prs=prs,
    )
    return _rows_to_result(rows_by_algo, primary)


def predictByParameters(
    acc: pd.DataFrame,
    segment: SegmentSpec | Mapping[str, Any],
    trapezoid_params: TrapezoidParams | Mapping[str, Any],
    phone_model: str = "",
    algorithms: Sequence[str] | None = None,
    resample: bool = True,
    gyro: pd.DataFrame | None = None,
    reconstruct: str = "none",
    prs: pd.DataFrame | None = None,
) -> PredictionResult:
    """Predict Δh for a segment using a manually edited trapezoid.

    The user overrides the fitted pulse shape; only the trapezoid estimator
    consumes the override (it recomputes Δh closed-form from the new shape
    while keeping the detected lobe centres). ZUPT ignores it and runs as in
    :func:`predictSegment`.

    Parameters
    ----------
    trapezoid_params:
        A :class:`TrapezoidParams` or a mapping ``{"W": float, "f": float,
        "abs_A": float}`` — half-width (s), plateau fraction (0–1), and
        absolute peak amplitude (m/s²).
    resample:
        Same as :func:`predictSegment` (default ``True``). The resampling is
        applied once, by the delegated ``predictSegment`` call.

    Returns
    -------
    PredictionResult
        Same shape as :func:`predictSegment`.
    """
    spec = coerce_segment(segment, func="predictByParameters")
    params = coerce_trapezoid(trapezoid_params, func="predictByParameters")
    spec = spec.model_copy(update={
        "trapezoid_override": TrapezoidOverride(
            W=params.W, f=params.f, abs_A=params.abs_A,
        ),
    })
    return predictSegment(
        acc, spec, phone_model=phone_model, algorithms=algorithms,
        resample=resample, gyro=gyro, reconstruct=reconstruct, prs=prs,
    )
