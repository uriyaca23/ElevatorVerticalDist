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
"""
from __future__ import annotations

import pandas as pd

from ._orchestration import _predict_segments, _resample_acc

__all__ = ["predictSegment", "predictByParameters"]


def _rows_to_result(rows_by_algo: dict[str, list[dict]],
                    primary: str) -> dict:
    """Flatten the single-segment ``_predict_segments`` output to one dict."""
    result: dict = {"primary": primary}
    for aid, rows in rows_by_algo.items():
        result[aid] = rows[0] if rows else None
    return result


def predictSegment(
    acc: pd.DataFrame,
    segment: dict,
    phone_model: str = "",
    algorithms: list[str] | None = None,
    resample: bool = True,
) -> dict:
    """Predict Δh for one ride segment.

    Parameters
    ----------
    acc:
        Raw accelerometer samples — columns ``timestamp_ms``, ``x``, ``y``,
        ``z``.
    segment:
        Ride interval — ``{"type": "up"|"down", "start_s": float,
        "end_s": float}``. May also carry a ``"trapezoid_override"`` dict
        (see :func:`predictByParameters`).
    phone_model:
        Optional phone identifier for the accelerometer noise model.
    algorithms:
        Subset of ``["trap", "zupt"]`` to run. ``None`` runs both.
    resample:
        When ``True`` (default), ``acc`` is first normalized onto the uniform
        50 Hz grid (gap-aware, time-correct), matching ``findSegments``. The
        ``segment`` ``start_s`` / ``end_s`` are interpreted on that grid, so
        pass the same ``acc`` and default ``resample`` you used for detection.
        Set ``False`` only when ``acc`` is already at the canonical cadence.

    Returns
    -------
    dict
        ``{"primary": "trap", "trap": <row>, "zupt": <row>}`` where each row
        has::

            delta_height_m   float   signed Δh (up +, down -), meters
            abs_height_m     float   |Δh|, meters
            accepted         bool    passed the quality filter
            quality_score    float   0 = excellent, higher = worse
            reject_reason    str     empty when accepted
            ci_half_width    float   90% CI half-width, meters (nan if rejected)
            meta             dict    algorithm-specific extras
            type, start_s, end_s, duration_s, segment
    """
    if resample:
        acc = _resample_acc(acc)
    rows_by_algo, primary = _predict_segments(
        acc, [segment], phone_model=phone_model, algorithms=algorithms,
    )
    return _rows_to_result(rows_by_algo, primary)


def predictByParameters(
    acc: pd.DataFrame,
    segment: dict,
    trapezoid_params: dict,
    phone_model: str = "",
    algorithms: list[str] | None = None,
    resample: bool = True,
) -> dict:
    """Predict Δh for a segment using a manually edited trapezoid.

    The user overrides the fitted pulse shape; only the trapezoid estimator
    consumes the override (it recomputes Δh closed-form from the new shape
    while keeping the detected lobe centres). ZUPT ignores it and runs as in
    :func:`predictSegment`.

    Parameters
    ----------
    trapezoid_params:
        ``{"W": float, "f": float, "abs_A": float}`` — half-width (s),
        plateau fraction (0–1), and absolute peak amplitude (m/s²).
    resample:
        Same as :func:`predictSegment` (default ``True``). The resampling is
        applied once, by the delegated ``predictSegment`` call.

    Returns
    -------
    dict
        Same shape as :func:`predictSegment`.
    """
    seg = {
        **segment,
        "trapezoid_override": {"mode": "manual", **trapezoid_params},
    }
    return predictSegment(
        acc, seg, phone_model=phone_model, algorithms=algorithms,
        resample=resample,
    )
