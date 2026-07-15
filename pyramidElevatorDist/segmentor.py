"""Public segmentation API.

Two functions:

* :func:`findSegments` — detect every elevator ride in an accelerometer
  trace and return the rich per-ride trapezoid fits.
* :func:`findSegmentParameters` — for a single user-marked interval, return
  the fitted trapezoid pulse-pair, its parameters, the per-lobe R² heatmaps,
  and the correlation map — everything the interactive editor shows for one
  segment.

Both load their (local, fixed) detector configuration internally; the caller
never passes hyperparameters. Input is always a pandas ``DataFrame`` with
columns ``timestamp_ms`` (Unix-epoch ms), ``x``, ``y``, ``z`` (raw
accelerometer, m/s²).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyramidElevatorDist.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E501
    detect as _detect,
    pair_filter as _pair_filter,
)

# The API is a thin façade: it only picks the cadence policy (external data →
# normalize to 50 Hz) and forwards. The Segmenter dispatcher does the detection
# *and* the input-cadence normalization; the shared core is pure orchestration.
from src.pipelines.inprocess import (
    RESAMPLE_TARGET_HZ,
    segment as _segment,
    find_matching_prediction as _find_matching_prediction,
)

__all__ = ["findSegments", "findSegmentParameters", "findSegmentsDetailed"]


def findSegments(acc: pd.DataFrame, phone_model: str = "",
                 resample: bool = True, gyro: pd.DataFrame | None = None,
                 reconstruct: str = "none") -> list[dict]:
    """Detect all elevator ride segments in an accelerometer trace.

    Parameters
    ----------
    acc:
        Raw accelerometer samples — columns ``timestamp_ms``, ``x``, ``y``,
        ``z``. May be at any (even variable) sample rate.
    phone_model:
        Optional phone identifier. When given, the detector tightens its
        amplitude floors using the phone's accelerometer noise σ.
    resample:
        When ``True`` (default), the trace is first normalized onto a uniform
        50 Hz grid (gap-aware, time-correct) before detection — the detector
        is tuned for that cadence and external data may arrive at any rate.
        Returned ride coordinates are relative seconds on this grid; pass the
        same ``acc`` (and the default ``resample``) to ``predictSegment`` so
        the coordinates line up. Set ``False`` only when ``acc`` is already at
        the canonical cadence.

    Returns
    -------
    list[dict]
        One dict per detected ride, each with::

            index            int    position in the list
            ride_type        str    "up" | "down"
            t_start_s        float  ride start, seconds (relative to acc[0])
            t_end_s          float  ride end, seconds
            duration_s       float  t_end_s - t_start_s
            lobe1, lobe2     dict   {t_c, a_peak, half_width_s, frac_flat, r2_local}
            joint_r2_mean    float  shared-shape mean R² across both lobes
            heatmap_energy   float  grid support of the match

        Empty list when the trace is empty/too short or no ride is found.
    """
    predictions, _state, _t0 = _segment(
        acc, phone_model=phone_model,
        resample_hz=(RESAMPLE_TARGET_HZ if resample else None),
        gyro=gyro, reconstruct=reconstruct,
    )
    return predictions


def _lobe_dict(t_c: float, a_peak: float, W: float, f: float,
               r2_local: float) -> dict:
    return {
        "t_c": float(t_c),
        "a_peak": float(a_peak),
        "half_width_s": float(W),
        "frac_flat": float(f),
        "r2_local": float(r2_local),
    }


def _fit_fresh(state: dict, t_lo: float, t_hi: float,
               ride_type: str | None):
    """Fit the trapezoid pulse-pair inside ``[t_lo, t_hi]`` from scratch.

    Mirrors the editor's manual-fit recipe: find the best +/- peak in the
    window, then run the shared-shape joint fit. Returns
    ``(lobe1, lobe2, joint_r2_mean, heatmap_energy, i1, i2)`` or ``None`` if
    the window lacks a usable +peak and -peak.
    """
    t = state["t"]
    a_smooth = state["a_smooth"]
    grid_w_s = state["grid_w_s"]
    grid_f = state["grid_f"]

    pos, neg = _detect._find_extrema_in_window(state, t_lo, t_hi)
    if ride_type == "up":
        first, second, s1 = pos, neg, +1.0
    elif ride_type == "down":
        first, second, s1 = neg, pos, -1.0
    else:
        if pos and neg and pos[0] < neg[0]:
            first, second, s1 = pos, neg, +1.0
        else:
            first, second, s1 = neg, pos, -1.0
    if first is None or second is None:
        return None
    i1, i2 = first[0], second[0]
    if i1 > i2:                       # enforce chronological order
        i1, i2, s1 = i2, i1, -s1
    s2 = -s1

    res = _pair_filter.joint_pair_score(
        a_smooth, t, i1, i2, s1, s2, grid_w_s, grid_f,
    )
    if res is None:
        return None
    score, W, f, A_abs, r2_1, r2_2, heatmap_energy = res
    lobe1 = _lobe_dict(t[i1], s1 * A_abs, W, f, r2_1)
    lobe2 = _lobe_dict(t[i2], s2 * A_abs, W, f, r2_2)
    return lobe1, lobe2, float(score), float(heatmap_energy), i1, i2


def _detail_from_state(
    state: dict, t_lo: float, t_hi: float, ride_type: str | None,
    predictions: list[dict] | None = None,
) -> dict | None:
    """Build one segment's detail (lobes, heatmaps, correlation) from an
    already-computed detector ``state`` + a ``[t_lo, t_hi]`` window.

    Split out of :func:`findSegmentParameters` so a single detector pass can
    serve many windows (see :func:`findSegmentsDetailed`). Pass ``predictions``
    precomputed to avoid re-running ``predict_pairs`` on every call.
    """
    t_arr = np.asarray(state["t"])
    if t_arr.size == 0:
        return None
    grid_w_s = state["grid_w_s"]
    grid_f = state["grid_f"]
    a_smooth = state["a_smooth"]

    if predictions is None:
        predictions = _pair_filter.predict_pairs(state, state["config"])
    matching = _find_matching_prediction(predictions, t_lo, t_hi)

    if matching is not None:
        lobe1 = dict(matching["lobe1"])
        lobe2 = dict(matching["lobe2"])
        joint_r2_mean = float(matching.get("joint_r2_mean", float("nan")))
        heatmap_energy = float(matching.get("heatmap_energy", float("nan")))
        rt = str(matching.get("ride_type", ride_type or "up"))
    else:
        fit = _fit_fresh(state, t_lo, t_hi, ride_type)
        if fit is None:
            return None
        lobe1, lobe2, joint_r2_mean, heatmap_energy, _i1, _i2 = fit
        if ride_type is not None:
            rt = ride_type
        else:
            rt = "up" if lobe1["a_peak"] > 0 else "down"

    i1 = int(np.clip(np.argmin(np.abs(t_arr - lobe1["t_c"])), 0, t_arr.size - 1))
    i2 = int(np.clip(np.argmin(np.abs(t_arr - lobe2["t_c"])), 0, t_arr.size - 1))
    heat1 = _detect.heatmap_at(a_smooth, state["t"], i1, grid_w_s, grid_f)
    heat2 = _detect.heatmap_at(a_smooth, state["t"], i2, grid_w_s, grid_f)

    W1 = float(lobe1["half_width_s"])
    W2 = float(lobe2["half_width_s"])
    return {
        "ride_type": rt,
        "t_start_s": float(lobe1["t_c"]) - W1,
        "t_end_s": float(lobe2["t_c"]) + W2,
        "lobe1": lobe1,
        "lobe2": lobe2,
        "joint_r2_mean": joint_r2_mean,
        "heatmap_energy": heatmap_energy,
        "heatmaps": {
            "lobe1": heat1,
            "lobe2": heat2,
            "grid_w_s": np.asarray(grid_w_s),
            "grid_f": np.asarray(grid_f),
        },
        "correlation": {
            "t": t_arr,
            "best_pos_r2": np.asarray(state["best_pos_r2"]),
            "best_neg_r2": np.asarray(state["best_neg_r2"]),
        },
    }


def findSegmentParameters(
    acc: pd.DataFrame,
    start_s: float,
    end_s: float,
    ride_type: str | None = None,
    phone_model: str = "",
    resample: bool = True,
    gyro: pd.DataFrame | None = None,
    reconstruct: str = "none",
) -> dict | None:
    """Fit the trapezoid pulse-pair + heatmaps for a marked interval.

    Use this for manual segment editing: the user marks ``[start_s, end_s]``
    (relative seconds on ``acc``'s own time axis) and gets back the trapezoid
    found inside, its parameters, and the visual diagnostics.

    ``resample`` (default ``True``) first normalizes ``acc`` onto the uniform
    50 Hz grid, matching ``findSegments``; ``start_s`` / ``end_s`` are
    interpreted on that grid. Set ``False`` only when ``acc`` is already at the
    canonical cadence.

    If the detector already proposed a ride overlapping the interval, that
    ride's lobe parameters are returned (identical to what the editor shows).
    Otherwise the trapezoid is fitted fresh inside the window.

    Returns ``None`` when the trace is unusable or the window has no usable
    +peak / -peak pair.

    Returns
    -------
    dict
        ::

            ride_type       str
            t_start_s       float   lobe1.t_c - half_width
            t_end_s         float   lobe2.t_c + half_width
            lobe1, lobe2    dict    {t_c, a_peak, half_width_s, frac_flat, r2_local}
            joint_r2_mean   float
            heatmap_energy  float
            heatmaps        dict    {lobe1: ndarray(nW,nF), lobe2: ndarray(nW,nF),
                                     grid_w_s: ndarray, grid_f: ndarray}
            correlation     dict    {t: ndarray, best_pos_r2: ndarray,
                                     best_neg_r2: ndarray}
    """
    # Detection (with its state) runs through the shared core / Segmenter
    # dispatcher, which also owns the optional resample + reconstruction.
    _preds, state, _t0 = _segment(
        acc, phone_model=phone_model, include_state=True,
        resample_hz=(RESAMPLE_TARGET_HZ if resample else None),
        gyro=gyro, reconstruct=reconstruct,
    )
    if not state:
        return None
    return _detail_from_state(state, float(start_s), float(end_s), ride_type)


def findSegmentsDetailed(
    acc: pd.DataFrame,
    phone_model: str = "",
    resample: bool = True,
    gyro: pd.DataFrame | None = None,
    reconstruct: str = "none",
) -> list[dict]:
    """Like :func:`findSegments`, but each ride dict also carries a ``"detail"``
    key — the full :func:`findSegmentParameters` result (lobes, heatmaps,
    correlation) — all computed in a SINGLE detector pass.

    This lets an interactive UI precompute every ride's detail once at load
    instead of re-running the detector on each click (the stateless per-segment
    call repeats the whole matched-filter pass every time). ``detail`` is
    ``None`` for the rare ride whose window yields no usable +/- pair.
    """
    preds, state, _t0 = _segment(
        acc, phone_model=phone_model, include_state=True,
        resample_hz=(RESAMPLE_TARGET_HZ if resample else None),
        gyro=gyro, reconstruct=reconstruct,
    )
    if not preds:
        return []
    predictions = (
        _pair_filter.predict_pairs(state, state["config"]) if state else []
    )
    out: list[dict] = []
    for p in preds:
        detail = (
            _detail_from_state(
                state, float(p["t_start_s"]), float(p["t_end_s"]),
                str(p.get("ride_type")), predictions=predictions,
            )
            if state else None
        )
        out.append({**p, "detail": detail})
    return out
