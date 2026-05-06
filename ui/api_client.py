"""In-process wrappers around the segmentation and prediction stages.

Originally a thin HTTP client around the FastAPI service in ``api/`` —
the Streamlit boutique pipeline now calls these wrappers directly so the
app runs without spinning up the API container. The function signatures
and return shapes mirror the ``/segment`` and ``/predict`` endpoints
exactly (see :mod:`api.main` for the HTTP-facing contract), so the
Streamlit step modules use them with no other changes.

The only behavioural difference vs. the HTTP path: state arrays come
back as ``np.ndarray`` and ``state['config']`` as the real
``DetectConfig`` dataclass, because nothing has been forced through
JSON. :func:`rehydrate_state` is therefore a no-op compatibility shim.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import pandas as pd

from src.prediction.algorithms import (
    PREDICT_ALGORITHM_CONFIG, PredictAlgorithm, Predictor,
)
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (
    detect as _detect,
)


# Stationary-window length used to calibrate gravity around a ride.
PRE_POST_WINDOW_SEC = 5.0
PRE_POST_MIN_SEC = 1.0

# Algorithms exposed — short id -> enum. The first entry is the
# "primary" the UI's sidebar list and PDF report default to. Kept in
# lock-step with ``api/main.py::_ACCEL_ALGO_MAP``.
_ACCEL_ALGO_MAP: dict[str, PredictAlgorithm] = {
    "trap": PredictAlgorithm.TRAPEZOID_ACCEL,
    "zupt": PredictAlgorithm.ZUPT_ACCEL,
}
_PRIMARY_ALGO_ID = "trap"

# Kept as an exported label so legacy error messages can still cite a
# "where" without being misleading.
API_URL = "in-process"


def _slice_acc(
    acc: pd.DataFrame, t0_ms: float, t_lo: float, t_hi: float,
) -> pd.DataFrame:
    ts = acc["timestamp_ms"].astype(float).to_numpy()
    lo_ms = t0_ms + t_lo * 1000.0
    hi_ms = t0_ms + t_hi * 1000.0
    mask = (ts >= lo_ms) & (ts < hi_ms)
    return acc.loc[mask].reset_index(drop=True)


def _slice_pre_post(
    acc: pd.DataFrame, t0_ms: float,
    seg_lo: float, seg_hi: float,
    prev_hi: Optional[float], next_lo: Optional[float],
    window_sec: float = PRE_POST_WINDOW_SEC,
    min_sec: float = PRE_POST_MIN_SEC,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    pre_lo = max(seg_lo - window_sec,
                 prev_hi if prev_hi is not None else seg_lo - window_sec)
    pre_hi = seg_lo
    post_lo = seg_hi
    post_hi = min(seg_hi + window_sec,
                  next_lo if next_lo is not None else seg_hi + window_sec)

    pre_df: Optional[pd.DataFrame] = None
    if pre_hi - pre_lo >= min_sec:
        pre_df = _slice_acc(acc, t0_ms, pre_lo, pre_hi)
        if pre_df.empty:
            pre_df = None
    post_df: Optional[pd.DataFrame] = None
    if post_hi - post_lo >= min_sec:
        post_df = _slice_acc(acc, t0_ms, post_lo, post_hi)
        if post_df.empty:
            post_df = None
    return pre_df, post_df


def _empty_pred_row(base: dict, reason: str) -> dict:
    return {
        **base,
        "delta_height_m": float("nan"),
        "abs_height_m":   float("nan"),
        "accepted":       False,
        "quality_score":  float("nan"),
        "reject_reason":  reason,
        "ci_half_width":  float("nan"),
        "meta":           {},
    }


def _selected_algos(req_algos: Optional[Iterable[str]]) -> list[str]:
    if req_algos is None:
        return list(_ACCEL_ALGO_MAP.keys())
    out: list[str] = []
    for a in req_algos:
        if a not in _ACCEL_ALGO_MAP:
            raise ValueError(
                f"unknown algorithm id: {a!r} "
                f"(valid: {sorted(_ACCEL_ALGO_MAP)})"
            )
        if a not in out:
            out.append(a)
    return out


def rehydrate_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """No-op shim kept for backwards compatibility.

    The HTTP client used to convert JSON-decoded lists back to
    ``np.ndarray`` and ``state['config']`` from a dict back to an
    attribute-accessible namespace. The in-process call returns those
    shapes already, so this function just returns its input.
    """
    return state


def segment(acc: pd.DataFrame, phone_model: str = "",
            include_state: bool = True) -> tuple[list[dict], dict | None, float | None]:
    """Detect ride intervals in an accelerometer trace.

    Returns ``(predictions, state, t0_ms)`` — same shape as the
    ``/segment`` endpoint. ``state`` is ``None`` when the detector
    produced nothing (e.g. empty trace) or when ``include_state=False``.
    """
    predictions, state = _detect.predict_intervals(
        acc, phone_model=phone_model,
    )
    t0_ms_raw = float(state.get("t0_ms", float("nan"))) if state else float("nan")
    t0_ms: float | None = t0_ms_raw if math.isfinite(t0_ms_raw) else None
    return predictions, (state if (include_state and state) else None), t0_ms


def predict(acc: pd.DataFrame,
            segments: Iterable[dict],
            phone_model: str = "",
            algorithms: list[str] | None = None) -> tuple[dict[str, list[dict]], str]:
    """Run Δh estimators on a list of ride intervals.

    ``segments`` is an iterable of dicts with ``type``, ``start_s``,
    ``end_s``. Returns ``(rows_by_algo, primary_algo_id)`` — same shape
    as the ``/predict`` endpoint.
    """
    segs = list(segments)
    if not segs:
        return {}, _PRIMARY_ALGO_ID

    t0_ms = float(acc["timestamp_ms"].iloc[0])

    chosen = _selected_algos(algorithms)
    predictors: dict[str, Predictor] = {
        aid: Predictor(PREDICT_ALGORITHM_CONFIG(algorithm=_ACCEL_ALGO_MAP[aid]))
        for aid in chosen
    }

    seg_starts = [float(s["start_s"]) for s in segs]
    seg_ends = [float(s["end_s"]) for s in segs]

    rows_by_algo: dict[str, list[dict]] = {aid: [] for aid in predictors}
    for pos, seg in enumerate(segs):
        t_lo = float(seg["start_s"])
        t_hi = float(seg["end_s"])
        rt = str(seg["type"])
        slice_df = _slice_acc(acc, t0_ms, t_lo, t_hi)
        prev_hi = seg_ends[pos - 1] if pos > 0 else None
        next_lo = seg_starts[pos + 1] if pos + 1 < len(segs) else None
        pre_df, post_df = _slice_pre_post(
            acc, t0_ms, t_lo, t_hi, prev_hi, next_lo,
        )
        base = {
            "segment":    int(pos),
            "type":       rt,
            "start_s":    t_lo,
            "end_s":      t_hi,
            "duration_s": t_hi - t_lo,
        }
        if slice_df.empty:
            for aid in predictors:
                rows_by_algo[aid].append(_empty_pred_row(base, "empty_slice"))
            continue
        for aid, predictor in predictors.items():
            try:
                out = predictor.predict(
                    slice_df,
                    phone_model=phone_model,
                    pre=pre_df, post=post_df,
                )
                dh = float(out.height_diff)
                ci = (float(out.ci_half_width)
                      if math.isfinite(out.ci_half_width) else float("nan"))
                signed = abs(dh) if rt == "up" else -abs(dh)
                rows_by_algo[aid].append({
                    **base,
                    "delta_height_m": signed,
                    "abs_height_m":   abs(dh),
                    "accepted":       bool(out.accepted),
                    "quality_score":  float(out.quality_score),
                    "reject_reason":  str(out.reject_reason or ""),
                    "ci_half_width":  ci,
                    "meta":           dict(out.meta) if out.meta else {},
                })
            except Exception as e:  # noqa: BLE001 — surface to caller as row
                rows_by_algo[aid].append(
                    _empty_pred_row(base, f"{type(e).__name__}: {e}"),
                )

    return rows_by_algo, _PRIMARY_ALGO_ID
