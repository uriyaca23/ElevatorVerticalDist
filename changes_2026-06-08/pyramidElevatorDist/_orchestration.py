"""Private orchestration shared by the public façade modules.

This module is **not** part of the public API. It mirrors the in-process
orchestration that the Streamlit boutique pipeline uses today
(``ui/api_client.py``) so the installed package reproduces the exact same
segmentation and prediction outputs without depending on the ``ui/``
application layer (which is not shipped in the wheel).

The slicing helpers, the per-algorithm signal policy, and the predict loop
below are kept byte-for-byte equivalent to ``ui/api_client.py`` —
``ui/api_client.py`` remains the source of truth; the regression tests in
``tests/`` pin this copy to it.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import pandas as pd

from src.data.loader import resample_sensor_with_gaps
from src.prediction.algorithms import (
    PREDICT_ALGORITHM_CONFIG, PredictAlgorithm, Predictor,
)
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E501
    detect as _detect,
)


# --------------------------------------------------------------------------
# Input normalization (public API entry point)
# --------------------------------------------------------------------------

# Canonical uniform cadence the detector + Δh estimators are tuned for.
RESAMPLE_TARGET_HZ = 50


def _resample_acc(acc: pd.DataFrame, target_hz: int = RESAMPLE_TARGET_HZ):
    """Normalize an external accelerometer trace onto a uniform ``target_hz``
    grid (default 50 Hz).

    The public API accepts data at any — possibly variable — sample rate, but
    the matched-filter detector and the Δh estimators are tuned for a clean
    uniform cadence. Resampling is gap-aware (splits on >1 s holes rather than
    interpolating across them) and time-correct (each output sample keeps its
    true timestamp, so ``findSegments`` ride coordinates line up with
    ``predictSegment``). The first timestamp and the overall time span are
    preserved, so relative-second segment coordinates stay valid.

    Returns the input unchanged when it is ``None``, lacks ``timestamp_ms``,
    or has fewer than 2 rows.
    """
    if (acc is None or "timestamp_ms" not in getattr(acc, "columns", [])
            or len(acc) < 2):
        return acc
    resampled, _intervals = resample_sensor_with_gaps(acc, target_hz=target_hz)
    return resampled


# --------------------------------------------------------------------------
# Constants — kept in lock-step with ui/api_client.py
# --------------------------------------------------------------------------

# Stationary-window length used to calibrate gravity around a ride.
PRE_POST_WINDOW_SEC = 5.0
PRE_POST_MIN_SEC = 1.0

# Algorithms exposed — short id -> enum. The first entry is the "primary"
# the sidebar list and PDF report default to.
_ACCEL_ALGO_MAP: dict[str, PredictAlgorithm] = {
    "trap": PredictAlgorithm.TRAPEZOID_ACCEL,
    "zupt": PredictAlgorithm.ZUPT_ACCEL,
}
_PRIMARY_ALGO_ID = "trap"

# Boutique-pipeline hybrid signal policy: trapezoid runs on the
# rotation-invariant |a|-g (matching what segmentation feeds it), ZUPT
# keeps the gravity-projected a_vert because its quality is bounded by the
# double-integration drift model, not the matched-filter signal choice.
_PER_ALGO_OVERRIDES: dict[str, dict] = {
    "trap": {"input_signal": "a_mag_minus_g"},
    "zupt": {},
}

# Segmentation signal: the boutique pipeline scores the matched filter on
# the rotation-invariant |a|-g residual.
_SEGMENT_INPUT_SIGNAL = "a_mag_minus_g"


# --------------------------------------------------------------------------
# Slicing helpers (copied from ui/api_client.py)
# --------------------------------------------------------------------------

def _slice_acc(
    acc: pd.DataFrame, t0_ms: float, t_lo: float, t_hi: float,
) -> pd.DataFrame:
    # Inclusive on both ends — matches the segmenter's slicing convention
    # so a row whose timestamp lands exactly on a segment boundary is
    # consistently kept rather than silently dropped on uneven sampling.
    ts = acc["timestamp_ms"].astype(float).to_numpy()
    lo_ms = t0_ms + t_lo * 1000.0
    hi_ms = t0_ms + t_hi * 1000.0
    mask = (ts >= lo_ms) & (ts <= hi_ms)
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


# --------------------------------------------------------------------------
# Segmentation primitive
# --------------------------------------------------------------------------

def _segment_cfg():
    """Build the matched-filter detector config from the live segmentation
    ``config.json`` (key ``acc_template_match``) — the same source of truth the
    ``Segmenter`` and the evaluators read. This is what makes tuned
    hyperparameters written by ``scripts/tune_hyperparameters.py`` take effect
    in the public API on the very next call, with no code change. The boutique
    signal policy (rotation-invariant ``|a|-g``) is applied as an override.
    """
    from src.segmentation.algorithms.configTypes import (
        SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, TemplateMatchConfig,
    )
    params = SEGMENT_ALGORITHM_CONFIG(
        algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
        overrides={"input_signal": _SEGMENT_INPUT_SIGNAL},
    ).load_params()
    return _detect.DetectConfig(**TemplateMatchConfig(**params).model_dump())


def _segment(acc: pd.DataFrame, phone_model: str = ""):
    """Run the trapezoid-template detector on the boutique |a|-g signal.

    Returns ``(predictions, state)`` exactly as
    :func:`detect.predict_intervals`. ``state`` is ``{}`` when the trace is
    unusable.
    """
    cfg = _segment_cfg()
    predictions, state = _detect.predict_intervals(
        acc, cfg, phone_model=phone_model,
    )
    return predictions, state


def find_matching_prediction(
    predictions: list[dict], t_lo: float, t_hi: float,
) -> dict | None:
    """Best-overlap detector prediction for a user-edited segment.

    Returns ``None`` when there is no overlap — happens when the user marks
    a segment the detector never proposed. Inlined from
    ``src/pipelines/streamlit/common.py`` so the package never imports the
    Streamlit layer.
    """
    best = None
    best_overlap = 0.0
    for p in predictions:
        s = float(p["t_start_s"])
        e = float(p["t_end_s"])
        overlap = max(0.0, min(e, t_hi) - max(s, t_lo))
        if overlap > best_overlap:
            best_overlap = overlap
            best = p
    return best


# --------------------------------------------------------------------------
# Prediction orchestration (copied from ui/api_client.py::predict)
# --------------------------------------------------------------------------

def _predict_segments(
    acc: pd.DataFrame,
    segments: Iterable[dict],
    phone_model: str = "",
    algorithms: list[str] | None = None,
) -> tuple[dict[str, list[dict]], str]:
    """Run the Δh estimators over a list of ride intervals.

    ``segments`` is an iterable of dicts with ``type``, ``start_s``,
    ``end_s`` (and optional ``trapezoid_override``). Neighbour clamping for
    the pre/post gravity windows uses each segment's position in the list,
    so callers wanting isolated single-segment behaviour pass a 1-element
    list. Returns ``(rows_by_algo, primary_algo_id)``.
    """
    segs = list(segments)
    if not segs:
        return {}, _PRIMARY_ALGO_ID

    t0_ms = float(acc["timestamp_ms"].iloc[0])

    chosen = _selected_algos(algorithms)
    predictors: dict[str, Predictor] = {
        aid: Predictor(PREDICT_ALGORITHM_CONFIG(
            algorithm=_ACCEL_ALGO_MAP[aid],
            overrides=_PER_ALGO_OVERRIDES.get(aid, {}),
        ))
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
        # Manual trapezoid override — only the trapezoid estimator consumes
        # it; other algorithms ignore the kwarg via the Predictor dispatcher.
        seg_override = seg.get("trapezoid_override")
        for aid, predictor in predictors.items():
            try:
                out = predictor.predict(
                    slice_df,
                    phone_model=phone_model,
                    pre=pre_df, post=post_df,
                    trapezoid_override=seg_override,
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
