"""FastAPI application — entry points.

Two POST endpoints:

* ``/segment`` — runs the accelerometer-only matched-filter detector and
  returns the rides it found together with the detector's full state dict
  (the bag the editor / heatmap UIs need to render diagnostics).
* ``/predict`` — runs every accelerometer-only Δh estimator over a list of
  ride intervals and returns one row per (segment, algorithm) pair.

Both endpoints are stateless: the client posts the accelerometer samples
each time. The Streamlit boutique UI in ``ui/`` is one consumer; downstream
apps are others.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Make ``src`` importable when uvicorn is launched from the repo root or
# from inside the api/ directory directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.prediction.algorithms import (  # noqa: E402
    PREDICT_ALGORITHM_CONFIG, PredictAlgorithm, Predictor,
)
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E402
    detect as _detect,
)

from .encoding import jsonify  # noqa: E402
from .schemas import (  # noqa: E402
    AccPayload, PredictRequest, SegmentRequest, SegmentRow,
)


app = FastAPI(
    title="ElevatorVerticalDist API",
    description=(
        "HTTP wrapper around the segmentation and prediction stages. "
        "Both endpoints are pure — they take accelerometer samples and "
        "return JSON; no on-disk state."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NaNSafeJSONResponse(JSONResponse):
    """JSON responses with non-finite floats encoded as ``null``.

    The body has already been passed through :func:`jsonify`, so by the
    time we hit ``json.dumps`` there should be no NaN/Inf left. We still
    pass ``allow_nan=False`` to fail loudly if a stray one slips through —
    bare ``NaN`` tokens break strict parsers and we'd rather know early.
    """

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


def _acc_to_df(acc: AccPayload) -> pd.DataFrame:
    n = len(acc.timestamp_ms)
    if not (len(acc.x) == len(acc.y) == len(acc.z) == n):
        raise HTTPException(
            status_code=400,
            detail=("acc arrays must all be the same length "
                    f"(timestamp_ms={n}, x={len(acc.x)}, "
                    f"y={len(acc.y)}, z={len(acc.z)})"),
        )
    if n == 0:
        raise HTTPException(status_code=400, detail="acc payload is empty")
    return pd.DataFrame({
        "timestamp_ms": np.asarray(acc.timestamp_ms, dtype=float),
        "x": np.asarray(acc.x, dtype=float),
        "y": np.asarray(acc.y, dtype=float),
        "z": np.asarray(acc.z, dtype=float),
    })


def _slice_acc(
    acc: pd.DataFrame, t0_ms: float, t_lo: float, t_hi: float,
) -> pd.DataFrame:
    ts = acc["timestamp_ms"].astype(float).to_numpy()
    lo_ms = t0_ms + t_lo * 1000.0
    hi_ms = t0_ms + t_hi * 1000.0
    mask = (ts >= lo_ms) & (ts < hi_ms)
    return acc.loc[mask].reset_index(drop=True)


# Stationary-window length used to calibrate gravity around a ride. Same
# constants the original step4 of the Streamlit pipeline used — copied
# here so the boutique UI now no longer owns slicing logic.
PRE_POST_WINDOW_SEC = 5.0
PRE_POST_MIN_SEC = 1.0


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


# Algorithms exposed on /predict — short id → enum. Order matters: the
# first entry is the "primary" the UI's sidebar list and PDF report
# default to.
_ACCEL_ALGO_MAP: dict[str, PredictAlgorithm] = {
    "trap": PredictAlgorithm.TRAPEZOID_ACCEL,
    "zupt": PredictAlgorithm.ZUPT_ACCEL,
}
_PRIMARY_ALGO_ID = "trap"


def _empty_pred_row(base: dict, reason: str) -> dict:
    return {
        **base,
        "delta_height_m": float("nan"),
        "abs_height_m":   float("nan"),
        "accepted":       False,
        "quality_score":  float("nan"),
        "reject_reason":  reason,
        "ci_half_width":  float("nan"),
    }


def _selected_algos(req_algos: Optional[Iterable[str]]) -> list[str]:
    if req_algos is None:
        return list(_ACCEL_ALGO_MAP.keys())
    out: list[str] = []
    for a in req_algos:
        if a not in _ACCEL_ALGO_MAP:
            raise HTTPException(
                status_code=400,
                detail=f"unknown algorithm id: {a!r} "
                       f"(valid: {sorted(_ACCEL_ALGO_MAP)})",
            )
        if a not in out:
            out.append(a)
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — used by docker-compose / Kubernetes."""
    return {"status": "ok"}


@app.post("/segment", response_class=_NaNSafeJSONResponse)
def segment(req: SegmentRequest) -> Any:
    """Detect ride intervals in an accelerometer trace.

    Returns a JSON object with:

    * ``predictions`` — list of detected rides (the same dicts the editor
      consumes: ``t_start_s``, ``t_end_s``, ``ride_type``, ``duration_s``,
      ``joint_r2_mean``, plus ``lobe1`` / ``lobe2`` shared-shape fits).
    * ``t0_ms`` — epoch origin used by the detector. Pass it back to
      ``/predict`` so the segment time axis matches.
    * ``state`` — full detector internals dict (numpy arrays converted to
      lists). Present iff ``include_state=true`` (the default). Holds
      everything the editor / heatmap UIs need: ``t``, ``a_vert``,
      ``a_smooth``, per-sign R² traces, peak indices, the (W, f) grid,
      and the effective ``DetectConfig`` after any phone-aware noise
      tightening.
    """
    acc_df = _acc_to_df(req.acc)
    predictions, state = _detect.predict_intervals(
        acc_df, phone_model=req.phone_model,
    )
    t0_ms = float(state.get("t0_ms", float("nan"))) if state else float("nan")
    payload: dict[str, Any] = {
        "predictions": jsonify(predictions),
        "t0_ms": t0_ms if math.isfinite(t0_ms) else None,
    }
    if req.include_state:
        payload["state"] = jsonify(state) if state else None
    return payload


@app.post("/predict", response_class=_NaNSafeJSONResponse)
def predict(req: PredictRequest) -> Any:
    """Run Δh estimators on a list of ride intervals.

    Each segment is sliced from the posted ACC stream by ``start_s`` /
    ``end_s`` (seconds relative to ``acc.timestamp_ms[0]``), and a
    ``±5 s`` stationary window is sliced on each side for gravity
    calibration. Neighbouring-segment boundaries clip those windows so
    back-to-back rides don't pollute each other.

    Response:

    * ``rows_by_algo`` — ``{algo_id: [row, ...]}`` with one row per input
      segment, in the same order. Each row carries
      ``segment``, ``type``, ``start_s``, ``end_s``, ``duration_s``,
      ``delta_height_m`` (signed), ``abs_height_m``, ``ci_half_width``,
      ``quality_score``, ``accepted``, ``reject_reason``.
    * ``primary`` — the short id whose rows are conventionally the
      "primary" prediction (currently ``"trap"``).
    """
    acc_df = _acc_to_df(req.acc)
    if not req.segments:
        return {"rows_by_algo": {}, "primary": _PRIMARY_ALGO_ID}

    t0_ms = float(acc_df["timestamp_ms"].iloc[0])

    chosen = _selected_algos(req.algorithms)
    predictors: dict[str, Predictor] = {
        aid: Predictor(PREDICT_ALGORITHM_CONFIG(algorithm=_ACCEL_ALGO_MAP[aid]))
        for aid in chosen
    }

    segs: list[SegmentRow] = list(req.segments)
    seg_starts = [float(s.start_s) for s in segs]
    seg_ends = [float(s.end_s) for s in segs]

    rows_by_algo: dict[str, list[dict]] = {aid: [] for aid in predictors}
    for pos, seg in enumerate(segs):
        t_lo = float(seg.start_s)
        t_hi = float(seg.end_s)
        rt = seg.type
        slice_df = _slice_acc(acc_df, t0_ms, t_lo, t_hi)
        prev_hi = seg_ends[pos - 1] if pos > 0 else None
        next_lo = seg_starts[pos + 1] if pos + 1 < len(segs) else None
        pre_df, post_df = _slice_pre_post(
            acc_df, t0_ms, t_lo, t_hi, prev_hi, next_lo,
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
                    phone_model=req.phone_model,
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
                })
            except Exception as e:  # noqa: BLE001 — surface to client as row
                rows_by_algo[aid].append(
                    _empty_pred_row(base, f"{type(e).__name__}: {e}"),
                )

    return {
        "rows_by_algo": jsonify(rows_by_algo),
        "primary": _PRIMARY_ALGO_ID,
    }
