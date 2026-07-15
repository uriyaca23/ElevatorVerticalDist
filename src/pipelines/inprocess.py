"""In-process boutique pipeline: the single shared segment + predict core.

This is the one place the "resample → segment → slice ride/pre/post →
predict-per-algorithm" orchestration lives. Both façades delegate here so
there is exactly one implementation:

* ``ui/api_client.py``            — the Streamlit boutique pipeline's adapter.
* ``pyramidElevatorDist``         — the shippable public package.

Everything runs *through the stage dispatchers* — :class:`Segmenter` for
detection (via :meth:`Segmenter.detect_raw`, which returns the rich per-ride
predictions plus the detector ``state`` the Streamlit heatmaps need) and
:class:`Predictor` for Δh — so a single config selects hyperparameters,
signal policy, and (optionally) accel+gyro orientation reconstruction.

This module is a *pure orchestrator*: it slices ride/pre/post windows and
assembles result rows, but owns **no** signal processing. Input-cadence
normalization now lives inside the dispatchers themselves (config
``resample_hz``), so :class:`Segmenter`/:class:`Predictor` accept data at any
sample rate and handle it internally — see ``Segmenter.detect_raw`` /
``Predictor.predict``.

The two knobs a caller may set:

* ``resample_hz`` — the cadence the dispatchers normalize the input onto
  (gap-aware, time-correct), passed straight through onto their config.
  ``None`` (default) consumes the data as-is; the Streamlit path leaves it
  ``None`` because it already resamples at load time
  (``src.data.load_data.enrich_loaded``), and the package sets it to 50 for
  arbitrary-cadence external input.
* ``gyro`` / ``reconstruct`` — when a gyroscope stream is supplied and
  ``reconstruct != "none"``, both stages first replace the accelerometer with
  a virtually-flat-phone signal (orientation tracked, gravity kept on +z).
  No-op otherwise, so the default path is byte-for-byte the legacy behaviour.

Boutique signal policy (unchanged): segmentation and the trapezoid predictor
score the rotation-invariant ``|a|-g`` residual; ZUPT keeps the
gravity-projected ``a_vert``. See ``docs/latex/main.tex`` §12--13.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import pandas as pd

from src.data.loader import resample_sensor_with_gaps
from pyramidElevatorDist.physics.barometric import pressure_to_altitude
from pyramidElevatorDist.physics.reconstruct_az import reconstruct_az
from pyramidElevatorDist.prediction.algorithms import (
    PREDICT_ALGORITHM_CONFIG, PredictAlgorithm, Predictor,
)
from pyramidElevatorDist.segmentation.algorithms.configTypes import (
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, TemplateMatchConfig,
)
from pyramidElevatorDist.segmentation.algorithms.segmenter import Segmenter
from pyramidElevatorDist.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E501
    detect as _detect,
)
from pyramidElevatorDist.segmentation.algorithms.accelerometer_only.template_match.fit_elevator_parameters.common import (  # noqa: E501
    _a_mag_minus_g, _estimate_fs_hz, _vertical_accel,
)


# --------------------------------------------------------------------------
# Constants (single source of truth; the façades re-export these)
# --------------------------------------------------------------------------

# Canonical uniform cadence the detector + Δh estimators are tuned for.
RESAMPLE_TARGET_HZ = 50

# Stationary-window length used to calibrate gravity around a ride.
PRE_POST_WINDOW_SEC = 5.0
PRE_POST_MIN_SEC = 1.0

# Algorithms exposed — short id -> enum. The first entry is the "primary"
# the UI sidebar list and PDF report default to.
_ACCEL_ALGO_MAP: dict[str, PredictAlgorithm] = {
    "trap": PredictAlgorithm.TRAPEZOID_ACCEL,
    "zupt": PredictAlgorithm.ZUPT_ACCEL,
}
_PRIMARY_ALGO_ID = "trap"

# Boutique hybrid signal policy: trapezoid + segmentation run on the
# rotation-invariant |a|-g; ZUPT keeps the gravity-projected a_vert because
# its quality is bounded by the double-integration drift model, not the
# matched-filter signal choice. See docs/latex/main.tex §12--13.
_SEGMENT_INPUT_SIGNAL = "a_mag_minus_g"
_PER_ALGO_OVERRIDES: dict[str, dict] = {
    "trap": {"input_signal": "a_mag_minus_g"},
    "zupt": {},
}


# --------------------------------------------------------------------------
# Input normalization
# --------------------------------------------------------------------------

def resample_acc(acc: pd.DataFrame, target_hz: int = RESAMPLE_TARGET_HZ):
    """Normalize an accelerometer trace onto a uniform ``target_hz`` grid.

    Gap-aware (splits on >1 s holes rather than interpolating across them)
    and time-correct (each output sample keeps its true timestamp, so
    ``segment`` ride coordinates line up with ``predict``). Returns the input
    unchanged when it is ``None``, lacks ``timestamp_ms``, or has < 2 rows.
    """
    if (acc is None or "timestamp_ms" not in getattr(acc, "columns", [])
            or len(acc) < 2):
        return acc
    resampled, _intervals = resample_sensor_with_gaps(acc, target_hz=target_hz)
    return resampled


# --------------------------------------------------------------------------
# Slicing helpers
# --------------------------------------------------------------------------

def _slice_acc(
    frame: pd.DataFrame, t0_ms: float, t_lo: float, t_hi: float,
) -> pd.DataFrame:
    # Inclusive on both ends — matches the segmenter's slicing convention so a
    # row whose timestamp lands exactly on a segment boundary is consistently
    # kept rather than silently dropped on uneven sampling. Works on any frame
    # carrying ``timestamp_ms`` (accelerometer or gyroscope).
    ts = frame["timestamp_ms"].astype(float).to_numpy()
    lo_ms = t0_ms + t_lo * 1000.0
    hi_ms = t0_ms + t_hi * 1000.0
    mask = (ts >= lo_ms) & (ts <= hi_ms)
    return frame.loc[mask].reset_index(drop=True)


def _slice_pre_post(
    frame: pd.DataFrame, t0_ms: float,
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
        pre_df = _slice_acc(frame, t0_ms, pre_lo, pre_hi)
        if pre_df.empty:
            pre_df = None
    post_df: Optional[pd.DataFrame] = None
    if post_hi - post_lo >= min_sec:
        post_df = _slice_acc(frame, t0_ms, post_lo, post_hi)
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
    """No-op compatibility shim (the in-process call already returns
    ``np.ndarray`` / real ``DetectConfig`` shapes, never JSON)."""
    return state


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------

def segment_cfg() -> "_detect.DetectConfig":
    """The boutique matched-filter detector config as a ``DetectConfig``.

    Built from the live segmentation ``config.json`` (key
    ``acc_template_match``) with the boutique ``|a|-g`` signal override, so
    tuned hyperparameters written by ``scripts/tune_hyperparameters.py`` take
    effect on the next call. Exposed for the editor detail panel
    (``findSegmentParameters``), which fits fresh trapezoids against it.
    """
    params = SEGMENT_ALGORITHM_CONFIG(
        algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
        overrides={"input_signal": _SEGMENT_INPUT_SIGNAL},
    ).load_params()
    return _detect.DetectConfig(**TemplateMatchConfig(**params).model_dump())


def segment(
    acc: pd.DataFrame,
    phone_model: str = "",
    include_state: bool = True,
    resample_hz: int | None = None,
    gyro: Optional[pd.DataFrame] = None,
    reconstruct: str = "none",
) -> tuple[list[dict], dict | None, float | None]:
    """Detect ride intervals — routed through :meth:`Segmenter.detect_raw`.

    Pure orchestration: it configures the dispatcher and returns
    ``(predictions, state, t0_ms)``; input-cadence normalization
    (``resample_hz``) is handled *inside* the ``Segmenter``. ``state`` is
    ``None`` when the detector produced nothing or ``include_state=False``.
    """
    cfg = SEGMENT_ALGORITHM_CONFIG(
        algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
        overrides={"input_signal": _SEGMENT_INPUT_SIGNAL},
        reconstruct=reconstruct,
        resample_hz=resample_hz,
    )
    predictions, state = Segmenter(cfg).detect_raw(
        acc, phone_model=phone_model, gyro=gyro,
    )
    t0_raw = float(state.get("t0_ms", float("nan"))) if state else float("nan")
    t0_ms: float | None = t0_raw if math.isfinite(t0_raw) else None
    return predictions, (state if (include_state and state) else None), t0_ms


def find_matching_prediction(
    predictions: list[dict], t_lo: float, t_hi: float,
) -> dict | None:
    """Best-overlap detector prediction for a user-edited interval.

    Returns ``None`` when there is no overlap — the user marked an interval
    the detector never proposed. Used by the interactive editor
    (``findSegmentParameters``) to reuse the detector's own fit when it exists.
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
# Display signal (UI overview / detail plots)
# --------------------------------------------------------------------------

def signal(
    acc: pd.DataFrame,
    gyro: Optional[pd.DataFrame] = None,
    reconstruct: str = "none",
    resample_hz: int | None = None,
) -> pd.DataFrame:
    """Whole-trace vertical-acceleration series for UI plots.

    Applies the same input-cadence normalization + gyro orientation
    reconstruction the dispatchers use, then returns the acceleration the UI
    draws — no detector ``state`` leaked. Columns:

    * ``timestamp_ms``
    * ``a_vert``  — gravity-projected vertical accel (m/s²), computed AFTER the
      gyro reconstruction: with ``reconstruct != "none"`` and a gyro stream
      this is the orientation-corrected "reconstructed a_z"; otherwise the
      plain vertical accel.
    * ``a_mag_g`` — ``|a| − g``, the rotation-invariant residual the detector
      matches on.

    Empty / column-less input → empty frame with the same columns.
    """
    cols = ["timestamp_ms", "a_vert", "a_mag_g"]
    if acc is None or len(acc) == 0 or "timestamp_ms" not in getattr(acc, "columns", []):
        return pd.DataFrame(columns=cols)
    if resample_hz:
        acc = resample_acc(acc, target_hz=resample_hz)
    acc = reconstruct_az(reconstruct, acc, gyro)
    ts = acc["timestamp_ms"].to_numpy(dtype=float)
    fs = _estimate_fs_hz(ts)
    ax = acc["x"].to_numpy(dtype=float)
    ay = acc["y"].to_numpy(dtype=float)
    az = acc["z"].to_numpy(dtype=float)
    return pd.DataFrame({
        "timestamp_ms": acc["timestamp_ms"].to_numpy(),
        "a_vert": _vertical_accel(ax, ay, az, fs),
        "a_mag_g": _a_mag_minus_g(ax, ay, az, fs),
    })


def barometric_altitude(
    prs: pd.DataFrame, temperature_c: float | None = None,
) -> pd.DataFrame:
    """Whole-trace barometric altitude — the ground-truth reference signal.

    ISA inversion of the pressure column (see ``pyramidElevatorDist.physics.barometric``), the
    same physics the barometer Δh estimator uses. Columns:

    * ``timestamp_ms``
    * ``altitude_m`` — metres above the ISA reference pressure.

    Empty / column-less input (or a frame without ``pressure``) → empty frame
    with those columns.
    """
    cols = ["timestamp_ms", "altitude_m"]
    if (prs is None or len(prs) == 0
            or "timestamp_ms" not in getattr(prs, "columns", [])
            or "pressure" not in getattr(prs, "columns", [])):
        return pd.DataFrame(columns=cols)
    alt = pressure_to_altitude(
        prs["pressure"].to_numpy(dtype=float), temperature_c=temperature_c,
    )
    return pd.DataFrame({
        "timestamp_ms": prs["timestamp_ms"].to_numpy(),
        "altitude_m": alt,
    })


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def predict(
    acc: pd.DataFrame,
    segments: Iterable[dict],
    phone_model: str = "",
    algorithms: list[str] | None = None,
    resample_hz: int | None = None,
    gyro: Optional[pd.DataFrame] = None,
    reconstruct: str = "none",
    prs: Optional[pd.DataFrame] = None,
) -> tuple[dict[str, list[dict]], str]:
    """Run the Δh estimators over a list of ride intervals via
    :meth:`Predictor.predict`.

    Pure orchestration: slices each ride window (and its pre/post gravity
    windows) from ``acc`` by relative seconds — which is cadence-independent —
    and hands them to :class:`Predictor`, which normalizes each window's
    sample rate itself (config ``resample_hz``). ``segments`` is an iterable of
    dicts with ``type``, ``start_s``, ``end_s`` (optionally
    ``trapezoid_override``). Neighbour clamping for the pre/post gravity windows
    uses each segment's position in the list.

    When a pressure frame ``prs`` (columns ``timestamp_ms``, ``pressure``) is
    supplied, a barometer Δh row is added under the ``"baro"`` key for each
    segment — the ground-truth reference. Its Δh keeps the measured sign (it is
    not forced to the ride's up/down label). Returns
    ``(rows_by_algo, primary_algo_id)``.
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
            reconstruct=reconstruct,
            resample_hz=resample_hz,
        ))
        for aid in chosen
    }
    # Barometer (ground-truth) estimator — only when a pressure stream is given.
    baro_predictor = (
        Predictor(PREDICT_ALGORITHM_CONFIG(
            algorithm=PredictAlgorithm.BAROMETER_HEIGHT_DIFF,
        ))
        if prs is not None else None
    )

    seg_starts = [float(s["start_s"]) for s in segs]
    seg_ends = [float(s["end_s"]) for s in segs]

    rows_by_algo: dict[str, list[dict]] = {aid: [] for aid in predictors}
    if baro_predictor is not None:
        rows_by_algo["baro"] = []
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
        # Matching gyroscope slices (only when a gyro stream is supplied).
        ride_gyro = pre_gyro = post_gyro = None
        if gyro is not None:
            ride_gyro = _slice_acc(gyro, t0_ms, t_lo, t_hi)
            pre_gyro, post_gyro = _slice_pre_post(
                gyro, t0_ms, t_lo, t_hi, prev_hi, next_lo,
            )
        base = {
            "segment":    int(pos),
            "type":       rt,
            "start_s":    t_lo,
            "end_s":      t_hi,
            "duration_s": t_hi - t_lo,
        }
        # Barometer ground-truth row — pressure-only, so it is computed
        # independently of the accelerometer slice (a ride with no ACC samples
        # can still have a GT Δh). Keeps the measured sign.
        if baro_predictor is not None:
            prs_slice = _slice_acc(prs, t0_ms, t_lo, t_hi)
            if prs_slice.empty or len(prs_slice) < 2:
                rows_by_algo["baro"].append(_empty_pred_row(base, "no_pressure"))
            else:
                try:
                    out = baro_predictor.predict(prs_slice)
                    dh = float(out.height_diff)
                    rows_by_algo["baro"].append({
                        **base,
                        "delta_height_m": dh,
                        "abs_height_m":   abs(dh),
                        "accepted":       bool(out.accepted),
                        "quality_score":  float(out.quality_score),
                        "reject_reason":  str(out.reject_reason or ""),
                        "ci_half_width":  (float(out.ci_half_width)
                                           if math.isfinite(out.ci_half_width)
                                           else float("nan")),
                        "meta":           dict(out.meta) if out.meta else {},
                    })
                except Exception as e:  # noqa: BLE001 — surface as a row
                    rows_by_algo["baro"].append(
                        _empty_pred_row(base, f"{type(e).__name__}: {e}"),
                    )
        if slice_df.empty:
            for aid in predictors:
                rows_by_algo[aid].append(_empty_pred_row(base, "empty_slice"))
            continue
        # Manual trapezoid override — only the trapezoid estimator consumes it;
        # other algorithms ignore the kwarg via the Predictor dispatcher.
        seg_override = seg.get("trapezoid_override")
        for aid, predictor in predictors.items():
            try:
                out = predictor.predict(
                    slice_df,
                    phone_model=phone_model,
                    pre=pre_df, post=post_df,
                    trapezoid_override=seg_override,
                    gyro=ride_gyro, pre_gyro=pre_gyro, post_gyro=post_gyro,
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
