"""Gap-aware sensor resampling onto a uniform target grid.

Pure pandas/numpy/scipy — no dependency on any data-loading layer. This is
the canonical implementation; ``src.data.loader`` re-exports these names for
the application side.
"""
from __future__ import annotations

import pandas as pd

# Target rate that every public load goes through. 50 Hz is fast enough for
# the elevator dynamics we care about (≪ 1 Hz signal bandwidth) and keeps the
# accelerometer / barometer on the same grid for downstream code.
TARGET_SAMPLE_RATE_HZ = 50

# Gap detection: any spacing between consecutive samples larger than
# 1/THRESHOLD_FREQUENCY_HZ seconds is treated as a "no data" gap that splits
# the recording into separate valid intervals. Below 1 Hz we effectively
# have no signal.
THRESHOLD_FREQUENCY_HZ = 1.0
GAP_THRESHOLD_S = 1.0 / THRESHOLD_FREQUENCY_HZ


def _is_uniformly_sampled(ts, rel_tol: float = 0.05) -> bool:
    """True when timestamp diffs have low coefficient of variation.

    `rel_tol` is the max allowed ``std(dt) / median(dt)`` ratio. ~5 %
    tolerates the typical phone-sensor jitter while flagging traces with
    dropped samples or burst-mode batching as non-uniform.
    """
    import numpy as np
    if len(ts) < 3:
        return True
    d = np.diff(np.asarray(ts, dtype=np.float64))
    md = float(np.median(d))
    if md <= 0 or len(d) == 0:
        return False
    return float(np.std(d)) / md <= rel_tol


def _detect_valid_intervals(
    ts_ms, gap_threshold_s: float = GAP_THRESHOLD_S,
) -> list[tuple[int, int]]:
    """Split a timestamp array into contiguous valid intervals.

    A "gap" is any spacing between consecutive samples greater than
    ``gap_threshold_s`` seconds — below the threshold frequency the data
    is too sparse to recover via interpolation and is treated as missing.

    Returns inclusive `[(start_ms, end_ms), ...]` covering the timestamp
    array minus the gaps. Empty input → ``[]``; a single sample → the
    span ``[(t, t)]`` (the resampler will short-circuit such cases).
    """
    import numpy as np
    ts = np.asarray(ts_ms, dtype=np.int64)
    if ts.size == 0:
        return []
    if ts.size == 1:
        return [(int(ts[0]), int(ts[0]))]
    diffs = np.diff(ts)
    gap_threshold_ms = int(round(gap_threshold_s * 1000.0))
    breaks = np.where(diffs > gap_threshold_ms)[0]
    if breaks.size == 0:
        return [(int(ts[0]), int(ts[-1]))]
    intervals: list[tuple[int, int]] = []
    start_idx = 0
    for b in breaks:
        intervals.append((int(ts[start_idx]), int(ts[b])))
        start_idx = b + 1
    intervals.append((int(ts[start_idx]), int(ts[-1])))
    return intervals


def _resample_sensor_to_hz(
    df: pd.DataFrame, target_hz: int = TARGET_SAMPLE_RATE_HZ,
) -> pd.DataFrame:
    """Resample one sensor frame onto a uniform `target_hz` grid.

    Numeric columns: ``scipy.signal.resample_poly`` (with its built-in
    anti-alias filter) when input timestamps are uniformly sampled —
    otherwise ``np.interp``. Non-numeric columns (e.g. the ``exp_name``
    tag) are propagated by nearest-neighbor lookup. The output spans
    ``[t0, t1]`` of the input and stays on int64 epoch-ms timestamps so
    downstream timestamp slicing is unaffected.

    No-op for empty / single-row frames or frames missing
    ``timestamp_ms``.
    """
    import numpy as np
    if df is None or df.empty or "timestamp_ms" not in df.columns:
        return df
    df = (df.sort_values("timestamp_ms")
            .drop_duplicates(subset="timestamp_ms")
            .reset_index(drop=True))
    if len(df) < 2:
        return df

    ts = df["timestamp_ms"].to_numpy(dtype=np.int64)
    t0, t1 = int(ts[0]), int(ts[-1])
    period_ms = 1000.0 / target_hz
    n = int(np.floor((t1 - t0) / period_ms)) + 1
    if n < 2:
        return df
    new_ts = np.round(t0 + np.arange(n) * period_ms).astype(np.int64)

    value_cols = [
        c for c in df.columns
        if c != "timestamp_ms" and pd.api.types.is_numeric_dtype(df[c])
    ]
    other_cols = [
        c for c in df.columns
        if c != "timestamp_ms" and c not in value_cols
    ]

    out = pd.DataFrame({"timestamp_ms": new_ts})

    if _is_uniformly_sampled(ts) and value_cols:
        from fractions import Fraction
        from scipy.signal import resample_poly
        median_dt_ms = float(np.median(np.diff(ts.astype(np.float64))))
        # Anti-aliased rate change via resample_poly, then RE-MAP its output to
        # the TRUE timestamps before placing on the uniform grid.
        #
        # resample_poly works purely in sample-count space — it assumes the
        # input is perfectly evenly spaced and discards the timestamps. Two
        # things break that assumption here: (1) the true rate is usually
        # non-integer (52.632 Hz), and (2) real phone ACC jitters between e.g.
        # 19 ms and 20 ms samples. Either way the output sample index no longer
        # corresponds linearly to wall-clock time, so naively stamping y onto a
        # uniform grid warps time — features drift by many seconds over a long
        # recording (small at the start, large at the end). We use up/down to
        # approximate target_hz / true_src_hz for the filter, then convert each
        # output sample back to its real time via the actual timestamps and
        # interpolate onto ``new_ts``. This keeps anti-aliasing AND time.
        src_hz = 1000.0 / median_dt_ms
        ratio = Fraction(target_hz / src_hz).limit_denominator(1000)
        up, down = ratio.numerator, ratio.denominator
        if up < 1 or down < 1:  # degenerate rate; fall back to integer ratio
            from math import gcd
            isrc = max(1, int(round(src_hz)))
            g = gcd(target_hz, isrc)
            up, down = target_hz // g, isrc // g
        ts_f = ts.astype(np.float64)
        src_idx = np.arange(len(df), dtype=np.float64)
        new_ts_f = new_ts.astype(np.float64)
        for c in value_cols:
            x = df[c].to_numpy(dtype=float)
            if np.isnan(x).all():
                out[c] = np.nan
                continue
            if np.isnan(x).any():
                # resample_poly's filter would smear NaNs across the
                # signal; fill them with linear interp first.
                idx = np.arange(len(x))
                m = ~np.isnan(x)
                x = np.interp(idx, idx[m], x[m])
            y = resample_poly(x, up=up, down=down)
            # Output sample k corresponds to input fractional index k*down/up;
            # map that to wall-clock time through the real timestamps, then
            # resample onto the uniform target grid at true times.
            t_of_y = np.interp(
                np.arange(len(y)) * (down / up), src_idx, ts_f,
            )
            out[c] = np.interp(new_ts_f, t_of_y, y)
    else:
        for c in value_cols:
            x = df[c].to_numpy(dtype=float)
            m = ~np.isnan(x)
            if not m.any():
                out[c] = np.nan
                continue
            out[c] = np.interp(
                new_ts.astype(np.float64),
                ts[m].astype(np.float64),
                x[m],
            )

    if other_cols:
        idx = np.searchsorted(ts, new_ts)
        idx = np.clip(idx, 0, len(ts) - 1)
        for c in other_cols:
            out[c] = df[c].to_numpy()[idx]

    return out


def _resample_sensor_with_gaps(
    df: pd.DataFrame,
    target_hz: int = TARGET_SAMPLE_RATE_HZ,
    gap_threshold_s: float = GAP_THRESHOLD_S,
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Gap-aware variant of :func:`_resample_sensor_to_hz`.

    Splits ``df`` on every consecutive-sample gap larger than
    ``gap_threshold_s`` seconds, resamples each contiguous valid interval
    independently to ``target_hz``, and concatenates the results. The
    returned DataFrame has *no rows* in the gap regions — downstream
    consumers see a clean 50 Hz signal punctuated by holes.

    Returns ``(resampled_df, valid_intervals)`` where ``valid_intervals``
    is a list of ``[start_ms, end_ms]`` (inclusive, on the original raw
    timestamp scale).
    """
    import numpy as np
    if df is None or df.empty or "timestamp_ms" not in df.columns:
        return df, []
    df = (df.sort_values("timestamp_ms")
            .drop_duplicates(subset="timestamp_ms")
            .reset_index(drop=True))
    if len(df) < 2:
        ts0 = int(df["timestamp_ms"].iloc[0]) if len(df) else 0
        return df, [(ts0, ts0)] if len(df) else []

    ts = df["timestamp_ms"].to_numpy(dtype=np.int64)
    intervals = _detect_valid_intervals(ts, gap_threshold_s=gap_threshold_s)
    if not intervals:
        return df.iloc[0:0].copy(), []

    resampled_chunks: list[pd.DataFrame] = []
    for s_ms, e_ms in intervals:
        mask = (ts >= s_ms) & (ts <= e_ms)
        chunk = df.loc[mask]
        if chunk.empty:
            continue
        chunk_resampled = _resample_sensor_to_hz(
            chunk.reset_index(drop=True), target_hz=target_hz,
        )
        if chunk_resampled is None or chunk_resampled.empty:
            continue
        resampled_chunks.append(chunk_resampled)

    if not resampled_chunks:
        return df.iloc[0:0].copy(), intervals
    out = pd.concat(resampled_chunks, ignore_index=True)
    return out, intervals


# Public aliases (the loader re-exported the underscore names under these).
detect_valid_intervals = _detect_valid_intervals
resample_sensor_to_hz = _resample_sensor_to_hz
resample_sensor_with_gaps = _resample_sensor_with_gaps
