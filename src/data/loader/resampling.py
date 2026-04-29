"""Uniform-50 Hz resampling for sensor frames.

Algorithms downstream (segmentation + prediction) all assume a 50 Hz uniform
sample grid. Real device logs are jittery, occasionally barometer-only at
25 Hz, sometimes burst into 100 Hz, and on rare occasions drop out entirely
for several seconds. This module is the loader-side fix that hides all of
that from the algorithms.

Pipeline applied to each sensor frame:

1. **Gap-split.** Anywhere `dt > gap_threshold_s` (default 1 s) is treated
   as a true measurement hole — there is no realistic phone configuration
   where the native rate is below 1 Hz. The frame is split on those holes
   into independent contiguous chunks.

2. **Even out at the chunk's natural rate.** For each chunk, estimate the
   device's native rate (median dt → snap to a common rate like 25/50/100
   Hz when within 5 %). Linearly interpolate the chunk onto a uniform grid
   at that natural rate. This step removes jitter while introducing the
   minimum possible number of synthetic samples (the new grid coincides
   with the original samples in the limit of zero jitter).

3. **Resample to 50 Hz.** Linear-interpolate the now-uniform chunk onto a
   uniform 50 Hz grid. If the natural rate is already 50 Hz, steps 2 and 3
   collapse into one resample.

4. **Concatenate** chunks back together. Inter-chunk gaps are preserved —
   we do not invent data across a >1 s hole. Within each chunk the rate is
   exactly 50 Hz.

Non-numeric columns (e.g. `exp_name`) are forward-filled by nearest-prior
sample. The `timestamp_ms` column is rebuilt as int64 ms on the new grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Public defaults. Algorithms downstream ASSUME 50 Hz uniform.
TARGET_HZ: float = 50.0

# Anything above this dt is considered a real measurement hole rather than
# jitter. 1 s is the largest plausible nominal sample period (1 Hz is the
# slowest configured rate on any phone in this dataset).
GAP_THRESHOLD_S: float = 1.0

# Common nominal device rates we snap to when the median is within 5 %.
# Snapping avoids drifting estimates like 49.9 Hz or 25.1 Hz that would
# add unnecessary interpolated points.
_COMMON_RATES_HZ: tuple[float, ...] = (1, 2, 5, 10, 20, 25, 40, 50, 100, 200, 400)

# Sensors that do not need resampling — slow, irregular, or never used by an
# algorithm that requires uniform sampling. GPS is the only one today.
_SKIP_SENSORS: frozenset[str] = frozenset({"GPS"})


def _estimate_natural_hz(t_ms: np.ndarray) -> float:
    """Return the device's natural rate (Hz) from a chunk's timestamps.

    Uses the median inter-sample dt, then snaps to the nearest common
    device rate when within 5 %. Falls back to the raw rounded value
    otherwise.
    """
    if t_ms.size < 2:
        return TARGET_HZ
    dt_ms = np.diff(t_ms)
    dt_ms = dt_ms[dt_ms > 0]
    if dt_ms.size == 0:
        return TARGET_HZ
    median_dt = float(np.median(dt_ms))
    nominal = 1000.0 / median_dt
    for c in _COMMON_RATES_HZ:
        if abs(nominal - c) / c <= 0.05:
            return float(c)
    return float(round(nominal, 2))


def _split_on_gaps(
    df: pd.DataFrame,
    time_col: str = "timestamp_ms",
    gap_threshold_ms: float = GAP_THRESHOLD_S * 1000.0,
) -> list[pd.DataFrame]:
    """Split `df` (sorted by `time_col`) into chunks at any dt > threshold."""
    if df.empty:
        return []
    if len(df) < 2:
        return [df.reset_index(drop=True)]
    t = df[time_col].to_numpy(dtype=float)
    dt = np.diff(t)
    breaks = np.where(dt > gap_threshold_ms)[0]
    if breaks.size == 0:
        return [df.reset_index(drop=True)]
    chunks: list[pd.DataFrame] = []
    start = 0
    for b in breaks:
        chunks.append(df.iloc[start:int(b) + 1].reset_index(drop=True))
        start = int(b) + 1
    chunks.append(df.iloc[start:].reset_index(drop=True))
    return chunks


def _resample_uniform(
    df: pd.DataFrame,
    target_hz: float,
    time_col: str = "timestamp_ms",
) -> pd.DataFrame:
    """Linear-interpolate `df` onto a uniform `target_hz` grid.

    Numeric columns are linearly interpolated, NaN-aware (NaN samples are
    excluded from the interpolation source). Non-numeric columns are
    forward-filled from the nearest-prior original sample. The output
    `timestamp_ms` is int64 ms on the new grid; `df[time_col].iloc[0]`
    anchors the first sample.

    For a chunk with <2 samples or with a degenerate time range, returns
    the input unchanged (resampling is undefined).
    """
    if df.empty or len(df) < 2:
        return df.reset_index(drop=True).copy()

    t = df[time_col].to_numpy(dtype=float)
    if t[-1] <= t[0]:
        return df.reset_index(drop=True).copy()

    period_ms = 1000.0 / float(target_hz)
    n_samples = int(np.floor((t[-1] - t[0]) / period_ms)) + 1
    if n_samples < 2:
        return df.reset_index(drop=True).copy()
    new_t = t[0] + np.arange(n_samples, dtype=float) * period_ms

    out: dict[str, np.ndarray] = {time_col: np.rint(new_t).astype("int64")}
    fwd_idx: np.ndarray | None = None  # lazy: only build if needed
    for col in df.columns:
        if col == time_col:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            v = s.to_numpy(dtype=float)
            mask = np.isfinite(v)
            if mask.sum() < 2:
                out[col] = np.full(n_samples, np.nan, dtype=float)
            else:
                out[col] = np.interp(new_t, t[mask], v[mask])
        else:
            if fwd_idx is None:
                fwd_idx = np.clip(
                    np.searchsorted(t, new_t, side="right") - 1, 0, len(t) - 1,
                )
            out[col] = s.to_numpy()[fwd_idx]
    return pd.DataFrame(out)


def is_uniform_at_hz(
    df: pd.DataFrame,
    target_hz: float = TARGET_HZ,
    time_col: str = "timestamp_ms",
    gap_threshold_s: float = GAP_THRESHOLD_S,
    tol_ms: float = 0.5,
) -> bool:
    """True iff every non-gap chunk is already uniform at `target_hz`.

    Used to short-circuit resampling on already-prepared cached data. A
    chunk is considered uniform when every dt within it is within `tol_ms`
    of the target period (default 1000/50 = 20 ms, ±0.5 ms).
    """
    if df.empty or len(df) < 2:
        return True
    target_period_ms = 1000.0 / target_hz
    for chunk in _split_on_gaps(df, time_col, gap_threshold_s * 1000.0):
        if len(chunk) < 2:
            continue
        dt = np.diff(chunk[time_col].to_numpy(dtype=float))
        if np.any(np.abs(dt - target_period_ms) > tol_ms):
            return False
    return True


def prepare_uniform_50hz(
    df: pd.DataFrame,
    time_col: str = "timestamp_ms",
    gap_threshold_s: float = GAP_THRESHOLD_S,
    target_hz: float = TARGET_HZ,
) -> pd.DataFrame:
    """Apply gap-split → even-out → 50 Hz resample to a sensor frame.

    See module docstring for the full algorithm. Idempotent: a frame that
    is already uniform at `target_hz` (with no >1 s gaps inside chunks)
    is returned unchanged.
    """
    if df.empty:
        return df.copy()
    if time_col not in df.columns:
        return df.copy()
    df = df.sort_values(time_col, kind="stable").reset_index(drop=True)

    if is_uniform_at_hz(df, target_hz=target_hz, time_col=time_col,
                       gap_threshold_s=gap_threshold_s):
        return df

    chunks = _split_on_gaps(df, time_col, gap_threshold_s * 1000.0)
    out_chunks: list[pd.DataFrame] = []
    for chunk in chunks:
        if len(chunk) < 2:
            out_chunks.append(chunk)
            continue
        t_chunk = chunk[time_col].to_numpy(dtype=float)
        natural_hz = _estimate_natural_hz(t_chunk)
        # Step 1: jitter-fix on the chunk's natural rate.
        if abs(natural_hz - target_hz) > 1e-6:
            uniform_natural = _resample_uniform(chunk, natural_hz, time_col)
            # Step 2: now upsample/downsample to the 50 Hz target.
            uniform_target = _resample_uniform(uniform_natural, target_hz, time_col)
        else:
            uniform_target = _resample_uniform(chunk, target_hz, time_col)
        out_chunks.append(uniform_target)

    if not out_chunks:
        return df.iloc[0:0].copy()
    out = pd.concat(out_chunks, ignore_index=True)
    return out


def prepare_sensors_uniform_50hz(
    sensors: dict[str, pd.DataFrame],
    skip: frozenset[str] = _SKIP_SENSORS,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Apply :func:`prepare_uniform_50hz` to every sensor frame in a dict.

    Sensors listed in `skip` (default: GPS) pass through untouched — they
    are sampled too irregularly or too slowly for uniform resampling to
    produce meaningful data, and no algorithm consumes them.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, df in sensors.items():
        if df is None or df.empty or name in skip:
            out[name] = df
            continue
        out[name] = prepare_uniform_50hz(df, **kwargs)
    return out
