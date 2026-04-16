"""Detect elevator-active segments from a height (altitude) time series.

Adapted from the ground-truth tagging logic in
`src/legacy/old_tagging/auto_tag_elevators.py`: threshold the smoothed
vertical velocity, then keep runs that are long enough and produce a
meaningful height change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# import via importlib to avoid the "class" reserved-keyword module name
import importlib
_config_mod = importlib.import_module("src.algorithms.segmentation_algorithms.class")
PressureFilterConfig = _config_mod.PressureFilterConfig


def filter_height(
    data: pd.DataFrame,
    config: PressureFilterConfig,
) -> np.ndarray:
    """Return the low-pass filtered height signal actually used by the
    segmenter. Exposed so plots can show the same signal that drives the
    decision boundary."""
    t = np.asarray(data[config.time_col].to_numpy(), dtype=float)
    z = np.asarray(data[config.height_col].to_numpy(), dtype=float)
    if len(t) < 2:
        return z
    dt = np.diff(t)
    fs = 1.0 / np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    lp_win = max(1, int(round(config.height_lowpass_sec * fs)))
    return pd.Series(z).rolling(window=lp_win, center=True, min_periods=1).mean().to_numpy()


def detect_elevator_segments_from_height(
    data: pd.DataFrame,
    config: PressureFilterConfig,
) -> pd.DataFrame:
    """Return a DataFrame with columns ``start_ci``, ``end_ci``,
    ``probability_ci`` (each a ``(low, high)`` tuple), ``duration``, ``type``.

    Point estimates are implied as the center of each CI. The pressure
    algorithm is deterministic, so all CIs collapse to zero-width.
    """
    columns = ["start_ci", "end_ci", "duration", "type", "probability_ci"]
    t = np.asarray(data[config.time_col].to_numpy(), dtype=float)
    z = np.asarray(data[config.height_col].to_numpy(), dtype=float)
    if len(t) < 2:
        return pd.DataFrame(columns=columns)

    dt = np.diff(t)
    fs = 1.0 / np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    lp_win = max(1, int(round(config.height_lowpass_sec * fs)))
    z_lp = pd.Series(z).rolling(window=lp_win, center=True, min_periods=1).mean().to_numpy()
    vz = np.zeros_like(t)
    vz[1:] = np.diff(z_lp) / np.where(dt > 0, dt, np.nan)
    vz = np.nan_to_num(vz, nan=0.0)

    win = max(1, int(round(config.smooth_window_sec * fs)))
    kernel = np.ones(win) / win
    vz_smooth = np.convolve(vz, kernel, mode="same")

    is_moving = np.abs(vz_smooth) > config.velocity_threshold
    edges = np.diff(is_moving.astype(int))
    starts = (np.where(edges == 1)[0] + 1).tolist()
    ends = (np.where(edges == -1)[0] + 1).tolist()
    if is_moving[0]:
        starts.insert(0, 0)
    if is_moving[-1]:
        ends.append(len(t) - 1)

    raw: list[tuple[int, int, int]] = []
    for s, e in zip(starts, ends):
        sign = 1 if np.mean(vz_smooth[s:e + 1]) >= 0 else -1
        raw.append((s, e, sign))

    merged: list[tuple[int, int, int]] = []
    last_sign: int | None = None
    for s, e, sign in raw:
        if (
            merged
            and sign == last_sign
            and t[s] - t[merged[-1][1]] < config.merge_gap_sec
        ):
            merged[-1] = (merged[-1][0], e, sign)
        else:
            merged.append((s, e, sign))
            last_sign = sign

    rows: list[dict] = []
    for s, e, sign in merged:
        if t[e] - t[s] <= config.min_duration_sec:
            continue
        if abs(z[e] - z[s]) <= config.min_height_diff_m:
            continue
        start_t = max(t[0], t[s] - config.pad_sec)
        end_t = min(t[-1], t[e] + config.pad_sec)
        rows.append({
            "start_ci": (float(start_t), float(start_t)),
            "end_ci": (float(end_t), float(end_t)),
            "duration": float(end_t - start_t),
            "type": "up" if sign > 0 else "down",
            "probability_ci": (1.0, 1.0),
        })

    return pd.DataFrame(rows, columns=columns)
