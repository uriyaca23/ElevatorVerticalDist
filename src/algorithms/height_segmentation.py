"""Detect elevator-active segments from a height (altitude) time series.

Adapted from the ground-truth tagging logic in
`src/legacy/old_tagging/auto_tag_elevators.py`: threshold the smoothed
vertical velocity, then keep runs that are long enough and produce a
meaningful height change.
"""

from __future__ import annotations

import numpy as np


def detect_elevator_segments_from_height(
    t_sec: np.ndarray,
    height_m: np.ndarray,
    velocity_threshold: float = 0.2,   # m/s
    smooth_window_sec: float = 1.0,
    min_duration_sec: float = 3.0,
    min_height_diff_m: float = 2.0,
    pad_sec: float = 1.0,
) -> list[tuple[float, float]]:
    """Return list of (start_time, end_time) seconds where the user is in an
    actively-moving elevator, based on barometric/GT height.
    """
    t = np.asarray(t_sec, dtype=float)
    z = np.asarray(height_m, dtype=float)
    if len(t) < 2:
        return []

    dt = np.diff(t)
    vz = np.zeros_like(t)
    vz[1:] = np.diff(z) / np.where(dt > 0, dt, np.nan)
    vz = np.nan_to_num(vz, nan=0.0)

    fs = 1.0 / np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    win = max(1, int(round(smooth_window_sec * fs)))
    kernel = np.ones(win) / win
    vz_smooth = np.convolve(vz, kernel, mode="same")

    is_moving = np.abs(vz_smooth) > velocity_threshold
    edges = np.diff(is_moving.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if is_moving[0]:
        starts = np.insert(starts, 0, 0)
    if is_moving[-1]:
        ends = np.append(ends, len(t) - 1)

    segments: list[tuple[float, float]] = []
    for s, e in zip(starts, ends):
        if t[e] - t[s] <= min_duration_sec:
            continue
        if abs(z[e] - z[s]) <= min_height_diff_m:
            continue
        start_t = max(t[0], t[s] - pad_sec)
        end_t = min(t[-1], t[e] + pad_sec)
        segments.append((float(start_t), float(end_t)))

    return segments
