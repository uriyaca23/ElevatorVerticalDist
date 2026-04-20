"""Shared accelerometer-only helpers used by both prediction algorithms.

Every function here works on bare numpy arrays so we can be explicit
about shapes and not care whether the caller is handing us a
``pd.DataFrame`` column or a raw float buffer.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass(data: np.ndarray, fs: float, cutoff: float = 3.0,
                   order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass. Returns ``data`` unchanged for
    pathologically short segments or when ``cutoff >= Nyquist``.
    """
    if len(data) < 15:
        return data.copy()
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return data.copy()
    b, a = butter(order, cutoff / nyq, btype="low")
    try:
        return filtfilt(b, a, data)
    except Exception:
        return data.copy()


def estimate_gravity_stationary(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    fs: float = 50.0, window_sec: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    """Robust gravity vector estimate from a (mostly) stationary segment.

    Splits the buffer into ``window_sec`` windows, takes per-axis means,
    then returns the per-axis median as the gravity vector. Stability
    (std of per-window means) is returned so the caller can decide how
    much to trust the vector.

    Returns ``(gvec, g_mag, stability)``. When the buffer is too short
    it returns ``([0, 0, 9.81], 9.81, +inf)``.
    """
    n = len(ax)
    if n < 10:
        return np.array([0.0, 0.0, 9.81]), 9.81, float("inf")

    win = max(10, int(fs * window_sec))
    n_win = max(1, n // win)

    gx_list, gy_list, gz_list = [], [], []
    for i in range(n_win):
        s = i * win
        e = min(s + win, n)
        gx_list.append(np.mean(ax[s:e]))
        gy_list.append(np.mean(ay[s:e]))
        gz_list.append(np.mean(az[s:e]))

    gx = float(np.median(gx_list))
    gy = float(np.median(gy_list))
    gz = float(np.median(gz_list))
    gvec = np.array([gx, gy, gz])
    g_mag = float(np.linalg.norm(gvec))
    stability = float(
        np.sqrt(np.var(gx_list) + np.var(gy_list) + np.var(gz_list))
    )
    return gvec, g_mag, stability


def vertical_accel_magnitude(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    g_ref: float | None = None,
) -> np.ndarray:
    """``|a(t)| - g`` where ``g`` is the pre/ride-average magnitude.

    Rotation-invariant by construction but loses sign on the
    orientation frame, so the caller has to recover direction
    elsewhere (e.g., from ZUPT drift sign, or first-pulse sign).
    """
    mag = np.sqrt(ax * ax + ay * ay + az * az)
    g = float(np.median(mag)) if g_ref is None else float(g_ref)
    return mag - g


def vertical_accel_projected(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    gvec: np.ndarray,
) -> np.ndarray:
    """Signed vertical acceleration via projection onto a reference
    gravity vector (from pre/post stationary windows).

    Returns ``(a · ĝ) − |g|`` where ``ĝ = g / |g|``. If ``|g|`` is
    degenerate, falls back to magnitude-based vertical accel.
    """
    g_mag = float(np.linalg.norm(gvec))
    if not np.isfinite(g_mag) or g_mag < 1e-6:
        return vertical_accel_magnitude(ax, ay, az)
    g_hat = gvec / g_mag
    return ax * g_hat[0] + ay * g_hat[1] + az * g_hat[2] - g_mag


def zupt_integrate(
    a_vert: np.ndarray, t_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Double-integrate vertical acceleration under ZUPT boundary conditions.

    * v(start) = 0, v(end) = 0 (linear-drift-correction enforces the
      terminal constraint).
    * Returns ``(position, velocity)`` both in meters / (m/s), with
      position starting at 0 at the first sample.

    Variable sampling rates are respected via the actual ``dt`` per
    step; phones ship ~50–200 Hz depending on model but the
    inter-sample gap is rarely perfectly uniform.
    """
    n = len(a_vert)
    if n < 2:
        return np.zeros(n), np.zeros(n)

    dt = np.diff(t_sec, prepend=t_sec[0])
    dt[0] = dt[1] if n > 1 else 0.01

    vel = np.cumsum(a_vert * dt)
    # Enforce v(end) = 0 via a linear drift correction
    drift = vel[-1]
    vel -= np.linspace(0.0, drift, n)

    pos = np.cumsum(vel * dt)
    return pos, vel
