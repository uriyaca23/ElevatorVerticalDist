"""Shared accelerometer preprocessing utilities.

Bare-numpy helpers used across segmentation and prediction stages:

* :func:`estimate_gravity_stationary` — robust gravity vector from a
  stationary (pre/post-ride) window.
* :func:`vertical_accel_projected` — signed vertical acceleration by
  projection onto a known gravity vector.
* :func:`vertical_accel_magnitude` — rotation-invariant ``|a| − g``
  fallback when gravity is not known.
* :func:`compute_a_vert` — convenience: estimate gravity from the
  passed samples themselves and project in one call.
* :func:`compute_velocity` — cumulative-sum integration with DC removal.
* :func:`zupt_integrate` — double-integrate vertical acceleration under
  ``v(start) = v(end) = 0`` with linear drift correction.
* :func:`lowpass` — ride-band low-pass (cutoff = 0.3 Hz) to suppress
  walking-cadence leakage while preserving ride-scale dynamics. For a
  general-purpose Butterworth filter, see
  :func:`src.utils.signal_processing.butter_lowpass`.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


VELOCITY_LPF_HZ = 0.3  # cut walking band; keep elevator ride dynamics


def lowpass(x: np.ndarray, fs: float, cutoff_hz: float = VELOCITY_LPF_HZ) -> np.ndarray:
    nyq = 0.5 * fs
    sos = butter(4, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, x)


def estimate_gravity_stationary(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    fs: float = 50.0, window_sec: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    """Robust gravity vector from a (mostly) stationary buffer.

    Splits the samples into ``window_sec`` windows, takes per-axis means,
    then returns the per-axis **median** as the gravity vector — more
    robust to brief disturbances than a single mean. ``stability`` is
    the joint std of per-window component estimates; use it to decide
    whether the vector can be trusted.

    Returns ``(gvec, |g|, stability)``. For short buffers returns
    ``([0, 0, 9.81], 9.81, +inf)``.

    Only meaningful on stationary input — during motion the samples
    contain both gravity and ride dynamics, so their mean is biased.
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

    gvec = np.array([
        float(np.median(gx_list)),
        float(np.median(gy_list)),
        float(np.median(gz_list)),
    ])
    g_mag = float(np.linalg.norm(gvec))
    stability = float(
        np.sqrt(np.var(gx_list) + np.var(gy_list) + np.var(gz_list))
    )
    return gvec, g_mag, stability


def vertical_accel_magnitude(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    g_ref: float | None = None,
) -> np.ndarray:
    """``|a(t)| − g`` where ``g`` is the median magnitude (or ``g_ref``).

    Rotation-invariant, but loses sign on the orientation frame — the
    caller must recover direction elsewhere (ZUPT drift sign, first-pulse
    sign, etc).
    """
    mag = np.sqrt(ax * ax + ay * ay + az * az)
    g = float(np.median(mag)) if g_ref is None else float(g_ref)
    return mag - g


def vertical_accel_projected(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    gvec: np.ndarray,
) -> np.ndarray:
    """Signed vertical acceleration via projection onto ``gvec``.

    Returns ``(a · ĝ) − |g|`` where ``ĝ = g / |g|``. Falls back to
    :func:`vertical_accel_magnitude` if ``gvec`` is degenerate.
    """
    g_mag = float(np.linalg.norm(gvec))
    if not np.isfinite(g_mag) or g_mag < 1e-6:
        return vertical_accel_magnitude(ax, ay, az)
    g_hat = gvec / g_mag
    return ax * g_hat[0] + ay * g_hat[1] + az * g_hat[2] - g_mag


def compute_a_vert(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float) -> np.ndarray:
    """Self-gravity-referenced vertical acceleration.

    Estimates gravity from the passed samples themselves (treating the
    full buffer as quasi-stationary) and projects onto it. Suitable for
    whole-session signals where no dedicated pre/post window is
    available; prefer :func:`vertical_accel_projected` with an
    externally-estimated ``gvec`` when one exists.
    """
    gvec, _, _ = estimate_gravity_stationary(ax, ay, az, fs=fs, window_sec=0.5)
    return vertical_accel_projected(ax, ay, az, gvec)


def compute_velocity(a_vert: np.ndarray, fs: float) -> np.ndarray:
    """Global integration of DC-removed a_vert."""
    return np.cumsum(a_vert - float(np.mean(a_vert))) / fs


def zupt_integrate(
    a_vert: np.ndarray, t_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Double-integrate vertical acceleration under ZUPT boundary conditions.

    * ``v(start) = 0``, ``v(end) = 0`` (linear-drift correction enforces
      the terminal constraint).
    * Returns ``(position, velocity)`` in meters / (m/s), with
      ``position[0] = 0``.

    Per-step ``dt`` is taken from ``t_sec`` so variable sampling rates
    are respected.
    """
    n = len(a_vert)
    if n < 2:
        return np.zeros(n), np.zeros(n)

    dt = np.diff(t_sec, prepend=t_sec[0])
    dt[0] = dt[1] if n > 1 else 0.01

    vel = np.cumsum(a_vert * dt)
    drift = vel[-1]
    vel -= np.linspace(0.0, drift, n)

    pos = np.cumsum(vel * dt)
    return pos, vel
