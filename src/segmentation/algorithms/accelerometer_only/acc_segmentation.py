"""Shared accelerometer preprocessing utilities.

Provides the gravity-projection front-end (`_compute_a_vert`), cumulative
velocity integration (`compute_velocity`), and the ride-band low-pass
(`lowpass`) used by the template-match detector.
"""

from __future__ import annotations

import importlib

import numpy as np
from scipy.signal import butter, sosfiltfilt


VELOCITY_LPF_HZ = 0.3  # cut walking band; keep elevator ride dynamics


def lowpass(x: np.ndarray, fs: float, cutoff_hz: float = VELOCITY_LPF_HZ) -> np.ndarray:
    nyq = 0.5 * fs
    sos = butter(4, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, x)


_quality = importlib.import_module("src.prediction.algorithms.quality_filter")
estimate_gravity_vector = _quality.estimate_gravity_vector


def _compute_a_vert(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float) -> np.ndarray:
    gvec, g_mag, _ = estimate_gravity_vector(ax, ay, az, fs=fs, window_sec=0.5)
    g_hat = gvec / (np.linalg.norm(gvec) + 1e-12)
    return ax * g_hat[0] + ay * g_hat[1] + az * g_hat[2] - g_mag


def compute_velocity(a_vert: np.ndarray, fs: float) -> np.ndarray:
    """Global integration of DC-removed a_vert."""
    return np.cumsum(a_vert - float(np.mean(a_vert))) / fs
