"""General-purpose signal-processing helpers.

Kept separate from :mod:`src.utils.accelerometer_utils` so callers with
no accelerometer context can grab plain filter / spectral utilities
without pulling in the projection stack.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass(
    data: np.ndarray, fs: float, cutoff: float = 3.0, order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass via ``filtfilt``.

    Returns ``data`` unchanged for pathologically short buffers or when
    ``cutoff >= Nyquist`` — the caller usually wants a passthrough in
    those cases rather than a crash.
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
