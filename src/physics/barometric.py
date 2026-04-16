"""Barometric altitude conversion + accelerometer-derived vertical velocity.

Altitude: International Standard Atmosphere (troposphere, h < 11 km)
inversion of the barometric formula,

    h = (T0 / L) * (1 - (P / P0) ** (R * L / (g * M)))

which with the standard constants collapses to the common form

    h_m = 44330 * (1 - (P / P0) ** (1 / 5.255)).

Velocity-from-accelerometer: pure cumulative integration of the
acceleration magnitude after subtracting its session-level mean. No
gravity projection (so device rotation during the session doesn't leak
horizontal acceleration into the vertical channel) and no per-window
ZUPT assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

P0_HPA = 1013.25  # sea-level standard pressure


def pressure_to_altitude(
    pressure_hpa: float | np.ndarray | pd.Series,
    p0_hpa: float = P0_HPA,
) -> float | np.ndarray | pd.Series:
    """Convert barometric pressure (hPa) to altitude (meters) above the
    reference pressure `p0_hpa` (default: sea-level standard, 1013.25 hPa).
    """
    return 44330.0 * (1.0 - (pressure_hpa / p0_hpa) ** (1.0 / 5.255))


def calculate_velocity_from_accelerometer(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Cumulative vertical velocity from raw 3-axis accelerometer.

    Recipe (matches the ``run_results/.../velocity.png`` panel):

        1. mag  = ‖(ax, ay, az)‖                 # rotation-invariant scalar
        2. a    = mag − mean(mag)                # subtract session-wide gravity
        3. v(t) = cumulative integral of a

    No gravity projection (so device rotation during the session doesn't
    leak horizontal acceleration into the "vertical" channel) and no ZUPT
    / per-window reset. ``v`` is a single session-wide curve.

    Args:
        ax, ay, az: 1-D arrays of accelerometer samples (m/s²).
        fs: sample rate in Hz (uniform sampling assumed).

    Returns:
        Velocity ``v`` (m/s), same length as the input.
    """
    ax_a = np.asarray(ax, dtype=float)
    ay_a = np.asarray(ay, dtype=float)
    az_a = np.asarray(az, dtype=float)

    mag = np.sqrt(ax_a * ax_a + ay_a * ay_a + az_a * az_a)
    a_lin = mag - float(np.mean(mag))
    return np.cumsum(a_lin) / float(fs)
