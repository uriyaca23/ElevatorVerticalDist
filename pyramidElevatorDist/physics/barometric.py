"""Barometric altitude conversion + accelerometer-derived vertical velocity.

Altitude: International Standard Atmosphere (troposphere, h < 11 km)
inversion of the barometric formula,

    h = (T0 / L) * (1 - (P / P0) ** (R * L / (g * M)))

With the standard lapse rate L = 0.0065 K/m and R*M/g = 5.255, this becomes

    h = (T0 / L) * (1 - (P / P0) ** (1 / 5.255))

so supplying a surface temperature `T0` (in Kelvin) linearly rescales the
altitude axis. At the standard T0 = 288.15 K (15 °C), `T0/L` = 44330.77,
recovering the common

    h_m = 44330 * (1 - (P / P0) ** (1 / 5.255)).

For elevator-scale height differences, varying T0 by ±15 °C (≈±5 %) shifts
a 50 m Δh by roughly ±2.5 m — enough to matter for floor-level snapping.

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
T0_STD_K = 288.15  # ISA sea-level temperature (15 °C)
LAPSE_RATE_K_PER_M = 0.0065  # ISA tropospheric lapse rate


def pressure_to_altitude(
    pressure_hpa: float | np.ndarray | pd.Series,
    p0_hpa: float = P0_HPA,
    temperature_c: float | None = None,
) -> float | np.ndarray | pd.Series:
    """Convert barometric pressure (hPa) to altitude (meters) above the
    reference pressure `p0_hpa` (default: sea-level standard, 1013.25 hPa).

    When `temperature_c` is provided, it is treated as the surface
    temperature T0 and scales the altitude axis via T0/L (ISA troposphere
    form); when ``None``, the standard 15 °C is used, reproducing the
    classic `44330 * (1 - (P/P0)^(1/5.255))` formula exactly.
    """
    t0_k = T0_STD_K if temperature_c is None else (float(temperature_c) + 273.15)
    scale = t0_k / LAPSE_RATE_K_PER_M
    return scale * (1.0 - (pressure_hpa / p0_hpa) ** (1.0 / 5.255))


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
