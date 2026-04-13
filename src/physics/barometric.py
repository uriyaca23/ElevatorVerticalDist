"""Barometric altitude conversion.

Uses the International Standard Atmosphere (troposphere, h < 11 km) inversion
of the barometric formula:

    h = (T0 / L) * (1 - (P / P0) ** (R * L / (g * M)))

with the standard constants below this gives the common simplified form:

    h_m = 44330 * (1 - (P / P0) ** (1 / 5.255))
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
