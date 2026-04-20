"""Predict the height difference of a segment from barometric pressure.

Converts each pressure sample to altitude via the ISA inversion in
``src/physics/barometric.py`` and returns ``altitude_end - altitude_start``.
A configurable edge-averaging window smooths the first/last samples to
damp barometer noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.physics.barometric import pressure_to_altitude
from src.prediction.algorithms.configTypes import BarometerHeightDiffConfig


def predict_height_difference_from_barometer(
    data: pd.DataFrame,
    config: BarometerHeightDiffConfig,
) -> float:
    pressure = np.asarray(data[config.pressure_col].to_numpy(), dtype=float)
    if pressure.size < 2:
        return 0.0

    altitude = pressure_to_altitude(pressure, p0_hpa=config.p0_hpa)
    k = max(1, min(int(config.edge_avg_samples), pressure.size // 2))
    start_alt = float(np.mean(altitude[:k]))
    end_alt = float(np.mean(altitude[-k:]))
    return end_alt - start_alt
