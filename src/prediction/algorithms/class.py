"""Pydantic config model for the prediction algorithm dispatcher.

Three algorithms are registered:
  * ``BAROMETER_HEIGHT_DIFF`` — ISA-inversion Δh from pressure.
  * ``ZUPT_ACCEL`` — Zero-Velocity Update double integration.
  * ``TRAPEZOID_ACCEL`` — shared-shape trapezoid pulse-pair matched-filter
    fit in the acceleration domain. Replaces the old 7-step S-curve
    velocity-domain NLS fitter — the new method is physically
    equivalent (the trapezoid is the jerk-limited acceleration profile
    with phases 1-3 collapsed to a single symmetric pulse), simpler
    (2 shape params + closed-form amplitude), and more accurate on
    our dataset.

Each algorithm has its own Pydantic config subclass. The top-level
``PREDICT_ALGORITHM_CONFIG`` selects one of them and (optionally)
overrides fields from ``config.json``.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import json

from pydantic import BaseModel, Field


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


class PredictAlgorithm(str, Enum):
    BAROMETER_HEIGHT_DIFF = "barometer_height_diff"
    ZUPT_ACCEL = "zupt_accel"
    TRAPEZOID_ACCEL = "trapezoid_accel"


class BarometerHeightDiffConfig(BaseModel):
    time_col: str = "timestamp_ms"
    pressure_col: str = "pressure"
    p0_hpa: float = 1013.25
    edge_avg_samples: int = 1


class PREDICT_ALGORITHM_CONFIG(BaseModel):
    algorithm: PredictAlgorithm = PredictAlgorithm.BAROMETER_HEIGHT_DIFF
    config_path: Path = DEFAULT_CONFIG_PATH
    overrides: dict[str, Any] = Field(default_factory=dict)

    def load_params(self) -> dict[str, Any]:
        with open(self.config_path, "r") as f:
            all_params = json.load(f) or {}
        params = dict(all_params.get(self.algorithm.value, {}))
        params.update(self.overrides)
        return params
