"""Pydantic config model for the segmentation algorithm dispatcher."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import json

from pydantic import BaseModel, Field


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


class SegmentAlgorithm(str, Enum):
    PRESSURE_FILTER = "pressure_filter"
    ACC_ONLY = "acc_only"
    ACC_TEMPLATE_MATCH = "acc_template_match"


class PressureFilterConfig(BaseModel):
    velocity_threshold: float = 0.15
    smooth_window_sec: float = 3.0
    height_lowpass_sec: float = 8.0
    min_duration_sec: float = 3.0
    min_height_diff_m: float = 2.0
    merge_gap_sec: float = 6.0
    pad_sec: float = 1.0
    time_col: str = "time"
    height_col: str = "height"


CALIBRATORS_DIR = Path(__file__).with_name("calibrators")


class AccOnlyConfig(BaseModel):
    time_col: str = "time"
    x_col: str = "x"
    y_col: str = "y"
    z_col: str = "z"
    fs_hz: float = 100.0
    window_sec: float = 4.0
    overlap: float = 0.5
    band_elev_hz: tuple[float, float] = (0.05, 0.5)
    band_walk_hz: tuple[float, float] = (1.2, 2.8)
    enter_threshold: float = 0.3
    exit_threshold: float = 0.05
    min_duration_sec: float = 3.0
    merge_gap_sec: float = 4.0
    pad_sec: float = 2.0
    detrend_sec: float = 90.0
    local_var_sec: float = 8.0
    alpha: float = 0.1
    lr_path: Path = CALIBRATORS_DIR / "lr_weights.json"
    ivap_path: Path = CALIBRATORS_DIR / "ivap.json"
    edge_conformal_path: Path = CALIBRATORS_DIR / "edge_conformal.json"


class TemplateMatchConfig(BaseModel):
    time_col: str = "time"
    x_col: str = "x"
    y_col: str = "y"
    z_col: str = "z"
    fs_hz: float = 100.0
    lpf_hz: float = 0.3
    template_len: int = 256
    entry_frac: float = 0.5
    exit_frac: float = 0.5
    enter_threshold: float = 0.55
    nms_radius_sec: float = 4.0
    min_pair_gap_sec: float = 3.0
    max_pair_gap_sec: float = 90.0
    vel_weight: float = 0.6
    pad_sec: float = 1.5
    templates_path: Path = CALIBRATORS_DIR / "templates.json"
    ivap_path: Path = CALIBRATORS_DIR / "ivap.json"
    edge_conformal_path: Path = CALIBRATORS_DIR / "edge_conformal.json"


class SEGMENT_ALGORITHM_CONFIG(BaseModel):
    algorithm: SegmentAlgorithm = SegmentAlgorithm.PRESSURE_FILTER
    config_path: Path = DEFAULT_CONFIG_PATH
    overrides: dict[str, Any] = Field(default_factory=dict)

    def load_params(self) -> dict[str, Any]:
        with open(self.config_path, "r") as f:
            all_params = json.load(f) or {}
        params = dict(all_params.get(self.algorithm.value, {}))
        params.update(self.overrides)
        return params
