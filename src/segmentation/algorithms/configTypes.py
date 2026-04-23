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


class TemplateMatchConfig(BaseModel):
    """Hyperparameters for the trapezoid-pulse-pair grid detector.

    Mirrors the fields of
    ``src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal.detect.DetectConfig``
    one-to-one. The Segmenter dispatches through that detector; this
    model is the public, Pydantic-validated face of the dataclass so
    ``config.json`` / ``SEGMENT_ALGORITHM_CONFIG.overrides`` work the
    same as for every other algorithm.

    Input DataFrame for this algorithm must carry ``timestamp_ms``,
    ``x``, ``y``, ``z`` columns — the detector reads these names
    directly.
    """
    # Peak-pick / same-sample NMS (detect stages 3–4). Defaults updated
    # in the iter_07 tuning round — see
    # ``src/segmentation/README.md`` ("Tuning round — 2026-04") for the
    # before/after table and rationale.
    r2_peak_thresh: float = 0.40
    min_peak_abs_a: float = 0.25
    nms_radius_s: float = 1.0
    same_sign_min_gap_s: float = 5.0

    # Pair filter (stages 5–6)
    min_ride_s: float = 0.0
    max_ride_s: float = 30.0
    joint_r2_thresh: float = 0.90
    min_pair_abs_a: float = 0.30
    heatmap_energy_thresh: float = 0.40
    # Quiet-middle filter added in iter_04. Reject pairs whose inter-lobe
    # plateau RMS exceeds ``quiet_middle_ratio × pair_A_abs``. Set ≥ 1.0
    # to disable.
    quiet_middle_ratio: float = 0.5

    # Segment padding for downstream integrators (ZUPT / trapezoid_accel).
    # Emitted ride interval is ``[t_c1 - W - ε, t_c2 + W + ε]``.
    # Optimum chosen by predictor-MAE sweep (see
    # ``improvement_iterations/_sweep_epsilon.py``). Set to 0.0 for the
    # zero-padded behaviour.
    segment_pad_eps_s: float = 0.25

    # (W, f) trapezoid-template grid
    w_min_s: float = 0.4
    w_max_s: float = 3.0
    n_w: int = 30
    f_min: float = 0.05
    f_max: float = 0.80
    n_f: int = 15

    # Phone-aware amplitude floor. When ``phone_model`` is passed to
    # ``Segmenter.detect``, ``min_peak_abs_a`` / ``min_pair_abs_a`` are
    # tightened to ``max(config_floor, multiplier · σ_a)``.
    noise_sigma_multiplier: float = 6.0


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
