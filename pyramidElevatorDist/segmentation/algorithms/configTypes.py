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
    ``pyramidElevatorDist.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal.detect.DetectConfig``
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
    min_peak_abs_a: float = 0.20
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

    # (W, f) trapezoid-template grid. ``w_min_s`` was lowered from 0.4 to
    # 0.3 in the iter_15 tuning round after diagnostics showed many
    # missed rides had ``pair_W`` pinned at the floor.
    w_min_s: float = 0.3
    w_max_s: float = 3.0
    n_w: int = 30
    f_min: float = 0.05
    f_max: float = 0.80
    n_f: int = 15

    # iter_13: prepend an explicit f=0 (pure-triangle) row to the (W, f)
    # grid built by ``DetectConfig.grid_f()``. One-floor / joined-pulse
    # rides have no cruise phase, so each lobe collapses to a triangle;
    # the per-pair argmax in ``joint_pair_score`` picks trapezoid vs.
    # triangle on a pair-by-pair basis with no separate branch needed.
    include_triangle_row: bool = True

    # Phone-aware amplitude floor. When ``phone_model`` is passed to
    # ``Segmenter.detect``, ``min_peak_abs_a`` / ``min_pair_abs_a`` are
    # tightened to ``max(config_floor, multiplier · σ_a)``.
    noise_sigma_multiplier: float = 6.0

    # Scalar acceleration signal the matched filter scores against.
    # ``"a_vert"``: signed projection onto the estimated gravity vector
    # (current default; orientation-aware but vulnerable to in-ride phone
    # rotation that invalidates the frozen ``ĝ``). ``"a_mag_minus_g"``:
    # rotation-invariant ``|a| − |ĝ|`` magnitude residual; robust to phone
    # tilt mid-ride, but picks up horizontal user motion. See
    # ``docs/latex/algorithm_report.tex`` for the trade-off discussion.
    input_signal: str = "a_vert"


class SEGMENT_ALGORITHM_CONFIG(BaseModel):
    algorithm: SegmentAlgorithm = SegmentAlgorithm.PRESSURE_FILTER
    config_path: Path = DEFAULT_CONFIG_PATH
    overrides: dict[str, Any] = Field(default_factory=dict)
    # Optional accel+gyro orientation reconstruction applied to the raw
    # accelerometer as the first step of ``Segmenter.detect`` (ACC template
    # match only). ``"none"`` = today's behavior; other values are keys of
    # ``pyramidElevatorDist.physics.reconstruct_az.RECONSTRUCTORS`` (Complementary, Mahony,
    # Madgwick, Valenti, ESKF). See ``docs/latex/algorithm_report.tex``.
    reconstruct: str = "none"
    # Optional input-cadence normalization. When set (e.g. ``50``), the
    # ACC_TEMPLATE_MATCH path first resamples the incoming trace onto a uniform
    # grid at this rate (gap-aware, time-correct) so the detector — tuned for a
    # fixed cadence — is robust to data arriving at any/variable sample rate.
    # ``None`` (default) = consume the trace as-is (native cadence); this keeps
    # the research evaluators and their reported numbers unchanged. The boutique
    # in-process pipeline sets it to 50; callers that pre-resample leave it None.
    resample_hz: int | None = None

    def load_params(self) -> dict[str, Any]:
        try:
            with open(self.config_path, "r") as f:
                all_params = json.load(f) or {}
        except (OSError, ValueError) as exc:
            from pyramidElevatorDist.exceptions import ConfigurationError
            raise ConfigurationError(
                f"cannot read the algorithm config at {self.config_path} "
                f"({type(exc).__name__}: {exc}). The packaged config.json "
                f"ships with the library - re-install the package or point "
                f"config_path at a valid JSON file."
            ) from exc
        params = dict(all_params.get(self.algorithm.value, {}))
        params.update(self.overrides)
        return params
