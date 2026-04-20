"""Algorithm: ``basicTrapezoidGrid`` — independent per-lobe trapezoid fit.

For every GT ride, search the :data:`common.GRID_W_S` × :data:`common.GRID_F`
template grid and pick the best ``(t_c, A, W, f)`` for lobe 1 and lobe 2
*independently*. Each lobe is restricted to its sign of the ride type and
to the half of the ride window it belongs to.

    up ride:   lobe 1 = +A (take-off),   lobe 2 = −A (landing)
    down ride: lobe 1 = −A (take-off),   lobe 2 = +A (landing)

This matches the original ``fit_trapezoid_pulses.py`` flavour — kept
verbatim so downstream labels/plots don't regress. The shared-shape
alternative lives in :mod:`constrained_grid`.

Outputs under ``labels/fit_elevator_paramater/basicTrapezoidGrid/``.

Run:
    venv/bin/python src/segmentation/algorithms/accelerometer_only/\
template_match/fit_elevator_parameters/basic_grid.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# Load ``common.py`` by file path so direct-script execution doesn't have
# to go through the ``src.segmentation.algorithms`` package __init__.
# Cache in ``sys.modules`` so ``@dataclass`` introspection of ``__module__``
# resolves correctly when a second fitter reloads the same file.
_HERE = Path(__file__).resolve().parent
_COMMON_MOD_NAME = "_fit_ep_common"
if _COMMON_MOD_NAME in sys.modules:
    _common = sys.modules[_COMMON_MOD_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_COMMON_MOD_NAME, _HERE / "common.py")
    _common = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    sys.modules[_COMMON_MOD_NAME] = _common
    _spec.loader.exec_module(_common)

LobeFit = _common.LobeFit
RideFit = _common.RideFit
GRID_W_S = _common.GRID_W_S
GRID_F = _common.GRID_F
LOBE1_REGION = _common.LOBE1_REGION
LOBE2_REGION = _common.LOBE2_REGION
match_one_template = _common.match_one_template
run_fitter = _common.run_fitter

OUT_DIR_NAME = "basicTrapezoidGrid"
TITLE_SUFFIX = "independent per-lobe trapezoid fit (matched-filter grid)"


def _grid_search_lobe(
    a: np.ndarray, t: np.ndarray,
    center_lo: float, center_hi: float, target_sign: float,
    grid_W: np.ndarray, grid_F: np.ndarray,
) -> LobeFit:
    """Best ``(t_c, A, W, f)`` in ``[center_lo, center_hi]`` with
    ``sign(A) == target_sign``."""
    n = a.size
    if n == 0:
        return LobeFit()

    in_region = (t >= center_lo) & (t <= center_hi)
    if not in_region.any():
        return LobeFit()

    best = LobeFit()
    best_score = -np.inf

    for W in grid_W:
        for f in grid_F:
            scan = match_one_template(a, t, float(W), float(f))
            r2 = scan.r2_local
            A_hat = scan.A_hat
            if not np.isfinite(r2).any():
                continue
            mask = in_region & np.isfinite(r2) & (np.sign(A_hat) == target_sign)
            if not mask.any():
                continue
            idx_candidates = np.where(mask)[0]
            best_idx = idx_candidates[np.argmax(r2[idx_candidates])]
            score = float(r2[best_idx])
            if score > best_score:
                best_score = score
                best = LobeFit(
                    t_c=float(t[best_idx]),
                    a_peak=float(A_hat[best_idx]),
                    half_width_s=float(W),
                    frac_flat=float(f),
                    r2_local=float(score),
                )
    return best


def fit_ride(
    t_ride: np.ndarray, a_smooth: np.ndarray,
    gt_t0: float, gt_t1: float,
    ride_idx: int, ride_type: str, fs: float,
) -> RideFit:
    duration = float(gt_t1 - gt_t0)
    fail = RideFit(index=ride_idx, ride_type=ride_type, duration_s=duration)
    if t_ride.size < 8 or duration <= 0:
        return fail

    ride_sign = 1.0 if ride_type == "up" else -1.0
    sign_lobe1 = +1.0 * ride_sign
    sign_lobe2 = -1.0 * ride_sign

    lo1 = gt_t0 + LOBE1_REGION[0] * duration
    hi1 = gt_t0 + LOBE1_REGION[1] * duration
    lo2 = gt_t0 + LOBE2_REGION[0] * duration
    hi2 = gt_t0 + LOBE2_REGION[1] * duration

    # Cap W by a fraction of the ride duration.
    W_cap = 0.5 * duration
    grid_W = GRID_W_S[GRID_W_S <= W_cap]
    if grid_W.size == 0:
        grid_W = GRID_W_S[:1]

    lobe1 = _grid_search_lobe(a_smooth, t_ride, lo1, hi1, sign_lobe1, grid_W, GRID_F)
    lobe2 = _grid_search_lobe(a_smooth, t_ride, lo2, hi2, sign_lobe2, grid_W, GRID_F)

    spacing: float | None = None
    if lobe1.t_c is not None and lobe2.t_c is not None:
        spacing = float(abs(lobe2.t_c - lobe1.t_c))

    return RideFit(
        index=ride_idx, ride_type=ride_type, duration_s=duration,
        lobe1=lobe1, lobe2=lobe2, lobe_centroid_spacing_s=spacing,
    )


def main() -> int:
    return run_fitter(OUT_DIR_NAME, fit_ride, title_suffix=TITLE_SUFFIX)


if __name__ == "__main__":
    sys.exit(main())
