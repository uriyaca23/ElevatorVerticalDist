"""Algorithm: ``basicTrapezoidGridWithConstraint`` — shared-shape per-ride fit.

The two lobes of an elevator ride (take-off vs landing) are physically the
same pulse up to sign — same ``|A|``, ``W``, ``f``. The basic fitter in
:mod:`basic_grid` lets each lobe pick its shape independently, so noise
or orientation drift can push the two apart. This fitter enforces the
constraint directly:

    lobe 1 :  a  ≈  +s * |A| * trapezoid(t − t_c1; W, f)
    lobe 2 :  a  ≈  −s * |A| * trapezoid(t − t_c2; W, f)

where ``s = +1`` for up rides and ``−1`` for down rides (same convention
as :mod:`basic_grid`). The shape ``(W, f)`` and magnitude ``|A|`` are
shared; only ``t_c`` differs between lobes.

Selection rule. For every ``(W, f)`` on the :data:`common.GRID_W_S` ×
:data:`common.GRID_F` grid:

  1. Slide the unit template and read off ``inner[i] = ⟨a, tpl⟩`` and
     ``power[i] = ⟨a, a⟩`` on every ±W window (:func:`common.match_one_template`).
  2. Restrict lobe 1 to ``LOBE1_REGION`` with the correct sign, same for
     lobe 2 on ``LOBE2_REGION``.
  3. For every remaining ``(i1, i2)`` pair, the least-squares optimal
     shared magnitude is

            |A|*  =  ( sign1·inner[i1] + sign2·inner[i2] )  /  (2·norm_t)

     and the per-lobe local R² under that constraint is

            R²_k  =  1 − ( P_k − 2·|A|*·s_k·inner[i_k] + |A|*²·norm_t ) / P_k.

  4. Score each pair by ``mean(R²_1, R²_2)`` and keep the argmax across
     all ``(W, f, i1, i2)``.

The search is fully exhaustive over the sign-valid pair grid (done with a
single broadcast per ``(W, f)``), so we're not relying on a top-K heuristic.

Outputs under ``labels/fit_elevator_paramater/basicTrapezoidGridWithConstraint/``.
Each :class:`~common.LobeFit` still carries its own ``half_width_s`` /
``frac_flat`` / ``a_peak`` — by construction ``|a_peak|``, ``half_width_s``,
``frac_flat`` are identical between lobe1 and lobe2 in this variant.

Run:
    venv/bin/python src/segmentation/algorithms/accelerometer_only/\
template_match/fit_elevator_parameters/constrained_grid.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from src.utils.trapezoid_template import search_shared_shape_pair

# See ``basic_grid.py`` for why we load ``common.py`` by file path.
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
run_fitter = _common.run_fitter

OUT_DIR_NAME = "basicTrapezoidGridWithConstraint"
TITLE_SUFFIX = "shared-shape per-ride trapezoid fit (|A|,W,f tied; max mean R²)"


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
    sign1 = +1.0 * ride_sign
    sign2 = -1.0 * ride_sign

    lo1 = gt_t0 + LOBE1_REGION[0] * duration
    hi1 = gt_t0 + LOBE1_REGION[1] * duration
    lo2 = gt_t0 + LOBE2_REGION[0] * duration
    hi2 = gt_t0 + LOBE2_REGION[1] * duration

    W_cap = 0.5 * duration
    grid_W = GRID_W_S[GRID_W_S <= W_cap]
    if grid_W.size == 0:
        grid_W = GRID_W_S[:1]

    best = search_shared_shape_pair(
        a_smooth, t_ride, lo1, hi1, lo2, hi2, sign1, sign2, grid_W, GRID_F,
    )
    if best is None:
        return fail

    lobe1 = LobeFit(
        t_c=float(t_ride[best.i1]), a_peak=float(sign1 * best.A_abs),
        half_width_s=best.W, frac_flat=best.f, r2_local=best.r2_1,
    )
    lobe2 = LobeFit(
        t_c=float(t_ride[best.i2]), a_peak=float(sign2 * best.A_abs),
        half_width_s=best.W, frac_flat=best.f, r2_local=best.r2_2,
    )
    return RideFit(
        index=ride_idx, ride_type=ride_type, duration_s=duration,
        lobe1=lobe1, lobe2=lobe2,
        lobe_centroid_spacing_s=float(abs(lobe2.t_c - lobe1.t_c)),
    )


def main() -> int:
    return run_fitter(OUT_DIR_NAME, fit_ride, title_suffix=TITLE_SUFFIX)


if __name__ == "__main__":
    sys.exit(main())
