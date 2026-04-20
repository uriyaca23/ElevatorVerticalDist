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
match_one_template = _common.match_one_template
run_fitter = _common.run_fitter

OUT_DIR_NAME = "basicTrapezoidGridWithConstraint"
TITLE_SUFFIX = "shared-shape per-ride trapezoid fit (|A|,W,f tied; max mean R²)"


def _search_best_pair(
    a: np.ndarray, t: np.ndarray,
    lo1: float, hi1: float, lo2: float, hi2: float,
    sign1: float, sign2: float,
    grid_W: np.ndarray, grid_F: np.ndarray,
) -> tuple[int, int, float, float, float, float, float] | None:
    """Exhaustive search for the shared-shape optimum.

    Returns ``(i1, i2, A_abs, W, f, r2_1, r2_2)`` or ``None`` if no
    sign-valid pair exists anywhere on the grid.
    """
    in1 = (t >= lo1) & (t <= hi1)
    in2 = (t >= lo2) & (t <= hi2)
    if not in1.any() or not in2.any():
        return None

    best: tuple[int, int, float, float, float, float, float] | None = None
    best_score = -np.inf

    for W in grid_W:
        for f in grid_F:
            scan = match_one_template(a, t, float(W), float(f))
            inner = scan.inner
            power = scan.local_power
            norm_t = scan.norm_t
            if norm_t < 1e-9:
                continue
            valid = np.isfinite(inner) & np.isfinite(power) & (power > 1e-9)
            m1 = in1 & valid & (sign1 * inner > 0.0)
            m2 = in2 & valid & (sign2 * inner > 0.0)
            if not m1.any() or not m2.any():
                continue

            i1s = np.where(m1)[0]
            i2s = np.where(m2)[0]

            # Sign-folded inner products (positive after the masks above).
            u1 = sign1 * inner[i1s]
            u2 = sign2 * inner[i2s]
            p1 = power[i1s]
            p2 = power[i2s]

            U1 = u1[:, None]
            U2 = u2[None, :]
            P1 = p1[:, None]
            P2 = p2[None, :]

            A = (U1 + U2) / (2.0 * norm_t)
            ss1 = P1 - 2.0 * A * U1 + A * A * norm_t
            ss2 = P2 - 2.0 * A * U2 + A * A * norm_t
            r2_1 = 1.0 - ss1 / P1
            r2_2 = 1.0 - ss2 / P2
            mean_r2 = 0.5 * (r2_1 + r2_2)

            flat = int(np.argmax(mean_r2))
            j1, j2 = np.unravel_index(flat, mean_r2.shape)
            score = float(mean_r2[j1, j2])
            if score > best_score:
                best_score = score
                i1_best = int(i1s[j1])
                i2_best = int(i2s[j2])
                best = (
                    i1_best, i2_best,
                    float(A[j1, j2]), float(W), float(f),
                    float(r2_1[j1, j2]), float(r2_2[j1, j2]),
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

    best = _search_best_pair(
        a_smooth, t_ride, lo1, hi1, lo2, hi2, sign1, sign2, grid_W, GRID_F,
    )
    if best is None:
        return fail

    i1, i2, A_abs, W, f, r2_1, r2_2 = best
    lobe1 = LobeFit(
        t_c=float(t_ride[i1]), a_peak=float(sign1 * A_abs),
        half_width_s=W, frac_flat=f, r2_local=r2_1,
    )
    lobe2 = LobeFit(
        t_c=float(t_ride[i2]), a_peak=float(sign2 * A_abs),
        half_width_s=W, frac_flat=f, r2_local=r2_2,
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
