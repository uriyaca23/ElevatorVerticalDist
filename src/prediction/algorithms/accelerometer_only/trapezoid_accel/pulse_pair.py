"""Shared-shape trapezoid pulse-pair fitter for accel-domain Δh estimation.

This module implements the algorithm that replaces the 7-phase kinematic
S-curve model with a 2-shape-parameter trapezoid pulse pair, derived
from the matched-filter NCC grid search introduced by Eyal in
``src/segmentation/.../template_match/``.

Physical setup
--------------
A jerk-limited elevator ride produces a symmetric pair of acceleration
lobes: a positive acceleration lobe while the cabin accelerates to its
cruise velocity, and a negative (equal-area) deceleration lobe while it
brakes to a stop. Each lobe is well-approximated by a trapezoid with
symmetric linear ramp-up + flat top + symmetric linear ramp-down,
parameterised by:

    W  — half-width (seconds)   [total lobe duration = 2W]
    f  — flat-fraction ∈ [0, 1] [f=0 ⇒ triangular pulse, f=1 ⇒ rectangle]
    A  — amplitude (m/s²)       [positive for up-acceleration lobe,
                                  negative for decel lobe]

The shared-shape constraint says a real elevator's accel lobe and
decel lobe share the same (W, f, |A|) — they differ only in sign and
in their centres t_c1 < t_c2. This is a direct consequence of
ZUPT (cabin returns to rest, so ∫a_up = −∫a_decel) plus the
jerk-limited motion profile (same machine-driven limits during both
phases). Dropping the constraint adds 3 degrees of freedom that are
physically meaningless and just buy noise-fitting on our data.

Δh is then analytic:

    v_peak = |A| · W · (1 + f)                    (integral of one lobe)
    Δh     = sign · v_peak · (t_c2 − t_c1)
           = sign · |A| · W · (1 + f) · (t_c2 − t_c1)

because the velocity profile is trapezoidal (ramp up during lobe 1,
cruise at v_peak, ramp down during lobe 2) and the integrated area
telescopes to ``v_peak · Δt_centre``.

Fit
---
We search a discrete (W, f) grid (same one Eyal used — 30 × 15 = 450
templates). For each grid point, the matched filter gives the LS-optimal
amplitude Â and local R² at every candidate centre i:

    Â(W, f, i) = ⟨a, tpl⟩_i / ⟨tpl, tpl⟩
    R²(W, f, i) = 1 − ‖a − Â · tpl‖² / ‖a‖²    on the ±W window

The shared-shape joint optimum for a (+/−) lobe pair is the closed-form
LS solution

    |A|*(W, f, i1, i2) = (u1 + u2) / (2 · ⟨tpl, tpl⟩)
    R²_k               = 1 − (P_k − 2|A|*·u_k + |A|*²·⟨tpl,tpl⟩) / P_k

where ``u_k = sign_k · ⟨a, tpl⟩_{i_k}`` and ``P_k = ⟨a, a⟩`` on the ±W
window around i_k. We score each pair by ``½(R²₁ + R²₂)`` and keep the
argmax across (W, f, i1, i2).

The search is fully exhaustive — for each (W, f) one broadcast forms
the full (i1 × i2) pair-cost surface, so there is no reliance on a
top-K heuristic.

Theoretical σ
-------------
Under white-noise σ_a, the LS variance of each parameter follows from
the Fisher information of the matched-filter likelihood:

    σ_A²  = σ_a² / ⟨tpl, tpl⟩
    σ_tc² = σ_a² / (|A|² · ⟨(dtpl/dt)², (dtpl/dt)⟩)

W and f are discrete, so σ_W² and σ_f² come from the local curvature
of R² on the grid (finite-difference second derivative, scaled by the
residual variance).

Delta method gives σ_Δh from the gradient of Δh(A, W, f, Δt_c):

    σ_Δh² = (∂Δh/∂A)² σ_A² + (∂Δh/∂W)² σ_W² + (∂Δh/∂f)² σ_f²
                           + (∂Δh/∂Δt_c)² σ_Δt_c²
    ∂Δh/∂A     = sign · W · (1 + f) · Δt_c
    ∂Δh/∂W     = sign · |A| · (1 + f) · Δt_c
    ∂Δh/∂f     = sign · |A| · W · Δt_c
    ∂Δh/∂Δt_c  = sign · |A| · W · (1 + f)

σ_Δt_c² ≈ 2 σ_tc² since t_c1 and t_c2 are independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Grid (same one Eyal calibrated on train; start W at 0.3 s so templates are
# wider than the accel sampling period and large enough to detect)
# ---------------------------------------------------------------------------

GRID_W_S: np.ndarray = np.linspace(0.30, 3.00, 30)
GRID_F: np.ndarray = np.linspace(0.00, 0.80, 15)

# For prediction we expect the GT window already brackets one ride. The
# lobe search regions carve the window into a first-half + second-half
# with overlap in the middle (centre lobes can be claimed by either
# side). These are fractions of the ride duration measured from t_gt0.
LOBE1_REGION: tuple[float, float] = (0.00, 0.60)
LOBE2_REGION: tuple[float, float] = (0.40, 1.00)


# ---------------------------------------------------------------------------
# Trapezoid kernel + matched-filter primitive
# ---------------------------------------------------------------------------

def trapezoid_kernel(t: np.ndarray, t_c: float, W: float, frac_flat: float) -> np.ndarray:
    """Unit-amplitude symmetric trapezoid centered at t_c, half-width W,
    flat fraction f ∈ [0, 1]."""
    frac_flat = max(0.0, min(1.0, float(frac_flat)))
    W = max(1e-6, float(W))
    flat_half = frac_flat * W
    ramp_width = W - flat_half + 1e-9
    dt = np.abs(t - t_c)
    return np.where(
        dt <= flat_half, 1.0,
        np.where(dt < W, (W - dt) / ramp_width, 0.0),
    )


def trapezoid_kernel_deriv(t: np.ndarray, t_c: float, W: float, frac_flat: float) -> np.ndarray:
    """Time derivative of ``trapezoid_kernel``. Slope is ±1/ramp_width
    on the ramps, 0 elsewhere. Needed for the matched-filter CRB on t_c.
    """
    frac_flat = max(0.0, min(1.0, float(frac_flat)))
    W = max(1e-6, float(W))
    flat_half = frac_flat * W
    ramp_width = W - flat_half + 1e-9
    s = np.sign(t - t_c)
    dt = np.abs(t - t_c)
    ramp = (dt > flat_half) & (dt < W)
    out = np.zeros_like(t)
    out[ramp] = -s[ramp] / ramp_width
    return out


class TemplateScan(NamedTuple):
    A_hat: np.ndarray        # (n,)   closed-form LS amplitude at each centre
    r2_local: np.ndarray     # (n,)   local R² of the single-template fit
    inner: np.ndarray        # (n,)   ⟨a, tpl⟩ at each centre
    local_power: np.ndarray  # (n,)   ⟨a, a⟩ on the ±W window
    norm_t: float            # ⟨tpl, tpl⟩ (scalar)


def match_one_template(a: np.ndarray, t: np.ndarray, W: float, frac_flat: float) -> TemplateScan:
    """Slide a unit trapezoid of shape (W, f) over the smoothed accel
    signal ``a`` sampled at times ``t``. Returns the LS amplitude, the
    local R², and the raw inner-product + local-power series needed by
    the pair fit.

    Edges where the ±W window falls off the signal are NaN so masks
    can propagate cleanly through downstream code.
    """
    n = a.size
    nan = np.full(n, np.nan)
    if n == 0:
        return TemplateScan(nan[:0], nan[:0], nan[:0], nan[:0], 0.0)

    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0 / 100.0
    K = max(3, int(round(2 * W / dt)))
    if K % 2 == 0:
        K += 1
    half = K // 2

    t_kernel = (np.arange(K) - half) * dt
    tpl = trapezoid_kernel(t_kernel, 0.0, W, frac_flat)
    norm_t = float(np.sum(tpl * tpl))
    if norm_t < 1e-9:
        return TemplateScan(nan, nan, nan, nan, 0.0)

    inner = np.convolve(a, tpl[::-1], mode="same")

    a2 = a * a
    csum = np.concatenate(([0.0], np.cumsum(a2)))
    local_power = np.full(n, np.nan)
    if n - half > half:
        idx = np.arange(half, n - half)
        local_power[idx] = csum[idx + half + 1] - csum[idx - half]

    A_hat = np.full(n, np.nan)
    valid = np.isfinite(local_power)
    A_hat[valid] = inner[valid] / norm_t

    r2_local = np.full(n, np.nan)
    denom = local_power[valid]
    with np.errstate(divide="ignore", invalid="ignore"):
        ss_res = denom - A_hat[valid] * inner[valid]
        r2 = 1.0 - ss_res / np.where(denom > 1e-9, denom, np.nan)
    r2_local[valid] = r2

    inner_masked = np.where(np.isfinite(local_power), inner, np.nan)
    return TemplateScan(A_hat, r2_local, inner_masked, local_power, norm_t)


# ---------------------------------------------------------------------------
# Shared-shape pair fit
# ---------------------------------------------------------------------------

@dataclass
class PulsePairFit:
    """Outcome of the shared-shape (W, f, |A|) pair fit.

    `A` is positive; the two lobes share |A| but differ in sign.
    `sign` is +1 if lobe 1 is the acceleration lobe (up ride), −1 if
    lobe 1 is the deceleration lobe (down ride). `t_c1 < t_c2` by
    construction.
    """
    A: float                  # shared |A| (m/s²)
    W: float                  # shared half-width (s)
    f: float                  # shared flat-fraction
    t_c1: float               # lobe-1 centre (s, ride-local)
    t_c2: float               # lobe-2 centre (s, ride-local)
    sign: int                 # +1 for up ride, −1 for down ride
    r2_1: float               # local R² of lobe 1
    r2_2: float               # local R² of lobe 2
    joint_r2: float           # mean(R²_1, R²_2)
    residuals: np.ndarray     # a − Â·(tpl1 + tpl2) across the ride window
    norm_t: float             # ⟨tpl, tpl⟩ at the chosen (W, f)
    local_power_1: float      # ⟨a, a⟩ on ±W around t_c1
    local_power_2: float      # ⟨a, a⟩ on ±W around t_c2

    @property
    def delta_t_c(self) -> float:
        """Centre-to-centre spacing (always positive)."""
        return float(self.t_c2 - self.t_c1)


def _search_pair_on_grid(
    a: np.ndarray, t: np.ndarray,
    lo1: float, hi1: float, lo2: float, hi2: float,
    sign1: float, sign2: float,
    grid_W: np.ndarray, grid_F: np.ndarray,
) -> Optional[tuple]:
    """Exhaustive (W, f, i1, i2) search for the shared-shape optimum.

    Returns a tuple ``(W, f, i1, i2, A_abs, r2_1, r2_2, norm_t, P1, P2)``
    or ``None`` when no sign-valid pair exists anywhere on the grid.
    """
    in1 = (t >= lo1) & (t <= hi1)
    in2 = (t >= lo2) & (t <= hi2)
    if not in1.any() or not in2.any():
        return None

    best_score = -np.inf
    best = None

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
            u1 = sign1 * inner[i1s]
            u2 = sign2 * inner[i2s]
            p1 = power[i1s]
            p2 = power[i2s]

            U1 = u1[:, None]
            U2 = u2[None, :]
            P1 = p1[:, None]
            P2 = p2[None, :]

            A_abs = (U1 + U2) / (2.0 * norm_t)
            ss1 = P1 - 2.0 * A_abs * U1 + A_abs * A_abs * norm_t
            ss2 = P2 - 2.0 * A_abs * U2 + A_abs * A_abs * norm_t
            r2_1 = 1.0 - ss1 / P1
            r2_2 = 1.0 - ss2 / P2
            mean_r2 = 0.5 * (r2_1 + r2_2)

            flat = int(np.argmax(mean_r2))
            j1, j2 = np.unravel_index(flat, mean_r2.shape)
            score = float(mean_r2[j1, j2])
            if score > best_score:
                best_score = score
                best = (
                    float(W), float(f),
                    int(i1s[j1]), int(i2s[j2]),
                    float(A_abs[j1, j2]),
                    float(r2_1[j1, j2]), float(r2_2[j1, j2]),
                    float(norm_t),
                    float(P1[j1, 0]), float(P2[0, j2]),
                )
    return best


def fit_shared_shape_pair(
    a: np.ndarray,
    t: np.ndarray,
    gt_t0: float,
    gt_t1: float,
    direction: Optional[int] = None,
    grid_W: np.ndarray = GRID_W_S,
    grid_F: np.ndarray = GRID_F,
    lobe1_region: tuple[float, float] = LOBE1_REGION,
    lobe2_region: tuple[float, float] = LOBE2_REGION,
) -> Optional[PulsePairFit]:
    """Run the shared-shape (W, f, |A|) fit on a single ride window.

    Parameters
    ----------
    a : np.ndarray
        Smoothed gravity-projected vertical acceleration (m/s²).
    t : np.ndarray
        Ride-local time axis (s), length ``len(a)``.
    gt_t0, gt_t1 : float
        GT ride boundaries on ``t`` (s).
    direction : int | None
        +1 to force an up ride (lobe 1 = +A), −1 for down, None to try
        both and keep the higher joint R². For prediction from labelled
        GT we always pass ±1; for blind use ``None``.

    Returns
    -------
    PulsePairFit or None when no sign-valid pair fits.
    """
    duration = float(gt_t1 - gt_t0)
    if t.size < 8 or duration <= 0:
        return None

    W_cap = 0.5 * duration
    grid_W_eff = grid_W[grid_W <= W_cap]
    if grid_W_eff.size == 0:
        grid_W_eff = grid_W[:1]

    lo1 = gt_t0 + lobe1_region[0] * duration
    hi1 = gt_t0 + lobe1_region[1] * duration
    lo2 = gt_t0 + lobe2_region[0] * duration
    hi2 = gt_t0 + lobe2_region[1] * duration

    def _one_direction(d: int) -> Optional[PulsePairFit]:
        sign1 = +1.0 * d
        sign2 = -1.0 * d
        found = _search_pair_on_grid(
            a, t, lo1, hi1, lo2, hi2, sign1, sign2,
            grid_W_eff, grid_F,
        )
        if found is None:
            return None
        W_b, f_b, i1, i2, A_b, r2_1, r2_2, norm_t_b, P1, P2 = found
        # Reconstruct residuals on the pair-fit support region
        t_c1 = float(t[i1])
        t_c2 = float(t[i2])
        tpl = (
            sign1 * A_b * trapezoid_kernel(t, t_c1, W_b, f_b)
            + sign2 * A_b * trapezoid_kernel(t, t_c2, W_b, f_b)
        )
        residuals = a - tpl
        return PulsePairFit(
            A=A_b, W=W_b, f=f_b,
            t_c1=t_c1, t_c2=t_c2,
            sign=d,
            r2_1=r2_1, r2_2=r2_2,
            joint_r2=0.5 * (r2_1 + r2_2),
            residuals=residuals,
            norm_t=norm_t_b,
            local_power_1=P1, local_power_2=P2,
        )

    if direction is not None:
        return _one_direction(int(direction))

    up = _one_direction(+1)
    down = _one_direction(-1)
    if up is None:
        return down
    if down is None:
        return up
    return up if up.joint_r2 >= down.joint_r2 else down


# ---------------------------------------------------------------------------
# Δh + theoretical σ
# ---------------------------------------------------------------------------

def height_from_fit(fit: PulsePairFit) -> float:
    """Analytic Δh from the shared-shape fit.

    Derivation: lobe 1 integrates to v_peak = |A| · W · (1 + f); lobe 2
    integrates to −v_peak; between them the velocity is at v_peak; the
    velocity profile is a trapezoid (ramp-up over 2W, cruise of length
    Δt_c − 2W, ramp-down over 2W) whose integrated area telescopes to
    ``v_peak · Δt_c``. Sign is inherited from ``fit.sign``.
    """
    v_peak = fit.A * fit.W * (1.0 + fit.f)
    return float(fit.sign * v_peak * fit.delta_t_c)


def theoretical_sigma_height(
    fit: PulsePairFit,
    sigma_a: float,
    dt_sec: float,
    *,
    grid_W: np.ndarray = GRID_W_S,
    grid_F: np.ndarray = GRID_F,
    joint_r2: float | None = None,
    r2_epsilon: float = 1e-3,
    cruise_sec: float = 0.0,
    anchored: bool = False,
    overlap_delta: float = 0.05,
) -> dict:
    """Delta-method σ of Δh, derived directly from

        Δh = sign · A · W · (1 + f) · Δt_c

    and the Cramér-Rao bound on the matched-filter LS fit.

    Parameters beyond the fit:
    ``joint_r2``     Mean of the two lobes' R² at the fit optimum.
                     If given, scales the effective sensor variance by
                     ``1 / max(R², r2_epsilon)`` — the Wald
                     post-regression form that widens the CI in every
                     parameter when the model fits the data poorly.
    ``cruise_sec``   Duration of the cruise window used by the
                     velocity-anchor step. Needed only when
                     ``anchored=True``.
    ``anchored``     When True, ``A`` was replaced by
                     ``|v_cruise| / (W(1+f))`` and σ_A is the variance
                     of the cruise-velocity estimator rather than the
                     matched-filter CRB.
    ``overlap_delta``   W-relative margin above which the overlap
                     inflation of σ_Δt_c starts applying. For a pair
                     with Δt_c close to ``2W`` the two centre
                     estimates are no longer independent, so the
                     naïve CRB under-reports the variance on the
                     *spacing*. We multiply σ_Δt_c² by
                     ``1 + (2W / (Δt_c - 2W))²`` in the soft overlap
                     zone (roughly Δt_c < 4W).

    Returns a dict with the individual parameter σ, the propagated
    σ_Δh, the R²-scaling and overlap factors so the caller can
    inspect which term dominates.
    """
    A = max(fit.A, 1e-6)
    W = max(fit.W, 1e-6)
    f = max(fit.f, 0.0)
    dt_c = float(fit.delta_t_c)

    # --- R²-scaled effective sensor variance ---
    # Scaling by 1/R² follows from the Wald expression for the posterior
    # variance in a constrained LS fit: a low-R² segment has more of
    # the signal energy in residuals the model cannot explain, and the
    # parameter uncertainties scale proportionally to that lack of fit.
    if joint_r2 is None:
        r2_scale = 1.0
    else:
        r2_scale = 1.0 / max(float(joint_r2), float(r2_epsilon))
    sigma_a2_eff = (sigma_a ** 2) * r2_scale

    # --- σ_A: velocity-anchored or matched-filter CRB ---
    # When the caller used the ZUPT-integrated cruise velocity to anchor
    # A (estimator.py's velocity_anchor_A path), the correct σ_A² is the
    # variance of the cruise-velocity estimator divided by (W(1+f))² —
    # not the matched-filter CRB. Mean of N i.i.d. samples of a
    # white-noise signal has variance σ_a² · dt / T_cruise.
    denom_va = W * (1.0 + f)
    if anchored and cruise_sec > 1e-3 and denom_va > 1e-6:
        sigma_A2 = sigma_a2_eff * dt_sec / (cruise_sec * denom_va ** 2)
    else:
        sigma_A2 = sigma_a2_eff / max(fit.norm_t, 1e-9)

    # --- σ_tc from the pulse slope energy (CRB of single-centre fit) ---
    # ⟨(dtpl/dt)², ...⟩ for a unit trapezoid: two ramps of width
    # ramp_width = W(1-f), slope ±1/ramp_width. Continuous integrand
    # evaluates to 2/ramp_width.
    ramp_width = max(W * (1.0 - f), dt_sec)
    slope_energy = 2.0 / max(ramp_width, dt_sec)
    sigma_tc2 = sigma_a2_eff / max(A ** 2 * slope_energy, 1e-9)
    sigma_dtc2_crb = 2.0 * sigma_tc2

    # --- Overlap inflation of σ_Δt_c (only when lobes actually overlap) ---
    # At Δt_c = 2W the two trapezoid lobes touch but do not overlap;
    # their supports are disjoint, the off-diagonal Fisher block is zero,
    # and the diagonal CRB is correct. Inflation therefore applies only
    # in the unphysical regime Δt_c < 2W, where the shared-shape model
    # no longer describes a real ride. We still emit a finite factor
    # there so downstream math stays well-conditioned, but we do not
    # penalise legitimate touching-lobe rides — those are handed off to
    # the joined-pulse fit in the estimator.
    two_W = 2.0 * W
    if dt_c >= two_W:
        overlap_factor = 1.0
    else:
        shortfall = (two_W - dt_c) / max(two_W, dt_sec)
        overlap_factor = 1.0 + (shortfall / max(overlap_delta, 1e-3)) ** 2
    sigma_dtc2 = sigma_dtc2_crb * overlap_factor

    # --- σ_W, σ_f from grid quantization ---
    def _nearest_step(grid: np.ndarray, v: float) -> float:
        if grid.size < 2:
            return 1e-3
        diffs = np.diff(grid)
        idx = int(np.clip(np.searchsorted(grid, v), 1, grid.size - 1))
        return float(diffs[idx - 1])

    dW = _nearest_step(grid_W, W)
    dF = _nearest_step(grid_F, f)
    sigma_W2 = (dW ** 2) / 12.0
    sigma_f2 = (dF ** 2) / 12.0

    # --- Delta method on Δh = sign · A · W · (1+f) · Δt_c ---
    d_dA = W * (1.0 + f) * dt_c
    d_dW = A * (1.0 + f) * dt_c
    d_df = A * W * dt_c
    d_dDtc = A * W * (1.0 + f)

    sigma_dh2 = (
        d_dA ** 2 * sigma_A2
        + d_dW ** 2 * sigma_W2
        + d_df ** 2 * sigma_f2
        + d_dDtc ** 2 * sigma_dtc2
    )

    return {
        "sigma_A": float(np.sqrt(sigma_A2)),
        "sigma_tc": float(np.sqrt(sigma_tc2)),
        "sigma_W": float(np.sqrt(sigma_W2)),
        "sigma_f": float(np.sqrt(sigma_f2)),
        "sigma_dh": float(np.sqrt(sigma_dh2)),
        "r2_scale": float(r2_scale),
        "overlap_factor": float(overlap_factor),
        "anchored_sigma_A": bool(anchored and cruise_sec > 1e-3 and denom_va > 1e-6),
        "contributions": {
            "amp":   float((d_dA ** 2) * sigma_A2),
            "width": float((d_dW ** 2) * sigma_W2),
            "flat":  float((d_df ** 2) * sigma_f2),
            "dtc":   float((d_dDtc ** 2) * sigma_dtc2),
        },
    }


# ---------------------------------------------------------------------------
# Joined pulse fit: Δt_c = 2W constrained (the short-ride regime)
# ---------------------------------------------------------------------------
#
# When a ride is short enough that the cabin never settles into a cruise
# velocity, the accelerometer trace is a single bipolar pulse: a +A
# trapezoid lobe immediately followed by a −A trapezoid lobe, with no
# constant-velocity gap between them. In the pair-fit parameterisation
# this corresponds to Δt_c = 2W exactly (lobes touching). We fit this
# regime with a constrained model that has four free parameters —
# (A, W, f, t_mid) — instead of the five the unconstrained pair fit uses.
#
# The 4-param model:
#     a_θ(t) = s·A·[τ_{W,f}(t − t_mid + W) − τ_{W,f}(t − t_mid − W)]
#     Δh     = s · A · W · (1+f) · 2W
#            = 2·s·A·W²·(1+f)
#
# The σ derivation is the delta method again, with gradients
#     ∂Δh/∂A = 2·s·W²·(1+f)
#     ∂Δh/∂W = 4·s·A·W·(1+f)
#     ∂Δh/∂f = 2·s·A·W²
#     ∂Δh/∂t_mid = 0                              (Δh invariant under shift)
#
# The CRB on A uses the *joined* template norm
#     ⟨joined, joined⟩ = 2·⟨τ, τ⟩
# because the two touching lobes occupy disjoint intervals. The CRB on
# t_mid is tighter than the pair-fit's σ_tc by a factor √2 for the same
# reason (twice the slope energy, since there are four ramps instead of
# two).

def joined_kernel(t: np.ndarray, t_mid: float, W: float, frac_flat: float) -> np.ndarray:
    """Unit-amplitude bipolar (+/−) joined trapezoid centred at t_mid.

    Support is [t_mid − 2W, t_mid + 2W]; the positive lobe occupies
    [t_mid − 2W, t_mid] and the negative lobe [t_mid, t_mid + 2W].
    """
    pos = trapezoid_kernel(t, t_mid - W, W, frac_flat)
    neg = trapezoid_kernel(t, t_mid + W, W, frac_flat)
    return pos - neg


def match_joined_template(
    a: np.ndarray, t: np.ndarray, W: float, frac_flat: float,
) -> TemplateScan:
    """Same API as :func:`match_one_template` but uses the joined
    bipolar template of width 4W. Returns per-sample LS amplitude
    and local R² for the joined shape.
    """
    n = a.size
    nan = np.full(n, np.nan)
    if n == 0:
        return TemplateScan(nan[:0], nan[:0], nan[:0], nan[:0], 0.0)

    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0 / 100.0
    K_half = max(3, int(round(2 * W / dt)))
    K = 2 * K_half + 1
    half = K // 2
    # Template must fit inside the signal, else np.convolve(mode='same')
    # returns max(N, K) samples and downstream masks desync.
    if K >= n:
        return TemplateScan(nan, nan, nan, nan, 0.0)

    t_kernel = (np.arange(K) - half) * dt
    tpl = joined_kernel(t_kernel, 0.0, W, frac_flat)
    norm_t = float(np.sum(tpl * tpl))
    if norm_t < 1e-9:
        return TemplateScan(nan, nan, nan, nan, 0.0)

    inner = np.convolve(a, tpl[::-1], mode="same")[:n]

    a2 = a * a
    csum = np.concatenate(([0.0], np.cumsum(a2)))
    local_power = np.full(n, np.nan)
    if n - half > half:
        idx = np.arange(half, n - half)
        local_power[idx] = csum[idx + half + 1] - csum[idx - half]

    A_hat = np.full(n, np.nan)
    valid = np.isfinite(local_power)
    A_hat[valid] = inner[valid] / norm_t

    r2_local = np.full(n, np.nan)
    denom = local_power[valid]
    with np.errstate(divide="ignore", invalid="ignore"):
        ss_res = denom - A_hat[valid] * inner[valid]
        r2 = 1.0 - ss_res / np.where(denom > 1e-9, denom, np.nan)
    r2_local[valid] = r2

    inner_masked = np.where(np.isfinite(local_power), inner, np.nan)
    return TemplateScan(A_hat, r2_local, inner_masked, local_power, norm_t)


def fit_joined_pulse(
    a: np.ndarray,
    t: np.ndarray,
    gt_t0: float,
    gt_t1: float,
    direction: Optional[int] = None,
    grid_W: np.ndarray = GRID_W_S,
    grid_F: np.ndarray = GRID_F,
) -> Optional[PulsePairFit]:
    """Constrained fit where the two lobes are forced to touch
    (Δt_c = 2W). This is the short-ride / no-cruise regime.

    Returns the same :class:`PulsePairFit` contract as
    :func:`fit_shared_shape_pair` so downstream code can branch on
    ``joint_r2``/``delta_t_c`` uniformly; here ``t_c1 = t_mid − W``
    and ``t_c2 = t_mid + W`` are derived from the single fitted
    centre ``t_mid``.
    """
    duration = float(gt_t1 - gt_t0)
    if t.size < 8 or duration <= 0:
        return None

    # For a fully-joined pulse, total support is 4W <= duration ⇒ W <= duration/4.
    W_cap = 0.25 * duration
    grid_W_eff = grid_W[grid_W <= W_cap]
    if grid_W_eff.size == 0:
        grid_W_eff = grid_W[:1]

    lo_mid = gt_t0 + 0.2 * duration
    hi_mid = gt_t0 + 0.8 * duration

    def _one_direction(d: int) -> Optional[PulsePairFit]:
        best_score = -np.inf
        best = None
        for W in grid_W_eff:
            for f in grid_F:
                scan = match_joined_template(a, t, float(W), float(f))
                inner = scan.inner
                power = scan.local_power
                norm_t = scan.norm_t
                if norm_t < 1e-9:
                    continue
                valid = np.isfinite(inner) & np.isfinite(power) & (power > 1e-9)
                in_mid = (t >= lo_mid) & (t <= hi_mid) & valid
                # Same sign convention as pair fit: d=+1 expects the
                # positive lobe first (inner > 0), d=−1 the opposite.
                in_mid = in_mid & (float(d) * inner > 0.0)
                if not in_mid.any():
                    continue
                idxs = np.where(in_mid)[0]
                u = float(d) * inner[idxs]
                p = power[idxs]
                A_abs = u / norm_t
                ss = p - 2.0 * A_abs * u + A_abs * A_abs * norm_t
                r2 = 1.0 - ss / np.where(p > 1e-9, p, np.nan)
                k = int(np.nanargmax(r2)) if np.any(np.isfinite(r2)) else -1
                if k < 0:
                    continue
                score = float(r2[k])
                if score > best_score:
                    best_score = score
                    best = (float(W), float(f), int(idxs[k]),
                            float(A_abs[k]), float(r2[k]), float(norm_t),
                            float(p[k]))
        if best is None:
            return None
        W_b, f_b, i_mid, A_b, r2_b, norm_t_b, P_mid = best
        t_mid = float(t[i_mid])
        t_c1 = t_mid - W_b
        t_c2 = t_mid + W_b
        # Reconstruct residuals on the 4W support for joint R² parity
        tpl = float(d) * A_b * joined_kernel(t, t_mid, W_b, f_b)
        residuals = a - tpl
        return PulsePairFit(
            A=A_b, W=W_b, f=f_b,
            t_c1=t_c1, t_c2=t_c2,
            sign=int(d),
            r2_1=r2_b, r2_2=r2_b,      # joined fit: one R²; replicate for contract parity
            joint_r2=r2_b,
            residuals=residuals,
            norm_t=norm_t_b,
            local_power_1=P_mid, local_power_2=P_mid,
        )

    if direction is not None:
        return _one_direction(int(direction))
    up = _one_direction(+1)
    down = _one_direction(-1)
    if up is None:
        return down
    if down is None:
        return up
    return up if up.joint_r2 >= down.joint_r2 else down


def theoretical_sigma_height_joined(
    fit: PulsePairFit,
    sigma_a: float,
    *,
    grid_W: np.ndarray = GRID_W_S,
    grid_F: np.ndarray = GRID_F,
    joint_r2: float | None = None,
    r2_epsilon: float = 1e-3,
) -> dict:
    """Delta-method σ of Δh for the joined-pulse fit (Δt_c = 2W
    constrained). Propagates through

        Δh = 2·s·A·W²·(1+f),

    four free parameters (A, W, f, t_mid), with CRB on A from the
    joined-template norm 2·⟨τ,τ⟩ and CRB on t_mid from the four-ramp
    slope energy. Δh is invariant under time shift of t_mid, so that
    parameter contributes nothing to σ_Δh.
    """
    A = max(fit.A, 1e-6)
    W = max(fit.W, 1e-6)
    f = max(fit.f, 0.0)

    if joint_r2 is None:
        r2_scale = 1.0
    else:
        r2_scale = 1.0 / max(float(joint_r2), float(r2_epsilon))
    sigma_a2_eff = (sigma_a ** 2) * r2_scale

    # σ_A: joined-template CRB. norm_joined = 2 · norm_single
    # (touching lobes with disjoint support).
    sigma_A2 = sigma_a2_eff / max(fit.norm_t, 1e-9)

    # σ_W, σ_f: grid quantization as in the pair fit.
    def _nearest_step(grid: np.ndarray, v: float) -> float:
        if grid.size < 2:
            return 1e-3
        diffs = np.diff(grid)
        idx = int(np.clip(np.searchsorted(grid, v), 1, grid.size - 1))
        return float(diffs[idx - 1])
    dW = _nearest_step(grid_W, W)
    dF = _nearest_step(grid_F, f)
    sigma_W2 = (dW ** 2) / 12.0
    sigma_f2 = (dF ** 2) / 12.0

    # Delta method: Δh = 2 A W² (1+f).
    one_plus_f = 1.0 + f
    d_dA = 2.0 * W * W * one_plus_f
    d_dW = 4.0 * A * W * one_plus_f
    d_df = 2.0 * A * W * W
    sigma_dh2 = (
        d_dA ** 2 * sigma_A2
        + d_dW ** 2 * sigma_W2
        + d_df ** 2 * sigma_f2
    )
    return {
        "sigma_A": float(np.sqrt(sigma_A2)),
        "sigma_W": float(np.sqrt(sigma_W2)),
        "sigma_f": float(np.sqrt(sigma_f2)),
        "sigma_dh": float(np.sqrt(sigma_dh2)),
        "r2_scale": float(r2_scale),
        "mode": "joined",
        "contributions": {
            "amp":   float((d_dA ** 2) * sigma_A2),
            "width": float((d_dW ** 2) * sigma_W2),
            "flat":  float((d_df ** 2) * sigma_f2),
        },
    }


def smooth_rolling_mean(x: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    """Rolling mean of ``x`` with a window of ``seconds`` s, matching
    the pre-processing used by Eyal's matched-filter pipeline. Kept
    here so we don't have to import pandas just for this."""
    w = max(3, int(round(seconds * fs)))
    # Simple centered moving average (edge-mode 'same')
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")
