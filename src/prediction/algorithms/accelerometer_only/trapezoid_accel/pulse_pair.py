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
) -> dict:
    """Delta-method σ of Δh given the residual noise σ_a.

    Returns a dict with the individual parameter σ plus the propagated
    σ_Δh, so the caller can both build a CI *and* inspect which term
    dominates when diagnosing CI miscalibration.

    The white-noise parameters obey:

        σ_A²      = σ_a² / ⟨tpl, tpl⟩
        σ_tc²     = σ_a² / (|A|² · ⟨(dtpl/dt)², (dtpl/dt)⟩)
        σ_Δtc²    = 2 · σ_tc²                      (independent centres)

    W, f are discrete; we approximate σ_W ≈ Δ_W / √12 and σ_f ≈ Δ_f/√12
    where Δ_W, Δ_f are the local grid spacings — this is the
    quantization-limited variance and is the correct lower bound when
    the true optimum sits between grid points.
    """
    A = max(fit.A, 1e-6)
    W = max(fit.W, 1e-6)
    f = max(fit.f, 0.0)
    dt_c = float(fit.delta_t_c)

    # --- σ_A from LS variance of the matched filter ---
    sigma_A2 = (sigma_a ** 2) / max(fit.norm_t, 1e-9)

    # --- σ_tc from the pulse slope energy ---
    # ⟨(dtpl/dt)², ...⟩ for a unit trapezoid: two ramps of width
    # ramp_width = W(1-f), slope ±1/ramp_width, so squared slope
    # is 1/ramp_width² integrated over two ramps of length
    # ramp_width·dt_sec/dt_sec samples = ramp_width seconds ×
    # samples/sec. Stated as a discrete sum at sampling period dt_sec:
    ramp_width = max(W * (1.0 - f), dt_sec)
    # Two ramps, each has ramp_width/dt samples of squared slope
    # (1/ramp_width)². Slope energy ≈ 2 · (1/ramp_width)² · (ramp_width / dt_sec)
    # = 2 / (ramp_width · dt_sec). Multiply by dt_sec to convert the
    # sum into the continuous ⟨(dtpl/dt)², dtpl/dt⟩ integrand:
    slope_energy = 2.0 / max(ramp_width, dt_sec)
    sigma_tc2 = (sigma_a ** 2) / max(A ** 2 * slope_energy, 1e-9)
    sigma_dtc2 = 2.0 * sigma_tc2

    # --- σ_W, σ_f from grid quantization ---
    # Use the nearest grid spacings. For an interior grid point this
    # is just the step; for edges we take the one-sided step.
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

    # --- Delta method ---
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
        "contributions": {
            "amp":   float((d_dA ** 2) * sigma_A2),
            "width": float((d_dW ** 2) * sigma_W2),
            "flat":  float((d_df ** 2) * sigma_f2),
            "dtc":   float((d_dDtc ** 2) * sigma_dtc2),
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
