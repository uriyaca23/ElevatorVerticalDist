"""Trapezoid pulse template + matched-filter primitives.

Stage-agnostic numerics: works on bare numpy arrays, no segmentation /
prediction types. Used by both stages — segmentation's template-match
detector (``src/segmentation/.../template_match/``) and prediction's
trapezoid-pulse-pair Δh estimator
(``src/prediction/algorithms/accelerometer_only/trapezoid_accel/``).

Contents:

* :func:`trapezoid_kernel` — unit-amplitude symmetric trapezoid.
* :class:`TemplateScan` + :func:`match_one_template` — slide one
  ``(W, frac_flat)`` template across a signal and return the per-sample
  closed-form LS amplitude, local R², inner product, and local signal
  power.
* :func:`search_shared_shape_pair` — exhaustive ``(W, f, i1, i2)``
  search for the shared-shape (|A| tied across two opposite-sign lobes)
  joint optimum.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


def trapezoid_kernel(t: np.ndarray, t_c: float, W: float, frac_flat: float) -> np.ndarray:
    """Unit-amplitude symmetric trapezoid centred at ``t_c``, half-width
    ``W``, flat fraction ``frac_flat ∈ [0, 1]`` (0 ⇒ triangle, 1 ⇒ rectangle).
    """
    frac_flat = max(0.0, min(1.0, float(frac_flat)))
    W = max(1e-6, float(W))
    flat_half = frac_flat * W
    ramp_width = W - flat_half + 1e-9
    dt = np.abs(t - t_c)
    return np.where(
        dt <= flat_half, 1.0,
        np.where(dt < W, (W - dt) / ramp_width, 0.0),
    )


class TemplateScan(NamedTuple):
    """Result of sliding one ``(W, frac_flat)`` template over a signal.

    ``A_hat[i]`` / ``r2_local[i]`` are the unconstrained least-squares
    amplitude and local R² if the template were centred at sample ``i``.
    ``inner[i] = A_hat[i] * norm_t`` is the raw ``⟨a, tpl⟩`` inner
    product, and ``local_power[i] = ⟨a, a⟩`` on the same ±W window.
    Positions whose ±W window falls off the signal are NaN.
    """

    A_hat: np.ndarray
    r2_local: np.ndarray
    inner: np.ndarray
    local_power: np.ndarray
    norm_t: float


def match_one_template(a: np.ndarray, t: np.ndarray, W: float, frac_flat: float) -> TemplateScan:
    """Slide a unit trapezoid of shape ``(W, frac_flat)`` over signal ``a``."""
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


class PairSearchResult(NamedTuple):
    """Outcome of :func:`search_shared_shape_pair`.

    ``A_abs`` is the shared closed-form LS magnitude. ``norm_t``,
    ``local_power_1`` and ``local_power_2`` are the matched-filter
    second-moment quantities at the optimum — kept around so callers
    that need the Cramér-Rao bound on Δh (prediction's σ derivation)
    don't have to recompute them.
    """

    W: float
    f: float
    i1: int
    i2: int
    A_abs: float
    r2_1: float
    r2_2: float
    norm_t: float
    local_power_1: float
    local_power_2: float


def search_shared_shape_pair(
    a: np.ndarray, t: np.ndarray,
    lo1: float, hi1: float, lo2: float, hi2: float,
    sign1: float, sign2: float,
    grid_W: np.ndarray, grid_F: np.ndarray,
) -> Optional[PairSearchResult]:
    """Exhaustive ``(W, f, i1, i2)`` search for the shared-shape optimum.

    For every ``(W, f)`` on the grid:

    * Slide the unit template via :func:`match_one_template`.
    * Restrict lobe 1 to ``[lo1, hi1]`` with sign ``sign1`` and lobe 2 to
      ``[lo2, hi2]`` with sign ``sign2``.
    * For every remaining ``(i1, i2)`` pair, the LS-optimal shared
      magnitude is ``|A| = (sign1·inner[i1] + sign2·inner[i2]) / (2·norm_t)``
      and the per-lobe local R² under that constraint follows from
      ``R² = 1 − (P − 2·|A|·s·inner + |A|²·norm_t) / P``.
    * Keep the ``(W, f, i1, i2)`` argmax of ``½(R²₁ + R²₂)``.

    Returns ``None`` when the search regions are empty or no sign-valid
    pair exists anywhere on the grid.
    """
    in1 = (t >= lo1) & (t <= hi1)
    in2 = (t >= lo2) & (t <= hi2)
    if not in1.any() or not in2.any():
        return None

    best: Optional[PairSearchResult] = None
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
                best = PairSearchResult(
                    W=float(W), f=float(f),
                    i1=int(i1s[j1]), i2=int(i2s[j2]),
                    A_abs=float(A_abs[j1, j2]),
                    r2_1=float(r2_1[j1, j2]), r2_2=float(r2_2[j1, j2]),
                    norm_t=float(norm_t),
                    local_power_1=float(P1[j1, 0]),
                    local_power_2=float(P2[0, j2]),
                )
    return best
