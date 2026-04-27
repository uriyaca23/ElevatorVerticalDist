"""Vectorized matched-filter primitives for the trapezoid template bank.

Drop-in replacements for the per-(W, f) Python loops that dominate
segmentation and prediction runtime:

* :func:`build_template_bank` — pre-build the (W, F) trapezoid kernel
  bank once (zero-padded to ``K_max`` for batched FFT).
* :func:`sweep_grid_max` — per-W FFT-batched matched-filter sweep that
  emits the per-sample argmax across the whole (W, F) grid. Replaces
  ``detect._sweep_best_template`` without storing the full grid.
* :func:`gather_inner_at_peaks` — gather matched-filter inner products
  at a sparse list of sample indices, batched across (W, F). Replaces
  the per-(W, f, pair) kernel-build inside ``pair_filter.joint_pair_score``.
* :func:`scan_template_bank` — convenience wrapper that returns the
  full (W, F, N) cube. Used by tests / small-segment prediction code.
* :func:`score_pair_at_peaks` — closed-form shared-shape joint R²
  cube for a (i1, i2) pair across all (W, F).

All routines are bare numpy (FFT through ``np.fft.rfft``/``irfft`` with
optional ``scipy.fft.next_fast_len`` for sizing). Numerically the only
difference from the original primitives is the matched-filter inner
product, where FFT-based correlation differs from direct correlation
by ~1e-10 ulp -- well below any threshold gate downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional accelerator
    from scipy.fft import next_fast_len as _next_fast_len  # type: ignore[import]
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - numpy-only fallback
    _HAVE_SCIPY = False
    _next_fast_len = None

from .trapezoid_template import trapezoid_kernel


__all__ = [
    "TemplateBank", "build_template_bank",
    "ScanResult", "scan_template_bank",
    "sweep_grid_max", "GridMaxResult",
    "gather_inner_at_peaks",
    "score_pair_at_peaks",
]


def _good_fft_len(n: int) -> int:
    if _HAVE_SCIPY:
        return int(_next_fast_len(n, real=True))
    # Power-of-two fallback.
    return 1 << (max(n - 1, 0)).bit_length() if n > 1 else 1


@dataclass
class TemplateBank:
    """Pre-built (W, F) template bank.

    ``tpls[w, f]`` is the unit-amplitude trapezoid kernel for ``W=grid_W[w]``
    and ``f=grid_F[f]``. Each row is zero-padded out to ``K_max`` so the
    whole bank can be FFT-convolved in one batched call. ``K_arr[w]`` is
    the actual support length per W, ``norm_t[w, f]`` the precomputed
    self-norm.
    """
    grid_W: np.ndarray   # [n_W]
    grid_F: np.ndarray   # [n_F]
    K_arr: np.ndarray    # [n_W] int — odd kernel length per W
    K_max: int
    half_max: int
    tpls: np.ndarray     # [n_W, n_F, K_max] float64 — zero-padded kernels
    norm_t: np.ndarray   # [n_W, n_F] float64
    dt: float


def build_template_bank(
    grid_W: np.ndarray, grid_F: np.ndarray, dt: float,
) -> TemplateBank:
    """Pre-compute the (W, F) trapezoid template bank for sample step ``dt``.

    Mirrors the ``K = max(3, round(2*W/dt)) | 1`` (force odd) kernel-size
    convention of :func:`src.utils.trapezoid_template.match_one_template`.
    """
    grid_W = np.ascontiguousarray(grid_W, dtype=np.float64)
    grid_F = np.ascontiguousarray(grid_F, dtype=np.float64)
    n_W = grid_W.size
    n_F = grid_F.size
    K_arr = np.maximum(3, np.round(2.0 * grid_W / dt).astype(int))
    K_arr = np.where(K_arr % 2 == 0, K_arr + 1, K_arr)
    K_max = int(K_arr.max())
    half_max = K_max // 2

    tpls = np.zeros((n_W, n_F, K_max), dtype=np.float64)
    for wi in range(n_W):
        K = int(K_arr[wi])
        half = K // 2
        offset = half_max - half
        t_k = (np.arange(K) - half) * dt
        for fi in range(n_F):
            tpls[wi, fi, offset:offset + K] = trapezoid_kernel(
                t_k, 0.0, float(grid_W[wi]), float(grid_F[fi]),
            )
    norm_t = np.sum(tpls * tpls, axis=-1)
    return TemplateBank(
        grid_W=grid_W, grid_F=grid_F, K_arr=K_arr,
        K_max=K_max, half_max=half_max, tpls=tpls, norm_t=norm_t, dt=dt,
    )


# ---------------------------------------------------------------------------
# Per-W batched FFT correlation core
# ---------------------------------------------------------------------------

def _correlate_w(
    a: np.ndarray, tpls_w: np.ndarray, K: int,
) -> np.ndarray:
    """Batched FFT correlation for one W.

    ``tpls_w`` has shape ``(n_F, K_max)`` (zero-padded) but only the
    central ``K`` samples are non-zero. ``K`` is the odd support length
    used to size the FFT for this W. Returns ``(n_F, n)`` matched-filter
    inner products.
    """
    n = a.size
    n_F = tpls_w.shape[0]
    if n == 0 or n_F == 0:
        return np.zeros((n_F, n), dtype=np.float64)

    K_max = tpls_w.shape[1]
    half_max = K_max // 2
    half = K // 2
    # The non-zero support of tpls_w is centred in the K_max window.
    # Slice out the centred K samples to FFT only what matters; this
    # keeps n_fft small for small W.
    offset = half_max - half
    tpls_active = tpls_w[:, offset:offset + K]
    # Reverse along K for correlation == convolution(reverse).
    tpls_rev = tpls_active[:, ::-1]

    n_full = n + K - 1
    n_fft = _good_fft_len(n_full)
    A_fft = np.fft.rfft(a, n_fft)
    T_fft = np.fft.rfft(tpls_rev, n_fft, axis=-1)
    Y = np.fft.irfft(A_fft[None, :] * T_fft, n_fft, axis=-1)[:, :n_full]
    # 'same' mode trim: skip the first (K - 1) // 2 = half samples.
    return Y[:, half:half + n]


# ---------------------------------------------------------------------------
# Convenience: full (W, F, N) cube — used by tests + small-segment fits
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    inner: np.ndarray    # [n_W, n_F, N] float64
    power: np.ndarray    # [n_W, N]      float64
    A_hat: np.ndarray    # [n_W, n_F, N] float64
    r2: np.ndarray       # [n_W, n_F, N] float64
    valid: np.ndarray    # [n_W, N]      bool
    bank: TemplateBank


def _per_w_power(a: np.ndarray, K_arr: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-W local power (cumsum of a^2) and validity mask."""
    n_W = K_arr.size
    power = np.full((n_W, n), np.nan)
    valid = np.zeros((n_W, n), dtype=bool)
    if n == 0:
        return power, valid
    csum = np.concatenate(([0.0], np.cumsum(a * a)))
    for wi in range(n_W):
        K = int(K_arr[wi])
        if K >= n:
            continue  # Mirror match_one_template's guard
        half = K // 2
        if n - half > half:
            lo, hi = half, n - half
            power[wi, lo:hi] = (
                csum[lo + half + 1: hi + half + 1]
                - csum[lo - half: hi - half]
            )
            valid[wi, lo:hi] = True
    return power, valid


def scan_template_bank(a: np.ndarray, bank: TemplateBank) -> ScanResult:
    """Compute inner / power / A_hat / R² for the whole (W, F, N) grid.

    Memory cost is roughly ``3 * n_W * n_F * N * 8`` bytes; for the
    default 30 × 16 grid on a 70k-sample signal that's ~770 MB. Use
    only on small segments (prediction-time per-ride windows). For
    whole-session scans use :func:`sweep_grid_max` instead.
    """
    n = a.size
    n_W, n_F, K_max = bank.tpls.shape
    K_arr = bank.K_arr

    inner = np.full((n_W, n_F, n), np.nan)
    power, valid = _per_w_power(a, K_arr, n)
    if n > 0:
        for wi in range(n_W):
            K = int(K_arr[wi])
            if K >= n:
                continue
            inner[wi] = _correlate_w(a, bank.tpls[wi], K)
        # Mask inner where invalid window; broadcast across F.
        inner = np.where(valid[:, None, :], inner, np.nan)

    norm_t = bank.norm_t  # [n_W, n_F]
    safe_norm = np.where(norm_t > 1e-9, norm_t, np.nan)[:, :, None]
    A_hat = inner / safe_norm
    pow_b = np.where(valid, power, np.nan)[:, None, :]
    safe_pow = np.where(pow_b > 1e-9, pow_b, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ss_res = pow_b - A_hat * inner
        r2 = 1.0 - ss_res / safe_pow
    return ScanResult(inner=inner, power=power, A_hat=A_hat,
                      r2=r2, valid=valid, bank=bank)


# ---------------------------------------------------------------------------
# Memory-light per-sample max sweep — replaces _sweep_best_template
# ---------------------------------------------------------------------------

@dataclass
class GridMaxResult:
    best_r2: np.ndarray     # [N]
    best_A: np.ndarray      # [N]
    best_W_idx: np.ndarray  # [N] int
    best_F_idx: np.ndarray  # [N] int
    best_pos_r2: np.ndarray; best_pos_A: np.ndarray
    best_neg_r2: np.ndarray; best_neg_A: np.ndarray
    bank: TemplateBank


def sweep_grid_max(a: np.ndarray, bank: TemplateBank) -> GridMaxResult:
    """Per-sample argmax over the full (W, F) grid (memory-light).

    Reduces inner products on-the-fly without materialising the full
    (W, F, N) cube. Per-sample fields exactly mirror what
    ``detect._sweep_best_template`` returns:

    * ``best_r2[i]``, ``best_A[i]`` — best across all (W, F) at sample i.
    * ``best_W_idx[i]``, ``best_F_idx[i]`` — argmax indices.
    * ``best_pos_*`` / ``best_neg_*`` — best across templates whose LS
      amplitude has the matching sign at i.
    """
    n = a.size
    n_W, n_F, _ = bank.tpls.shape
    K_arr = bank.K_arr
    norm_t = bank.norm_t

    best_r2 = np.full(n, -np.inf)
    best_A = np.zeros(n)
    best_W_idx = np.full(n, -1, dtype=np.int32)
    best_F_idx = np.full(n, -1, dtype=np.int32)
    best_pos_r2 = np.full(n, -np.inf); best_pos_A = np.zeros(n)
    best_neg_r2 = np.full(n, -np.inf); best_neg_A = np.zeros(n)
    if n == 0:
        return GridMaxResult(best_r2, best_A, best_W_idx, best_F_idx,
                             best_pos_r2, best_pos_A, best_neg_r2, best_neg_A,
                             bank)

    power, valid = _per_w_power(a, K_arr, n)

    for wi in range(n_W):
        K = int(K_arr[wi])
        if K >= n:
            continue
        half = K // 2
        lo, hi = half, n - half
        if hi <= lo:
            continue
        inner_w = _correlate_w(a, bank.tpls[wi], K)  # [n_F, n]
        norm_w = norm_t[wi]                          # [n_F]
        # Mask outside the valid window
        # A_hat_w[f, i] = inner_w[f, i] / norm_w[f]
        safe_norm = np.where(norm_w > 1e-9, norm_w, np.nan)
        A_hat_w = inner_w / safe_norm[:, None]
        p_w = power[wi]                               # [n]
        safe_p = np.where(p_w > 1e-9, p_w, np.nan)
        ss_res = p_w[None, :] - A_hat_w * inner_w
        with np.errstate(invalid="ignore", divide="ignore"):
            r2_w = 1.0 - ss_res / safe_p[None, :]
        # Mask to valid window
        r2_w[:, :lo] = -np.inf
        r2_w[:, hi:] = -np.inf
        # Per-sample best across F at this W
        f_best = np.argmax(r2_w, axis=0)             # [n]
        idx_n = np.arange(n)
        r2_best_w = r2_w[f_best, idx_n]
        A_best_w = A_hat_w[f_best, idx_n]
        # Update global
        upd = np.isfinite(r2_best_w) & (r2_best_w > best_r2)
        best_r2[upd] = r2_best_w[upd]
        best_A[upd] = A_best_w[upd]
        best_W_idx[upd] = wi
        best_F_idx[upd] = f_best[upd]
        # Per-sign updates: per-F filter, then argmax across F restricted
        # by sign.
        # For each sample i, look across F at this W for templates with
        # A_hat sign matching ±1.
        pos_mask = A_hat_w > 0
        neg_mask = A_hat_w < 0
        # Mask r2 to only valid sign-matching cells
        r2_pos = np.where(pos_mask & np.isfinite(r2_w), r2_w, -np.inf)
        r2_neg = np.where(neg_mask & np.isfinite(r2_w), r2_w, -np.inf)
        # Per-sample best across F under the sign mask
        f_pos = np.argmax(r2_pos, axis=0)
        r2_pos_best = r2_pos[f_pos, idx_n]
        A_pos_best = A_hat_w[f_pos, idx_n]
        f_neg = np.argmax(r2_neg, axis=0)
        r2_neg_best = r2_neg[f_neg, idx_n]
        A_neg_best = A_hat_w[f_neg, idx_n]
        upd_pos = np.isfinite(r2_pos_best) & (r2_pos_best > best_pos_r2)
        best_pos_r2[upd_pos] = r2_pos_best[upd_pos]
        best_pos_A[upd_pos] = A_pos_best[upd_pos]
        upd_neg = np.isfinite(r2_neg_best) & (r2_neg_best > best_neg_r2)
        best_neg_r2[upd_neg] = r2_neg_best[upd_neg]
        best_neg_A[upd_neg] = A_neg_best[upd_neg]

    return GridMaxResult(best_r2, best_A, best_W_idx, best_F_idx,
                         best_pos_r2, best_pos_A, best_neg_r2, best_neg_A,
                         bank)


# ---------------------------------------------------------------------------
# Pair scoring — gather inner / power at sparse peak indices
# ---------------------------------------------------------------------------

def gather_inner_at_peaks(
    a: np.ndarray, bank: TemplateBank, peaks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute inner / power / valid only at the requested peak indices.

    Returns ``(inner, power, valid)`` shaped ``(n_W, n_F, K_peaks)``,
    ``(n_W, K_peaks)``, ``(n_W, K_peaks)`` respectively. For each W,
    one matrix multiply gives all F templates' inner products at the
    requested centres -- essentially the exact same computation
    ``joint_pair_score`` was doing in a Python loop, but vectorized.
    """
    n = a.size
    K_peaks = peaks.size
    n_W, n_F, _ = bank.tpls.shape
    K_arr = bank.K_arr
    K_max = bank.K_max
    half_max = K_max // 2

    inner = np.full((n_W, n_F, K_peaks), np.nan)
    power = np.full((n_W, K_peaks), np.nan)
    valid = np.zeros((n_W, K_peaks), dtype=bool)
    if K_peaks == 0 or n == 0:
        return inner, power, valid

    a2 = a * a
    csum = np.concatenate(([0.0], np.cumsum(a2)))

    for wi in range(n_W):
        K = int(K_arr[wi])
        half = K // 2
        if K >= n:
            continue
        # Which peaks have a fully-contained ±half window?
        ok = (peaks >= half) & (peaks < n - half)
        if not ok.any():
            continue
        idx_ok = np.where(ok)[0]
        peaks_ok = peaks[idx_ok]
        # Build (K_ok, K) windows by fancy indexing.
        offsets = np.arange(-half, half + 1)
        win_idx = peaks_ok[:, None] + offsets[None, :]
        windows = a[win_idx]  # [K_ok, K]
        # Templates: tpls[wi, :, half_max-half : half_max+half+1] is [n_F, K]
        offset = half_max - half
        tpls_w = bank.tpls[wi, :, offset:offset + K]  # [n_F, K]
        # Inner product: (n_F, K) @ (K, K_ok) = (n_F, K_ok)
        u_ok = tpls_w @ windows.T
        inner[wi, :, idx_ok] = u_ok.T  # idx assignment with axis=2 trick
        # Local power at each peak: cumsum range
        power[wi, idx_ok] = (
            csum[peaks_ok + half + 1] - csum[peaks_ok - half]
        )
        valid[wi, idx_ok] = power[wi, idx_ok] > 1e-9
    return inner, power, valid


def score_pair_at_peaks(
    inner: np.ndarray, power: np.ndarray, valid: np.ndarray,
    norm_t: np.ndarray,
    k1: int, k2: int, s1: float, s2: float,
) -> Optional[tuple[float, int, int, float, float, float, float]]:
    """Vectorized closed-form joint shared-shape (W, F) optimum for one pair.

    ``inner`` / ``power`` / ``valid`` come from :func:`gather_inner_at_peaks`.
    ``k1``/``k2`` are the column indices into those arrays; ``s1``/``s2``
    are the expected ±1 signs at the two lobes.

    Returns ``(score, wi*, fi*, A_abs, r2_1, r2_2, heatmap_energy)`` or
    ``None`` if no (W, F) cell satisfies the sign constraint.
    """
    n_W, n_F = norm_t.shape
    # Gather lobe vectors — (n_W, n_F)
    u1 = s1 * inner[:, :, k1]
    u2 = s2 * inner[:, :, k2]
    p1 = power[:, k1][:, None]                 # broadcast over F
    p2 = power[:, k2][:, None]
    valid_w = valid[:, k1] & valid[:, k2]      # [n_W]
    valid_grid = (valid_w[:, None]
                  & np.isfinite(u1) & np.isfinite(u2)
                  & (norm_t > 1e-9))
    sign_ok = (u1 > 0.0) & (u2 > 0.0)
    A_abs = (u1 + u2) / np.where(norm_t > 1e-9, 2.0 * norm_t, np.nan)
    A2 = A_abs * A_abs
    ss_1 = p1 - 2.0 * A_abs * u1 + A2 * norm_t
    ss_2 = p2 - 2.0 * A_abs * u2 + A2 * norm_t
    r2_1 = 1.0 - ss_1 / np.where(p1 > 1e-9, p1, np.nan)
    r2_2 = 1.0 - ss_2 / np.where(p2 > 1e-9, p2, np.nan)
    score = 0.5 * (r2_1 + r2_2)

    # Mask scoring cells: must be valid + sign-ok + A_abs > 0.
    grid_eligible = valid_grid & sign_ok & (A_abs > 0.0)
    score_masked = np.where(grid_eligible, score, -np.inf)

    if not np.isfinite(score_masked).any():
        return None

    flat_argmax = int(np.argmax(score_masked))
    wi_star, fi_star = np.unravel_index(flat_argmax, score_masked.shape)
    best_score = float(score_masked[wi_star, fi_star])
    if not np.isfinite(best_score):
        return None

    # Heatmap energy: mean of clamped joint R² across all valid cells
    # (matches the original semantics — sign-failing cells contribute 0,
    # invalid windows are dropped from the count).
    valid_for_energy = valid_grid  # contains both valid windows + finite norms
    contrib = np.where(grid_eligible & (score > 0.0), score, 0.0)
    contrib = np.where(valid_for_energy, contrib, 0.0)
    grid_cell_count = int(valid_for_energy.sum())
    heatmap_energy = float(contrib.sum() / max(grid_cell_count, 1))

    return (
        best_score,
        int(wi_star), int(fi_star),
        float(A_abs[wi_star, fi_star]),
        float(r2_1[wi_star, fi_star]),
        float(r2_2[wi_star, fi_star]),
        heatmap_energy,
    )
