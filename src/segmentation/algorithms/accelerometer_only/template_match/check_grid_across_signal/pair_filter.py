"""Pair-filter — the "clearing algorithm" that turns raw detection peaks
into final interval predictions.

Inputs from :mod:`detect` (stages 1–4):
    state dict with ``t``, ``a_smooth``, ``final_peaks``, ``signs``.

Pipeline:
    * For every ``(+peak, -peak)`` pair whose gap is in
      ``[min_ride_s, max_ride_s]``, compute the best shared-shape joint
      fit via :func:`joint_pair_score` over the full ``(W, f)`` grid.
    * Accept if joint R² ≥ ``joint_r2_thresh`` and shared ``|A|`` ≥
      ``min_pair_abs_a``.
    * Greedy conflict resolution: sort accepted candidates by joint R²
      desc, commit each pair only if neither lobe is already taken and
      the pair's interval does not intersect any already-accepted one.

Kept as a standalone module so :mod:`detect` owns the detection
primitives and this module owns the ride-segment decision logic.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..fit_elevator_parameters.common import (
    LobeFit, trapezoid_kernel,
)
from src.utils.trapezoid_fast import (
    build_template_bank, gather_inner_at_peaks, score_pair_at_peaks,
)


def joint_pair_score(
    a: np.ndarray, t: np.ndarray,
    i1: int, i2: int, s1: float, s2: float,
    grid_w_s: np.ndarray | None = None,
    grid_f: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, float, float] | None:
    """Best shared-shape joint mean-R² across the full ``(W, f)`` grid
    for one pair. Returns
    ``(score, W, f, A_abs, r2_1, r2_2, heatmap_energy)`` or ``None``
    if the window is unusable.

    ``heatmap_energy`` is the mean of ``max(joint_R², 0)`` over every
    valid ``(W, f)`` cell — i.e. how broadly the grid supports the
    match. A true elevator ride lights up a wide band of templates; a
    spurious spike matches only a narrow sliver, producing a mostly
    dark heatmap and low energy.

    ``grid_w_s`` / ``grid_f`` default to :data:`detect.DEFAULT_CONFIG`'s
    grid so external callers (dump-mistakes, editor, diagnose_window)
    without a config still work. :func:`predict_pairs` passes the grid
    the detector actually used.
    """
    if grid_w_s is None or grid_f is None:
        from .detect import DEFAULT_CONFIG
        if grid_w_s is None:
            grid_w_s = DEFAULT_CONFIG.grid_w_s()
        if grid_f is None:
            grid_f = DEFAULT_CONFIG.grid_f()
    n = a.size
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
    best: tuple[float, float, float, float, float, float] | None = None
    best_score = -np.inf
    grid_score_sum = 0.0
    grid_cell_count = 0
    for W in grid_w_s:
        K = max(3, int(round(2 * W / dt)))
        if K % 2 == 0:
            K += 1
        half = K // 2
        if i1 - half < 0 or i1 + half >= n or i2 - half < 0 or i2 + half >= n:
            continue
        win1 = a[i1 - half: i1 + half + 1]
        win2 = a[i2 - half: i2 + half + 1]
        p1 = float(np.sum(win1 * win1))
        p2 = float(np.sum(win2 * win2))
        if p1 < 1e-9 or p2 < 1e-9:
            continue
        t_k = (np.arange(K) - half) * dt
        for f in grid_f:
            tpl = trapezoid_kernel(t_k, 0.0, float(W), float(f))
            norm_t = float(np.sum(tpl * tpl))
            if norm_t < 1e-9:
                continue
            inner_1 = float(np.dot(win1, tpl))
            inner_2 = float(np.dot(win2, tpl))
            u1 = s1 * inner_1
            u2 = s2 * inner_2
            if u1 <= 0 or u2 <= 0:
                # Cell doesn't support the requested sign pattern —
                # count it as zero energy, not "missing".
                grid_score_sum += 0.0
                grid_cell_count += 1
                continue
            A_abs = (u1 + u2) / (2.0 * norm_t)
            if A_abs <= 0:
                grid_score_sum += 0.0
                grid_cell_count += 1
                continue
            ss_1 = p1 - 2.0 * A_abs * u1 + A_abs * A_abs * norm_t
            ss_2 = p2 - 2.0 * A_abs * u2 + A_abs * A_abs * norm_t
            r2_1 = 1.0 - ss_1 / p1
            r2_2 = 1.0 - ss_2 / p2
            score = 0.5 * (r2_1 + r2_2)
            grid_score_sum += score if score > 0.0 else 0.0
            grid_cell_count += 1
            if score > best_score:
                best_score = score
                best = (score, float(W), float(f), A_abs, r2_1, r2_2)
    if best is None:
        return None
    heatmap_energy = (
        grid_score_sum / grid_cell_count if grid_cell_count > 0 else 0.0
    )
    return (*best, float(heatmap_energy))


def predict_pairs(state: dict, config) -> list[dict]:
    """Clear the detection peaks into final interval predictions.

    ``state`` must contain ``t``, ``a_smooth``, ``final_peaks``, ``signs``
    (all from :func:`detect.detect`). ``config`` is a :class:`DetectConfig`
    — only its pair-filter fields are consulted.
    """
    t = state["t"]
    a_smooth = state["a_smooth"]
    peaks = state["final_peaks"]
    signs = state["signs"]
    # Reuse the grid the detector built; fall back to config if the
    # state came from an older caller that didn't store it.
    grid_w_s = state.get("grid_w_s")
    if grid_w_s is None:
        grid_w_s = config.grid_w_s()
    grid_f = state.get("grid_f")
    if grid_f is None:
        grid_f = config.grid_f()

    pos = [i for i in peaks if signs[i] > 0]
    neg = [i for i in peaks if signs[i] < 0]

    # Vectorized matched-filter precompute. Build the template bank once
    # for the signal's sample step, then batch-gather inner / power /
    # validity at every candidate peak — score_pair_at_peaks then becomes
    # a closed-form per-pair lookup instead of recomputing 30×16 templates
    # per pair (the original joint_pair_score hot path).
    fast_state = state.get("_fast_pair_state")
    peaks_arr = np.asarray(peaks, dtype=np.int64)
    if fast_state is None or fast_state.get("peaks_id") is not id(peaks):
        dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
        bank = build_template_bank(np.asarray(grid_w_s), np.asarray(grid_f), dt)
        inner_pk, power_pk, valid_pk = gather_inner_at_peaks(
            a_smooth, bank, peaks_arr,
        )
        # Index map peak_index_in_signal -> column in inner_pk arrays.
        peak_to_col = {int(p): k for k, p in enumerate(peaks_arr)}
        fast_state = {
            "bank": bank, "inner_pk": inner_pk, "power_pk": power_pk,
            "valid_pk": valid_pk, "peak_to_col": peak_to_col,
            "peaks_id": id(peaks),
        }
        # Cache on state so :func:`diagnose_window` and other callers
        # that reuse the same state dict don't pay the cost twice.
        state["_fast_pair_state"] = fast_state
    bank = fast_state["bank"]
    inner_pk = fast_state["inner_pk"]
    power_pk = fast_state["power_pk"]
    valid_pk = fast_state["valid_pk"]
    peak_to_col = fast_state["peak_to_col"]

    candidates: list[
        tuple[float, int, int, float, float, float, float, float, float, float]
    ] = []

    def _try_pair(i1: int, i2: int, s1: float, s2: float) -> None:
        if i2 <= i1:
            return
        gap = t[i2] - t[i1]
        if gap < config.min_ride_s or gap > config.max_ride_s:
            return
        k1 = peak_to_col.get(int(i1))
        k2 = peak_to_col.get(int(i2))
        if k1 is None or k2 is None:
            return
        res = score_pair_at_peaks(
            inner_pk, power_pk, valid_pk, bank.norm_t,
            k1=k1, k2=k2, s1=s1, s2=s2,
        )
        if res is None:
            return
        score, wi_star, fi_star, A_abs, r2_1, r2_2, heatmap_energy = res
        W = float(bank.grid_W[wi_star])
        f = float(bank.grid_F[fi_star])
        if score < config.joint_r2_thresh:
            return
        if A_abs < config.min_pair_abs_a:
            return
        if heatmap_energy < config.heatmap_energy_thresh:
            return
        # Quiet-middle check: between the two lobes the cabin cruises at
        # constant velocity, so a_smooth should be close to zero. Walking
        # FPs have continuous motion; their middle RMS is as large as the
        # lobe amplitude. Reject pairs whose middle isn't quiet.
        qmr = getattr(config, "quiet_middle_ratio", 1.0)
        if qmr < 1.0:
            mid_lo_t = t[i1] + W
            mid_hi_t = t[i2] - W
            if mid_hi_t > mid_lo_t:
                mid_mask = (t >= mid_lo_t) & (t <= mid_hi_t)
                if mid_mask.any():
                    mid = a_smooth[mid_mask]
                    mid_rms = float(np.sqrt(np.mean(mid * mid)))
                    if mid_rms > qmr * A_abs:
                        return
        candidates.append(
            (score, i1, i2, s1, W, f, A_abs, r2_1, r2_2, heatmap_energy)
        )

    for i1 in pos:
        for i2 in neg:
            _try_pair(i1, i2, +1.0, -1.0)
    for i1 in neg:
        for i2 in pos:
            _try_pair(i1, i2, -1.0, +1.0)

    # Greedy conflict resolution — accept pairs in descending
    # (score − duration-penalty) order, rejecting any that share a
    # lobe with, or overlap in time with, an already-accepted pair.
    #
    # The duration penalty (λ = 0.01 per second) was chosen by the
    # pair-filter iteration sweep under ``pair_filter_iterations/`` —
    # see iter_04_dur_penalty_heavy. It broke the baseline's "super
    # pair" failure mode, where a take-off from ride 1 and a landing
    # from ride 6 paired at a high shared-shape R² and swallowed every
    # GT ride between them. Cost: it biases toward short gaps too far,
    # which brings in a smaller back-to-back-dwell failure mode (see
    # the iteration log for next-step ideas — band penalty, min-gap
    # floor, time-sorted greedy).
    _DURATION_PENALTY_LAMBDA = 0.01
    candidates.sort(
        key=lambda c: c[0] - _DURATION_PENALTY_LAMBDA * float(t[c[2]] - t[c[1]]),
        reverse=True,
    )
    used: set[int] = set()
    accepted_ranges: list[tuple[float, float]] = []
    accepted: list[
        tuple[float, int, int, float, float, float, float, float, float, float]
    ] = []
    for cand in candidates:
        _score, i1, i2, *_ = cand
        if i1 in used or i2 in used:
            continue
        ts, te = (t[i1], t[i2]) if t[i1] < t[i2] else (t[i2], t[i1])
        if any(not (te <= a_s or ts >= a_e) for a_s, a_e in accepted_ranges):
            continue
        used.add(i1)
        used.add(i2)
        accepted_ranges.append((ts, te))
        accepted.append(cand)
    accepted.sort(key=lambda x: x[1])  # chronological

    predictions: list[dict] = []
    for idx, (score, i1, i2, s1, W, f, A_abs, r2_1, r2_2, heatmap_energy) in enumerate(accepted):
        t_c1 = float(t[i1])
        t_c2 = float(t[i2])
        # Ride endpoints span the whole trapezoid pulse on each side:
        # t_start = centre1 − W − ε (take-off pulse left edge, padded),
        # t_end   = centre2 + W + ε (landing pulse right edge, padded).
        # The ε padding prevents downstream integrators (ZUPT,
        # trapezoid_accel) from clipping the acceleration tails. See the
        # ε-sweep under improvement_iterations/_sweep_epsilon.py.
        eps = float(getattr(config, "segment_pad_eps_s", 0.0))
        t_start = t_c1 - float(W) - eps
        t_end = t_c2 + float(W) + eps
        ride_type = "up" if s1 > 0 else "down"
        lobe1 = LobeFit(
            t_c=t_c1, a_peak=float(s1 * A_abs),
            half_width_s=W, frac_flat=f, r2_local=r2_1,
        )
        lobe2 = LobeFit(
            t_c=t_c2, a_peak=float(-s1 * A_abs),
            half_width_s=W, frac_flat=f, r2_local=r2_2,
        )
        predictions.append({
            "index": idx,
            "ride_type": ride_type,
            "t_start_s": t_start,
            "t_end_s": t_end,
            "duration_s": t_end - t_start,
            "lobe1": asdict(lobe1),
            "lobe2": asdict(lobe2),
            "joint_r2_mean": float(score),
            "heatmap_energy": float(heatmap_energy),
        })
    return predictions
