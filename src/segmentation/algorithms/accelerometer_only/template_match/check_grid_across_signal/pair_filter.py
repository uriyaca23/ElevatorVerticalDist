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
    GRID_W_S, GRID_F, LobeFit, trapezoid_kernel,
)


def joint_pair_score(
    a: np.ndarray, t: np.ndarray,
    i1: int, i2: int, s1: float, s2: float,
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
    """
    n = a.size
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
    best: tuple[float, float, float, float, float, float] | None = None
    best_score = -np.inf
    grid_score_sum = 0.0
    grid_cell_count = 0
    for W in GRID_W_S:
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
        for f in GRID_F:
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

    pos = [i for i in peaks if signs[i] > 0]
    neg = [i for i in peaks if signs[i] < 0]

    candidates: list[
        tuple[float, int, int, float, float, float, float, float, float, float]
    ] = []

    def _try_pair(i1: int, i2: int, s1: float, s2: float) -> None:
        if i2 <= i1:
            return
        gap = t[i2] - t[i1]
        if gap < config.min_ride_s or gap > config.max_ride_s:
            return
        res = joint_pair_score(a_smooth, t, i1, i2, s1, s2)
        if res is None:
            return
        score, W, f, A_abs, r2_1, r2_2, heatmap_energy = res
        if score < config.joint_r2_thresh:
            return
        if A_abs < config.min_pair_abs_a:
            return
        if heatmap_energy < config.heatmap_energy_thresh:
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
        # t_start = centre1 − W (take-off pulse left edge), t_end =
        # centre2 + W (landing pulse right edge).
        t_start = t_c1 - float(W)
        t_end = t_c2 + float(W)
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
