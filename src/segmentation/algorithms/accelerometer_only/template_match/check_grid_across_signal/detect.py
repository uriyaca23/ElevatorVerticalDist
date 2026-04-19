"""Whole-signal trapezoid detector with (+lobe, −lobe) pair filter.

Pipeline per experiment (all in-memory — no file output):

  1. Preprocess. Build ``a_vert`` (gravity-projected vertical accel) on
     the whole session, smooth with the same ``SMOOTH_SEC`` used by the
     fitters.

  2. Detection sweep. For every ``(W, f)`` in the 30×15 grid from
     ``fit_elevator_parameters.common`` run ``match_one_template`` over
     the full signal. At each sample keep a running argmax across the
     grid:  ``best_r2[i]``, signed amplitude ``best_A[i]``, and the
     winning ``(W_idx, f_idx)``.

  3. Peak-pick. Local maxima of ``best_r2`` above ``R2_PEAK_THRESH``
     followed by NMS of ``NMS_RADIUS_S`` seconds. Each candidate carries
     its signed amplitude.

  4. Pair filter. For every (+candidate, −candidate) pair whose time gap
     lies in ``[MIN_RIDE_S, MAX_RIDE_S]``, refit a **shared-shape**
     trapezoid pair by re-searching the full ``(W, f)`` grid (closed
     form from ``constrained_grid``): the LS-optimal shared ``|A|`` and
     the per-lobe constrained R² are available analytically per pair.
     Accept if the mean of the two per-lobe R² clears
     ``JOINT_R2_THRESH``.

  5. Greedy conflict resolution. Sort accepted pairs by joint R² desc
     and commit pairs whose lobes aren't already taken.

Visualization and browsing live in ``src/data/prediction_editor.py``,
which imports :func:`compute_predictions` and runs it against the freshly
loaded sensors — no ``predictions.json`` or ``detection.png`` on disk.

Run (CLI — prints per-experiment counts, writes nothing):
    venv/bin/python src/segmentation/algorithms/accelerometer_only/\
template_match/check_grid_across_signal/detect.py                 # all TRAIN
    venv/bin/python src/segmentation/algorithms/accelerometer_only/\
template_match/check_grid_across_signal/detect.py --only <exp>    # one exp
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np  # noqa: E402

# --------------------------------------------------------------------------
# Bootstrap — load common.py by file path to avoid the broken package
# __init__ chain. Same pattern as basic_grid / constrained_grid /
# scripts/plot_failed_fits.
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_COMMON_PATH = _HERE.parent / "fit_elevator_parameters" / "common.py"
_COMMON_MOD_NAME = "_fit_ep_common"
if _COMMON_MOD_NAME in sys.modules:
    _common = sys.modules[_COMMON_MOD_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_COMMON_MOD_NAME, _COMMON_PATH)
    _common = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    sys.modules[_COMMON_MOD_NAME] = _common
    _spec.loader.exec_module(_common)

LobeFit = _common.LobeFit
GRID_W_S = _common.GRID_W_S
GRID_F = _common.GRID_F
SMOOTH_SEC = _common.SMOOTH_SEC
match_one_template = _common.match_one_template
trapezoid_kernel = _common.trapezoid_kernel
_estimate_fs_hz = _common._estimate_fs_hz
_vertical_accel = _common._vertical_accel
_smooth = _common._smooth
getExperimentData = _common.getExperimentData
list_experiments = _common.list_experiments

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
# Minimum LS-fit |amplitude| (m/s²) for a matched-filter peak to count as
# a candidate pulse. Without this, tiny sub-noise trapezoid matches on
# flat regions of the signal give r²≈1.0 with essentially random sign
# and produce a flood of bogus pairs.
MIN_PEAK_ABS_A = 0.5
R2_PEAK_THRESH = 0.80
# Small NMS right after peak-picking — prevents two candidates of *any*
# sign landing within half a second of each other (duplicate peaks from
# the same physical pulse).
NMS_RADIUS_S = 0.5
# Per-sign NMS: two same-sign candidates (two ``+`` lobes or two ``−``
# lobes) must be at least this far apart. Back-to-back elevator rides
# still have landing→take-off with opposite signs, so real physics
# doesn't put two same-sign pulses within ~30 s.
SAME_SIGN_MIN_GAP_S = 10.0
# Minimum and maximum gap between the two lobes of a ride. ``MIN_RIDE_S``
# is the opposite-sign (take-off → landing) min gap — set to 0 to let
# the pair filter accept back-to-back lobes purely on joint R² + shape.
MIN_RIDE_S = 0.0
MAX_RIDE_S = 120
JOINT_R2_THRESH = 0.75
# Shared |A| also has to be above this for a pair to be accepted.
MIN_PAIR_ABS_A = 0.5

# --------------------------------------------------------------------------
# Detection sweep
# --------------------------------------------------------------------------

def _sweep_best_template(a: np.ndarray, t: np.ndarray):
    """Per-sample argmax over the full (W, f) grid.

    Returns ``(best_r2, best_A, best_W_idx, best_f_idx, best_pos_r2,
    best_pos_A, best_neg_r2, best_neg_A)`` — each length ``len(a)``.
    The ``best_*_r2`` / ``best_*_A`` pairs are the per-sign argmax — the
    strongest template whose signed amplitude is positive / negative at
    that sample. They let the UI show *why* a sign lost even when the
    unsigned ``best_r2`` picked the other sign. Samples whose ±W window
    falls off the signal end up with ``-inf`` R² and never win.
    """
    n = a.size
    best_r2 = np.full(n, -np.inf)
    best_A = np.zeros(n)
    best_W_idx = np.full(n, -1, dtype=np.int32)
    best_f_idx = np.full(n, -1, dtype=np.int32)
    best_pos_r2 = np.full(n, -np.inf)
    best_pos_A = np.zeros(n)
    best_neg_r2 = np.full(n, -np.inf)
    best_neg_A = np.zeros(n)
    for wi, W in enumerate(GRID_W_S):
        for fi, f in enumerate(GRID_F):
            scan = match_one_template(a, t, float(W), float(f))
            r2 = scan.r2_local
            A = scan.A_hat
            mask = np.isfinite(r2) & (r2 > best_r2)
            best_r2[mask] = r2[mask]
            best_A[mask] = A[mask]
            best_W_idx[mask] = wi
            best_f_idx[mask] = fi
            pos_m = np.isfinite(r2) & (A > 0) & (r2 > best_pos_r2)
            best_pos_r2[pos_m] = r2[pos_m]
            best_pos_A[pos_m] = A[pos_m]
            neg_m = np.isfinite(r2) & (A < 0) & (r2 > best_neg_r2)
            best_neg_r2[neg_m] = r2[neg_m]
            best_neg_A[neg_m] = A[neg_m]
    return (best_r2, best_A, best_W_idx, best_f_idx,
            best_pos_r2, best_pos_A, best_neg_r2, best_neg_A)


def _peak_pick(r2: np.ndarray, thresh: float, nms_samples: int) -> list[int]:
    """Local maxima above ``thresh`` with NMS of ±``nms_samples``."""
    n = r2.size
    r2_clean = np.where(np.isfinite(r2), r2, -np.inf)
    above = np.where(r2_clean >= thresh)[0]
    if above.size == 0:
        return []
    # Local maxima filter — r2[i] >= r2[i-1] and r2[i] >= r2[i+1].
    local_max: list[int] = []
    for i in above:
        if i == 0 or i == n - 1:
            continue
        if r2_clean[i] >= r2_clean[i - 1] and r2_clean[i] >= r2_clean[i + 1]:
            local_max.append(int(i))
    # NMS — greedy by r² desc.
    local_max.sort(key=lambda j: r2_clean[j], reverse=True)
    taken = np.zeros(n, dtype=bool)
    chosen: list[int] = []
    for i in local_max:
        if taken[i]:
            continue
        chosen.append(i)
        lo = max(0, i - nms_samples)
        hi = min(n, i + nms_samples + 1)
        taken[lo:hi] = True
    chosen.sort()
    return chosen


def _same_sign_nms(
    peaks: list[int], r2: np.ndarray, signs: np.ndarray,
    t: np.ndarray, min_gap_s: float,
) -> list[int]:
    """Per-sign NMS in seconds. Keeps the highest-``r2`` candidate in
    each ``min_gap_s`` window per sign."""
    def _one_sign(ixs: list[int]) -> list[int]:
        if len(ixs) <= 1:
            return ixs
        ixs_sorted = sorted(ixs, key=lambda i: r2[i], reverse=True)
        kept: list[int] = []
        for i in ixs_sorted:
            if all(abs(t[i] - t[j]) >= min_gap_s for j in kept):
                kept.append(i)
        return sorted(kept)

    pos = [i for i in peaks if signs[i] > 0]
    neg = [i for i in peaks if signs[i] < 0]
    return sorted(_one_sign(pos) + _one_sign(neg))


# --------------------------------------------------------------------------
# Pair-stage joint-R² test
# --------------------------------------------------------------------------

def _joint_pair_score(
    a: np.ndarray, t: np.ndarray,
    i1: int, i2: int, s1: float, s2: float,
) -> tuple[float, float, float, float, float, float] | None:
    """Best shared-shape joint mean-R² across the full grid for one pair.

    Returns ``(score, W, f, A_abs, r2_1, r2_2)`` or ``None``.
    """
    n = a.size
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
    best: tuple[float, float, float, float, float, float] | None = None
    best_score = -np.inf
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
                continue
            A_abs = (u1 + u2) / (2.0 * norm_t)
            if A_abs <= 0:
                continue
            ss_1 = p1 - 2.0 * A_abs * u1 + A_abs * A_abs * norm_t
            ss_2 = p2 - 2.0 * A_abs * u2 + A_abs * A_abs * norm_t
            r2_1 = 1.0 - ss_1 / p1
            r2_2 = 1.0 - ss_2 / p2
            score = 0.5 * (r2_1 + r2_2)
            if score > best_score:
                best_score = score
                best = (score, float(W), float(f), A_abs, r2_1, r2_2)
    return best


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _predict_pairs(
    t: np.ndarray, a_smooth: np.ndarray, peaks: list[int], signs: np.ndarray,
) -> list[dict]:
    pos = [i for i in peaks if signs[i] > 0]
    neg = [i for i in peaks if signs[i] < 0]

    candidates: list[tuple[float, int, int, float, float, float, float, float, float]] = []

    def _try_pair(i1: int, i2: int, s1: float, s2: float) -> None:
        if i2 <= i1:
            return
        dt = t[i2] - t[i1]
        if dt < MIN_RIDE_S or dt > MAX_RIDE_S:
            return
        res = _joint_pair_score(a_smooth, t, i1, i2, s1, s2)
        if res is None:
            return
        score, W, f, A_abs, r2_1, r2_2 = res
        if score < JOINT_R2_THRESH:
            return
        if A_abs < MIN_PAIR_ABS_A:
            return
        candidates.append((score, i1, i2, s1, W, f, A_abs, r2_1, r2_2))

    for i1 in pos:
        for i2 in neg:
            _try_pair(i1, i2, +1.0, -1.0)
    for i1 in neg:
        for i2 in pos:
            _try_pair(i1, i2, -1.0, +1.0)

    # Greedy — accept pairs in descending score order. Enforce two
    # constraints: (a) no lobe can be reused across pairs; (b) accepted
    # time intervals must not intersect — a new ride can only start after
    # the previous one has ended.
    candidates.sort(key=lambda x: x[0], reverse=True)
    used: set[int] = set()
    accepted_ranges: list[tuple[float, float]] = []
    accepted: list[tuple[float, int, int, float, float, float, float, float, float]] = []
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
    for idx, (score, i1, i2, s1, W, f, A_abs, r2_1, r2_2) in enumerate(accepted):
        t_start = float(t[i1])
        t_end = float(t[i2])
        ride_type = "up" if s1 > 0 else "down"
        lobe1 = LobeFit(
            t_c=t_start, a_peak=float(s1 * A_abs),
            half_width_s=W, frac_flat=f, r2_local=r2_1,
        )
        lobe2 = LobeFit(
            t_c=t_end, a_peak=float(-s1 * A_abs),
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
        })
    return predictions


def preprocess_and_sweep(acc) -> dict | None:
    """Preprocess the ACC stream and run the full (W, f) sweep.

    Returns a dict with every array the pair stage (and the UI
    diagnostics) needs, or ``None`` if the input is unusable. Split out
    of :func:`compute_predictions` so the UI can reuse the sweep for
    both predictions and GT-interval diagnostics without running it twice.
    """
    if acc is None or acc.empty:
        return None
    ts_ms = acc["timestamp_ms"].to_numpy(dtype=float)
    if ts_ms.size == 0:
        return None
    t0_ms = float(ts_ms[0])
    fs = _estimate_fs_hz(ts_ms)
    t = (ts_ms - t0_ms) / 1000.0
    ax_ = acc["x"].to_numpy(dtype=float)
    ay_ = acc["y"].to_numpy(dtype=float)
    az_ = acc["z"].to_numpy(dtype=float)
    a_vert = _vertical_accel(ax_, ay_, az_, fs)
    a_smooth = _smooth(a_vert, fs, SMOOTH_SEC)

    (best_r2, best_A, best_W_idx, best_f_idx,
     best_pos_r2, best_pos_A, best_neg_r2, best_neg_A) = _sweep_best_template(
        a_smooth, t,
    )

    nms_samples = max(1, int(round(NMS_RADIUS_S * fs)))
    amp_gate = np.abs(best_A) >= MIN_PEAK_ABS_A
    best_r2_gated = np.where(amp_gate, best_r2, -np.inf)
    initial_peaks = _peak_pick(best_r2_gated, R2_PEAK_THRESH, nms_samples)
    signs = np.sign(best_A)
    final_peaks = _same_sign_nms(
        initial_peaks, best_r2_gated, signs, t, SAME_SIGN_MIN_GAP_S,
    )

    return {
        "t0_ms": t0_ms,
        "fs": fs,
        "t": t,
        "a_vert": a_vert,
        "a_smooth": a_smooth,
        "best_r2": best_r2,
        "best_A": best_A,
        "best_W_idx": best_W_idx,
        "best_f_idx": best_f_idx,
        "best_pos_r2": best_pos_r2,
        "best_pos_A": best_pos_A,
        "best_neg_r2": best_neg_r2,
        "best_neg_A": best_neg_A,
        "best_r2_gated": best_r2_gated,
        "signs": signs,
        "initial_peaks": initial_peaks,
        "final_peaks": final_peaks,
    }


def compute_predictions(acc) -> list[dict]:
    """Run the full detector + pair filter on one ACC DataFrame."""
    state = preprocess_and_sweep(acc)
    if state is None:
        return []
    return _predict_pairs(
        state["t"], state["a_smooth"], state["final_peaks"], state["signs"],
    )


def _find_extrema_in_window(
    state: dict, t_lo: float, t_hi: float,
) -> tuple[tuple[int, float, float] | None, tuple[int, float, float] | None]:
    """Return ``((pos_idx, pos_A, pos_r2), (neg_idx, neg_A, neg_r2))`` —
    the sample with largest positive ``best_A`` and the sample with the
    most negative ``best_A`` in ``[t_lo, t_hi]``. Either side can be
    ``None`` if the window has no same-sign sample with finite R².
    """
    t = state["t"]
    mask = (t >= t_lo) & (t <= t_hi)
    if not mask.any():
        return None, None
    best_A = state["best_A"]
    best_r2 = state["best_r2"]
    idxs = np.where(mask & np.isfinite(best_r2))[0]
    if idxs.size == 0:
        return None, None
    pos_idxs = idxs[best_A[idxs] > 0]
    neg_idxs = idxs[best_A[idxs] < 0]
    pos = None
    neg = None
    if pos_idxs.size:
        j = int(pos_idxs[np.argmax(best_A[pos_idxs])])
        pos = (j, float(best_A[j]), float(best_r2[j]))
    if neg_idxs.size:
        j = int(neg_idxs[np.argmin(best_A[neg_idxs])])
        neg = (j, float(best_A[j]), float(best_r2[j]))
    return pos, neg


def diagnose_window(
    state: dict, t_lo: float, t_hi: float, ride_type: str | None = None,
) -> dict:
    """Explain what happened inside ``[t_lo, t_hi]`` w.r.t. the detector.

    Meant for GT-interval diagnostics in the UI: given a GT ride window
    that did (or didn't) get detected, report the best positive- and
    negative-sign matches found inside it, whether each clears the peak
    thresholds, and — if both sides exist — the shared-shape joint fit.
    Each failure mode is tagged in ``verdict_lines`` so the user can see
    exactly which tunable would have let the ride through.
    """
    t = state["t"]
    a_smooth = state["a_smooth"]

    pos, neg = _find_extrema_in_window(state, t_lo, t_hi)

    # For an explicit ride_type, order the two lobes chronologically so
    # we always test the right pairing direction.
    if ride_type == "up":
        first, second, s1, s2 = pos, neg, +1.0, -1.0
    elif ride_type == "down":
        first, second, s1, s2 = neg, pos, -1.0, +1.0
    else:
        # Pair direction by time if type unknown.
        if pos and neg and pos[0] < neg[0]:
            first, second, s1, s2 = pos, neg, +1.0, -1.0
        else:
            first, second, s1, s2 = neg, pos, -1.0, +1.0

    lines: list[str] = []

    # Peak-threshold checks.
    def _peak_line(tag: str, peak):
        if peak is None:
            return f"  {tag} lobe: no sample with that sign in the window."
        i, A, r2 = peak
        flags = []
        if not (r2 >= R2_PEAK_THRESH):
            flags.append(f"R²={r2:.2f} < {R2_PEAK_THRESH:.2f}")
        if not (abs(A) >= MIN_PEAK_ABS_A):
            flags.append(f"|A|={abs(A):.2f} < {MIN_PEAK_ABS_A:.2f}")
        ok = "OK" if not flags else "FAIL"
        reasons = " & ".join(flags) if flags else "clears thresholds"
        return (
            f"  {tag} lobe: t={t[i]:.1f}s  A={A:+.2f}  R²={r2:.2f}  [{ok}: {reasons}]"
        )

    lines.append(_peak_line("+", pos))
    lines.append(_peak_line("−", neg))

    # Pair-stage check if both peaks exist and order is physical.
    pair_info: dict | None = None
    if first is not None and second is not None:
        i1, _, _ = first
        i2, _, _ = second
        if i1 >= i2:
            lines.append("  pair: requested lobes are not in chronological order.")
        else:
            gap = float(t[i2] - t[i1])
            res = _joint_pair_score(a_smooth, t, i1, i2, s1, s2)
            if res is None:
                lines.append(
                    f"  pair: joint fit unavailable (window too short or no sign match)."
                )
            else:
                score, W, f, A_abs, r2_1, r2_2 = res
                flags = []
                if not (MIN_RIDE_S <= gap <= MAX_RIDE_S):
                    flags.append(
                        f"gap={gap:.1f}s outside [{MIN_RIDE_S}, {MAX_RIDE_S}]"
                    )
                if not (score >= JOINT_R2_THRESH):
                    flags.append(
                        f"joint R²={score:.2f} < {JOINT_R2_THRESH:.2f}"
                    )
                if not (A_abs >= MIN_PAIR_ABS_A):
                    flags.append(
                        f"pair |A|={A_abs:.2f} < {MIN_PAIR_ABS_A:.2f}"
                    )
                ok = "accepted" if not flags else "rejected"
                reasons = "; ".join(flags) if flags else "all thresholds pass"
                lines.append(
                    f"  pair: gap={gap:.1f}s  W={W:.2f}  f={f:.2f}  "
                    f"|A|={A_abs:.2f}  R²={score:.3f}  [{ok}: {reasons}]"
                )
                pair_info = {
                    "i1": int(i1), "i2": int(i2),
                    "t_c1": float(t[i1]), "t_c2": float(t[i2]),
                    "W": float(W), "frac_flat": float(f),
                    "A_abs": float(A_abs),
                    "r2_1": float(r2_1), "r2_2": float(r2_2),
                    "joint_r2_mean": float(score),
                    "gap_s": gap,
                    "reject_flags": flags,
                }
    else:
        lines.append("  pair: need both a + peak and a − peak to even attempt.")

    return {
        "t_lo": t_lo, "t_hi": t_hi,
        "ride_type": ride_type,
        "pos_peak": pos,
        "neg_peak": neg,
        "first_peak": first, "second_peak": second,
        "pair": pair_info,
        "verdict_lines": lines,
    }


def main() -> int:
    """CLI sanity check — prints gt/pred counts per experiment, writes nothing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="process a single experiment by name")
    args = parser.parse_args()

    names = [args.only] if args.only else list_experiments(kind="train")
    print(f"running detector on {len(names)} experiments")
    total_gt = 0
    total_pred = 0
    rows: list[tuple[str, int, int]] = []
    for n in names:
        try:
            sensors, gt, _meta = getExperimentData(n)
        except Exception as exc:
            print(f"[error] {n}: {type(exc).__name__}: {exc}")
            continue
        preds = compute_predictions(sensors.get("ACC"))
        n_gt = int(gt["type"].isin(("up", "down")).sum()) if gt is not None else 0
        total_gt += n_gt
        total_pred += len(preds)
        rows.append((n, n_gt, len(preds)))
        print(f"[ok]    {n}: gt={n_gt}  pred={len(preds)}")

    if rows:
        print(
            f"\n{len(rows)} experiments — "
            f"GT total {total_gt}, predicted total {total_pred} "
            f"(pred/gt = {total_pred / max(total_gt, 1):.2f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
