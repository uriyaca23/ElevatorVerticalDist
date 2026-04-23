"""Whole-signal trapezoid detector — detection stage + top-level wrapper.

Two halves:

* **Detection** (this file). Preprocess the ACC stream, run the
  ``(W, f)`` template sweep over the full smoothed signal, then distill
  the per-sample argmax arrays to a set of signed candidate peaks via
  R² + |A| thresholds and two NMS passes (small dedup + same-sign).
  Output: a state dict (``detect`` function).

* **Clearing algorithm** (see :mod:`pair_filter`). Promotes a subset of
  those candidates to final interval predictions via a shared-shape
  joint-R² fit and a greedy conflict resolver.

Top-level wrapper :func:`predict_intervals` runs both and returns
``(predictions, plotting_info)`` — the second item is the state dict, so
the editor UI and :func:`diagnose_window` can reuse it for inspection.

All tunables live in :class:`DetectConfig`. Callers pass a config
instance rather than mutating module globals.

CLI (prints per-experiment counts, writes nothing):
    venv/bin/python src/.../check_grid_across_signal/detect.py
    venv/bin/python src/.../check_grid_across_signal/detect.py --only <exp>
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace

import numpy as np

from src.utils.sensor_noise import get_phone_accel_noise_sigma

from ..fit_elevator_parameters.common import (
    SMOOTH_SEC,
    match_one_template, trapezoid_kernel,
    _estimate_fs_hz, _vertical_accel, _smooth,
    getExperimentData, list_experiments,
)
from . import pair_filter


# --------------------------------------------------------------------------
# Config — the single knob bag. Pass this to :func:`predict_intervals`.
# --------------------------------------------------------------------------
@dataclass
class DetectConfig:
    """All detector tunables bundled.

    Peak-pick / same-sign NMS (stages 3–4, this module):

    * ``r2_peak_thresh``       min unsigned R² at peak-pick.
    * ``min_peak_abs_a``       amplitude floor for a peak candidate
                               (``|A_hat|`` threshold at the peak sample).
    * ``nms_radius_s``         small same-sample NMS dedup radius (s).
    * ``same_sign_min_gap_s``  min gap between two same-sign candidates.

    Pair-filter (stages 5–6, :mod:`pair_filter`):

    * ``min_ride_s``             min (take-off → landing) gap of a ride.
    * ``max_ride_s``             max gap — any longer and we reject the pair.
    * ``joint_r2_thresh``        min shared-shape joint R² to accept a pair.
    * ``min_pair_abs_a``         min shared |A| of the accepted pair.
    * ``heatmap_energy_thresh``  min pair heatmap energy — the mean of
                                 clamped joint R² across the full (W, f)
                                 grid. Rejects pairs where only a narrow
                                 sliver of the grid supports the match
                                 (mostly-dark heatmaps in the editor UI).

    ``(W, f)`` trapezoid-template grid (used by detection, pair filter
    and heatmap visualisation). ``grid_w_s()`` / ``grid_f()`` build the
    linspaces on demand — sweep by overriding these on the config, not
    by patching :mod:`..fit_elevator_parameters.common`.

    * ``w_min_s`` / ``w_max_s`` / ``n_w``  bounds + count of the half-width axis.
    * ``f_min`` / ``f_max`` / ``n_f``      bounds + count of the flat-fraction axis.

    Phone-aware noise floor (opt-in via ``phone_model`` on
    :func:`detect` / :func:`predict_intervals`):

    * ``noise_sigma_multiplier`` — when a phone model is supplied, the
      per-peak and per-pair ``|A|`` floors are lifted to
      ``max(min_peak_abs_a, multiplier · σ_a)`` (and analogously for the
      pair floor), where ``σ_a`` is the phone's accelerometer white-noise
      σ at the session's sampling rate (see
      :mod:`src.utils.sensor_noise`). A multiplier of ~6 is a standard
      "well above noise" gate. Hard-coded floors win when the phone is
      unknown or noisier than expected — this is a tightening knob, not
      a loosening one.
    """
    # Defaults below are the combined-best 1-D sweep winners
    # (see ``scripts/sweep_acc_segmentation.py`` +
    # ``elevator_reports/seg_acc_sweep/summary.json``).
    r2_peak_thresh: float = 0.55

    min_peak_abs_a: float = 0.4

    nms_radius_s: float = 1.0


    same_sign_min_gap_s: float = 5.0
    min_ride_s: float = 0.0
    max_ride_s: float = 30.0


    joint_r2_thresh: float = 0.90
    min_pair_abs_a: float = 0.30
    heatmap_energy_thresh: float = 0.40

    # Quiet-middle filter (iter_04): reject pairs whose plateau between the
    # two lobes is not quiet. Real elevator rides cruise at constant
    # velocity so a_vert ≈ 0 in the middle; walking FPs have continuous
    # motion and their middle RMS is comparable to the lobe amplitude.
    # Filter fires iff quiet_middle_rms(middle) > ratio * pair_A_abs. Set
    # ratio >= 1.0 to disable.
    quiet_middle_ratio: float = 0.5

    w_min_s: float = 0.4
    w_max_s: float = 3.0
    n_w: int = 30
    f_min: float = 0.05
    f_max: float = 0.80
    n_f: int = 15

    noise_sigma_multiplier: float = 6.0

    def grid_w_s(self) -> np.ndarray:
        return np.linspace(self.w_min_s, self.w_max_s, self.n_w)

    def grid_f(self) -> np.ndarray:
        return np.linspace(self.f_min, self.f_max, self.n_f)


DEFAULT_CONFIG = DetectConfig()


# --------------------------------------------------------------------------
# Detection — stages 1–4
# --------------------------------------------------------------------------

def _sweep_best_template(
    a: np.ndarray, t: np.ndarray,
    grid_w_s: np.ndarray, grid_f: np.ndarray,
):
    """Per-sample argmax over the full ``(W, f)`` grid.

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
    for wi, W in enumerate(grid_w_s):
        for fi, f in enumerate(grid_f):
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
    local_max: list[int] = []
    for i in above:
        if i == 0 or i == n - 1:
            continue
        if r2_clean[i] >= r2_clean[i - 1] and r2_clean[i] >= r2_clean[i + 1]:
            local_max.append(int(i))
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
    """Per-sign NMS in seconds — highest-``r2`` wins each ``min_gap_s``
    window; two same-sign candidates end up no closer than that."""
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


def _apply_phone_noise_floor(
    config: DetectConfig, phone_model: str, fs: float,
) -> tuple[DetectConfig, float]:
    """Tighten the amplitude floors to the phone's noise σ when known.

    Returns ``(effective_config, sigma_a)``. When ``phone_model`` is
    empty the config is returned unchanged and ``sigma_a = 0.0``.
    """
    if not phone_model:
        return config, 0.0
    sigma_a = float(get_phone_accel_noise_sigma(phone_model, fs))
    floor = config.noise_sigma_multiplier * sigma_a
    return replace(
        config,
        min_peak_abs_a=max(config.min_peak_abs_a, floor),
        min_pair_abs_a=max(config.min_pair_abs_a, floor),
    ), sigma_a


def detect(
    acc, config: DetectConfig = DEFAULT_CONFIG, phone_model: str = "",
) -> dict | None:
    """Run detection stages 1–4. Returns the state dict or ``None`` if
    ``acc`` is empty/unusable.

    When ``phone_model`` is provided, the ``|A|`` floors are tightened
    to the phone's accelerometer noise σ (see
    :func:`_apply_phone_noise_floor`); the resulting effective config is
    echoed in ``state["config"]`` so downstream code and the editor UI
    see exactly the thresholds used.

    Dict keys (everything the editor UI, :mod:`pair_filter` and
    :func:`diagnose_window` consume):

    * inputs / preprocess: ``t0_ms``, ``fs``, ``t``, ``a_vert``, ``a_smooth``
    * per-sample sweep:    ``best_r2``, ``best_A``, ``best_W_idx``,
      ``best_f_idx``, ``best_pos_r2``, ``best_pos_A``, ``best_neg_r2``,
      ``best_neg_A``
    * peak-pick:           ``best_r2_gated``, ``signs``, ``initial_peaks``,
      ``final_peaks``
    * phone-aware:         ``phone_model``, ``sigma_a_m_s2``
    * echo:                ``config`` — effective config after phone floor
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

    config, sigma_a = _apply_phone_noise_floor(config, phone_model, fs)
    grid_w_s = config.grid_w_s()
    grid_f = config.grid_f()

    (best_r2, best_A, best_W_idx, best_f_idx,
     best_pos_r2, best_pos_A, best_neg_r2, best_neg_A) = _sweep_best_template(
        a_smooth, t, grid_w_s, grid_f,
    )

    nms_samples = max(1, int(round(config.nms_radius_s * fs)))
    amp_gate = np.abs(best_A) >= config.min_peak_abs_a
    best_r2_gated = np.where(amp_gate, best_r2, -np.inf)
    initial_peaks = _peak_pick(best_r2_gated, config.r2_peak_thresh, nms_samples)
    signs = np.sign(best_A)
    final_peaks = _same_sign_nms(
        initial_peaks, best_r2_gated, signs, t, config.same_sign_min_gap_s,
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
        "grid_w_s": grid_w_s,
        "grid_f": grid_f,
        "config": config,
        "phone_model": phone_model,
        "sigma_a_m_s2": sigma_a,
    }


# --------------------------------------------------------------------------
# Top-level wrapper — detection + pair filter
# --------------------------------------------------------------------------

def predict_intervals(
    acc, config: DetectConfig | None = None, phone_model: str = "",
) -> tuple[list[dict], dict]:
    """Full pipeline. Returns ``(predictions, plotting_info)``.

    * ``predictions`` — list of dicts with ``t_start_s``, ``t_end_s``,
      ``ride_type``, ``duration_s``, ``joint_r2_mean``, plus the two
      lobe fits. Schema mirrors
      :class:`fit_elevator_parameters.common.RideFit`.
    * ``plotting_info`` — the detection state dict (see :func:`detect`);
      contains every intermediate the editor / :func:`diagnose_window`
      needs. Empty dict if ``acc`` is unusable.

    ``phone_model`` (optional) enables the chip-spec-derived noise floor
    on the ``|A|`` thresholds — see :func:`detect`.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    state = detect(acc, cfg, phone_model=phone_model)
    if state is None:
        return [], {}
    # Use the effective (phone-aware) config that detect emitted.
    predictions = pair_filter.predict_pairs(state, state["config"])
    return predictions, state


# --------------------------------------------------------------------------
# UI helpers — pulled out of editor.py so it stays display-only. Pure
# numpy over :func:`detect` state; no Tk / matplotlib dependencies.
# --------------------------------------------------------------------------

def heatmap_at(
    a: np.ndarray, t: np.ndarray, i_center: int,
    grid_w_s: np.ndarray | None = None,
    grid_f: np.ndarray | None = None,
) -> np.ndarray:
    """``(nW, nF)`` local R² of every ``(W, f)`` template at sample
    ``i_center``. NaN cells = template window falls off the signal or
    local signal power is zero. Used to render the detail-panel
    heatmap in the editor.

    ``grid_w_s`` / ``grid_f`` default to :data:`DEFAULT_CONFIG`'s grid
    so external callers without a config still work; the editor and
    dump-mistakes pipelines pass the grid the detector actually used
    (``state["grid_w_s"]`` / ``state["grid_f"]``)."""
    if grid_w_s is None:
        grid_w_s = DEFAULT_CONFIG.grid_w_s()
    if grid_f is None:
        grid_f = DEFAULT_CONFIG.grid_f()
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
    n = a.size
    nW, nF = len(grid_w_s), len(grid_f)
    out = np.full((nW, nF), np.nan)
    for wi, W in enumerate(grid_w_s):
        K = max(3, int(round(2 * W / dt)))
        if K % 2 == 0:
            K += 1
        half = K // 2
        if i_center - half < 0 or i_center + half >= n:
            continue
        win = a[i_center - half: i_center + half + 1]
        p = float(np.dot(win, win))
        if p < 1e-9:
            continue
        t_k = (np.arange(K) - half) * dt
        for fi, f in enumerate(grid_f):
            tpl = trapezoid_kernel(t_k, 0.0, float(W), float(f))
            norm_t = float(np.dot(tpl, tpl))
            if norm_t < 1e-9:
                continue
            inner = float(np.dot(win, tpl))
            A_hat = inner / norm_t
            out[wi, fi] = 1.0 - (p - A_hat * inner) / p
    return out


def find_local_maxima(
    arr: np.ndarray, t: np.ndarray,
    t_lo: float, t_hi: float,
    min_val: float = 0.5, min_gap_s: float = 1.0,
) -> list[int]:
    """Strict interior local maxima of ``arr`` within ``[t_lo, t_hi]``
    above ``min_val``, with greedy time-gap NMS. The editor picks
    signed-R² annotation dots with this — caller-defined threshold and
    gap, not the detector config."""
    n = arr.size
    mask = (t >= t_lo) & (t <= t_hi) & np.isfinite(arr)
    cands: list[int] = []
    for i in np.where(mask)[0]:
        if i <= 0 or i >= n - 1:
            continue
        if arr[i] < min_val:
            continue
        if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
            cands.append(int(i))
    cands.sort(key=lambda j: arr[j], reverse=True)
    kept: list[int] = []
    for i in cands:
        if all(abs(t[i] - t[j]) >= min_gap_s for j in kept):
            kept.append(i)
    kept.sort()
    return kept


# Signed-R² peak classification — status tags in pipeline order. The
# editor uses them as legend keys; the first failing stage wins.
PEAK_STATUS_ACCEPTED = "accepted"
PEAK_STATUS_UNPAIRED = "unpaired (greedy)"
PEAK_STATUS_SAME_SIGN_NMS = "same-sign NMS"
PEAK_STATUS_LOCAL_NMS = "NMS (local)"
PEAK_STATUS_OPP_SIGN = "lost to opp sign"
PEAK_STATUS_LOW_R2 = "R²<thr"
PEAK_STATUS_LOW_A = "|A|<thr"


def classify_peak(
    state: dict, i: int, sign: int,
    predictions: list[dict],
    config: DetectConfig | None = None,
) -> str:
    """Status tag for the signed-R² peak at sample ``i``. Walks the
    stages in pipeline order; returns the first stage that kept or
    dropped the peak. ``sign`` is ``+1`` / ``-1`` — picks the per-sign
    ``best_*_A`` / ``best_*_r2`` arrays."""
    cfg = config if config is not None else state.get("config", DEFAULT_CONFIG)
    t = state["t"]
    A_field = "best_pos_A" if sign > 0 else "best_neg_A"
    R2_field = "best_pos_r2" if sign > 0 else "best_neg_r2"
    A_hat = float(state[A_field][i])
    r2 = float(state[R2_field][i])
    if abs(A_hat) < cfg.min_peak_abs_a:
        return PEAK_STATUS_LOW_A
    if r2 < cfg.r2_peak_thresh:
        return PEAK_STATUS_LOW_R2
    if np.sign(state["best_A"][i]) != sign:
        return PEAK_STATUS_OPP_SIGN
    if i not in set(state["initial_peaks"]):
        return PEAK_STATUS_LOCAL_NMS
    if i not in set(state["final_peaks"]):
        return PEAK_STATUS_SAME_SIGN_NMS
    t_i = float(t[i])
    for p in predictions:
        if (abs(float(p["lobe1"]["t_c"]) - t_i) < 0.05
                or abs(float(p["lobe2"]["t_c"]) - t_i) < 0.05):
            return PEAK_STATUS_ACCEPTED
    return PEAK_STATUS_UNPAIRED



# --------------------------------------------------------------------------
# Diagnostic — explain why a given time window did (or didn't) get detected
# --------------------------------------------------------------------------

def _find_extrema_in_window(
    state: dict, t_lo: float, t_hi: float,
) -> tuple[tuple[int, float, float] | None, tuple[int, float, float] | None]:
    """``((pos_idx, pos_A, pos_r2), (neg_idx, neg_A, neg_r2))`` — the
    largest-``A`` positive sample and the most-negative-``A`` sample in
    ``[t_lo, t_hi]``. Either side can be ``None`` if the window has no
    same-sign sample with finite R²."""
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
    state: dict, t_lo: float, t_hi: float,
    ride_type: str | None = None,
    config: DetectConfig | None = None,
) -> dict:
    """Explain what happened inside ``[t_lo, t_hi]`` w.r.t. the detector.

    Meant for GT-interval diagnostics in the UI: given a GT ride window
    that did (or didn't) get detected, report the best positive- and
    negative-sign matches found inside it, whether each clears the peak
    thresholds, and — if both sides exist — the shared-shape joint fit.
    Each failure mode is tagged in ``verdict_lines`` so the user sees
    exactly which tunable would have let the ride through.

    Uses ``config`` if passed, otherwise the config that produced
    ``state``, otherwise :data:`DEFAULT_CONFIG`.
    """
    cfg = config if config is not None else state.get("config", DEFAULT_CONFIG)
    t = state["t"]
    a_smooth = state["a_smooth"]

    pos, neg = _find_extrema_in_window(state, t_lo, t_hi)

    if ride_type == "up":
        first, second, s1, s2 = pos, neg, +1.0, -1.0
    elif ride_type == "down":
        first, second, s1, s2 = neg, pos, -1.0, +1.0
    else:
        if pos and neg and pos[0] < neg[0]:
            first, second, s1, s2 = pos, neg, +1.0, -1.0
        else:
            first, second, s1, s2 = neg, pos, -1.0, +1.0

    lines: list[str] = []

    def _peak_line(tag: str, peak):
        if peak is None:
            return f"  {tag} lobe: no sample with that sign in the window."
        i, A, r2 = peak
        flags = []
        if not (r2 >= cfg.r2_peak_thresh):
            flags.append(f"R²={r2:.2f} < {cfg.r2_peak_thresh:.2f}")
        if not (abs(A) >= cfg.min_peak_abs_a):
            flags.append(f"|A|={abs(A):.2f} < {cfg.min_peak_abs_a:.2f}")
        ok = "OK" if not flags else "FAIL"
        reasons = " & ".join(flags) if flags else "clears thresholds"
        return (
            f"  {tag} lobe: t={t[i]:.1f}s  A={A:+.2f}  R²={r2:.2f}  "
            f"[{ok}: {reasons}]"
        )

    lines.append(_peak_line("+", pos))
    lines.append(_peak_line("−", neg))

    pair_info: dict | None = None
    if first is not None and second is not None:
        i1, _, _ = first
        i2, _, _ = second
        if i1 >= i2:
            lines.append("  pair: requested lobes are not in chronological order.")
        else:
            gap = float(t[i2] - t[i1])
            res = pair_filter.joint_pair_score(
                a_smooth, t, i1, i2, s1, s2,
                state.get("grid_w_s", cfg.grid_w_s()),
                state.get("grid_f", cfg.grid_f()),
            )
            if res is None:
                lines.append(
                    "  pair: joint fit unavailable (window too short or no sign match)."
                )
            else:
                score, W, f, A_abs, r2_1, r2_2, heatmap_energy = res
                flags = []
                if not (cfg.min_ride_s <= gap <= cfg.max_ride_s):
                    flags.append(
                        f"gap={gap:.1f}s outside "
                        f"[{cfg.min_ride_s}, {cfg.max_ride_s}]"
                    )
                if not (score >= cfg.joint_r2_thresh):
                    flags.append(
                        f"joint R²={score:.2f} < {cfg.joint_r2_thresh:.2f}"
                    )
                if not (A_abs >= cfg.min_pair_abs_a):
                    flags.append(
                        f"pair |A|={A_abs:.2f} < {cfg.min_pair_abs_a:.2f}"
                    )
                if not (heatmap_energy >= cfg.heatmap_energy_thresh):
                    flags.append(
                        f"heatmap energy={heatmap_energy:.3f} < "
                        f"{cfg.heatmap_energy_thresh:.3f}"
                    )
                ok = "accepted" if not flags else "rejected"
                reasons = "; ".join(flags) if flags else "all thresholds pass"
                lines.append(
                    f"  pair: gap={gap:.1f}s  W={W:.2f}  f={f:.2f}  "
                    f"|A|={A_abs:.2f}  R²={score:.3f}  "
                    f"heatmap_energy={heatmap_energy:.3f}  "
                    f"[{ok}: {reasons}]"
                )
                pair_info = {
                    "i1": int(i1), "i2": int(i2),
                    "t_c1": float(t[i1]), "t_c2": float(t[i2]),
                    "W": float(W), "frac_flat": float(f),
                    "A_abs": float(A_abs),
                    "r2_1": float(r2_1), "r2_2": float(r2_2),
                    "joint_r2_mean": float(score),
                    "heatmap_energy": float(heatmap_energy),
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


# --------------------------------------------------------------------------
# CLI — sanity check; prints gt/pred counts per experiment, writes nothing
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="process a single experiment by name")
    args = parser.parse_args()

    names = [args.only] if args.only else list_experiments(kind="train")
    print(f"running detector on {len(names)} experiments")
    total_gt = 0
    total_pred = 0
    for n in names:
        try:
            sensors, gt, _meta = getExperimentData(n)
        except Exception as exc:
            print(f"[error] {n}: {type(exc).__name__}: {exc}")
            continue
        preds, _state = predict_intervals(sensors.get("ACC"))
        n_gt = int(gt["type"].isin(("up", "down")).sum()) if gt is not None else 0
        total_gt += n_gt
        total_pred += len(preds)
        print(f"[ok]    {n}: gt={n_gt}  pred={len(preds)}")

    if names:
        print(
            f"\n{len(names)} experiments — "
            f"GT total {total_gt}, predicted total {total_pred} "
            f"(pred/gt = {total_pred / max(total_gt, 1):.2f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
