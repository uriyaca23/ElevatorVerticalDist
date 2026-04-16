"""Sliding-window NCC of pulse templates against the session signal.

For each candidate window (length = template length), compute normalized
cross-correlation (Pearson r) between the z-scored window and z-scored
template. Fuse velocity-based and acceleration-based scores, peak-pick
with NMS, then greedily pair start peaks (pulse-up) with later end peaks
(pulse-down) under plausible-duration bounds.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from .._calibration import IVAP, load_edge_conformal
from ..accelerometer_only.acc_segmentation import (
    _compute_a_vert, compute_velocity, lowpass,
)
from .templates import Templates, load_templates

_config_mod = importlib.import_module("src.segmentation.algorithms.class")
TemplateMatchConfig = _config_mod.TemplateMatchConfig

OUTPUT_COLUMNS = ["start_ci", "end_ci", "duration", "type", "probability_ci"]


def _zscore(x: np.ndarray) -> np.ndarray:
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - m) / s


def ncc_slide(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Pearson NCC of `template` against every length-len(template) window
    of `signal`. Output length = len(signal) (centered, edges zero-padded).
    """
    n = len(template)
    L = len(signal)
    if L < n:
        return np.zeros(L)
    tz = _zscore(template)
    # rolling mean/std
    s = pd.Series(signal)
    win_mean = s.rolling(window=n, center=False, min_periods=n).mean().to_numpy()
    win_std = s.rolling(window=n, center=False, min_periods=n).std(ddof=0).to_numpy()
    # raw correlation = sum(window * tz) / n  (after centering window)
    # = (conv(signal, tz_rev) - n*win_mean*tz_mean) / n; tz mean is 0 ⇒ second term 0
    conv = np.convolve(signal, tz[::-1], mode="valid") / n
    # full-length output indexed by window-end; we want centered
    score_end = conv / np.maximum(win_std[n - 1:], 1e-12)
    out = np.zeros(L)
    half = n // 2
    out[half:half + len(score_end)] = score_end
    return out


def _peak_pick(score: np.ndarray, t: np.ndarray, threshold: float, nms_radius: int) -> list[int]:
    above = np.where(score >= threshold)[0]
    if len(above) == 0:
        return []
    order = np.argsort(score[above])[::-1]
    chosen: list[int] = []
    taken = np.zeros(len(score), dtype=bool)
    for idx in above[order]:
        if taken[idx]:
            continue
        chosen.append(int(idx))
        lo = max(0, idx - nms_radius)
        hi = min(len(score), idx + nms_radius + 1)
        taken[lo:hi] = True
    chosen.sort()
    return chosen


def _pair_starts_ends(
    starts: list[int], ends: list[int],
    start_score: np.ndarray, end_score: np.ndarray,
    t: np.ndarray,
    min_gap_sec: float, max_gap_sec: float,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used_end = set()
    for si in starts:
        ts = t[si]
        best = None
        best_score = -np.inf
        for ei in ends:
            if ei in used_end or ei <= si:
                continue
            dt = t[ei] - ts
            if dt < min_gap_sec or dt > max_gap_sec:
                continue
            combined = start_score[si] + end_score[ei]
            if combined > best_score:
                best_score = combined
                best = ei
        if best is not None:
            pairs.append((si, best))
            used_end.add(best)
    return pairs


def compute_match_scores(
    data: pd.DataFrame,
    templates: Templates,
    config: TemplateMatchConfig,
) -> dict:
    """Compute fused start/end NCC scores for the whole session.

    Returns a dict with keys: t, v_lpf, a_vert, start_score, end_score,
    ncc_vel_up, ncc_vel_down, ncc_acc_up, ncc_acc_down.
    """
    fs = float(config.fs_hz)
    t = data[config.time_col].to_numpy(dtype=float)
    ax = data[config.x_col].to_numpy(dtype=float)
    ay = data[config.y_col].to_numpy(dtype=float)
    az = data[config.z_col].to_numpy(dtype=float)
    a_vert = _compute_a_vert(ax, ay, az, fs)
    v_lpf = lowpass(compute_velocity(a_vert, fs), fs, cutoff_hz=float(config.lpf_hz))

    nv_up = ncc_slide(v_lpf, templates.pulse_up_v)
    nv_dn = ncc_slide(v_lpf, templates.pulse_down_v)
    na_up = ncc_slide(a_vert, templates.pulse_up_a)
    na_dn = ncc_slide(a_vert, templates.pulse_down_a)
    w = float(config.vel_weight)
    start_score = w * nv_up + (1.0 - w) * na_up
    end_score = w * nv_dn + (1.0 - w) * na_dn
    return {
        "t": t, "v_lpf": v_lpf, "a_vert": a_vert,
        "ncc_vel_up": nv_up, "ncc_vel_down": nv_dn,
        "ncc_acc_up": na_up, "ncc_acc_down": na_dn,
        "start_score": start_score, "end_score": end_score,
    }


def detect_elevator_segments_from_template_match(
    data: pd.DataFrame,
    config: TemplateMatchConfig,
) -> pd.DataFrame:
    try:
        templates = load_templates(config.templates_path)
    except FileNotFoundError:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if templates.meta.get("n_rides", 0) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    fs = float(config.fs_hz)
    scores = compute_match_scores(data, templates, config)
    t = scores["t"]
    nms = max(1, int(round(config.nms_radius_sec * fs)))
    starts = _peak_pick(scores["start_score"], t, config.enter_threshold, nms)
    ends = _peak_pick(scores["end_score"], t, config.enter_threshold, nms)
    pairs = _pair_starts_ends(
        starts, ends, scores["start_score"], scores["end_score"], t,
        float(config.min_pair_gap_sec), float(config.max_pair_gap_sec),
    )
    if not pairs:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        ivap = IVAP.load(config.ivap_path)
    except Exception:
        ivap = None
    try:
        _alpha, start_q, end_q = load_edge_conformal(config.edge_conformal_path)
    except Exception:
        start_q = end_q = float(config.pad_sec)

    rows = []
    for si, ei in pairs:
        t_s = float(t[si])
        t_e = float(t[ei])
        s_seg = 0.5 * (float(scores["start_score"][si]) + float(scores["end_score"][ei]))
        if ivap is not None:
            try:
                _p, p_lo, p_hi = ivap.predict(s_seg)
            except Exception:
                p_lo, p_hi = 0.0, 1.0
        else:
            p_lo, p_hi = 0.0, 1.0
        rows.append({
            "start_ci": (t_s - start_q, t_s + start_q),
            "end_ci": (t_e - end_q, t_e + end_q),
            "duration": t_e - t_s,
            "type": "ride",
            "probability_ci": (float(p_lo), float(p_hi)),
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
