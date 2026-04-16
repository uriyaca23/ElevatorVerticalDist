"""ACC-only elevator segmentation via drift-residual variance.

Statistical model: integrating accelerometer noise gives velocity that
behaves like Brownian motion (white ACC noise → Var[v(t)] = sigma^2 * t),
with extra low-frequency bias drift on top (1/f component). Under that
null, the velocity is smoothly drifting and locally low-variance. An
elevator ride injects a bounded bump in v concentrated in ~0.03-0.3 Hz.

Detector: remove the drift with a long-window rolling median of v, then
score the *short-window variance of the residual* against its own
session-level median. This is scale-free, training-free per-session, and
directly tests the null "drift only" vs "excess variance from a ride".

    ACC (time, x, y, z)  ->  gravity projection -> a_vert
                          ->  cumulative integration -> v(t)
                          ->  v - rolling_median_long(v) -> residual r
                          ->  local variance of r, normalized by median -> score s(t)
                          ->  hysteresis segmentation  -> (start, end)
                          ->  IVAP calibration  -> (p, p_lo, p_hi)
                          ->  conformal edge CIs  -> start/end CI
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .._calibration import IVAP, load_edge_conformal

VELOCITY_LPF_HZ = 0.3  # cut walking band; keep elevator ride dynamics


def lowpass(x: np.ndarray, fs: float, cutoff_hz: float = VELOCITY_LPF_HZ) -> np.ndarray:
    nyq = 0.5 * fs
    sos = butter(4, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, x)

_quality = importlib.import_module("src.predection.algorithms.quality_filter")
estimate_gravity_vector = _quality.estimate_gravity_vector

_config_mod = importlib.import_module("src.segmentation.algorithms.class")
AccOnlyConfig = _config_mod.AccOnlyConfig


OUTPUT_COLUMNS = ["start_ci", "end_ci", "duration", "type", "probability_ci"]


def _compute_a_vert(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float) -> np.ndarray:
    gvec, g_mag, _ = estimate_gravity_vector(ax, ay, az, fs=fs, window_sec=0.5)
    g_hat = gvec / (np.linalg.norm(gvec) + 1e-12)
    return ax * g_hat[0] + ay * g_hat[1] + az * g_hat[2] - g_mag


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def step_rate(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float,
    window_sec: float = 4.0,
) -> np.ndarray:
    """Steps per second in a rolling ``window_sec`` box.

    Peaks are detected on the 0.8-3 Hz bandpass of \|a\| with a refractory
    distance equal to the fastest human cadence (~0.3 s) and prominence
    matched to typical heel-strike amplitude. Texting/phone-handling rarely
    produces sustained peaks at this cadence, so step-rate is a cleaner
    stillness proxy than walkband RMS.
    """
    from scipy.signal import find_peaks
    mag = np.sqrt(ax * ax + ay * ay + az * az)
    a_hp = _bandpass(mag, fs, 0.8, 3.0)
    peaks, _ = find_peaks(a_hp, distance=int(0.3 * fs), prominence=0.3)
    w = max(3, int(round(window_sec * fs)))
    indicator = np.zeros_like(mag)
    indicator[peaks] = 1.0
    count = pd.Series(indicator).rolling(w, center=True, min_periods=1).sum().to_numpy()
    return count / window_sec


def sliding_zupt_disp(a_vert: np.ndarray, fs: float, window_sec: float = 10.0) -> np.ndarray:
    """Sliding-window ZUPT-integrated vertical displacement.

    For every sample t we pretend the ``window_sec`` window centered at t
    begins and ends at zero velocity, remove linear drift, and compute the
    peak-to-peak displacement inside. Elevator rides produce ~floor height;
    walking integrates to ~0 after the drift correction because walking's
    net acceleration over the window is ~0.

    Vectorized using cumulative sums:
        V(t) = ∫ a dτ,  S(t) = ∫ V dτ.
    Inside window [a,b], ZUPT-corrected displacement from a to τ reduces to
        d(τ) = S(τ)-S(a) - V(a)*(τ-a) - slope*(τ-a)**2 / 2,
    where slope = (V(b)-V(a))/(b-a). We score each center by evaluating |d|
    at the midpoint, which is a tight proxy for peak-to-peak excursion.
    """
    n = len(a_vert)
    a = a_vert - float(np.mean(a_vert))
    V = np.cumsum(a) / fs
    S = np.cumsum(V) / fs
    W = max(4, int(round(window_sec * fs)))
    half = W // 2
    out = np.zeros(n, dtype=float)
    a_idx = np.arange(n) - half
    b_idx = np.arange(n) + half
    a_idx = np.clip(a_idx, 0, n - 1)
    b_idx = np.clip(b_idx, 0, n - 1)
    span = (b_idx - a_idx) / fs  # seconds
    span = np.where(span < 1.0 / fs, 1.0 / fs, span)
    Va = V[a_idx]; Vb = V[b_idx]
    Sa = S[a_idx]; Sb = S[b_idx]
    slope = (Vb - Va) / span
    mid_idx = (a_idx + b_idx) // 2
    Sm = S[mid_idx]; Vm = V[mid_idx]
    dt = (mid_idx - a_idx) / fs
    # d(mid) = Sm - Sa - Va*dt - slope*dt^2/2
    d_mid = Sm - Sa - Va * dt - 0.5 * slope * dt * dt
    out = np.abs(d_mid)
    return out


def zupt_integrate(a_vert: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-velocity-update integration over a stillness window.

    Assumes v(0) = v(end) = 0 (the window is bracketed by true stance). We
    remove the window-mean of a_vert, integrate to v, force the velocity to
    the zero-endpoint boundary by subtracting a linear ramp (the classic
    constant-drift ZUPT correction), then integrate v to displacement d.
    Returns (v_corrected, d).
    """
    a = a_vert - float(np.mean(a_vert))
    v = np.cumsum(a) / fs
    n = len(v)
    if n < 2:
        return v, np.zeros_like(v)
    ramp = np.linspace(0.0, v[-1], n)
    v_c = v - ramp
    d = np.cumsum(v_c) / fs
    return v_c, d


def walkband_rms(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float,
    lo: float = 1.2, hi: float = 2.8, window_sec: float = 2.0,
) -> np.ndarray:
    """Rolling RMS of |a| bandpassed in the walking band. HIGH during walking;
    LOW when standing still (including inside an elevator ride)."""
    mag = np.sqrt(ax * ax + ay * ay + az * az)
    ab = _bandpass(mag, fs, lo, hi)
    w = max(3, int(round(window_sec * fs)))
    return np.sqrt(
        pd.Series(ab * ab).rolling(w, center=True, min_periods=1).mean().to_numpy()
    )


def stillness_hysteresis_segments(
    t: np.ndarray, walk: np.ndarray,
    enter: float, exit_: float,
    min_duration_sec: float, merge_gap_sec: float, pad_sec: float,
    t_min: float, t_max: float,
) -> list[tuple[float, float, float]]:
    """Candidate = interval where ``walk <= enter`` (stillness begins) until
    ``walk >= exit_`` (walking resumes). Symmetric to score-based hysteresis
    but thresholds are on a signal that is LOW during the event."""
    in_seg = False
    raw: list[tuple[int, int]] = []
    cur_start = 0
    for i, s in enumerate(walk):
        if not in_seg and s <= enter:
            in_seg = True
            cur_start = i
        elif in_seg and s >= exit_:
            raw.append((cur_start, i))
            in_seg = False
    if in_seg:
        raw.append((cur_start, len(walk) - 1))

    merged: list[tuple[int, int]] = []
    for s, e in raw:
        if merged and t[s] - t[merged[-1][1]] < merge_gap_sec:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    out: list[tuple[float, float, float]] = []
    for s, e in merged:
        t_start = float(t[s]); t_end = float(t[e])
        if t_end - t_start <= min_duration_sec:
            continue
        t_start = max(t_min, t_start - pad_sec)
        t_end = min(t_max, t_end + pad_sec)
        mean_stillness = float(np.mean(walk[s:e + 1]))
        out.append((t_start, t_end, mean_stillness))
    return out


def compute_velocity(a_vert: np.ndarray, fs: float) -> np.ndarray:
    """Global integration of DC-removed a_vert."""
    return np.cumsum(a_vert - float(np.mean(a_vert))) / fs


def drift_residual_score(
    a_vert: np.ndarray, fs: float,
    detrend_sec: float,
    local_var_sec: float,
) -> np.ndarray:
    """Score(t) = log ratio of local variance of drift residual to its
    session-level median. Under the drift-only null this is ~ 0; during a
    ride it spikes by 1-3 orders of magnitude."""
    v = compute_velocity(a_vert, fs)
    v = lowpass(v, fs)
    win_long = max(5, int(round(detrend_sec * fs)))
    trend = pd.Series(v).rolling(window=win_long, center=True, min_periods=1).median().to_numpy()
    r = v - trend
    win_short = max(3, int(round(local_var_sec * fs)))
    # local variance of residual
    r2 = r * r
    local_var = pd.Series(r2).rolling(window=win_short, center=True, min_periods=1).mean().to_numpy()
    med = float(np.median(local_var)) + 1e-12
    return np.log10((local_var + 1e-12) / med)


def hysteresis_segments(
    t: np.ndarray, scores: np.ndarray,
    enter: float, exit_: float,
    min_duration_sec: float, merge_gap_sec: float, pad_sec: float,
    t_min: float, t_max: float,
) -> list[tuple[float, float, float]]:
    in_seg = False
    raw: list[tuple[int, int]] = []
    cur_start = 0
    for i, s in enumerate(scores):
        if not in_seg and s >= enter:
            in_seg = True
            cur_start = i
        elif in_seg and s <= exit_:
            raw.append((cur_start, i))
            in_seg = False
    if in_seg:
        raw.append((cur_start, len(scores) - 1))

    merged: list[tuple[int, int]] = []
    for s, e in raw:
        if merged and t[s] - t[merged[-1][1]] < merge_gap_sec:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    out: list[tuple[float, float, float]] = []
    for s, e in merged:
        t_start = float(t[s])
        t_end = float(t[e])
        if t_end - t_start <= min_duration_sec:
            continue
        t_start = max(t_min, t_start - pad_sec)
        t_end = min(t_max, t_end + pad_sec)
        mean_score = float(np.mean(scores[s:e + 1]))
        out.append((t_start, t_end, mean_score))
    return out


def detect_elevator_segments_from_acc(
    data: pd.DataFrame,
    config: AccOnlyConfig,
) -> pd.DataFrame:
    t = data[config.time_col].to_numpy(dtype=float)
    ax = data[config.x_col].to_numpy(dtype=float)
    ay = data[config.y_col].to_numpy(dtype=float)
    az = data[config.z_col].to_numpy(dtype=float)
    if len(t) < int(config.window_sec * config.fs_hz):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    a_vert = _compute_a_vert(ax, ay, az, config.fs_hz)
    fs = float(config.fs_hz)

    # --- Sliding-ZUPT + vertical-variance detector ------------------------
    # 1) Sliding ZUPT displacement score (drift-free, ride-band amplitude).
    # 2) var(a_vert) over 4 s : LOW inside elevator, HIGH during walking.
    # 3) Combined score = log(disp) - 0.5 * log(var+eps). Hysteresis on
    #    session-normalized combined score, min_duration gate, NMS.
    # 4) Endpoint refinement: tighten segment to the window where the
    #    sliding-ZUPT disp is above 40% of the segment's peak.
    min_duration = float(config.min_duration_sec)
    pad = float(config.pad_sec)

    disp = sliding_zupt_disp(a_vert, fs, window_sec=10.0)
    win4 = max(3, int(round(4.0 * fs)))
    var_av = pd.Series(a_vert * a_vert).rolling(win4, center=True, min_periods=1).mean().to_numpy() - (
        pd.Series(a_vert).rolling(win4, center=True, min_periods=1).mean().to_numpy() ** 2
    )
    log_disp = np.log10(np.maximum(disp, 1e-4))
    log_var = np.log10(np.maximum(var_av, 1e-6))
    score = log_disp - 0.7 * log_var
    score_norm = score - float(np.median(score))
    scores = score_norm

    raw_segments = hysteresis_segments(
        t, score_norm,
        enter=float(config.enter_threshold),
        exit_=float(config.exit_threshold),
        min_duration_sec=min_duration,
        merge_gap_sec=float(config.merge_gap_sec),
        pad_sec=pad,
        t_min=float(t[0]), t_max=float(t[-1]),
    )

    # Tighten endpoints: restrict to where the sliding-ZUPT disp is above
    # 0.4 * peak-inside-segment; walking spillover at edges is excluded.
    kept: list[tuple[float, float, float]] = []
    for t_start, t_end, s_seg in raw_segments:
        i0 = int(np.searchsorted(t, t_start))
        i1 = int(np.searchsorted(t, t_end))
        if i1 <= i0:
            continue
        peak = float(np.max(disp[i0:i1 + 1]))
        if peak <= 0:
            continue
        mask = disp[i0:i1 + 1] > 0.4 * peak
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        j0, j1 = int(idxs[0]), int(idxs[-1])
        new_start = max(float(t[0]), float(t[i0 + j0]) - pad)
        new_end = min(float(t[-1]), float(t[i0 + j1]) + pad)
        if new_end - new_start <= min_duration:
            continue
        kept.append((new_start, new_end, s_seg))
    raw_segments = kept
    if not raw_segments:
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
    for t_start, t_end, s_seg in raw_segments:
        if ivap is not None:
            _p, p_lo, p_hi = ivap.predict(s_seg)
        else:
            p_lo, p_hi = 0.0, 1.0
        rows.append({
            "start_ci": (float(t_start - start_q), float(t_start + start_q)),
            "end_ci": (float(t_end - end_q), float(t_end + end_q)),
            "duration": float(t_end - t_start),
            "type": "unknown",
            "probability_ci": (float(p_lo), float(p_hi)),
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
