"""ACC-only elevator segmentation with calibrated probabilities.

Pipeline:
    ACC (time, x, y, z)  ->  gravity projection -> a_vert
                          ->  sliding windows  -> features
                          ->  L2 logistic regression  -> window score s_w
                          ->  hysteresis segmentation  -> (start, end)
                          ->  segment score aggregation
                          ->  IVAP calibration  -> (p, p_lo, p_hi)
                          ->  conformal edge CIs  -> start_ci_sec, end_ci_sec

The classifier weights and calibrators are fit offline by
``train_acc_calibrator.py`` and loaded from JSON artifacts.
"""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .._calibration import IVAP, load_edge_conformal

# predection_algorithms package name is spelled that way on disk
_quality = importlib.import_module("src.algorithms.predection_algorithms.quality_filter")
estimate_gravity_vector = _quality.estimate_gravity_vector

_config_mod = importlib.import_module("src.algorithms.segmentation_algorithms.class")
AccOnlyConfig = _config_mod.AccOnlyConfig


OUTPUT_COLUMNS = ["start_ci", "end_ci", "duration", "type", "probability_ci"]


def _compute_a_vert(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float) -> np.ndarray:
    gvec, g_mag, _ = estimate_gravity_vector(ax, ay, az, fs=fs, window_sec=0.5)
    g_hat = gvec / (np.linalg.norm(gvec) + 1e-12)
    a_vert = ax * g_hat[0] + ay * g_hat[1] + az * g_hat[2] - g_mag
    return a_vert


def _band_energy(sos, signal: np.ndarray) -> float:
    filt = sosfiltfilt(sos, signal)
    return float(np.mean(filt * filt))


def _window_features(
    a_vert: np.ndarray,
    fs: float,
    band_elev_hz: tuple[float, float],
    band_walk_hz: tuple[float, float],
) -> np.ndarray:
    """5-dim feature vector for one window."""
    nyq = 0.5 * fs
    sos_elev = butter(4, [band_elev_hz[0] / nyq, band_elev_hz[1] / nyq], btype="band", output="sos")
    sos_walk = butter(4, [band_walk_hz[0] / nyq, band_walk_hz[1] / nyq], btype="band", output="sos")
    e_elev = _band_energy(sos_elev, a_vert)
    e_walk = _band_energy(sos_walk, a_vert)
    energy_ratio = e_elev / (e_elev + e_walk + 1e-12)

    var = float(np.var(a_vert))
    rms = float(np.sqrt(np.mean(a_vert * a_vert)))
    jerk = np.diff(a_vert) * fs
    mean_abs_jerk = float(np.mean(np.abs(jerk))) if len(jerk) else 0.0

    sign = np.sign(a_vert)
    runs = 0
    run_len_sum = 0
    cur = 0
    cur_sign = 0
    for s in sign:
        if s == cur_sign and s != 0:
            cur += 1
        else:
            if cur_sign != 0:
                run_len_sum += cur
                runs += 1
            cur = 1 if s != 0 else 0
            cur_sign = int(s)
    if cur_sign != 0:
        run_len_sum += cur
        runs += 1
    mean_run = (run_len_sum / runs) if runs else 0.0
    mean_run_sec = mean_run / fs

    out = np.array([var, rms, energy_ratio, mean_abs_jerk, mean_run_sec], dtype=float)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


FEATURE_NAMES = ["var", "rms", "energy_ratio", "mean_abs_jerk", "mean_run_sec"]


def build_windows(
    t: np.ndarray, a_vert: np.ndarray, fs: float,
    window_sec: float, overlap: float,
    band_elev_hz: tuple[float, float], band_walk_hz: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (centers_sec, feature_matrix[N, D])."""
    win = int(round(window_sec * fs))
    step = max(1, int(round(win * (1.0 - overlap))))
    n = len(a_vert)
    if n < win:
        return np.empty(0), np.empty((0, len(FEATURE_NAMES)))
    starts = np.arange(0, n - win + 1, step)
    feats = np.empty((len(starts), len(FEATURE_NAMES)))
    centers = np.empty(len(starts))
    for i, s in enumerate(starts):
        seg = a_vert[s:s + win]
        feats[i] = _window_features(seg, fs, band_elev_hz, band_walk_hz)
        centers[i] = t[s + win // 2]
    return centers, feats


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def score_windows(features: np.ndarray, lr_payload: dict) -> np.ndarray:
    mean = np.asarray(lr_payload["mean"])
    std = np.asarray(lr_payload["std"])
    coef = np.asarray(lr_payload["coef"])
    intercept = float(lr_payload["intercept"])
    std_safe = np.where(std > 0, std, 1.0)
    feats = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    x = (feats - mean) / std_safe
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return _sigmoid(x @ coef + intercept)


def hysteresis_segments(
    centers: np.ndarray, scores: np.ndarray,
    enter: float, exit_: float,
    min_duration_sec: float, merge_gap_sec: float, pad_sec: float,
    t_min: float, t_max: float,
) -> list[tuple[float, float, float]]:
    """Return list of (start_sec, end_sec, mean_score)."""
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
        if merged and centers[s] - centers[merged[-1][1]] < merge_gap_sec:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    out: list[tuple[float, float, float]] = []
    for s, e in merged:
        t_start = float(centers[s])
        t_end = float(centers[e])
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
    """Segment elevator rides from ACC only and emit calibrated probabilities."""
    t = data[config.time_col].to_numpy(dtype=float)
    ax = data[config.x_col].to_numpy(dtype=float)
    ay = data[config.y_col].to_numpy(dtype=float)
    az = data[config.z_col].to_numpy(dtype=float)
    if len(t) < int(config.window_sec * config.fs_hz):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    a_vert = _compute_a_vert(ax, ay, az, config.fs_hz)
    centers, feats = build_windows(
        t, a_vert, config.fs_hz, config.window_sec, config.overlap,
        tuple(config.band_elev_hz), tuple(config.band_walk_hz),
    )
    if len(centers) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    with open(config.lr_path, "r") as f:
        lr_payload = json.load(f)
    scores = score_windows(feats, lr_payload)

    raw_segments = hysteresis_segments(
        centers, scores,
        enter=config.enter_threshold, exit_=config.exit_threshold,
        min_duration_sec=config.min_duration_sec,
        merge_gap_sec=config.merge_gap_sec,
        pad_sec=config.pad_sec,
        t_min=float(t[0]), t_max=float(t[-1]),
    )
    if not raw_segments:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    ivap = IVAP.load(config.ivap_path)
    _alpha, start_q, end_q = load_edge_conformal(config.edge_conformal_path)

    rows = []
    for t_start, t_end, s_seg in raw_segments:
        _p, p_lo, p_hi = ivap.predict(s_seg)
        rows.append({
            "start_ci": (float(t_start - start_q), float(t_start + start_q)),
            "end_ci": (float(t_end - end_q), float(t_end + end_q)),
            "duration": float(t_end - t_start),
            "type": "unknown",
            "probability_ci": (float(p_lo), float(p_hi)),
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
