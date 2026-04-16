"""Diagnostic plots for the template-matching segmenter:

- ``plot_templates_overlay`` — per-GT-segment LPF velocity slice with
  the entry-half / exit-half templates rescaled back onto each ride to
  visually check extraction quality.
- ``plot_match_scores`` — session-wide LPF velocity with detected and GT
  segments shaded, plus the start/end NCC score traces with peaks.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.algorithms.segmentation_algorithms import ci_center
from src.algorithms.segmentation_algorithms.accelerometer_only.acc_segmentation import (
    _compute_a_vert, compute_velocity, lowpass,
)


def _resample(x: np.ndarray, n_out: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(n_out)
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, n_out)
    return np.interp(dst, src, x)


def plot_templates_overlay(
    acc_frame: pd.DataFrame,
    gt_segments: pd.DataFrame,
    templates,
    config,
) -> plt.Figure:
    fs = float(config.fs_hz)
    t = acc_frame[config.time_col].to_numpy(dtype=float)
    ax = acc_frame[config.x_col].to_numpy(dtype=float)
    ay = acc_frame[config.y_col].to_numpy(dtype=float)
    az = acc_frame[config.z_col].to_numpy(dtype=float)
    a_vert = _compute_a_vert(ax, ay, az, fs)
    v_lpf = lowpass(compute_velocity(a_vert, fs), fs, cutoff_hz=float(config.lpf_hz))

    n_rides = len(gt_segments)
    if n_rides == 0:
        fig, ax_ = plt.subplots(figsize=(6, 3))
        ax_.text(0.5, 0.5, "no GT segments", ha="center", va="center")
        ax_.set_axis_off()
        return fig

    cols = min(4, n_rides)
    rows = math.ceil(n_rides / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.6 * rows), squeeze=False)
    for i, (_, row) in enumerate(gt_segments.iterrows()):
        r, c = divmod(i, cols)
        ax_ = axes[r][c]
        s = ci_center(row["start_ci"])
        e = ci_center(row["end_ci"])
        i0 = int(np.searchsorted(t, s))
        i1 = int(np.searchsorted(t, e))
        if i1 - i0 < 8:
            ax_.set_axis_off()
            continue
        seg_t = t[i0:i1] - t[i0]
        seg_v = v_lpf[i0:i1]
        L = len(seg_v)
        n_entry = max(2, int(round(L * float(config.entry_frac))))
        n_exit = max(2, int(round(L * float(config.exit_frac))))

        up_t = seg_t[:n_entry]
        dn_t = seg_t[-n_exit:]
        # templates are in raw velocity units; resample to this ride's half-length
        up_tmpl = _resample(templates.pulse_up_v, n_entry)
        dn_tmpl = _resample(templates.pulse_down_v, n_exit)

        ax_.plot(seg_t, seg_v, color="tab:blue", linewidth=1.2, label="LPF v")
        ax_.plot(up_t, up_tmpl, color="tab:green", linewidth=1.6, alpha=0.85, label="pulse-up tmpl")
        ax_.plot(dn_t, dn_tmpl, color="tab:red", linewidth=1.6, alpha=0.85, label="pulse-down tmpl")
        ax_.set_title(f"ride {i}  ({e - s:.1f}s)", fontsize=9)
        ax_.grid(True, alpha=0.3)
        if i == 0:
            ax_.legend(fontsize=7, loc="best")
    for j in range(n_rides, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_axis_off()
    fig.suptitle(f"Per-ride LPF velocity with extracted templates  "
                 f"(name={templates.meta.get('name', '')}, n={templates.meta.get('n_rides', 0)})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_match_scores(
    t: np.ndarray, v_lpf: np.ndarray,
    start_score: np.ndarray, end_score: np.ndarray,
    detected_segments: pd.DataFrame, gt_segments: pd.DataFrame,
    threshold: float,
) -> plt.Figure:
    fig, (ax_v, ax_s) = plt.subplots(2, 1, figsize=(15, 6.0), sharex=True)

    ax_v.plot(t, v_lpf, linewidth=0.9, color="tab:blue", label="LPF v")
    for _, row in gt_segments.iterrows():
        ax_v.axvspan(ci_center(row["start_ci"]), ci_center(row["end_ci"]),
                     color="tab:green", alpha=0.18)
    for _, row in detected_segments.iterrows():
        ax_v.axvspan(ci_center(row["start_ci"]), ci_center(row["end_ci"]),
                     color="tab:red", alpha=0.18)
    ax_v.set_ylabel("LPF v (m/s)")
    ax_v.set_title("Session velocity — green=GT, red=template-match detections")
    ax_v.grid(True, alpha=0.3)

    ax_s.plot(t, start_score, color="tab:green", linewidth=0.9, label="start (pulse-up)")
    ax_s.plot(t, end_score, color="tab:red", linewidth=0.9, label="end (pulse-down)")
    ax_s.axhline(threshold, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    for _, row in detected_segments.iterrows():
        s = ci_center(row["start_ci"]); e = ci_center(row["end_ci"])
        ax_s.axvline(s, color="tab:green", linewidth=0.6, alpha=0.6)
        ax_s.axvline(e, color="tab:red", linewidth=0.6, alpha=0.6)
    ax_s.set_xlabel("time (s)")
    ax_s.set_ylabel("fused NCC")
    ax_s.set_title(f"Match scores (threshold={threshold:.2f})")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="best", fontsize=9)

    fig.tight_layout()
    return fig
