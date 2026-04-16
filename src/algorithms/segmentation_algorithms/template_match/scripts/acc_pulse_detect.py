"""Acceleration-domain matched-filter pulse detection.

Differences from pulse_detect.py:
  1. Templates are the *derivative* of velocity templates — bipolar acc signatures
     (positive spike at ride start, negative spike at end, near-zero during cruise).
  2. The filter runs on raw `|a| − mean|a|` (+ mild bandpass), NOT on integrated
     velocity. This avoids integration drift.
  3. Score = correlation / ||template||₂ (unnormalized by local signal RMS).
     Amplitude is preserved, so tiny wiggles get tiny scores and real rides
     get big scores.

Outputs:
    results/05_acc_matched_filter/
        acc_pulse_templates.png
        acc_pulse_confidence_{name}.png
        acc_pulse_detections_{name}.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, butter, sosfiltfilt, find_peaks, peak_widths

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter,
)
from src.tests.segmentations.main import build_acc_frame, build_height_frame
from src.algorithms.segmentation_algorithms.template_match.scripts.velocity_templates import (
    trapezoid_velocity, parabola_velocity,
)
from src.algorithms.segmentation_algorithms.template_match.scripts.strategy_search import load_rides
from src.algorithms.segmentation_algorithms.template_match.scripts.pulse_detect import (
    cluster_templates, FS_TEMPLATE,
)


OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "05_acc_matched_filter"

N_TEMPLATES_PER_SHAPE = 5

# Score threshold. Score units are m/s (peak of correlation ~= integrated velocity
# amplitude of the matched ride). Walking bumps ~0.1-0.3 m/s, real rides 0.5-3 m/s.
SCORE_THRESH = 0.45
MIN_DURATION = 0.5      # lowered — was rejecting short peaks that were above threshold
# GT gap stats (both experimenters combined, 81 gaps):
#   min=6.32s  p10=7.04s  p25=7.68s  median=9.71s
# So two real rides are never closer than ~6.3s. Any two detections within
# 5s of each other must belong to the same ride — merge them.
MERGE_GAP_SEC = 5.0
SMOOTH_SEC = 4.0   # heavier smoothing so each hill collapses to a single peak


# --------------------- Acceleration templates ---------------------
def velocity_template(pick: dict, shape: str) -> np.ndarray:
    if shape == "trapezoid":
        t = pick["trap"]; W = t["t_end"] - t["t_start"]
        ts = np.arange(0.0, W, 1.0 / FS_TEMPLATE)
        return trapezoid_velocity(ts, 0.0, W, t["a_max"], t["v_max"])
    p = pick["par"]; W = p["W"]
    ts = np.arange(0.0, W, 1.0 / FS_TEMPLATE)
    return parabola_velocity(ts, W / 2, W / 2, p["v_peak"], p["p"])


def acc_template(v_tpl: np.ndarray, fs: float = FS_TEMPLATE) -> np.ndarray:
    """Derivative: gives bipolar +spike / flat / -spike signature."""
    a = np.gradient(v_tpl, 1.0 / fs)
    return a - a.mean()  # zero-mean for matched filter


# --------------------- Session velocity (same as main_acc.py/pulse_detect.py) ---------------------
def session_signal(acc_frame) -> tuple[np.ndarray, np.ndarray, float]:
    """cumsum(|a| - mean|a|) -> Butterworth 4th-order LPF @ 0.3 Hz.
    Matches the velocity panel from main_acc.py (run_results/.../velocity.png).
    """
    ts = acc_frame["time"].to_numpy()
    mag = acc_frame["mag"].to_numpy()
    dt = np.diff(ts, prepend=ts[0])
    fs = 1.0 / np.median(dt[dt > 0])
    a_lin = mag - mag.mean()
    vel = np.cumsum(a_lin * dt)
    sos = butter(4, 0.3 / (0.5 * fs), btype="low", output="sos")
    vel_lpf = sosfiltfilt(sos, vel)
    return ts, vel_lpf, fs


# --------------------- Matched filter (template-energy normalization only) ---------------------
def matched_score(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Template-energy-normalized matched filter.

    score[i] = Σ_j signal[i+j] · template[j] / ||template||²

    At exact match (signal_window == template), score = 1.
    At amplitude-scaled match (signal_window == k · template), score = k.
    So score units are the *effective template amplitude* at position i.
    Scale is comparable across templates of different length/size.
    """
    tpl = template - template.mean()
    tpl_energy = np.dot(tpl, tpl) + 1e-9
    c = fftconvolve(signal, tpl[::-1] / tpl_energy, mode="same")
    L = len(tpl); half = L // 2
    if half > 0 and len(c) > 2 * half:
        c[:half] = 0.0
        c[-half:] = 0.0
    return c


def best_score(signal: np.ndarray, templates: list[np.ndarray]) -> np.ndarray:
    best = np.zeros_like(signal)
    for tpl in templates:
        s = np.abs(matched_score(signal, tpl))
        best = np.maximum(best, s)
    return best


def smooth(x: np.ndarray, fs: float, sec: float) -> np.ndarray:
    w = max(3, int(sec * fs))
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def resample_to(fs_src: float, fs_dst: float, tpl: np.ndarray) -> np.ndarray:
    if abs(fs_src - fs_dst) < 0.5:
        return tpl
    n_new = max(5, int(len(tpl) * fs_dst / fs_src))
    return np.interp(np.linspace(0, 1, n_new), np.linspace(0, 1, len(tpl)), tpl)


# --------------------- Detection & plotting ---------------------
# GT min gap is 6.3s; set peak-distance just below that so we don't merge
# genuinely separate rides, but still reject closer duplicate peaks.
# scipy.find_peaks with `distance=` uses a greedy "keep the tallest in each
# window" algorithm — exactly "if peaks are too close, take the higher one".
PEAK_MIN_DISTANCE_SEC = 5.0
PEAK_PROMINENCE_FRAC = 0.02   # tiny — so we don't miss real hills
PEAK_WIDTH_REL_HEIGHT = 0.6   # where to measure peak width (0 = at top, 1 = at base)


def detections_from_peaks(score: np.ndarray, ts: np.ndarray, fs: float
                          ) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Find local maxima (hills) in the score. Each hill → one detection.

    Width is taken at 60% down from the peak height (i.e. where the hill
    drops to 40% of peak-prominence). Minimum separation between peaks is
    5 s (just below the smallest GT inter-ride gap of 6.3 s).

    Returns (detections, peak_indices).
    """
    distance = max(1, int(PEAK_MIN_DISTANCE_SEC * fs))
    span = float(score.max() - np.median(score))
    prominence = max(1e-4, PEAK_PROMINENCE_FRAC * span)
    peak_idx, props = find_peaks(score, distance=distance, prominence=prominence)
    if len(peak_idx) == 0:
        return [], peak_idx
    widths, _, left_ips, right_ips = peak_widths(
        score, peak_idx, rel_height=PEAK_WIDTH_REL_HEIGHT)
    dets = []
    for L, R in zip(left_ips, right_ips):
        L_i = max(0, int(L)); R_i = min(len(ts) - 1, int(R))
        dets.append((ts[L_i], ts[R_i]))
    # merge any remaining overlaps
    dets.sort()
    merged = []
    for s, e in dets:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged, peak_idx


def iou_hit(det, gts):
    for g_s, g_e in gts:
        inter = max(0, min(det[1], g_e) - max(det[0], g_s))
        union = max(det[1], g_e) - min(det[0], g_s)
        if union > 0 and inter / union > 0.3:
            return True
    return False


def render(name: str, ts, a_bp, score, detections, segments, fs, peak_idx):
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

    ax = axes[0]
    ax.plot(ts, a_bp, color="0.5", lw=0.5)
    for _, row in segments.iterrows():
        ax.axvspan(row["start_ci"][0], row["end_ci"][1], color="yellow", alpha=0.3)
    ax.set_ylabel("|a| − mean (BP)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="yellow", alpha=0.4, label="GT")], loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ts, score, color="black", lw=1.0, label="matched-filter score")
    if len(peak_idx):
        ax.plot(ts[peak_idx], score[peak_idx], "rx", ms=9, mew=1.6,
                label=f"{len(peak_idx)} peaks (local maxima)")
    for _, row in segments.iterrows():
        ax.axvspan(row["start_ci"][0], row["end_ci"][1], color="yellow", alpha=0.3)
    ax.set_ylabel("score")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for _, row in segments.iterrows():
        up = row["type"] == "up"
        ax.axvspan(row["start_ci"][0], row["end_ci"][1],
                   color="tab:blue" if up else "tab:red", alpha=0.2)
    for s, e in detections:
        ax.axvspan(s, e, color="tab:green", alpha=0.45)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title("Detections (green) vs GT (blue=up, red=down)")
    ax.grid(True, alpha=0.3)

    gt_windows = [(r["start_ci"][0], r["end_ci"][1]) for _, r in segments.iterrows()]
    tp = sum(1 for d in detections if iou_hit(d, gt_windows))
    fp = len(detections) - tp
    hit = sum(1 for g in gt_windows if iou_hit(g, detections))
    miss = len(gt_windows) - hit
    fig.suptitle(
        f"Acc-domain matched filter — {name}   "
        f"det={len(detections)}  TP={tp}  FP={fp}  GT hit={hit}/{len(gt_windows)}  miss={miss}",
        fontsize=12,
    )
    print(f"  {name}: det={len(detections)} TP={tp} FP={fp} hit={hit}/{len(gt_windows)} miss={miss}")

    fig.tight_layout()
    out = OUT_DIR / f"acc_pulse_detections_{name}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  -> {out}")


def render_confidence_only(name: str, ts, score_trap, score_par, segments, fs,
                           peaks_trap, peaks_par):
    """Split plot — trap score top, parabola score bottom; each panel shows
    its OWN local maxima (peaks found on that score trace)."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    ax = axes[0]
    ax.plot(ts, score_trap, color="tab:orange", lw=1.0, label="trapezoid score")
    for _, row in segments.iterrows():
        ax.axvspan(row["start_ci"][0], row["end_ci"][1], color="yellow", alpha=0.3)
    if len(peaks_trap):
        ax.plot(ts[peaks_trap], score_trap[peaks_trap], "rx", ms=8, mew=1.5,
                label=f"{len(peaks_trap)} trap peaks")
    ax.set_ylabel("score"); ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3); ax.set_title(f"Acc matched-filter score — {name}")

    ax = axes[1]
    ax.plot(ts, score_par, color="tab:blue", lw=1.0, label="parabola score")
    for _, row in segments.iterrows():
        ax.axvspan(row["start_ci"][0], row["end_ci"][1], color="yellow", alpha=0.3)
    if len(peaks_par):
        ax.plot(ts[peaks_par], score_par[peaks_par], "rx", ms=8, mew=1.5,
                label=f"{len(peaks_par)} par peaks")
    ax.set_ylabel("score"); ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / f"acc_pulse_confidence_{name}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  -> {out}")


def run(name: str, trap_v: list[np.ndarray], par_v: list[np.ndarray]) -> None:
    data = load_experimenter(name)
    t0 = float(data["ACC"]["timestamp_ms"].iloc[0])
    acc_frame = build_acc_frame(data["ACC"], t0)
    height_frame = build_height_frame(data["PRS"], t0)

    ts, sig, fs = session_signal(acc_frame)
    print(f"{name}: fs={fs:.1f} Hz, {len(ts)/fs:.0f} s")

    trap_r = [resample_to(FS_TEMPLATE, fs, t) for t in trap_v]
    par_r = [resample_to(FS_TEMPLATE, fs, t) for t in par_v]

    score_trap = best_score(sig, trap_r)
    score_par = best_score(sig, par_r)
    score_all = np.maximum(score_trap, score_par)
    print(f"    score_all: max={score_all.max():.3f}  p95={np.percentile(score_all, 95):.3f}  "
          f"p50={np.percentile(score_all, 50):.3f}")
    score_trap_s = smooth(score_trap, fs, SMOOTH_SEC)
    score_par_s = smooth(score_par, fs, SMOOTH_SEC)
    score_all_s = smooth(score_all, fs, SMOOTH_SEC)

    cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)
    segments = Segmenter(cfg).detect(height_frame)

    # Peak-based detection: each local maximum ("hill") becomes one detection.
    detections, peak_idx = detections_from_peaks(score_all_s, ts, fs)
    _, peaks_trap = detections_from_peaks(score_trap_s, ts, fs)
    _, peaks_par = detections_from_peaks(score_par_s, ts, fs)
    print(f"    peaks: combined={len(peak_idx)}  trap={len(peaks_trap)}  par={len(peaks_par)}  "
          f"-> {len(detections)} detections")
    render(name, ts, sig, score_all_s, detections, segments, fs, peak_idx)
    render_confidence_only(name, ts, score_trap_s, score_par_s, segments, fs,
                            peaks_trap, peaks_par)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rides, labels = load_rides()
    trap_picks = cluster_templates(rides, labels, "trapezoid", N_TEMPLATES_PER_SHAPE)
    par_picks = cluster_templates(rides, labels, "parabola", N_TEMPLATES_PER_SHAPE)
    print(f"Templates: {len(trap_picks)} trap, {len(par_picks)} par")

    trap_v = [velocity_template(p, "trapezoid") for p in trap_picks]
    par_v = [velocity_template(p, "parabola") for p in par_picks]
    trap_a = [acc_template(v) for v in trap_v]
    par_a = [acc_template(v) for v in par_v]

    # Template plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    for i, v in enumerate(trap_v):
        x = np.arange(len(v)) / FS_TEMPLATE
        axes[0, 0].plot(x, v, lw=1.2, alpha=0.85, label=trap_picks[i]["key"])
    axes[0, 0].set_title("Trapezoid velocity templates"); axes[0, 0].legend(fontsize=6, ncol=2)
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].set_ylabel("v (m/s)")

    for i, a in enumerate(trap_a):
        x = np.arange(len(a)) / FS_TEMPLATE
        axes[1, 0].plot(x, a, lw=1.2, alpha=0.85)
    axes[1, 0].set_title("Trapezoid acceleration templates (derivative)")
    axes[1, 0].axhline(0, color="k", lw=0.4); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylabel("a (m/s²)"); axes[1, 0].set_xlabel("time (s)")

    for i, v in enumerate(par_v):
        x = np.arange(len(v)) / FS_TEMPLATE
        axes[0, 1].plot(x, v, lw=1.2, alpha=0.85, label=par_picks[i]["key"])
    axes[0, 1].set_title("Parabola velocity templates"); axes[0, 1].legend(fontsize=6, ncol=2)
    axes[0, 1].grid(True, alpha=0.3)

    for i, a in enumerate(par_a):
        x = np.arange(len(a)) / FS_TEMPLATE
        axes[1, 1].plot(x, a, lw=1.2, alpha=0.85)
    axes[1, 1].set_title("Parabola acceleration templates (derivative)")
    axes[1, 1].axhline(0, color="k", lw=0.4); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "acc_pulse_templates.png", dpi=120); plt.close(fig)

    for name in ("oria", "roy_turgman"):
        run(name, trap_v, par_v)


if __name__ == "__main__":
    main()
