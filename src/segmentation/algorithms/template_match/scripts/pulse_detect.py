"""Matched-filter pulse detection using shape templates.

Steps:
  1. Load GT rides + labels (via strategy_search.load_rides).
  2. K-means cluster trapezoid-labeled rides on (a_max, v_max, W) -> pick 10
     representative templates; same for parabola on (v_peak, W, p).
  3. Render each cluster representative as a velocity template at fs=100 Hz.
  4. Compute a session-wide velocity-like signal per experimenter.
  5. Sliding normalized cross-correlation of each template on that signal.
  6. Confidence = max over templates (both polarities) of |NCC|.
  7. Plot confidence + session velocity + GT overlay.

Output:
    src/.../template_match/results/pulse_detect_{experimenter}.png

Run:
    python3 -m src.algorithms.segmentation_algorithms.template_match.scripts.pulse_detect
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, fftconvolve

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter,
)
from src.tests.segmentations.main import build_acc_frame, build_height_frame
from src.algorithms.segmentation_algorithms.template_match.scripts.velocity_templates import (
    trapezoid_velocity, parabola_velocity, lowpass,
)
from src.algorithms.segmentation_algorithms.template_match.scripts.strategy_search import load_rides


OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "04_pulse_detection"
FS_TEMPLATE = 100.0  # Hz, common sampling for templates
N_TEMPLATES_PER_SHAPE = 5
CONFIDENCE_SMOOTH_SEC = 1.5  # rolling-mean window applied to |NCC|

# Detection gates (on top of smoothed NCC confidence)
NCC_THRESH = 0.55       # minimum shape similarity
VEL_THRESH = 0.25       # m/s — local peak |v| must exceed this (amplitude gate)
ACC_VAR_MAX = 1.2       # m/s² — local std of raw |a| inside ride must be low (standing still)
MIN_DURATION = 3.0      # s — merge + discard too-short runs
MERGE_GAP_SEC = 2.0     # s — merge runs closer than this


# ------------------- Cluster-centered template selection -------------------
def cluster_templates(rides, labels, shape: str, k: int = 10) -> list[dict]:
    matched = [r for r in rides if labels[r["key"]] == shape]
    if shape == "trapezoid":
        feats = np.array([[r["trap"]["a_max"], r["trap"]["v_max"],
                           r["trap"]["t_end"] - r["trap"]["t_start"]] for r in matched])
    else:
        feats = np.array([[r["par"]["v_peak"], r["par"]["W"], r["par"]["p"]]
                          for r in matched])
    # normalize
    mu, sd = feats.mean(axis=0), feats.std(axis=0) + 1e-9
    X = (feats - mu) / sd
    # simple k-means
    rng = np.random.default_rng(0)
    k_eff = min(k, len(X))
    idx = rng.choice(len(X), k_eff, replace=False)
    centers = X[idx].copy()
    for _ in range(50):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
        new_centers = np.array([X[assign == c].mean(axis=0) if np.any(assign == c)
                                else centers[c] for c in range(k_eff)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    picks = []
    for c in range(k_eff):
        mask = assign == c
        if not mask.any():
            continue
        members = np.where(mask)[0]
        best = members[np.argmin(np.linalg.norm(X[mask] - centers[c], axis=1))]
        picks.append(matched[best])
    return picks


def render_template(pick: dict, shape: str) -> np.ndarray:
    if shape == "trapezoid":
        t = pick["trap"]
        W = t["t_end"] - t["t_start"]
        ts = np.arange(0.0, W, 1.0 / FS_TEMPLATE)
        v = trapezoid_velocity(ts, 0.0, W, t["a_max"], t["v_max"])
    else:
        p = pick["par"]
        W_full = p["W"]
        ts = np.arange(0.0, W_full, 1.0 / FS_TEMPLATE)
        # parabola_velocity expects (t, t_c, W_half, v_peak, p)
        v = parabola_velocity(ts, W_full / 2.0, W_full / 2.0, p["v_peak"], p["p"])
    # zero-mean
    return v - v.mean()


# ------------------- Session velocity (matches older main_acc.py definition) -------------------
def session_velocity(acc_frame) -> tuple[np.ndarray, np.ndarray, float]:
    """Cumulative integral of DC-removed |acc|, then 4th-order Butterworth
    LPF at 0.3 Hz — reproduces the velocity panel from main_acc.py
    (run_results/.../velocity.png)."""
    from scipy.signal import butter, sosfiltfilt
    ts = acc_frame["time"].to_numpy()
    mag = acc_frame["mag"].to_numpy()
    dt = np.diff(ts, prepend=ts[0])
    fs = 1.0 / np.median(dt[dt > 0])
    a_lin = mag - mag.mean()
    vel = np.cumsum(a_lin * dt)
    sos = butter(4, 0.3 / (0.5 * fs), btype="low", output="sos")
    vel_lpf = sosfiltfilt(sos, vel)
    return ts, vel_lpf, fs


# ------------------- NCC matched filter -------------------
def ncc(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Sliding normalized cross-correlation (zero-mean template & local-window signal)."""
    t = template - template.mean()
    t_norm = np.linalg.norm(t) + 1e-9
    L = len(t)
    # cross-correlate via fftconvolve with reversed template
    num = fftconvolve(signal - signal.mean(), t[::-1] / t_norm, mode="same")
    kernel = np.ones(L) / L
    loc_mean = fftconvolve(signal, kernel, mode="same")
    loc_sq = fftconvolve(signal ** 2, kernel, mode="same")
    loc_var = np.maximum(loc_sq - loc_mean ** 2, 1e-12)
    loc_energy = np.sqrt(loc_var * L)
    out = num / (loc_energy + 1e-9)
    # Zero the edges where the template extends past the signal (zero-padding
    # in fftconvolve biases both numerator and local stats -> spurious spikes).
    half = L // 2
    if half > 0 and len(out) > 2 * half:
        out[:half] = 0.0
        out[-half:] = 0.0
    return out


def confidence_max(signal: np.ndarray, templates: list[np.ndarray]) -> np.ndarray:
    """Max over templates & polarities of |NCC|."""
    best = np.zeros_like(signal)
    for tpl in templates:
        c = ncc(signal, tpl)
        # take absolute value -> covers both up and down rides (template mirror)
        best = np.maximum(best, np.abs(c))
    return best


# ------------------- Gated detection -------------------
def apply_gates(ts, vel_lpf, acc_frame, conf, fs, *,
                ncc_thresh=NCC_THRESH, vel_thresh=VEL_THRESH,
                acc_var_max=ACC_VAR_MAX, min_duration=MIN_DURATION,
                merge_gap=MERGE_GAP_SEC):
    """Return list of (t_start, t_end) detections passing all gates."""
    # Resample raw |a| to the same time base as ts
    acc_t = acc_frame["time"].to_numpy()
    acc_m = acc_frame["mag"].to_numpy()
    a_mag = np.interp(ts, acc_t, acc_m)

    w_short = max(3, int(2.0 * fs))  # 2 s local window
    roll = pd.Series(a_mag).rolling(w_short, center=True, min_periods=1)
    a_std = roll.std().to_numpy()
    v_abs = np.abs(vel_lpf)
    v_local_peak = pd.Series(v_abs).rolling(w_short, center=True, min_periods=1).max().to_numpy()

    passes = (conf > ncc_thresh) & (v_local_peak > vel_thresh) & (a_std < acc_var_max)

    # find contiguous runs, merge gaps, drop too-short
    edges = np.diff(passes.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if passes[0]: starts.insert(0, 0)
    if passes[-1]: ends.append(len(passes) - 1)

    runs = []
    for s, e in zip(starts, ends):
        runs.append((ts[s], ts[min(e, len(ts)-1)]))
    # merge close
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    # drop short
    return [(s, e) for s, e in merged if e - s >= min_duration]


def render_detection_plot(name, ts, vel_lpf, acc_frame, conf, detections, segments, fs):
    acc_t = acc_frame["time"].to_numpy()
    acc_m = acc_frame["mag"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

    # Top: |a| with GT (yellow) and detections (green)
    ax = axes[0]
    ax.plot(acc_t, acc_m, color="0.5", lw=0.4)
    for _, row in segments.iterrows():
        ax.axvspan(row["start_ci"][0], row["end_ci"][1], color="yellow", alpha=0.35)
    for s, e in detections:
        ax.axvspan(s, e, color="tab:green", alpha=0.35)
    ax.set_ylabel("|acc| (m/s²)")
    ax.set_title(f"Gated elevator detections — {name}  "
                 f"(NCC>{NCC_THRESH}, |v|>{VEL_THRESH}, σ|a|<{ACC_VAR_MAX}, dur≥{MIN_DURATION}s)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="yellow", alpha=0.4, label="GT"),
                       Patch(color="tab:green", alpha=0.4, label="detection")],
              loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Middle: confidence + threshold line
    ax = axes[1]
    ax.plot(ts, conf, color="black", lw=0.9)
    ax.axhline(NCC_THRESH, color="red", lw=0.6, ls="--", label=f"NCC thresh = {NCC_THRESH}")
    for s, e in detections:
        ax.axvspan(s, e, color="tab:green", alpha=0.15)
    ax.set_ylabel("max |NCC|")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom: velocity
    ax = axes[2]
    ax.plot(ts, vel_lpf, color="tab:gray", lw=0.8)
    for _, row in segments.iterrows():
        up = row["type"] == "up"
        ax.axvspan(row["start_ci"][0], row["end_ci"][1],
                   color="tab:blue" if up else "tab:red", alpha=0.15)
    for s, e in detections:
        ax.axvspan(s, e, color="tab:green", alpha=0.15)
    ax.axhline(VEL_THRESH, color="red", lw=0.4, ls=":", alpha=0.5)
    ax.axhline(-VEL_THRESH, color="red", lw=0.4, ls=":", alpha=0.5)
    ax.axhline(0, color="k", lw=0.3)
    ax.set_xlabel("time (s)"); ax.set_ylabel("vel (m/s)")
    ax.grid(True, alpha=0.3)

    # counts
    def iou_match(det, gts):
        for g_s, g_e in gts:
            inter = max(0, min(det[1], g_e) - max(det[0], g_s))
            union = max(det[1], g_e) - min(det[0], g_s)
            if union > 0 and inter / union > 0.3:
                return True
        return False
    gt_windows = [(r["start_ci"][0], r["end_ci"][1]) for _, r in segments.iterrows()]
    tp = sum(1 for d in detections if iou_match(d, gt_windows))
    fp = len(detections) - tp
    hit_gt = sum(1 for g in gt_windows if iou_match(g, detections))
    miss = len(gt_windows) - hit_gt
    axes[0].text(
        0.01, 0.97,
        f"detections={len(detections)}  TP={tp}  FP={fp}  "
        f"GT_hit={hit_gt}/{len(gt_windows)}  FN={miss}",
        transform=axes[0].transAxes, fontsize=9, va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    print(f"  {name}: det={len(detections)}  TP={tp} FP={fp}  GT hit {hit_gt}/{len(gt_windows)} miss={miss}")

    fig.tight_layout()
    out = OUT_DIR / f"pulse_detections_{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  -> {out}")


# ------------------- Main per-experimenter run -------------------
def run(name: str, trap_templates: list[np.ndarray], par_templates: list[np.ndarray],
        trap_info: list[dict], par_info: list[dict]) -> None:
    data = load_experimenter(name)
    t0_ms = float(data["ACC"]["timestamp_ms"].iloc[0])
    acc_frame = build_acc_frame(data["ACC"], t0_ms)
    height_frame = build_height_frame(data["PRS"], t0_ms)

    ts, vel_lpf, fs = session_velocity(acc_frame)
    print(f"{name}: fs={fs:.1f} Hz, {len(ts)/fs:.0f} s")

    # Resample/reinterpolate templates to session fs (they were built at 100 Hz)
    def resample(tpl: np.ndarray) -> np.ndarray:
        if abs(fs - FS_TEMPLATE) < 0.5:
            return tpl
        target_n = int(len(tpl) * fs / FS_TEMPLATE)
        xp = np.linspace(0, 1, len(tpl))
        xn = np.linspace(0, 1, target_n)
        return np.interp(xn, xp, tpl)

    trap_r = [resample(t) for t in trap_templates]
    par_r = [resample(t) for t in par_templates]

    conf_trap = confidence_max(vel_lpf, trap_r)
    conf_par = confidence_max(vel_lpf, par_r)

    # Smooth confidence curves with a rolling mean for readability
    smooth_win = max(3, int(CONFIDENCE_SMOOTH_SEC * fs))
    conf_trap_s = pd.Series(conf_trap).rolling(smooth_win, center=True, min_periods=1).mean().to_numpy()
    conf_par_s = pd.Series(conf_par).rolling(smooth_win, center=True, min_periods=1).mean().to_numpy()

    # GT segments
    cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)
    segments = Segmenter(cfg).detect(height_frame)

    def _plot_conf(ax, conf, color, shape_label):
        ax.plot(ts, conf, color=color, lw=1.2, label=f"max |NCC| — {shape_label}")
        for _, row in segments.iterrows():
            ax.axvspan(row["start_ci"][0], row["end_ci"][1], color="yellow", alpha=0.25)
        ax.set_ylabel("confidence")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_vel(ax):
        ax.plot(ts, vel_lpf, color="tab:gray", lw=0.8, label="session velocity (LPF)")
        for _, row in segments.iterrows():
            up = row["type"] == "up"
            ax.axvspan(row["start_ci"][0], row["end_ci"][1],
                       color="tab:blue" if up else "tab:red", alpha=0.15,
                       label="GT up" if up else "GT down")
        ax.axhline(0, color="k", lw=0.3)
        ax.set_xlabel("time (s)"); ax.set_ylabel("velocity (m/s)")
        handles, lbls = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, lbls):
            seen.setdefault(l, h)
        ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # ---- TRAPEZOID plot ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    _plot_conf(a1, conf_trap_s, "tab:orange", f"{len(trap_r)} trapezoid templates")
    a1.set_title(f"Trapezoid matched-filter confidence — {name}  (smoothed {CONFIDENCE_SMOOTH_SEC}s)")
    _plot_vel(a2)
    fig.tight_layout()
    out_trap = OUT_DIR / f"pulse_detect_trapezoid_{name}.png"
    fig.savefig(out_trap, dpi=120); plt.close(fig)
    print(f"  -> {out_trap}")

    # ---- PARABOLA plot ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    _plot_conf(a1, conf_par_s, "tab:blue", f"{len(par_r)} parabola templates")
    a1.set_title(f"Parabola matched-filter confidence — {name}  (smoothed {CONFIDENCE_SMOOTH_SEC}s)")
    _plot_vel(a2)
    fig.tight_layout()
    out_par = OUT_DIR / f"pulse_detect_parabola_{name}.png"
    fig.savefig(out_par, dpi=120); plt.close(fig)
    print(f"  -> {out_par}")

    # ---- Gated detection ----
    detections = apply_gates(
        ts, vel_lpf, acc_frame, np.maximum(conf_trap_s, conf_par_s), fs,
        ncc_thresh=NCC_THRESH, vel_thresh=VEL_THRESH,
        acc_var_max=ACC_VAR_MAX, min_duration=MIN_DURATION,
    )
    render_detection_plot(name, ts, vel_lpf, acc_frame, np.maximum(conf_trap_s, conf_par_s),
                          detections, segments, fs)


def main() -> None:
    rides, labels = load_rides()
    trap_picks = cluster_templates(rides, labels, "trapezoid", N_TEMPLATES_PER_SHAPE)
    par_picks = cluster_templates(rides, labels, "parabola", N_TEMPLATES_PER_SHAPE)
    print(f"Selected {len(trap_picks)} trapezoid + {len(par_picks)} parabola templates")

    trap_templates = [render_template(p, "trapezoid") for p in trap_picks]
    par_templates = [render_template(p, "parabola") for p in par_picks]

    # Write a summary of chosen templates
    OUT_DIR.mkdir(exist_ok=True)
    lines = ["# Pulse-detection templates", "", "## Trapezoid templates"]
    for p in trap_picks:
        t = p["trap"]
        lines.append(f"- {p['key']}: a_max={t['a_max']:.3f}  v_max={t['v_max']:.3f}  "
                     f"W={t['t_end']-t['t_start']:.2f}s")
    lines += ["", "## Parabola templates"]
    for p in par_picks:
        pp = p["par"]
        lines.append(f"- {p['key']}: v_peak={pp['v_peak']:.3f}  W={pp['W']:.2f}s  p={pp['p']:.2f}")
    (OUT_DIR / "pulse_detect_templates.md").write_text("\n".join(lines) + "\n")

    # Also plot the templates themselves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    for i, t in enumerate(trap_templates):
        x = np.arange(len(t)) / FS_TEMPLATE
        ax1.plot(x, t + t.mean(), lw=1.2, alpha=0.8, label=trap_picks[i]["key"])
    ax1.set_title(f"{len(trap_templates)} trapezoid templates")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("v (m/s)")
    ax1.legend(fontsize=6, ncol=2)
    ax1.grid(True, alpha=0.3)
    for i, t in enumerate(par_templates):
        x = np.arange(len(t)) / FS_TEMPLATE
        ax2.plot(x, t + t.mean(), lw=1.2, alpha=0.8, label=par_picks[i]["key"])
    ax2.set_title(f"{len(par_templates)} parabola templates")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("v (m/s)")
    ax2.legend(fontsize=6, ncol=2)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pulse_detect_templates.png", dpi=120)
    plt.close(fig)

    for name in ("oria", "roy_turgman"):
        run(name, trap_templates, par_templates, trap_picks, par_picks)


if __name__ == "__main__":
    main()
