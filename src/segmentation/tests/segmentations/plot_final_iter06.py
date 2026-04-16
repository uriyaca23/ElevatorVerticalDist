"""Rich multi-panel plot of iteration 06 (the best algorithm found): step-rate
stillness + ZUPT displacement. Saves per-experimenter PNG in the 06 folder."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter, ci_center,
    SegmentationMetrics,
)
from src.algorithms.segmentation_algorithms.accelerometer_only.acc_segmentation import (
    _compute_a_vert, step_rate, zupt_integrate,
)
from src.tests.segmentations.main_acc import build_acc_frame, build_height_frame

OUT_DIR = REPO_ROOT / "run_results" / "acc_segmentation_iterations" / "06_steprate_zupt"
MAX_STEP_RATE = 0.3
MIN_DURATION = 6.0
MIN_DISP = 0.5
PAD = 1.0


def detect(t, ax, ay, az, fs):
    a_vert = _compute_a_vert(ax, ay, az, fs)
    sr = step_rate(ax, ay, az, fs, 4.0)
    still = sr < MAX_STEP_RATE
    n = len(still)
    intervals = []
    i = 0
    while i < n:
        if still[i]:
            j = i
            while j < n and still[j]:
                j += 1
            if (t[j - 1] - t[i]) >= MIN_DURATION:
                intervals.append((i, j - 1))
            i = j
        else:
            i += 1

    # Per-session traces used by the diagnostic plot:
    #   v_raw_trace  — integrated a_vert inside each stillness interval,
    #                  starting at zero at the interval's left edge (before
    #                  any drift correction).
    #   ramp_trace   — the linear drift we subtract per interval (line from
    #                  0 at left edge to v_raw(e) at right edge).
    #   v_c_trace    — v_raw - ramp (post-ZUPT velocity, ends at 0).
    #   zupt_disp_trace — integrated v_c (displacement, ends near 0 too).
    v_raw_trace = np.full_like(t, np.nan, dtype=float)
    ramp_trace = np.full_like(t, np.nan, dtype=float)
    v_c_trace = np.full_like(t, np.nan, dtype=float)
    zupt_disp_trace = np.full_like(t, np.nan, dtype=float)

    windows = []
    for s, e in intervals:
        seg = a_vert[s:e + 1]
        a_local = seg - float(np.mean(seg))
        v_raw = np.cumsum(a_local) / fs
        nlen = len(v_raw)
        ramp = np.linspace(0.0, v_raw[-1], nlen)
        v_c = v_raw - ramp
        d = np.cumsum(v_c) / fs

        v_raw_trace[s:e + 1] = v_raw
        ramp_trace[s:e + 1] = ramp
        v_c_trace[s:e + 1] = v_c
        zupt_disp_trace[s:e + 1] = d - d.min()

        d_pp = float(np.max(d) - np.min(d))
        if d_pp < MIN_DISP:
            continue
        v_abs = np.abs(v_c)
        if v_abs.max() < 1e-6:
            continue
        mask = v_abs > 0.2 * v_abs.max()
        idxs = np.where(mask)[0]
        if not len(idxs):
            continue
        windows.append((float(t[s + idxs[0]]) - PAD, float(t[s + idxs[-1]]) + PAD))
    return {
        "step_rate": sr,
        "a_vert": a_vert,
        "v_raw": v_raw_trace,
        "ramp": ramp_trace,
        "v_c": v_c_trace,
        "zupt_disp": zupt_disp_trace,
        "intervals": intervals,
        "windows": windows,
    }


def plot_one(name: str):
    d = load_experimenter(name)
    t0 = float(d["ACC"]["timestamp_ms"].iloc[0])
    acc = build_acc_frame(d["ACC"], t0)
    h = build_height_frame(d["PRS"], t0)
    gt = Segmenter(SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)).detect(h)
    fs = 100.0
    t = acc["time"].to_numpy()
    ax, ay, az = [acc[c].to_numpy() for c in "xyz"]
    mag = np.sqrt(ax * ax + ay * ay + az * az)

    det = detect(t, ax, ay, az, fs)
    sr = det["step_rate"]; a_vert = det["a_vert"]
    v_raw = det["v_raw"]; ramp = det["ramp"]; v_c = det["v_c"]
    zupt = det["zupt_disp"]
    intervals = det["intervals"]; windows = det["windows"]

    pred_df = pd.DataFrame([
        {"start_ci": (s, s), "end_ci": (e, e), "duration": e - s,
         "type": "unknown", "probability_ci": (0.5, 0.5)}
        for s, e in windows
    ], columns=["start_ci", "end_ci", "duration", "type", "probability_ci"])
    res = SegmentationMetrics.iou_match_segments(pred_df, gt, iou_threshold=0.3)

    # figure layout: 6 signal panels + 1 short bar panel for GT/pred.
    fig, axes = plt.subplots(
        7, 1, figsize=(16, 14), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1.1, 1.1, 1, 0.5]},
    )

    # Panel 1: |a| magnitude, lightly smoothed
    axes[0].plot(t, pd.Series(mag).rolling(10, center=True, min_periods=1).mean(),
                 lw=0.5, color="0.35")
    axes[0].set_ylabel("|a|  (m/s²)")
    axes[0].set_title(
        f"Iteration 06 — step-rate stillness + ZUPT  ({name})    "
        f"TP={res.tp}  FP={res.fp}  FN={res.fn}    "
        f"recall={res.recall:.2f}   precision={res.precision:.2f}"
    )

    # Panel 2: step rate + stillness threshold (the GATE)
    axes[1].plot(t, sr, lw=0.7, color="tab:orange", label="step rate (4 s rolling)")
    axes[1].axhline(MAX_STEP_RATE, color="k", ls="--", lw=1.0,
                    label=f"stillness threshold = {MAX_STEP_RATE} steps/s")
    for s, e in intervals:
        axes[1].axvspan(t[s], t[e], color="tab:blue", alpha=0.10)
    axes[1].set_ylabel("steps / s")
    axes[1].legend(loc="upper right", fontsize=9)

    # Panel 3: a_vert (gravity-removed, light smoothing)
    a_smooth = pd.Series(a_vert).rolling(20, center=True, min_periods=1).mean().to_numpy()
    axes[2].plot(t, a_smooth, lw=0.5, color="0.4", label="a_vert (gravity removed)")
    axes[2].axhline(0, color="k", lw=0.5, alpha=0.5)
    axes[2].set_ylabel("a_vert  (m/s²)")
    axes[2].legend(loc="upper right", fontsize=9)

    # Panel 4: raw velocity v_raw (integrated a_vert) + the LINEAR DRIFT
    # ramp we subtract in each stillness interval. Both traces are only
    # defined inside stillness intervals (NaN outside = gaps in the line).
    axes[3].plot(t, v_raw, lw=0.9, color="tab:blue",
                 label="v_raw = ∫ a_vert  (per stillness interval)")
    axes[3].plot(t, ramp, lw=1.3, color="tab:red", ls="--",
                 label="linear drift (ZUPT ramp)  — subtracted")
    axes[3].axhline(0, color="k", lw=0.5, alpha=0.5)
    axes[3].set_ylabel("velocity  (m/s)")
    axes[3].legend(loc="upper right", fontsize=9)

    # Panel 5: post-ZUPT velocity v_c = v_raw - ramp. Zero at interval ends
    # by construction; nonzero in the middle only when there is real motion.
    axes[4].plot(t, v_c, lw=0.9, color="tab:purple",
                 label="v after ZUPT  (v_raw − ramp)")
    axes[4].axhline(0, color="k", lw=0.5, alpha=0.5)
    axes[4].set_ylabel("v  (m/s)")
    axes[4].legend(loc="upper right", fontsize=9)

    # Panel 6: ZUPT displacement trace + DISPLACEMENT THRESHOLD (the gate)
    axes[5].plot(t, zupt, lw=0.9, color="tab:blue",
                 label="ZUPT displacement  d = ∫ v_c")
    axes[5].axhline(MIN_DISP, color="tab:red", ls="--", lw=1.0,
                    label=f"min ride displacement = {MIN_DISP} m")
    axes[5].set_ylabel("Δd  (m)")
    axes[5].legend(loc="upper right", fontsize=9)

    # Panel 7: segment timeline — GT and predictions on SEPARATE rows
    ax_seg = axes[6]
    for _, r in gt.iterrows():
        s, e = ci_center(r["start_ci"]), ci_center(r["end_ci"])
        ax_seg.barh(y=1.0, left=s, width=e - s, height=0.55,
                    color="tab:green", alpha=0.85,
                    edgecolor="black", linewidth=0.3)
    for s, e in windows:
        ax_seg.barh(y=0.0, left=s, width=e - s, height=0.55,
                    color="tab:red", alpha=0.85,
                    edgecolor="black", linewidth=0.3)
    ax_seg.set_yticks([0.0, 1.0])
    ax_seg.set_yticklabels(["Predicted", "Ground truth"])
    ax_seg.set_ylim(-0.6, 1.6)
    ax_seg.set_xlabel("time (s)")

    for a in axes:
        a.grid(True, alpha=0.25)

    fig.tight_layout()
    out = OUT_DIR / f"{name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}  (TP={res.tp} FP={res.fp} FN={res.fn})")
    return res


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n in ("uriya", "roy_turgeman"):
        plot_one(n)
