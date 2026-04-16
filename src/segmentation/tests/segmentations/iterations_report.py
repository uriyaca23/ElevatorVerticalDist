"""Regenerate every acc-segmentation iteration tried this session, saving a
score/GT/detection PNG and a README with numbers for each iteration.

Output: ``run_results/acc_segmentation_iterations/iter_{N}_{slug}/``"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter, ci_center,
    SegmentationMetrics,
)
from src.algorithms.segmentation_algorithms.accelerometer_only.acc_segmentation import (
    _compute_a_vert, lowpass, compute_velocity, drift_residual_score,
    walkband_rms, _bandpass, hysteresis_segments, stillness_hysteresis_segments,
    step_rate, zupt_integrate, sliding_zupt_disp,
)
from src.tests.segmentations.main_acc import build_acc_frame, build_height_frame


OUT_ROOT = REPO_ROOT / "run_results" / "acc_segmentation_iterations"


@dataclass
class IterResult:
    name: str
    slug: str
    description: str
    method_summary: str
    score_label: str
    score_hysteresis: str  # "above" or "below"
    per_exp: dict  # exp -> dict(score, detections, gt, t, tp, fp, fn, recall, precision)


def build_frames(exp: str):
    d = load_experimenter(exp)
    t0 = float(d["ACC"]["timestamp_ms"].iloc[0])
    acc = build_acc_frame(d["ACC"], t0)
    h = build_height_frame(d["PRS"], t0)
    gt = Segmenter(SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)).detect(h)
    return acc, gt


def evaluate(pred_windows: list[tuple[float, float]], gt) -> tuple[int, int, int]:
    pred_df = pd.DataFrame([
        {"start_ci": (s, s), "end_ci": (e, e), "duration": e - s,
         "type": "unknown", "probability_ci": (0.5, 0.5)}
        for s, e in pred_windows
    ], columns=["start_ci", "end_ci", "duration", "type", "probability_ci"])
    res = SegmentationMetrics.iou_match_segments(pred_df, gt, iou_threshold=0.3)
    return res.tp, res.fp, res.fn


def plot_iter(out_dir: Path, res: IterResult):
    out_dir.mkdir(parents=True, exist_ok=True)
    for exp, d in res.per_exp.items():
        fig, ax = plt.subplots(figsize=(14, 3.5))
        ax.plot(d["t"], d["score"], lw=0.8, color="tab:blue", label=res.score_label)
        for _, r in d["gt"].iterrows():
            ax.axvspan(ci_center(r["start_ci"]), ci_center(r["end_ci"]),
                       color="tab:green", alpha=0.18, label="GT" if _ == 0 else None)
        for s, e in d["detections"]:
            ax.axvspan(s, e, color="tab:red", alpha=0.28)
        ax.set_title(
            f"{res.name} — {exp}  TP={d['tp']} FP={d['fp']} FN={d['fn']}  "
            f"recall={d['recall']:.2f} precision={d['precision']:.2f}"
        )
        ax.set_xlabel("time (s)"); ax.set_ylabel(res.score_label)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{exp}.png", dpi=130)
        plt.close(fig)

    readme = out_dir / "README.md"
    lines = [f"# {res.name}", "", f"**Slug:** `{res.slug}`", "",
             "## Method", res.method_summary, "",
             "## Results", "", "| Experimenter | TP | FP | FN | Recall | Precision |",
             "|---|---|---|---|---|---|"]
    for exp, d in res.per_exp.items():
        lines.append(
            f"| {exp} | {d['tp']} | {d['fp']} | {d['fn']} | {d['recall']:.2f} | {d['precision']:.2f} |"
        )
    lines += ["", "## Artifacts", "- `{experimenter}.png` — score trace, GT (green), predictions (red)"]
    readme.write_text("\n".join(lines) + "\n")


# --------------- iteration 1: legacy drift-residual (baseline) -----------------

def iter_01(exp: str):
    acc, gt = build_frames(exp)
    fs = 100.0
    t = acc["time"].to_numpy()
    ax, ay, az = [acc[c].to_numpy() for c in "xyz"]
    a_vert = _compute_a_vert(ax, ay, az, fs)
    score = drift_residual_score(a_vert, fs, detrend_sec=90.0, local_var_sec=8.0)
    raw = hysteresis_segments(
        t, score, enter=0.3, exit_=0.05, min_duration_sec=3.0,
        merge_gap_sec=4.0, pad_sec=2.0, t_min=float(t[0]), t_max=float(t[-1]),
    )
    windows = [(s, e) for s, e, _ in raw]
    tp, fp, fn = evaluate(windows, gt)
    return {
        "t": t, "score": score, "detections": windows, "gt": gt,
        "tp": tp, "fp": fp, "fn": fn,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
    }


# --------------- iteration 2: walkband RMS stillness + displacement gate -------

def iter_02(exp: str):
    acc, gt = build_frames(exp)
    fs = 100.0
    t = acc["time"].to_numpy()
    ax, ay, az = [acc[c].to_numpy() for c in "xyz"]
    walk = walkband_rms(ax, ay, az, fs, 1.2, 2.8, 2.0)
    raw = stillness_hysteresis_segments(
        t, walk, enter=0.25, exit_=0.5,
        min_duration_sec=6.0, merge_gap_sec=3.0, pad_sec=1.0,
        t_min=float(t[0]), t_max=float(t[-1]),
    )
    a_vert = _compute_a_vert(ax, ay, az, fs)
    v = lowpass(compute_velocity(a_vert, fs), fs)
    keep = []
    for s, e, _ in raw:
        i0, i1 = int(np.searchsorted(t, s)), int(np.searchsorted(t, e))
        if i1 > i0 and np.max(np.abs(v[i0:i1] - v[i0])) >= 0.15:
            keep.append((s, e))
    tp, fp, fn = evaluate(keep, gt)
    return {
        "t": t, "score": walk, "detections": keep, "gt": gt,
        "tp": tp, "fp": fp, "fn": fn,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
    }


# --------------- iteration 3: session-normalized walkband ----------------------

def iter_03(exp: str):
    acc, gt = build_frames(exp)
    fs = 100.0
    t = acc["time"].to_numpy()
    ax, ay, az = [acc[c].to_numpy() for c in "xyz"]
    walk = walkband_rms(ax, ay, az, fs, 1.2, 2.8, 2.0)
    walk_n = walk / (float(np.percentile(walk, 75)) + 1e-9)
    raw = stillness_hysteresis_segments(
        t, walk_n, enter=0.6, exit_=1.2,
        min_duration_sec=6.0, merge_gap_sec=3.0, pad_sec=1.0,
        t_min=float(t[0]), t_max=float(t[-1]),
    )
    a_vert = _compute_a_vert(ax, ay, az, fs)
    v = lowpass(compute_velocity(a_vert, fs), fs)
    keep = []
    for s, e, _ in raw:
        i0, i1 = int(np.searchsorted(t, s)), int(np.searchsorted(t, e))
        if i1 > i0 and np.max(np.abs(v[i0:i1] - v[i0])) >= 0.15:
            keep.append((s, e))
    tp, fp, fn = evaluate(keep, gt)
    return {
        "t": t, "score": walk_n, "detections": keep, "gt": gt,
        "tp": tp, "fp": fp, "fn": fn,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
    }


# --------------- iteration 4: bandpassed integrated a_vert (raw) ---------------

def iter_04(exp: str):
    acc, gt = build_frames(exp)
    fs = 100.0
    t = acc["time"].to_numpy()
    ax, ay, az = [acc[c].to_numpy() for c in "xyz"]
    a_vert = _compute_a_vert(ax, ay, az, fs)
    a_band = _bandpass(a_vert, fs, 0.03, 0.3)
    v_band = np.cumsum(a_band) / fs
    W = int(8 * fs)
    disp = pd.Series(np.abs(v_band)).rolling(W, center=True, min_periods=1).max().to_numpy()
    raw = hysteresis_segments(
        t, disp, enter=2.0, exit_=1.0,
        min_duration_sec=6.0, merge_gap_sec=3.0, pad_sec=1.0,
        t_min=float(t[0]), t_max=float(t[-1]),
    )
    walk = walkband_rms(ax, ay, az, fs, 1.2, 2.8, 2.0)
    walk_n = walk / (float(np.percentile(walk, 75)) + 1e-9)
    keep = []
    for s, e, _ in raw:
        i0, i1 = int(np.searchsorted(t, s)), int(np.searchsorted(t, e))
        if i1 > i0 and float(np.mean(walk_n[i0:i1])) <= 1.2:
            keep.append((s, e))
    tp, fp, fn = evaluate(keep, gt)
    return {
        "t": t, "score": disp, "detections": keep, "gt": gt,
        "tp": tp, "fp": fp, "fn": fn,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
    }


# --------------- iteration 5: session-normalized disp + stillness AND ----------

def iter_05(exp: str):
    acc, gt = build_frames(exp)
    fs = 100.0
    t = acc["time"].to_numpy()
    ax, ay, az = [acc[c].to_numpy() for c in "xyz"]
    a_vert = _compute_a_vert(ax, ay, az, fs)
    a_band = _bandpass(a_vert, fs, 0.03, 0.3)
    v_band = np.cumsum(a_band) / fs
    W = int(8 * fs)
    disp_raw = pd.Series(np.abs(v_band)).rolling(W, center=True, min_periods=1).max().to_numpy()
    disp = disp_raw / (float(np.median(disp_raw)) + 1e-9)
    walk = walkband_rms(ax, ay, az, fs, 1.2, 2.8, 2.0)
    walk_n = walk / (float(np.percentile(walk, 75)) + 1e-9)
    raw = hysteresis_segments(
        t, disp, enter=1.1, exit_=0.9,
        min_duration_sec=6.0, merge_gap_sec=3.0, pad_sec=1.0,
        t_min=float(t[0]), t_max=float(t[-1]),
    )
    keep = []
    for s, e, _ in raw:
        i0, i1 = int(np.searchsorted(t, s)), int(np.searchsorted(t, e))
        if i1 > i0 and float(np.mean(walk_n[i0:i1])) <= 1.2:
            keep.append((s, e))
    tp, fp, fn = evaluate(keep, gt)
    return {
        "t": t, "score": disp, "detections": keep, "gt": gt,
        "tp": tp, "fp": fp, "fn": fn,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
    }


# --------------- iteration 6: step-rate stillness + ZUPT integrate ------------

def iter_06(exp: str):
    acc, gt = build_frames(exp)
    fs = 100.0
    t = acc["time"].to_numpy()
    ax, ay, az = [acc[c].to_numpy() for c in "xyz"]
    a_vert = _compute_a_vert(ax, ay, az, fs)
    sr = step_rate(ax, ay, az, fs, 4.0)
    max_sr = 0.3
    min_dur = 6.0
    min_disp = 0.5

    still = sr < max_sr
    n = len(still); i = 0
    intervals = []
    while i < n:
        if still[i]:
            j = i
            while j < n and still[j]:
                j += 1
            if (t[j-1] - t[i]) >= min_dur:
                intervals.append((i, j-1))
            i = j
        else:
            i += 1
    windows = []
    for s, e in intervals:
        v_c, d = zupt_integrate(a_vert[s:e+1], fs)
        if float(np.max(d) - np.min(d)) < min_disp:
            continue
        v_abs = np.abs(v_c)
        if v_abs.max() < 1e-6:
            continue
        mask = v_abs > 0.2 * v_abs.max()
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        windows.append((float(t[s + idxs[0]]) - 1.0, float(t[s + idxs[-1]]) + 1.0))
    tp, fp, fn = evaluate(windows, gt)
    return {
        "t": t, "score": sr, "detections": windows, "gt": gt,
        "tp": tp, "fp": fp, "fn": fn,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
    }


# --------------- iteration 7: sliding ZUPT + var(a_vert) combined -------------

def iter_07(exp: str):
    acc, gt = build_frames(exp)
    fs = 100.0
    t = acc["time"].to_numpy()
    ax, ay, az = [acc[c].to_numpy() for c in "xyz"]
    a_vert = _compute_a_vert(ax, ay, az, fs)
    disp = sliding_zupt_disp(a_vert, fs, 10.0)
    W = int(4 * fs)
    var_av = pd.Series(a_vert * a_vert).rolling(W, center=True, min_periods=1).mean().to_numpy() - (
        pd.Series(a_vert).rolling(W, center=True, min_periods=1).mean().to_numpy() ** 2
    )
    sc = np.log10(np.maximum(disp, 1e-4)) - 0.7 * np.log10(np.maximum(var_av, 1e-6))
    scn = sc - float(np.median(sc))
    raw = hysteresis_segments(
        t, scn, enter=0.45, exit_=0.0,
        min_duration_sec=6.0, merge_gap_sec=3.0, pad_sec=1.0,
        t_min=float(t[0]), t_max=float(t[-1]),
    )
    windows = []
    for s, e, _ in raw:
        i0, i1 = int(np.searchsorted(t, s)), int(np.searchsorted(t, e))
        peak = float(disp[i0:i1+1].max()) if i1 > i0 else 0.0
        if peak <= 0:
            continue
        mask = disp[i0:i1+1] > 0.4 * peak
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        windows.append((float(t[i0 + idxs[0]]) - 1.0, float(t[i0 + idxs[-1]]) + 1.0))
    tp, fp, fn = evaluate(windows, gt)
    return {
        "t": t, "score": scn, "detections": windows, "gt": gt,
        "tp": tp, "fp": fp, "fn": fn,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
    }


ITERATIONS = [
    (iter_01, "01_baseline_drift_residual",
     "Iteration 1 — legacy drift-residual variance",
     "Variance of (v - long-rolling-median(v)) relative to session median. Hysteresis at >=0.3 (enter) / <=0.05 (exit).",
     "drift-residual score"),
    (iter_02, "02_walkband_stillness",
     "Iteration 2 — raw walk-band RMS stillness + displacement gate",
     "Rolling 2s RMS of 1.2-2.8 Hz bandpass of |a|. Stillness hysteresis (walk<=0.25 enter / >=0.5 exit). Keep only if LPF velocity swing >= 0.15 m/s inside window.",
     "walk-band RMS (raw)"),
    (iter_03, "03_walkband_session_norm",
     "Iteration 3 — session-normalized walk-band stillness",
     "Same as iter 2 but walk-RMS divided by session p75, so thresholds are scale-free across experimenters (0.6 enter / 1.2 exit).",
     "walk-band RMS / p75"),
    (iter_04, "04_bandpass_disp_raw",
     "Iteration 4 — bandpassed integrated a_vert displacement",
     "Bandpass a_vert in 0.03-0.3 Hz (elevator band), integrate, take peak |v| in 8 s rolling window. Threshold 2.0 enter / 1.0 exit. Stillness AND-gate: mean walk_norm <= 1.2 inside window.",
     "|v_bandpass|  peak (m/s)"),
    (iter_05, "05_disp_session_norm",
     "Iteration 5 — session-normalized displacement + stillness AND",
     "Iter 4 with displacement divided by its own session median. Thresholds 1.1 enter / 0.9 exit. Same stillness AND-gate.",
     "disp / session median"),
    (iter_06, "06_steprate_zupt",
     "Iteration 6 — step-rate stillness + ZUPT-integrated displacement",
     "PEDOMETER-style step detection (Pan-Tompkins-like peaks on 0.8-3 Hz bandpass of |a|, refractory 0.3 s). Stillness = <0.3 steps/s sustained >=6 s. Inside each stillness interval: ZUPT integration (v forced to 0 at both ends, linear-drift removed) gives drift-free displacement. Keep if Δd >=0.5 m. Endpoint refinement via |v| > 0.2*max|v|.",
     "step rate (steps/s)"),
    (iter_07, "07_sliding_zupt_plus_var",
     "Iteration 7 — sliding-ZUPT displacement AND-gated by var(a_vert)",
     "Sliding 10-s ZUPT at every sample (closed-form via double cumsum); walking integrates to near-0 displacement, elevator to ~floor height. Combined with log var(a_vert,4s) as 'quiet vertical axis' gate: score = log10 disp - 0.7 * log10 var. Hysteresis 0.45 enter / 0.0 exit on the session-median-centered score; refine endpoints with disp > 0.4*peak.",
     "log disp - 0.7 log var (session-centered)"),
]

if __name__ == "__main__":
    summary_lines = ["# Iteration summary",
                     "",
                     "Test bar: **recall >= 0.9 AND precision >= 0.7** on both experimenters,",
                     "best-match-per-GT IOU >= 0.3. Accelerometer-only input.",
                     "",
                     "| # | Name | oria R / P | roy R / P |", "|---|---|---|---|"]
    for fn, slug, name, method_summary, score_label in ITERATIONS:
        per_exp = {}
        for exp in ("oria", "roy_turgman"):
            per_exp[exp] = fn(exp)
        res = IterResult(name=name, slug=slug, description="", method_summary=method_summary,
                         score_label=score_label, score_hysteresis="above", per_exp=per_exp)
        plot_iter(OUT_ROOT / slug, res)
        o, r = per_exp["oria"], per_exp["roy_turgman"]
        summary_lines.append(
            f"| {slug.split('_')[0]} | {name} | {o['recall']:.2f} / {o['precision']:.2f} | "
            f"{r['recall']:.2f} / {r['precision']:.2f} |"
        )
        print(f"done: {slug}")
    (OUT_ROOT / "README.md").write_text("\n".join(summary_lines) + "\n")
    print(f"\nSummary: {OUT_ROOT/'README.md'}")
