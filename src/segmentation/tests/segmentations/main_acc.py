"""Compare pressure-GT segmentation with the ACC-only segmenter end-to-end.

Saves every run to ``run_results/YYYY_MM_DD_HH_MM_SS/`` with metrics,
config, segments, plot and notes.

Run with:
    python -m src.tests.segmentations.main_acc [experimenter] [note]
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive: figures save to disk, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


VELOCITY_LPF_HZ = 0.3  # cut walking (>1 Hz) and keep elevator band (<0.5 Hz)


def lowpass_velocity(v: np.ndarray, fs: float, cutoff_hz: float = VELOCITY_LPF_HZ) -> np.ndarray:
    """Zero-phase Butterworth low-pass on the velocity. Removes walking /
    handling transients so the elevator-band bump stands out."""
    nyq = 0.5 * fs
    sos = butter(4, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, v)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter,
    SegmentationMetrics, ci_center,
    TemplateMatchConfig, fit_templates, save_templates, compute_match_scores,
)
from src.plotting import plot_templates_overlay, plot_match_scores


RUN_RESULTS_ROOT = Path(__file__).resolve().parents[3] / "run_results"
CALIBRATORS_DIR = (
    Path(__file__).resolve().parents[2]
    / "algorithms" / "segmentation_algorithms" / "calibrators"
)
CONFIG_JSON = (
    Path(__file__).resolve().parents[2]
    / "algorithms" / "segmentation_algorithms" / "config.json"
)


def build_acc_frame(acc_raw: pd.DataFrame, t0_ms: float) -> pd.DataFrame:
    t = (acc_raw["timestamp_ms"].to_numpy() - t0_ms) / 1000.0
    mag = np.sqrt(acc_raw["x"] ** 2 + acc_raw["y"] ** 2 + acc_raw["z"] ** 2).to_numpy()
    return pd.DataFrame({
        "time": t,
        "x": acc_raw["x"].to_numpy(),
        "y": acc_raw["y"].to_numpy(),
        "z": acc_raw["z"].to_numpy(),
        "mag": mag,
    })


def build_height_frame(prs: pd.DataFrame, t0_ms: float) -> pd.DataFrame:
    t = (prs["timestamp_ms"].to_numpy() - t0_ms) / 1000.0
    h = prs["GT_height_m"].to_numpy()
    h_smooth = pd.Series(h).rolling(window=51, center=True, min_periods=1).median().to_numpy()
    return pd.DataFrame({"time": t, "height": h_smooth})


def compute_metrics(acc_segments: pd.DataFrame, gt_segments: pd.DataFrame) -> dict:
    det = SegmentationMetrics.match_segments(acc_segments, gt_segments)
    probs = np.array([0.5 * (lo + hi) for lo, hi in acc_segments["probability_ci"]]) if len(acc_segments) else np.array([])
    p_lo = np.array([lo for lo, _ in acc_segments["probability_ci"]]) if len(acc_segments) else np.array([])
    p_hi = np.array([hi for _, hi in acc_segments["probability_ci"]]) if len(acc_segments) else np.array([])
    labels = np.zeros(len(acc_segments), dtype=int)
    for pi in det.matched_pred_idx:
        labels[pi] = 1
    start_res = np.array([abs(ci_center(acc_segments.iloc[pi]["start_ci"]) - ci_center(gt_segments.iloc[gi]["start_ci"]))
                          for pi, gi in zip(det.matched_pred_idx, det.matched_gt_idx)])
    end_res = np.array([abs(ci_center(acc_segments.iloc[pi]["end_ci"]) - ci_center(gt_segments.iloc[gi]["end_ci"]))
                        for pi, gi in zip(det.matched_pred_idx, det.matched_gt_idx)])
    start_q = 0.5 * (acc_segments.iloc[0]["start_ci"][1] - acc_segments.iloc[0]["start_ci"][0]) if len(acc_segments) else 0.0
    end_q = 0.5 * (acc_segments.iloc[0]["end_ci"][1] - acc_segments.iloc[0]["end_ci"][0]) if len(acc_segments) else 0.0
    return SegmentationMetrics.summary(
        acc_segments, gt_segments,
        probs=probs, labels=labels, p_lo=p_lo, p_hi=p_hi,
        start_residuals_sec=start_res, end_residuals_sec=end_res,
        start_q_sec=start_q, end_q_sec=end_q,
    )


def _segments_to_records(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "start_ci": list(r["start_ci"]),
            "end_ci": list(r["end_ci"]),
            "duration": float(r["duration"]),
            "type": str(r["type"]),
            "probability_ci": list(r["probability_ci"]),
        })
    return rows


def make_interval_plot(
    gt_segments: pd.DataFrame, acc_segments: pd.DataFrame, session_duration: float,
) -> plt.Figure:
    """Timeline bars of GT (green) and ACC (red) segments. Blue vertical
    lines mark the start_ci / end_ci endpoints so CI width is visible; each
    bar is labeled with the segment index.
    """
    fig, ax = plt.subplots(figsize=(16, 3.2))
    for i, (_, row) in enumerate(gt_segments.iterrows()):
        (s_lo, s_hi) = row["start_ci"]
        (e_lo, e_hi) = row["end_ci"]
        s_mid = ci_center(row["start_ci"])
        e_mid = ci_center(row["end_ci"])
        ax.barh(y=1.0, left=s_mid, width=e_mid - s_mid, height=0.55,
                color="tab:green", alpha=0.85, edgecolor="black", linewidth=0.3)
        for x in (s_lo, s_hi, e_lo, e_hi):
            ax.vlines(x, 0.72, 1.28, colors="tab:blue", linewidth=1.0, alpha=0.9)
        ax.text(0.5 * (s_mid + e_mid), 1.0, str(i),
                ha="center", va="center", fontsize=7, color="black")
    for i, (_, row) in enumerate(acc_segments.iterrows()):
        (s_lo, s_hi) = row["start_ci"]
        (e_lo, e_hi) = row["end_ci"]
        s_mid = ci_center(row["start_ci"])
        e_mid = ci_center(row["end_ci"])
        ax.barh(y=0.0, left=s_mid, width=e_mid - s_mid, height=0.55,
                color="tab:red", alpha=0.85, edgecolor="black", linewidth=0.3)
        for x in (s_lo, s_hi, e_lo, e_hi):
            ax.vlines(x, -0.28, 0.28, colors="tab:blue", linewidth=1.0, alpha=0.9)
        ax.text(0.5 * (s_mid + e_mid), 0.0, str(i),
                ha="center", va="center", fontsize=7, color="white")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ACC-only", "Pressure GT"])
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, session_duration)
    ax.set_ylim(-0.6, 1.6)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title(
        f"Segment timeline — {len(gt_segments)} GT / {len(acc_segments)} predicted "
        f"(blue lines = start_ci / end_ci bounds)"
    )
    fig.tight_layout()
    return fig


def compute_velocity(acc_frame: pd.DataFrame) -> np.ndarray:
    """Integrate DC-removed acceleration magnitude — matches the
    ``build_integrals_frame`` definition in ``tests/segmentations/main.py``.
    """
    t = acc_frame["time"].to_numpy()
    a_lin = acc_frame["mag"].to_numpy() - acc_frame["mag"].mean()
    dt = np.diff(t, prepend=t[0])
    return np.cumsum(a_lin * dt)


def make_velocity_plots(
    acc_frame: pd.DataFrame, fs: float,
    gt_segments: pd.DataFrame, acc_segments: pd.DataFrame,
    window_sec: float,
    height_frame: pd.DataFrame | None = None,
) -> plt.Figure:
    """Single panel: raw and LPF'd cumulative velocity, plus (on a twin
    axis) the LPF barometer height for sanity comparison against GT."""
    t = acc_frame["time"].to_numpy()
    v_cumulative = compute_velocity(acc_frame)
    v_lpf = lowpass_velocity(v_cumulative, fs)

    fig, ax_cum = plt.subplots(figsize=(14, 4.2))

    ax_cum.plot(t, v_cumulative, linewidth=0.7, color="tab:blue", alpha=0.55, label="raw v")
    ax_cum.plot(t, v_lpf, linewidth=1.4, color="tab:red", label=f"LPF v (cutoff {VELOCITY_LPF_HZ} Hz)")
    ax_cum.set_ylabel("cumulative v (m/s)")
    ax_cum.set_title("Cumulative velocity — raw vs low-pass filtered (+ barometer height)")
    ax_cum.grid(True, alpha=0.3)
    for _, row in gt_segments.iterrows():
        ax_cum.axvspan(ci_center(row["start_ci"]), ci_center(row["end_ci"]),
                       color="tab:green", alpha=0.18)
    for _, row in acc_segments.iterrows():
        ax_cum.axvspan(ci_center(row["start_ci"]), ci_center(row["end_ci"]),
                       color="tab:red", alpha=0.12)

    handles, labels = ax_cum.get_legend_handles_labels()
    if height_frame is not None and len(height_frame) > 0:
        ax_h = ax_cum.twinx()
        h_t = height_frame["time"].to_numpy()
        h = height_frame["height"].to_numpy()
        # extra LPF on the already-median-smoothed height to match the
        # visual smoothness of LPF v
        win = max(11, int(round(fs * 1.0)))
        h_lpf = pd.Series(h).rolling(window=win, center=True, min_periods=1).mean().to_numpy()
        line_h, = ax_h.plot(h_t, h_lpf, linewidth=1.4, color="tab:purple",
                            alpha=0.85, label="barometer height (LPF)")
        ax_h.set_ylabel("height (m)", color="tab:purple")
        ax_h.tick_params(axis="y", colors="tab:purple")
        handles.append(line_h)
        labels.append(line_h.get_label())

    ax_cum.legend(handles, labels, loc="best", fontsize=9)
    ax_cum.set_xlabel("time (s)")
    return fig


def save_run(
    run_dir: Path, name: str, note: str,
    acc_segments: pd.DataFrame, gt_segments: pd.DataFrame,
    metrics: dict,
    vel_fig: plt.Figure | None = None,
    interval_fig: plt.Figure | None = None,
    template_overlay_fig: plt.Figure | None = None,
    match_scores_fig: plt.Figure | None = None,
    tm_segments: pd.DataFrame | None = None,
    tm_metrics: dict | None = None,
    tm_interval_fig: plt.Figure | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    with open(run_dir / "acc_segments.json", "w") as f:
        json.dump(_segments_to_records(acc_segments), f, indent=2)
    with open(run_dir / "gt_segments.json", "w") as f:
        json.dump(_segments_to_records(gt_segments), f, indent=2)
    shutil.copy2(CONFIG_JSON, run_dir / "config.json")
    if CALIBRATORS_DIR.is_dir():
        dst = run_dir / "calibrators"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(CALIBRATORS_DIR, dst)
    if vel_fig is not None:
        vel_fig.savefig(run_dir / "velocity.png", dpi=140)
        plt.close(vel_fig)
    if interval_fig is not None:
        interval_fig.savefig(run_dir / "intervals.png", dpi=140)
        plt.close(interval_fig)
    if template_overlay_fig is not None:
        template_overlay_fig.savefig(run_dir / "templates_overlay.png", dpi=140)
        plt.close(template_overlay_fig)
    if match_scores_fig is not None:
        match_scores_fig.savefig(run_dir / "match_scores.png", dpi=140)
        plt.close(match_scores_fig)
    if tm_interval_fig is not None:
        tm_interval_fig.savefig(run_dir / "intervals_template_match.png", dpi=140)
        plt.close(tm_interval_fig)
    if tm_segments is not None:
        with open(run_dir / "tm_segments.json", "w") as f:
            json.dump(_segments_to_records(tm_segments), f, indent=2)
    if tm_metrics is not None:
        with open(run_dir / "tm_metrics.json", "w") as f:
            json.dump(tm_metrics, f, indent=2, default=float)
    with open(run_dir / "notes.md", "w") as f:
        det = metrics.get("detection", {})
        cal = metrics.get("calibration", {})
        f.write(f"# Run: {run_dir.name}\n\n")
        f.write(f"**Experimenter:** {name}\n\n")
        if note:
            f.write(f"**Note:** {note}\n\n")
        f.write("## Detection\n")
        f.write(f"- TP: {det.get('tp')}  FP: {det.get('fp')}  FN: {det.get('fn')}\n")
        f.write(f"- Precision: {det.get('precision'):.3f}  Recall: {det.get('recall'):.3f}  F1: {det.get('f1'):.3f}\n\n")
        if cal:
            f.write("## Calibration\n")
            f.write(f"- ECE: {cal.get('ece'):.3f}  Brier: {cal.get('brier'):.3f}  N: {cal.get('n')}\n")
            if "prob_ci_coverage" in cal:
                f.write(f"- Prob-CI coverage: {cal['prob_ci_coverage']:.3f}\n")
        f.write("\n## Artifacts\n")
        f.write("- `metrics.json` — full metrics summary\n")
        f.write("- `acc_segments.json` / `gt_segments.json` — segment outputs\n")
        f.write("- `config.json` — segmentation hyperparameters\n")
        f.write("- `calibrators/` — fitted LR/IVAP/edge-conformal artifacts\n")
        f.write("- `velocity.png` / `intervals.png` — diagnostic plots\n")


def main(name: str = "uriya", note: str = "") -> None:
    data = load_experimenter(name)
    acc_raw = data["ACC"]
    prs = data.get("PRS")
    t0_ms = float(acc_raw["timestamp_ms"].iloc[0])
    acc_frame = build_acc_frame(acc_raw, t0_ms)
    height_frame = build_height_frame(prs, t0_ms) if prs is not None else None

    pressure_segmenter = Segmenter(SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER))
    gt_segments = pressure_segmenter.detect(height_frame) if height_frame is not None else pd.DataFrame()
    acc_segmenter = Segmenter(SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.ACC_ONLY))
    acc_data = acc_frame[["time", "x", "y", "z"]]
    acc_segments = acc_segmenter.detect(acc_data)

    print(f"Pressure GT: {len(gt_segments)} segments")
    print(f"ACC-only:    {len(acc_segments)} segments")

    metrics = compute_metrics(acc_segments, gt_segments)
    print(f"\n=== Metrics ({name}) [acc_only] ===")
    print(json.dumps(metrics, indent=2, default=float))
    print("===========================================\n")

    params = acc_segmenter.params
    vel_fig = make_velocity_plots(
        acc_frame, fs=float(params.get("fs_hz", 100.0)),
        gt_segments=gt_segments, acc_segments=acc_segments,
        window_sec=float(params.get("window_sec", 2.0)),
        height_frame=height_frame,
    )
    interval_fig = make_interval_plot(
        gt_segments, acc_segments, session_duration=float(acc_frame["time"].iloc[-1]),
    )

    # ---- template-match path: fit per-experimenter templates on GT, then match ----
    tm_cfg = TemplateMatchConfig()
    tm_segments = pd.DataFrame()
    tm_metrics = None
    template_overlay_fig = match_scores_fig = tm_interval_fig = None
    if len(gt_segments) > 0:
        templates = fit_templates(acc_data, gt_segments, tm_cfg, name=name)
        templates_path = CALIBRATORS_DIR / f"templates_{name}.json"
        save_templates(templates, templates_path)
        tm_cfg = TemplateMatchConfig(templates_path=templates_path)
        tm_segmenter = Segmenter(SEGMENT_ALGORITHM_CONFIG(
            algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
            overrides={"templates_path": str(templates_path)},
        ))
        tm_segments = tm_segmenter.detect(acc_data)
        print(f"Template-match: {len(tm_segments)} segments  (n_rides used to fit: {templates.meta.get('n_rides', 0)})")
        tm_metrics = compute_metrics(tm_segments, gt_segments)
        print(f"\n=== Metrics ({name}) [acc_template_match] ===")
        print(json.dumps(tm_metrics, indent=2, default=float))
        print("===========================================\n")

        template_overlay_fig = plot_templates_overlay(acc_data, gt_segments, templates, tm_cfg)
        scores = compute_match_scores(acc_data, templates, tm_cfg)
        match_scores_fig = plot_match_scores(
            scores["t"], scores["v_lpf"],
            scores["start_score"], scores["end_score"],
            tm_segments, gt_segments,
            threshold=float(tm_cfg.enter_threshold),
        )
        tm_interval_fig = make_interval_plot(
            gt_segments, tm_segments, session_duration=float(acc_frame["time"].iloc[-1]),
        )

    stamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    run_dir = RUN_RESULTS_ROOT / f"{stamp}_{name}"
    save_run(run_dir, name, note, acc_segments, gt_segments, metrics,
             vel_fig=vel_fig, interval_fig=interval_fig,
             template_overlay_fig=template_overlay_fig,
             match_scores_fig=match_scores_fig,
             tm_segments=tm_segments if len(tm_segments) else None,
             tm_metrics=tm_metrics,
             tm_interval_fig=tm_interval_fig)
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    from src.data.loader import DATA_ROOT
    args = sys.argv[1:]
    first = args[0] if args else "uriya"
    note = " ".join(args[1:]) if len(args) > 1 else ""
    if first == "all":
        names = sorted([p.name for p in Path(DATA_ROOT).iterdir()
                        if p.is_dir() and not p.name.startswith(".")])
    else:
        names = [first]
    for n in names:
        print(f"\n######## {n} ########")
        main(n, note)
