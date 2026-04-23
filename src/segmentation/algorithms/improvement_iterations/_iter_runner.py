"""Improvement-iteration runner for the acc_template_match segmenter.

Runs the live detector + pair-filter across every experiment, classifies
each GT interval as clean / missed / gt_merged / gt_split, writes a
diagnostic PNG for every non-clean GT, and emits a metrics.json + per_gt.csv
+ notes.md stub for a single iteration.

Usage:
    venv/bin/python -m src.segmentation.algorithms.improvement_iterations._iter_runner \
        --iter 00 --slug baseline \
        --what "Current production pair_filter.py + config.json"

A fresh invocation overwrites the iteration folder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.loader import getExperimentData, list_experiments  # noqa: E402
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E402
    detect as _detect_mod,
    pair_filter as _pair_mod,
)
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal.detect import (  # noqa: E402
    DEFAULT_CONFIG, DetectConfig, classify_peak, detect, diagnose_window,
    heatmap_at, predict_intervals,
)
from src.segmentation.algorithms.accelerometer_only.template_match.fit_elevator_parameters.common import (  # noqa: E402
    trapezoid_kernel,
)
from src.segmentation.algorithms.metrics import IntervalPredictionMetrics  # noqa: E402
from src.segmentation.algorithms.metrics.metrics import (  # noqa: E402
    DEFAULT_MIN_OVERLAP_FRAC, DEFAULT_MIN_OVERLAP_S, _intervals_match,
)


ITER_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Per-experiment run — returns everything the classifier + renderer need.
# --------------------------------------------------------------------------
def _run_experiment(name: str) -> dict | None:
    """Load one experiment, run detect + pair-filter, return a bundle."""
    try:
        sensors, gt_df, metadata = getExperimentData(name)
    except Exception as exc:
        print(f"[error] {name}: {type(exc).__name__}: {exc}")
        return None
    acc = sensors.get("ACC")
    if acc is None or acc.empty or len(acc) < 2:
        print(f"[skip]  {name}: no ACC")
        return None
    phone_model = ""
    if metadata:
        for key in ("phone_model", "phone", "model", "device_model"):
            v = metadata.get(key)
            if v:
                phone_model = str(v)
                break
    try:
        preds, state = predict_intervals(acc, DEFAULT_CONFIG, phone_model=phone_model)
    except Exception as exc:
        print(f"[error] {name}: detector crashed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None
    if not state:
        return None

    gt_rides: list[dict] = []
    if gt_df is not None and not gt_df.empty:
        t0_ms = float(state["t0_ms"])
        for _, row in gt_df.iterrows():
            if row.get("type") not in ("up", "down"):
                continue
            # Optional signalClearRecording column — skip unclear rides.
            clear = row.get("signalClearRecording", True)
            if clear is False:
                continue
            gt_rides.append({
                "type":      row["type"],
                "t_start_s": (float(row["start_ms"]) - t0_ms) / 1000.0,
                "t_end_s":   (float(row["end_ms"])   - t0_ms) / 1000.0,
            })
    return {
        "name": name,
        "phone_model": phone_model,
        "state": state,
        "preds": preds,
        "gt_rides": gt_rides,
    }


# --------------------------------------------------------------------------
# Per-GT / per-pred classification — mirrors IntervalPredictionMetrics but
# keeps the per-item labels (from_intervals only returns aggregate counts).
# --------------------------------------------------------------------------
def _classify(
    gt_rides: list[dict], preds: list[dict],
    min_overlap_s: float = DEFAULT_MIN_OVERLAP_S,
    min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
) -> tuple[list[dict], list[dict]]:
    """Return (gt_labels, pred_labels). Each label carries status +
    overlapping-index list, ready for CSV export and PNG titling."""
    n_g, n_p = len(gt_rides), len(preds)
    g2p: list[list[int]] = [[] for _ in range(n_g)]
    p2g: list[list[int]] = [[] for _ in range(n_p)]
    for i, g in enumerate(gt_rides):
        for j, p in enumerate(preds):
            if _intervals_match(
                g["t_start_s"], g["t_end_s"],
                p["t_start_s"], p["t_end_s"],
                min_overlap_s, min_overlap_frac,
            ):
                g2p[i].append(j)
                p2g[j].append(i)

    gt_labels: list[dict] = []
    for i, g in enumerate(gt_rides):
        ps = g2p[i]
        if len(ps) == 0:
            status = "missed"
        elif len(ps) == 1:
            status = "clean" if len(p2g[ps[0]]) == 1 else "gt_merged"
        else:
            status = "gt_split"
        gt_labels.append({
            "gt_idx": i, "status": status,
            "type": g["type"],
            "t_start_s": g["t_start_s"], "t_end_s": g["t_end_s"],
            "pred_idxs": ps,
            # verdict_* filled by _attach_verdicts()
        })
    pred_labels: list[dict] = []
    for j, p in enumerate(preds):
        gs = p2g[j]
        if len(gs) == 0:
            status = "fp"
        elif len(gs) == 1:
            status = "clean" if len(g2p[gs[0]]) == 1 else "pred_split_part"
        else:
            status = "pred_merged"
        pred_labels.append({
            "pred_idx": j, "status": status,
            "ride_type": p.get("ride_type"),
            "t_start_s": p["t_start_s"], "t_end_s": p["t_end_s"],
            "gt_idxs": gs,
            "joint_r2_mean": float(p.get("joint_r2_mean", float("nan"))),
            "heatmap_energy": float(p.get("heatmap_energy", float("nan"))),
        })
    return gt_labels, pred_labels


# --------------------------------------------------------------------------
# Diagnostic PNG — one figure per non-clean GT.
# --------------------------------------------------------------------------
def _summarize_verdict(state: dict, t_lo: float, t_hi: float, ride_type: str) -> dict:
    """Extract pair-fit diagnostic fields per GT for the CSV."""
    try:
        diag = diagnose_window(state, t_lo, t_hi, ride_type=ride_type)
    except Exception:
        return {}
    out: dict = {
        "pos_r2": float("nan"), "pos_A": float("nan"),
        "neg_r2": float("nan"), "neg_A": float("nan"),
        "pair_joint_r2": float("nan"), "pair_A_abs": float("nan"),
        "pair_heatmap_energy": float("nan"), "pair_gap_s": float("nan"),
        "pair_W": float("nan"), "pair_frac_flat": float("nan"),
        "pair_reject_flags": "",
    }
    pos = diag.get("pos_peak")
    neg = diag.get("neg_peak")
    if pos is not None:
        _, A, r2 = pos
        out["pos_A"] = float(A); out["pos_r2"] = float(r2)
    if neg is not None:
        _, A, r2 = neg
        out["neg_A"] = float(A); out["neg_r2"] = float(r2)
    pair = diag.get("pair")
    if pair is not None:
        out["pair_joint_r2"] = pair["joint_r2_mean"]
        out["pair_A_abs"] = pair["A_abs"]
        out["pair_heatmap_energy"] = pair["heatmap_energy"]
        out["pair_gap_s"] = pair["gap_s"]
        out["pair_W"] = pair["W"]
        out["pair_frac_flat"] = pair["frac_flat"]
        out["pair_reject_flags"] = ";".join(pair.get("reject_flags", []))
    return out


_PEAK_COLOR = {
    "accepted":         "#2ca02c",
    "unpaired (greedy)":"#ff7f0e",
    "same-sign NMS":    "#9467bd",
    "NMS (local)":      "#e377c2",
    "lost to opp sign": "#7f7f7f",
    "R²<thr":           "#bfbfbf",
    "|A|<thr":          "#d9d9d9",
}


def _window_mask(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (t >= lo) & (t <= hi)


def _draw_mistake_png(
    out_path: Path,
    exp_name: str,
    gt_label: dict,
    gt_rides: list[dict],
    pred_labels: list[dict],
    preds: list[dict],
    state: dict,
    pad_s: float = 30.0,
) -> None:
    """Static 3-row figure with subplots mimicking the editor view:
      Row 1  (full width): a_vert + a_smooth + GT shade + fitted trapezoids
      Row 2  (full width): signed-R² panel (best_pos_r2 / best_neg_r2)
      Row 3: left = GT-vs-pred strip (full experiment); middle = heatmap at
             best +sample in window; right = heatmap at best −sample.
    """
    t = state["t"]
    a_vert = state["a_vert"]
    a_smooth = state["a_smooth"]
    best_pos_r2 = state["best_pos_r2"]
    best_neg_r2 = state["best_neg_r2"]
    grid_w_s = state["grid_w_s"]
    grid_f = state["grid_f"]
    cfg: DetectConfig = state["config"]

    g_lo = gt_label["t_start_s"]
    g_hi = gt_label["t_end_s"]
    w_lo = max(float(t[0]), g_lo - pad_s)
    w_hi = min(float(t[-1]), g_hi + pad_s)

    # Run the diagnose-window primitive on this GT — gives the verdict text
    # and tells us which ± samples to put in the heatmaps.
    diag = diagnose_window(state, g_lo, g_hi, ride_type=gt_label["type"])

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1.8, 1.1, 1.3], width_ratios=[2.2, 1.0, 1.0],
        hspace=0.45, wspace=0.35,
    )
    ax_sig = fig.add_subplot(gs[0, :])
    ax_r2 = fig.add_subplot(gs[1, :], sharex=ax_sig)
    ax_tl = fig.add_subplot(gs[2, 0])
    ax_hp = fig.add_subplot(gs[2, 1])
    ax_hn = fig.add_subplot(gs[2, 2])

    # ---- Row 1: signal
    m = _window_mask(t, w_lo, w_hi)
    ax_sig.plot(t[m], a_vert[m], color="#555", lw=0.8, label="a_vert")
    ax_sig.plot(t[m], a_smooth[m], color="tab:orange", lw=1.4, label="a_smooth")
    ax_sig.axhline(0, color="k", lw=0.3)
    ax_sig.axvspan(
        g_lo, g_hi,
        color=("tab:green" if gt_label["type"] == "up" else "tab:red"),
        alpha=0.12, label=f"GT {gt_label['type']}",
    )
    # Overlay fitted trapezoids for predictions whose lobes land in window.
    for p in preds:
        for lobe in (p["lobe1"], p["lobe2"]):
            tc = float(lobe["t_c"])
            if tc < w_lo - 3 or tc > w_hi + 3:
                continue
            W = float(lobe["half_width_s"])
            f = float(lobe["frac_flat"])
            Ap = float(lobe["a_peak"])
            t_local_mask = _window_mask(t, tc - W, tc + W)
            if t_local_mask.any():
                tpl = Ap * trapezoid_kernel(t[t_local_mask], tc, W, f)
                ax_sig.plot(t[t_local_mask], tpl, color="tab:red", lw=1.4, alpha=0.85)
    # Predicted ride envelopes as vertical dashed lines (for predictions
    # that overlap this GT).
    for p_lab in pred_labels:
        if gt_label["gt_idx"] not in p_lab["gt_idxs"] and not (
            p_lab["t_end_s"] >= w_lo and p_lab["t_start_s"] <= w_hi
        ):
            continue
        ax_sig.axvline(p_lab["t_start_s"], color="tab:red", ls="--", lw=0.7, alpha=0.6)
        ax_sig.axvline(p_lab["t_end_s"], color="tab:red", ls="--", lw=0.7, alpha=0.6)

    # Peak scatter — walk final_peaks in window, colour-code via classify_peak.
    final_peaks = state.get("final_peaks", [])
    initial_peaks = state.get("initial_peaks", [])
    used_labels: set[str] = set()
    all_candidate_ix = set(initial_peaks) | set(final_peaks)
    for i in all_candidate_ix:
        if t[i] < w_lo or t[i] > w_hi:
            continue
        sign_val = int(np.sign(state["best_A"][i]))
        if sign_val == 0:
            continue
        status = classify_peak(state, i, sign_val, preds, cfg)
        color = _PEAK_COLOR.get(status, "gray")
        A_plot = state["best_pos_A"][i] if sign_val > 0 else state["best_neg_A"][i]
        label = status if status not in used_labels else None
        used_labels.add(status)
        ax_sig.scatter(
            [t[i]], [A_plot], s=55, color=color, edgecolor="black", lw=0.4,
            zorder=5, label=label,
        )
    ax_sig.set_ylabel("a (m/s²)")
    ax_sig.set_xlim(w_lo, w_hi)
    ax_sig.grid(alpha=0.25)
    ax_sig.legend(loc="upper right", fontsize=8, ncol=2)

    # ---- Row 2: signed-R²
    ax_r2.plot(t[m], np.clip(best_pos_r2[m], -0.2, 1.05), color="tab:blue", lw=0.9,
               label="best_pos_r2")
    ax_r2.plot(t[m], np.clip(best_neg_r2[m], -0.2, 1.05), color="tab:red",  lw=0.9,
               label="best_neg_r2")
    ax_r2.axhline(cfg.r2_peak_thresh, color="gray", ls="--", lw=0.7,
                  label=f"r2_peak_thresh={cfg.r2_peak_thresh:.2f}")
    ax_r2.axhline(cfg.joint_r2_thresh, color="purple", ls=":", lw=0.7,
                  label=f"joint_r2_thresh={cfg.joint_r2_thresh:.2f}")
    ax_r2.axvspan(g_lo, g_hi, color="green" if gt_label["type"] == "up" else "red",
                  alpha=0.08)
    # Scatter the same peaks on the R² line.
    for i in all_candidate_ix:
        if t[i] < w_lo or t[i] > w_hi:
            continue
        sign_val = int(np.sign(state["best_A"][i]))
        if sign_val == 0:
            continue
        status = classify_peak(state, i, sign_val, preds, cfg)
        color = _PEAK_COLOR.get(status, "gray")
        r2_v = best_pos_r2[i] if sign_val > 0 else best_neg_r2[i]
        ax_r2.scatter(
            [t[i]], [np.clip(r2_v, -0.2, 1.05)], s=40, color=color,
            edgecolor="black", lw=0.4, zorder=5,
        )
    ax_r2.set_ylabel("signed R²")
    ax_r2.set_ylim(-0.2, 1.05)
    ax_r2.grid(alpha=0.25)
    ax_r2.legend(loc="upper right", fontsize=7, ncol=2)

    # ---- Row 3 left: GT-vs-pred timeline (full experiment)
    t_full_lo = float(t[0])
    t_full_hi = float(t[-1])
    for g in gt_rides:
        color = "tab:green" if g["type"] == "up" else "tab:red"
        ax_tl.barh(
            0.7, g["t_end_s"] - g["t_start_s"], left=g["t_start_s"],
            height=0.35, color=color, alpha=0.45,
        )
    ax_tl.barh(
        0.7, g_hi - g_lo, left=g_lo, height=0.35,
        color="black", alpha=0.25, hatch="///",
    )
    for p_lab in pred_labels:
        c = {
            "clean":           "tab:green",
            "pred_merged":     "tab:orange",
            "pred_split_part": "tab:purple",
            "fp":              "tab:gray",
        }.get(p_lab["status"], "tab:red")
        ax_tl.barh(
            0.2, p_lab["t_end_s"] - p_lab["t_start_s"], left=p_lab["t_start_s"],
            height=0.35, color=c, alpha=0.65,
        )
    ax_tl.set_xlim(t_full_lo, t_full_hi)
    ax_tl.set_ylim(0.0, 1.1)
    ax_tl.set_yticks([0.2, 0.7], labels=["pred", "gt"])
    ax_tl.set_xlabel("t (s)")
    ax_tl.grid(alpha=0.2, axis="x")
    ax_tl.set_title(f"whole experiment  (this GT highlighted)", fontsize=9)

    # ---- Row 3 middle / right: heatmaps at best + / − sample in GT window.
    pos = diag.get("pos_peak")
    neg = diag.get("neg_peak")
    _draw_heatmap(
        ax_hp, state["a_smooth"], t, pos, grid_w_s, grid_f, cfg, title="heatmap @ best +",
    )
    _draw_heatmap(
        ax_hn, state["a_smooth"], t, neg, grid_w_s, grid_f, cfg, title="heatmap @ best −",
    )

    # ---- title — verdict
    verdict = "\n".join(diag["verdict_lines"])
    header = (
        f"{exp_name}  |  gt#{gt_label['gt_idx']}  {gt_label['type']}  "
        f"[{gt_label['status']}]  "
        f"t=[{g_lo:.1f}, {g_hi:.1f}]  dur={g_hi - g_lo:.1f}s"
    )
    fig.suptitle(header + "\n" + verdict, fontsize=9, ha="left", x=0.01, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _draw_heatmap(
    ax, a_smooth: np.ndarray, t: np.ndarray,
    peak_info: tuple[int, float, float] | None,
    grid_w_s: np.ndarray, grid_f: np.ndarray,
    cfg: DetectConfig, title: str,
) -> None:
    if peak_info is None:
        ax.text(0.5, 0.5, "no same-sign sample\nin GT window",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        return
    i_c, A_val, r2_val = peak_info
    hm = heatmap_at(a_smooth, t, i_c, grid_w_s, grid_f)
    # imshow: rows = W (y), cols = f (x). origin='lower' so small W at bottom.
    extent = [grid_f[0], grid_f[-1], grid_w_s[0], grid_w_s[-1]]
    im = ax.imshow(
        hm, origin="lower", aspect="auto", extent=extent,
        vmin=0.0, vmax=1.0, cmap="viridis",
    )
    # Mark the argmax
    if np.any(np.isfinite(hm)):
        wi, fi = np.unravel_index(np.nanargmax(hm), hm.shape)
        ax.scatter([grid_f[fi]], [grid_w_s[wi]], marker="x", color="red", s=70, lw=2.5)
    ax.set_xlabel("frac_flat")
    ax.set_ylabel("W (s)")
    ax.set_title(
        f"{title}\nt={t[i_c]:.1f}s  A={A_val:+.2f}  R²={r2_val:.2f}",
        fontsize=9,
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _per_exp_summary_png(out_path: Path, per_exp: list[tuple[str, IntervalPredictionMetrics]]):
    fig, ax = plt.subplots(figsize=(13, max(5, 0.35 * len(per_exp) + 2)))
    names = [e[0] for e in per_exp]
    clean = np.array([m.clean for _, m in per_exp], dtype=float)
    missed = np.array([m.missed for _, m in per_exp], dtype=float)
    merged = np.array([m.gt_merged for _, m in per_exp], dtype=float)
    split = np.array([m.gt_split for _, m in per_exp], dtype=float)
    fp = np.array([m.fp for _, m in per_exp], dtype=float)
    y = np.arange(len(names))
    ax.barh(y, clean, color="tab:green", label="clean")
    ax.barh(y, missed, left=clean, color="tab:red", label="missed")
    ax.barh(y, merged, left=clean + missed, color="tab:orange", label="gt_merged")
    ax.barh(y, split, left=clean + missed + merged, color="tab:purple", label="gt_split")
    ax.barh(y, fp, left=clean + missed + merged + split, color="tab:gray", label="fp")
    ax.set_yticks(y, labels=[n[:50] for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def run_iteration(
    iter_dir: Path,
    experiments: list[str],
    max_mistake_pngs: int | None = None,
    description: str = "",
) -> dict:
    iter_dir.mkdir(parents=True, exist_ok=True)
    mistakes_dir = iter_dir / "mistakes"
    mistakes_dir.mkdir(exist_ok=True)

    per_exp: list[tuple[str, IntervalPredictionMetrics]] = []
    all_gt_rows: list[dict] = []
    pooled_gt: list[dict] = []
    pooled_pred: list[dict] = []

    t_start = time.time()
    print(f"Running on {len(experiments)} experiments...")
    n_pngs_written = 0
    for idx, name in enumerate(experiments):
        t_exp = time.time()
        bundle = _run_experiment(name)
        if bundle is None:
            per_exp.append((name, IntervalPredictionMetrics()))
            continue
        gt_rides = bundle["gt_rides"]
        preds = bundle["preds"]
        state = bundle["state"]
        gt_labels, pred_labels = _classify(gt_rides, preds)
        m = IntervalPredictionMetrics.from_intervals(gt_rides, preds)
        per_exp.append((name, m))
        pooled_offset = (idx + 1) * 1e9
        pooled_gt.extend(
            {**g, "t_start_s": g["t_start_s"] + pooled_offset,
             "t_end_s":   g["t_end_s"]   + pooled_offset}
            for g in gt_rides
        )
        pooled_pred.extend(
            {**p, "t_start_s": p["t_start_s"] + pooled_offset,
             "t_end_s":   p["t_end_s"]   + pooled_offset}
            for p in preds
        )
        for gl in gt_labels:
            # Record diagnose_window verdict for every GT (including clean,
            # so we can see why they passed). Catches the exact threshold
            # that rejected each missed / merged GT.
            diag_info = _summarize_verdict(
                state, gl["t_start_s"], gl["t_end_s"], gl["type"],
            )
            row = dict(gl)
            row["exp"] = name
            row.update(diag_info)
            all_gt_rows.append(row)
            if gl["status"] == "clean":
                continue
            if max_mistake_pngs is not None and n_pngs_written >= max_mistake_pngs:
                continue
            png_name = f"{name}__gt{gl['gt_idx']:02d}__{gl['status']}.png"
            try:
                _draw_mistake_png(
                    mistakes_dir / png_name,
                    exp_name=name,
                    gt_label=gl,
                    gt_rides=gt_rides,
                    pred_labels=pred_labels,
                    preds=preds,
                    state=state,
                )
                n_pngs_written += 1
            except Exception as exc:
                print(f"  [png-err] {png_name}: {type(exc).__name__}: {exc}")
        dt = time.time() - t_exp
        print(
            f"  [{idx + 1:2d}/{len(experiments)}] {name[:60]:<60} "
            f"gt={m.n_gt:3d} pr={m.n_pred:3d} "
            f"cl={m.clean:3d} mi={m.missed:3d} gm={m.gt_merged:3d} "
            f"gs={m.gt_split:3d} fp={m.fp:3d} ({dt:4.1f}s)"
        )

    total = IntervalPredictionMetrics.sum(m for _, m in per_exp)
    iou_metrics = IntervalPredictionMetrics.iou_f1(
        pooled_gt, pooled_pred, iou_threshold=0.5,
    )

    # Write per_gt.csv
    df = pd.DataFrame(all_gt_rows, columns=[
        "exp", "gt_idx", "status", "type", "t_start_s", "t_end_s", "pred_idxs",
        "pos_r2", "pos_A", "neg_r2", "neg_A",
        "pair_joint_r2", "pair_A_abs", "pair_heatmap_energy",
        "pair_gap_s", "pair_W", "pair_frac_flat", "pair_reject_flags",
    ])
    df["pred_idxs"] = df["pred_idxs"].apply(
        lambda v: ",".join(map(str, v)) if isinstance(v, list) else ""
    )
    df.to_csv(iter_dir / "per_gt.csv", index=False)

    # Write metrics.json
    metrics_payload = {
        "description": description,
        "n_experiments": len(per_exp),
        "elapsed_s": time.time() - t_start,
        "config": asdict(DEFAULT_CONFIG),
        "total": total.as_dict(),
        "iou": iou_metrics,
        "per_exp": [(n, m.as_dict()) for n, m in per_exp],
        "mistakes_total": (total.missed + total.gt_merged + total.gt_split + total.fp),
        "n_gt_total": total.n_gt,
    }
    (iter_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    # Write per_exp_summary.png
    _per_exp_summary_png(iter_dir / "per_exp_summary.png", per_exp)

    # Write notes.md stub if missing
    notes_path = iter_dir / "notes.md"
    if not notes_path.exists():
        notes_path.write_text(_notes_template(iter_dir.name, total, iou_metrics, description))

    print(
        f"\nTotal: clean={total.clean}/{total.n_gt} "
        f"mistakes={metrics_payload['mistakes_total']} "
        f"(miss={total.missed} gm={total.gt_merged} gs={total.gt_split} fp={total.fp}) "
        f"f1={total.score():.3f} iou_f1@0.5={iou_metrics['iou_f1@0.5']:.3f} "
        f"pngs={n_pngs_written} in {iter_dir}"
    )
    return metrics_payload


def _notes_template(
    slug: str, total: IntervalPredictionMetrics, iou: dict, description: str,
) -> str:
    return f"""# Iteration: {slug}

**What changed:** {description or "(TODO)"}

## Metrics

| metric | value |
|---|---|
| n_gt | {total.n_gt} |
| clean | {total.clean} |
| missed | {total.missed} |
| gt_merged | {total.gt_merged} |
| gt_split | {total.gt_split} |
| pred_merged | {total.pred_merged} |
| fp | {total.fp} |
| mistakes_total | {total.missed + total.gt_merged + total.gt_split + total.fp} |
| f1_like | {total.score():.3f} |
| iou_f1@0.5 | {iou.get("iou_f1@0.5", 0):.3f} |
| mean IoU (matched) | {iou.get("iou_mean@0.5", 0):.3f} |

## Observations

- (fill in from looking at a sample of `mistakes/*.png`)

## Next iteration hypothesis

- (TODO)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", required=True, help="iteration number, e.g. 00")
    ap.add_argument("--slug", required=True, help="short slug, e.g. baseline")
    ap.add_argument("--what", default="", help="one-line description for notes.md")
    ap.add_argument("--kind", default="all", choices=["all", "train", "test"])
    ap.add_argument("--only", default=None, help="comma-separated subset")
    ap.add_argument(
        "--max-pngs", type=int, default=None,
        help="cap on number of mistake PNGs to write (debug only)",
    )
    args = ap.parse_args()

    if args.only:
        experiments = [s.strip() for s in args.only.split(",") if s.strip()]
    else:
        experiments = list_experiments(kind=args.kind)
    iter_dir = ITER_ROOT / f"iter_{args.iter}_{args.slug}"
    run_iteration(
        iter_dir=iter_dir,
        experiments=experiments,
        max_mistake_pngs=args.max_pngs,
        description=args.what,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
