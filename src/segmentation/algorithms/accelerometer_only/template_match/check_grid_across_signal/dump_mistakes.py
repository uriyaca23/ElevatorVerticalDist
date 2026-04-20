"""Render a diagnostic figure per detector mistake across all train
experiments.

A "mistake" is any GT ride or prediction that is *not* in a clean
one-to-one match under
:class:`IntervalPredictionMetrics.from_intervals`. Each mistake gets
one PNG that mirrors the editor's detail panel: two (W, f) R² heatmaps
on the top row (at the two lobe centres), the zoomed vertical-accel
signal with the fitted trapezoid pair overlaid in the middle, and the
per-sign R² trace with colour-coded peak status at the bottom.

Output layout::

    labels/check_grid_across_signal/mistakes/
        <exp-name>/
            gt_missed_<idx>_t<start>.png
            gt_merged_<idx>_t<start>.png
            gt_split_<idx>_t<start>.png
            pred_fp_<idx>_t<start>.png
            pred_merged_<idx>_t<start>.png
            pred_split_part_<idx>_t<start>.png

Run:

    PYTHONPATH=. venv/bin/python -m src.segmentation.algorithms.\
accelerometer_only.template_match.check_grid_across_signal.dump_mistakes
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.data.loader import list_experiments, getExperimentData
from src.segmentation.algorithms.accelerometer_only.template_match.fit_elevator_parameters.common import (
    trapezoid_kernel,
)
from . import detect, pair_filter


_HERE = Path(__file__).resolve().parent
# template_match/labels/check_grid_across_signal/ — the shared root for
# every artefact this sub-package writes or reads (best config, sweep
# CSV, pair-filter iteration log, mistakes dump).
LABELS_ROOT = _HERE.parent / "labels" / "check_grid_across_signal"
OUT_ROOT = LABELS_ROOT / "mistakes"

PAD_S = 5.0
PAD_WIDE_S = 15.0


# --------------------------------------------------------------------------
# Match graph (same overlap rule as IntervalPredictionMetrics)
# --------------------------------------------------------------------------
def _overlap_s(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _intervals_match(g0, g1, p0, p1, min_overlap_s=1.0, min_overlap_frac=0.3):
    ov = _overlap_s(g0, g1, p0, p1)
    if ov <= 0.0:
        return False
    shortest = max(1e-3, min(g1 - g0, p1 - p0))
    return ov >= min_overlap_s or ov / shortest >= min_overlap_frac


def build_match_graph(gt_rides, predictions):
    gt_to_preds = [[] for _ in gt_rides]
    pred_to_gts = [[] for _ in predictions]
    for i, g in enumerate(gt_rides):
        for j, p in enumerate(predictions):
            if _intervals_match(
                g["t_start_s"], g["t_end_s"],
                p["t_start_s"], p["t_end_s"],
            ):
                gt_to_preds[i].append(j)
                pred_to_gts[j].append(i)
    return gt_to_preds, pred_to_gts


def _extract_gt_rides(gt, t0_ms):
    rides = []
    if gt is None or gt.empty:
        return rides
    for _, row in gt.iterrows():
        if row.get("type") not in ("up", "down"):
            continue
        rides.append({
            "type": row["type"],
            "t_start_s": (float(row["start_ms"]) - t0_ms) / 1000.0,
            "t_end_s":   (float(row["end_ms"])   - t0_ms) / 1000.0,
        })
    return rides


# --------------------------------------------------------------------------
# Best +/- samples inside a GT window
# --------------------------------------------------------------------------
def _best_signed_in_window(state, t_lo, t_hi):
    t = state["t"]
    mask = (t >= t_lo) & (t <= t_hi)
    if not mask.any():
        return None, None
    best_A = state["best_A"]
    best_r2 = state["best_r2"]
    idxs = np.where(mask & np.isfinite(best_r2))[0]
    if idxs.size == 0:
        return None, None
    pos_idxs = idxs[best_A[idxs] > 0]
    neg_idxs = idxs[best_A[idxs] < 0]
    pos = neg = None
    if pos_idxs.size:
        j = int(pos_idxs[np.argmax(best_A[pos_idxs])])
        pos = (j, float(best_A[j]), float(best_r2[j]))
    if neg_idxs.size:
        j = int(neg_idxs[np.argmin(best_A[neg_idxs])])
        neg = (j, float(best_A[j]), float(best_r2[j]))
    return pos, neg


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
_STATUS_COLORS = {
    detect.PEAK_STATUS_ACCEPTED:      "#27ae60",
    detect.PEAK_STATUS_UNPAIRED:      "#f39c12",
    detect.PEAK_STATUS_SAME_SIGN_NMS: "#9b59b6",
    detect.PEAK_STATUS_LOCAL_NMS:     "#8e44ad",
    detect.PEAK_STATUS_OPP_SIGN:      "#34495e",
    detect.PEAK_STATUS_LOW_R2:        "#7f8c8d",
    detect.PEAK_STATUS_LOW_A:         "#95a5a6",
}


def _draw_heatmap(ax, heat, title, grid_w_s, grid_f, mark_W=None, mark_f=None):
    im = ax.imshow(
        heat, origin="lower", aspect="auto",
        extent=(grid_f[0], grid_f[-1], grid_w_s[0], grid_w_s[-1]),
        cmap="viridis", vmin=0.0, vmax=1.0,
    )
    if mark_W is not None and mark_f is not None:
        ax.plot([mark_f], [mark_W], marker="x", color="#e74c3c",
                markersize=10, markeredgewidth=2.0)
    ax.set_xlabel("plateau $f$")
    ax.set_ylabel("half-width $W$ (s)")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)


def _draw_signed_r2(ax, state, t_lo, t_hi, predictions):
    t = state["t"]
    pos_r2 = state["best_pos_r2"]
    neg_r2 = state["best_neg_r2"]
    mask = (t >= t_lo) & (t <= t_hi)
    pos_plot = np.where(np.isfinite(pos_r2), pos_r2, np.nan)
    neg_plot = np.where(np.isfinite(neg_r2), neg_r2, np.nan)
    ax.plot(t[mask], pos_plot[mask], color="#2980b9", lw=0.9,
            label="max $R^2_{+}$")
    ax.plot(t[mask], neg_plot[mask], color="#c0392b", lw=0.9,
            label="max $R^2_{-}$")
    cfg = state.get("config", detect.DEFAULT_CONFIG)
    ax.axhline(cfg.r2_peak_thresh, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylabel("$R^2$ (per sign)")
    ax.set_xlabel("t (s, ACC-local)")
    ax.grid(True, alpha=0.25)

    seen = set()
    for sign, arr in ((+1, pos_r2), (-1, neg_r2)):
        peaks = detect.find_local_maxima(arr, t, t_lo, t_hi,
                                         min_val=0.5, min_gap_s=1.0)
        for i in peaks:
            tag = detect.classify_peak(state, i, sign, predictions)
            ax.scatter([t[i]], [arr[i]], s=36, zorder=5,
                       color=_STATUS_COLORS.get(tag, "#000"),
                       edgecolor="black", linewidth=0.4)
            seen.add(tag)
    status_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=_STATUS_COLORS[tag],
               markeredgecolor="black", markeredgewidth=0.4, markersize=6,
               label=tag)
        for tag in (
            detect.PEAK_STATUS_ACCEPTED, detect.PEAK_STATUS_UNPAIRED,
            detect.PEAK_STATUS_SAME_SIGN_NMS, detect.PEAK_STATUS_LOCAL_NMS,
            detect.PEAK_STATUS_OPP_SIGN, detect.PEAK_STATUS_LOW_R2,
            detect.PEAK_STATUS_LOW_A,
        ) if tag in seen
    ]
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(
        line_handles + status_handles,
        line_labels + [h.get_label() for h in status_handles],
        fontsize=6, loc="lower left", ncol=2, frameon=True,
    )


def render_mistake_figure(
    state, predictions, title, out_path,
    pos_peak, neg_peak, pair_W, pair_f,
    t_lo, t_hi,
    gt_span=None, pred_span=None,
    all_gt_spans=None, all_pred_spans=None,
):
    t = state["t"]
    a_vert = state["a_vert"]
    a_smooth = state["a_smooth"]
    grid_w_s = state["grid_w_s"]
    grid_f = state["grid_f"]

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 0.9, 0.7], hspace=0.7, wspace=0.28,
    )
    ax_h1 = fig.add_subplot(gs[0, 0])
    ax_h2 = fig.add_subplot(gs[0, 1])
    ax_sig = fig.add_subplot(gs[1, :])
    ax_rt = fig.add_subplot(gs[2, :])
    fig.suptitle(title, fontsize=10)

    def _heat_panel(ax, peak, sign_label):
        if peak is None:
            ax.text(0.5, 0.5, f"no {sign_label} sample in window",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888", style="italic")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{sign_label} lobe — n/a", fontsize=9)
            return
        i, A, r2 = peak
        heat = detect.heatmap_at(a_smooth, t, i, grid_w_s, grid_f)
        _draw_heatmap(
            ax, heat,
            f"{sign_label} lobe @ t={t[i]:.1f}s  A={A:+.2f}  R²={r2:.2f}",
            grid_w_s, grid_f,
            mark_W=pair_W, mark_f=pair_f,
        )

    _heat_panel(ax_h1, pos_peak, "+")
    _heat_panel(ax_h2, neg_peak, "−")

    mask = (t >= t_lo) & (t <= t_hi)
    ax_sig.plot(t[mask], a_vert[mask],   color="#2c3e50", lw=0.7, label="$a_\\mathrm{vert}$")
    ax_sig.plot(t[mask], a_smooth[mask], color="#e67e22", lw=1.1, label="smoothed")
    ax_sig.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.5)

    for g0, g1, gt_type in (all_gt_spans or []):
        if g1 < t_lo or g0 > t_hi:
            continue
        color = "#27ae60" if gt_type == "up" else "#e74c3c"
        ax_sig.axvspan(g0, g1, color=color, alpha=0.10, zorder=0)
    for p0, p1, pred_type in (all_pred_spans or []):
        if p1 < t_lo or p0 > t_hi:
            continue
        color = "#1f3a5f" if pred_type == "up" else "#7d3c98"
        ax_sig.axvspan(p0, p1, color=color, alpha=0.10, zorder=0, ymin=0.5)

    if gt_span is not None:
        g0, g1, gt_type = gt_span
        color = "#27ae60" if gt_type == "up" else "#e74c3c"
        ax_sig.axvspan(g0, g1, color=color, alpha=0.35, zorder=0,
                       label=f"GT ({gt_type})")
    if pred_span is not None:
        p0, p1, pred_type = pred_span
        color = "#1f3a5f" if pred_type == "up" else "#7d3c98"
        ax_sig.axvline(p0, color=color, lw=1.0, ls="--", alpha=0.9,
                       label=f"pred ({pred_type})")
        ax_sig.axvline(p1, color=color, lw=1.0, ls="--", alpha=0.9)

    if pos_peak is not None and neg_peak is not None \
            and pair_W is not None and pair_f is not None:
        A_abs = 0.5 * (abs(pos_peak[1]) + abs(neg_peak[1]))
        for i, sign in ((pos_peak[0], +1.0), (neg_peak[0], -1.0)):
            t_c = float(t[i])
            tt = np.linspace(t_c - pair_W, t_c + pair_W, 120)
            yy = sign * A_abs * trapezoid_kernel(tt, t_c, pair_W, pair_f)
            ax_sig.plot(tt, yy, color="#c0392b", lw=1.5)

    ax_sig.set_xlim(t_lo, t_hi)
    ax_sig.set_xlabel("t (s, ACC-local)")
    ax_sig.set_ylabel("a (m/s²)")
    ax_sig.grid(True, alpha=0.25)
    ax_sig.legend(fontsize=7, loc="upper right", ncol=2)

    _draw_signed_r2(ax_rt, state, t_lo, t_hi, predictions)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Per-exp driver
# --------------------------------------------------------------------------
def dump_for_experiment(name, cfg, out_root):
    counts = {
        "gt_missed": 0, "gt_merged": 0, "gt_split": 0,
        "pred_fp": 0, "pred_merged": 0, "pred_split_part": 0,
    }
    try:
        sensors, gt, _ = getExperimentData(name)
    except Exception as exc:
        print(f"  [error] {name}: {type(exc).__name__}: {exc}", flush=True)
        return counts
    preds, state = detect.predict_intervals(sensors.get("ACC"), cfg)
    if not state:
        return counts
    gt_rides = _extract_gt_rides(gt, state["t0_ms"])
    if not gt_rides and not preds:
        return counts
    gt_to_preds, pred_to_gts = build_match_graph(gt_rides, preds)

    all_gt_spans = [(g["t_start_s"], g["t_end_s"], g["type"]) for g in gt_rides]
    all_pred_spans = [(p["t_start_s"], p["t_end_s"], p["ride_type"]) for p in preds]

    exp_dir = out_root / name

    for i, g in enumerate(gt_rides):
        ps = gt_to_preds[i]
        if len(ps) == 1 and len(pred_to_gts[ps[0]]) == 1:
            continue  # clean
        if len(ps) == 0:
            kind, wide = "gt_missed", False
        elif len(ps) == 1:
            kind, wide = "gt_merged", True
        else:
            kind, wide = "gt_split", True
        counts[kind] += 1

        pad = PAD_WIDE_S if wide else PAD_S
        t_lo = g["t_start_s"] - pad
        t_hi = g["t_end_s"] + pad
        pos, neg = _best_signed_in_window(state, g["t_start_s"], g["t_end_s"])
        pair_W = pair_f = None
        if pos is not None and neg is not None:
            first, second, s1, s2 = (
                (pos, neg, +1.0, -1.0) if g["type"] == "up"
                else (neg, pos, -1.0, +1.0)
            )
            if first[0] < second[0]:
                res = pair_filter.joint_pair_score(
                    state["a_smooth"], state["t"],
                    first[0], second[0], s1, s2,
                    state["grid_w_s"], state["grid_f"],
                )
                if res is not None:
                    _, pair_W, pair_f, *_ = res

        title = (
            f"{name}\n{kind} — GT #{i} ({g['type']}, "
            f"t∈[{g['t_start_s']:.1f}, {g['t_end_s']:.1f}]s, "
            f"dur={g['t_end_s'] - g['t_start_s']:.1f}s)"
        )
        out_path = exp_dir / f"{kind}_{i:03d}_t{int(g['t_start_s']):06d}.png"
        render_mistake_figure(
            state, preds, title, out_path, pos, neg,
            pair_W, pair_f, t_lo, t_hi,
            gt_span=(g["t_start_s"], g["t_end_s"], g["type"]),
            all_gt_spans=all_gt_spans,
            all_pred_spans=all_pred_spans,
        )

    for j, p in enumerate(preds):
        gs = pred_to_gts[j]
        if len(gs) == 1 and len(gt_to_preds[gs[0]]) == 1:
            continue
        if len(gs) == 0:
            kind, wide = "pred_fp", False
        elif len(gs) == 1:
            kind, wide = "pred_split_part", True
        else:
            kind, wide = "pred_merged", True
        counts[kind] += 1

        pad = PAD_WIDE_S if wide else PAD_S
        t_lo = p["t_start_s"] - pad
        t_hi = p["t_end_s"] + pad
        t_c1 = float(p["lobe1"]["t_c"])
        t_c2 = float(p["lobe2"]["t_c"])
        i1 = int(np.argmin(np.abs(state["t"] - t_c1)))
        i2 = int(np.argmin(np.abs(state["t"] - t_c2)))
        s1 = np.sign(float(p["lobe1"]["a_peak"]))
        pos_peak = (i1, float(p["lobe1"]["a_peak"]), float(p["lobe1"]["r2_local"]))
        neg_peak = (i2, float(p["lobe2"]["a_peak"]), float(p["lobe2"]["r2_local"]))
        if s1 < 0:
            pos_peak, neg_peak = neg_peak, pos_peak
        pair_W = float(p["lobe1"]["half_width_s"])
        pair_f = float(p["lobe1"]["frac_flat"])

        title = (
            f"{name}\n{kind} — pred #{j} ({p['ride_type']}, "
            f"t∈[{p['t_start_s']:.1f}, {p['t_end_s']:.1f}]s, "
            f"joint R²={p['joint_r2_mean']:.3f})"
        )
        out_path = exp_dir / f"{kind}_{j:03d}_t{int(p['t_start_s']):06d}.png"
        render_mistake_figure(
            state, preds, title, out_path, pos_peak, neg_peak,
            pair_W, pair_f, t_lo, t_hi,
            pred_span=(p["t_start_s"], p["t_end_s"], p["ride_type"]),
            all_gt_spans=all_gt_spans,
            all_pred_spans=all_pred_spans,
        )

    return counts


def main():
    cfg_json = LABELS_ROOT / "best_detect_config.json"
    if cfg_json.exists():
        cfg = detect.DetectConfig(**json.loads(cfg_json.read_text())["config"])
    else:
        cfg = detect.DEFAULT_CONFIG
    print(f"using config: {cfg}", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    names = list_experiments(kind="train")
    print(f"rendering mistakes for {len(names)} experiments "
          f"→ {OUT_ROOT}", flush=True)

    total = {k: 0 for k in (
        "gt_missed", "gt_merged", "gt_split",
        "pred_fp", "pred_merged", "pred_split_part",
    )}
    t0 = time.time()
    for n in names:
        counts = dump_for_experiment(n, cfg, OUT_ROOT)
        for k, v in counts.items():
            total[k] += v
        print(f"  [{time.time() - t0:6.1f}s] {n} — "
              + ", ".join(f"{k}={v}" for k, v in counts.items() if v),
              flush=True)

    grand = sum(total.values())
    print(f"\ntotal mistakes rendered: {grand}")
    for k, v in total.items():
        print(f"  {k}: {v}")

    lines = [
        "# Detector mistakes — diagnostic dump",
        "",
        f"Rendered at config = `{cfg}` over {len(names)} train experiments.",
        "",
        "Each subfolder is one experiment. Each PNG is one mistake. File "
        "name format: ``<kind>_<index>_t<start-second>.png``. Kinds:",
        "",
        "| kind | meaning |",
        "|---|---|",
        "| `gt_missed` | GT ride not covered by any prediction. |",
        "| `gt_merged` | GT ride is one of several covered by a single pred. |",
        "| `gt_split` | GT ride covered by ≥2 predictions. |",
        "| `pred_fp` | Pred lands on outside (no overlapping GT). |",
        "| `pred_merged` | One pred swallows several GTs. |",
        "| `pred_split_part` | Pred is one of several sharing a single GT. |",
        "",
        "## Layout per figure",
        "",
        "Three rows per PNG (mirror the editor's detail panel):",
        "",
        "1. **(W, f) R² heatmaps at the two lobe centres.** "
        "Positive lobe on the left, negative on the right. The red × "
        "marks the joint-fit template (when both lobes exist); for "
        "GT-side mistakes that's the shared-shape fit over the best "
        "± samples inside the GT window.",
        "2. **Signal zoom.** `a_vert` (dark blue) and the smoothed "
        "trace (orange) over the mistake plus context pad. All GT "
        "spans in view are faintly shaded; the focal GT or pred is "
        "highlighted. Fitted trapezoid pair drawn in red when available.",
        "3. **Signed-R² trace.** The editor's colour-coded peak "
        "status panel: green = accepted, orange = unpaired (greedy), "
        "purple / darker purple = NMS-suppressed, slate = lost to the "
        "opposite sign, grey shades = below-threshold.",
        "",
        "## Totals",
        "",
        "| kind | count |",
        "|---|---|",
    ]
    for k, v in total.items():
        lines.append(f"| `{k}` | {v} |")
    lines += [f"| **total** | **{grand}** |", ""]
    (OUT_ROOT / "README.md").write_text("\n".join(lines))
    print(f"\nwrote {OUT_ROOT / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
