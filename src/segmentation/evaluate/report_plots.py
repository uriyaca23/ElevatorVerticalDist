"""Extra plot helpers for the LaTeX-bound segmentation evaluation report.

Kept separate from :mod:`plots` so the live evaluator stays minimal and
this module owns all of the multi-panel / scatter / per-experiment work
the report bundles up.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.segmentation.algorithms.metrics import IntervalPredictionMetrics
from src.physics import calculate_velocity_from_accelerometer


# Shared color palette so every figure in the report uses the same legend.
STATUS_COLORS: dict[str, str] = {
    "clean":       "#27ae60",
    "missed":      "#e74c3c",
    "fp":          "#7f8c8d",
    "gt_merged":   "#e67e22",
    "pred_merged": "#e67e22",
    "gt_split":    "#8e44ad",
    "pred_split_part": "#8e44ad",
}
TYPE_COLORS: dict[str, str] = {"up": "#27ae60", "down": "#e74c3c"}


# --------------------------------------------------------------------------
# CDF / PDF
# --------------------------------------------------------------------------
def _cdf_axes(ax, values: Sequence[float], xlabel: str) -> None:
    if len(values):
        xs = np.sort(np.asarray(values, dtype=float))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, lw=1.5, color="tab:blue")
        ax.axhline(0.5, color="gray", lw=0.5, ls="--")
        ax.set_xlim(xs.min(), xs.max())
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("empirical CDF")
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)


def _pdf_axes(
    ax, values: Sequence[float], xlabel: str, bins: int = 30,
) -> None:
    if len(values):
        arr = np.asarray(values, dtype=float)
        ax.hist(
            arr, bins=bins, color="tab:blue", alpha=0.55,
            edgecolor="white", density=True,
        )
        # Overlay a Gaussian KDE when we have enough samples.
        if arr.size >= 5 and np.std(arr) > 0:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(arr)
            xs = np.linspace(arr.min(), arr.max(), 200)
            ax.plot(xs, kde(xs), color="tab:red", lw=1.4)
        ax.axvline(float(np.median(arr)), color="black",
                   lw=0.8, ls="--", alpha=0.6)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)


def cdf_pdf_pair(
    values: Sequence[float], title: str, xlabel: str, out_path: Path,
    bins: int = 30,
) -> None:
    """Side-by-side CDF (left) + histogram + KDE (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _cdf_axes(axes[0], values, xlabel)
    _pdf_axes(axes[1], values, xlabel, bins=bins)
    fig.suptitle(f"{title}  (n={len(values)})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Scatter plots
# --------------------------------------------------------------------------
def iou_vs_duration_scatter(
    matched_pairs: list[dict],
    out_path: Path,
    iou_threshold: float = 0.5,
) -> None:
    """IoU of each matched pair vs GT ride duration (s)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not matched_pairs:
        ax.text(0.5, 0.5, "no matched pairs", ha="center", va="center",
                transform=ax.transAxes)
    else:
        durations = np.array(
            [p["gt_end_s"] - p["gt_start_s"] for p in matched_pairs],
            dtype=float,
        )
        ious = np.array([p["iou"] for p in matched_pairs], dtype=float)
        ax.scatter(durations, ious, s=22, alpha=0.55,
                   color="tab:blue", edgecolor="none")
        ax.axhline(iou_threshold, color="tab:red", lw=0.8, ls="--",
                   label=f"IoU = {iou_threshold:.2f}")
        ax.set_xlim(0, max(1.0, float(durations.max()) * 1.05))
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("GT ride duration (s)")
    ax.set_ylabel("IoU")
    ax.set_title("Edge quality vs ride duration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def pred_vs_gt_duration_scatter(
    matched_pairs: list[dict], out_path: Path,
) -> None:
    """Predicted-duration vs GT-duration scatter, with y=x reference."""
    fig, ax = plt.subplots(figsize=(6, 5))
    if not matched_pairs:
        ax.text(0.5, 0.5, "no matched pairs", ha="center", va="center",
                transform=ax.transAxes)
    else:
        gd = np.array(
            [p["gt_end_s"] - p["gt_start_s"] for p in matched_pairs],
            dtype=float,
        )
        pd_ = np.array(
            [p["pred_end_s"] - p["pred_start_s"] for p in matched_pairs],
            dtype=float,
        )
        ious = np.array([p["iou"] for p in matched_pairs], dtype=float)
        sc = ax.scatter(gd, pd_, c=ious, cmap="viridis", s=24,
                        alpha=0.75, edgecolor="none", vmin=0.0, vmax=1.0)
        m = max(float(gd.max()), float(pd_.max())) * 1.05
        ax.plot([0, m], [0, m], "k--", lw=0.8, alpha=0.6, label="$y=x$")
        ax.set_xlim(0, m)
        ax.set_ylim(0, m)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("IoU")
        ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("GT duration (s)")
    ax.set_ylabel("predicted duration (s)")
    ax.set_title("Predicted vs ground-truth ride duration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Failure-mode bars
# --------------------------------------------------------------------------
_LABELS = ["clean", "missed", "fp", "pred_merged", "gt_split"]


def _bar_values(m: IntervalPredictionMetrics) -> list[int]:
    return [m.clean, m.missed, m.fp, m.pred_merged, m.gt_split]


def failure_modes_split_bar(
    train_total: IntervalPredictionMetrics,
    test_total:  IntervalPredictionMetrics,
    out_path: Path,
) -> None:
    """Grouped bar chart: train vs test counts per failure mode."""
    x = np.arange(len(_LABELS))
    w = 0.36
    train_vals = _bar_values(train_total)
    test_vals  = _bar_values(test_total)
    colors = [STATUS_COLORS[lab] for lab in _LABELS]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_tr = ax.bar(x - w / 2, train_vals, width=w, label="train",
                     color=colors, edgecolor="black", linewidth=0.5)
    bars_te = ax.bar(x + w / 2, test_vals,  width=w, label="test",
                     color=colors, edgecolor="black", linewidth=0.5,
                     alpha=0.55, hatch="//")
    for bar, v in list(zip(bars_tr, train_vals)) + list(zip(bars_te, test_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(_LABELS)
    ax.set_ylabel("count")
    ax.set_title(
        f"Failure modes by split  "
        f"(train: {train_total.n_gt} GT / {train_total.n_pred} pred; "
        f"test: {test_total.n_gt} GT / {test_total.n_pred} pred)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def per_experiment_failure_bar(
    per_exp: list[tuple[str, IntervalPredictionMetrics]],
    out_path: Path,
    label_short: dict[str, str] | None = None,
) -> None:
    """Stacked-bar chart of failure modes for every experiment."""
    if not per_exp:
        return
    # Sort experiments by descending GT count so the visual reading order
    # foregrounds the busy buildings.
    items = sorted(per_exp, key=lambda kv: kv[1].n_gt, reverse=True)
    names = [label_short[n] if label_short and n in label_short else n
             for n, _ in items]
    clean = np.array([m.clean       for _, m in items])
    missed = np.array([m.missed     for _, m in items])
    fps    = np.array([m.fp         for _, m in items])
    merged = np.array([m.pred_merged for _, m in items])
    split  = np.array([m.gt_split   for _, m in items])

    x = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(items)), 5))
    bottom = np.zeros(len(items))
    for arr, label in [
        (clean,  "clean"),
        (missed, "missed"),
        (fps,    "fp"),
        (merged, "pred_merged"),
        (split,  "gt_split"),
    ]:
        ax.bar(x, arr, bottom=bottom,
               color=STATUS_COLORS[label], edgecolor="white",
               linewidth=0.4, label=label)
        bottom = bottom + arr
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("count (rides)")
    ax.set_title(
        f"Failure mode distribution per experiment "
        f"(n={len(items)} experiments)"
    )
    ax.legend(loc="upper right", fontsize=9, ncol=5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def phone_breakdown_bar(
    per_exp: list[tuple[str, IntervalPredictionMetrics]],
    phone_for_exp: dict[str, str],
    out_path: Path,
) -> None:
    """Aggregate failure-mode counts grouped by phone model."""
    by_phone: "OrderedDict[str, IntervalPredictionMetrics]" = OrderedDict()
    for name, m in per_exp:
        phone = phone_for_exp.get(name, "unknown")
        by_phone[phone] = by_phone.get(phone, IntervalPredictionMetrics()) + m
    if not by_phone:
        return

    phones = list(by_phone.keys())
    x = np.arange(len(phones))
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(phones) + 4), 5))
    bottom = np.zeros(len(phones))
    for label in _LABELS:
        vals = np.array([
            getattr(by_phone[p],
                    "pred_merged" if label == "pred_merged" else label)
            for p in phones
        ])
        ax.bar(x, vals, bottom=bottom,
               color=STATUS_COLORS[label], edgecolor="white",
               linewidth=0.4, label=label)
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels(phones, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("count (rides)")
    ax.set_title("Failure mode distribution by phone model")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Per-experiment timeline (signal + GT + predictions)
# --------------------------------------------------------------------------
def _classify_match(
    gt_rides: list[dict], preds: list[dict],
) -> tuple[list[str], list[str]]:
    """Return (gt_status_per_idx, pred_status_per_idx).

    Uses the same overlap rule as :class:`IntervalPredictionMetrics`
    (>=1 s OR >=30 % of the shorter interval), keeping per-ride labels
    aligned with the aggregate failure-mode counts.
    """
    from src.segmentation.algorithms.metrics.metrics import (
        DEFAULT_MIN_OVERLAP_FRAC, DEFAULT_MIN_OVERLAP_S, _intervals_match,
    )
    gt_to: list[list[int]] = [[] for _ in gt_rides]
    pr_to: list[list[int]] = [[] for _ in preds]
    for i, g in enumerate(gt_rides):
        for j, p in enumerate(preds):
            if _intervals_match(
                g["t_start_s"], g["t_end_s"],
                p["t_start_s"], p["t_end_s"],
                DEFAULT_MIN_OVERLAP_S, DEFAULT_MIN_OVERLAP_FRAC,
            ):
                gt_to[i].append(j)
                pr_to[j].append(i)

    gt_status = []
    for ps in gt_to:
        if len(ps) == 0:
            gt_status.append("missed")
        elif len(ps) == 1:
            if len(pr_to[ps[0]]) == 1:
                gt_status.append("clean")
            else:
                gt_status.append("gt_merged")
        else:
            gt_status.append("gt_split")

    pred_status = []
    for gs in pr_to:
        if len(gs) == 0:
            pred_status.append("fp")
        elif len(gs) == 1:
            if len(gt_to[gs[0]]) == 1:
                pred_status.append("clean")
            else:
                pred_status.append("pred_split_part")
        else:
            pred_status.append("pred_merged")
    return gt_status, pred_status


def per_experiment_timeline(
    name: str,
    acc: pd.DataFrame,
    gt_rides: list[dict],
    preds: list[dict],
    out_path: Path,
) -> None:
    """Acc magnitude + smoothed velocity, with GT shading and pred bands.

    GT intervals are shaded behind the trace using the same colour map
    as :func:`src.plotting.experiment_overview.plot_experiment_overview`.
    Predicted intervals are drawn as coloured bands above the upper axes,
    coloured by their match status (clean / fp / pred_merged / split).
    Each clean match is annotated with its IoU.
    """
    if acc.empty:
        return
    ts = acc["timestamp_ms"].to_numpy(dtype=float)
    t0 = float(ts[0])
    t = (ts - t0) / 1000.0
    ax_a = acc["x"].to_numpy(dtype=float)
    ay_a = acc["y"].to_numpy(dtype=float)
    az_a = acc["z"].to_numpy(dtype=float)
    mag = np.sqrt(ax_a * ax_a + ay_a * ay_a + az_a * az_a)
    fs_hz = 1000.0 / max(1.0, float(np.median(np.diff(ts)))) if len(ts) >= 2 else 100.0
    v = calculate_velocity_from_accelerometer(ax_a, ay_a, az_a, fs_hz)

    gt_status, pred_status = _classify_match(gt_rides, preds)

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    # GT shading (up = green, down = red)
    for g in gt_rides:
        c = TYPE_COLORS.get(str(g.get("type", "")), "#bdc3c7")
        for ax in axes:
            ax.axvspan(g["t_start_s"], g["t_end_s"], color=c, alpha=0.18,
                       zorder=0)

    # Predicted-interval bands above the top axes (in axis-fraction space).
    top_ax = axes[0]
    y_band_lo, y_band_hi = 1.02, 1.08
    trans = top_ax.get_xaxis_transform()
    for j, p in enumerate(preds):
        st = pred_status[j]
        c = STATUS_COLORS.get(st, "#34495e")
        rect = mpatches.Rectangle(
            (p["t_start_s"], y_band_lo),
            p["t_end_s"] - p["t_start_s"],
            y_band_hi - y_band_lo,
            transform=trans, color=c, alpha=0.85,
            clip_on=False, linewidth=0.0,
        )
        top_ax.add_patch(rect)

    # Annotate clean matches with their IoU.
    paired_idx_by_gt = {}
    for j, gs in enumerate([
        [i for i, g in enumerate(gt_rides)
         if max(0.0, min(p["t_end_s"], g["t_end_s"])
                - max(p["t_start_s"], g["t_start_s"])) > 0]
        for p in preds
    ]):
        for i in gs:
            paired_idx_by_gt.setdefault(i, []).append(j)

    for i, g in enumerate(gt_rides):
        if gt_status[i] != "clean" or i not in paired_idx_by_gt:
            continue
        # Among overlapping preds, pick the one with the highest IoU
        best_iou = 0.0
        best_p = None
        for j in paired_idx_by_gt[i]:
            p = preds[j]
            inter = max(0.0, min(p["t_end_s"], g["t_end_s"])
                            - max(p["t_start_s"], g["t_start_s"]))
            union = max(p["t_end_s"], g["t_end_s"]) \
                    - min(p["t_start_s"], g["t_start_s"])
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_p = p
        if best_p is None:
            continue
        x_text = 0.5 * (best_p["t_start_s"] + best_p["t_end_s"])
        top_ax.text(x_text, 1.10, f"IoU={best_iou:.2f}",
                    transform=trans, ha="center", va="bottom",
                    fontsize=7, color="#2c3e50", clip_on=False)

    # Top panel: |a|
    top_ax.plot(t, mag, linewidth=0.5, color="#2c3e50", alpha=0.9)
    top_ax.axhline(9.81, color="gray", linewidth=0.4, linestyle="--",
                   alpha=0.5)
    top_ax.set_ylabel("|a| (m/s²)")
    top_ax.set_ylim(
        max(0.0, float(np.percentile(mag, 0.5)) - 1.0),
        float(np.percentile(mag, 99.5)) + 1.0,
    )
    top_ax.grid(True, alpha=0.25)

    # Bottom panel: vertical velocity
    axes[1].plot(t, v, linewidth=0.8, color="#2980b9", alpha=0.95)
    axes[1].axhline(0.0, color="gray", linewidth=0.4, alpha=0.5)
    axes[1].set_ylabel("vz (m/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.25)

    handles = [
        mpatches.Patch(color=TYPE_COLORS["up"],   alpha=0.45, label="GT up"),
        mpatches.Patch(color=TYPE_COLORS["down"], alpha=0.45, label="GT down"),
        mpatches.Patch(color=STATUS_COLORS["clean"],       label="pred: clean"),
        mpatches.Patch(color=STATUS_COLORS["fp"],          label="pred: false positive"),
        mpatches.Patch(color=STATUS_COLORS["pred_merged"], label="pred: merged/split"),
    ]
    top_ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=5)

    n_clean = sum(1 for s in gt_status if s == "clean")
    n_miss  = sum(1 for s in gt_status if s == "missed")
    n_fp    = sum(1 for s in pred_status if s == "fp")
    fig.suptitle(
        f"{name} — GT vs predicted intervals  "
        f"(GT={len(gt_rides)}, pred={len(preds)}, clean={n_clean}, "
        f"missed={n_miss}, fp={n_fp})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Constraint-justification plots (driven by per_gt.csv diagnostics)
# --------------------------------------------------------------------------
_HIST_STATUS_ORDER = ["clean", "missed", "gt_merged", "gt_split"]


def score_hist_by_status(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 30,
    log_y: bool = True,
) -> None:
    """Stacked histogram of ``score_col`` separated by GT status.

    Visualises why the deployed threshold is a good cut-point: when the
    clean-status density mass sits well above the threshold and the
    missed-status density mass sits well below it, the constraint is
    earning its keep.
    """
    df_v = df[df[score_col].notna()]
    if df_v.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    arrs, labels, colors = [], [], []
    for status in _HIST_STATUS_ORDER:
        mask = df_v["status"] == status
        if mask.any():
            arrs.append(df_v.loc[mask, score_col].to_numpy(dtype=float))
            labels.append(f"{status} (n={int(mask.sum())})")
            colors.append(STATUS_COLORS[status])

    lo = float(min(a.min() for a in arrs))
    hi = float(max(a.max() for a in arrs))
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist(arrs, bins=edges, stacked=True, color=colors, label=labels,
            edgecolor="white", linewidth=0.3)
    ax.axvline(threshold, color="black", lw=1.4, ls="--",
               label=f"threshold = {threshold:g}")
    if log_y:
        ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count" + (" (symlog)" if log_y else ""))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def constraint_2d_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_threshold: float,
    y_threshold: float,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Scatter of two paired scores with threshold lines drawing the
    accept rectangle.

    Each GT is one point coloured by status, so the spatial separation
    between clean (above-and-right of the elbow) and missed (below or
    left) tells you whether the joint constraint is doing real work.
    """
    df_v = df[df[x_col].notna() & df[y_col].notna()]
    if df_v.empty:
        return

    # Wider canvas leaves room for an external legend (right side) so
    # the legend never overlaps the scatter cloud.
    fig, ax = plt.subplots(figsize=(9, 5))
    for status in _HIST_STATUS_ORDER:
        mask = df_v["status"] == status
        if not mask.any():
            continue
        ax.scatter(
            df_v.loc[mask, x_col], df_v.loc[mask, y_col],
            s=22, alpha=0.65, color=STATUS_COLORS[status],
            edgecolor="none",
            label=f"{status} (n={int(mask.sum())})",
        )
    ax.axvline(x_threshold, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.axhline(y_threshold, color="black", lw=1.0, ls="--", alpha=0.7)

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    ax.fill_betweenx([y_threshold, yhi], x_threshold, xhi,
                     color="#27ae60", alpha=0.05, zorder=0)
    ax.text(
        xhi - 0.02 * (xhi - xlo), yhi - 0.05 * (yhi - ylo),
        "accept", ha="right", va="top", fontsize=9, color="#1e8449",
        alpha=0.8,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Park the legend outside the right edge so it can never occlude
    # the data cloud — the missed mass extends across most of the lower
    # half of the plot.
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=8, frameon=True, borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def reject_reason_bar(
    df: pd.DataFrame, out_path: Path, top_n: int = 8,
) -> None:
    """Bar chart of which pair-filter gate fired most often, on the
    subset of non-clean GTs that did produce a candidate pair.
    """
    sub = df[(df["status"] != "clean") & df["pair_reject_flags"].notna()]
    sub = sub[sub["pair_reject_flags"] != ""]
    if sub.empty:
        return

    # The flag string can chain multiple reasons with ";". Bucket each
    # GT by its leading reason name (ignoring the numeric value), so
    # "joint R²=0.85 < 0.90" and "joint R²=0.62 < 0.90" both count as
    # "joint R² < 0.90".
    def _bucket(s: str) -> str:
        first = s.split(";")[0].strip()
        for key in ("joint R²", "pair |A|", "heatmap energy",
                    "ride duration", "quiet middle"):
            if key in first:
                return f"{key} below floor"
        return first[:40]

    counts = sub["pair_reject_flags"].apply(_bucket).value_counts()
    counts = counts.head(top_n)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color="#e74c3c", edgecolor="white")
    for bar, v in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                str(int(v)), va="center", fontsize=9)
    ax.set_xlabel("non-clean GTs whose candidate pair fired this gate")
    ax.set_title("Which pair-filter gate caught the most rejected pairs?")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def peak_score_hist_combined(
    df: pd.DataFrame,
    pos_col: str, neg_col: str,
    threshold: float,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 30,
    take_abs: bool = False,
) -> None:
    """Histogram pooling ``pos_col`` and |``neg_col``| (or both signed)
    by status. Useful for peak R² and peak |A|, each of which is
    diagnosed twice (one per lobe sign) per GT.
    """
    rows: list[tuple[str, float]] = []
    for _, r in df.iterrows():
        for col in (pos_col, neg_col):
            v = r[col]
            if pd.isna(v):
                continue
            rows.append((r["status"], abs(float(v)) if take_abs else float(v)))
    if not rows:
        return
    sub = pd.DataFrame(rows, columns=["status", "value"])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    arrs, labels, colors = [], [], []
    for status in _HIST_STATUS_ORDER:
        mask = sub["status"] == status
        if mask.any():
            arrs.append(sub.loc[mask, "value"].to_numpy(dtype=float))
            labels.append(f"{status} (n={int(mask.sum())})")
            colors.append(STATUS_COLORS[status])
    lo = float(min(a.min() for a in arrs))
    hi = float(max(a.max() for a in arrs))
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist(arrs, bins=edges, stacked=True, color=colors, label=labels,
            edgecolor="white", linewidth=0.3)
    ax.axvline(threshold, color="black", lw=1.4, ls="--",
               label=f"threshold = {threshold:g}")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (symlog)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
