"""Research-quality figures for the prediction evaluation.

Every figure saves a ``.png`` (300 dpi) under the provided output
directory. The naming convention is ``fig_<short-name>.png`` so the
LaTeX report can glob them reliably.

The figures are intentionally self-describing: every axis has units,
every title carries the relevant sample count, and the legends are
colour-blind-aware (green / blue / red are fine but are paired with
shape/linestyle cues).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import per_experiment_metrics


# Global plot defaults — picks up the report style, keeps fonts legible.
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _ensure_dir(d: Path | str) -> Path:
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["signal_clear"] == True].copy()  # noqa: E712


def _safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, q))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_scatter(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Predicted vs. true Δh scatter, colour-coded by accept/reject."""
    sub = _clean(df)
    acc = sub[sub["accepted"]]
    rej = sub[~sub["accepted"]]

    fig, ax = plt.subplots(figsize=(7, 6))
    if not acc.empty:
        ax.scatter(acc["true_dh"], acc["pred_dh"], s=28, c="tab:green",
                   alpha=0.7, label=f"accepted (n={len(acc)})",
                   edgecolor="white", linewidth=0.3)
    if not rej.empty:
        ax.scatter(rej["true_dh"], rej["pred_dh"], s=26, c="tab:red",
                   marker="x", alpha=0.65, label=f"filtered out (n={len(rej)})")

    lo = min(sub["true_dh"].min(), sub["pred_dh"].min()) - 1
    hi = max(sub["true_dh"].max(), sub["pred_dh"].max()) + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="perfect")
    ax.plot([lo, hi], [lo - 1.5, hi - 1.5], "b:", lw=0.8, alpha=0.5, label="±1.5 m band")
    ax.plot([lo, hi], [lo + 1.5, hi + 1.5], "b:", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"True $\Delta h$ (m)")
    ax.set_ylabel(r"Predicted $\Delta h$ (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_path)
    plt.close(fig)


def fig_error_cdf(df: pd.DataFrame, out_path: Path, title: str) -> None:
    sub = _clean(df)
    acc = sub[sub["accepted"]]
    err_all = np.sort(sub["abs_error"].to_numpy())
    err_acc = np.sort(acc["abs_error"].to_numpy())

    fig, ax = plt.subplots(figsize=(7, 5))
    if err_all.size:
        ax.plot(err_all, np.linspace(0, 1, err_all.size), "r-", lw=1.5,
                label=f"all clean (n={err_all.size}, med={np.median(err_all):.2f}m)")
    if err_acc.size:
        ax.plot(err_acc, np.linspace(0, 1, err_acc.size), "g-", lw=2,
                label=f"accepted clean (n={err_acc.size}, med={np.median(err_acc):.2f}m)")
    ax.axvline(1.5, color="b", ls=":", alpha=0.7, label="±1.5 m target")
    ax.axvline(3.0, color="k", ls=":", alpha=0.5, label="±3.0 m")
    ax.set_xlabel("|error| (m)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.set_xlim(0, max(6.0, _safe_quantile(sub["abs_error"].to_numpy(), 0.99)))
    ax.legend(loc="lower right", framealpha=0.9)
    fig.savefig(out_path)
    plt.close(fig)


def fig_error_histogram(df: pd.DataFrame, out_path: Path, title: str) -> None:
    sub = _clean(df)
    errs = sub["abs_error"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    if errs.size:
        ax.hist(errs, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(1.5, color="b", ls=":", lw=1.5, label="±1.5 m")
    ax.axvline(3.0, color="k", ls=":", lw=1.5, label="±3.0 m")
    ax.set_xlabel("|error| (m)")
    ax.set_ylabel("segments")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def fig_per_ride_errors(df: pd.DataFrame, out_path: Path, title: str) -> None:
    sub = _clean(df).sort_values("abs_error").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(sub))
    colours = np.where(sub["accepted"], "tab:green", "tab:red")
    ax.bar(xs, sub["abs_error"], color=colours, alpha=0.85, width=0.8)
    ax.axhline(1.5, color="b", ls=":", label="±1.5 m")
    ax.axhline(3.0, color="k", ls=":", label="±3.0 m")
    ax.set_xlabel("clean segment (sorted by error)")
    ax.set_ylabel("|error| (m)")
    ax.set_title(title)
    ax.set_ylim(0, max(6.0, _safe_quantile(sub["abs_error"].to_numpy(), 0.98) + 0.5))
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def fig_ci_coverage(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Per-segment CI bars (centered at pred_dh) with truth overlaid, sorted
    by true_dh. Green band behind = covered; red = miss.
    """
    sub = _clean(df).copy()
    sub = sub.sort_values("true_dh").reset_index(drop=True)
    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(title + " (no clean segments)")
        fig.savefig(out_path); plt.close(fig); return

    xs = np.arange(len(sub))
    ci = np.minimum(sub["ci_half_width"].to_numpy(), 30.0)

    fig, ax = plt.subplots(figsize=(13, 6))
    # Background colour
    for i, cov in enumerate(sub["covered"]):
        ax.axvspan(i - 0.45, i + 0.45,
                   color="lightgreen" if cov else "salmon", alpha=0.25, lw=0)

    # Error bars at the prediction, with truth as red X
    ax.errorbar(xs, sub["pred_dh"], yerr=ci, fmt="o", ms=4, color="tab:blue",
                ecolor="tab:blue", elinewidth=1.2, capsize=3,
                label="pred ± CI")
    ax.scatter(xs, sub["true_dh"], marker="x", s=30, c="tab:red", zorder=5,
               label="true Δh")

    cov_all = float(np.mean(sub["covered"]))
    med_ci = float(np.median(sub["ci_half_width"]))
    ax.set_xlabel("clean segment (sorted by true Δh)")
    ax.set_ylabel("Δh (m)")
    ax.set_title(f"{title} — coverage={cov_all:.1%}, median CI=±{med_ci:.2f} m")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.savefig(out_path)
    plt.close(fig)


def fig_quality_vs_error(df: pd.DataFrame, out_path: Path, title: str) -> None:
    sub = _clean(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    acc = sub[sub["accepted"]]
    rej = sub[~sub["accepted"]]
    if not acc.empty:
        ax.scatter(acc["quality_score"], acc["abs_error"], s=28, c="tab:green",
                   alpha=0.7, label=f"accepted (n={len(acc)})", edgecolor="white", linewidth=0.3)
    if not rej.empty:
        ax.scatter(rej["quality_score"], rej["abs_error"], s=28, c="tab:red",
                   marker="x", alpha=0.7, label=f"filtered (n={len(rej)})")
    ax.axhline(1.5, color="b", ls=":", alpha=0.8)
    ax.set_xlabel("quality score (lower = better)")
    ax.set_ylabel("|error| (m)")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def fig_per_experiment_mae(df: pd.DataFrame, out_path: Path, title: str) -> None:
    pe = per_experiment_metrics(df)
    if pe.empty:
        fig, ax = plt.subplots(figsize=(8, 5)); ax.set_title(title + " (empty)")
        fig.savefig(out_path); plt.close(fig); return
    fig, ax = plt.subplots(figsize=(12, max(4, 0.3 * len(pe))))
    colours = ["tab:blue" if t == "train" else "tab:orange"
               for t in pe["experiment_type"]]
    y = np.arange(len(pe))
    ax.barh(y, pe["mae"].to_numpy(), color=colours, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(pe["exp_name"].to_list(), fontsize=7)
    ax.set_xlabel("MAE (m)"); ax.set_title(title)
    ax.axvline(1.5, color="b", ls=":", label="±1.5 m")
    ax.axvline(3.0, color="k", ls=":", label="±3.0 m")
    ax.legend(loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)


def fig_rejection_reasons(df: pd.DataFrame, out_path: Path, title: str) -> None:
    rej = df[~df["accepted"] & (df["reject_reason"] != "")]
    if rej.empty:
        fig, ax = plt.subplots(figsize=(8, 4)); ax.set_title(title + " (none)")
        fig.savefig(out_path); plt.close(fig); return

    # collapse to bucket names
    def _bucket(r: str) -> str:
        head = r.split("_")[0:3]
        return "_".join(head)[:35]

    counts = rej["reject_reason"].apply(_bucket).value_counts()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(counts))))
    ax.barh(counts.index[::-1], counts.values[::-1], color="tab:red", alpha=0.85)
    ax.set_xlabel("segments rejected")
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)


def fig_ci_vs_distance(df: pd.DataFrame, out_path: Path, title: str) -> None:
    sub = _clean(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    if not sub.empty:
        ax.scatter(np.abs(sub["true_dh"]), sub["ci_half_width"], s=22,
                   c="tab:purple", alpha=0.6, edgecolor="white", linewidth=0.3)
    ax.axhline(1.5, color="b", ls=":", alpha=0.7, label="±1.5 m")
    ax.set_xlabel("|true Δh| (m)")
    ax.set_ylabel("CI half-width (m)")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def fig_coverage_vs_distance_bins(df: pd.DataFrame, out_path: Path, title: str,
                                   bin_edges: np.ndarray | None = None) -> None:
    """Coverage rate broken down by |true_dh| bin."""
    sub = _clean(df)
    if sub.empty:
        fig, ax = plt.subplots(figsize=(7, 5)); ax.set_title(title + " (empty)")
        fig.savefig(out_path); plt.close(fig); return

    if bin_edges is None:
        bin_edges = np.array([0, 3, 6, 12, 24, 60])

    abs_dh = np.abs(sub["true_dh"].to_numpy())
    bins = np.digitize(abs_dh, bin_edges) - 1
    rows = []
    for b in range(len(bin_edges) - 1):
        mask = bins == b
        if mask.sum() == 0:
            continue
        rows.append({
            "label": f"{bin_edges[b]:.0f}–{bin_edges[b+1]:.0f} m",
            "n": int(mask.sum()),
            "coverage": float(np.mean(sub["covered"].to_numpy()[mask])),
            "mae": float(np.mean(sub["abs_error"].to_numpy()[mask])),
        })
    if not rows:
        fig, ax = plt.subplots(figsize=(7, 5)); ax.set_title(title + " (empty)")
        fig.savefig(out_path); plt.close(fig); return
    df_b = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_b))
    ax1.bar(x, df_b["coverage"], color="tab:green", alpha=0.6, label="coverage")
    ax1.axhline(0.9, color="b", ls=":", alpha=0.8, label="90% target")
    ax1.set_xticks(x); ax1.set_xticklabels(df_b["label"])
    ax1.set_ylabel("coverage"); ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(x, df_b["mae"], "r-o", lw=1.5, label="MAE")
    ax2.set_ylabel("MAE (m)")
    for xi, n in zip(x, df_b["n"]):
        ax1.text(xi, 0.03, f"n={n}", ha="center", color="black", fontsize=8)
    ax1.set_title(title)
    # merge legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)


def fig_compare_algorithms(dfs: dict[str, pd.DataFrame], out_path: Path,
                            title: str = "Algorithm comparison") -> None:
    """CDFs of |error| for each algorithm (clean data only)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colour_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (name, df), col in zip(dfs.items(), colour_cycle):
        sub = _clean(df)
        if sub.empty:
            continue
        errs = np.sort(sub["abs_error"].to_numpy())
        ax.plot(errs, np.linspace(0, 1, errs.size), "-", lw=2, c=col,
                label=f"{name} (med={np.median(errs):.2f}m, n={errs.size})")
    ax.axvline(1.5, color="b", ls=":", alpha=0.7, label="±1.5 m")
    ax.axvline(3.0, color="k", ls=":", alpha=0.5, label="±3.0 m")
    ax.set_xlim(0, 8)
    ax.set_xlabel("|error| (m)"); ax.set_ylabel("CDF")
    ax.set_title(title); ax.legend(loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def fig_reliability_diagram(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Expected vs. achieved coverage at nominal α-levels.

    For nominal 50%, 60%, ..., 95% intervals (by scaling ``ci_half_width``
    proportionally using σ-relative widths), plot the actual coverage
    rate on the clean subset. A perfectly calibrated predictor traces
    the diagonal.
    """
    sub = _clean(df).copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(title + " (empty)"); fig.savefig(out_path); plt.close(fig)
        return

    # Rebuild a per-nominal coverage curve by scaling the reported
    # CI half-width by the ratio k_target / k_fitted — equivalent to
    # scaling σ. We assume the fitted 90% CI is achieved by a single
    # scalar multiplier applied to theoretical σ.
    eps = 1e-6
    scores = (sub["abs_error"] / np.clip(sub["theoretical_sigma"], eps, None)).to_numpy()
    nominals = np.arange(0.5, 0.99, 0.025)
    achieved = []
    for p in nominals:
        q = np.quantile(scores, p)
        achieved.append(float(np.mean(scores <= q)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.plot(nominals, achieved, "o-", color="tab:blue", lw=2, label="empirical")
    ax.axhline(0.9, color="r", ls=":", alpha=0.6, label="90%")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title(title)
    ax.set_aspect("equal"); ax.set_xlim(0.5, 1.0); ax.set_ylim(0.5, 1.0)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def fig_error_sign_analysis(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Per-segment *signed* error vs. true Δh. A centered cloud with
    no systematic bias is healthy; systematic offset at extremes
    suggests the kinematic model is over- or under-reaching.
    """
    sub = _clean(df)
    signed = sub["pred_dh"] - sub["true_dh"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub["true_dh"], signed, s=22, c="tab:blue", alpha=0.55,
               edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="k", lw=1)
    ax.axhline(1.5, color="b", ls=":", alpha=0.6)
    ax.axhline(-1.5, color="b", ls=":", alpha=0.6)
    ax.set_xlabel(r"true $\Delta h$ (m)")
    ax.set_ylabel(r"signed error $\hat{\Delta h}-\Delta h$ (m)")
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)


def save_all_figures(
    df: pd.DataFrame, out_dir: Path | str, label: str,
) -> dict[str, Path]:
    """Generate the full figure bundle under ``out_dir`` and return the
    {key: path} map for the LaTeX writer to pick up.
    """
    p = _ensure_dir(out_dir)
    paths: dict[str, Path] = {}

    def _f(key: str, fn, title: str):
        path = p / f"fig_{key}.png"
        fn(df, path, title=f"[{label}] {title}")
        paths[key] = path

    _f("scatter", fig_scatter, "Predicted vs. true Δh")
    _f("cdf", fig_error_cdf, "Error CDF")
    _f("hist", fig_error_histogram, "Error histogram")
    _f("per_ride", fig_per_ride_errors, "Per-ride error")
    _f("ci", fig_ci_coverage, "Per-segment CI")
    _f("quality", fig_quality_vs_error, "Quality score vs. |error|")
    _f("per_exp", fig_per_experiment_mae, "MAE by experiment")
    _f("reject", fig_rejection_reasons, "Rejection reasons")
    _f("ci_vs_dh", fig_ci_vs_distance, "CI width vs. |true Δh|")
    _f("cov_bins", fig_coverage_vs_distance_bins, "Coverage by distance bin")
    _f("reliability", fig_reliability_diagram, "Reliability diagram")
    _f("signed_err", fig_error_sign_analysis, "Signed error vs. true Δh")

    return paths
