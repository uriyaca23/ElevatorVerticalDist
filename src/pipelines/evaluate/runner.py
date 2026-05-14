"""End-to-end pipeline evaluation primitives.

Three independent concerns live here:

* :func:`run_experiment` — runs one experiment through segmentation +
  prediction + barometer, returning per-GT and per-prediction record
  dicts. The schema mirrors what
  ``scripts/pipeline_evaluation_report.py`` uses so the LaTeX report
  stays a thin wrapper over this module.

* :func:`build_views` — given the pooled per-GT / per-pred DataFrames,
  builds the three error views (GT-segments, matched-segmenter, all
  segments vs barometer). Optionally restricts to predictions the
  quality filter accepted (``accepted_only=True``).

* The figure-rendering helpers (:func:`cdf_overlay`, :func:`bar_overall`,
  …) and :func:`render_view_figures` that drives them. They're the same
  figures the pipeline subsection of ``docs/latex/main.tex`` consumes.

The CLI entry point :mod:`src.pipelines.evaluate.evaluateOnData` wraps
these with experiment-list filters and a timestamped output directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.loader import getExperimentData
from src.physics.barometric import pressure_to_altitude
from src.prediction.algorithms.configTypes import (
    PREDICT_ALGORITHM_CONFIG,
    PredictAlgorithm,
)
from src.prediction.algorithms.predictor import Predictor
from src.segmentation.algorithms.configTypes import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
)
from src.segmentation.algorithms.metrics.metrics import (
    DEFAULT_MIN_OVERLAP_FRAC,
    DEFAULT_MIN_OVERLAP_S,
    _intervals_match,
)
from src.segmentation.algorithms.segmenter import Segmenter
from src.segmentation.evaluate.evaluator import (
    _gt_to_interval_dicts,
    _phone_model_from_metadata,
    _prepare_segmenter_input,
    _segments_to_interval_dicts,
)


PRE_MS = 3000
POST_MS = 3000

COLOR = {"gt": "#2980b9", "matched": "#27ae60", "all": "#e74c3c"}
LABEL = {
    "gt":      "GT segments  (predictor on every GT interval, vs gt $\\Delta h$)",
    "matched": "Matched segmenter  (predictor on clean-matched preds, vs matched GT $\\Delta h$)",
    "all":     "All segments  (predictor on every pred, vs barometer $\\Delta h$ on same window)",
}


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Knobs the per-experiment runner needs.

    Defaults wire the deployed accelerometer-only stack: trapezoid
    template-match segmentation + trapezoid pulse-pair prediction. The
    optional ``calibration_path`` (when present and readable) is loaded
    onto the predictor before any prediction is made.
    """
    seg_cfg: SEGMENT_ALGORITHM_CONFIG = field(
        default_factory=lambda: SEGMENT_ALGORITHM_CONFIG(
            algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
        )
    )
    pred_cfg: PREDICT_ALGORITHM_CONFIG = field(
        default_factory=lambda: PREDICT_ALGORITHM_CONFIG(
            algorithm=PredictAlgorithm.TRAPEZOID_ACCEL,
        )
    )
    calibration_path: Optional[Path] = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _slice_acc(acc: pd.DataFrame, lo_ms: float, hi_ms: float) -> pd.DataFrame:
    ts = acc["timestamp_ms"].astype(float).to_numpy()
    mask = (ts >= lo_ms) & (ts < hi_ms)
    return acc.loc[mask].reset_index(drop=True)


def _predict_for_window(
    predictor: Predictor,
    acc: pd.DataFrame,
    t_start_ms: float, t_end_ms: float,
    phone_model: str,
):
    seg = _slice_acc(acc, t_start_ms, t_end_ms)
    if len(seg) < 5:
        return None
    pre  = _slice_acc(acc, t_start_ms - PRE_MS, t_start_ms)
    post = _slice_acc(acc, t_end_ms,            t_end_ms + POST_MS)
    try:
        out = predictor.predict(seg, phone_model=phone_model, pre=pre, post=post)
    except Exception:
        return None
    return (
        float(out.height_diff),
        float(out.ci_half_width) if np.isfinite(out.ci_half_width) else float("inf"),
        bool(out.accepted),
        float(out.quality_score),
    )


def _barometer_dh_over_interval(
    prs: pd.DataFrame, t_start_ms: float, t_end_ms: float,
    temperature_c: float | None = None, edge_k: int = 3,
) -> float | None:
    if prs is None or prs.empty or "pressure" not in prs.columns:
        return None
    ts = prs["timestamp_ms"].astype("int64").to_numpy()
    p_all = prs["pressure"].to_numpy(dtype=float)
    alt_all = pressure_to_altitude(p_all, temperature_c=temperature_c)
    mask = (ts >= int(t_start_ms)) & (ts < int(t_end_ms))
    seg = np.asarray(alt_all)[mask]
    if seg.size < 2:
        return None
    k = max(1, min(edge_k, seg.size // 2))
    return float(np.mean(seg[-k:]) - np.mean(seg[:k]))


def _classify(gt_rides, preds):
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
    gt_status, gt_match = [], []
    for ps in gt_to:
        if len(ps) == 0:
            gt_status.append("missed"); gt_match.append(-1)
        elif len(ps) == 1:
            if len(pr_to[ps[0]]) == 1:
                gt_status.append("clean"); gt_match.append(ps[0])
            else:
                gt_status.append("gt_merged"); gt_match.append(ps[0])
        else:
            gt_status.append("gt_split"); gt_match.append(ps[0])
    pred_status, pred_match = [], []
    for gs in pr_to:
        if len(gs) == 0:
            pred_status.append("fp"); pred_match.append(-1)
        elif len(gs) == 1 and len(gt_to[gs[0]]) == 1:
            pred_status.append("clean"); pred_match.append(gs[0])
        else:
            pred_status.append("entangled"); pred_match.append(gs[0])
    return gt_status, pred_status, gt_match, pred_match


# --------------------------------------------------------------------------
# Per-experiment runner
# --------------------------------------------------------------------------
def run_experiment(
    exp_name: str,
    config: PipelineConfig,
    predictor: Predictor,
) -> tuple[list[dict], list[dict]]:
    """Process one experiment end-to-end.

    Returns ``(gt_records, seg_records)`` — per-GT-ride rows (with
    ``true_dh``, predictor's oracle output on the GT interval, status
    label) and per-predicted-segment rows (with ``pred_dh``,
    barometer-on-pred-interval ``baro_dh``, ``matched_gt_dh`` for
    clean predictions).

    ``predictor`` should be pre-configured (calibration loaded etc.).
    """
    sensors, gt, metadata = getExperimentData(exp_name)
    if "ACC" not in sensors or sensors["ACC"].empty:
        return [], []
    acc = sensors["ACC"]
    prs = sensors.get("PRS")
    t0_ms = float(acc["timestamp_ms"].iloc[0])
    phone = _phone_model_from_metadata(metadata)
    exp_kind = "test" if "beityitzchaki" in exp_name.lower() else "train"

    temp_c: float | None = None
    if metadata and metadata.get("temperature_c"):
        try:
            temp_c = float(metadata["temperature_c"])
        except (TypeError, ValueError):
            temp_c = None

    # Segmentation
    data, _t0 = _prepare_segmenter_input(sensors, config.seg_cfg.algorithm)
    segmenter = Segmenter(config.seg_cfg)
    segments_df = segmenter.detect(data, phone_model=phone)
    gt_rides = _gt_to_interval_dicts(gt, t0_ms)
    preds = _segments_to_interval_dicts(segments_df)
    gt_status, pred_status, gt_match, pred_match = _classify(gt_rides, preds)

    # Truth Δh and noise-polarity per GT ride (only up/down rows in gt)
    gt_dh_per_ride: list[float] = []
    gt_signal_clear_per_ride: list[bool] = []
    for _, row in gt.iterrows():
        if row.get("type") not in ("up", "down"):
            continue
        gt_dh_per_ride.append(
            float(row.get("height_diff_m", float("nan")))
        )
        sc = row.get("signalClearRecording", True)
        gt_signal_clear_per_ride.append(
            bool(sc) if sc is not None else True
        )
    if len(gt_dh_per_ride) != len(gt_rides):
        gt_dh_per_ride = (
            gt_dh_per_ride + [float("nan")] * len(gt_rides)
        )[: len(gt_rides)]
        gt_signal_clear_per_ride = (
            gt_signal_clear_per_ride + [True] * len(gt_rides)
        )[: len(gt_rides)]

    # Predictor on every GT interval (oracle view)
    gt_records: list[dict] = []
    for i, g in enumerate(gt_rides):
        s_ms = t0_ms + g["t_start_s"] * 1000.0
        e_ms = t0_ms + g["t_end_s"]   * 1000.0
        oracle = _predict_for_window(predictor, acc, s_ms, e_ms, phone)
        gt_records.append({
            "exp": exp_name, "kind": exp_kind, "gt_idx": i,
            "type": g.get("type"),
            "duration_s": g["t_end_s"] - g["t_start_s"],
            "true_dh": gt_dh_per_ride[i],
            "signal_clear": gt_signal_clear_per_ride[i],
            "status": gt_status[i],
            "oracle_pred_dh":  oracle[0] if oracle else float("nan"),
            "oracle_ci":       oracle[1] if oracle else float("inf"),
            "oracle_accepted": oracle[2] if oracle else False,
        })

    # Predictor on every predicted segment + barometer truth
    seg_records: list[dict] = []
    for j, p in enumerate(preds):
        s_ms = t0_ms + p["t_start_s"] * 1000.0
        e_ms = t0_ms + p["t_end_s"]   * 1000.0
        out = _predict_for_window(predictor, acc, s_ms, e_ms, phone)
        baro_dh = _barometer_dh_over_interval(prs, s_ms, e_ms, temp_c)
        match_i = pred_match[j]
        matched_gt_dh = (
            gt_dh_per_ride[match_i]
            if 0 <= match_i < len(gt_dh_per_ride) else float("nan")
        )
        matched_signal_clear = (
            gt_signal_clear_per_ride[match_i]
            if 0 <= match_i < len(gt_signal_clear_per_ride) else None
        )
        seg_records.append({
            "exp": exp_name, "kind": exp_kind, "pred_idx": j,
            "type": p.get("type"),
            "duration_s": p["t_end_s"] - p["t_start_s"],
            "status": pred_status[j],
            "matched_gt_idx": match_i,
            "matched_gt_dh":  matched_gt_dh,
            # FP preds have no matched GT → signal_clear is None; the
            # noise filter in build_views drops these from clean/noisy
            # passes and keeps them in the "both" pass.
            "signal_clear":   matched_signal_clear,
            "pred_dh":        out[0] if out else float("nan"),
            "pred_ci":        out[1] if out else float("inf"),
            "pred_accepted":  out[2] if out else False,
            "baro_dh":        baro_dh if baro_dh is not None else float("nan"),
        })

    return gt_records, seg_records


# --------------------------------------------------------------------------
# Three views
# --------------------------------------------------------------------------
def _summary(arr: np.ndarray) -> dict:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mae": float("nan"), "median": float("nan"),
                "rmse": float("nan"),
                "p_within_0_5m": float("nan"),
                "p_within_1_5m": float("nan")}
    return {
        "n": int(arr.size),
        "mae":          float(np.mean(arr)),
        "median":       float(np.median(arr)),
        "rmse":         float(np.sqrt(np.mean(arr * arr))),
        "p_within_0_5m": float((arr <= 0.5).mean()),
        "p_within_1_5m": float((arr <= 1.5).mean()),
    }


def _apply_noise_filter(
    gt_df: pd.DataFrame, seg_df: pd.DataFrame,
    signal_clear: bool | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice the pooled DataFrames to a noise polarity.

    ``signal_clear`` is ``True`` for clean only, ``False`` for noisy only,
    ``None`` to keep all rows. seg rows with ``signal_clear is None``
    (FP preds with no matched GT) are dropped from clean/noisy passes
    and retained in the ``None`` ("both") pass.
    """
    if signal_clear is None:
        return gt_df, seg_df
    gt_df = gt_df[gt_df["signal_clear"] == signal_clear]
    seg_df = seg_df[seg_df["signal_clear"] == signal_clear]
    return gt_df, seg_df


def build_views(
    gt_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    accepted_only: bool = False,
    signal_clear: bool | None = None,
) -> dict[str, dict]:
    """Build the three error views (gt / matched / all) from the pooled
    DataFrames :func:`run_experiment` produces.

    ``accepted_only`` restricts every view to predictions the quality
    filter accepted (``oracle_accepted`` for the GT view,
    ``pred_accepted`` for the matched / all views) — the production
    deployment view. ``signal_clear`` slices to one noise polarity (see
    :func:`_apply_noise_filter`).
    """
    gt_df, seg_df = _apply_noise_filter(gt_df, seg_df, signal_clear)
    if accepted_only:
        gt_df = gt_df[gt_df["oracle_accepted"] == True]    # noqa: E712
        seg_df = seg_df[seg_df["pred_accepted"] == True]   # noqa: E712

    # GT view
    gt_v = gt_df.dropna(subset=["true_dh", "oracle_pred_dh"])
    err_a = (gt_v["oracle_pred_dh"] - gt_v["true_dh"]).abs().to_numpy(dtype=float)

    # Matched view
    matched = seg_df[
        (seg_df["status"] == "clean")
        & seg_df["matched_gt_dh"].notna()
        & seg_df["pred_dh"].notna()
    ]
    err_b = (matched["pred_dh"] - matched["matched_gt_dh"]).abs().to_numpy(dtype=float)

    # All view
    all_v = seg_df.dropna(subset=["pred_dh", "baro_dh"])
    err_c = (all_v["pred_dh"] - all_v["baro_dh"]).abs().to_numpy(dtype=float)

    return {
        "gt":      {"err": err_a, "summary": _summary(err_a)},
        "matched": {"err": err_b, "summary": _summary(err_b)},
        "all":     {"err": err_c, "summary": _summary(err_c)},
    }


# --------------------------------------------------------------------------
# Plot helpers (the main.tex pipeline figures)
# --------------------------------------------------------------------------
def cdf_overlay(views: dict, title: str, out_path: Path,
                xmax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key in ("gt", "matched", "all"):
        arr = views[key]["err"]
        if arr.size == 0:
            continue
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, lw=1.6, color=COLOR[key],
                label=f"{LABEL[key]}  (n={len(xs)})")
    ax.axvline(1.5, color="gray", lw=0.6, ls="--",
               label="$\\pm$1.5 m one-floor target")
    ax.set_ylim(0.0, 1.0)
    if xmax is not None:
        ax.set_xlim(0.0, xmax)
    ax.set_xlabel("absolute $\\Delta h$ error (m)")
    ax.set_ylabel("empirical CDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def bar_overall(views_pooled, views_train, views_test, out_path: Path) -> None:
    rows = ["GT segments", "Matched segmenter", "All vs barometer"]
    keys = ["gt", "matched", "all"]
    pooled = [views_pooled[k]["summary"]["mae"] for k in keys]
    train  = [views_train[k]["summary"]["mae"]  for k in keys]
    test   = [views_test[k]["summary"]["mae"]   for k in keys]

    x = np.arange(len(rows))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, pooled, width=w, color="#34495e", label="Pooled")
    ax.bar(x,     train,  width=w, color="#2980b9", label="Train")
    ax.bar(x + w, test,   width=w, color="#e67e22", label="Test (held-out)")
    for px, vals in zip(x, list(zip(pooled, train, test))):
        for off, v in zip([-w, 0, w], vals):
            if np.isfinite(v):
                ax.text(px + off, v, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8)
    ax.axhline(1.5, color="gray", lw=0.6, ls="--", label="1.5 m target")
    ax.set_xticks(x)
    ax.set_xticklabels(rows)
    ax.set_ylabel("MAE on $|\\Delta h|$ (m)")
    ax.set_title("MAE per view, pooled / train / test")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def signed_pdf_pair(views: dict, src_df: dict, out_path: Path) -> None:
    XLIM = 5.0
    BIN_W = 0.5
    edges = np.arange(-XLIM, XLIM + BIN_W, BIN_W)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key in zip(axes, ("gt", "matched", "all")):
        arr = src_df[key]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        n_out = int((np.abs(arr) > XLIM).sum())
        ax.hist(arr, bins=edges, color=COLOR[key], alpha=0.65,
                edgecolor="white", density=True)
        ax.axvline(0, color="black", lw=0.5)
        ax.axvline(float(np.median(arr)), color="#c0392b", ls="--", lw=0.8,
                   label=f"median = {np.median(arr):+.2f} m")
        ax.set_xlim(-XLIM, XLIM)
        ax.set_xticks(np.arange(-XLIM, XLIM + 1.0, 1.0))
        ax.set_xlabel("signed error: pred $-$ truth (m)  [$\\pm$5 m clipped]")
        ax.set_ylabel("density")
        ax.set_title(f"{key.upper()} view  "
                     f"(n={len(arr)}; {n_out} outside $\\pm$5 m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def scatter_three(gt_df: pd.DataFrame, seg_df: pd.DataFrame,
                  out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    sub = gt_df.dropna(subset=["true_dh", "oracle_pred_dh"])
    axes[0].scatter(sub["true_dh"], sub["oracle_pred_dh"],
                    s=22, alpha=0.6, color=COLOR["gt"], edgecolor="none")
    matched = seg_df[
        (seg_df["status"] == "clean")
        & seg_df["matched_gt_dh"].notna()
        & seg_df["pred_dh"].notna()
    ]
    axes[1].scatter(matched["matched_gt_dh"], matched["pred_dh"],
                    s=22, alpha=0.6, color=COLOR["matched"], edgecolor="none")
    all_v = seg_df.dropna(subset=["pred_dh", "baro_dh"])
    axes[2].scatter(all_v["baro_dh"], all_v["pred_dh"],
                    s=22, alpha=0.6, color=COLOR["all"], edgecolor="none")

    titles = ["GT view (oracle)", "Matched-segmenter view",
              "All segments vs barometer"]
    xlabels = ["true $\\Delta h$ (m)", "matched-GT $\\Delta h$ (m)",
               "barometer $\\Delta h$ on same interval (m)"]
    for ax, t, xl in zip(axes, titles, xlabels):
        lo, hi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        m = max(abs(lo), abs(hi), abs(ylo), abs(yhi))
        ax.plot([-m, m], [-m, m], "k--", lw=0.7, alpha=0.6)
        ax.set_xlim(-m, m); ax.set_ylim(-m, m)
        ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.4, alpha=0.5)
        ax.set_xlabel(xl)
        ax.set_ylabel("predicted $\\Delta h$ (m)")
        ax.set_title(t)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def fp_predicted_dh_hist(seg_df: pd.DataFrame, out_path: Path) -> None:
    fp = seg_df[seg_df["status"] == "fp"].dropna(subset=["pred_dh"])
    XLIM = 5.0
    BIN_W = 0.5
    edges_signed = np.arange(-XLIM, XLIM + BIN_W, BIN_W)
    edges_abs    = np.arange(0.0, XLIM + BIN_W, BIN_W)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if fp.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "no FPs", ha="center", va="center",
                    transform=ax.transAxes)
    else:
        sd = fp["pred_dh"].to_numpy(dtype=float)
        ad = np.abs(sd)
        n_out_s = int((np.abs(sd) > XLIM).sum())
        n_out_a = int((ad > XLIM).sum())

        axes[0].hist(sd, bins=edges_signed, color="#7f8c8d", alpha=0.85,
                     edgecolor="white")
        axes[0].axvline(0, color="black", ls="--", lw=0.5)
        axes[0].axvline(float(np.median(sd)), color="#c0392b", ls="--", lw=0.8,
                        label=f"median = {np.median(sd):+.2f} m")
        axes[0].set_xlim(-XLIM, XLIM)
        axes[0].set_xlabel("predicted $\\Delta h$ on FP segments (m)")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"Signed $\\Delta h$ on FP predictions  "
                          f"(n={len(sd)}; {n_out_s} outside $\\pm$5 m)")
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].hist(ad, bins=edges_abs, color="#e74c3c", alpha=0.85,
                     edgecolor="white")
        axes[1].axvline(float(np.median(ad)), color="black", ls="--", lw=0.8,
                        label=f"median = {np.median(ad):.2f} m")
        axes[1].axvline(1.5, color="gray", ls="--", lw=0.6,
                        label="1.5 m one-floor")
        axes[1].set_xlim(0.0, XLIM)
        axes[1].set_xlabel("$|$predicted $\\Delta h|$ on FP segments (m)")
        axes[1].set_ylabel("count")
        axes[1].set_title(f"Absolute $\\Delta h$ on FPs  "
                          f"(mean = {np.mean(ad):.2f} m; "
                          f"{n_out_a} outside $5$ m)")
        axes[1].grid(True, axis="y", alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def fp_vs_clean_predicted_dh(seg_df: pd.DataFrame, out_path: Path) -> None:
    fp = seg_df[(seg_df["status"] == "fp") & seg_df["pred_dh"].notna()]
    clean = seg_df[(seg_df["status"] == "clean") & seg_df["pred_dh"].notna()]
    XLIM = 5.0
    BIN_W = 0.5
    edges = np.arange(-XLIM, XLIM + BIN_W, BIN_W)

    def _stats(arr: np.ndarray):
        if arr.size == 0:
            return float("nan"), float("nan"), 0
        n_in = int(((arr >= -XLIM) & (arr <= XLIM)).sum())
        return float(np.median(arr)), float(np.median(np.abs(arr))), arr.size - n_in

    fp_arr = fp["pred_dh"].to_numpy(dtype=float)
    clean_arr = clean["pred_dh"].to_numpy(dtype=float)
    fp_med, fp_amed, fp_out = _stats(fp_arr)
    cl_med, cl_amed, cl_out = _stats(clean_arr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, arr, color, label, med, amed, n_out in [
        (axes[0], fp_arr,    "#7f8c8d", "FP",    fp_med, fp_amed, fp_out),
        (axes[1], clean_arr, "#27ae60", "clean", cl_med, cl_amed, cl_out),
    ]:
        ax.hist(arr, bins=edges, color=color, alpha=0.85,
                edgecolor="white", density=True)
        ax.axvline(0, color="black", ls="--", lw=0.5)
        ax.axvline(med, color="#c0392b", ls="--", lw=0.8,
                   label=f"median = {med:+.2f} m")
        ax.set_xlim(-XLIM, XLIM)
        ax.set_xlabel("predicted $\\Delta h$ on segment (m)")
        ax.set_ylabel("density")
        ax.set_title(f"{label} predictions  (n={arr.size}; "
                     f"median $|\\Delta h|$={amed:.2f} m; "
                     f"{n_out} outside)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.hist(fp_arr,    bins=edges, color="#7f8c8d", alpha=0.6,
            edgecolor="white", density=True, label=f"FP  (n={fp_arr.size})")
    ax.hist(clean_arr, bins=edges, color="#27ae60", alpha=0.5,
            edgecolor="white", density=True, label=f"clean  (n={clean_arr.size})")
    ax.axvline(0, color="black", ls="--", lw=0.5)
    ax.set_xlim(-XLIM, XLIM)
    ax.set_xlabel("predicted $\\Delta h$ on segment (m)")
    ax.set_ylabel("density")
    ax.set_title("Overlay: FP vs clean")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def fp_predicted_altitude(seg_df: pd.DataFrame, out_path: Path) -> None:
    sub = seg_df.dropna(subset=["pred_dh"]).copy()
    if sub.empty:
        return
    rows: list[dict] = []
    fp_total: dict[str, float] = {}
    for exp, grp in sub.groupby("exp"):
        grp = grp.sort_values("pred_idx")
        cum = grp["pred_dh"].cumsum().to_numpy()
        statuses = grp["status"].to_numpy()
        dhs      = grp["pred_dh"].to_numpy(dtype=float)
        fp_only_sum = 0.0
        for i in range(len(grp)):
            if statuses[i] == "fp":
                fp_only_sum += dhs[i]
                rows.append({
                    "exp": exp,
                    "alt_at_fp_end": float(cum[i]),
                    "fp_dh": float(dhs[i]),
                })
        fp_total[exp] = fp_only_sum
    if not rows:
        return
    df = pd.DataFrame(rows)
    arr = df["alt_at_fp_end"].to_numpy()
    XLIM = float(max(20.0, np.ceil(np.percentile(np.abs(arr), 95))))
    BIN_W = 0.5 if XLIM <= 10 else (1.0 if XLIM <= 30 else 2.0)
    edges = np.arange(-XLIM, XLIM + BIN_W, BIN_W)
    n_out = int((np.abs(arr) > XLIM).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(arr, bins=edges, color="#7f8c8d", alpha=0.85,
                 edgecolor="white")
    axes[0].axvline(0, color="black", ls="--", lw=0.5)
    axes[0].axvline(float(np.median(arr)), color="#c0392b", ls="--", lw=0.8,
                    label=f"median = {np.median(arr):+.2f} m")
    axes[0].set_xlim(-XLIM, XLIM)
    axes[0].set_xlabel("system's predicted altitude at FP (m)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Predicted altitude at the moment of each FP  "
                      f"(n={len(arr)}; {n_out} outside)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    items = sorted(fp_total.items(), key=lambda kv: kv[1])
    items = [(e, v) for e, v in items if v != 0.0]
    short = lambda nm: (nm.split("_")[0][:3] + "/"
                        + (nm.split("_")[1][:11] if len(nm.split("_")) > 1
                           else ""))
    names = [short(e) for e, _ in items]
    vals = [v for _, v in items]
    colors = ["#27ae60" if v >= 0 else "#c0392b" for v in vals]
    y = np.arange(len(items))
    axes[1].barh(y, vals, color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names, fontsize=7)
    axes[1].axvline(0, color="black", lw=0.5)
    axes[1].set_xlabel("total altitude FPs would inject (m)")
    axes[1].set_title(f"Per-experiment phantom-altitude bias from FPs  "
                      f"(n={len(items)})")
    axes[1].grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def clean_predicted_altitude(gt_df: pd.DataFrame, seg_df: pd.DataFrame,
                             out_path: Path) -> None:
    sub_seg = seg_df.dropna(subset=["pred_dh"]).copy()
    sub_gt  = gt_df.dropna(subset=["true_dh"]).copy()
    if sub_seg.empty or sub_gt.empty:
        return
    rows: list[dict] = []
    for exp, seg_e in sub_seg.groupby("exp"):
        gt_e = sub_gt[sub_gt["exp"] == exp]
        if gt_e.empty:
            continue
        seg_e = seg_e.sort_values("pred_idx").reset_index(drop=True)
        gt_e  = gt_e.sort_values("gt_idx").reset_index(drop=True)
        seg_cum = seg_e["pred_dh"].cumsum().to_numpy()
        gt_cum  = gt_e["true_dh"].cumsum().to_numpy()
        gt_idx_lookup = {int(gt_e.loc[k, "gt_idx"]): k
                         for k in range(len(gt_e))}
        for i in range(len(seg_e)):
            if seg_e.loc[i, "status"] != "clean":
                continue
            gt_idx = int(seg_e.loc[i, "matched_gt_idx"])
            if gt_idx < 0 or gt_idx not in gt_idx_lookup:
                continue
            rows.append({
                "exp": exp,
                "pred_alt": float(seg_cum[i]),
                "true_alt": float(gt_cum[gt_idx_lookup[gt_idx]]),
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["err"] = df["pred_alt"] - df["true_alt"]
    arr_pred = df["pred_alt"].to_numpy()
    arr_err  = df["err"].to_numpy()
    AX_LIM_S = float(max(20.0, np.ceil(np.percentile(np.abs(arr_pred), 95))))
    BIN_W = 0.5 if AX_LIM_S <= 10 else (1.0 if AX_LIM_S <= 30 else 2.0)
    edges_alt = np.arange(-AX_LIM_S, AX_LIM_S + BIN_W, BIN_W)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].hist(arr_pred, bins=edges_alt, color=COLOR["matched"],
                 alpha=0.85, edgecolor="white")
    axes[0].axvline(0, color="black", ls="--", lw=0.5)
    axes[0].axvline(float(np.median(arr_pred)), color="#c0392b", ls="--",
                    lw=0.8, label=f"median = {np.median(arr_pred):+.2f} m")
    axes[0].set_xlim(-AX_LIM_S, AX_LIM_S)
    axes[0].set_xlabel("system's predicted altitude (m)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Predicted altitude at each correct prediction  "
                      f"(n={len(arr_pred)})")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    ERR_LIM = 5.0
    BIN_E = 0.5
    edges_err = np.arange(-ERR_LIM, ERR_LIM + BIN_E, BIN_E)
    axes[1].hist(arr_err, bins=edges_err, color="#16a085", alpha=0.85,
                 edgecolor="white")
    axes[1].axvline(0, color="black", ls="--", lw=0.5)
    axes[1].axvline(float(np.median(arr_err)), color="#c0392b", ls="--",
                    lw=0.8, label=f"median = {np.median(arr_err):+.2f} m")
    axes[1].set_xlim(-ERR_LIM, ERR_LIM)
    axes[1].set_xlabel("cumulative altitude error: pred $-$ true (m)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Residual at each correct prediction  "
                      f"(n={len(arr_err)})")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def per_exp_mae_bar(gt_df: pd.DataFrame, seg_df: pd.DataFrame,
                    out_path: Path) -> None:
    rows: list[dict] = []
    for exp in sorted(gt_df["exp"].unique()):
        gt_e  = gt_df[gt_df["exp"] == exp]
        seg_e = seg_df[seg_df["exp"] == exp]
        a = (gt_e.dropna(subset=["true_dh", "oracle_pred_dh"])
                  .assign(err=lambda d: (d["oracle_pred_dh"] - d["true_dh"]).abs())
                  ["err"].to_numpy())
        m = seg_e[(seg_e["status"] == "clean")
                  & seg_e["matched_gt_dh"].notna()
                  & seg_e["pred_dh"].notna()]
        b = ((m["pred_dh"] - m["matched_gt_dh"]).abs().to_numpy()
             if not m.empty else np.array([]))
        c = (seg_e.dropna(subset=["pred_dh", "baro_dh"])
                  .assign(err=lambda d: (d["pred_dh"] - d["baro_dh"]).abs())
                  ["err"].to_numpy())
        rows.append({
            "exp": exp, "n_gt": len(gt_e),
            "gt":      float(np.mean(a)) if len(a) else float("nan"),
            "matched": float(np.mean(b)) if len(b) else float("nan"),
            "all":     float(np.mean(c)) if len(c) else float("nan"),
        })
    rows.sort(key=lambda r: r["n_gt"], reverse=True)
    short = lambda nm: (nm.split("_")[0][:3] + "/"
                        + (nm.split("_")[1][:11] if len(nm.split("_")) > 1
                           else ""))
    names = [short(r["exp"]) for r in rows]
    x = np.arange(len(rows))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(11, 0.55 * len(rows)), 5))
    ax.bar(x - w, [r["gt"]      for r in rows], width=w,
           color=COLOR["gt"],      label="GT view")
    ax.bar(x,     [r["matched"] for r in rows], width=w,
           color=COLOR["matched"], label="Matched view")
    ax.bar(x + w, [r["all"]     for r in rows], width=w,
           color=COLOR["all"],     label="All vs barometer")
    ax.axhline(1.5, color="gray", lw=0.6, ls="--", label="1.5 m target")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE on $|\\Delta h|$ (m)")
    ax.set_title("Per-experiment MAE for the three views")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


DURATION_BIN_EDGES = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 90.0])


def _gt_view_arrays(gt_df: pd.DataFrame):
    """Return (duration_s, abs_err, covered) over the GT (oracle) view."""
    sub = gt_df.dropna(subset=["true_dh", "oracle_pred_dh"])
    if sub.empty:
        return (np.array([]), np.array([]), np.array([], dtype=bool))
    err = (sub["oracle_pred_dh"] - sub["true_dh"]).abs().to_numpy(dtype=float)
    ci = sub["oracle_ci"].to_numpy(dtype=float)
    covered = err <= ci
    return (sub["duration_s"].to_numpy(dtype=float), err, covered)


def coverage_vs_duration_bins(gt_df: pd.DataFrame, out_path: Path,
                              title: str = "Coverage by ride-duration bin",
                              bin_edges: np.ndarray | None = None) -> None:
    """Pipeline coverage rate per ride-duration bin (oracle GT view).

    Bars = coverage (P(|err| ≤ oracle CI)); overlaid line = MAE.
    """
    dur, err, cov = _gt_view_arrays(gt_df)
    if bin_edges is None:
        bin_edges = DURATION_BIN_EDGES
    fig, ax1 = plt.subplots(figsize=(8, 5))
    if dur.size == 0:
        ax1.text(0.5, 0.5, "no data", ha="center", va="center",
                 transform=ax1.transAxes)
        ax1.set_title(title); fig.savefig(out_path, dpi=120); plt.close(fig)
        return
    bins = np.digitize(dur, bin_edges) - 1
    rows = []
    for b in range(len(bin_edges) - 1):
        m = bins == b
        if m.sum() == 0:
            continue
        rows.append({
            "label": f"{bin_edges[b]:.0f}–{bin_edges[b+1]:.0f} s",
            "n":        int(m.sum()),
            "coverage": float(np.mean(cov[m])),
            "mae":      float(np.mean(err[m])),
        })
    if not rows:
        ax1.text(0.5, 0.5, "no data in bins", ha="center", va="center",
                 transform=ax1.transAxes)
        ax1.set_title(title); fig.savefig(out_path, dpi=120); plt.close(fig)
        return
    df_b = pd.DataFrame(rows)
    x = np.arange(len(df_b))
    ax1.bar(x, df_b["coverage"], color=COLOR["gt"], alpha=0.6, label="coverage")
    ax1.axhline(0.9, color="b", ls=":", alpha=0.8, label="90% target")
    ax1.set_xticks(x); ax1.set_xticklabels(df_b["label"])
    ax1.set_xlabel("ride duration (s)")
    ax1.set_ylabel("coverage"); ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(x, df_b["mae"], "r-o", lw=1.5, label="MAE")
    ax2.set_ylabel("MAE (m)")
    for xi, n in zip(x, df_b["n"]):
        ax1.text(xi, 0.03, f"n={n}", ha="center", color="black", fontsize=8)
    ax1.set_title(title)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def error_vs_duration_bins(gt_df: pd.DataFrame, out_path: Path,
                           title: str = "Δh error by ride-duration bin",
                           bin_edges: np.ndarray | None = None) -> None:
    """Pipeline signed + absolute error per ride-duration bin (oracle GT view)."""
    dur, abs_err, _cov = _gt_view_arrays(gt_df)
    sub = gt_df.dropna(subset=["true_dh", "oracle_pred_dh"])
    if bin_edges is None:
        bin_edges = DURATION_BIN_EDGES
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if dur.size == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
        fig.suptitle(title); fig.tight_layout()
        fig.savefig(out_path, dpi=120); plt.close(fig); return
    signed = (sub["oracle_pred_dh"] - sub["true_dh"]).to_numpy(dtype=float)
    bins = np.digitize(dur, bin_edges) - 1
    rows = []
    for b in range(len(bin_edges) - 1):
        m = bins == b
        if m.sum() == 0:
            continue
        rows.append({
            "label": f"{bin_edges[b]:.0f}–{bin_edges[b+1]:.0f} s",
            "n":             int(m.sum()),
            "median_signed": float(np.median(signed[m])),
            "mae":           float(np.mean(abs_err[m])),
            "p95_abs":       float(np.quantile(abs_err[m], 0.95)),
        })
    if not rows:
        for ax in axes:
            ax.text(0.5, 0.5, "no data in bins", ha="center", va="center",
                    transform=ax.transAxes)
        fig.suptitle(title); fig.tight_layout()
        fig.savefig(out_path, dpi=120); plt.close(fig); return
    df_b = pd.DataFrame(rows)
    x = np.arange(len(df_b))

    axes[0].bar(x, df_b["median_signed"], color=COLOR["gt"], alpha=0.7,
                label="median signed error")
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].axhline(1.5, color="b", ls=":", alpha=0.6, label="±1.5 m")
    axes[0].axhline(-1.5, color="b", ls=":", alpha=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(df_b["label"])
    axes[0].set_xlabel("ride duration (s)")
    axes[0].set_ylabel("signed error: pred $-$ truth (m)")
    axes[0].set_title("Signed error by duration bin")
    for xi, n in zip(x, df_b["n"]):
        axes[0].text(xi, axes[0].get_ylim()[0], f"n={n}",
                     ha="center", va="bottom", fontsize=8, color="black")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    w = 0.4
    axes[1].bar(x - w / 2, df_b["mae"], width=w, color="tab:red",
                alpha=0.75, label="MAE")
    axes[1].bar(x + w / 2, df_b["p95_abs"], width=w, color="tab:orange",
                alpha=0.75, label="P95 |error|")
    axes[1].axhline(1.5, color="b", ls=":", alpha=0.6, label="±1.5 m")
    axes[1].set_xticks(x); axes[1].set_xticklabels(df_b["label"])
    axes[1].set_xlabel("ride duration (s)")
    axes[1].set_ylabel("|error| (m)")
    axes[1].set_title("Absolute error by duration bin")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)


def baro_vs_gt_consistency(seg_df: pd.DataFrame, out_path: Path) -> None:
    matched = seg_df[
        (seg_df["status"] == "clean")
        & seg_df["matched_gt_dh"].notna()
        & seg_df["baro_dh"].notna()
    ]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    if matched.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
    else:
        ax.scatter(matched["matched_gt_dh"], matched["baro_dh"],
                   s=22, alpha=0.6, color="#16a085", edgecolor="none",
                   label=f"n={len(matched)}")
        m = max(abs(matched["matched_gt_dh"]).max(),
                abs(matched["baro_dh"]).max()) * 1.05
        ax.plot([-m, m], [-m, m], "k--", lw=0.7, alpha=0.6, label="$y = x$")
        ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.4, alpha=0.5)
    ax.set_xlabel("matched-GT $\\Delta h$ (m, snapped)")
    ax.set_ylabel("barometer $\\Delta h$ on segmenter interval (m)")
    ax.set_title("Sanity: barometer-on-pred-interval vs snapped GT $\\Delta h$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# All-figures driver
# --------------------------------------------------------------------------
def render_view_figures(
    gt_df: pd.DataFrame, seg_df: pd.DataFrame,
    out_dir: Path, suffix: str = "",
) -> dict[str, Path]:
    """Render every pipeline figure that the LaTeX report references.

    ``suffix`` lands on each filename (e.g. ``"_acc"``) so two passes
    (raw and accepted-only) can coexist in the same directory without
    overwriting each other.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled = build_views(gt_df, seg_df)
    train  = build_views(gt_df[gt_df["kind"] == "train"],
                         seg_df[seg_df["kind"] == "train"])
    test   = build_views(gt_df[gt_df["kind"] == "test"],
                         seg_df[seg_df["kind"] == "test"])

    paths: dict[str, Path] = {}
    def _w(name: str) -> Path:
        p = out_dir / f"{name}{suffix}.png"
        paths[name] = p
        return p

    cdf_overlay(pooled, "CDF — pooled (full range)", _w("cdf_pooled"), xmax=10.0)
    cdf_overlay(pooled, "CDF — pooled (zoom 0–3 m)",
                _w("cdf_pooled_zoom"), xmax=3.0)
    cdf_overlay(train,  "CDF — train split", _w("cdf_train"), xmax=10.0)
    cdf_overlay(test,   "CDF — test split (held-out)",
                _w("cdf_test"), xmax=10.0)
    bar_overall(pooled, train, test, _w("bar_mae_overall"))
    per_exp_mae_bar(gt_df, seg_df, _w("per_exp_mae"))
    scatter_three(gt_df, seg_df, _w("scatter_three"))

    signed_pdf_pair(
        pooled,
        src_df={
            "gt": (gt_df.dropna(subset=["true_dh", "oracle_pred_dh"])
                   .assign(s=lambda d: d["oracle_pred_dh"] - d["true_dh"])
                   ["s"].to_numpy()),
            "matched": (seg_df[(seg_df["status"] == "clean")
                               & seg_df["matched_gt_dh"].notna()
                               & seg_df["pred_dh"].notna()]
                        .assign(s=lambda d: d["pred_dh"] - d["matched_gt_dh"])
                        ["s"].to_numpy()),
            "all": (seg_df.dropna(subset=["pred_dh", "baro_dh"])
                    .assign(s=lambda d: d["pred_dh"] - d["baro_dh"])
                    ["s"].to_numpy()),
        },
        out_path=_w("signed_error_pdf"),
    )

    fp_predicted_dh_hist(seg_df, _w("fp_predicted_dh"))
    fp_predicted_altitude(seg_df, _w("fp_predicted_altitude"))
    fp_vs_clean_predicted_dh(seg_df, _w("fp_vs_clean_dh"))
    clean_predicted_altitude(gt_df, seg_df, _w("clean_predicted_altitude"))

    # Pipeline's mirror of the prediction per-duration plots — uses the
    # GT (oracle) view so both axes (true_dh and oracle_ci) are well-defined.
    coverage_vs_duration_bins(gt_df, _w("coverage_vs_duration"))
    error_vs_duration_bins(gt_df, _w("err_vs_duration"))

    if not suffix:
        baro_vs_gt_consistency(seg_df, _w("baro_vs_gt_sanity"))
    return paths


def run_all_experiments(
    experiments: Iterable[str],
    config: PipelineConfig,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run :func:`run_experiment` over every experiment, returning the
    pooled per-GT and per-pred DataFrames.
    """
    predictor = Predictor(config.pred_cfg)
    if config.calibration_path is not None and Path(config.calibration_path).exists():
        try:
            predictor.load_calibration(config.calibration_path)
            if verbose:
                print(f"loaded calibration from {Path(config.calibration_path).name}")
        except Exception as exc:
            print(f"  [warn] calibration load failed: {exc}")

    all_gt: list[dict] = []
    all_seg: list[dict] = []
    experiments = list(experiments)
    for k, exp in enumerate(experiments, 1):
        try:
            gtr, segr = run_experiment(exp, config, predictor)
        except Exception as exc:
            print(f"  [{k:2d}/{len(experiments)}] {exp[:60]}: "
                  f"{type(exc).__name__}: {exc}")
            continue
        all_gt.extend(gtr)
        all_seg.extend(segr)
        if verbose:
            n_fp = sum(1 for r in segr if r["status"] == "fp")
            print(f"  [{k:2d}/{len(experiments)}] {exp[:60]:<60} "
                  f"gt={len(gtr)} pr={len(segr)} fp={n_fp}", flush=True)

    return pd.DataFrame(all_gt), pd.DataFrame(all_seg)
