"""End-to-end pipeline evaluation: data → segmentation → prediction → Δh.

Three views, each answering a different operational question. In every
view the *prediction* comes from the deployed accelerometer predictor
(``TRAPEZOID_ACCEL``); only the input interval and the truth source
change.

* **GT-segments view** — predictor reads each ground-truth interval
  directly. Truth = ``gt['height_diff_m']`` (barometer-derived).
  Tells us how good the predictor itself is when given perfect
  segmentation.

* **Matched-segmenter view** ("segmenter when correct") — predictor
  reads the segmenter's interval, restricted to predictions that
  overlap a GT (no false negatives in this view, since we are
  scoring per matched prediction). Truth = the matched GT row's
  ``height_diff_m``. Tells us how good the prediction is when the
  segmenter has actually identified a real ride, with the small edge
  inaccuracy a real detector produces.

* **All-segments-vs-barometer view** — predictor reads every
  segmenter prediction (including false positives). Truth =
  barometer-derived Δh integrated over the *same* predicted
  interval. Tells us how well the accelerometer-only predictor
  agrees with the barometer regardless of whether the segment
  corresponds to a true ride; FP segments get scored too because
  the barometer can answer "what altitude change actually happened
  on this interval" with no need for a label.

CLI: ``python scripts/pipeline_evaluation_report.py``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.data.loader import getExperimentData, list_experiments  # noqa: E402
from src.physics.barometric import pressure_to_altitude  # noqa: E402
from src.segmentation.algorithms.configTypes import (  # noqa: E402
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm,
)
from src.segmentation.algorithms.segmenter import Segmenter  # noqa: E402
from src.segmentation.algorithms.metrics.metrics import (  # noqa: E402
    DEFAULT_MIN_OVERLAP_FRAC, DEFAULT_MIN_OVERLAP_S, _intervals_match,
)
from src.segmentation.evaluate.evaluator import (  # noqa: E402
    _prepare_segmenter_input, _gt_to_interval_dicts,
    _segments_to_interval_dicts, _phone_model_from_metadata,
)
from src.prediction.algorithms.configTypes import (  # noqa: E402
    PREDICT_ALGORITHM_CONFIG, PredictAlgorithm,
)
from src.prediction.algorithms.predictor import Predictor  # noqa: E402


OUT_ROOT = REPO_ROOT / "docs" / "latex" / "figures" / "pipeline"
PRE_MS  = 3000   # mirrors src/prediction/evaluation/dataset.py default
POST_MS = 3000

CALIBRATION_PATH = (
    REPO_ROOT / "src" / "data" / "structuredData" / "test_results"
    / "prediction" / "train" / "calibration_trapezoid.json"
)


# --------------------------------------------------------------------------
# Per-ride prediction helpers
# --------------------------------------------------------------------------
def _slice_acc(acc: pd.DataFrame, lo_ms: float, hi_ms: float) -> pd.DataFrame:
    ts = acc["timestamp_ms"].astype(float).to_numpy()
    mask = (ts >= lo_ms) & (ts < hi_ms)
    return acc.loc[mask].reset_index(drop=True)


def _predict_for_window(
    predictor: Predictor,
    acc: pd.DataFrame,
    t_start_ms: float,
    t_end_ms: float,
    phone_model: str,
):
    """Run one prediction with PRE_MS/POST_MS pre/post windows.

    Returns ``(pred_dh, ci_half_width, accepted, quality_score)`` or
    ``None`` if the slice is too short to predict.
    """
    seg = _slice_acc(acc, t_start_ms, t_end_ms)
    if len(seg) < 5:
        return None
    pre  = _slice_acc(acc, t_start_ms - PRE_MS,  t_start_ms)
    post = _slice_acc(acc, t_end_ms,             t_end_ms + POST_MS)
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
    prs: pd.DataFrame,
    t_start_ms: float, t_end_ms: float,
    temperature_c: float | None = None,
    edge_k: int = 3,
) -> float | None:
    """Barometer-derived Δh over an arbitrary [t_start_ms, t_end_ms].

    Mirrors :func:`src.data.loader.pipeline._compute_raw_dh_per_segment`
    but works on any interval. Returns ``None`` when the barometer is
    absent or has fewer than 2 samples in the window.
    """
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


# --------------------------------------------------------------------------
# Match GT to predictions (overlap rule)
# --------------------------------------------------------------------------
def _classify(gt_rides, preds):
    """Return (gt_status_per_idx, pred_status_per_idx,
                gt_to_pred_idx, pred_to_gt_idx).

    gt_to_pred_idx[i] = predicted-interval index that overlaps GT i
    (only when len(matches)>=1 — for clean it's the unique partner; for
    merge/split it's an arbitrary overlapping prediction). -1 = missed.
    pred_to_gt_idx[j] = symmetric.
    """
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
def _run_experiment(
    exp_name: str,
    seg_cfg: SEGMENT_ALGORITHM_CONFIG,
    predictor: Predictor,
):
    """Process one experiment end-to-end. Returns (gt_records, seg_records).

    * gt_records: one row per GT ride. Columns include ``true_dh``
      (barometer-snapped GT) and ``oracle_pred_dh`` (predictor output
      on the GT interval).
    * seg_records: one row per *predicted* segment. Columns include
      ``pred_dh`` (predictor on segmenter interval), ``baro_dh``
      (barometer Δh over that same interval), ``status``,
      ``matched_gt_dh`` (GT Δh of the matched GT, NaN for FPs).
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

    # -- segmentation --
    data, _t0 = _prepare_segmenter_input(sensors, seg_cfg.algorithm)
    segmenter = Segmenter(seg_cfg)
    segments_df = segmenter.detect(data, phone_model=phone)
    gt_rides = _gt_to_interval_dicts(gt, t0_ms)
    preds = _segments_to_interval_dicts(segments_df)

    gt_status, pred_status, gt_match, pred_match = _classify(gt_rides, preds)

    # -- gt_height_diff_m, aligned to gt_rides (only up/down rows) --
    gt_dh_per_ride: list[float] = []
    for _, row in gt.iterrows():
        if row.get("type") not in ("up", "down"):
            continue
        gt_dh_per_ride.append(float(row.get("height_diff_m", float("nan"))))
    if len(gt_dh_per_ride) != len(gt_rides):
        gt_dh_per_ride = (gt_dh_per_ride
                          + [float("nan")] * len(gt_rides))[: len(gt_rides)]

    # -- predictor on every GT interval --
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
            "status": gt_status[i],
            "oracle_pred_dh":  oracle[0] if oracle else float("nan"),
            "oracle_ci":       oracle[1] if oracle else float("inf"),
            "oracle_accepted": oracle[2] if oracle else False,
        })

    # -- predictor on every predicted segment + barometer truth --
    seg_records: list[dict] = []
    for j, p in enumerate(preds):
        s_ms = t0_ms + p["t_start_s"] * 1000.0
        e_ms = t0_ms + p["t_end_s"]   * 1000.0
        out = _predict_for_window(predictor, acc, s_ms, e_ms, phone)
        baro_dh = _barometer_dh_over_interval(prs, s_ms, e_ms, temp_c)
        match_i = pred_match[j]
        matched_gt_dh = (gt_dh_per_ride[match_i]
                         if 0 <= match_i < len(gt_dh_per_ride) else float("nan"))
        seg_records.append({
            "exp": exp_name, "kind": exp_kind, "pred_idx": j,
            "type": p.get("type"),
            "duration_s": p["t_end_s"] - p["t_start_s"],
            "status": pred_status[j],
            "matched_gt_idx": match_i,
            "matched_gt_dh":  matched_gt_dh,
            "pred_dh":        out[0] if out else float("nan"),
            "pred_ci":        out[1] if out else float("inf"),
            "pred_accepted":  out[2] if out else False,
            "baro_dh":        baro_dh if baro_dh is not None else float("nan"),
        })
    return gt_records, seg_records


# --------------------------------------------------------------------------
# View construction
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


def _build_views(
    gt_df: pd.DataFrame, seg_df: pd.DataFrame,
    accepted_only: bool = False,
) -> dict[str, dict]:
    """Three error arrays, one per view, plus summaries.

    When ``accepted_only`` is True, restrict every view to predictions
    that the predictor's quality filter accepted (``oracle_accepted``
    for the GT view, ``pred_accepted`` for the matched / all views).
    This is the production deployment view --- a downstream consumer
    that trusts only accepted predictions.
    """
    if accepted_only:
        gt_df  = gt_df[gt_df["oracle_accepted"] == True]
        seg_df = seg_df[seg_df["pred_accepted"] == True]

    # View A — GT segments → predictor → vs gt['height_diff_m']
    gt_v = gt_df.dropna(subset=["true_dh", "oracle_pred_dh"])
    err_a = (gt_v["oracle_pred_dh"] - gt_v["true_dh"]).abs().to_numpy(dtype=float)

    # View B — Matched segmenter (clean) → predictor → vs matched GT dh
    matched = seg_df[
        (seg_df["status"] == "clean")
        & seg_df["matched_gt_dh"].notna()
        & seg_df["pred_dh"].notna()
    ]
    err_b = (matched["pred_dh"] - matched["matched_gt_dh"]).abs().to_numpy(dtype=float)

    # View C — All predicted segments → predictor → vs barometer Δh
    all_v = seg_df.dropna(subset=["pred_dh", "baro_dh"])
    err_c = (all_v["pred_dh"] - all_v["baro_dh"]).abs().to_numpy(dtype=float)

    return {
        "gt":      {"err": err_a, "summary": _summary(err_a)},
        "matched": {"err": err_b, "summary": _summary(err_b)},
        "all":     {"err": err_c, "summary": _summary(err_c)},
    }


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------
COLOR = {"gt": "#2980b9", "matched": "#27ae60", "all": "#e74c3c"}
LABEL = {
    "gt":      "GT segments  (predictor on every GT interval, vs gt $\\Delta h$)",
    "matched": "Matched segmenter  (predictor on clean-matched preds, vs matched GT $\\Delta h$)",
    "all":     "All segments  (predictor on every pred, vs barometer $\\Delta h$ on same window)",
}


def cdf_overlay(
    views: dict, title: str, out_path: Path,
    xmax: float | None = None,
) -> None:
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
    """Histogram of signed (pred - truth) for each view, x-axis clipped
    to ±5 m so the bulk distribution is visible (a handful of large-
    magnitude outliers on damped phones would otherwise dominate the
    range and hide the centre)."""
    XLIM = 5.0
    BIN_WIDTH = 0.5  # 0.5 m bins → 20 bins across [-5, +5]
    edges = np.arange(-XLIM, XLIM + BIN_WIDTH, BIN_WIDTH)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key in zip(axes, ("gt", "matched", "all")):
        arr = src_df[key]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        n_in   = int(((arr >= -XLIM) & (arr <= XLIM)).sum())
        n_out  = int((np.abs(arr) > XLIM).sum())
        ax.hist(arr, bins=edges, color=COLOR[key], alpha=0.65,
                edgecolor="white", density=True)
        ax.axvline(0, color="black", lw=0.5)
        ax.axvline(float(np.median(arr)), color="#c0392b", ls="--", lw=0.8,
                   label=f"median = {np.median(arr):+.2f} m")
        ax.set_xlim(-XLIM, XLIM)
        ax.set_xticks(np.arange(-XLIM, XLIM + 1.0, 1.0))
        ax.set_xlabel("signed error: pred $-$ truth (m)  "
                      "[$\\pm$5 m clipped]")
        ax.set_ylabel("density")
        ax.set_title(f"{key.upper()} view  "
                     f"(n={len(arr)}; "
                     f"{n_out} outside $\\pm$5 m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def scatter_three(
    gt_df: pd.DataFrame, seg_df: pd.DataFrame, out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    # GT view
    sub = gt_df.dropna(subset=["true_dh", "oracle_pred_dh"])
    axes[0].scatter(sub["true_dh"], sub["oracle_pred_dh"],
                    s=22, alpha=0.6, color=COLOR["gt"], edgecolor="none")
    # Matched view
    matched = seg_df[
        (seg_df["status"] == "clean")
        & seg_df["matched_gt_dh"].notna()
        & seg_df["pred_dh"].notna()
    ]
    axes[1].scatter(matched["matched_gt_dh"], matched["pred_dh"],
                    s=22, alpha=0.6, color=COLOR["matched"], edgecolor="none")
    # All view
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
    """FP-segment predicted-Δh histograms with 0.5 m bins and a zoomed
    x-axis (signed: ±5 m, absolute: 0–5 m). Outliers past the clip edge
    are tallied in the title rather than rescaling the whole axis."""
    fp = seg_df[seg_df["status"] == "fp"].dropna(subset=["pred_dh"])
    XLIM   = 5.0
    BIN_W  = 0.5
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
        axes[0].set_xticks(np.arange(-XLIM, XLIM + 1.0, 1.0))
        axes[0].set_xlabel("predicted $\\Delta h$ on FP segments (m)  "
                           "[0.5 m bins, $\\pm$5 m clipped]")
        axes[0].set_ylabel("count")
        axes[0].set_title(
            f"Signed $\\Delta h$ on FP predictions  "
            f"(n={len(sd)}; {n_out_s} outside $\\pm$5 m)"
        )
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].hist(ad, bins=edges_abs, color="#e74c3c", alpha=0.85,
                     edgecolor="white")
        axes[1].axvline(float(np.median(ad)), color="black", ls="--", lw=0.8,
                        label=f"median = {np.median(ad):.2f} m")
        axes[1].axvline(1.5, color="gray", ls="--", lw=0.6,
                        label="1.5 m one-floor")
        axes[1].set_xlim(0.0, XLIM)
        axes[1].set_xticks(np.arange(0.0, XLIM + 1.0, 1.0))
        axes[1].set_xlabel("$|$predicted $\\Delta h|$ on FP segments (m)  "
                           "[0.5 m bins, 5 m clipped]")
        axes[1].set_ylabel("count")
        axes[1].set_title(
            f"Absolute $\\Delta h$ on FPs  "
            f"(mean = {np.mean(ad):.2f} m; "
            f"{n_out_a} outside $5$ m)"
        )
        axes[1].grid(True, axis="y", alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def fp_vs_clean_predicted_dh(
    seg_df: pd.DataFrame, out_path: Path,
) -> None:
    """Side-by-side comparison of predicted-Δh distributions for FP vs
    clean segments, x clipped to ±5 m with 0.5 m bins. Densities so
    the shapes are comparable across the very different n's. Answers:
    *can a magnitude-only rule separate phantom rides from real ones?*
    """
    fp = seg_df[(seg_df["status"] == "fp") & seg_df["pred_dh"].notna()]
    clean = seg_df[(seg_df["status"] == "clean") & seg_df["pred_dh"].notna()]

    XLIM = 5.0
    BIN_W = 0.5
    edges = np.arange(-XLIM, XLIM + BIN_W, BIN_W)

    def _stats(arr: np.ndarray) -> tuple[float, float, int]:
        if arr.size == 0:
            return float("nan"), float("nan"), 0
        n_in = int(((arr >= -XLIM) & (arr <= XLIM)).sum())
        return float(np.median(arr)), float(np.median(np.abs(arr))), arr.size - n_in

    fp_arr   = fp["pred_dh"].to_numpy(dtype=float)
    clean_arr = clean["pred_dh"].to_numpy(dtype=float)
    fp_med, fp_amed, fp_out = _stats(fp_arr)
    cl_med, cl_amed, cl_out = _stats(clean_arr)

    # Two side-by-side panels with shared y-axis so the densities are
    # directly comparable. Plus a third overlay panel so you can read
    # the separation between the two populations at a glance.
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
        ax.set_xticks(np.arange(-XLIM, XLIM + 1.0, 1.0))
        ax.set_xlabel("predicted $\\Delta h$ on segment (m)  "
                      "[0.5 m bins, $\\pm$5 m clipped]")
        ax.set_ylabel("density")
        ax.set_title(
            f"{label} predictions  (n={arr.size}; "
            f"median $|\\Delta h|$={amed:.2f} m; "
            f"{n_out} outside)"
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    # Overlay panel (right): both densities on the same axes.
    ax = axes[2]
    ax.hist(fp_arr,    bins=edges, color="#7f8c8d", alpha=0.6,
            edgecolor="white", density=True, label=f"FP  (n={fp_arr.size})")
    ax.hist(clean_arr, bins=edges, color="#27ae60", alpha=0.5,
            edgecolor="white", density=True, label=f"clean  (n={clean_arr.size})")
    ax.axvline(0, color="black", ls="--", lw=0.5)
    ax.set_xlim(-XLIM, XLIM)
    ax.set_xticks(np.arange(-XLIM, XLIM + 1.0, 1.0))
    ax.set_xlabel("predicted $\\Delta h$ on segment (m)")
    ax.set_ylabel("density")
    ax.set_title("Overlay: FP vs clean")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def fp_predicted_altitude(seg_df: pd.DataFrame, out_path: Path) -> None:
    """Per-FP **predicted altitude** (not Δh) the system would report
    at the moment of the FP, taking the cumulative sum of every prior
    predicted Δh from the start of the recording. Plus, per-experiment,
    the total phantom altitude FPs alone contribute.

    Where the per-FP-Δh plots show \"how big is each phantom step\",
    this plot shows \"where does the system claim the user is when the
    FP fires\" --- the natural absolute-altitude analogue.
    """
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

    # Decide a sensible clip range: cover ~95 % of the data, snapped to
    # a multiple of the bin width.
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
    axes[0].set_xlabel(
        f"system's predicted altitude (m, integrated from $t = 0$)  "
        f"[{BIN_W:g}-m bins, $\\pm${int(XLIM)} m clipped]"
    )
    axes[0].set_ylabel("count")
    axes[0].set_title(
        f"Predicted altitude at the moment of each FP  "
        f"(n={len(arr)}; {n_out} outside)"
    )
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    # Right: per-experiment total altitude FPs alone contribute.
    items = sorted(fp_total.items(), key=lambda kv: kv[1])
    items = [(e, v) for e, v in items if v != 0.0]
    short = lambda nm: (nm.split("_")[0][:3] + "/" +
                        (nm.split("_")[1][:11] if len(nm.split("_")) > 1
                         else ""))
    names = [short(e) for e, _ in items]
    vals  = [v for _, v in items]
    colors = ["#27ae60" if v >= 0 else "#c0392b" for v in vals]
    y = np.arange(len(items))
    axes[1].barh(y, vals, color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names, fontsize=7)
    axes[1].axvline(0, color="black", lw=0.5)
    for yi, v in zip(y, vals):
        axes[1].text(v, yi, f"  {v:+.1f} m",
                     va="center", ha="left" if v >= 0 else "right",
                     fontsize=7)
    axes[1].set_xlabel("total altitude FPs would inject into a running tally (m)")
    axes[1].set_title(
        f"Per-experiment phantom-altitude bias from FPs  "
        f"(n={len(items)} experiments with FPs)"
    )
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def clean_predicted_altitude(
    gt_df: pd.DataFrame, seg_df: pd.DataFrame, out_path: Path,
) -> None:
    """Per *clean* (correctly matched) prediction, the system's running
    cumulative altitude at the end of the prediction --- and how that
    tracks the true cumulative altitude at the matched GT's end.

    Mirrors :func:`fp_predicted_altitude` for the success case: where
    the FP version asks \"what altitude does the system claim when it's
    wrong\", this asks \"what altitude does the system claim when it's
    right, and how close is that to the truth\".
    """
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
    arr_true = df["true_alt"].to_numpy()
    arr_err  = df["err"].to_numpy()

    AX_LIM_S = float(max(20.0, np.ceil(
        max(np.percentile(np.abs(arr_pred), 95),
            np.percentile(np.abs(arr_true), 95))
    )))
    BIN_W = 0.5 if AX_LIM_S <= 10 else (1.0 if AX_LIM_S <= 30 else 2.0)
    edges_alt = np.arange(-AX_LIM_S, AX_LIM_S + BIN_W, BIN_W)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    # ---- Panel 1: histogram of predicted cumulative altitude ----
    n_out_alt = int((np.abs(arr_pred) > AX_LIM_S).sum())
    axes[0].hist(arr_pred, bins=edges_alt, color=COLOR["matched"],
                 alpha=0.85, edgecolor="white")
    axes[0].axvline(0, color="black", ls="--", lw=0.5)
    axes[0].axvline(float(np.median(arr_pred)), color="#c0392b", ls="--",
                    lw=0.8, label=f"median = {np.median(arr_pred):+.2f} m")
    axes[0].set_xlim(-AX_LIM_S, AX_LIM_S)
    axes[0].set_xlabel(
        f"system's predicted altitude (m, integrated from $t = 0$)  "
        f"[{BIN_W:g}-m bins, $\\pm${int(AX_LIM_S)} m clipped]"
    )
    axes[0].set_ylabel("count")
    axes[0].set_title(
        f"Predicted altitude at each correct prediction  "
        f"(n={len(arr_pred)}; {n_out_alt} outside)"
    )
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    # ---- Panel 2: residual histogram ----
    ERR_LIM = 5.0
    BIN_E = 0.5
    edges_err = np.arange(-ERR_LIM, ERR_LIM + BIN_E, BIN_E)
    n_out_err = int((np.abs(arr_err) > ERR_LIM).sum())
    axes[1].hist(arr_err, bins=edges_err, color="#16a085", alpha=0.85,
                 edgecolor="white")
    axes[1].axvline(0, color="black", ls="--", lw=0.5)
    axes[1].axvline(float(np.median(arr_err)), color="#c0392b", ls="--",
                    lw=0.8, label=f"median = {np.median(arr_err):+.2f} m")
    axes[1].set_xlim(-ERR_LIM, ERR_LIM)
    axes[1].set_xlabel(
        "cumulative altitude error: pred $-$ true (m)  "
        f"[{BIN_E:g}-m bins, $\\pm${int(ERR_LIM)} m clipped]"
    )
    axes[1].set_ylabel("count")
    axes[1].set_title(
        f"Residual at each correct prediction  "
        f"(n={len(arr_err)}; {n_out_err} outside)"
    )
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def per_exp_mae_bar(
    gt_df: pd.DataFrame, seg_df: pd.DataFrame, out_path: Path,
) -> None:
    """Per-experiment MAE bars for the three views."""
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
            "exp": exp,
            "n_gt": len(gt_e),
            "gt":      float(np.mean(a)) if len(a) else float("nan"),
            "matched": float(np.mean(b)) if len(b) else float("nan"),
            "all":     float(np.mean(c)) if len(c) else float("nan"),
        })
    rows.sort(key=lambda r: r["n_gt"], reverse=True)
    short = lambda nm: (nm.split("_")[0][:3] + "/" +
                        (nm.split("_")[1][:11] if len(nm.split("_")) > 1 else ""))
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


def baro_vs_gt_consistency(seg_df: pd.DataFrame, out_path: Path) -> None:
    """Sanity scatter: barometer Δh vs matched-GT Δh on clean-matched
    intervals. Should hug the diagonal — the barometer is the truth
    used to derive the GT in the first place; this just checks the
    interval-level integration is consistent."""
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
# LaTeX emit
# --------------------------------------------------------------------------
def _fmt_m(v: float) -> str:
    return f"{v:.2f} m" if np.isfinite(v) else "---"


def _fmt_pct(v: float) -> str:
    return f"{100 * v:.1f}\\%" if np.isfinite(v) else "---"


def write_macros(
    out_path: Path,
    pooled: dict, train: dict, test: dict,
    n_fp: int,
    fp_median_signed: float, fp_median_abs: float, fp_mean_abs: float,
    pooled_acc: dict | None = None,
    train_acc: dict | None = None,
    test_acc: dict | None = None,
    accept_stats: dict | None = None,
    macro_prefix: str = "Pipe",
) -> None:
    """Emit pipeline macros. When the optional ``*_acc`` views are
    provided, also emit a parallel block of ``Acc``-suffixed macros
    (e.g. ``\\PipePooledGtMaeAcc``) plus accept-rate macros.

    ``macro_prefix`` lets a second run (e.g. with ``input_signal=
    a_mag_minus_g``) emit a parallel set of commands under a different
    LaTeX namespace (``\\PipeAmgPooledGtMae`` etc.) without colliding
    with the baseline."""
    P = macro_prefix
    lines = ["% Auto-generated by scripts/pipeline_evaluation_report.py"]

    def block(prefix: str, src: dict, acc: bool = False) -> list[str]:
        ll: list[str] = []
        suffix = "Acc" if acc else ""
        for view in ("gt", "matched", "all"):
            s = src[view]["summary"]
            tag = view.capitalize()
            ll.append(f"\\newcommand{{\\{P}{prefix}{tag}N{suffix}}}{{{s['n']}}}")
            ll.append(f"\\newcommand{{\\{P}{prefix}{tag}Mae{suffix}}}{{{s['mae']:.2f}}}")
            ll.append(f"\\newcommand{{\\{P}{prefix}{tag}Median{suffix}}}{{{s['median']:.2f}}}")
            ll.append(f"\\newcommand{{\\{P}{prefix}{tag}Rmse{suffix}}}{{{s['rmse']:.2f}}}")
            ll.append(f"\\newcommand{{\\{P}{prefix}{tag}Half{suffix}}}"
                      f"{{{100*s['p_within_0_5m']:.1f}\\%}}")
            ll.append(f"\\newcommand{{\\{P}{prefix}{tag}Floor{suffix}}}"
                      f"{{{100*s['p_within_1_5m']:.1f}\\%}}")
        return ll

    lines.extend(block("Pooled", pooled))
    lines.extend(block("Train",  train))
    lines.extend(block("Test",   test))
    lines.append(f"\\newcommand{{\\{P}FpN}}{{{n_fp}}}")
    lines.append(f"\\newcommand{{\\{P}FpMedianSigned}}{{{fp_median_signed:+.2f}}}")
    lines.append(f"\\newcommand{{\\{P}FpMedianAbs}}{{{fp_median_abs:.2f}}}")
    lines.append(f"\\newcommand{{\\{P}FpMeanAbs}}{{{fp_mean_abs:.2f}}}")

    if (pooled_acc is not None and train_acc is not None
            and test_acc is not None):
        lines.extend(block("Pooled", pooled_acc, acc=True))
        lines.extend(block("Train",  train_acc,  acc=True))
        lines.extend(block("Test",   test_acc,   acc=True))

    if accept_stats is not None:
        for k, v in accept_stats.items():
            lines.append(f"\\newcommand{{\\{P}{k}}}{{{v}}}")

    out_path.write_text("\n".join(lines) + "\n")


def write_headline_table(out_path: Path, label: str, src: dict) -> None:
    lines = [
        f"% Auto-generated headline pipeline metrics ({label})",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"View & $n$ & MAE & Median & RMSE & $\le$0.5\,m & $\le$1.5\,m \\",
        r"\midrule",
    ]
    rows = [
        ("GT segments (predictor on every GT)", "gt"),
        ("Matched segmenter (clean predictions)", "matched"),
        ("All segments vs barometer",            "all"),
    ]
    for name, key in rows:
        s = src[key]["summary"]
        lines.append(
            f"{name} & {s['n']} & {_fmt_m(s['mae'])} & "
            f"{_fmt_m(s['median'])} & {_fmt_m(s['rmse'])} & "
            f"{_fmt_pct(s['p_within_0_5m'])} & "
            f"{_fmt_pct(s['p_within_1_5m'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    out_path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline evaluation. Defaults reproduce "
                    "the historical run; flags below swap input signals "
                    "and output dirs for the |a|-g comparison.",
    )
    parser.add_argument("--seg-signal", default="a_vert",
                        choices=("a_vert", "a_mag_minus_g"),
                        help="input_signal override for the segmentation "
                             "matched filter (default: a_vert).")
    parser.add_argument("--pred-signal", default="a_vert",
                        choices=("a_vert", "a_mag_minus_g"),
                        help="input_signal override for the trapezoid "
                             "predictor (default: a_vert).")
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT,
                        help="Directory to write figures + macros into "
                             "(default: docs/latex/figures/pipeline).")
    parser.add_argument("--macro-prefix", default="Pipe",
                        help="LaTeX command prefix; use a unique value "
                             "(e.g. PipeAmg) when running a parallel "
                             "config to avoid \\newcommand collisions.")
    args = parser.parse_args(argv)

    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"writing artifacts under {out_root}")
    print(f"  seg input_signal={args.seg_signal}  "
          f"pred input_signal={args.pred_signal}  "
          f"macro prefix=\\{args.macro_prefix}...")

    seg_cfg = SEGMENT_ALGORITHM_CONFIG(
        algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
        overrides={"input_signal": args.seg_signal},
    )
    pred_cfg = PREDICT_ALGORITHM_CONFIG(
        algorithm=PredictAlgorithm.TRAPEZOID_ACCEL,
        overrides={"input_signal": args.pred_signal},
    )
    predictor = Predictor(pred_cfg)
    if CALIBRATION_PATH.exists():
        try:
            predictor.load_calibration(CALIBRATION_PATH)
            print(f"loaded calibration from {CALIBRATION_PATH.name}")
        except Exception as exc:
            print(f"  [warn] calibration load failed: {exc}")

    experiments = list_experiments(kind="all")
    print(f"running pipeline on {len(experiments)} experiments")
    t0 = time.time()
    all_gt: list[dict] = []
    all_seg: list[dict] = []
    for k, exp in enumerate(experiments, 1):
        try:
            gtr, segr = _run_experiment(exp, seg_cfg, predictor)
        except Exception as exc:
            print(f"  [{k:2d}/{len(experiments)}] {exp[:60]}: "
                  f"{type(exc).__name__}: {exc}")
            continue
        all_gt.extend(gtr)
        all_seg.extend(segr)
        n_fp = sum(1 for r in segr if r["status"] == "fp")
        print(f"  [{k:2d}/{len(experiments)}] {exp[:60]:<60} "
              f"gt={len(gtr)} pr={len(segr)} fp={n_fp} "
              f"({time.time() - t0:.1f}s)", flush=True)

    gt_df  = pd.DataFrame(all_gt)
    seg_df = pd.DataFrame(all_seg)
    gt_df.to_csv(out_root / "gt_records.csv", index=False)
    seg_df.to_csv(out_root / "seg_records.csv", index=False)

    # ----- aggregates -----
    pooled = _build_views(gt_df, seg_df)
    train  = _build_views(gt_df[gt_df["kind"] == "train"],
                           seg_df[seg_df["kind"] == "train"])
    test   = _build_views(gt_df[gt_df["kind"] == "test"],
                           seg_df[seg_df["kind"] == "test"])

    # FP statistics
    fp = seg_df[(seg_df["status"] == "fp") & seg_df["pred_dh"].notna()]
    if not fp.empty:
        fp_signed_med = float(np.median(fp["pred_dh"]))
        fp_abs = fp["pred_dh"].abs()
        fp_median_abs = float(np.median(fp_abs))
        fp_mean_abs   = float(np.mean(fp_abs))
    else:
        fp_signed_med = fp_median_abs = fp_mean_abs = 0.0

    # ----- figures -----
    print("\nrendering figures")
    cdf_overlay(pooled, "CDF of $|\\Delta h$ error$|$ — pooled (full range)",
                out_root / "cdf_pooled.png", xmax=10.0)
    cdf_overlay(pooled, "CDF (zoom to $0$--$3$ m)",
                out_root / "cdf_pooled_zoom.png", xmax=3.0)
    cdf_overlay(train,  "CDF of $|\\Delta h$ error$|$ — train split",
                out_root / "cdf_train.png",  xmax=10.0)
    cdf_overlay(test,   "CDF of $|\\Delta h$ error$|$ — test split (held-out)",
                out_root / "cdf_test.png",   xmax=10.0)
    bar_overall(pooled, train, test, out_root / "bar_mae_overall.png")
    per_exp_mae_bar(gt_df, seg_df, out_root / "per_exp_mae.png")
    scatter_three(gt_df, seg_df, out_root / "scatter_three.png")

    signed_pdf_pair(
        pooled,
        src_df={
            "gt": (gt_df.dropna(subset=["true_dh", "oracle_pred_dh"])
                   .assign(s=lambda d: d["oracle_pred_dh"] - d["true_dh"])
                   ["s"].to_numpy()),
            "matched": (
                seg_df[(seg_df["status"] == "clean")
                       & seg_df["matched_gt_dh"].notna()
                       & seg_df["pred_dh"].notna()]
                .assign(s=lambda d: d["pred_dh"] - d["matched_gt_dh"])
                ["s"].to_numpy()
            ),
            "all": (seg_df.dropna(subset=["pred_dh", "baro_dh"])
                    .assign(s=lambda d: d["pred_dh"] - d["baro_dh"])
                    ["s"].to_numpy()),
        },
        out_path=out_root / "signed_error_pdf.png",
    )
    fp_predicted_dh_hist(seg_df, out_root / "fp_predicted_dh.png")
    fp_predicted_altitude(seg_df, out_root / "fp_predicted_altitude.png")
    fp_vs_clean_predicted_dh(
        seg_df, out_root / "fp_vs_clean_dh.png",
    )
    clean_predicted_altitude(
        gt_df, seg_df, out_root / "clean_predicted_altitude.png",
    )
    baro_vs_gt_consistency(seg_df, out_root / "baro_vs_gt_sanity.png")

    # ----- accepted-only (post-quality-filter) parallel pass -----
    print("\nrendering accepted-only (post-quality-filter) figures")
    gt_acc  = gt_df[gt_df["oracle_accepted"] == True]
    seg_acc = seg_df[seg_df["pred_accepted"] == True]
    pooled_acc = _build_views(gt_df, seg_df, accepted_only=True)
    train_acc  = _build_views(
        gt_df[gt_df["kind"] == "train"],
        seg_df[seg_df["kind"] == "train"], accepted_only=True,
    )
    test_acc   = _build_views(
        gt_df[gt_df["kind"] == "test"],
        seg_df[seg_df["kind"] == "test"], accepted_only=True,
    )

    cdf_overlay(pooled_acc,
                "CDF — pooled, accepted-only (post quality filter)",
                out_root / "cdf_pooled_acc.png", xmax=10.0)
    cdf_overlay(pooled_acc, "CDF — pooled accepted (zoom)",
                out_root / "cdf_pooled_zoom_acc.png", xmax=3.0)
    cdf_overlay(train_acc,  "CDF — train, accepted-only",
                out_root / "cdf_train_acc.png",  xmax=10.0)
    cdf_overlay(test_acc,   "CDF — test, accepted-only",
                out_root / "cdf_test_acc.png",   xmax=10.0)
    bar_overall(pooled_acc, train_acc, test_acc,
                out_root / "bar_mae_overall_acc.png")
    per_exp_mae_bar(gt_acc, seg_acc, out_root / "per_exp_mae_acc.png")
    scatter_three(gt_acc, seg_acc, out_root / "scatter_three_acc.png")
    signed_pdf_pair(
        pooled_acc,
        src_df={
            "gt": (gt_acc.dropna(subset=["true_dh", "oracle_pred_dh"])
                   .assign(s=lambda d: d["oracle_pred_dh"] - d["true_dh"])
                   ["s"].to_numpy()),
            "matched": (
                seg_acc[(seg_acc["status"] == "clean")
                       & seg_acc["matched_gt_dh"].notna()
                       & seg_acc["pred_dh"].notna()]
                .assign(s=lambda d: d["pred_dh"] - d["matched_gt_dh"])
                ["s"].to_numpy()
            ),
            "all": (seg_acc.dropna(subset=["pred_dh", "baro_dh"])
                    .assign(s=lambda d: d["pred_dh"] - d["baro_dh"])
                    ["s"].to_numpy()),
        },
        out_path=out_root / "signed_error_pdf_acc.png",
    )
    fp_predicted_dh_hist(seg_acc, out_root / "fp_predicted_dh_acc.png")
    fp_predicted_altitude(seg_acc, out_root / "fp_predicted_altitude_acc.png")
    fp_vs_clean_predicted_dh(seg_acc, out_root / "fp_vs_clean_dh_acc.png")
    clean_predicted_altitude(
        gt_acc, seg_acc, out_root / "clean_predicted_altitude_acc.png",
    )

    # Accepted-only headline tables
    write_headline_table(out_root / "headline_pooled_acc.tex",
                         "pooled, accepted-only", pooled_acc)
    write_headline_table(out_root / "headline_train_acc.tex",
                         "train, accepted-only", train_acc)
    write_headline_table(out_root / "headline_test_acc.tex",
                         "test, accepted-only", test_acc)

    # FP statistics under the accepted-only filter
    fp_acc = seg_acc[(seg_acc["status"] == "fp")
                      & seg_acc["pred_dh"].notna()]
    n_fp_acc       = int(len(fp_acc))
    n_fp_total     = int((seg_df["status"] == "fp").sum())
    n_clean_total  = int((seg_df["status"] == "clean").sum())
    n_clean_acc    = int(((seg_df["status"] == "clean")
                          & (seg_df["pred_accepted"] == True)).sum())
    n_seg_total    = int(len(seg_df))
    n_seg_acc      = int(len(seg_acc))
    n_gt_total     = int(len(gt_df))
    n_gt_acc       = int(len(gt_acc))
    accept_stats = {
        "AcceptRateAll":     f"{100.0 * n_seg_acc / max(1, n_seg_total):.1f}\\%",
        "AcceptRateClean":   f"{100.0 * n_clean_acc / max(1, n_clean_total):.1f}\\%",
        "AcceptRateFp":      f"{100.0 * n_fp_acc / max(1, n_fp_total):.1f}\\%",
        "AcceptRateGt":      f"{100.0 * n_gt_acc / max(1, n_gt_total):.1f}\\%",
        "FpAcceptedN":       str(n_fp_acc),
        "FpRejectedN":       str(n_fp_total - n_fp_acc),
        "CleanAcceptedN":    str(n_clean_acc),
        "CleanRejectedN":    str(n_clean_total - n_clean_acc),
        "SegAcceptedN":      str(n_seg_acc),
        "SegTotalN":         str(n_seg_total),
        "GtAcceptedN":       str(n_gt_acc),
        "GtTotalN":          str(n_gt_total),
    }

    # ----- LaTeX outputs -----
    print("writing LaTeX tables and macros")
    write_macros(
        out_root / "results_macros.tex",
        pooled, train, test,
        n_fp=int((seg_df["status"] == "fp").sum()),
        fp_median_signed=fp_signed_med,
        fp_median_abs=fp_median_abs, fp_mean_abs=fp_mean_abs,
        pooled_acc=pooled_acc, train_acc=train_acc, test_acc=test_acc,
        accept_stats=accept_stats,
        macro_prefix=args.macro_prefix,
    )
    write_headline_table(out_root / "headline_pooled.tex", "pooled", pooled)
    write_headline_table(out_root / "headline_train.tex",  "train",  train)
    write_headline_table(out_root / "headline_test.tex",   "test",   test)

    # ----- summary -----
    print("\nsummary (pooled):")
    for view in ("gt", "matched", "all"):
        s = pooled[view]["summary"]
        print(f"  {view:8s}: n={s['n']:4d}  MAE={s['mae']:.3f} m  "
              f"median={s['median']:.3f} m  rmse={s['rmse']:.3f} m  "
              f"<=1.5m={100*s['p_within_1_5m']:.1f}%")
    print(f"  fps: n={int((seg_df['status'] == 'fp').sum())}  "
          f"|dh| median={fp_median_abs:.2f} m mean={fp_mean_abs:.2f} m")
    print(f"\nartifacts under: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
