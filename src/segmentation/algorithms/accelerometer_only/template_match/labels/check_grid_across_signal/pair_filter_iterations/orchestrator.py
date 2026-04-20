"""10-iteration autonomous exploration of pair-filter variants.

Keeps the production :mod:`pair_filter` module untouched — every variant
below is a pure function that shares the shape of
:func:`pair_filter.predict_pairs` but swaps in new acceptance / ranking
rules. The orchestrator loads the 22 training experiments once, runs
every variant against the winning :class:`DetectConfig` from the prior
sweep, and writes per-iteration metrics + diagnostic graphs + notes
into ``pair_filter_iterations/iter_NN_<slug>/``. A final ``README.md``
summarises progress across all 10 iterations.

Design goal of each iteration: attack the specific failure mode the
baseline sweep surfaced — 208/415 GT rides (50 %) are swallowed by a
pred that also covers other GTs (``gt_merged``). Every variant targets
a different way to break up these merges, or combines improvements
that worked alone.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import list_experiments, getExperimentData
from src.segmentation.algorithms.metrics import IntervalPredictionMetrics
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (
    detect, pair_filter,
)
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal.detect import (
    DetectConfig,
)
from src.segmentation.algorithms.accelerometer_only.template_match.fit_elevator_parameters.common import (
    LobeFit,
)
from dataclasses import asdict as dc_asdict


# Paths are anchored to this file so the script runs from any cwd.
_HERE = Path(__file__).resolve().parent
OUT_ROOT = _HERE  # labels/check_grid_across_signal/pair_filter_iterations/
BEST_CONFIG_PATH = _HERE.parent / "best_detect_config.json"


# ==========================================================================
# Parametric pair filter — superset of baseline logic
# ==========================================================================
def predict_pairs_variant(
    state: dict, cfg: DetectConfig,
    *,
    # Greedy ranking — subtract lambda * gap_seconds from the pair score.
    duration_penalty_lambda: float = 0.0,
    # Pre-accept filter: the middle of the pair must be quieter than
    # ``quiet_middle_ratio * A_abs`` on ``a_smooth``. Defence against
    # "super pairs" that swallow intermediate rides.
    require_quiet_middle: bool = False,
    quiet_middle_ratio: float = 0.5,
    # Mutual-nearest-neighbour gate: only consider a (+, -) pair if each
    # lobe is the other's nearest same-admissibility-window opposite.
    require_mutual_nearest: bool = False,
    # Acceptance: min of per-lobe R² ≥ joint_r2_thresh (tighter than
    # the default mean rule, which lets one great lobe drag a mediocre
    # one across the line).
    require_min_r2: bool = False,
) -> list[dict]:
    """Drop-in replacement for :func:`pair_filter.predict_pairs` with
    optional extra rules. Defaults collapse to the baseline."""
    from dataclasses import asdict as _asdict

    t = state["t"]
    a_smooth = state["a_smooth"]
    peaks = state["final_peaks"]
    signs = state["signs"]

    pos = [i for i in peaks if signs[i] > 0]
    neg = [i for i in peaks if signs[i] < 0]

    # Pre-compute mutual-nearest tables if requested.
    pos_to_near_neg: dict[int, int] = {}
    neg_to_near_pos: dict[int, int] = {}
    if require_mutual_nearest:
        def nearest(src: int, candidates: list[int]) -> int | None:
            best, best_d = None, float("inf")
            for j in candidates:
                dt = abs(float(t[j] - t[src]))
                if cfg.min_ride_s <= dt <= cfg.max_ride_s and dt < best_d:
                    best, best_d = j, dt
            return best

        for p in pos:
            j = nearest(p, neg)
            if j is not None:
                pos_to_near_neg[p] = j
        for n in neg:
            j = nearest(n, pos)
            if j is not None:
                neg_to_near_pos[n] = j

    dt_step = float(np.median(np.diff(t))) if t.size > 1 else 0.01

    candidates: list[tuple] = []

    def _try_pair(i1: int, i2: int, s1: float, s2: float) -> None:
        if i2 <= i1:
            return
        gap = float(t[i2] - t[i1])
        if gap < cfg.min_ride_s or gap > cfg.max_ride_s:
            return
        if require_mutual_nearest:
            if s1 > 0:
                if pos_to_near_neg.get(i1) != i2 or neg_to_near_pos.get(i2) != i1:
                    return
            else:
                if neg_to_near_pos.get(i1) != i2 or pos_to_near_neg.get(i2) != i1:
                    return
        res = pair_filter.joint_pair_score(a_smooth, t, i1, i2, s1, s2)
        if res is None:
            return
        score, W, f, A_abs, r2_1, r2_2, _heatmap_energy = res
        if require_min_r2:
            if min(r2_1, r2_2) < cfg.joint_r2_thresh:
                return
        else:
            if score < cfg.joint_r2_thresh:
                return
        if A_abs < cfg.min_pair_abs_a:
            return
        if require_quiet_middle:
            half_samples = max(1, int(round(W / dt_step)))
            mid_lo = i1 + half_samples
            mid_hi = i2 - half_samples
            if mid_hi > mid_lo:
                mid_max = float(np.max(np.abs(a_smooth[mid_lo:mid_hi])))
                if mid_max > quiet_middle_ratio * A_abs:
                    return
        candidates.append((score, gap, i1, i2, s1, W, f, A_abs, r2_1, r2_2))

    for i1 in pos:
        for i2 in neg:
            _try_pair(i1, i2, +1.0, -1.0)
    for i1 in neg:
        for i2 in pos:
            _try_pair(i1, i2, -1.0, +1.0)

    # Greedy with optional duration penalty.
    candidates.sort(
        key=lambda x: x[0] - duration_penalty_lambda * x[1],
        reverse=True,
    )
    used: set[int] = set()
    accepted_ranges: list[tuple[float, float]] = []
    accepted: list[tuple] = []
    for cand in candidates:
        _score, _gap, i1, i2, *_ = cand
        if i1 in used or i2 in used:
            continue
        ts, te = (t[i1], t[i2]) if t[i1] < t[i2] else (t[i2], t[i1])
        if any(not (te <= a_s or ts >= a_e) for a_s, a_e in accepted_ranges):
            continue
        used.add(i1); used.add(i2)
        accepted_ranges.append((ts, te))
        accepted.append(cand)
    accepted.sort(key=lambda x: x[2])

    predictions: list[dict] = []
    for idx, (score, _gap, i1, i2, s1, W, f, A_abs, r2_1, r2_2) in enumerate(accepted):
        t_start = float(t[i1])
        t_end = float(t[i2])
        ride_type = "up" if s1 > 0 else "down"
        lobe1 = LobeFit(t_c=t_start, a_peak=float(s1 * A_abs),
                        half_width_s=W, frac_flat=f, r2_local=r2_1)
        lobe2 = LobeFit(t_c=t_end, a_peak=float(-s1 * A_abs),
                        half_width_s=W, frac_flat=f, r2_local=r2_2)
        predictions.append({
            "index": idx, "ride_type": ride_type,
            "t_start_s": t_start, "t_end_s": t_end,
            "duration_s": t_end - t_start,
            "lobe1": _asdict(lobe1), "lobe2": _asdict(lobe2),
            "joint_r2_mean": float(score),
        })
    return predictions


# ==========================================================================
# The 10 variants
# ==========================================================================
# Every entry: (slug, description, kwargs-for-predict_pairs_variant).
VARIANTS: list[tuple[str, str, dict]] = [
    ("01_baseline",
     "Baseline — current pair_filter.predict_pairs (greedy by mean R²).",
     {}),
    ("02_dur_penalty_light",
     "Duration penalty λ=0.001: rank by score - 0.001·Δt to nudge short "
     "pairs ahead of long ones at similar scores.",
     {"duration_penalty_lambda": 0.001}),
    ("03_dur_penalty_medium",
     "Duration penalty λ=0.003: stronger nudge against super-pairs.",
     {"duration_penalty_lambda": 0.003}),
    ("04_dur_penalty_heavy",
     "Duration penalty λ=0.01: aggressive — a 100 s pair must beat a 10 s "
     "pair by ≥ 0.9 R² to outrank it.",
     {"duration_penalty_lambda": 0.01}),
    ("05_mutual_nearest",
     "Mutual nearest neighbour: only pair + with − (or vice versa) if "
     "each is the other's nearest admissible opposite. Structural attack "
     "on lobe-jumping.",
     {"require_mutual_nearest": True}),
    ("06_quiet_middle",
     "Quiet-middle constraint: reject if |a_smooth| between lobes ever "
     "exceeds 0.5·A_abs. A real ride's middle is cruise — nearly zero "
     "acceleration.",
     {"require_quiet_middle": True, "quiet_middle_ratio": 0.5}),
    ("07_min_r2",
     "Min R² acceptance: min(r2_1, r2_2) ≥ joint_r2_thresh instead of "
     "mean ≥ thresh. Forces both lobes to agree with the shared shape.",
     {"require_min_r2": True}),
    ("08_mutual_plus_dur",
     "Combine 05 + 03: mutual nearest pairing, then duration penalty "
     "λ=0.003 as tiebreaker.",
     {"require_mutual_nearest": True, "duration_penalty_lambda": 0.003}),
    ("09_quiet_plus_minr2",
     "Combine 06 + 07: quiet-middle constraint + both-lobe R² gate.",
     {"require_quiet_middle": True, "quiet_middle_ratio": 0.5,
      "require_min_r2": True}),
    ("10_combined_final",
     "Kitchen sink: mutual nearest + quiet middle + min R² + λ=0.003 "
     "duration penalty.",
     {"require_mutual_nearest": True, "require_quiet_middle": True,
      "quiet_middle_ratio": 0.5, "require_min_r2": True,
      "duration_penalty_lambda": 0.003}),
]


# ==========================================================================
# Harness
# ==========================================================================
def _extract_gt_rides(gt: pd.DataFrame, t0_ms: float) -> list[dict]:
    rides = []
    if gt is None or gt.empty:
        return rides
    for _, row in gt.iterrows():
        if row.get("type") not in ("up", "down"):
            continue
        rides.append({
            "type": row["type"],
            "t_start_s": (float(row["start_ms"]) - t0_ms) / 1000.0,
            "t_end_s":   (float(row["end_ms"]) - t0_ms) / 1000.0,
        })
    return rides


def _rerun_peaks(state: dict, cfg: DetectConfig) -> dict:
    nms_samples = max(1, int(round(cfg.nms_radius_s * state["fs"])))
    amp_gate = np.abs(state["best_A"]) >= cfg.min_peak_abs_a
    best_r2_gated = np.where(amp_gate, state["best_r2"], -np.inf)
    initial_peaks = detect._peak_pick(
        best_r2_gated, cfg.r2_peak_thresh, nms_samples,
    )
    final_peaks = detect._same_sign_nms(
        initial_peaks, best_r2_gated, state["signs"],
        state["t"], cfg.same_sign_min_gap_s,
    )
    return {
        **state,
        "best_r2_gated": best_r2_gated,
        "initial_peaks": initial_peaks,
        "final_peaks": final_peaks,
        "config": cfg,
    }


def load_experiments() -> list[dict]:
    """Load all 22 training exps + run the (W, f) sweep once each.

    The sweep is the expensive part; we cache the state dict so each
    variant only re-runs the cheap peak-pick + pair-filter stages."""
    names = list_experiments(kind="train")
    print(f"preparing {len(names)} experiments…", flush=True)
    exps: list[dict] = []
    t0 = time.time()
    for n in names:
        try:
            sensors, gt, _ = getExperimentData(n)
        except Exception as e:
            print(f"  [error] {n}: {e}", flush=True)
            continue
        state = detect.detect(sensors.get("ACC"), DetectConfig())
        if state is None:
            continue
        exps.append({
            "name": n,
            "state": state,
            "gt_rides": _extract_gt_rides(gt, state["t0_ms"]),
        })
        print(f"  [ok] {n} ({time.time() - t0:.1f}s)", flush=True)
    return exps


def evaluate_variant(
    exps: list[dict], cfg: DetectConfig, fn,
) -> tuple[IntervalPredictionMetrics, list[tuple[str, IntervalPredictionMetrics, list[dict]]], dict]:
    """Run ``fn`` (a variant of predict_pairs) on every exp and score.

    Returns (total, per_exp_with_preds, iou_metrics).
    """
    per_exp = []
    pooled_gt: list[dict] = []
    pooled_pred: list[dict] = []
    for e in exps:
        state = _rerun_peaks(e["state"], cfg)
        preds = fn(state, cfg)
        m = IntervalPredictionMetrics.from_intervals(e["gt_rides"], preds)
        per_exp.append((e["name"], m, preds))
        offset = (len(pooled_gt) + len(pooled_pred)) * 1e6 + 1e9
        pooled_gt.extend({**g, "t_start_s": g["t_start_s"] + offset,
                          "t_end_s":   g["t_end_s"] + offset} for g in e["gt_rides"])
        pooled_pred.extend({"t_start_s": p["t_start_s"] + offset,
                            "t_end_s":   p["t_end_s"] + offset} for p in preds)
    total = IntervalPredictionMetrics.sum(m for _, m, _ in per_exp)
    iou = IntervalPredictionMetrics.iou_f1(pooled_gt, pooled_pred, iou_threshold=0.5)
    return total, per_exp, iou


def _worst_merge_exp(per_exp: list[tuple[str, IntervalPredictionMetrics, list[dict]]]) -> str:
    """Exp with the worst merge count — best stress-test visual."""
    return max(per_exp, key=lambda x: x[1].gt_merged + x[1].pred_merged)[0]


def plot_timeline(
    exp_name: str, gt_rides: list[dict], preds: list[dict], out_path: Path,
    title: str,
) -> None:
    """Horizontal timeline of GT (green/red) and pred (blue/purple)
    intervals for one experiment. Overlap is what the eye should check.
    """
    if not gt_rides and not preds:
        return
    all_starts = [g["t_start_s"] for g in gt_rides] + [p["t_start_s"] for p in preds]
    all_ends = [g["t_end_s"] for g in gt_rides] + [p["t_end_s"] for p in preds]
    t_lo = min(all_starts) - 30 if all_starts else 0
    t_hi = max(all_ends) + 30 if all_ends else 1

    fig, ax = plt.subplots(figsize=(14, 3.2))
    # GT on row 1, predictions on row 0.
    for g in gt_rides:
        color = "#27ae60" if g["type"] == "up" else "#e74c3c"
        ax.barh(1, g["t_end_s"] - g["t_start_s"], left=g["t_start_s"],
                height=0.6, color=color, alpha=0.75, edgecolor="black", linewidth=0.3)
    for p in preds:
        color = "#1f3a5f" if p["ride_type"] == "up" else "#7d3c98"
        ax.barh(0, p["t_end_s"] - p["t_start_s"], left=p["t_start_s"],
                height=0.6, color=color, alpha=0.75, edgecolor="black", linewidth=0.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["pred", "GT"])
    ax.set_xlabel("t (s, ACC-local)")
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(-0.6, 1.6)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title(f"{title}\n{exp_name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_error_bars(
    per_exp: list[tuple[str, IntervalPredictionMetrics, list[dict]]],
    out_path: Path, title: str,
) -> None:
    """Per-exp stacked bar of clean / missed / merged / split / fp counts."""
    names = [n for n, _, _ in per_exp]
    clean = [m.clean for _, m, _ in per_exp]
    missed = [m.missed for _, m, _ in per_exp]
    merged = [m.gt_merged for _, m, _ in per_exp]
    split = [m.gt_split for _, m, _ in per_exp]
    fp = [m.fp for _, m, _ in per_exp]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(14, 5.2))
    bottom = np.zeros(len(names))
    for label, vals, color in (
        ("clean", clean,     "#27ae60"),
        ("missed", missed,   "#95a5a6"),
        ("merged", merged,   "#e74c3c"),
        ("split", split,     "#f39c12"),
        ("fp",   fp,         "#7d3c98"),
    ):
        ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.3)
        bottom = bottom + np.asarray(vals)
    ax.set_xticks(x)
    short = [n.split("_")[1][:12] + "…" + n.split("_")[-1] for n in names]
    ax.set_xticklabels(short, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("count (GT-side + pred-side errors)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_iter_notes(
    out_dir: Path, slug: str, description: str, kwargs: dict,
    total: IntervalPredictionMetrics, iou: dict,
    per_exp: list[tuple[str, IntervalPredictionMetrics, list[dict]]],
    worst_exp: str,
) -> None:
    r = total.rates()
    lines = [
        f"# Iteration: {slug}",
        "",
        f"**What changed:** {description}",
        "",
        f"**Variant kwargs:** `{kwargs}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---|",
        f"| clean | {total.clean} / {total.n_gt} |",
        f"| missed | {total.missed} |",
        f"| gt_merged | {total.gt_merged} |",
        f"| gt_split | {total.gt_split} |",
        f"| pred_merged | {total.pred_merged} |",
        f"| fp | {total.fp} |",
        f"| **f1_like** | **{r['f1_like']:.3f}** |",
        f"| **IoU-F1 @ 0.5** | **{iou['iou_f1@0.5']:.3f}** |",
        f"| recall | {r['recall']:.3f} |",
        f"| precision | {r['precision']:.3f} |",
        f"| mean IoU (matched) | {iou['iou_mean@0.5']:.3f} |",
        "",
        "## Per-exp breakdown",
        "",
        "| exp | gt | pred | clean | miss | merged | split | fp |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m, preds in per_exp:
        lines.append(
            f"| {name} | {m.n_gt} | {m.n_pred} | {m.clean} | {m.missed} "
            f"| {m.pred_merged} | {m.gt_split} | {m.fp} |"
        )
    lines += [
        "",
        "## Diagnostic plots",
        "",
        "- `errors_bar.png` — per-exp stacked breakdown of clean / missed / merged / split / fp.",
        f"- `timeline_{worst_exp[:40]}…png` — GT (top row) vs. pred (bottom row) intervals "
        f"for the exp with the worst merge count, to inspect the swallowing pattern.",
        "",
    ]
    (out_dir / "notes.md").write_text("\n".join(lines))


def run_all_iterations() -> list[dict]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    best = json.loads(BEST_CONFIG_PATH.read_text())
    cfg = DetectConfig(**best["config"])
    print(f"config for all iterations: {cfg}", flush=True)

    exps = load_experiments()

    summary: list[dict] = []
    for slug, description, kwargs in VARIANTS:
        iter_dir = OUT_ROOT / f"iter_{slug}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== iter {slug} ===", flush=True)
        print(f"    {description}", flush=True)

        def fn(state, c, _kw=kwargs):
            return predict_pairs_variant(state, c, **_kw)

        t0 = time.time()
        total, per_exp, iou = evaluate_variant(exps, cfg, fn)
        dt = time.time() - t0
        r = total.rates()
        print(
            f"    clean={total.clean}  miss={total.missed}  "
            f"merged={total.gt_merged}  split={total.gt_split}  fp={total.fp}  "
            f"| f1_like={r['f1_like']:.3f}  iou_f1={iou['iou_f1@0.5']:.3f}  "
            f"({dt:.1f}s)", flush=True,
        )

        worst = _worst_merge_exp(per_exp)
        worst_entry = next(x for x in per_exp if x[0] == worst)
        worst_gt = next(e["gt_rides"] for e in exps if e["name"] == worst)

        plot_timeline(
            worst, worst_gt, worst_entry[2],
            iter_dir / f"timeline_{worst[:40]}.png",
            f"iter {slug}: {description}",
        )
        plot_error_bars(per_exp, iter_dir / "errors_bar.png",
                        f"iter {slug} — per-exp error breakdown")

        (iter_dir / "metrics.json").write_text(json.dumps({
            "slug": slug, "description": description, "kwargs": kwargs,
            "totals": total.as_dict(), "iou": iou,
            "per_exp": [(n, m.as_dict()) for n, m, _ in per_exp],
            "elapsed_s": dt,
        }, indent=2))
        write_iter_notes(iter_dir, slug, description, kwargs, total, iou, per_exp, worst)

        summary.append({
            "slug": slug, "description": description,
            "clean": total.clean, "missed": total.missed,
            "gt_merged": total.gt_merged, "gt_split": total.gt_split,
            "fp": total.fp,
            "f1_like": r["f1_like"], "iou_f1": iou["iou_f1@0.5"],
            "recall": r["recall"], "precision": r["precision"],
            "mean_iou": iou["iou_mean@0.5"],
        })

    return summary


def write_summary_readme(summary: list[dict]) -> None:
    df = pd.DataFrame(summary)
    # Plot f1_like + iou_f1 progression.
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    ax.plot(x, df["f1_like"], "o-", color="#2980b9", label="f1_like (four-mode composite)")
    ax.plot(x, df["iou_f1"], "s-", color="#e67e22", label="IoU-F1 @ 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(df["slug"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("Pair-filter iteration scores")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, max(0.3, df[["f1_like", "iou_f1"]].to_numpy().max() * 1.15))
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "progress.png", dpi=120)
    plt.close(fig)

    best_idx = int(df["f1_like"].idxmax())
    best = df.iloc[best_idx]
    lines = [
        "# Pair-Filter Iteration Log",
        "",
        "Ten autonomous iterations on the pair-filter stage of the ",
        "`check_grid_across_signal/` trapezoid detector. Every iteration ",
        "keeps the detection stage frozen at the winning DetectConfig ",
        "from the earlier hyperparameter sweep ",
        "(`elevator_reports/best_detect_config.json`) and only varies the ",
        "pair-filter algorithm.",
        "",
        "## Why we're here",
        "",
        "Baseline sweep identified the dominant failure mode: `gt_merged` ",
        "= 208 / 415 (50 %). A single prediction often spans several GT ",
        "rides because its two lobes — one true take-off and one true ",
        "landing from *different* rides — happen to share a template ",
        "shape. Parameter tuning alone can't fix this; the pair-filter's ",
        "greedy rule needs structural changes.",
        "",
        "## Summary table (sorted by run order)",
        "",
        "| slug | clean | miss | merged | split | fp | f1_like | IoU-F1 | recall | precision |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['slug']} | {row['clean']} | {row['missed']} | "
            f"{row['gt_merged']} | {row['gt_split']} | {row['fp']} | "
            f"{row['f1_like']:.3f} | {row['iou_f1']:.3f} | "
            f"{row['recall']:.3f} | {row['precision']:.3f} |"
        )
    lines += [
        "",
        "## Progress chart",
        "",
        "![iteration progression](progress.png)",
        "",
        "## Winner",
        "",
        f"**`{best['slug']}`** — {best['description']}",
        "",
        f"* f1_like = **{best['f1_like']:.3f}** (baseline: {summary[0]['f1_like']:.3f}, ",
        f"  Δ = {best['f1_like'] - summary[0]['f1_like']:+.3f})",
        f"* IoU-F1 @ 0.5 = **{best['iou_f1']:.3f}** (baseline: {summary[0]['iou_f1']:.3f}, ",
        f"  Δ = {best['iou_f1'] - summary[0]['iou_f1']:+.3f})",
        f"* clean = {best['clean']} / 415 GTs (recall {best['recall']:.1%})",
        f"* gt_merged = {best['gt_merged']} (baseline {summary[0]['gt_merged']}, ",
        f"  Δ = {best['gt_merged'] - summary[0]['gt_merged']:+d})",
        "",
        "## Per-iteration notes",
        "",
    ]
    for row in summary:
        lines.append(f"### {row['slug']}")
        lines.append("")
        lines.append(row["description"])
        lines.append("")
        lines.append(
            f"clean={row['clean']}  miss={row['missed']}  "
            f"merged={row['gt_merged']}  split={row['gt_split']}  "
            f"fp={row['fp']}  | f1_like={row['f1_like']:.3f}  "
            f"IoU-F1={row['iou_f1']:.3f}"
        )
        lines.append("")
        lines.append(f"See `iter_{row['slug']}/notes.md` for the per-exp breakdown ")
        lines.append("and diagnostic plots.")
        lines.append("")
    (OUT_ROOT / "README.md").write_text("\n".join(lines))
    print(f"\nwrote {OUT_ROOT / 'README.md'}", flush=True)


def main() -> int:
    summary = run_all_iterations()
    write_summary_readme(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
