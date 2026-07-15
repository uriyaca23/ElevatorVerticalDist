"""
Hyperparameter tuner for the segmentation and prediction stages.

Sweeps a curated grid of hyperparameters over ALL experiments (test + train
pooled), scoring every candidate with building-grouped k-fold cross-validation
on CLEAR segments only, then writes the winning params into the live
``config.json`` of each stage (with a timestamped backup + a printed diff).

Run from the project root with the venv active:

    source venv/bin/activate

    # tune both stages, write winners into config.json (backup + diff):
    python scripts/tune_hyperparameters.py --stage both

    # tune just one stage:
    python scripts/tune_hyperparameters.py --stage segmentation
    python scripts/tune_hyperparameters.py --stage prediction

    # preview only — rank configs + print the diff, but DON'T touch config.json:
    python scripts/tune_hyperparameters.py --stage both --dry-run

    # knobs:
    #   --folds 5                 number of k-fold splits (default 5)
    #   --grid-seg  tuning/grid_segmentation.json    editable param grid
    #   --grid-pred tuning/grid_prediction.json      editable param grid
    #   --min-coverage 0.90       required calibrated coverage for prediction
    #   --min-accept-rate 0.40    floor on clean accept-rate (prediction)
    #   --max-configs 0           cap grid size (0 = no cap)
    #   --include-corrupted       keep *__corrupted* experiments (off by default)
    #   --out-dir tuning          where result CSVs / backups / calibration go

What you get out of it
----------------------
0. ``<out-dir>/segment_qc/<experiment>.png`` — at the START of the
   segmentation run, one PNG per experiment showing the ``|a|-g`` signal
   the detector scores with the GT up/down ride intervals shaded behind it
   (same colors as src/data/gt_editor.py). A timing sanity check that the
   GT lines up with the acceleration pulses on the SAME resampled time
   domain the detector sees. Disable with ``--no-qc-plots``.
1. ``<out-dir>/results_segmentation.csv`` and
   ``<out-dir>/results_prediction_<algo>.csv`` — every candidate config ranked,
   with the k-fold-averaged metrics (and per-fold spread).
2. A console summary: the winning config per stage/algorithm and the
   improvement vs the current defaults.
3. Updated ``config.json`` files (backed up first; skipped under --dry-run):
     - pyramidElevatorDist/segmentation/algorithms/config.json   (key: acc_template_match)
     - pyramidElevatorDist/prediction/algorithms/config.json      (keys: zupt_accel, trapezoid_accel)
   These are the live source of truth — every Segmenter()/Predictor() reads
   them via load_params(), so the new values take effect on the next run.
4. Refit conformal calibration for the winning prediction configs:
   ``<out-dir>/calibration_<algo>.json`` (fit on all clear samples).

Note on k-fold here: predictions/detections depend only on the config, so each
candidate is run over the data ONCE; the folds only change how scores are
aggregated. For PREDICTION the conformal CI is genuinely refit per fold and
coverage measured on the held-out fold (an honest generalization estimate).
For SEGMENTATION the detector has no learned parameters, so the folds give a
stability estimate (per-fold spread) rather than train/val separation — the
grid search itself is the only fitting, and Option A intentionally pools all
data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless: only ever writes PNGs, never shows a window
import matplotlib.pyplot as plt  # noqa: E402

# Allow running as a plain script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import (  # noqa: E402
    getExperimentData,
    load_experiment_index,
    resolve_experiments,
)
from src.data.loadFromDB import LoadedSignal  # noqa: E402
from src.data.load_data import enrich_loaded  # noqa: E402
from pyramidElevatorDist.utils.accelerometer_utils import vertical_accel_magnitude  # noqa: E402
from pyramidElevatorDist.segmentation.algorithms.configTypes import (  # noqa: E402
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
)
from pyramidElevatorDist.segmentation.algorithms.configTypes import (  # noqa: E402
    DEFAULT_CONFIG_PATH as SEG_CONFIG_PATH,
)
from pyramidElevatorDist.segmentation.algorithms.segmenter import Segmenter  # noqa: E402
from pyramidElevatorDist.segmentation.algorithms.metrics import IntervalPredictionMetrics  # noqa: E402

from pyramidElevatorDist.prediction.algorithms import (  # noqa: E402
    PREDICT_ALGORITHM_CONFIG,
    PredictAlgorithm,
    Predictor,
    ZuptAccelConfig,
)
from pyramidElevatorDist.prediction.algorithms.accelerometer_only.trapezoid_accel import (  # noqa: E402
    TrapezoidAccelConfig,
)
from pyramidElevatorDist.prediction.algorithms.configTypes import (  # noqa: E402
    DEFAULT_CONFIG_PATH as PRED_CONFIG_PATH,
)
from src.prediction.evaluation.dataset import load_all_segments  # noqa: E402
from src.prediction.evaluation.runner import run_predictions  # noqa: E402
from pyramidElevatorDist.utils.conformal import ConformalCalibrator  # noqa: E402


# ---------------------------------------------------------------------------
# Default curated grids (editable — written to disk on first run if absent).
# Keep these modest: every value multiplies the number of full passes. Widen
# the JSON files once you've seen which params actually move the metric.
# ---------------------------------------------------------------------------
DEFAULT_GRID_SEGMENTATION: dict[str, list] = {
    # Peak-pick / pair acceptance — the highest-impact knobs of the matched
    # filter (see TemplateMatchConfig docstrings in configTypes.py).
    "r2_peak_thresh":   [0.35, 0.40, 0.45],
    "joint_r2_thresh":  [0.88, 0.90, 0.92],
    "min_pair_abs_a":   [0.25, 0.30, 0.35],
    "segment_pad_eps_s": [0.20, 0.25, 0.30],
}

DEFAULT_GRID_PREDICTION: dict[str, dict[str, list]] = {
    "zupt_accel": {
        "active_threshold_m_s2": [0.08, 0.10, 0.12],
        "quality_score_reject":  [5.0, 6.0, 7.0],
        "min_displacement_m":    [0.3, 0.4, 0.5],
        "alpha":                 [0.06, 0.08, 0.10],
    },
    "trapezoid_accel": {
        "smooth_sec":           [0.30, 0.40, 0.50],
        "quality_score_reject": [5.0, 6.0, 7.0],
        "min_r2_short":         [0.30, 0.35, 0.40],
        "alpha":                [0.06, 0.08, 0.10],
    },
}

_PRED_ALGO_ENUM = {
    "zupt_accel": PredictAlgorithm.ZUPT_ACCEL,
    "trapezoid_accel": PredictAlgorithm.TRAPEZOID_ACCEL,
}
_PRED_ALGO_CONFIG_CLS = {
    "zupt_accel": ZuptAccelConfig,
    "trapezoid_accel": TrapezoidAccelConfig,
}


# ---------------------------------------------------------------------------
# Experiment selection + building-grouped folds
# ---------------------------------------------------------------------------
def select_experiments(include_corrupted: bool) -> list[str]:
    """All experiments (train + test pooled), minus corrupted variants."""
    exps = resolve_experiments(kind="all")
    if not include_corrupted:
        kept = [e for e in exps if "corrupted" not in e.lower()]
        dropped = sorted(set(exps) - set(kept))
        if dropped:
            print(f"[select] excluding {len(dropped)} corrupted variant(s):")
            for d in dropped:
                print(f"           - {d}")
        exps = kept
    return exps


def build_building_folds(
    experiments: list[str], k: int, index: dict,
) -> tuple[int, dict[str, int]]:
    """Assign whole buildings to k folds, balancing experiment count.

    Returns ``(k_effective, fold_of_experiment)``. Grouping by building
    (metadata ``location``) keeps every recording from one building in the
    same fold, so a config is never validated on a building it was scored
    against — mirroring the cross-building test the project cares about.
    """
    by_building: dict[str, list[str]] = defaultdict(list)
    for e in experiments:
        loc = str((index.get(e, {}) or {}).get("location", "") or "").strip()
        by_building[loc or "(unknown)"].append(e)

    buildings = sorted(by_building, key=lambda b: len(by_building[b]), reverse=True)
    k_eff = max(1, min(k, len(buildings)))
    fold_load = [0] * k_eff
    fold_of: dict[str, int] = {}
    for b in buildings:
        j = min(range(k_eff), key=lambda i: fold_load[i])
        for e in by_building[b]:
            fold_of[e] = j
        fold_load[j] += len(by_building[b])

    print(f"[folds] {len(buildings)} building(s) -> {k_eff} fold(s); "
          f"experiments per fold: {fold_load}")
    if k_eff < k:
        print(f"[folds] note: only {len(buildings)} buildings available, "
              f"so k was reduced from {k} to {k_eff}.")
    return k_eff, fold_of


def load_grid(path: Path, default: dict) -> dict:
    """Load a grid JSON, writing the curated default if the file is absent."""
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(default, indent=2))
    print(f"[grid] wrote default grid to {path} (edit it to change the search)")
    return default


def grid_combos(grid: dict[str, list], max_configs: int) -> list[dict]:
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]
    if max_configs and len(combos) > max_configs:
        print(f"[grid] capping {len(combos)} combos at --max-configs={max_configs}")
        combos = combos[:max_configs]
    return combos


def _fold_mean_std(per_fold: list[float]) -> tuple[float, float]:
    vals = np.asarray([v for v in per_fold if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std())


# ===========================================================================
# SEGMENTATION
# ===========================================================================
def _prepare_acc_input(sensors: dict, name: str) -> tuple[pd.DataFrame, float] | None:
    """Shape an experiment's ACC into the detector's input frame.

    Routes the raw ACC through ``enrich_loaded(resample=True)`` — exactly
    as ``src/data/gt_editor.py`` and the live pipeline do — so the tuner
    scores the detector on the SAME 50 Hz, gap-split time domain every
    other tool uses, and ``t0`` is anchored on the *resampled* first sample
    (not the raw one). Without this, a trimmed sparse leading region would
    shift this script's time origin relative to gt_editor, making the GT
    bands appear offset (see the gt_editor.load_experiment comment).
    """
    acc = sensors.get("ACC")
    if acc is None or acc.empty:
        return None
    base = LoadedSignal(
        acc=acc, source=name, meta={},
        prs=sensors.get("PRS"), gyr=sensors.get("GYR"),
        mag=sensors.get("MAG"), ori=sensors.get("ORI"),
    )
    acc = enrich_loaded(base, resample=True).acc
    if acc is None or acc.empty:
        return None
    t0_ms = float(acc["timestamp_ms"].iloc[0])
    df = pd.DataFrame({
        "timestamp_ms": acc["timestamp_ms"].to_numpy(dtype=float),
        "x": acc["x"].to_numpy(dtype=float),
        "y": acc["y"].to_numpy(dtype=float),
        "z": acc["z"].to_numpy(dtype=float),
    })
    return df, t0_ms


def _gt_rides(gt: pd.DataFrame, t0_ms: float) -> tuple[list[dict], list[dict]]:
    """Split GT rides into (clear, unclear) center-interval dicts."""
    clear: list[dict] = []
    unclear: list[dict] = []
    if gt is None or gt.empty:
        return clear, unclear
    for _, row in gt.iterrows():
        if row.get("type") not in ("up", "down"):
            continue
        d = {
            "type": row["type"],
            "t_start_s": (float(row["start_ms"]) - t0_ms) / 1000.0,
            "t_end_s": (float(row["end_ms"]) - t0_ms) / 1000.0,
        }
        sc = row.get("signalClearRecording", True)
        (clear if (bool(sc) if sc is not None else True) else unclear).append(d)
    return clear, unclear


def _segments_to_intervals(segments: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    if segments is None or segments.empty:
        return out
    for _, row in segments.iterrows():
        s_lo, s_hi = row["start_ci"]
        e_lo, e_hi = row["end_ci"]
        out.append({
            "t_start_s": 0.5 * (float(s_lo) + float(s_hi)),
            "t_end_s": 0.5 * (float(e_lo) + float(e_hi)),
            "type": row.get("type"),
        })
    return out


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) - max(a0, b0) > 0.0


def _drop_preds_in_unclear(preds: list[dict], unclear: list[dict]) -> list[dict]:
    """Drop predictions overlapping an UNCLEAR GT ride so the detector is
    neither rewarded nor penalised for those regions (we score clear rides
    only, but a detection landing on an unclear ride is not a true FP)."""
    if not unclear:
        return preds
    kept = []
    for p in preds:
        if any(_overlaps(p["t_start_s"], p["t_end_s"], u["t_start_s"], u["t_end_s"])
               for u in unclear):
            continue
        kept.append(p)
    return kept


def preload_segmentation(experiments: list[str]) -> dict[str, dict]:
    """Load + shape each experiment's ACC input and GT once (reused across
    every candidate config — only detection re-runs per config)."""
    cache: dict[str, dict] = {}
    for name in experiments:
        try:
            sensors, gt, meta = getExperimentData(name)
        except Exception as exc:
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
            continue
        prepared = _prepare_acc_input(sensors, name)
        if prepared is None or len(prepared[0]) < 2:
            print(f"  [skip] {name}: no usable ACC")
            continue
        df, t0_ms = prepared
        clear, unclear = _gt_rides(gt, t0_ms)
        if not clear:
            print(f"  [skip] {name}: no clear up/down rides")
            continue
        cache[name] = {
            "df": df,
            "phone": str((meta or {}).get("phone", "")),
            "clear": clear,
            "unclear": unclear,
        }
    return cache


# GT band colors mirror src/data/gt_editor.py (TYPE_COLORS) so the QC plot
# reads the same as the editor GUI: up=green, down=red, outside=grey.
_GT_TYPE_COLORS = {"up": "#2ca02c", "down": "#d62728", "outside": "#b8b8b8"}


def save_segment_qc_plots(cache: dict[str, dict], out_dir: Path) -> Path:
    """Write one PNG per experiment: the ``|a|-g`` residual the matched
    filter scores, with the GT up/down ride intervals shaded behind it.

    A timing sanity check — it confirms the GT bands line up with the
    acceleration pulses on the SAME resampled time domain the detector
    sees (the recurring "are we in the same time domain?" question). The
    signal and GT-band colors mirror ``src/data/gt_editor.py`` so this
    reads identically to the editor GUI. Clear rides (what the tuner
    actually scores) are solid; unclear rides are hatched + faded so you
    can tell at a glance which intervals count.
    """
    qc_dir = out_dir / "segment_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    for name, entry in cache.items():
        df = entry["df"]
        ts = df["timestamp_ms"].to_numpy(dtype=float)
        t = (ts - ts[0]) / 1000.0  # seconds since the resampled first sample
        sig = vertical_accel_magnitude(
            df["x"].to_numpy(dtype=float),
            df["y"].to_numpy(dtype=float),
            df["z"].to_numpy(dtype=float),
        )
        fig, ax = plt.subplots(figsize=(18, 3.2))
        ax.plot(t, sig, color="tab:blue", lw=0.5, zorder=2)
        ax.axhline(0.0, color="gray", lw=0.4, ls="--", alpha=0.5)
        seen: set[str] = set()
        for grp, rides, alpha, hatch in (
            ("clear", entry["clear"], 0.25, None),
            ("unclear", entry["unclear"], 0.12, "//"),
        ):
            for r in rides:
                c = _GT_TYPE_COLORS.get(str(r["type"]), "#cccccc")
                lbl = f"{r['type']} ({grp})"
                ax.axvspan(
                    r["t_start_s"], r["t_end_s"], color=c, alpha=alpha,
                    hatch=hatch, zorder=0,
                    label=lbl if lbl not in seen else None,
                )
                seen.add(lbl)
        ax.set_xlabel("time (s since first resampled sample)")
        ax.set_ylabel("|a|−g (m/s²)")
        ax.set_title(
            f"{name}   ({len(entry['clear'])} clear / "
            f"{len(entry['unclear'])} unclear up·down rides)",
            fontsize=9, loc="left",
        )
        ax.grid(True, alpha=0.3)
        if seen:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)
        fig.tight_layout()
        fig.savefig(qc_dir / f"{name}.png", dpi=110)
        plt.close(fig)
    print(f"[qc] wrote {len(cache)} segment QC plot(s) -> {qc_dir}")
    return qc_dir


def score_segmentation_config(
    overrides: dict, cache: dict[str, dict], fold_of: dict[str, int], k: int,
) -> dict:
    """Detect on every experiment once, aggregate f1_like per fold."""
    cfg = SEGMENT_ALGORITHM_CONFIG(
        algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH, overrides=overrides,
    )
    segmenter = Segmenter(cfg)

    fold_metrics: list[list[IntervalPredictionMetrics]] = [[] for _ in range(k)]
    for name, entry in cache.items():
        f = fold_of.get(name, 0)
        try:
            segments = segmenter.detect(entry["df"], phone_model=entry["phone"])
        except Exception:
            preds: list[dict] = []
        else:
            preds = _drop_preds_in_unclear(
                _segments_to_intervals(segments), entry["unclear"],
            )
        m = IntervalPredictionMetrics.from_intervals(entry["clear"], preds)
        fold_metrics[f].append(m)

    per_fold_f1: list[float] = []
    for ms in fold_metrics:
        if not ms:
            continue
        total = IntervalPredictionMetrics.sum(ms)
        per_fold_f1.append(total.score())
    mean_f1, std_f1 = _fold_mean_std(per_fold_f1)
    return {
        "mean_f1_like": mean_f1,
        "std_f1_like": std_f1,
        "n_folds_scored": len(per_fold_f1),
    }


def tune_segmentation(
    cache: dict, fold_of: dict, k: int, grid: dict, max_configs: int,
) -> pd.DataFrame:
    combos = grid_combos(grid, max_configs)
    print(f"\n[segmentation] {len(combos)} configs x {len(cache)} experiments")
    rows: list[dict] = []
    t0 = time.time()
    for i, combo in enumerate(combos):
        res = score_segmentation_config(combo, cache, fold_of, k)
        rows.append({**combo, **res})
        if i == 0 or (i + 1) % 5 == 0 or i == len(combos) - 1:
            print(f"  [{i+1:4d}/{len(combos)}] mean_f1_like={res['mean_f1_like']:.3f}"
                  f" (+/-{res['std_f1_like']:.3f})  [{time.time()-t0:.0f}s]",
                  flush=True)
    df = (pd.DataFrame(rows)
          .sort_values("mean_f1_like", ascending=False)
          .reset_index(drop=True))
    return df


# ===========================================================================
# PREDICTION
# ===========================================================================
def preload_prediction(experiments: list[str]) -> list:
    """All CLEAR elevator segments with finite ground-truth Δh, across every
    experiment (folds applied later by the record's building)."""
    records = load_all_segments(experiments=experiments)
    clear = [
        r for r in records
        if r.signal_clear and np.isfinite(r.true_dh)
    ]
    print(f"[prediction] {len(records)} segments loaded; "
          f"{len(clear)} clear with finite Δh kept")
    return clear


def _predict_rec_rows(
    algo_key: str, pred_overrides: dict, records: list, fold_of: dict[str, int],
) -> tuple[list[dict], float, float, float]:
    """Run predictions once for a (non-alpha) config and extract the
    per-record quantities that are invariant to the conformal multiplier.

    Returns ``(rec_rows, ci_floor, ci_cap, default_alpha)``.
    """
    algo_enum = _PRED_ALGO_ENUM[algo_key]
    cfg = PREDICT_ALGORITHM_CONFIG(algorithm=algo_enum, overrides=pred_overrides)
    params = cfg.load_params()
    algo_cfg = _PRED_ALGO_CONFIG_CLS[algo_key](**params)
    ci_floor = float(getattr(algo_cfg, "ci_absolute_floor_m", 0.5))
    ci_cap = float(getattr(algo_cfg, "ci_absolute_cap_m", 60.0))
    default_alpha = float(getattr(algo_cfg, "alpha", 0.08))

    predictor = Predictor(cfg)
    preds = run_predictions(predictor, records)
    rec_rows = [{
        "fold": fold_of.get(rp.record.exp_name, 0),
        "sigma": float(rp.output.theoretical_sigma),
        "abs_err": float(abs(rp.output.height_diff - rp.record.true_dh)),
        "accepted": bool(rp.output.accepted),
    } for rp in preds]
    return rec_rows, ci_floor, ci_cap, default_alpha


def _score_with_alpha(
    rec_rows: list[dict], k: int, alpha: float, ci_floor: float, ci_cap: float,
    use_accepted_subset: bool, min_coverage: float, min_accept: float,
) -> dict:
    """k-fold coverage / MAE / accept-rate for a fixed prediction set at a
    given conformal ``alpha`` (cheap: no re-prediction).

    Trapezoid is scored on the ACCEPTED-clean subset (predictions the quality
    filter trusts; rejected ones carry an infinite CI). ZUPT is scored on the
    full clean subset, matching how each algorithm reports coverage.
    """
    per_fold_cov: list[float] = []
    per_fold_mae: list[float] = []
    per_fold_acc: list[float] = []
    for f in range(k):
        calib = [r for r in rec_rows if r["fold"] != f]
        val = [r for r in rec_rows if r["fold"] == f]
        if not val:
            continue
        calib_pool = [r for r in calib if (r["accepted"] or not use_accepted_subset)]
        cc = ConformalCalibrator(alpha=alpha)
        cc.fit([r["abs_err"] for r in calib_pool], [r["sigma"] for r in calib_pool])

        eval_set = [r for r in val if (r["accepted"] or not use_accepted_subset)]
        per_fold_acc.append(sum(1 for r in val if r["accepted"]) / len(val))
        if not eval_set:
            continue
        covered, errs = [], []
        for r in eval_set:
            ci = max(ci_floor, min(cc.half_width(r["sigma"]), ci_cap))
            covered.append(1.0 if r["abs_err"] <= ci else 0.0)
            errs.append(r["abs_err"])
        per_fold_cov.append(float(np.mean(covered)))
        per_fold_mae.append(float(np.mean(errs)))

    mean_cov, std_cov = _fold_mean_std(per_fold_cov)
    mean_mae, std_mae = _fold_mean_std(per_fold_mae)
    mean_acc, _ = _fold_mean_std(per_fold_acc)
    feasible = (
        np.isfinite(mean_cov) and np.isfinite(mean_mae)
        and mean_cov >= min_coverage and mean_acc >= min_accept
    )
    return {
        "mean_coverage": mean_cov, "std_coverage": std_cov,
        "mean_mae": mean_mae, "std_mae": std_mae,
        "mean_accept_rate": mean_acc, "feasible": bool(feasible),
    }


def tune_prediction_algo(
    algo_key: str, records: list, fold_of: dict, k: int, grid: dict,
    max_configs: int, min_coverage: float, min_accept: float,
) -> pd.DataFrame:
    # ``alpha`` only scales the conformal CI (coverage) — it does NOT change
    # the predictions. So we run predictions once per non-alpha combo and
    # sweep alpha for free in the cheap conformal loop.
    alpha_values = list(grid.get("alpha", [None]))
    pred_grid = {key: vals for key, vals in grid.items() if key != "alpha"}
    pred_combos = grid_combos(pred_grid, max_configs) if pred_grid else [{}]
    use_accepted_subset = (algo_key == "trapezoid_accel")

    print(f"\n[prediction:{algo_key}] {len(pred_combos)} prediction passes "
          f"x {len(alpha_values)} alpha(s) on {len(records)} segments")
    rows: list[dict] = []
    t0 = time.time()
    for i, pcombo in enumerate(pred_combos):
        rec_rows, ci_floor, ci_cap, default_alpha = _predict_rec_rows(
            algo_key, pcombo, records, fold_of)
        for a in alpha_values:
            alpha = default_alpha if a is None else float(a)
            res = _score_with_alpha(
                rec_rows, k, alpha, ci_floor, ci_cap,
                use_accepted_subset, min_coverage, min_accept)
            rows.append({**pcombo, "alpha": alpha, **res})
        if i == 0:
            per = time.time() - t0
            print(f"  [pass 1/{len(pred_combos)}] {per:.0f}s/pass  "
                  f"~ETA {per * len(pred_combos) / 60:.1f} min total", flush=True)
        if (i + 1) % 5 == 0 or i == len(pred_combos) - 1:
            best = min((r for r in rows if r["feasible"]),
                       key=lambda r: r["mean_mae"], default=None)
            tag = (f"best feasible MAE={best['mean_mae']:.2f}m"
                   if best else "no feasible config yet")
            print(f"  [{i+1:4d}/{len(pred_combos)}] {tag}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    # Feasible configs first (coverage + accept-rate met), then lowest MAE.
    df = (df.sort_values(["feasible", "mean_mae"], ascending=[False, True])
          .reset_index(drop=True))
    return df


def refit_prediction_calibration(
    algo_key: str, overrides: dict, records: list, out_dir: Path,
) -> None:
    """Fit + save conformal calibration for the winning config over ALL clear
    samples, so the 90% CI is honest with the new hyperparameters."""
    cfg = PREDICT_ALGORITHM_CONFIG(
        algorithm=_PRED_ALGO_ENUM[algo_key], overrides=overrides,
    )
    predictor = Predictor(cfg)
    preds = run_predictions(predictor, records)
    from src.prediction.evaluation.runner import to_calibration_samples
    calib = predictor.calibrate(to_calibration_samples(preds))
    out = out_dir / f"calibration_{algo_key}.json"
    predictor.save_calibration(out)
    print(f"  [{algo_key}] refit calibration -> {out}  ({calib})")


# ===========================================================================
# Persistence
# ===========================================================================
def _grid_keys(grid: dict) -> list[str]:
    return list(grid.keys())


def winner_overrides(df: pd.DataFrame, grid_keys: list[str]) -> dict:
    """Extract just the swept-param columns from the top-ranked row."""
    top = df.iloc[0]
    return {k: _json_safe(top[k]) for k in grid_keys}


def _json_safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def persist_config(
    config_path: Path, algo_key: str, overrides: dict, dry_run: bool,
) -> None:
    """Merge ``overrides`` into ``config.json`` under ``algo_key`` (backup +
    diff). No-op write under --dry-run."""
    current = json.loads(config_path.read_text())
    old_section = dict(current.get(algo_key, {}))
    new_section = {**old_section, **overrides}

    print(f"\n[persist] {config_path}  (section: {algo_key})")
    for key in sorted(overrides):
        old = old_section.get(key, "<unset>")
        new = new_section[key]
        flag = "" if old == new else "  <-- changed"
        print(f"    {key}: {old}  ->  {new}{flag}")

    if dry_run:
        print("    [dry-run] config.json NOT modified.")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = config_path.with_suffix(config_path.suffix + f".bak-{stamp}")
    shutil.copy2(config_path, backup)
    current[algo_key] = new_section
    config_path.write_text(json.dumps(current, indent=2) + "\n")
    print(f"    backup: {backup.name}")
    print(f"    wrote new params into {config_path.name}")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--stage", choices=("segmentation", "prediction", "both"),
                    default="both")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--grid-seg", type=Path,
                    default=_REPO_ROOT / "tuning" / "grid_segmentation.json")
    ap.add_argument("--grid-pred", type=Path,
                    default=_REPO_ROOT / "tuning" / "grid_prediction.json")
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--min-accept-rate", type=float, default=0.40)
    ap.add_argument("--max-configs", type=int, default=0)
    ap.add_argument("--include-corrupted", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "tuning")
    ap.add_argument("--dry-run", action="store_true",
                    help="Rank + print diffs but do not modify config.json.")
    ap.add_argument("--no-qc-plots", action="store_true",
                    help="Skip the per-experiment ACC+GT timing QC PNGs "
                         "written to <out-dir>/segment_qc/ at the start of "
                         "the segmentation run.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = load_experiment_index()
    experiments = select_experiments(args.include_corrupted)
    if not experiments:
        raise SystemExit("No experiments selected.")
    print(f"[select] {len(experiments)} experiments pooled (train + test).")
    k, fold_of = build_building_folds(experiments, args.folds, index)

    do_seg = args.stage in ("segmentation", "both")
    do_pred = args.stage in ("prediction", "both")

    summary: list[str] = []

    if do_seg:
        grid = load_grid(args.grid_seg, DEFAULT_GRID_SEGMENTATION)
        cache = preload_segmentation(experiments)
        if not cache:
            print("[segmentation] no usable experiments; skipping.")
        else:
            if not args.no_qc_plots:
                save_segment_qc_plots(cache, args.out_dir)
            df = tune_segmentation(cache, fold_of, k, grid, args.max_configs)
            out_csv = args.out_dir / "results_segmentation.csv"
            df.to_csv(out_csv, index=False)
            print(f"[segmentation] ranked results -> {out_csv}")
            keys = _grid_keys(grid)
            win = winner_overrides(df, keys)
            base = _current_f1_baseline(cache, fold_of, k)
            best = float(df.iloc[0]["mean_f1_like"])
            if best > base:
                summary.append(
                    f"segmentation: f1_like {base:.3f} -> {best:.3f}  win={win}")
                persist_config(
                    SEG_CONFIG_PATH, "acc_template_match", win, args.dry_run)
            else:
                summary.append(
                    f"segmentation: best grid f1_like {best:.3f} did NOT beat "
                    f"current defaults {base:.3f} -- keeping defaults "
                    f"(widen the grid to search further).")
                print(f"\n[segmentation] best ({best:.3f}) <= baseline "
                      f"({base:.3f}); config.json left unchanged.")

    if do_pred:
        grid_all = load_grid(args.grid_pred, DEFAULT_GRID_PREDICTION)
        records = preload_prediction(experiments)
        if not records:
            print("[prediction] no clear segments; skipping.")
        else:
            for algo_key, grid in grid_all.items():
                df = tune_prediction_algo(
                    algo_key, records, fold_of, k, grid,
                    args.max_configs, args.min_coverage, args.min_accept_rate,
                )
                out_csv = args.out_dir / f"results_prediction_{algo_key}.csv"
                df.to_csv(out_csv, index=False)
                print(f"[prediction:{algo_key}] ranked results -> {out_csv}")

                if not bool(df.iloc[0]["feasible"]):
                    msg = (f"prediction:{algo_key}: NO config met "
                           f"coverage>={args.min_coverage:.0%} & "
                           f"accept>={args.min_accept_rate:.0%} -- not persisting. "
                           f"Best coverage seen: {df['mean_coverage'].max():.1%}.")
                    print(f"  [warn] {msg}")
                    summary.append(msg)
                    continue

                keys = _grid_keys(grid)
                win = winner_overrides(df, keys)
                top = df.iloc[0]
                summary.append(
                    f"prediction:{algo_key}: MAE={top['mean_mae']:.2f}m at "
                    f"coverage={top['mean_coverage']:.1%} "
                    f"(accept={top['mean_accept_rate']:.0%})  win={win}")
                persist_config(PRED_CONFIG_PATH, algo_key, win, args.dry_run)
                if not args.dry_run:
                    refit_prediction_calibration(
                        algo_key, win, records, args.out_dir)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for line in summary:
        print("  " + line)
    if args.dry_run:
        print("\n(--dry-run: no config.json files were modified.)")


def _current_f1_baseline(cache: dict, fold_of: dict, k: int) -> float:
    """f1_like of the CURRENT config.json defaults (empty overrides)."""
    res = score_segmentation_config({}, cache, fold_of, k)
    return float(res["mean_f1_like"])


if __name__ == "__main__":
    main()
