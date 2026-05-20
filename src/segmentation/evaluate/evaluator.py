"""Generic evaluation harness for any :class:`Segmenter`-compatible algorithm.

Two top-level entry points — both go through ``Segmenter.detect`` so they
work uniformly with every algorithm registered in
:class:`SegmentAlgorithm`:

* :func:`sweep_hyperparameters` — grid-search ``overrides`` on a base
  :class:`SEGMENT_ALGORITHM_CONFIG`, scoring each combo across a fixed
  set of experiments. Returns a sorted DataFrame of
  ``{params} + IntervalPredictionMetrics + iou_f1@0.5``.

* :func:`evaluate_algorithm` — single-config run across all experiments
  with per-exp metrics, aggregate totals, and a standard set of
  diagnostic plots (CDFs of IoU / edge residuals / duration error, plus
  a failure-mode bar chart).

Both rely on :func:`_prepare_segmenter_input` to adapt the loaded
``sensors`` dict into the DataFrame shape each algorithm expects. Adding
a new :class:`SegmentAlgorithm` requires one branch there.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import getExperimentData, resolve_experiments
from src.segmentation.algorithms.configTypes import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
)
from src.segmentation.algorithms.segmenter import Segmenter
from src.segmentation.algorithms.metrics import IntervalPredictionMetrics

from . import plots


# --------------------------------------------------------------------------
# Input / output adapters
# --------------------------------------------------------------------------
def _prepare_segmenter_input(
    sensors: dict[str, pd.DataFrame], algo: SegmentAlgorithm,
) -> tuple[pd.DataFrame, float]:
    """Build the DataFrame fed to :meth:`Segmenter.detect` from raw sensors.

    Returns ``(detect_input, t0_ms)`` where ``t0_ms`` is the Unix-epoch
    start time used to align GT ride timestamps into the session-relative
    seconds that the segmenter emits.

    Extend this with a new branch when adding algorithms to
    :class:`SegmentAlgorithm`.
    """
    if algo is SegmentAlgorithm.PRESSURE_FILTER:
        prs = sensors.get("PRS")
        if prs is None or prs.empty:
            raise KeyError("PRS")
        t0_ms = float(prs["timestamp_ms"].iloc[0])
        return pd.DataFrame({
            "time":   (prs["timestamp_ms"].to_numpy(dtype=float) - t0_ms) / 1000.0,
            "height": prs["GT_height_m"].to_numpy(dtype=float),
        }), t0_ms
    if algo is SegmentAlgorithm.ACC_TEMPLATE_MATCH:
        acc = sensors.get("ACC")
        if acc is None or acc.empty:
            raise KeyError("ACC")
        t0_ms = float(acc["timestamp_ms"].iloc[0])
        # Active detector reads ``timestamp_ms`` directly and derives its
        # own t0 / fs — pass raw ms, not seconds-relative.
        return pd.DataFrame({
            "timestamp_ms": acc["timestamp_ms"].to_numpy(dtype=float),
            "x":            acc["x"].to_numpy(dtype=float),
            "y":            acc["y"].to_numpy(dtype=float),
            "z":            acc["z"].to_numpy(dtype=float),
        }), t0_ms
    raise ValueError(f"Unsupported algorithm: {algo}")


def _gt_to_interval_dicts(gt: pd.DataFrame | None, t0_ms: float) -> list[dict]:
    rides: list[dict] = []
    if gt is None or gt.empty:
        return rides
    for _, row in gt.iterrows():
        if row.get("type") not in ("up", "down"):
            continue
        # ``signalClearRecording`` is the gt.csv quality marker (True = clean,
        # False = noisy). Default to True when the column is missing so older
        # GT files don't silently get classified as noisy.
        sc = row.get("signalClearRecording", True)
        rides.append({
            "type":         row["type"],
            "t_start_s":    (float(row["start_ms"]) - t0_ms) / 1000.0,
            "t_end_s":      (float(row["end_ms"])   - t0_ms) / 1000.0,
            "signal_clear": bool(sc) if sc is not None else True,
        })
    return rides


def _segments_to_interval_dicts(segments: pd.DataFrame) -> list[dict]:
    """Collapse CI-valued segmenter output to center-point interval dicts."""
    out: list[dict] = []
    if segments is None or segments.empty:
        return out
    for _, row in segments.iterrows():
        s_lo, s_hi = row["start_ci"]
        e_lo, e_hi = row["end_ci"]
        out.append({
            "t_start_s": 0.5 * (float(s_lo) + float(s_hi)),
            "t_end_s":   0.5 * (float(e_lo) + float(e_hi)),
            "type":      row.get("type"),
        })
    return out


# --------------------------------------------------------------------------
# Per-experiment runner
# --------------------------------------------------------------------------
@dataclass
class _ExpResult:
    name: str
    gt_rides: list[dict]
    preds: list[dict]


def _phone_model_from_metadata(metadata: dict | None) -> str:
    """Resolve the phone model string from an experiment's metadata dict.

    Returns ``""`` when metadata is absent or the field isn't populated,
    which makes the phone-aware noise floor a no-op (the hard-coded
    amplitude floor wins; see ``TemplateMatchConfig.noise_sigma_multiplier``).
    """
    if not metadata:
        return ""
    for key in ("phone_model", "phone", "model", "device_model"):
        v = metadata.get(key)
        if v:
            return str(v)
    return ""


def _run_on_experiments(
    config: SEGMENT_ALGORITHM_CONFIG,
    experiments: list[str],
    verbose: bool = False,
    phone_model: str | None = None,
) -> list[_ExpResult]:
    """Run the segmenter across a list of experiments.

    ``phone_model`` (optional) forces a single phone model for every
    experiment; when ``None``, each experiment's metadata is consulted
    via :func:`_phone_model_from_metadata`. Ignored by algorithms that
    don't use chip-spec floors (e.g. the pressure filter).
    """
    segmenter = Segmenter(config)
    out: list[_ExpResult] = []
    for name in experiments:
        try:
            sensors, gt, metadata = getExperimentData(name)
        except Exception as exc:
            if verbose:
                print(f"  [error] {name}: {type(exc).__name__}: {exc}")
            continue
        try:
            data, t0_ms = _prepare_segmenter_input(sensors, config.algorithm)
        except KeyError as missing:
            if verbose:
                print(f"  [skip]  {name}: missing sensor {missing}")
            continue
        if len(data) < 2:
            if verbose:
                print(f"  [skip]  {name}: too few samples")
            continue
        phone = (
            phone_model
            if phone_model is not None
            else _phone_model_from_metadata(metadata)
        )
        segments = segmenter.detect(data, phone_model=phone)
        out.append(_ExpResult(
            name=name,
            gt_rides=_gt_to_interval_dicts(gt, t0_ms),
            preds=_segments_to_interval_dicts(segments),
        ))
    return out


def _pool_intervals(
    exps: list[_ExpResult],
) -> tuple[list[dict], list[dict]]:
    """Offset each exp's timeline so pooled IoU never cross-matches."""
    pooled_gt: list[dict] = []
    pooled_pred: list[dict] = []
    for i, e in enumerate(exps):
        offset = (i + 1) * 1e9
        pooled_gt.extend(
            {**g, "t_start_s": g["t_start_s"] + offset,
             "t_end_s":   g["t_end_s"]   + offset}
            for g in e.gt_rides
        )
        pooled_pred.extend(
            {**p, "t_start_s": p["t_start_s"] + offset,
             "t_end_s":   p["t_end_s"]   + offset}
            for p in e.preds
        )
    return pooled_gt, pooled_pred


# --------------------------------------------------------------------------
# Public API — hyperparameter sweep
# --------------------------------------------------------------------------
def sweep_hyperparameters(
    base_config: SEGMENT_ALGORITHM_CONFIG,
    param_grid: dict[str, list],
    experiments: list[str] | None = None,
    out_csv: Path | str | None = None,
    log_every: int = 5,
    phone_model: str | None = None,
) -> pd.DataFrame:
    """Grid-sweep ``param_grid`` as overrides on top of ``base_config``.

    Each key in ``param_grid`` must be a valid field of the chosen
    algorithm's config (``PressureFilterConfig``, ``TemplateMatchConfig``,
    …) — values are merged into ``base_config.overrides`` per combo.

    Returns a DataFrame with one row per combo: the param columns plus
    aggregated :class:`IntervalPredictionMetrics` counts/rates and pooled
    ``iou_f1@0.5``. Sorted by ``f1_like`` descending.
    """
    if experiments is None:
        experiments = resolve_experiments(kind="train")
    keys = list(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    rows: list[dict] = []
    t0 = time.time()
    for i, combo in enumerate(combos):
        overrides = {**base_config.overrides, **dict(zip(keys, combo))}
        cfg = base_config.model_copy(update={"overrides": overrides})
        exps = _run_on_experiments(cfg, experiments, phone_model=phone_model)
        total = IntervalPredictionMetrics.sum(
            IntervalPredictionMetrics.from_intervals(e.gt_rides, e.preds)
            for e in exps
        )
        pooled_gt, pooled_pred = _pool_intervals(exps)
        iou_metrics = IntervalPredictionMetrics.iou_f1(
            pooled_gt, pooled_pred, iou_threshold=0.5,
        )
        rows.append({**dict(zip(keys, combo)), **total.as_dict(), **iou_metrics})
        if i == 0 or (i + 1) % log_every == 0 or i == len(combos) - 1:
            print(
                f"  [{i + 1:5d}/{len(combos)}] "
                f"f1_like={total.score():.3f} "
                f"iou_f1={iou_metrics['iou_f1@0.5']:.3f} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
    df = (
        pd.DataFrame(rows)
        .sort_values(["f1_like", "iou_f1@0.5"], ascending=[False, False])
        .reset_index(drop=True)
    )
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


# --------------------------------------------------------------------------
# Public API — single-config evaluation
# --------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    """Everything :func:`evaluate_algorithm` produces (before plotting)."""

    total: IntervalPredictionMetrics
    per_exp: list[tuple[str, IntervalPredictionMetrics]]
    iou_metrics: dict[str, float]
    matched_pairs: list[dict] = field(default_factory=list)
    plot_paths: dict[str, Path] = field(default_factory=dict)

    def as_summary_dict(self) -> dict:
        return {
            **self.total.as_dict(),
            **self.iou_metrics,
            "n_experiments": len(self.per_exp),
        }


def _collect_matched_pairs(exps: list[_ExpResult]) -> list[dict]:
    """Greedy best-IoU matching per exp → list of residual records.

    Each record holds the matched GT/pred interval, their IoU, and the
    signed edge / duration residuals — the raw input to all CDF plots.
    """
    records: list[dict] = []
    for e in exps:
        g_iv = [(g["t_start_s"], g["t_end_s"]) for g in e.gt_rides]
        p_iv = [(p["t_start_s"], p["t_end_s"]) for p in e.preds]
        used = [False] * len(p_iv)
        order = sorted(
            range(len(g_iv)),
            key=lambda j: max(
                (_iou(p_iv[k], g_iv[j]) for k in range(len(p_iv))),
                default=0.0,
            ),
            reverse=True,
        )
        for j in order:
            g = g_iv[j]
            best_i, best_iou = -1, 0.0
            for i, p in enumerate(p_iv):
                if used[i]:
                    continue
                v = _iou(p, g)
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_i < 0 or best_iou <= 0.0:
                continue
            used[best_i] = True
            p = p_iv[best_i]
            records.append({
                "exp":              e.name,
                "iou":              best_iou,
                "start_residual_s": p[0] - g[0],
                "end_residual_s":   p[1] - g[1],
                "duration_error_s": (p[1] - p[0]) - (g[1] - g[0]),
                "gt_start_s":  g[0], "gt_end_s":  g[1],
                "pred_start_s": p[0], "pred_end_s": p[1],
                # Noise polarity comes from the matched GT row; the noise
                # loop in the segmentation evaluator uses this to slice the
                # matched-pair set per pass (clean / noisy / both).
                "signal_clear": bool(e.gt_rides[j].get("signal_clear", True)),
            })
    return records


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def evaluate_algorithm(
    config: SEGMENT_ALGORITHM_CONFIG,
    experiments: list[str] | None = None,
    out_dir: Path | str | None = None,
    kind: str = "train",
    phone_model: str | None = None,
) -> EvaluationResult:
    """Run ``config``'s detector across ``experiments`` and produce a
    full evaluation bundle.

    Metrics: per-exp + aggregate :class:`IntervalPredictionMetrics` plus
    pooled ``iou_f1@0.5``.

    Plots (written to ``out_dir`` when provided):

    * ``cdf_iou.png``              — CDF of matched-pair IoU.
    * ``cdf_start_residual.png``   — CDF of signed start-edge error (s).
    * ``cdf_end_residual.png``     — CDF of signed end-edge error (s).
    * ``cdf_duration_error.png``   — CDF of duration error (s).
    * ``failure_modes.png``        — bar chart of clean / miss / merge / split / fp.
    """
    if experiments is None:
        experiments = resolve_experiments(kind=kind)
    exps = _run_on_experiments(
        config, experiments, verbose=True, phone_model=phone_model,
    )

    per_exp = [
        (e.name, IntervalPredictionMetrics.from_intervals(e.gt_rides, e.preds))
        for e in exps
    ]
    total = IntervalPredictionMetrics.sum(m for _, m in per_exp)
    pooled_gt, pooled_pred = _pool_intervals(exps)
    iou_metrics = IntervalPredictionMetrics.iou_f1(
        pooled_gt, pooled_pred, iou_threshold=0.5,
    )
    matched_pairs = _collect_matched_pairs(exps)

    plot_paths: dict[str, Path] = {}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = plots.render_all(matched_pairs, total, out_dir)

    return EvaluationResult(
        total=total,
        per_exp=per_exp,
        iou_metrics=iou_metrics,
        matched_pairs=matched_pairs,
        plot_paths=plot_paths,
    )
