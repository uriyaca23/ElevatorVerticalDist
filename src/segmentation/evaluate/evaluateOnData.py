"""Reproducible segmentation evaluation across an experiment subset.

Runs the active accelerometer-only template-match detector against every
experiment that survives the requested filters, then renders the figures
that appear in the *Segmentation & Detection / Evaluation* subsection of
``docs/latex/main.tex`` (failure-mode bars, IoU CDF/PDF, per-experiment
stack, phone-model breakdown, three picked timelines) and the headline
``IntervalPredictionMetrics`` for train / test / pooled / cleaned-pooled.

Typical usage::

    venv/bin/python -m src.segmentation.evaluate.evaluateOnData
    venv/bin/python -m src.segmentation.evaluate.evaluateOnData \\
        --kind train --source experiment \\
        --exclude UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026

Each invocation writes a timestamped directory ``run_YYYYMMDD-HHMMSS/``
under ``--out-root`` (default ``elevator_reports/seg_eval``) containing
the figures, ``run_settings.json`` (every CLI flag, the resolved
experiment list, the active config dump), ``metrics.json``, and
``per_experiment.csv``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src.data.loader import (
    EXPERIMENT_TYPES,
    VALID_SOURCES,
    classify_experiment_type,
    getExperimentData,
    list_experiments,
)
from src.segmentation.algorithms.configTypes import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
)
from src.segmentation.algorithms.metrics import IntervalPredictionMetrics

from . import evaluator, plots as live_plots, report_plots


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_ROOT = REPO_ROOT / "elevator_reports" / "seg_eval"
ITER16_PER_GT_CSV = (
    REPO_ROOT
    / "src" / "segmentation" / "algorithms" / "improvement_iterations"
    / "iter_16_lower_peak_a" / "per_gt.csv"
)


# --------------------------------------------------------------------------
# Naming + phone helpers (mirror scripts/segmentation_evaluation_report.py)
# --------------------------------------------------------------------------
def _short_label(exp_name: str) -> str:
    parts = exp_name.split("_")
    initials = "".join(c for c in parts[0] if c.isupper()) or parts[0][:2].upper()
    location = parts[1] if len(parts) > 1 else ""
    phone = parts[2] if len(parts) > 2 else ""
    loc_short = (
        "milleniumHotel" if location.startswith("millenium") and "Outsi" not in location
        else "milleniumOutsi" if location.startswith("millenium")
        else "BarIlan2" if "BarIlan" in location
        else "beitYitzchaki" if "beitYitzchaki" in location
        else location
    )
    phone_short = (
        "Pix10" if "Pixel" in phone or "Pixel10" in phone
        else "ZF6"  if "Flip6" in phone or "F931B" in phone
        else "S23"  if "SM-S911B" in phone
        else "A23"  if "SM-A235F" in phone
        else "Xmi"  if "Xiaomi" in phone
        else phone[:6]
    )
    return f"{initials}/{loc_short}/{phone_short}"


def _phone_canonical(metadata: dict | None) -> str:
    if not metadata:
        return "unknown"
    raw = (
        metadata.get("phone") or metadata.get("phone_model")
        or metadata.get("model") or metadata.get("device_model")
        or "unknown"
    )
    s = str(raw)
    if "Pixel" in s:                return "Google Pixel 10"
    if "Flip6" in s or "F931B" in s: return "Galaxy Z Flip6"
    if "SM-S911B" in s:              return "Galaxy S23"
    if "SM-A235F" in s:              return "Galaxy A23"
    if "Xiaomi" in s:                return "Xiaomi 22101320I"
    return s


# --------------------------------------------------------------------------
# Experiment filtering
# --------------------------------------------------------------------------
def _experiment_metadata(name: str) -> dict | None:
    """Return the metadata row for ``name`` or ``None`` if unavailable.

    Used to apply ``--source`` filters before running detection.
    """
    try:
        _, _, meta = getExperimentData(name)
    except Exception:
        return None
    return meta


def _resolve_experiments(
    kind: str,
    sources: list[str] | None,
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[str]:
    if include:
        candidates = list(include)
    else:
        candidates = list_experiments(kind="all")
    excluded = set(exclude or [])
    out: list[str] = []
    for name in candidates:
        if name in excluded:
            continue
        if kind != "all" and classify_experiment_type(name) != kind:
            continue
        if sources:
            meta = _experiment_metadata(name)
            src = (meta or {}).get("source", "")
            if src not in sources:
                continue
        out.append(name)
    return out


# --------------------------------------------------------------------------
# Per-split + cleaned aggregation
# --------------------------------------------------------------------------
def _evaluate_subset(
    cfg: SEGMENT_ALGORITHM_CONFIG,
    experiments: list[str],
    out_dir: Path,
    label: str,
):
    """Run the detector on ``experiments`` and dump live-plot bundle.

    Returns ``(raw, per_exp, total, iou, matched_pairs)`` so the caller
    can assemble pooled and cleaned-pooled views without re-running.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{label}] {len(experiments)} experiments")
    raw = evaluator._run_on_experiments(cfg, experiments, verbose=True)

    per_exp = [
        (e.name,
         IntervalPredictionMetrics.from_intervals(e.gt_rides, e.preds))
        for e in raw
    ]
    total = IntervalPredictionMetrics.sum(m for _, m in per_exp)
    pooled_gt, pooled_pred = evaluator._pool_intervals(raw)
    iou_metrics = IntervalPredictionMetrics.iou_f1(
        pooled_gt, pooled_pred, iou_threshold=0.5,
    )
    matched_pairs = evaluator._collect_matched_pairs(raw)
    live_plots.render_all(matched_pairs, total, out_dir)

    return raw, per_exp, total, iou_metrics, matched_pairs


def _aggregate_filtered(raw_results, exclude_set: set[str]):
    kept = [e for e in raw_results if e.name not in exclude_set]
    per_exp = [
        (e.name,
         IntervalPredictionMetrics.from_intervals(e.gt_rides, e.preds))
        for e in kept
    ]
    total = IntervalPredictionMetrics.sum(m for _, m in per_exp)
    pooled_gt, pooled_pred = evaluator._pool_intervals(kept)
    iou = IntervalPredictionMetrics.iou_f1(
        pooled_gt, pooled_pred, iou_threshold=0.5,
    )
    pairs = evaluator._collect_matched_pairs(kept)
    return per_exp, total, iou, pairs


# --------------------------------------------------------------------------
# Timeline picker (best / typical / worst)
# --------------------------------------------------------------------------
def _pick_timeline_examples(per_exp_pool, raw_pool):
    by_name = {e.name: e for e in raw_pool}
    scored = [(n, m, m.rates()["f1_like"]) for n, m in per_exp_pool]
    scored = [s for s in scored if s[1].n_gt >= 6] or scored
    if not scored:
        return []
    by_score = sorted(scored, key=lambda t: t[2])
    picks = [
        ("best",    by_score[-1]),
        ("typical", by_score[len(by_score) // 2]),
        ("worst",   by_score[0]),
    ]
    return [(label, item[0], by_name[item[0]])
            for label, item in picks if item[0] in by_name]


# --------------------------------------------------------------------------
# CSV / metric dumps
# --------------------------------------------------------------------------
def _per_exp_csv(per_exp, out_path: Path) -> None:
    rows = []
    for name, m in per_exp:
        r = m.rates()
        rows.append({
            "exp": name, "label": _short_label(name),
            "kind": classify_experiment_type(name),
            **m.as_dict(),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _metrics_payload(label: str, total, iou, n_exp):
    return {
        "label": label, "n_experiments": n_exp,
        "total": total.as_dict(), "iou": iou,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="evaluateOnData",
        description="Reproducible segmentation evaluation: filter "
                    "experiments, run the active detector, render every "
                    "figure consumed by the segmentation subsection of "
                    "docs/latex/main.tex.",
    )
    p.add_argument(
        "--algorithm", default=SegmentAlgorithm.ACC_TEMPLATE_MATCH.value,
        choices=[a.value for a in SegmentAlgorithm],
        help="Detector to run (defaults to acc_template_match).",
    )
    p.add_argument(
        "--kind", default="all",
        choices=("all", *EXPERIMENT_TYPES),
        help="Restrict to train, test, or all experiments.",
    )
    p.add_argument(
        "--source", action="append", default=None,
        choices=list(VALID_SOURCES),
        help="Filter by metadata.source — repeat to allow multiple "
             "(e.g. --source experiment --source ido). Default: any.",
    )
    p.add_argument(
        "--include", nargs="*", default=None,
        help="Whitelist of experiment names. When provided, only these "
             "names are considered (still subject to other filters).",
    )
    p.add_argument(
        "--exclude", nargs="*", default=None,
        help="Drop these experiment names from the run.",
    )
    p.add_argument(
        "--cleaned-exclude", nargs="*", default=(
            "UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026",
            "UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2",
            "eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1",
        ),
        help="Experiments excluded only from the *cleaned* aggregates "
             "(parallel of the 'after removing three outliers' panel). "
             "Pass an empty list to skip the cleaned pass entirely.",
    )
    p.add_argument(
        "--out-root", type=Path, default=DEFAULT_OUT_ROOT,
        help="Base directory; the run is written to "
             "<out-root>/run_<timestamp>/.",
    )
    p.add_argument(
        "--run-name", default=None,
        help="Override the timestamp folder name (used as-is).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm(args.algorithm))

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_root / (args.run_name or f"run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing run artefacts under {run_dir}")

    cleaned_exclude = set(args.cleaned_exclude or [])

    # --- resolve filtered experiment list per split + pooled ---
    train_exps = _resolve_experiments(
        kind="train", sources=args.source,
        include=args.include, exclude=args.exclude,
    ) if args.kind in ("all", "train") else []
    test_exps = _resolve_experiments(
        kind="test", sources=args.source,
        include=args.include, exclude=args.exclude,
    ) if args.kind in ("all", "test") else []
    all_exps = sorted(set(train_exps) | set(test_exps))

    if not all_exps:
        print("no experiments survived filtering; nothing to do",
              file=sys.stderr)
        return 1

    # Persist run settings as soon as we know the resolved list — even if
    # detection later raises, the user has a record of what was attempted.
    settings = {
        "timestamp": timestamp,
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "config": {
            "algorithm": cfg.algorithm.value,
            "config_path": str(cfg.config_path),
            "overrides": cfg.overrides,
            "active_params": cfg.load_params(),
        },
        "experiments": {
            "train": train_exps, "test": test_exps,
            "n_train": len(train_exps), "n_test": len(test_exps),
            "n_total": len(all_exps),
            "cleaned_exclude": sorted(cleaned_exclude),
        },
    }
    (run_dir / "run_settings.json").write_text(
        json.dumps(settings, indent=2, default=str)
    )

    # --- run detection per split ---
    t0 = time.time()
    train_raw = train_per_exp = test_raw = test_per_exp = None
    train_total = test_total = None
    train_iou = test_iou = None
    train_pairs = test_pairs = []

    if train_exps:
        (train_raw, train_per_exp, train_total, train_iou, train_pairs) = \
            _evaluate_subset(cfg, train_exps, run_dir / "train", "train")
    if test_exps:
        (test_raw,  test_per_exp,  test_total,  test_iou,  test_pairs) = \
            _evaluate_subset(cfg, test_exps,  run_dir / "test", "test")
    print(f"\ndetection finished in {time.time() - t0:.1f}s")

    pooled_pairs = (train_pairs or []) + (test_pairs or [])
    pooled_per_exp = (train_per_exp or []) + (test_per_exp or [])
    pooled_raw = (train_raw or []) + (test_raw or [])
    pooled_total = sum((m for _, m in pooled_per_exp),
                       start=IntervalPredictionMetrics())

    # --- combined figures (mirroring main.tex segmentation results) ---
    print("\nrendering combined figures")
    if train_total is not None and test_total is not None:
        report_plots.failure_modes_split_bar(
            train_total, test_total,
            run_dir / "failure_modes_train_vs_test.png",
        )
    report_plots.per_experiment_failure_bar(
        pooled_per_exp, run_dir / "per_experiment_failure_bar.png",
        label_short={n: _short_label(n) for n, _ in pooled_per_exp},
    )
    report_plots.cdf_pdf_pair(
        [p["iou"] for p in pooled_pairs],
        title="IoU over matched pairs", xlabel="IoU",
        out_path=run_dir / "cdf_pdf_iou.png",
    )
    report_plots.iou_vs_duration_scatter(
        pooled_pairs, run_dir / "iou_vs_duration.png",
    )
    report_plots.pred_vs_gt_duration_scatter(
        pooled_pairs, run_dir / "pred_vs_gt_duration.png",
    )

    # Phone-model breakdown — needs metadata per experiment
    phone_for_exp: dict[str, str] = {}
    for e in pooled_raw:
        meta = _experiment_metadata(e.name)
        phone_for_exp[e.name] = _phone_canonical(meta)
    report_plots.phone_breakdown_bar(
        pooled_per_exp, phone_for_exp, run_dir / "phone_breakdown.png",
    )

    # Per-experiment timelines (best / typical / worst)
    print("\nrendering per-experiment timeline picks")
    for label, name, exp_result in _pick_timeline_examples(
        pooled_per_exp, pooled_raw,
    ):
        try:
            sensors, _, _ = getExperimentData(name)
        except Exception as exc:
            print(f"  [skip] timeline_{label}: {name} → {exc}")
            continue
        acc = sensors.get("ACC")
        if acc is None or acc.empty:
            continue
        report_plots.per_experiment_timeline(
            name, acc, exp_result.gt_rides, exp_result.preds,
            run_dir / f"timeline_{label}.png",
        )
        print(f"  timeline_{label}: {name}")

    # --- cleaned-pooled (drops cleaned_exclude) ---
    cleaned_payload = None
    if cleaned_exclude and pooled_raw:
        print(f"\ncleaned aggregates (excluded: {sorted(cleaned_exclude)})")
        c_train = _aggregate_filtered(train_raw or [], cleaned_exclude)
        c_test  = _aggregate_filtered(test_raw  or [], cleaned_exclude)
        c_train_per_exp, c_train_total, c_train_iou, c_train_pairs = c_train
        c_test_per_exp,  c_test_total,  c_test_iou,  c_test_pairs  = c_test
        c_pairs = c_train_pairs + c_test_pairs
        c_per_exp = c_train_per_exp + c_test_per_exp

        if c_train_total is not None and c_test_total is not None and \
           train_total is not None and test_total is not None:
            report_plots.failure_modes_split_bar(
                c_train_total, c_test_total,
                run_dir / "failure_modes_train_vs_test_cleaned.png",
            )
        report_plots.per_experiment_failure_bar(
            c_per_exp, run_dir / "per_experiment_failure_bar_cleaned.png",
            label_short={n: _short_label(n) for n, _ in c_per_exp},
        )
        report_plots.cdf_pdf_pair(
            [p["iou"] for p in c_pairs],
            title="IoU over matched pairs (cleaned set)",
            xlabel="IoU", out_path=run_dir / "cdf_pdf_iou_cleaned.png",
        )
        report_plots.iou_vs_duration_scatter(
            c_pairs, run_dir / "iou_vs_duration_cleaned.png",
        )
        cleaned_payload = {
            "train": _metrics_payload("train_cleaned", c_train_total,
                                      c_train_iou, len(c_train_per_exp))
                       if c_train_total else None,
            "test":  _metrics_payload("test_cleaned",  c_test_total,
                                      c_test_iou,  len(c_test_per_exp))
                       if c_test_total else None,
        }

    # --- constraint-justification plots from iter_16 per_gt.csv ---
    if ITER16_PER_GT_CSV.exists():
        print(f"\nrendering constraint plots from {ITER16_PER_GT_CSV.name}")
        per_gt = pd.read_csv(ITER16_PER_GT_CSV)
        active = cfg.load_params()
        report_plots.score_hist_by_status(
            per_gt, "pair_joint_r2", threshold=active["joint_r2_thresh"],
            title="Pair joint $R^2$ by GT status",
            xlabel="pair_joint_r2",
            out_path=run_dir / "constraint_pair_joint_r2.png",
        )
        report_plots.score_hist_by_status(
            per_gt, "pair_A_abs", threshold=active["min_pair_abs_a"],
            title="Pair amplitude $|A|$ by GT status",
            xlabel="pair_A_abs (m/s²)",
            out_path=run_dir / "constraint_pair_A.png",
        )
        report_plots.score_hist_by_status(
            per_gt, "pair_heatmap_energy",
            threshold=active["heatmap_energy_thresh"],
            title="Heatmap energy by GT status",
            xlabel="pair_heatmap_energy",
            out_path=run_dir / "constraint_heatmap_energy.png",
        )
        report_plots.peak_score_hist_combined(
            per_gt, "pos_r2", "neg_r2", threshold=active["r2_peak_thresh"],
            title="Peak signed $R^2$ by GT status (both lobes pooled)",
            xlabel="peak signed $R^2$",
            out_path=run_dir / "constraint_peak_R2.png",
        )
        report_plots.peak_score_hist_combined(
            per_gt, "pos_A", "neg_A", threshold=active["min_peak_abs_a"],
            title="Peak amplitude $|A|$ by GT status (both lobes pooled)",
            xlabel="peak $|A|$ (m/s²)", take_abs=True,
            out_path=run_dir / "constraint_peak_A.png",
        )
        report_plots.constraint_2d_scatter(
            per_gt, x_col="pair_joint_r2", y_col="pair_A_abs",
            x_threshold=active["joint_r2_thresh"],
            y_threshold=active["min_pair_abs_a"],
            title="Pair joint $R^2$ vs pair $|A|$",
            xlabel="pair_joint_r2", ylabel="pair_A_abs (m/s²)",
            out_path=run_dir / "constraint_jointR2_vs_pairA.png",
        )
        report_plots.reject_reason_bar(
            per_gt, run_dir / "constraint_reject_reasons.png",
        )
    else:
        print(f"  [skip] {ITER16_PER_GT_CSV} not found — "
              "constraint plots omitted")

    # --- per-experiment CSVs + metrics dump ---
    if pooled_per_exp:
        _per_exp_csv(pooled_per_exp, run_dir / "per_experiment.csv")

    metrics_dump = {
        "train":  _metrics_payload("train",  train_total, train_iou,
                                   len(train_per_exp or []))
                    if train_total else None,
        "test":   _metrics_payload("test",   test_total,  test_iou,
                                   len(test_per_exp or []))
                    if test_total else None,
        "pooled": _metrics_payload(
            "pooled", pooled_total,
            IntervalPredictionMetrics.iou_f1(
                *evaluator._pool_intervals(pooled_raw),
                iou_threshold=0.5,
            ) if pooled_raw else {},
            len(pooled_per_exp),
        ) if pooled_per_exp else None,
        "cleaned": cleaned_payload,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_dump, indent=2, default=str)
    )

    # --- summary printout ---
    print("\nsummary")
    if train_total is not None:
        print(f"  train : gt={train_total.n_gt} pred={train_total.n_pred} "
              f"clean={train_total.clean} miss={train_total.missed} "
              f"fp={train_total.fp} f1*={train_total.score():.3f} "
              f"iou_f1={train_iou['iou_f1@0.5']:.3f}")
    if test_total is not None:
        print(f"  test  : gt={test_total.n_gt} pred={test_total.n_pred} "
              f"clean={test_total.clean} miss={test_total.missed} "
              f"fp={test_total.fp} f1*={test_total.score():.3f} "
              f"iou_f1={test_iou['iou_f1@0.5']:.3f}")
    if pooled_total.n_gt:
        print(f"  pooled: gt={pooled_total.n_gt} pred={pooled_total.n_pred} "
              f"clean={pooled_total.clean} miss={pooled_total.missed} "
              f"fp={pooled_total.fp} f1*={pooled_total.score():.3f}")
    print(f"\nartefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
