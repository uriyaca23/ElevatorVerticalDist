"""Reproducible segmentation evaluation across an experiment subset.

Runs the active accelerometer-only template-match detector against every
experiment that survives the requested filters, then renders the
evaluation figures (failure-mode bars, IoU CDF/PDF, per-experiment
stack, phone-model breakdown, three picked timelines) plus a
``metrics.json`` with the run's full interval metrics and a GT-side
detection breakdown by noise class (clean vs noisy rides).

Segmentation always runs on the full resolved experiment set — there is
no noise filter. ``metrics.json`` reports the run's ``overall`` interval
metrics and, under ``by_noise``, how many clean vs noisy GT rides the
detector caught — so one run shows its accuracy on each noise class.
False positives stay in ``overall`` only: a prediction that matches no
GT ride cannot be attributed to a noise class. ``--kind`` / ``--source``
/ ``--include`` / ``--exclude`` pick *which experiments* feed the run.

Typical usage::

    # 1. Defaults: every source, train+test
    venv/bin/python -m src.segmentation.evaluate.evaluateOnData

    # 2. One source only
    venv/bin/python -m src.segmentation.evaluate.evaluateOnData \\
        --source experiment

    # 3. Two sources (Ido + real-world, skip lab experiments)
    venv/bin/python -m src.segmentation.evaluate.evaluateOnData \\
        --source ido --source real_world

    # 4. Train only / test only
    venv/bin/python -m src.segmentation.evaluate.evaluateOnData --kind train
    venv/bin/python -m src.segmentation.evaluate.evaluateOnData --kind test

    # 5. Test split, Ido source
    venv/bin/python -m src.segmentation.evaluate.evaluateOnData \\
        --kind test --source ido

    # 6. Drop a known-bad experiment
    venv/bin/python -m src.segmentation.evaluate.evaluateOnData \\
        --source experiment \\
        --exclude UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026

    # 7. Whitelist a couple of experiments
    venv/bin/python -m src.segmentation.evaluate.evaluateOnData \\
        --include eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1 \\
                  UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1

    # 8. Custom output root + stable run name (no timestamp)
    venv/bin/python -m src.segmentation.evaluate.evaluateOnData \\
        --source experiment \\
        --out-root /tmp/seg_eval --run-name source_experiment_only

    # 9. Run a different detector (e.g. the pressure-filter fallback)
    venv/bin/python -m src.segmentation.evaluate.evaluateOnData \\
        --algorithm pressure_filter

Each invocation writes a timestamped directory ``run_YYYYMMDD-HHMMSS/``
under ``--out-root`` (default ``elevator_reports/seg_eval``). Every
figure, ``per_experiment.csv``, ``metrics.json`` and ``run_settings.json``
is written flat into that one folder — no sub-directories. The figure
bundle is rendered three times — for the full set and for the clean /
noisy subsets — with ``_clean`` / ``_noisy`` filename suffixes (the
full set carries no suffix). ``metrics.json`` holds the run's
``overall`` metrics plus a ``by_noise`` detection breakdown (clean vs
noisy GT rides).
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
    return sorted(out)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def _aggregate_filtered(raw_results, exclude_set: set[str]):
    """Compute ``(per_exp, total, iou, matched_pairs)`` from raw results.

    ``exclude_set`` drops experiments by name — pass an empty set for the
    plain aggregate, or the ``--cleaned-exclude`` list for the cleaned
    outlier-removal view.
    """
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


def _filter_raw_by_noise(
    raw_results: list, keep_clean: bool | None,
) -> list:
    """Return a copy of ``raw_results`` with ``gt_rides`` filtered by noise.

    ``keep_clean`` is ``True`` to retain only ``signal_clear==True`` GTs,
    ``False`` for the opposite, ``None`` to keep everything. The
    ``signal_clear`` flag is set by the evaluator from the gt.csv column
    ``signalClearRecording``. Predictions are left untouched so a GT
    ride's detected/missed status is unchanged by the filter — used to
    score clean vs noisy GT rides independently within one run.
    """
    if keep_clean is None:
        return list(raw_results)
    out = []
    for e in raw_results:
        filtered_gt = [
            g for g in e.gt_rides
            if bool(g.get("signal_clear", True)) == keep_clean
        ]
        out.append(evaluator._ExpResult(
            name=e.name, gt_rides=filtered_gt, preds=list(e.preds),
        ))
    return out


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


def _detection_line(label: str, n_gt: int, missed: int, extra: str = "") -> str:
    """One-line console summary of GT-side detection for a noise class."""
    if not n_gt:
        return f"  {label:7s}: (no GT rides in this run)"
    detected = n_gt - missed
    return (f"  {label:7s}: gt={n_gt} detected={detected} missed={missed} "
            f"rate={detected / n_gt:.1%}{extra}")


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _render_figures(
    raw: list,
    run_dir: Path,
    suffix: str,
    phone_for_exp: dict[str, str],
) -> None:
    """Render the full figure bundle for one noise subset of ``raw``.

    ``suffix`` ("" / "_clean" / "_noisy") is appended to every filename
    so the three subsets share one flat run directory.
    """
    per_exp, total, _iou, pairs = _aggregate_filtered(raw, exclude_set=set())
    label = suffix.lstrip("_") or "all"
    print(f"  figures [{label}]: gt={total.n_gt}, pairs={len(pairs)}")

    live_plots.render_all(pairs, total, run_dir, suffix=suffix)

    if per_exp:
        report_plots.per_experiment_failure_bar(
            per_exp, run_dir / f"per_experiment_failure_bar{suffix}.png",
            label_short={n: _short_label(n) for n, _ in per_exp},
        )
    if pairs:
        report_plots.cdf_pdf_pair(
            [p["iou"] for p in pairs],
            title=f"IoU over matched pairs ({label})",
            xlabel="IoU", out_path=run_dir / f"cdf_pdf_iou{suffix}.png",
        )
        report_plots.iou_vs_duration_scatter(
            pairs, run_dir / f"iou_vs_duration{suffix}.png",
        )
        report_plots.pred_vs_gt_duration_scatter(
            pairs, run_dir / f"pred_vs_gt_duration{suffix}.png",
        )
    if per_exp and phone_for_exp:
        report_plots.phone_breakdown_bar(
            per_exp, phone_for_exp, run_dir / f"phone_breakdown{suffix}.png",
        )

    # timelines (best / typical / worst) — picked from this subset
    if per_exp and raw:
        for tlabel, name, exp_result in _pick_timeline_examples(per_exp, raw):
            try:
                sensors, _, _ = getExperimentData(name)
            except Exception as exc:
                print(f"    [skip] timeline_{tlabel}{suffix}: {name} → {exc}")
                continue
            acc = sensors.get("ACC")
            if acc is None or acc.empty:
                continue
            report_plots.per_experiment_timeline(
                name, acc, exp_result.gt_rides, exp_result.preds,
                run_dir / f"timeline_{tlabel}{suffix}.png",
            )

    if per_exp:
        _per_exp_csv(per_exp, run_dir / f"per_experiment{suffix}.csv")


def _render_run(
    run_dir: Path,
    raw: list,
    kind_label: str,
    phone_for_exp: dict[str, str],
    cleaned_exclude: set[str],
) -> dict:
    """Render figure bundles for the full set + the clean / noisy
    subsets (flat, ``_clean`` / ``_noisy`` filename suffixes), then
    assemble the dict written to ``metrics.json``.
    """
    print("rendering figure sets — all / clean / noisy:")
    for suffix, keep in (("", None), ("_clean", True), ("_noisy", False)):
        _render_figures(_filter_raw_by_noise(raw, keep),
                        run_dir, suffix, phone_for_exp)

    per_exp, total, iou, _ = _aggregate_filtered(raw, exclude_set=set())

    # cleaned aggregate — an orthogonal outlier-removal view (full set
    # only); renders parallel ``*_cleaned`` figures into the flat folder.
    cleaned_payload = None
    if cleaned_exclude and raw:
        print(f"cleaned aggregate (excluded: {sorted(cleaned_exclude)})")
        c_per_exp, c_total, c_iou, c_pairs = _aggregate_filtered(
            raw, cleaned_exclude,
        )
        if c_per_exp:
            report_plots.per_experiment_failure_bar(
                c_per_exp, run_dir / "per_experiment_failure_bar_cleaned.png",
                label_short={n: _short_label(n) for n, _ in c_per_exp},
            )
        if c_pairs:
            report_plots.cdf_pdf_pair(
                [p["iou"] for p in c_pairs],
                title="IoU over matched pairs (cleaned set)",
                xlabel="IoU", out_path=run_dir / "cdf_pdf_iou_cleaned.png",
            )
            report_plots.iou_vs_duration_scatter(
                c_pairs, run_dir / "iou_vs_duration_cleaned.png",
            )
        cleaned_payload = (
            _metrics_payload("cleaned", c_total, c_iou, len(c_per_exp))
            if c_per_exp else None
        )

    # metrics.json carries the run's full interval metrics (``overall``)
    # plus a GT-side detection breakdown by the signal_clear flag
    # (``by_noise`` → clean / noisy). Only GT-side numbers are split:
    # ``detected`` / ``missed`` of a GT ride depend solely on whether a
    # prediction overlaps *it*, so they partition cleanly by noise class.
    # fp / precision do NOT — a false positive matches no GT ride at all,
    # so it cannot be attributed to a class; those live in ``overall``.
    by_noise: dict[str, dict] = {}
    for lbl, keep in (("clean", True), ("noisy", False)):
        sub = _filter_raw_by_noise(raw, keep)
        _, s_total, _, _ = _aggregate_filtered(sub, set())
        detected = s_total.n_gt - s_total.missed
        by_noise[lbl] = {
            "label": lbl,
            "n_gt": s_total.n_gt,
            "detected": detected,
            "missed": s_total.missed,
            "detection_rate": detected / s_total.n_gt if s_total.n_gt else 0.0,
        }

    print("\nnoise breakdown — GT rides detected per noise class:")
    print(_detection_line(
        "overall", total.n_gt, total.missed,
        extra=(f"  [n_pred={total.n_pred} fp={total.fp} "
               f"f1*={total.score():.3f}]") if total.n_gt else "",
    ))
    for lbl in ("clean", "noisy"):
        d = by_noise[lbl]
        print(_detection_line(lbl, d["n_gt"], d["missed"]))

    metrics = {
        "kind": kind_label,
        "overall": _metrics_payload("overall", total, iou, len(per_exp)),
        "by_noise": by_noise,
        "cleaned": cleaned_payload,
    }
    return metrics


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="evaluateOnData",
        description="Reproducible segmentation evaluation: filter "
                    "experiments, run the active detector, and render "
                    "every evaluation figure into the run directory.",
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
        choices=[*VALID_SOURCES, "all"],
        help="Filter by metadata.source — repeat to allow multiple "
             "(e.g. --source experiment --source ido). Pass 'all' (or "
             "omit the flag) to keep every source.",
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
        help="Experiments excluded only from the *cleaned* aggregate "
             "(parallel of the 'after removing three outliers' panel). "
             "Pass an empty list to skip the cleaned figures entirely.",
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
    args = p.parse_args(argv)
    # 'all' is a convenience alias for "no source filter".
    if args.source and "all" in args.source:
        args.source = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm(args.algorithm))

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_root / (args.run_name or f"run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing run artefacts under {run_dir}")

    cleaned_exclude = set(args.cleaned_exclude or [])

    experiments = _resolve_experiments(
        kind=args.kind, sources=args.source,
        include=args.include, exclude=args.exclude,
    )
    if not experiments:
        print("no experiments survived filtering; nothing to do",
              file=sys.stderr)
        return 1

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
            "names": experiments,
            "n": len(experiments),
            "cleaned_exclude": sorted(cleaned_exclude),
        },
        "kind": args.kind,
    }
    (run_dir / "run_settings.json").write_text(
        json.dumps(settings, indent=2, default=str)
    )

    # --- run detection on the full resolved experiment set ---
    t0 = time.time()
    raw = evaluator._run_on_experiments(cfg, experiments, verbose=True)
    print(f"\ndetection finished in {time.time() - t0:.1f}s")

    phone_for_exp: dict[str, str] = {}
    for e in raw:
        meta = _experiment_metadata(e.name)
        phone_for_exp[e.name] = _phone_canonical(meta)

    # --- render flat into run_dir; metrics.json carries the noise split ---
    metrics = _render_run(
        run_dir, raw, args.kind, phone_for_exp, cleaned_exclude,
    )

    # --- constraint plots — config-only, independent of the data ---
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

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )

    print(f"\nartefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
