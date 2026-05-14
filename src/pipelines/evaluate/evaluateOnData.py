"""Reproducible end-to-end pipeline evaluation.

Runs every experiment surviving the requested filters through the
deployed segmentation + prediction stack, compares against barometer
truth, and renders the figures consumed by the *Pipeline* section of
``docs/latex/main.tex`` for both passes — full and accepted-only
(post-quality-filter).

Typical usage::

    venv/bin/python -m src.pipelines.evaluate.evaluateOnData
    venv/bin/python -m src.pipelines.evaluate.evaluateOnData \\
        --kind all --source experiment \\
        --exclude UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026

A timestamped ``run_YYYYMMDD-HHMMSS/`` directory is created under
``--out-root`` (default ``elevator_reports/pipeline_eval``):

* ``run_settings.json``      — every flag, resolved experiments,
                               active config dump.
* ``gt_records.csv`` /
  ``seg_records.csv``        — pooled per-GT and per-prediction rows.
* All pipeline figures (``cdf_pooled.png``, ``bar_mae_overall.png``,
  ``scatter_three.png``, ``signed_error_pdf.png``, ``fp_*.png``,
  ``clean_predicted_altitude.png``, ``per_exp_mae.png``,
  ``baro_vs_gt_sanity.png``).
* The same set with the ``_acc`` suffix for the accepted-only pass.
* ``metrics.json`` with the three views' summaries (full + accepted)
  and FP / accept-rate counts.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import (
    EXPERIMENT_TYPES,
    VALID_SOURCES,
    classify_experiment_type,
    getExperimentData,
    list_experiments,
)
from src.prediction.algorithms.configTypes import (
    PREDICT_ALGORITHM_CONFIG,
    PredictAlgorithm,
)
from src.segmentation.algorithms.configTypes import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
)

from .runner import (
    PipelineConfig,
    build_views,
    render_view_figures,
    run_all_experiments,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_ROOT = REPO_ROOT / "elevator_reports" / "pipeline_eval"
DEFAULT_CALIBRATION = (
    REPO_ROOT / "src" / "data" / "structuredData" / "test_results"
    / "prediction" / "train" / "calibration_trapezoid.json"
)
DEFAULT_LATEX_OUT = REPO_ROOT / "docs" / "latex" / "results_noise_pipeline.tex"

# (label, signal_clear) — slices gt_df/seg_df by gt-side signal_clear.
# FP preds (signal_clear is None) are dropped from clean/noisy passes
# and kept in "both".
NOISE_PASSES: list[tuple[str, "bool | None"]] = [
    ("clean", True),
    ("noisy", False),
    ("both",  None),
]


def _experiment_metadata(name: str) -> dict | None:
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
    candidates = list(include) if include else list_experiments(kind="all")
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


def _summary_block(gt_df: pd.DataFrame, seg_df: pd.DataFrame,
                   accepted_only: bool) -> dict:
    pooled = build_views(gt_df, seg_df, accepted_only=accepted_only)
    train  = build_views(gt_df[gt_df["kind"] == "train"],
                         seg_df[seg_df["kind"] == "train"],
                         accepted_only=accepted_only)
    test   = build_views(gt_df[gt_df["kind"] == "test"],
                         seg_df[seg_df["kind"] == "test"],
                         accepted_only=accepted_only)
    return {
        "pooled": {k: v["summary"] for k, v in pooled.items()},
        "train":  {k: v["summary"] for k, v in train.items()},
        "test":   {k: v["summary"] for k, v in test.items()},
    }


def _accept_stats(seg_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    n_seg_total   = int(len(seg_df))
    n_seg_acc     = int((seg_df["pred_accepted"] == True).sum())  # noqa: E712
    n_clean_total = int((seg_df["status"] == "clean").sum())
    n_clean_acc   = int(((seg_df["status"] == "clean")
                         & (seg_df["pred_accepted"] == True)).sum())  # noqa: E712
    n_fp_total    = int((seg_df["status"] == "fp").sum())
    n_fp_acc      = int(((seg_df["status"] == "fp")
                         & (seg_df["pred_accepted"] == True)).sum())  # noqa: E712
    n_gt_total    = int(len(gt_df))
    n_gt_acc      = int((gt_df["oracle_accepted"] == True).sum())  # noqa: E712
    pct = lambda a, b: float(100.0 * a / b) if b else 0.0
    return {
        "accept_rate_all":    pct(n_seg_acc, n_seg_total),
        "accept_rate_clean":  pct(n_clean_acc, n_clean_total),
        "accept_rate_fp":     pct(n_fp_acc, n_fp_total),
        "accept_rate_gt":     pct(n_gt_acc, n_gt_total),
        "n_seg_total": n_seg_total, "n_seg_acc": n_seg_acc,
        "n_clean_total": n_clean_total, "n_clean_acc": n_clean_acc,
        "n_fp_total": n_fp_total, "n_fp_acc": n_fp_acc,
        "n_gt_total": n_gt_total, "n_gt_acc": n_gt_acc,
    }


def _fp_stats(seg_df: pd.DataFrame) -> dict:
    fp = seg_df[(seg_df["status"] == "fp") & seg_df["pred_dh"].notna()]
    if fp.empty:
        return {"n": 0, "median_signed": 0.0, "median_abs": 0.0,
                "mean_abs": 0.0}
    return {
        "n": int(len(fp)),
        "median_signed": float(np.median(fp["pred_dh"])),
        "median_abs":    float(np.median(fp["pred_dh"].abs())),
        "mean_abs":      float(np.mean(fp["pred_dh"].abs())),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="evaluateOnData",
        description="Reproducible end-to-end pipeline evaluation: filter "
                    "experiments, run segmentation + prediction + barometer, "
                    "and render every figure consumed by the pipeline "
                    "section of docs/latex/main.tex.",
    )
    p.add_argument(
        "--seg-algorithm", default=SegmentAlgorithm.ACC_TEMPLATE_MATCH.value,
        choices=[a.value for a in SegmentAlgorithm],
        help="Segmentation algorithm (default acc_template_match).",
    )
    p.add_argument(
        "--pred-algorithm", default=PredictAlgorithm.TRAPEZOID_ACCEL.value,
        choices=[a.value for a in PredictAlgorithm],
        help="Prediction algorithm (default trapezoid_accel).",
    )
    p.add_argument(
        "--kind", default="all",
        choices=("all", *EXPERIMENT_TYPES),
        help="Restrict to train, test, or all experiments.",
    )
    p.add_argument(
        "--source", action="append", default=None,
        choices=list(VALID_SOURCES),
        help="Filter by metadata.source — repeatable.",
    )
    p.add_argument(
        "--include", nargs="*", default=None,
        help="Whitelist of experiment names.",
    )
    p.add_argument(
        "--exclude", nargs="*", default=None,
        help="Drop these experiment names from the run.",
    )
    p.add_argument(
        "--calibration-path", type=Path, default=DEFAULT_CALIBRATION,
        help="Conformal-calibration JSON to load onto the predictor "
             "(default: the trapezoid checkpoint produced by "
             "src.prediction.evaluation.evaluateOnData on the train half).",
    )
    p.add_argument(
        "--out-root", type=Path, default=DEFAULT_OUT_ROOT,
        help="Base directory; output → <out-root>/run_<timestamp>/.",
    )
    p.add_argument(
        "--run-name", default=None,
        help="Override the timestamp folder name.",
    )
    p.add_argument(
        "--latex-out", type=Path, default=DEFAULT_LATEX_OUT,
        help=(
            "LaTeX snippet path that main.tex \\inputs "
            f"(default: {DEFAULT_LATEX_OUT}). Pass empty string to skip."
        ),
    )
    return p.parse_args(argv)


def _latex_relpath(target: Path, latex_file: Path) -> str:
    try:
        return Path(target).resolve().relative_to(
            Path(latex_file).resolve().parent
        ).as_posix()
    except ValueError:
        try:
            return ("../" * len(latex_file.resolve().parent.relative_to(REPO_ROOT).parts)
                    + Path(target).resolve().relative_to(REPO_ROOT).as_posix())
        except ValueError:
            return Path(target).resolve().as_posix()


def _write_latex_snippet(
    latex_path: Path,
    run_dir: Path,
    args: argparse.Namespace,
    pass_dirs: dict[str, Path],
    pass_metrics: dict,
) -> None:
    if not latex_path:
        return
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    source = ", ".join(args.source) if args.source else "all sources"

    lines: list[str] = []
    lines.append(r"% Auto-generated by src.pipelines.evaluate.evaluateOnData.")
    lines.append(r"\section{Pipeline --- results on clean vs.\ noisy data}")
    lines.append(rf"\noindent\textit{{Source filter: {source}.\quad Kind: {args.kind}.\quad "
                 rf"Run: \texttt{{{run_dir.name}}}.}}")
    lines.append("")

    fig_keys = [
        ("cdf_pooled.png",           r"CDF of $|\Delta h|$ error"),
        ("scatter_three.png",        r"Pred vs.\ truth (3 views)"),
        ("coverage_vs_duration.png", r"Coverage by ride-duration bin"),
        ("err_vs_duration.png",      r"$\Delta h$ error by ride-duration bin"),
    ]
    for pass_name in ("clean", "noisy", "both"):
        if pass_name not in pass_dirs:
            continue
        pass_dir = pass_dirs[pass_name]
        summary = (pass_metrics.get(pass_name, {})
                                .get("full", {})
                                .get("pooled", {}).get("gt", {}))
        n = summary.get("n", 0)
        mae = summary.get("mae", float("nan"))
        med = summary.get("median", float("nan"))
        within = 100.0 * summary.get("p_within_1_5m", 0.0)
        lines.append(
            rf"\subsection*{{Noise subset: {pass_name} "
            rf"(GT view, n={n}, MAE={mae:.3f}\,m, median={med:.3f}\,m, "
            rf"$\le$1.5\,m={within:.1f}\%)}}"
        )
        lines.append(r"\begin{figure}[H]\centering")
        for fname, _ in fig_keys:
            img = pass_dir / fname
            if not img.exists():
                continue
            rel = _latex_relpath(img, latex_path)
            lines.append(
                rf"  \includegraphics[width=0.48\linewidth]{{{rel}}}\hfill"
            )
        lines.append(rf"  \caption{{Pipeline, {pass_name} subset --- CDF, "
                     rf"scatter, coverage and error by ride-duration bin.}}")
        lines.append(rf"  \label{{fig:pipe-noise-{pass_name}}}")
        lines.append(r"\end{figure}")
        lines.append("")

    latex_path.write_text("\n".join(lines))
    print(f"\nLaTeX snippet → {latex_path}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_root / (args.run_name or f"run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing run artefacts under {run_dir}")

    seg_cfg  = SEGMENT_ALGORITHM_CONFIG(
        algorithm=SegmentAlgorithm(args.seg_algorithm))
    pred_cfg = PREDICT_ALGORITHM_CONFIG(
        algorithm=PredictAlgorithm(args.pred_algorithm))
    calibration_path = (
        args.calibration_path
        if args.calibration_path and Path(args.calibration_path).exists()
        else None
    )
    config = PipelineConfig(
        seg_cfg=seg_cfg, pred_cfg=pred_cfg,
        calibration_path=calibration_path,
    )

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
        "configs": {
            "segmentation": {
                "algorithm": seg_cfg.algorithm.value,
                "config_path": str(seg_cfg.config_path),
                "overrides": seg_cfg.overrides,
                "active_params": seg_cfg.load_params(),
            },
            "prediction": {
                "algorithm": pred_cfg.algorithm.value,
                "config_path": str(pred_cfg.config_path),
                "overrides": pred_cfg.overrides,
                "active_params": pred_cfg.load_params(),
                "calibration_path":
                    str(calibration_path) if calibration_path else None,
            },
        },
        "experiments": {
            "n": len(experiments),
            "names": experiments,
            "n_train": sum(1 for e in experiments
                            if classify_experiment_type(e) == "train"),
            "n_test":  sum(1 for e in experiments
                            if classify_experiment_type(e) == "test"),
        },
    }
    (run_dir / "run_settings.json").write_text(
        json.dumps(settings, indent=2, default=str)
    )

    # --- run pipeline ---
    print(f"\nrunning pipeline on {len(experiments)} experiments")
    t0 = time.time()
    gt_df, seg_df = run_all_experiments(experiments, config, verbose=True)
    print(f"\npipeline finished in {time.time() - t0:.1f}s "
          f"({len(gt_df)} GT rows / {len(seg_df)} pred rows)")

    if gt_df.empty and seg_df.empty:
        print("no data after pipeline; nothing to render", file=sys.stderr)
        return 2

    gt_df.to_csv(run_dir  / "gt_records.csv", index=False)
    seg_df.to_csv(run_dir / "seg_records.csv", index=False)

    # --- 3-pass noise loop ---
    pass_dirs: dict[str, Path] = {}
    pass_metrics: dict[str, dict] = {}
    for pass_name, sc in NOISE_PASSES:
        pass_dir = run_dir / pass_name
        pass_dir.mkdir(parents=True, exist_ok=True)
        pass_dirs[pass_name] = pass_dir

        # Slice gt/seg to this noise polarity. FP preds (signal_clear is
        # None) survive only the "both" pass.
        if sc is None:
            gt_pass = gt_df
            seg_pass = seg_df
        else:
            gt_pass  = gt_df[gt_df["signal_clear"] == sc]
            seg_pass = seg_df[seg_df["signal_clear"] == sc]
        n_gt, n_seg = len(gt_pass), len(seg_pass)
        print(f"\n=== noise pass: {pass_name}  "
              f"(gt={n_gt}, pred={n_seg}) ===")
        if n_gt == 0 and n_seg == 0:
            pass_metrics[pass_name] = {"note": "empty subset"}
            continue

        gt_pass.to_csv(pass_dir / "gt_records.csv", index=False)
        seg_pass.to_csv(pass_dir / "seg_records.csv", index=False)

        print(f"[{pass_name}] rendering pipeline figures (full pass)")
        render_view_figures(gt_pass, seg_pass, pass_dir, suffix="")

        print(f"[{pass_name}] rendering pipeline figures (accepted-only pass)")
        gt_acc  = gt_pass[gt_pass["oracle_accepted"] == True]   # noqa: E712
        seg_acc = seg_pass[seg_pass["pred_accepted"] == True]   # noqa: E712
        render_view_figures(gt_acc, seg_acc, pass_dir, suffix="_acc")

        metrics = {
            "full":          _summary_block(gt_pass, seg_pass, accepted_only=False),
            "accepted_only": _summary_block(gt_pass, seg_pass, accepted_only=True),
            "fp_stats":      _fp_stats(seg_pass),
            "fp_stats_acc":  _fp_stats(seg_acc),
            "accept_stats":  _accept_stats(seg_pass, gt_pass),
        }
        (pass_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str)
        )
        pass_metrics[pass_name] = metrics

        print(f"[{pass_name}] summary (pooled, full pass):")
        for view, s in metrics["full"]["pooled"].items():
            print(f"  {view:8s}: n={s['n']:4d}  MAE={s['mae']:.3f} m  "
                  f"median={s['median']:.3f} m  rmse={s['rmse']:.3f} m  "
                  f"<=1.5m={100*s['p_within_1_5m']:.1f}%")
        fp = metrics["fp_stats"]
        print(f"  fps: n={fp['n']}  |dh| median={fp['median_abs']:.2f} m "
              f"mean={fp['mean_abs']:.2f} m")

    (run_dir / "metrics.json").write_text(
        json.dumps(pass_metrics, indent=2, default=str)
    )

    if str(args.latex_out):
        _write_latex_snippet(args.latex_out, run_dir, args,
                             pass_dirs, pass_metrics)

    print(f"\nartefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
