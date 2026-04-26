"""End-to-end driver for the LaTeX segmentation-evaluation subsection.

Reads the active iter_16 config from
``src/segmentation/algorithms/config.json``, runs the active detector
across the train / test splits, and writes every artifact the
``\\subsection{Evaluation}`` block in ``docs/latex/main.tex`` consumes:

* ``docs/latex/figures/seg_eval/{train,test}/`` — CDFs and failure-mode
  bars produced by the live evaluator (:func:`evaluate_algorithm`).
* ``docs/latex/figures/seg_eval/`` — combined figures (train-vs-test
  bars, per-experiment stacks, edge-quality CDF+PDF pairs, scatters,
  phone-model breakdown, three picked per-experiment timelines).
* ``docs/latex/figures/seg_eval/results_macros.tex`` — every inline
  number quoted in the LaTeX, namespaced ``\\Seg{Train,Test,Pooled}…``.
* ``docs/latex/figures/seg_eval/results_table_{train,test}.tex`` —
  headline booktabs tables.
* ``docs/latex/figures/seg_eval/per_exp_{train,test}.tex`` — wide
  per-experiment tables.

CLI: ``python scripts/segmentation_evaluation_report.py``
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make ``src.…`` importable when invoked from any cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import getExperimentData, list_experiments  # noqa: E402
from src.segmentation.algorithms.configTypes import (  # noqa: E402
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
)
from src.segmentation.algorithms.metrics import (  # noqa: E402
    IntervalPredictionMetrics,
)
from src.segmentation.evaluate import evaluator  # noqa: E402
from src.segmentation.evaluate import report_plots  # noqa: E402


OUT_ROOT = REPO_ROOT / "docs" / "latex" / "figures" / "seg_eval"

# Two recordings the user asked us to set aside as outliers in the
# "after removing bad experiments" subsubsection. Both are damped /
# weak-signal Pixel\,10 sessions where the lobes never form at the
# deployed amplitude floor — they are exactly the rides flagged in
# Section~\ref{sec:trapezoid-detect-tuning} as needing a per-session
# noise-scaled floor. They stay in the headline numbers; the cleaned
# section only re-aggregates to show the operating point on the rest
# of the dataset.
EXCLUDED_EXPERIMENTS = {
    "UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026",
    "UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2",
}

ITER16_PER_GT_CSV = (
    REPO_ROOT
    / "src" / "segmentation" / "algorithms" / "improvement_iterations"
    / "iter_16_lower_peak_a" / "per_gt.csv"
)


# --------------------------------------------------------------------------
# Naming helpers
# --------------------------------------------------------------------------
def _short_label(exp_name: str) -> str:
    """Compact LaTeX-safe label for a row in the per-exp table.

    Mirrors the prediction section's `EY/Mansour1/S23` style: shrink the
    experimenter prefix to initials, the location to a tag, and the phone
    to the last token.
    """
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
        metadata.get("phone")
        or metadata.get("phone_model")
        or metadata.get("model")
        or metadata.get("device_model")
        or "unknown"
    )
    s = str(raw)
    if "Pixel" in s:
        return "Google Pixel 10"
    if "Flip6" in s or "F931B" in s:
        return "Galaxy Z Flip6"
    if "SM-S911B" in s:
        return "Galaxy S23"
    if "SM-A235F" in s:
        return "Galaxy A23"
    if "Xiaomi" in s:
        return "Xiaomi 22101320I"
    return s


# --------------------------------------------------------------------------
# Per-split runner — wraps evaluator.evaluate_algorithm and keeps the
# raw _ExpResult list so we can reuse it for the timeline plots without
# re-running detection.
# --------------------------------------------------------------------------
def _evaluate_split(
    cfg: SEGMENT_ALGORITHM_CONFIG,
    kind: str,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    exps = list_experiments(kind=kind)
    print(f"\n[{kind}] {len(exps)} experiments")
    raw = evaluator._run_on_experiments(cfg, exps, verbose=True)

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

    # Drop the live evaluator's standard CDF + failure-mode bar set.
    from src.segmentation.evaluate import plots as live_plots
    live_plots.render_all(matched_pairs, total, out_dir)

    metrics = {
        "kind": kind,
        "n_experiments": len(per_exp),
        "total": total.as_dict(),
        "iou": iou_metrics,
        "per_exp": [(name, m.as_dict()) for name, m in per_exp],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return raw, per_exp, total, iou_metrics, matched_pairs


# --------------------------------------------------------------------------
# LaTeX emitters
# --------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}\\%"


def _fmt_ratio(x: float) -> str:
    return f"{x:.3f}"


def _aggregate_filtered(
    raw_results: list,
    exclude: set[str],
):
    """Re-aggregate a list of ``_ExpResult`` skipping ``exclude`` names.

    Returns ``(per_exp, total, iou_metrics, matched_pairs)`` over the
    surviving experiments — same shape as one
    :func:`_evaluate_split` half so downstream emitters can treat both
    paths uniformly.
    """
    kept = [e for e in raw_results if e.name not in exclude]
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
    matched_pairs = evaluator._collect_matched_pairs(kept)
    return per_exp, total, iou, matched_pairs


def _macro_block(prefix: str, total: IntervalPredictionMetrics,
                 iou: dict[str, float], n_exp: int) -> list[str]:
    rates = total.rates()
    rows = [
        ("NExp",        str(n_exp)),
        ("NGt",         str(total.n_gt)),
        ("NPred",       str(total.n_pred)),
        ("Clean",       str(total.clean)),
        ("Missed",      str(total.missed)),
        ("Fp",          str(total.fp)),
        ("GtMerged",    str(total.gt_merged)),
        ("GtSplit",     str(total.gt_split)),
        ("PredMerged",  str(total.pred_merged)),
        ("PredSplit",   str(total.pred_split_part)),
        ("FOne",        _fmt_ratio(rates["f1_like"])),
        ("FOnePct",     _fmt_pct(rates["f1_like"])),
        ("Recall",      _fmt_pct(rates["recall"])),
        ("Precision",   _fmt_pct(rates["precision"])),
        ("MissRate",    _fmt_pct(rates["miss_rate"])),
        ("FpRate",      _fmt_pct(rates["fp_rate"])),
        ("MergeRate",   _fmt_pct(rates["merge_rate"])),
        ("IoUFOne",     _fmt_ratio(iou["iou_f1@0.5"])),
        ("IoUPrec",     _fmt_pct(iou["iou_precision@0.5"])),
        ("IoURecall",   _fmt_pct(iou["iou_recall@0.5"])),
        ("IoUMean",     _fmt_ratio(iou["iou_mean@0.5"])),
        ("IoUTp",       str(int(iou["iou_tp@0.5"]))),
        ("IoUFp",       str(int(iou["iou_fp@0.5"]))),
        ("IoUFn",       str(int(iou["iou_fn@0.5"]))),
    ]
    return [f"\\newcommand{{\\Seg{prefix}{name}}}{{{val}}}"
            for name, val in rows]


def write_macros_file(
    out_path: Path,
    train_total: IntervalPredictionMetrics,
    train_iou:   dict[str, float],
    train_n:     int,
    test_total:  IntervalPredictionMetrics,
    test_iou:    dict[str, float],
    test_n:      int,
    cleaned_train_total: IntervalPredictionMetrics | None = None,
    cleaned_train_iou:   dict[str, float] | None = None,
    cleaned_train_n:     int | None = None,
    cleaned_test_total:  IntervalPredictionMetrics | None = None,
    cleaned_test_iou:    dict[str, float] | None = None,
    cleaned_test_n:      int | None = None,
) -> None:
    pooled = train_total + test_total
    pooled_gt: list[dict] = []
    pooled_pred: list[dict] = []
    # Pool IoU from the union of train+test pooled intervals — we re-derive
    # below using the iter_16 reference number for consistency.
    pooled_iou = {
        "iou_f1@0.5":        (train_iou["iou_f1@0.5"]        + test_iou["iou_f1@0.5"])        / 2.0,
        "iou_precision@0.5": (train_iou["iou_precision@0.5"] + test_iou["iou_precision@0.5"]) / 2.0,
        "iou_recall@0.5":    (train_iou["iou_recall@0.5"]    + test_iou["iou_recall@0.5"])    / 2.0,
        "iou_mean@0.5":      (train_iou["iou_mean@0.5"]      + test_iou["iou_mean@0.5"])      / 2.0,
        "iou_tp@0.5": train_iou["iou_tp@0.5"] + test_iou["iou_tp@0.5"],
        "iou_fp@0.5": train_iou["iou_fp@0.5"] + test_iou["iou_fp@0.5"],
        "iou_fn@0.5": train_iou["iou_fn@0.5"] + test_iou["iou_fn@0.5"],
    }
    lines = ["% Auto-generated by scripts/segmentation_evaluation_report.py"]
    lines.extend(_macro_block("Train",  train_total, train_iou,   train_n))
    lines.extend(_macro_block("Test",   test_total,  test_iou,    test_n))
    lines.extend(_macro_block("Pooled", pooled,      pooled_iou,  train_n + test_n))

    if (cleaned_train_total is not None and cleaned_train_iou is not None
            and cleaned_train_n is not None
            and cleaned_test_total is not None and cleaned_test_iou is not None
            and cleaned_test_n is not None):
        cleaned_pooled = cleaned_train_total + cleaned_test_total
        cleaned_pooled_iou = {
            "iou_f1@0.5":        (cleaned_train_iou["iou_f1@0.5"]        + cleaned_test_iou["iou_f1@0.5"])        / 2.0,
            "iou_precision@0.5": (cleaned_train_iou["iou_precision@0.5"] + cleaned_test_iou["iou_precision@0.5"]) / 2.0,
            "iou_recall@0.5":    (cleaned_train_iou["iou_recall@0.5"]    + cleaned_test_iou["iou_recall@0.5"])    / 2.0,
            "iou_mean@0.5":      (cleaned_train_iou["iou_mean@0.5"]      + cleaned_test_iou["iou_mean@0.5"])      / 2.0,
            "iou_tp@0.5": cleaned_train_iou["iou_tp@0.5"] + cleaned_test_iou["iou_tp@0.5"],
            "iou_fp@0.5": cleaned_train_iou["iou_fp@0.5"] + cleaned_test_iou["iou_fp@0.5"],
            "iou_fn@0.5": cleaned_train_iou["iou_fn@0.5"] + cleaned_test_iou["iou_fn@0.5"],
        }
        lines.extend(_macro_block(
            "CleanTrain", cleaned_train_total, cleaned_train_iou, cleaned_train_n,
        ))
        lines.extend(_macro_block(
            "CleanTest", cleaned_test_total, cleaned_test_iou, cleaned_test_n,
        ))
        lines.extend(_macro_block(
            "CleanPooled", cleaned_pooled, cleaned_pooled_iou,
            cleaned_train_n + cleaned_test_n,
        ))
        # Convenience deltas vs the unfiltered pooled headline so the
        # LaTeX can quote the lift in one inline.
        full_pooled = train_total + test_total
        delta_clean = cleaned_pooled.clean - full_pooled.clean
        delta_miss  = cleaned_pooled.missed - full_pooled.missed
        lines.append(
            f"\\newcommand{{\\SegCleanGtRemoved}}"
            f"{{{full_pooled.n_gt - cleaned_pooled.n_gt}}}"
        )
        lines.append(
            f"\\newcommand{{\\SegCleanExpRemoved}}"
            f"{{{(train_n + test_n) - (cleaned_train_n + cleaned_test_n)}}}"
        )
        lines.append(
            f"\\newcommand{{\\SegCleanFOneLift}}"
            f"{{{cleaned_pooled.score() - full_pooled.score():+.3f}}}"
        )
        lines.append(
            f"\\newcommand{{\\SegCleanIoUFOneLift}}"
            f"{{{cleaned_pooled_iou['iou_f1@0.5'] - pooled_iou['iou_f1@0.5']:+.3f}}}"
        )

    out_path.write_text("\n".join(lines) + "\n")


def write_headline_table(
    out_path: Path,
    label_split: str,
    total: IntervalPredictionMetrics,
    iou: dict[str, float],
) -> None:
    rates = total.rates()
    rows = (
        f"\\resizebox{{\\textwidth}}{{!}}{{%\n"
        f"\\begin{{tabular}}{{lrrrrrrrrr}}\n"
        f"\\toprule\n"
        f"Split & $n_{{\\rm gt}}$ & $n_{{\\rm pred}}$ & "
        f"clean & missed & FP & merged & split & "
        f"$F1^\\ast$ & IoU-F1@0.5 \\\\\n"
        f"\\midrule\n"
        f"\\textbf{{{label_split}}} & {total.n_gt} & {total.n_pred} & "
        f"{total.clean} & {total.missed} & {total.fp} & "
        f"{total.pred_merged} & {total.gt_split} & "
        f"{rates['f1_like']:.3f} & {iou['iou_f1@0.5']:.3f} \\\\\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}%\n"
        f"}}\n"
    )
    out_path.write_text(
        f"% Auto-generated headline metrics for the {label_split} split\n"
        + rows
    )


def write_per_exp_table(
    out_path: Path,
    label_split: str,
    per_exp: list[tuple[str, IntervalPredictionMetrics]],
) -> None:
    lines = [
        f"% Auto-generated per-experiment table ({label_split})",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"\textbf{Experiment} & $n_{\rm gt}$ & $n_{\rm pred}$ & "
        r"clean & missed & FP & merged & split & "
        r"$F1^\ast$ & recall & precision \\",
        r"\midrule",
    ]
    items = sorted(per_exp, key=lambda kv: kv[1].n_gt, reverse=True)
    for name, m in items:
        r = m.rates()
        label = _short_label(name).replace("_", r"\_")
        lines.append(
            f"\\texttt{{{label}}} & {m.n_gt} & {m.n_pred} & "
            f"{m.clean} & {m.missed} & {m.fp} & "
            f"{m.pred_merged} & {m.gt_split} & "
            f"{r['f1_like']:.2f} & {_fmt_pct(r['recall'])} & "
            f"{_fmt_pct(r['precision'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    out_path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Timeline picker — choose three representative experiments
# --------------------------------------------------------------------------
def _pick_timeline_examples(
    train_raw, train_per_exp,
    test_raw,  test_per_exp,
) -> list[tuple[str, str, list]]:
    """Return [(label, exp_name, _ExpResult), …] for the timeline plots.

    Picks: best train (highest f1_like with n_gt>=8), typical train
    (closest to median f1_like), worst-case (lowest f1_like).
    """
    by_name = {e.name: e for e in (train_raw + test_raw)}
    all_pe = train_per_exp + test_per_exp
    scored = [(name, m, m.rates()["f1_like"]) for name, m in all_pe]
    scored = [s for s in scored if s[1].n_gt >= 6]
    if not scored:
        scored = [(name, m, m.rates()["f1_like"]) for name, m in all_pe]

    by_score = sorted(scored, key=lambda t: t[2])
    worst = by_score[0]
    best  = by_score[-1]
    median_idx = len(by_score) // 2
    typical = by_score[median_idx]

    out = []
    for label, item in [("best", best), ("typical", typical), ("worst", worst)]:
        name = item[0]
        if name in by_name:
            out.append((label, name, by_name[name]))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"writing artifacts under {OUT_ROOT}")
    cfg = SEGMENT_ALGORITHM_CONFIG(
        algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
    )

    t0 = time.time()
    train_raw, train_per_exp, train_total, train_iou, train_pairs = \
        _evaluate_split(cfg, "train", OUT_ROOT / "train")
    test_raw,  test_per_exp,  test_total,  test_iou,  test_pairs  = \
        _evaluate_split(cfg, "test",  OUT_ROOT / "test")
    print(f"\ndetection finished in {time.time() - t0:.1f}s")

    pooled_pairs = train_pairs + test_pairs
    pooled_per_exp = train_per_exp + test_per_exp
    pooled_total = train_total + test_total
    pooled_raw = train_raw + test_raw

    # Phone-model lookup (used by the breakdown bar).
    phone_for_exp: dict[str, str] = {}
    for e in pooled_raw:
        try:
            _, _, metadata = getExperimentData(e.name)
        except Exception:
            metadata = None
        phone_for_exp[e.name] = _phone_canonical(metadata)

    # ----- combined figures -----
    print("\nrendering combined figures")
    report_plots.failure_modes_split_bar(
        train_total, test_total,
        OUT_ROOT / "failure_modes_train_vs_test.png",
    )
    report_plots.per_experiment_failure_bar(
        pooled_per_exp,
        OUT_ROOT / "per_experiment_failure_bar.png",
        label_short={n: _short_label(n) for n, _ in pooled_per_exp},
    )
    report_plots.cdf_pdf_pair(
        [p["iou"] for p in pooled_pairs],
        title="IoU over matched pairs",
        xlabel="IoU",
        out_path=OUT_ROOT / "cdf_pdf_iou.png",
    )
    report_plots.cdf_pdf_pair(
        [p["duration_error_s"] for p in pooled_pairs],
        title="Duration error  (pred $-$ gt)",
        xlabel="duration error (s)",
        out_path=OUT_ROOT / "cdf_pdf_duration_error.png",
    )
    report_plots.cdf_pdf_pair(
        [p["start_residual_s"] for p in pooled_pairs],
        title="Start-edge residual  (pred $-$ gt)",
        xlabel="start residual (s)",
        out_path=OUT_ROOT / "cdf_pdf_start_residual.png",
    )
    report_plots.cdf_pdf_pair(
        [p["end_residual_s"] for p in pooled_pairs],
        title="End-edge residual  (pred $-$ gt)",
        xlabel="end residual (s)",
        out_path=OUT_ROOT / "cdf_pdf_end_residual.png",
    )
    report_plots.iou_vs_duration_scatter(
        pooled_pairs, OUT_ROOT / "iou_vs_duration.png",
    )
    report_plots.pred_vs_gt_duration_scatter(
        pooled_pairs, OUT_ROOT / "pred_vs_gt_duration.png",
    )
    report_plots.phone_breakdown_bar(
        pooled_per_exp, phone_for_exp,
        OUT_ROOT / "phone_breakdown.png",
    )

    # ----- per-experiment timelines -----
    print("\nrendering per-experiment timeline picks")
    picks = _pick_timeline_examples(train_raw, train_per_exp,
                                    test_raw,  test_per_exp)
    for label, name, exp_result in picks:
        try:
            sensors, _, _ = getExperimentData(name)
        except Exception as exc:
            print(f"  [skip] timeline_{label}: {name} → {exc}")
            continue
        acc = sensors.get("ACC")
        if acc is None or acc.empty:
            continue
        out_path = OUT_ROOT / f"timeline_{label}.png"
        report_plots.per_experiment_timeline(
            name, acc, exp_result.gt_rides, exp_result.preds, out_path,
        )
        print(f"  timeline_{label}: {name}")

    # ----- LaTeX tables + macros -----
    print("\nwriting LaTeX tables and macros")
    write_macros_file(
        OUT_ROOT / "results_macros.tex",
        train_total, train_iou, len(train_per_exp),
        test_total,  test_iou,  len(test_per_exp),
    )
    write_headline_table(
        OUT_ROOT / "results_table_train.tex", "train",
        train_total, train_iou,
    )
    write_headline_table(
        OUT_ROOT / "results_table_test.tex",  "test",
        test_total,  test_iou,
    )
    write_per_exp_table(
        OUT_ROOT / "per_exp_train.tex", "train", train_per_exp,
    )
    write_per_exp_table(
        OUT_ROOT / "per_exp_test.tex",  "test",  test_per_exp,
    )

    # ----- cleaned-set aggregates (drop EXCLUDED_EXPERIMENTS) -----
    print("\nbuilding cleaned-set artifacts (excluding outlier experiments)")
    print(f"  excluded: {sorted(EXCLUDED_EXPERIMENTS)}")
    (clean_train_per_exp, clean_train_total,
     clean_train_iou, clean_train_pairs) = _aggregate_filtered(
        train_raw, EXCLUDED_EXPERIMENTS,
    )
    (clean_test_per_exp,  clean_test_total,
     clean_test_iou,  clean_test_pairs) = _aggregate_filtered(
        test_raw, EXCLUDED_EXPERIMENTS,
    )
    clean_pooled_pairs = clean_train_pairs + clean_test_pairs
    clean_pooled_per_exp = clean_train_per_exp + clean_test_per_exp

    write_headline_table(
        OUT_ROOT / "results_table_train_cleaned.tex", "train (cleaned)",
        clean_train_total, clean_train_iou,
    )
    write_headline_table(
        OUT_ROOT / "results_table_test_cleaned.tex",  "test (cleaned)",
        clean_test_total,  clean_test_iou,
    )
    write_per_exp_table(
        OUT_ROOT / "per_exp_train_cleaned.tex", "train (cleaned)",
        clean_train_per_exp,
    )
    write_per_exp_table(
        OUT_ROOT / "per_exp_test_cleaned.tex", "test (cleaned)",
        clean_test_per_exp,
    )
    report_plots.failure_modes_split_bar(
        clean_train_total, clean_test_total,
        OUT_ROOT / "failure_modes_train_vs_test_cleaned.png",
    )
    report_plots.per_experiment_failure_bar(
        clean_pooled_per_exp,
        OUT_ROOT / "per_experiment_failure_bar_cleaned.png",
        label_short={n: _short_label(n) for n, _ in clean_pooled_per_exp},
    )
    report_plots.cdf_pdf_pair(
        [p["iou"] for p in clean_pooled_pairs],
        title="IoU over matched pairs (cleaned set)",
        xlabel="IoU",
        out_path=OUT_ROOT / "cdf_pdf_iou_cleaned.png",
    )
    report_plots.iou_vs_duration_scatter(
        clean_pooled_pairs,
        OUT_ROOT / "iou_vs_duration_cleaned.png",
    )

    # Re-write the macros file with the cleaned-set block appended.
    write_macros_file(
        OUT_ROOT / "results_macros.tex",
        train_total, train_iou, len(train_per_exp),
        test_total,  test_iou,  len(test_per_exp),
        cleaned_train_total=clean_train_total,
        cleaned_train_iou=clean_train_iou,
        cleaned_train_n=len(clean_train_per_exp),
        cleaned_test_total=clean_test_total,
        cleaned_test_iou=clean_test_iou,
        cleaned_test_n=len(clean_test_per_exp),
    )

    # ----- constraint-justification plots from iter_16 per_gt.csv -----
    if ITER16_PER_GT_CSV.exists():
        print(f"\nrendering constraint plots from {ITER16_PER_GT_CSV.name}")
        import pandas as pd
        per_gt = pd.read_csv(ITER16_PER_GT_CSV)
        # Active iter_16 thresholds (mirror config.json + configTypes.py).
        cfg_active = SEGMENT_ALGORITHM_CONFIG(
            algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
        ).load_params()

        report_plots.score_hist_by_status(
            per_gt, "pair_joint_r2",
            threshold=cfg_active["joint_r2_thresh"],
            title="Pair joint $R^2$ by GT status",
            xlabel="pair_joint_r2",
            out_path=OUT_ROOT / "constraint_pair_joint_r2.png",
        )
        report_plots.score_hist_by_status(
            per_gt, "pair_A_abs",
            threshold=cfg_active["min_pair_abs_a"],
            title="Pair amplitude $|A|$ by GT status",
            xlabel="pair_A_abs (m/s²)",
            out_path=OUT_ROOT / "constraint_pair_A.png",
        )
        report_plots.score_hist_by_status(
            per_gt, "pair_heatmap_energy",
            threshold=cfg_active["heatmap_energy_thresh"],
            title="Heatmap energy by GT status",
            xlabel="pair_heatmap_energy",
            out_path=OUT_ROOT / "constraint_heatmap_energy.png",
        )
        report_plots.peak_score_hist_combined(
            per_gt, "pos_r2", "neg_r2",
            threshold=cfg_active["r2_peak_thresh"],
            title="Peak signed $R^2$ by GT status (both lobes pooled)",
            xlabel="peak signed $R^2$",
            out_path=OUT_ROOT / "constraint_peak_R2.png",
        )
        report_plots.peak_score_hist_combined(
            per_gt, "pos_A", "neg_A",
            threshold=cfg_active["min_peak_abs_a"],
            title="Peak amplitude $|A|$ by GT status (both lobes pooled)",
            xlabel="peak $|A|$ (m/s²)",
            take_abs=True,
            out_path=OUT_ROOT / "constraint_peak_A.png",
        )
        report_plots.constraint_2d_scatter(
            per_gt,
            x_col="pair_joint_r2",
            y_col="pair_A_abs",
            x_threshold=cfg_active["joint_r2_thresh"],
            y_threshold=cfg_active["min_pair_abs_a"],
            title="Pair joint $R^2$ vs pair $|A|$",
            xlabel="pair_joint_r2",
            ylabel="pair_A_abs (m/s²)",
            out_path=OUT_ROOT / "constraint_jointR2_vs_pairA.png",
        )
        report_plots.reject_reason_bar(
            per_gt, OUT_ROOT / "constraint_reject_reasons.png",
        )
    else:
        print(f"  [skip] {ITER16_PER_GT_CSV} not found")

    # ----- summary printout -----
    print("\nsummary")
    print(f"  train: gt={train_total.n_gt} pred={train_total.n_pred} "
          f"clean={train_total.clean} miss={train_total.missed} "
          f"fp={train_total.fp} f1_like={train_total.score():.3f} "
          f"iou_f1={train_iou['iou_f1@0.5']:.3f}")
    print(f"  test : gt={test_total.n_gt} pred={test_total.n_pred} "
          f"clean={test_total.clean} miss={test_total.missed} "
          f"fp={test_total.fp} f1_like={test_total.score():.3f} "
          f"iou_f1={test_iou['iou_f1@0.5']:.3f}")
    print(f"  pooled: gt={pooled_total.n_gt} pred={pooled_total.n_pred} "
          f"clean={pooled_total.clean} miss={pooled_total.missed} "
          f"fp={pooled_total.fp} f1_like={pooled_total.score():.3f}")
    print(f"\nartifacts under: {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
