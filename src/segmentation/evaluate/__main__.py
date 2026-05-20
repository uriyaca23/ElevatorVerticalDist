"""CLI wrapper.

Single config:
    venv/bin/python -m src.segmentation.evaluate \\
        --algorithm pressure_filter \\
        --out-dir elevator_reports/seg_eval

Grid sweep (grid supplied as JSON):
    venv/bin/python -m src.segmentation.evaluate \\
        --algorithm pressure_filter --sweep grid.json \\
        --out-csv elevator_reports/seg_sweep.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.loader import resolve_experiments
from src.segmentation.algorithms.configTypes import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
)

from .evaluator import evaluate_algorithm, sweep_hyperparameters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", default=SegmentAlgorithm.PRESSURE_FILTER.value,
        choices=[a.value for a in SegmentAlgorithm],
    )
    parser.add_argument("--kind", default="train",
                        choices=("train", "test", "all"))
    parser.add_argument("--only", help="evaluate a single experiment by name")
    parser.add_argument("--out-dir", type=Path,
                        help="write CDF + failure-mode plots here")
    parser.add_argument("--sweep", type=Path,
                        help="JSON file with {param: [values, ...]} grid")
    parser.add_argument("--out-csv", type=Path,
                        help="CSV path for sweep results (sorted by f1_like)")
    parser.add_argument("--top", type=int, default=20,
                        help="print this many rows after a sweep")
    parser.add_argument("--phone-model", default=None,
                        help="force a single phone model for the chip-spec "
                             "amplitude floor (acc_template_match only). "
                             "When omitted, each experiment's metadata is "
                             "consulted; unknown phones keep the hard-coded "
                             "floor.")
    args = parser.parse_args()

    cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm(args.algorithm))
    experiments = [args.only] if args.only else resolve_experiments(kind=args.kind)
    if not experiments:
        print("no experiments found", file=sys.stderr)
        return 1

    if args.sweep is not None:
        grid = json.loads(args.sweep.read_text())
        df = sweep_hyperparameters(
            base_config=cfg, param_grid=grid,
            experiments=experiments, out_csv=args.out_csv,
            phone_model=args.phone_model,
        )
        print(df.head(args.top).to_string(index=False))
        if args.out_csv is not None:
            print(f"\nwrote full sweep → {args.out_csv}")
        return 0

    result = evaluate_algorithm(
        config=cfg, experiments=experiments, out_dir=args.out_dir,
        phone_model=args.phone_model,
    )
    header = (
        f"{'exp':60s} {'gt':>3s} {'pred':>4s} {'clean':>5s} "
        f"{'miss':>4s} {'merge':>5s} {'split':>5s} {'fp':>3s}"
    )
    print(header)
    print("-" * len(header))
    for name, m in result.per_exp:
        short = name if len(name) <= 60 else name[:57] + "..."
        print(
            f"{short:60s} {m.n_gt:3d} {m.n_pred:4d} "
            f"{m.clean:5d} {m.missed:4d} "
            f"{m.pred_merged:5d} {m.gt_split:5d} {m.fp:3d}"
        )
    r = result.total.rates()
    print(
        f"\nTOTAL: gt={result.total.n_gt} pred={result.total.n_pred} "
        f"clean={result.total.clean} miss={result.total.missed} "
        f"merge={result.total.pred_merged} split={result.total.gt_split} "
        f"fp={result.total.fp}"
    )
    print(
        f"RATES: f1_like={r['f1_like']:.3f} recall={r['recall']:.3f} "
        f"precision={r['precision']:.3f}"
    )
    iou = result.iou_metrics
    print(
        f"IOU  : f1@0.5={iou['iou_f1@0.5']:.3f} "
        f"p@0.5={iou['iou_precision@0.5']:.3f} "
        f"r@0.5={iou['iou_recall@0.5']:.3f} "
        f"mean_iou={iou['iou_mean@0.5']:.3f}"
    )
    if result.plot_paths:
        print("\nplots:")
        for k, p in result.plot_paths.items():
            print(f"  {k:15s} {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
