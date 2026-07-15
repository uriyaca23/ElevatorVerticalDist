"""CLI for the whole-signal trapezoid detector — sanity check.

Prints per-experiment gt/pred counts, writes nothing. This is the
application-side runner for the packaged detector
(:mod:`pyramidElevatorDist.segmentation...check_grid_across_signal.detect`).

Usage:
    venv/bin/python -m src.segmentation.run_detector
    venv/bin/python -m src.segmentation.run_detector --only <exp>
"""
from __future__ import annotations

import argparse
import sys

from src.data.loader import getExperimentData, list_experiments
from pyramidElevatorDist.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal.detect import (  # noqa: E501
    predict_intervals,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="process a single experiment by name")
    args = parser.parse_args()

    names = [args.only] if args.only else list_experiments(kind="train")
    print(f"running detector on {len(names)} experiments")
    total_gt = 0
    total_pred = 0
    for n in names:
        try:
            sensors, gt, _meta = getExperimentData(n)
        except Exception as exc:
            print(f"[error] {n}: {type(exc).__name__}: {exc}")
            continue
        preds, _state = predict_intervals(sensors.get("ACC"))
        n_gt = int(gt["type"].isin(("up", "down")).sum()) if gt is not None else 0
        total_gt += n_gt
        total_pred += len(preds)
        print(f"[ok]    {n}: gt={n_gt}  pred={len(preds)}")

    if names:
        print(
            f"\n{len(names)} experiments — "
            f"GT total {total_gt}, predicted total {total_pred} "
            f"(pred/gt = {total_pred / max(total_gt, 1):.2f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
