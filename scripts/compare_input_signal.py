"""Side-by-side: ``a_vert`` vs ``|a| − g`` as the matched-filter input.

Runs the ACC_TEMPLATE_MATCH segmentation detector across all TRAIN
experiments twice — once with the default gravity-projected ``a_vert``
and once with the rotation-invariant ``|a| − |ĝ|`` magnitude residual —
and prints the headline detection metrics side by side.

Motivation: in the field we see real-world sessions where the user
rotates the phone mid-ride; the frozen-gravity projection then
collapses the trapezoid signature, while the magnitude residual keeps
it. This script answers whether swapping the input signal recovers (or
loses) detection performance on the lab TRAIN set, where the phone is
mostly stationary and the two signals should be approximately equal.

Usage:
    venv/bin/python -m scripts.compare_input_signal
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import list_experiments  # noqa: E402
from src.segmentation.algorithms.configTypes import (  # noqa: E402
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
)
from src.segmentation.evaluate.evaluator import evaluate_algorithm  # noqa: E402


def _run(signal: str, experiments: list[str]):
    cfg = SEGMENT_ALGORITHM_CONFIG(
        algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
        overrides={"input_signal": signal},
    )
    return evaluate_algorithm(config=cfg, experiments=experiments, out_dir=None)


def _row(label: str, res) -> str:
    r = res.total.rates()
    iou = res.iou_metrics
    return (
        f"{label:20s} "
        f"gt={res.total.n_gt:3d} pred={res.total.n_pred:4d} "
        f"clean={res.total.clean:4d} miss={res.total.missed:3d} "
        f"merge={res.total.pred_merged:3d} split={res.total.gt_split:3d} "
        f"fp={res.total.fp:3d}  "
        f"f1={r['f1_like']:.3f} P={r['precision']:.3f} R={r['recall']:.3f}  "
        f"IoU_f1@0.5={iou['iou_f1@0.5']:.3f} "
        f"meanIoU={iou['iou_mean@0.5']:.3f}"
    )


def main() -> int:
    experiments = list_experiments(kind="train")
    print(f"evaluating {len(experiments)} TRAIN experiments\n")

    res_vert = _run("a_vert", experiments)
    res_mag = _run("a_mag_minus_g", experiments)

    print(_row("a_vert (baseline)", res_vert))
    print(_row("|a| - g          ", res_mag))

    d = lambda a, b: f"{(b - a):+.3f}"
    rv = res_vert.total.rates(); rm = res_mag.total.rates()
    iv = res_vert.iou_metrics;   im = res_mag.iou_metrics
    print("\nDelta (|a|-g  −  a_vert):")
    print(f"  f1_like      {d(rv['f1_like'],     rm['f1_like'])}")
    print(f"  precision    {d(rv['precision'],   rm['precision'])}")
    print(f"  recall       {d(rv['recall'],      rm['recall'])}")
    print(f"  IoU_f1@0.5   {d(iv['iou_f1@0.5'],  im['iou_f1@0.5'])}")
    print(f"  meanIoU@0.5  {d(iv['iou_mean@0.5'], im['iou_mean@0.5'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
