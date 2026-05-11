"""End-to-end pipeline comparison: ``a_vert`` vs ``|a| − g``.

Runs the full pipeline (segmentation → prediction) across every train
and test experiment with each input signal, then computes the
end-to-end Δh errors on the three views from
``pipeline_evaluation_report.py``:

  * GT view       — predictor on every GT interval (oracle).
  * Matched view  — predictor on segmenter-detected intervals that
                    cleanly match a GT (real end-to-end on caught rides).
  * All view      — predictor on every segmenter interval, scored against
                    barometer Δh integrated over the same interval (the
                    deployment-truth view; counts false positives too).

Outputs:
  * stdout       — per-view, per-signal MAE / median / RMSE table.
  * /tmp/pipeline_compare.csv  — raw seg+gt records dump.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import list_experiments  # noqa: E402
from src.segmentation.algorithms.configTypes import (  # noqa: E402
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm,
)
from src.prediction.algorithms.configTypes import (  # noqa: E402
    PREDICT_ALGORITHM_CONFIG, PredictAlgorithm,
)
from src.prediction.algorithms.predictor import Predictor  # noqa: E402

# Reuse the heavy lifting from the existing pipeline report.
from pipeline_evaluation_report import (  # type: ignore  # noqa: E402
    _run_experiment, _build_views,
)


SIGNALS = ["a_vert", "a_mag_minus_g"]


def _say(msg: str) -> None:
    print(msg, flush=True)


def _run_signal(signal: str, train_exps: list[str], test_exps: list[str]):
    seg_cfg = SEGMENT_ALGORITHM_CONFIG(
        algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
        overrides={"input_signal": signal},
    )
    pred_cfg = PREDICT_ALGORITHM_CONFIG(
        algorithm=PredictAlgorithm.TRAPEZOID_ACCEL,
        overrides={"input_signal": signal},
    )
    p = Predictor(pred_cfg)

    gt_rows: list[dict] = []
    seg_rows: list[dict] = []
    for kind, exps in (("train", train_exps), ("test", test_exps)):
        for name in exps:
            t0 = time.time()
            gt_recs, seg_recs = _run_experiment(name, seg_cfg, p)
            for r in gt_recs:
                r["signal"] = signal
            for r in seg_recs:
                r["signal"] = signal
            gt_rows.extend(gt_recs)
            seg_rows.extend(seg_recs)
            _say(f"  [{signal}/{kind}] {name}: "
                 f"{len(gt_recs)} gt, {len(seg_recs)} preds "
                 f"({time.time() - t0:.1f}s)")
    return gt_rows, seg_rows


def main() -> int:
    train_exps = list_experiments(kind="train")
    test_exps  = list_experiments(kind="test")
    _say(f"resolving experiments: train={len(train_exps)} test={len(test_exps)}")

    gt_all: list[dict] = []
    seg_all: list[dict] = []
    for signal in SIGNALS:
        _say(f"\n=== signal: {signal} ===")
        t_signal = time.time()
        g, s = _run_signal(signal, train_exps, test_exps)
        gt_all.extend(g); seg_all.extend(s)
        _say(f"  [{signal}] done in {time.time() - t_signal:.0f}s "
             f"({len(g)} gt rows, {len(s)} pred rows)")

    gt_df = pd.DataFrame(gt_all)
    seg_df = pd.DataFrame(seg_all)
    gt_df.to_csv("/tmp/pipeline_compare_gt.csv", index=False)
    seg_df.to_csv("/tmp/pipeline_compare_seg.csv", index=False)
    _say("\nwrote /tmp/pipeline_compare_{gt,seg}.csv")

    _say("\n" + "=" * 95)
    _say("END-TO-END PIPELINE COMPARISON")
    _say("=" * 95)
    for kind_label, kind in (("ALL (train + test)", None), ("TRAIN", "train"),
                              ("TEST", "test")):
        _say(f"\n--- {kind_label} ---")
        for signal in SIGNALS:
            gt_sub = gt_df[gt_df["signal"] == signal]
            seg_sub = seg_df[seg_df["signal"] == signal]
            if kind is not None:
                gt_sub = gt_sub[gt_sub["kind"] == kind]
                seg_sub = seg_sub[seg_sub["kind"] == kind]
            views = _build_views(gt_sub, seg_sub, accepted_only=False)
            for vname in ("gt", "matched", "all"):
                s = views[vname]["summary"]
                _say(f"  [{signal:14s}][{vname:7s}] "
                     f"n={s['n']:4d}  MAE={s['mae']:6.2f}m  "
                     f"med={s['median']:5.2f}m  RMSE={s['rmse']:6.2f}m  "
                     f"<0.5m={s['p_within_0_5m']:5.1%}  "
                     f"<1.5m={s['p_within_1_5m']:5.1%}")

    # Headline deltas (matched view, test set)
    _say("\nHEADLINE: matched-view TEST set")
    for signal in SIGNALS:
        gt_sub = gt_df[(gt_df["signal"] == signal) & (gt_df["kind"] == "test")]
        seg_sub = seg_df[(seg_df["signal"] == signal) & (seg_df["kind"] == "test")]
        v = _build_views(gt_sub, seg_sub)["matched"]["summary"]
        _say(f"  {signal:14s}: n={v['n']}  MAE={v['mae']:.2f}m  RMSE={v['rmse']:.2f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
