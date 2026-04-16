"""Pass bar: per-experimenter recall >= 0.9 AND precision >= 0.7 on best-match IOU
(threshold 0.3), accelerometer-only predictor vs pressure-based GT."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter, SegmentationMetrics,
    ci_center,
)
from src.tests.segmentations.main_acc import build_acc_frame, build_height_frame

IOU_THRESHOLD = 0.3
MIN_RECALL = 0.9
MIN_PRECISION = 0.7

DEBUG_DIR = REPO_ROOT / "run_results" / "acc_segmentation_iterations"


def _segments_to_list(df: pd.DataFrame) -> list[tuple[float, float]]:
    return [
        (ci_center(s), ci_center(e))
        for s, e in zip(df["start_ci"].to_list(), df["end_ci"].to_list())
    ] if len(df) else []


def _dump_debug(name: str, pred: pd.DataFrame, gt: pd.DataFrame, res) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experimenter": name,
        "tp": res.tp, "fp": res.fp, "fn": res.fn,
        "recall": res.recall, "precision": res.precision, "f1": res.f1,
        "iou_threshold": IOU_THRESHOLD,
        "gt_segments": _segments_to_list(gt),
        "pred_segments": _segments_to_list(pred),
        "matched_pairs": [
            {"pred_idx": pi, "gt_idx": gi, "iou": float(io)}
            for pi, gi, io in zip(res.matched_pred_idx, res.matched_gt_idx, res.ious)
        ],
        "missed_gt_idx": [
            j for j in range(len(gt)) if j not in set(res.matched_gt_idx)
        ],
        "fp_pred_idx": [
            i for i in range(len(pred)) if i not in set(res.matched_pred_idx)
        ],
    }
    (DEBUG_DIR / f"fail_{name}.json").write_text(json.dumps(payload, indent=2, default=float))


@pytest.mark.parametrize("experimenter", ["oria", "roy_turgman"])
def test_acc_recall_precision(experimenter: str) -> None:
    data = load_experimenter(experimenter)
    acc_raw = data["ACC"]
    prs = data["PRS"]
    t0_ms = float(acc_raw["timestamp_ms"].iloc[0])
    acc_frame = build_acc_frame(acc_raw, t0_ms)
    height_frame = build_height_frame(prs, t0_ms)

    gt = Segmenter(SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)).detect(height_frame)
    pred = Segmenter(SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.ACC_ONLY)).detect(
        acc_frame[["time", "x", "y", "z"]]
    )

    res = SegmentationMetrics.iou_match_segments(pred, gt, iou_threshold=IOU_THRESHOLD)
    if res.recall < MIN_RECALL or res.precision < MIN_PRECISION:
        _dump_debug(experimenter, pred, gt, res)

    assert res.recall >= MIN_RECALL, (
        f"{experimenter}: recall={res.recall:.3f} tp={res.tp} fn={res.fn} (want >={MIN_RECALL})"
    )
    assert res.precision >= MIN_PRECISION, (
        f"{experimenter}: precision={res.precision:.3f} tp={res.tp} fp={res.fp} (want >={MIN_PRECISION})"
    )
