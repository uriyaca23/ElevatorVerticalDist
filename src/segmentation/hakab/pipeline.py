"""Per-experiment template-match detection check.

Iterates every experiment discovered by the data-loader pipeline, fits
pulse templates from that experiment's own GT, runs the template-match
segmenter, and reports detection metrics (precision/recall/F1 at IoU 0.3)
against GT.

Note: templates are fitted per-experiment from the same GT we evaluate
against — this is in-sample and therefore optimistic. Useful as a sanity
check that the detector *can* match its training distribution. For an
honest evaluation, fit templates on held-out experiments.

Run:
    venv/bin/python -m src.segmentation.hakab.pipeline
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import getExperimentData, list_experiments
from src.segmentation.algorithms import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
    SegmentationMetrics,
    Segmenter,
    TemplateMatchConfig,
    fit_templates,
    save_templates,
)

GT_SCHEMA_COLUMNS = ["start_ci", "end_ci", "duration", "type", "probability_ci"]
IOU_THRESHOLD = 0.3


def build_acc_frame(acc_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Loader ACC → segmenter-ready (time, x, y, z). Returns frame and t0_ms."""
    t0_ms = float(acc_df["timestamp_ms"].iloc[0])
    t_sec = (acc_df["timestamp_ms"].to_numpy(dtype=float) - t0_ms) / 1000.0
    return pd.DataFrame({
        "time": t_sec,
        "x": acc_df["x"].to_numpy(dtype=float),
        "y": acc_df["y"].to_numpy(dtype=float),
        "z": acc_df["z"].to_numpy(dtype=float),
    }), t0_ms


def build_gt_frame(gt_df: pd.DataFrame, t0_ms: float) -> pd.DataFrame:
    """Loader GT → segmenter CI schema in seconds; drops 'outside' rows."""
    rides = gt_df[gt_df["type"].isin(("up", "down"))]
    if rides.empty:
        return pd.DataFrame(columns=GT_SCHEMA_COLUMNS)
    starts = (rides["start_ms"].to_numpy(dtype=float) - t0_ms) / 1000.0
    ends = (rides["end_ms"].to_numpy(dtype=float) - t0_ms) / 1000.0
    rows = [
        {
            "start_ci": (float(s), float(s)),
            "end_ci": (float(e), float(e)),
            "duration": float(e - s),
            "type": typ,
            "probability_ci": (1.0, 1.0),
        }
        for s, e, typ in zip(starts, ends, rides["type"].tolist())
    ]
    return pd.DataFrame(rows, columns=GT_SCHEMA_COLUMNS)


def estimate_fs(acc_frame: pd.DataFrame, default: float = 100.0) -> float:
    t = acc_frame["time"].to_numpy()
    if len(t) < 2:
        return default
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt if dt > 0 else default


def evaluate_experiment(name: str) -> dict | None:
    """Detect rides for one experiment and score against GT.

    Returns None when the experiment has no ACC data or no ride intervals.
    """
    sensors, gt, _meta = getExperimentData(name)
    if "ACC" not in sensors or sensors["ACC"].empty:
        return None

    acc_frame, t0_ms = build_acc_frame(sensors["ACC"])
    gt_frame = build_gt_frame(gt, t0_ms)
    if gt_frame.empty:
        return None

    fs = estimate_fs(acc_frame)
    tm_cfg = TemplateMatchConfig(fs_hz=fs)

    templates = fit_templates(acc_frame, gt_frame, tm_cfg, name=name)
    if templates.meta.get("n_rides", 0) == 0:
        return None

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tpl_path = Path(f.name)
    save_templates(templates, tpl_path)
    try:
        seg_cfg = SEGMENT_ALGORITHM_CONFIG(
            algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
            overrides={"fs_hz": fs, "templates_path": str(tpl_path)},
        )
        pred = Segmenter(seg_cfg).detect(acc_frame)
    finally:
        tpl_path.unlink(missing_ok=True)

    det = SegmentationMetrics.iou_match_segments(pred, gt_frame, iou_threshold=IOU_THRESHOLD)
    return {
        "name": name,
        "n_gt": len(gt_frame),
        "n_pred": len(pred),
        "tp": det.tp,
        "fp": det.fp,
        "fn": det.fn,
        "precision": det.precision,
        "recall": det.recall,
        "f1": det.f1,
        "mean_iou": float(np.mean(det.ious)) if det.ious else 0.0,
    }


def evaluate_all() -> list[dict]:
    rows: list[dict] = []
    for name in list_experiments('train'):
        try:
            row = evaluate_experiment(name)
        except Exception as exc:
            print(f"[error] {name}: {type(exc).__name__}: {exc}")
            continue
        if row is None:
            print(f"[skip]  {name}: no ACC or no rides in GT")
            continue
        rows.append(row)
        print(
            f"[ok]    {name}: "
            f"gt={row['n_gt']} pred={row['n_pred']} "
            f"tp={row['tp']} fp={row['fp']} fn={row['fn']} "
            f"P={row['precision']:.2f} R={row['recall']:.2f} F1={row['f1']:.2f} "
            f"mIoU={row['mean_iou']:.2f}"
        )
    return rows


def main() -> int:
    rows = evaluate_all()
    if not rows:
        print("no experiments evaluated")
        return 1

    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print()
    print(
        f"aggregate over {len(rows)} experiments: "
        f"tp={tp} fp={fp} fn={fn} P={p:.3f} R={r:.3f} F1={f1:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
