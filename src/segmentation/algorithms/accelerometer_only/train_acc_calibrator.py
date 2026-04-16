"""Fit IVAP + edge conformal quantiles for the drift-residual detector.

The detector itself is training-free (scale-free per session). This script
only fits the calibrators: it runs the detector on every session, collects
(segment_score, is_true_positive) pairs against pressure GT, and fits:

    - IVAP: segment score -> calibrated probability with CI
    - EdgeTimeConformal: residual |t_pred - t_gt| -> 1-alpha quantile

Run:
    python -m src.algorithms.segmentation_algorithms.accelerometer_only.train_acc_calibrator [names...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.loader import load_experimenter, DATA_ROOT
from src.algorithms.segmentation_algorithms.barometer_only import detect_elevator_segments_from_height
from src.algorithms.segmentation_algorithms.accelerometer_only.acc_segmentation import (
    _compute_a_vert, drift_residual_score, hysteresis_segments,
)
from src.algorithms.segmentation_algorithms._calibration import (
    IVAP, EdgeTimeConformal, save_edge_conformal,
)
from src.algorithms.segmentation_algorithms.metrics import SegmentationMetrics, ci_center

import importlib
_config_mod = importlib.import_module("src.algorithms.segmentation_algorithms.class")
PressureFilterConfig = _config_mod.PressureFilterConfig
AccOnlyConfig = _config_mod.AccOnlyConfig
CALIBRATORS_DIR = _config_mod.CALIBRATORS_DIR


def run_detector(name: str, cfg: AccOnlyConfig, pcfg: PressureFilterConfig) -> dict:
    data = load_experimenter(name)
    acc = data["ACC"]
    prs = data.get("PRS")
    if prs is None or "GT_height_m" not in prs.columns:
        raise RuntimeError(f"{name}: no pressure data for GT labels")

    t0_ms = float(acc["timestamp_ms"].iloc[0])
    t = (acc["timestamp_ms"].to_numpy() - t0_ms) / 1000.0
    ax = acc["x"].to_numpy(dtype=float)
    ay = acc["y"].to_numpy(dtype=float)
    az = acc["z"].to_numpy(dtype=float)

    t_prs = (prs["timestamp_ms"].to_numpy() - t0_ms) / 1000.0
    height_df = pd.DataFrame({"time": t_prs, "height": prs["GT_height_m"].to_numpy()})
    gt = detect_elevator_segments_from_height(height_df, pcfg)

    a_vert = _compute_a_vert(ax, ay, az, cfg.fs_hz)
    scores = drift_residual_score(a_vert, cfg.fs_hz,
                                  detrend_sec=cfg.detrend_sec,
                                  local_var_sec=cfg.local_var_sec)
    preds = hysteresis_segments(
        t, scores,
        enter=cfg.enter_threshold, exit_=cfg.exit_threshold,
        min_duration_sec=cfg.min_duration_sec,
        merge_gap_sec=cfg.merge_gap_sec, pad_sec=cfg.pad_sec,
        t_min=float(t[0]), t_max=float(t[-1]),
    )
    return {"name": name, "preds": preds, "gt": gt}


def _preds_to_df(preds):
    if not preds:
        return pd.DataFrame(columns=["start_ci", "end_ci", "score"])
    return pd.DataFrame([
        {"start_ci": (s, s), "end_ci": (e, e), "score": sc} for s, e, sc in preds
    ])


def main(names: list[str]) -> None:
    cfg = AccOnlyConfig()
    pcfg = PressureFilterConfig()
    print(f"Running drift-residual detector on {len(names)} session(s): {names}")

    all_scores, all_labels = [], []
    all_start_res, all_end_res = [], []
    for name in names:
        r = run_detector(name, cfg, pcfg)
        preds, gt = r["preds"], r["gt"]
        pred_df = _preds_to_df(preds)
        det = SegmentationMetrics.match_segments(pred_df, gt, iou_threshold=0.5)
        labels = [0] * len(preds)
        for pi in det.matched_pred_idx:
            labels[pi] = 1
        scores = [p[2] for p in preds]
        loose = SegmentationMetrics.match_segments(pred_df, gt, iou_threshold=0.3)
        sr = [abs(preds[pi][0] - ci_center(gt.iloc[gi]["start_ci"]))
              for pi, gi in zip(loose.matched_pred_idx, loose.matched_gt_idx)]
        er = [abs(preds[pi][1] - ci_center(gt.iloc[gi]["end_ci"]))
              for pi, gi in zip(loose.matched_pred_idx, loose.matched_gt_idx)]
        all_scores += scores
        all_labels += labels
        all_start_res += sr
        all_end_res += er
        print(f"  {name}: preds={len(preds)}  TP@IoU0.5={sum(labels)}  GT={len(gt)}")

    seg_scores = np.asarray(all_scores)
    seg_labels = np.asarray(all_labels)
    print(f"Calibration pool: {len(seg_scores)} segments, {int(seg_labels.sum())} positives")

    CALIBRATORS_DIR.mkdir(parents=True, exist_ok=True)
    if len(seg_scores) > 0 and len(np.unique(seg_labels)) > 1:
        ivap = IVAP().fit(seg_scores, seg_labels.astype(float))
        ivap.save(CALIBRATORS_DIR / "ivap.json")
        p, p_lo, p_hi = ivap.predict(seg_scores)
        ece = SegmentationMetrics.ece(np.asarray(p), seg_labels)
        br = SegmentationMetrics.brier(np.asarray(p), seg_labels)
        cov = SegmentationMetrics.prob_ci_coverage(seg_labels, np.asarray(p_lo), np.asarray(p_hi))
        print(f"IVAP:  ECE={ece:.3f}  Brier={br:.3f}  CI_coverage={cov:.2f}")
    else:
        IVAP(np.array([0.0, 1.0]), np.array([0.0, 1.0])).save(CALIBRATORS_DIR / "ivap.json")
        print("IVAP: no calibration data, saved degenerate fallback")

    _edge_floor = float(cfg.pad_sec + cfg.local_var_sec * 0.5)
    start_c = EdgeTimeConformal(alpha=cfg.alpha).fit(
        np.asarray(all_start_res) if len(all_start_res) >= 5 else np.array([_edge_floor]))
    end_c = EdgeTimeConformal(alpha=cfg.alpha).fit(
        np.asarray(all_end_res) if len(all_end_res) >= 5 else np.array([_edge_floor]))
    save_edge_conformal(start_c, end_c, CALIBRATORS_DIR / "edge_conformal.json")
    print(f"Edge conformal: start_q={start_c.quantile():.2f}s  end_q={end_c.quantile():.2f}s  alpha={cfg.alpha}")
    print(f"Artifacts written to {CALIBRATORS_DIR}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        data_root = Path(DATA_ROOT)
        args = sorted([p.name for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    main(args)
