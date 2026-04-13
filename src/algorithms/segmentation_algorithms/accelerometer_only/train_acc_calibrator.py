"""Fit the LR classifier, IVAP calibrator, and edge conformal quantiles.

Ground-truth labels come from the pressure segmenter on the same sessions.
Uses leave-one-session-out (LOSO) when multiple sessions are available;
falls back to a 50/50 temporal split within a single session.

Artifacts are written under ``calibrators/`` next to this file.
Run with:
    python -m src.algorithms.segmentation_algorithms.accelerometer_only.train_acc_calibrator [names...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# allow running as a script
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.loader import load_experimenter, DATA_ROOT
from src.algorithms.segmentation_algorithms.barometer_only import detect_elevator_segments_from_height
from src.algorithms.segmentation_algorithms.accelerometer_only.acc_segmentation import (
    _compute_a_vert, build_windows, FEATURE_NAMES, hysteresis_segments, score_windows,
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


# ---------- per-session feature / label construction ----------

def build_session(name: str, cfg: AccOnlyConfig, pcfg: PressureFilterConfig) -> dict:
    data = load_experimenter(name)
    acc = data["ACC"]
    prs = data.get("PRS")
    if prs is None or "GT_height_m" not in prs.columns:
        raise RuntimeError(f"{name}: no pressure data for GT labels")

    t0_ms = float(acc["timestamp_ms"].iloc[0])
    t_acc = (acc["timestamp_ms"].to_numpy() - t0_ms) / 1000.0
    ax = acc["x"].to_numpy(dtype=float)
    ay = acc["y"].to_numpy(dtype=float)
    az = acc["z"].to_numpy(dtype=float)

    t_prs = (prs["timestamp_ms"].to_numpy() - t0_ms) / 1000.0
    h = prs["GT_height_m"].to_numpy()
    h_smooth = pd.Series(h).rolling(window=51, center=True, min_periods=1).median().to_numpy()
    height_df = pd.DataFrame({"time": t_prs, "height": h_smooth})
    gt = detect_elevator_segments_from_height(height_df, pcfg)

    a_vert = _compute_a_vert(ax, ay, az, cfg.fs_hz)
    centers, feats = build_windows(
        t_acc, a_vert, cfg.fs_hz, cfg.window_sec, cfg.overlap,
        tuple(cfg.band_elev_hz), tuple(cfg.band_walk_hz),
    )

    labels = np.zeros(len(centers), dtype=int)
    for _, row in gt.iterrows():
        gs = ci_center(row["start_ci"])
        ge = ci_center(row["end_ci"])
        labels[(centers >= gs) & (centers <= ge)] = 1

    return {
        "name": name,
        "t_acc": t_acc,
        "centers": centers,
        "features": feats,
        "labels": labels,
        "gt": gt,
    }


# ---------- LR fitting ----------

def fit_lr(X: np.ndarray, y: np.ndarray) -> dict:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    Xs = (X - mean) / std_safe
    if len(np.unique(y)) < 2:
        # degenerate — fall back to zero classifier
        return {
            "feature_names": FEATURE_NAMES,
            "mean": mean.tolist(), "std": std.tolist(),
            "coef": [0.0] * X.shape[1], "intercept": 0.0,
        }
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
    clf.fit(Xs, y)
    return {
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(), "std": std.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
    }


# ---------- segment-level calibration pairs ----------

def segments_from_scores(centers: np.ndarray, scores: np.ndarray, cfg: AccOnlyConfig,
                         t_min: float, t_max: float) -> list[tuple[float, float, float]]:
    return hysteresis_segments(
        centers, scores,
        enter=cfg.enter_threshold, exit_=cfg.exit_threshold,
        min_duration_sec=cfg.min_duration_sec,
        merge_gap_sec=cfg.merge_gap_sec, pad_sec=cfg.pad_sec,
        t_min=t_min, t_max=t_max,
    )


def _preds_to_df(preds: list[tuple[float, float, float]]) -> pd.DataFrame:
    """Wrap raw (start, end, score) triples as a CI-schema DataFrame with
    degenerate CIs, so SegmentationMetrics can consume it directly."""
    if not preds:
        return pd.DataFrame(columns=["start_ci", "end_ci", "score"])
    rows = [{"start_ci": (s, s), "end_ci": (e, e), "score": sc} for s, e, sc in preds]
    return pd.DataFrame(rows)


def evaluate_fold(preds: list[tuple[float, float, float]], gt: pd.DataFrame,
                  iou_threshold: float = 0.5, match_threshold: float = 0.3,
                  ) -> tuple[list[int], list[float], list[float], list[float]]:
    """Return (labels, scores, start_residuals, end_residuals) for a fold.

    ``match_threshold`` is the loose IoU threshold used to collect edge
    residuals (we want residuals even when IoU is below the strict F1
    threshold but the segments clearly correspond).
    """
    pred_df = _preds_to_df(preds)
    det = SegmentationMetrics.match_segments(pred_df, gt, iou_threshold=iou_threshold)
    labels = [0] * len(preds)
    for pi in det.matched_pred_idx:
        labels[pi] = 1
    scores = [p[2] for p in preds]

    # edge residuals: looser IoU match so we can pool residuals for conformal fit
    loose = SegmentationMetrics.match_segments(pred_df, gt, iou_threshold=match_threshold)
    start_res, end_res = [], []
    for pi, gi in zip(loose.matched_pred_idx, loose.matched_gt_idx):
        start_res.append(abs(preds[pi][0] - ci_center(gt.iloc[gi]["start_ci"])))
        end_res.append(abs(preds[pi][1] - ci_center(gt.iloc[gi]["end_ci"])))
    return labels, scores, start_res, end_res


# ---------- orchestration ----------

def _split_by_time(session: dict, split_at: float) -> tuple[dict, dict]:
    """Temporally split a single session into 'before' and 'after' halves."""
    c = session["centers"]
    feats = session["features"]
    labels = session["labels"]
    mask_a = c < split_at
    mask_b = ~mask_a
    gt = session["gt"]
    gt_starts = np.array([ci_center(x) for x in gt["start_ci"]]) if len(gt) else np.array([])
    gt_ends = np.array([ci_center(x) for x in gt["end_ci"]]) if len(gt) else np.array([])
    gt_a = gt.iloc[np.where(gt_ends <= split_at)[0]].reset_index(drop=True)
    gt_b = gt.iloc[np.where(gt_starts >= split_at)[0]].reset_index(drop=True)
    return (
        {**session, "centers": c[mask_a], "features": feats[mask_a], "labels": labels[mask_a], "gt": gt_a},
        {**session, "centers": c[mask_b], "features": feats[mask_b], "labels": labels[mask_b], "gt": gt_b},
    )


def main(names: list[str]) -> None:
    cfg = AccOnlyConfig()
    pcfg = PressureFilterConfig()

    print(f"Loading {len(names)} session(s): {names}")
    sessions = [build_session(n, cfg, pcfg) for n in names]
    for s in sessions:
        pos = int(s["labels"].sum())
        print(f"  {s['name']}: windows={len(s['centers'])} positives={pos} GT_segments={len(s['gt'])}")

    # ---- build (session_id, feature, label) list and LOSO folds ----
    if len(sessions) >= 2:
        folds = [(i, list(range(len(sessions))) ) for i in range(len(sessions))]
        folds = [(test, [j for j in train if j != test]) for test, train in folds]
        fold_sessions = sessions
    else:
        # single session -> temporal split at median window center
        c = sessions[0]["centers"]
        split_at = float(np.median(c))
        a, b = _split_by_time(sessions[0], split_at)
        fold_sessions = [a, b]
        folds = [(0, [1]), (1, [0])]
        print(f"  single-session fallback: temporal split at t={split_at:.1f}s")

    # ---- outer LOSO: evaluate ----
    all_seg_scores: list[float] = []
    all_seg_labels: list[int] = []
    all_start_res: list[float] = []
    all_end_res: list[float] = []

    for test_idx, train_idx in folds:
        Xtr = np.vstack([fold_sessions[i]["features"] for i in train_idx])
        ytr = np.concatenate([fold_sessions[i]["labels"] for i in train_idx])
        lr_fold = fit_lr(Xtr, ytr)

        te = fold_sessions[test_idx]
        if len(te["features"]) == 0:
            continue
        w_scores = score_windows(te["features"], lr_fold)
        preds = segments_from_scores(
            te["centers"], w_scores, cfg,
            t_min=float(te["t_acc"][0]), t_max=float(te["t_acc"][-1]),
        )
        labels, scores, sr, er = evaluate_fold(preds, te["gt"])
        all_seg_scores.extend(scores)
        all_seg_labels.extend(labels)
        all_start_res.extend(sr)
        all_end_res.extend(er)
        print(f"  fold test={fold_sessions[test_idx]['name']}: preds={len(preds)} TP={sum(labels)}")

    seg_scores = np.asarray(all_seg_scores)
    seg_labels = np.asarray(all_seg_labels)
    print(f"Calibration pool: {len(seg_scores)} segments, {int(seg_labels.sum())} positives")

    # ---- fit final artifacts on all data (LR) + LOSO-held-out scores (IVAP + edge) ----
    Xall = np.vstack([s["features"] for s in sessions])
    yall = np.concatenate([s["labels"] for s in sessions])
    lr_final = fit_lr(Xall, yall)

    CALIBRATORS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATORS_DIR / "lr_weights.json", "w") as f:
        json.dump(lr_final, f, indent=2)

    if len(seg_scores) > 0:
        ivap = IVAP().fit(seg_scores, seg_labels.astype(float))
        ivap.save(CALIBRATORS_DIR / "ivap.json")
        p, p_lo, p_hi = ivap.predict(seg_scores)
        calibrated_p = np.asarray(p)
        bin_ece = SegmentationMetrics.ece(calibrated_p, seg_labels)
        bin_br = SegmentationMetrics.brier(calibrated_p, seg_labels)
        covered = SegmentationMetrics.prob_ci_coverage(
            seg_labels, np.asarray(p_lo), np.asarray(p_hi)
        )
        print(f"IVAP on held-out:  ECE={bin_ece:.3f}  Brier={bin_br:.3f}  CI_coverage={covered:.2f}")
    else:
        IVAP(np.array([0.0, 1.0]), np.array([0.0, 1.0])).save(CALIBRATORS_DIR / "ivap.json")
        print("IVAP: no calibration data, saved degenerate fallback")

    start_c = EdgeTimeConformal(alpha=cfg.alpha).fit(np.asarray(all_start_res) if all_start_res else np.array([0.0]))
    end_c = EdgeTimeConformal(alpha=cfg.alpha).fit(np.asarray(all_end_res) if all_end_res else np.array([0.0]))
    save_edge_conformal(start_c, end_c, CALIBRATORS_DIR / "edge_conformal.json")
    print(f"Edge conformal: start_q={start_c.quantile():.2f}s  end_q={end_c.quantile():.2f}s  alpha={cfg.alpha}")
    print(f"Artifacts written to {CALIBRATORS_DIR}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        data_root = Path(DATA_ROOT)
        args = sorted([p.name for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    main(args)
