"""Find the optimal segment-padding epsilon.

Each ride prediction algorithm double-integrates the accelerometer over
the segmentation-emitted interval. If the interval is tight on the
lobes' physical extent, the integrator misses part of the acceleration
burst and under-reports Δh. Widening the interval by ε on each side
should recover the missing tails, up to the point where noise
contributes more than signal.

This script sweeps ε over a grid, slicing GT rides ±ε on each side,
running both accelerometer-only predictors (ZUPT and Trapezoid), and
aggregating the mean absolute error against the GT ``height_diff_m``.
The ε that minimises the combined MAE is the optimum.

Usage:
    venv/bin/python -m src.segmentation.algorithms.improvement_iterations._sweep_epsilon
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import getExperimentData, list_experiments
from src.prediction.algorithms.configTypes import (
    PREDICT_ALGORITHM_CONFIG, PredictAlgorithm,
)
from src.prediction.algorithms.predictor import Predictor


ITER_ROOT = Path(__file__).resolve().parent
PRE_MS_FIXED = 3000.0
POST_MS_FIXED = 3000.0
# ε grid (seconds on each side).
EPSILON_GRID_S = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def _slice(acc: pd.DataFrame, lo_ms: float, hi_ms: float) -> pd.DataFrame:
    ts = acc["timestamp_ms"].to_numpy(dtype=float)
    mask = (ts >= lo_ms) & (ts < hi_ms)
    return acc.loc[mask].reset_index(drop=True)


def _phone_model(metadata: dict | None) -> str:
    if not metadata:
        return ""
    for key in ("phone_model", "phone", "model", "device_model"):
        v = metadata.get(key)
        if v:
            return str(v)
    return ""


def _build_predictors() -> dict[str, Predictor]:
    return {
        "zupt": Predictor(PREDICT_ALGORITHM_CONFIG(
            algorithm=PredictAlgorithm.ZUPT_ACCEL,
        )),
        "trapezoid": Predictor(PREDICT_ALGORITHM_CONFIG(
            algorithm=PredictAlgorithm.TRAPEZOID_ACCEL,
        )),
    }


def sweep() -> pd.DataFrame:
    predictors = _build_predictors()
    rows: list[dict] = []
    names = list_experiments(kind="all")
    print(f"Sweeping ε over {len(EPSILON_GRID_S)} values on {len(names)} experiments")
    t0 = time.time()
    for idx, name in enumerate(names):
        try:
            sensors, gt, metadata = getExperimentData(name)
        except Exception as exc:
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
            continue
        acc = sensors.get("ACC")
        if acc is None or acc.empty:
            continue
        phone_model = _phone_model(metadata)
        if gt is None or gt.empty:
            continue
        for gt_idx, row in gt.iterrows():
            if row.get("type") not in ("up", "down"):
                continue
            if row.get("signalClearRecording", True) is False:
                continue
            h_true = row.get("height_diff_m")
            if h_true is None or (isinstance(h_true, float) and np.isnan(h_true)):
                continue
            gt_start_ms = float(row["start_ms"])
            gt_end_ms = float(row["end_ms"])
            for eps_s in EPSILON_GRID_S:
                eps_ms = eps_s * 1000.0
                ride_lo_ms = gt_start_ms - eps_ms
                ride_hi_ms = gt_end_ms + eps_ms
                ride = _slice(acc, ride_lo_ms, ride_hi_ms)
                pre = _slice(acc, ride_lo_ms - PRE_MS_FIXED, ride_lo_ms)
                post = _slice(acc, ride_hi_ms, ride_hi_ms + POST_MS_FIXED)
                if len(ride) < 20 or len(pre) < 20 or len(post) < 20:
                    continue
                for algo_name, pred in predictors.items():
                    try:
                        out = pred.predict(ride, phone_model=phone_model, pre=pre, post=post)
                    except Exception as exc:
                        continue
                    rows.append({
                        "exp":      name,
                        "gt_idx":   int(gt_idx),
                        "type":     row["type"],
                        "algo":     algo_name,
                        "eps_s":    eps_s,
                        "pred_dh":  float(out.height_diff),
                        "true_dh":  float(h_true),
                        "abs_err":  abs(float(out.height_diff) - float(h_true)),
                        "accepted": bool(out.accepted),
                        "reject_reason": out.reject_reason,
                        "n_ride_samples": len(ride),
                    })
        print(f"  [{idx + 1:2d}/{len(names)}] {name[:60]:<60} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(algo, ε) mean / median absolute error + acceptance rate."""
    out = (
        df.groupby(["algo", "eps_s"])
          .agg(
              n=("abs_err", "size"),
              n_accepted=("accepted", "sum"),
              mae=("abs_err", "mean"),
              median_abs=("abs_err", "median"),
              p90_abs=("abs_err", lambda s: float(np.quantile(s, 0.9))),
              accepted_mae=("abs_err", lambda s: float(np.mean(s[df.loc[s.index, "accepted"]])) if s[df.loc[s.index, "accepted"]].size else float("nan")),
          )
          .reset_index()
    )
    out["accept_rate"] = out["n_accepted"] / out["n"]
    return out


def main() -> int:
    out_dir = ITER_ROOT / "epsilon_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = sweep()
    df.to_csv(out_dir / "per_ride.csv", index=False)
    print(f"Wrote {len(df)} rows → {out_dir / 'per_ride.csv'}")

    summary = summarise(df)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print("\n=== MAE by (algo, ε) — ACCEPTED rides only (best to worst per algo) ===")
    for algo in sorted(summary["algo"].unique()):
        sub = summary[summary["algo"] == algo].sort_values("accepted_mae")
        print(f"\n-- {algo} --")
        print(sub[["eps_s", "n", "n_accepted", "accept_rate", "mae", "accepted_mae", "median_abs"]].to_string(index=False))

    # Combined: pick ε minimising mean(accepted_mae_zupt, accepted_mae_trap).
    wide = summary.pivot(index="eps_s", columns="algo", values="accepted_mae")
    wide["combined"] = wide.mean(axis=1)
    wide = wide.sort_values("combined")
    print("\n=== Combined ranking (mean accepted MAE across algos) ===")
    print(wide.to_string())
    best_eps = float(wide.index[0])

    payload = {
        "epsilon_grid_s": EPSILON_GRID_S,
        "best_eps_s": best_eps,
        "best_combined_mae": float(wide.iloc[0]["combined"]),
        "summary": summary.to_dict(orient="records"),
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2))
    print(f"\nBest ε = {best_eps:.2f} s  (combined accepted MAE = {wide.iloc[0]['combined']:.3f} m)")
    print(f"Wrote {out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
