"""Reproduce the boutique pipeline's prediction-step call signature on
additional experiments to confirm the universal "no_gravity_calibration"
rejection mode.

Mirrors src/pipelines/streamlit/step4_prediction.py::_run_predictions:
    out = predictor.predict(seg, phone_model="")
i.e. *no* pre/post stationary windows are passed.

This script also runs a control where pre/post windows are extracted
from the surrounding `outside` rows of the same gt.csv, to show that
the same code path accepts segments when calibration data is supplied.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.prediction.algorithms.configTypes import (  # noqa: E402
    PREDICT_ALGORITHM_CONFIG, PredictAlgorithm,
)
from src.prediction.algorithms.predictor import Predictor  # noqa: E402

DATA_ROOT = REPO / "src" / "data" / "structuredData" / "data"


def _slice_acc(acc: pd.DataFrame, t_lo_ms: float, t_hi_ms: float) -> pd.DataFrame:
    m = (acc["timestamp_ms"] >= t_lo_ms) & (acc["timestamp_ms"] <= t_hi_ms)
    return acc.loc[m].reset_index(drop=True)


def run_one_experiment(exp_dir: Path, with_pre_post: bool) -> dict:
    acc = pd.read_csv(exp_dir / "ACC.csv")
    gt = pd.read_csv(exp_dir / "gt.csv")
    rides = gt[gt["type"].isin(["up", "down"])].reset_index(drop=True)

    cfg = PREDICT_ALGORITHM_CONFIG(algorithm=PredictAlgorithm.TRAPEZOID_ACCEL)
    pred = Predictor(cfg)

    counts: dict[str, int] = {}
    accepted = 0
    n = len(rides)
    pre_dur_ms = 5_000
    post_dur_ms = 5_000

    for _, row in rides.iterrows():
        t0 = float(row["start_ms"]); t1 = float(row["end_ms"])
        seg = _slice_acc(acc, t0, t1)
        if seg.empty:
            counts["empty_slice"] = counts.get("empty_slice", 0) + 1
            continue
        if with_pre_post:
            pre = _slice_acc(acc, t0 - pre_dur_ms, t0)
            post = _slice_acc(acc, t1, t1 + post_dur_ms)
        else:
            pre = None; post = None
        try:
            out = pred.predict(seg, phone_model="", pre=pre, post=post)
            if out.accepted:
                accepted += 1
                key = "ACCEPTED"
            else:
                key = out.reject_reason or "rejected_no_reason"
            counts[key] = counts.get(key, 0) + 1
        except Exception as e:
            key = f"EXC:{type(e).__name__}: {e}"
            counts[key] = counts.get(key, 0) + 1

    return {
        "exp": exp_dir.name,
        "with_pre_post": with_pre_post,
        "n_rides": n,
        "accepted": accepted,
        "counts": counts,
    }


def main() -> None:
    targets = [
        # The original mistake-folder experiment
        "eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2",
        # A different building / different phone
        "UriyaCohenEliya_acroBuilding_GooglePixel10_15-04-2026_exp4",
        # A third one
        "eyalyakir_beitMansour1_SamsungSM-S911B_15-04-2026_exp5",
    ]
    print(f"Repo: {REPO}\nData root: {DATA_ROOT}\n")
    for name in targets:
        d = DATA_ROOT / name
        if not d.exists():
            print(f"  [skip] {name} not found"); continue
        for with_pp in (False, True):
            r = run_one_experiment(d, with_pre_post=with_pp)
            tag = "WITHOUT pre/post (mimic boutique pipeline)" if not with_pp else "WITH 5s pre/post (control)"
            print(f"\n=== {r['exp']} -- {tag} ===")
            print(f"  rides={r['n_rides']}, accepted={r['accepted']}/{r['n_rides']}")
            for k, v in sorted(r["counts"].items(), key=lambda kv: -kv[1]):
                print(f"    {v:3d}  {k}")


if __name__ == "__main__":
    main()
