"""End-to-end repro of the boutique pipeline path: segmenter → prediction
with pre/post slicing exactly the way the fixed step4_prediction.py does.
Targets several experiments and runs both algorithms (trapezoid and ZUPT)
so we can verify the fixes are consistent across:

* Different buildings / phones (3+ experiments).
* Both accelerometer-only algorithms (trapezoid + ZUPT).
* Tight matched-filter segments (the boutique pipeline's actual segmenter
  output) and not just GT intervals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.pipelines.streamlit.step4_prediction import _slice_acc, _slice_pre_post  # noqa: E402
from src.prediction.algorithms.configTypes import (  # noqa: E402
    PREDICT_ALGORITHM_CONFIG, PredictAlgorithm,
)
from src.prediction.algorithms.predictor import Predictor  # noqa: E402
from src.segmentation.algorithms.configTypes import (  # noqa: E402
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm,
)
from src.segmentation.algorithms.segmenter import Segmenter  # noqa: E402

DATA_ROOT = REPO / "src" / "data" / "structuredData" / "data"
TARGETS = [
    "eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2",
    "UriyaCohenEliya_acroBuilding_GooglePixel10_15-04-2026_exp4",
    "eyalyakir_beitMansour1_SamsungSM-S911B_15-04-2026_exp5",
    "UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1",
]


def _flatten(seg_row: pd.Series) -> tuple[float, float, str]:
    s = seg_row["start_ci"]; e = seg_row["end_ci"]
    s = float(s[0]) if hasattr(s, "__len__") else float(s)
    e = float(e[0]) if hasattr(e, "__len__") else float(e)
    return s, e, str(seg_row["type"]).lower()


def run_one(target: str) -> dict:
    exp_dir = DATA_ROOT / target
    if not exp_dir.exists():
        return {"target": target, "skipped": True}
    acc = pd.read_csv(exp_dir / "ACC.csv")
    t0_ms = float(acc["timestamp_ms"].iloc[0])

    seg = Segmenter(SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH))
    segs = seg.detect(acc)
    n_segs = len(segs)
    starts = [_flatten(r)[0] for _, r in segs.iterrows()]
    ends = [_flatten(r)[1] for _, r in segs.iterrows()]

    results: dict[str, dict] = {}
    for algo in (PredictAlgorithm.TRAPEZOID_ACCEL, PredictAlgorithm.ZUPT_ACCEL):
        pred = Predictor(PREDICT_ALGORITHM_CONFIG(algorithm=algo))
        accepted = 0
        reasons: dict[str, int] = {}
        methods: dict[str, int] = {}
        excs = 0
        for i, (_, row) in enumerate(segs.iterrows()):
            t_lo, t_hi, _rt = _flatten(row)
            s = _slice_acc(acc, t0_ms, t_lo, t_hi)
            prev_hi = ends[i - 1] if i > 0 else None
            next_lo = starts[i + 1] if i + 1 < len(starts) else None
            pre, post = _slice_pre_post(acc, t0_ms, t_lo, t_hi, prev_hi, next_lo)
            try:
                out = pred.predict(s, phone_model="", pre=pre, post=post)
                m = (out.meta or {}).get("vert_method") or (out.meta or {}).get("method") or "n/a"
                methods[m] = methods.get(m, 0) + 1
                if out.accepted:
                    accepted += 1
                else:
                    key = out.reject_reason or "rejected_no_reason"
                    reasons[key] = reasons.get(key, 0) + 1
            except Exception as e:
                excs += 1
                key = f"EXC:{type(e).__name__}: {e}"
                reasons[key] = reasons.get(key, 0) + 1
        results[algo.value] = {
            "accepted": accepted, "n": n_segs,
            "reasons": reasons, "methods": methods, "exceptions": excs,
        }
    return {"target": target, "n_segs": n_segs, "by_algo": results}


def main() -> None:
    for t in TARGETS:
        r = run_one(t)
        if r.get("skipped"):
            print(f"[skip] {t} not found"); continue
        print(f"\n=== {r['target']}  ({r['n_segs']} detected segments) ===")
        for algo, data in r["by_algo"].items():
            print(f"  {algo}: {data['accepted']}/{data['n']} accepted, "
                  f"exceptions={data['exceptions']}")
            print(f"    methods: {data['methods']}")
            if data["reasons"]:
                items = sorted(data["reasons"].items(), key=lambda kv: -kv[1])[:6]
                print(f"    top rejection reasons: {items}")


if __name__ == "__main__":
    main()
