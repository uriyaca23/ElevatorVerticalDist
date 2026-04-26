"""Inference-time benchmark for segmentation + prediction.

Measures per-segment runtime on a fixed pool of experiments. Captures:
  * Segmentation (Segmenter.detect on the full ACC stream).
  * Prediction (Predictor.predict per detected segment, both ZUPT and trapezoid).

The benchmark is "per ACC segment" — the unit the user wants to minimise.
For segmentation the per-segment cost is amortised across detected rides.

Outputs JSON for diff-able runs (baseline vs optimised) at
benchmarks/results/<tag>.json.

Usage:
    python -m benchmarks.bench_inference --tag baseline
    python -m benchmarks.bench_inference --tag opt1 --compare baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
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
RESULTS_DIR = REPO / "benchmarks" / "results"

TARGETS = [
    "eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2",
    "UriyaCohenEliya_acroBuilding_GooglePixel10_15-04-2026_exp4",
    "eyalyakir_beitMansour1_SamsungSM-S911B_15-04-2026_exp5",
    "UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1",
    "UriyaCohenEliya_beitYitzchakiRaanana_GooglePixel10_15-04-2026_exp6",
    "eyalyakir_acroBuilding_SamsungSM-S911B_15-04-2026_exp4",
    "UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2",
    "UriyaCohenEliya_beitMansour1_GooglePixel10_15-04-2026_exp5",
]


def _flatten(seg_row: pd.Series) -> tuple[float, float, str]:
    s = seg_row["start_ci"]; e = seg_row["end_ci"]
    s = float(s[0]) if hasattr(s, "__len__") else float(s)
    e = float(e[0]) if hasattr(e, "__len__") else float(e)
    return s, e, str(seg_row["type"]).lower()


def bench_one(target: str, n_warmup: int = 1, n_repeat: int = 3) -> dict:
    exp_dir = DATA_ROOT / target
    if not exp_dir.exists():
        return {"target": target, "skipped": True}
    acc = pd.read_csv(exp_dir / "ACC.csv")
    t0_ms = float(acc["timestamp_ms"].iloc[0])
    n_acc = int(len(acc))
    duration_s = float((acc["timestamp_ms"].iloc[-1] - acc["timestamp_ms"].iloc[0]) / 1000.0)

    seg_cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH)
    seg = Segmenter(seg_cfg)

    # Warm-up segmentation (loads config files etc.)
    for _ in range(n_warmup):
        segs = seg.detect(acc)
    # Time segmentation
    seg_times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        segs = seg.detect(acc)
        seg_times.append(time.perf_counter() - t0)
    seg_time = float(min(seg_times))
    n_segs = int(len(segs))
    starts = [_flatten(r)[0] for _, r in segs.iterrows()]
    ends = [_flatten(r)[1] for _, r in segs.iterrows()]

    pred_results: dict[str, dict] = {}
    for algo in (PredictAlgorithm.TRAPEZOID_ACCEL, PredictAlgorithm.ZUPT_ACCEL):
        pred = Predictor(PREDICT_ALGORITHM_CONFIG(algorithm=algo))

        # Build all segment slices once (shared across warmup/repeat).
        seg_inputs = []
        for i, (_, row) in enumerate(segs.iterrows()):
            t_lo, t_hi, _rt = _flatten(row)
            s = _slice_acc(acc, t0_ms, t_lo, t_hi)
            prev_hi = ends[i - 1] if i > 0 else None
            next_lo = starts[i + 1] if i + 1 < len(starts) else None
            pre, post = _slice_pre_post(acc, t0_ms, t_lo, t_hi, prev_hi, next_lo)
            seg_inputs.append((s, pre, post))

        # Warmup
        for _ in range(n_warmup):
            for s, pre, post in seg_inputs:
                pred.predict(s, phone_model="", pre=pre, post=post)

        # Capture height_diff + accepted/rejected for accuracy comparison
        ref_run = []
        for s, pre, post in seg_inputs:
            out = pred.predict(s, phone_model="", pre=pre, post=post)
            ref_run.append({
                "height_diff": float(out.height_diff),
                "ci_half_width": float(out.ci_half_width),
                "theoretical_sigma": float(out.theoretical_sigma),
                "accepted": bool(out.accepted),
                "reject_reason": out.reject_reason or "",
                "quality_score": float(out.quality_score),
            })

        # Time per-segment prediction (best of n_repeat over the full sweep)
        t_per_seg = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            for s, pre, post in seg_inputs:
                pred.predict(s, phone_model="", pre=pre, post=post)
            t_per_seg.append(time.perf_counter() - t0)
        total_pred_time = float(min(t_per_seg))
        per_seg = total_pred_time / max(n_segs, 1)

        pred_results[algo.value] = {
            "total_time_s": total_pred_time,
            "per_segment_s": per_seg,
            "n_segments": n_segs,
            "outputs": ref_run,
        }

    return {
        "target": target,
        "n_acc_samples": n_acc,
        "duration_s": duration_s,
        "n_segments": n_segs,
        "segmentation": {
            "total_time_s": seg_time,
            "per_segment_s": seg_time / max(n_segs, 1),
            "per_acc_sample_us": seg_time / n_acc * 1e6,
        },
        "prediction": pred_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Result tag (filename)")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--targets", nargs="*", default=None,
                        help="Override target list (whitespace-separated)")
    parser.add_argument("--compare", default=None,
                        help="Compare against another tag's results")
    args = parser.parse_args()

    targets = args.targets or TARGETS
    print(f"Benchmarking {len(targets)} experiments, repeat={args.repeat}")
    results = []
    for t in targets:
        print(f"  [{t}] ...", flush=True)
        try:
            r = bench_one(t, n_repeat=args.repeat)
        except Exception as e:
            print(f"    SKIP: {type(e).__name__}: {e}")
            continue
        if r.get("skipped"):
            print("    SKIP: not found")
            continue
        results.append(r)
        print(f"    n_segs={r['n_segments']}  seg={r['segmentation']['total_time_s']*1000:.1f}ms  "
              f"trap/seg={r['prediction']['trapezoid_accel']['per_segment_s']*1000:.2f}ms  "
              f"zupt/seg={r['prediction']['zupt_accel']['per_segment_s']*1000:.2f}ms")

    # Aggregate
    n_total_segs = sum(r["n_segments"] for r in results)
    seg_total_time = sum(r["segmentation"]["total_time_s"] for r in results)
    trap_total = sum(r["prediction"]["trapezoid_accel"]["total_time_s"] for r in results)
    zupt_total = sum(r["prediction"]["zupt_accel"]["total_time_s"] for r in results)
    summary = {
        "tag": args.tag,
        "n_experiments": len(results),
        "n_segments_total": int(n_total_segs),
        "segmentation_total_s": float(seg_total_time),
        "segmentation_per_seg_s": float(seg_total_time / max(n_total_segs, 1)),
        "trapezoid_total_s": float(trap_total),
        "trapezoid_per_seg_s": float(trap_total / max(n_total_segs, 1)),
        "zupt_total_s": float(zupt_total),
        "zupt_per_seg_s": float(zupt_total / max(n_total_segs, 1)),
        "all_per_seg_s": float((seg_total_time + trap_total + zupt_total) / max(n_total_segs, 1)),
    }
    print("\nSUMMARY:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.tag}.json"
    out_path.write_text(json.dumps({"summary": summary, "details": results}, indent=2))
    print(f"\nWrote {out_path}")

    if args.compare:
        cmp_path = RESULTS_DIR / f"{args.compare}.json"
        if not cmp_path.exists():
            print(f"compare baseline not found: {cmp_path}")
            return 0
        baseline = json.loads(cmp_path.read_text())
        bs = baseline["summary"]
        print(f"\n=== Compare vs '{args.compare}' ===")
        for k in ("segmentation_per_seg_s", "trapezoid_per_seg_s",
                  "zupt_per_seg_s", "all_per_seg_s"):
            old = bs[k]; new = summary[k]
            if old > 0:
                speedup = old / max(new, 1e-12)
                print(f"  {k}: {old*1000:.2f}ms -> {new*1000:.2f}ms  ({speedup:.2f}x)")
        # Accuracy comparison
        print("\nAccuracy diff (per-experiment):")
        for r_new, r_old in zip(results, baseline["details"]):
            if r_new["target"] != r_old["target"]:
                continue
            for algo in ("trapezoid_accel", "zupt_accel"):
                old_outs = r_old["prediction"][algo]["outputs"]
                new_outs = r_new["prediction"][algo]["outputs"]
                if len(old_outs) != len(new_outs):
                    print(f"  {r_new['target']}/{algo}: count mismatch {len(old_outs)} vs {len(new_outs)}")
                    continue
                hd_old = np.array([o["height_diff"] for o in old_outs])
                hd_new = np.array([o["height_diff"] for o in new_outs])
                acc_old = np.array([o["accepted"] for o in old_outs])
                acc_new = np.array([o["accepted"] for o in new_outs])
                hd_diff = np.abs(hd_old - hd_new)
                acc_disagree = int((acc_old != acc_new).sum())
                if hd_diff.size:
                    print(f"  {r_new['target']}/{algo}: max|Δh diff|={hd_diff.max():.4f}m  "
                          f"mean={hd_diff.mean():.4f}m  accept_disagree={acc_disagree}/{len(old_outs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
