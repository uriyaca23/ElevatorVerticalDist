"""Side-by-side prediction evaluation: ``a_vert`` vs ``|a| − g``.

Runs each accelerometer predictor (ZUPT + trapezoid) twice — once with
the default gravity-projected vertical, once with the rotation-invariant
magnitude residual — fits conformal on the train half, applies to test.
Stdout streams progress live; no figures, no run dir.

Each (algo, signal) does TWO inference passes:
  1. ``run_predictions(train)``       — fits calibration; gives train MAE/RMSE.
  2. ``run_predictions(test_after_calibrate)`` — gives test metrics with
      the freshly-fit conformal multiplier.

We deliberately skip the third "re-run train with calibrated multiplier"
pass that ``evaluateOnData.py`` performs, because MAE/RMSE/medAE do not
depend on the multiplier — only coverage and CI width do, and we report
those on TEST.

Usage:
    venv/bin/python -u -m scripts.compare_prediction_input_signal
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import list_experiments  # noqa: E402
from src.prediction.algorithms import (  # noqa: E402
    PREDICT_ALGORITHM_CONFIG,
    PredictAlgorithm,
    Predictor,
)
from src.prediction.evaluation.dataset import build_segment_records  # noqa: E402
from src.prediction.evaluation.metrics import compute_metrics  # noqa: E402
from src.prediction.evaluation.runner import (  # noqa: E402
    collect_predictions,
    run_predictions,
    to_calibration_samples,
)


ALGOS = [
    ("zupt", PredictAlgorithm.ZUPT_ACCEL),
    ("trapezoid", PredictAlgorithm.TRAPEZOID_ACCEL),
]
SIGNALS = ["a_vert", "a_mag_minus_g"]


def _say(msg: str) -> None:
    print(msg, flush=True)


def _load(experiments: list[str], label: str) -> list:
    t0 = time.time()
    recs = []
    for name in experiments:
        recs.extend(build_segment_records(name, verbose=False))
    _say(f"  loaded {label}: {len(recs)} segments  ({time.time() - t0:.1f}s)")
    return recs


def _run_one(algo_name: str, algo_enum, signal: str, train_recs, test_recs) -> dict:
    _say(f"\n[{algo_name} / {signal}] starting ...")
    cfg = PREDICT_ALGORITHM_CONFIG(
        algorithm=algo_enum, overrides={"input_signal": signal},
    )
    p = Predictor(cfg)

    t0 = time.time()
    train_preds = run_predictions(p, train_recs)
    _say(f"  train inference done  ({time.time() - t0:.1f}s, "
         f"{len(train_preds)} preds)")

    t1 = time.time()
    calib = p.calibrate(to_calibration_samples(train_preds))
    _say(f"  calibration: {calib}  ({time.time() - t1:.1f}s)")

    t2 = time.time()
    test_preds_cal = run_predictions(p, test_recs)
    _say(f"  test inference done  ({time.time() - t2:.1f}s, "
         f"{len(test_preds_cal)} preds)")

    train_df = collect_predictions(train_preds)
    test_df = collect_predictions(test_preds_cal)
    return {
        "calib": calib,
        "train": compute_metrics(train_df),
        "test":  compute_metrics(test_df),
        "time_sec": time.time() - t0,
    }


def _fmt_row(label: str, m) -> str:
    return (
        f"{label:30s} "
        f"n_clean={m.n_clean:3d}  "
        f"MAE={m.clean_mae:5.2f}m  "
        f"medAE={m.clean_median_abs_err:5.2f}m  "
        f"RMSE={m.clean_rmse:5.2f}m  "
        f"cov90={m.clean_coverage_90:5.1%}  "
        f"medCI=±{m.clean_median_ci:5.2f}m  "
        f"<1.5m={m.clean_frac_within_1_5m:5.1%}  "
        f"<3m={m.clean_frac_within_3m:5.1%}"
    )


def main() -> int:
    train_exps = list_experiments(kind="train")
    test_exps  = list_experiments(kind="test")
    _say(f"resolving experiments: train={len(train_exps)} test={len(test_exps)}")
    train_recs = _load(train_exps, "train")
    test_recs  = _load(test_exps,  "test ")

    results: dict[tuple[str, str], dict] = {}
    for algo_name, algo_enum in ALGOS:
        for signal in SIGNALS:
            results[(algo_name, signal)] = _run_one(
                algo_name, algo_enum, signal, train_recs, test_recs,
            )

    _say("\n" + "=" * 95)
    _say("FINAL RESULTS")
    _say("=" * 95)
    for algo_name, _ in ALGOS:
        _say(f"\n--- {algo_name.upper()} ---")
        for split in ("train", "test"):
            _say(f"  [{split}]")
            base = results[(algo_name, "a_vert")][split]
            new  = results[(algo_name, "a_mag_minus_g")][split]
            _say("    " + _fmt_row("a_vert    (baseline)", base))
            _say("    " + _fmt_row("|a| - g   (treatment)", new))
            d = lambda a, b: f"{(b - a):+.2f}"
            dp = lambda a, b: f"{(b - a) * 100:+.1f}pp"
            _say(
                f"    {'Δ (treatment − baseline)':30s} "
                f"n_clean={new.n_clean - base.n_clean:+3d}    "
                f"MAE={d(base.clean_mae, new.clean_mae)}m  "
                f"medAE={d(base.clean_median_abs_err, new.clean_median_abs_err)}m  "
                f"RMSE={d(base.clean_rmse, new.clean_rmse)}m  "
                f"cov90={dp(base.clean_coverage_90, new.clean_coverage_90)}  "
                f"medCI={d(base.clean_median_ci, new.clean_median_ci)}m  "
                f"<1.5m={dp(base.clean_frac_within_1_5m, new.clean_frac_within_1_5m)}  "
                f"<3m={dp(base.clean_frac_within_3m, new.clean_frac_within_3m)}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
