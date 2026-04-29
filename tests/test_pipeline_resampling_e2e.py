"""End-to-end sanity check: resampling preserves prediction quality on real data.

Loads one experiment two ways:

    A. raw CSV read directly from disk (simulating the pre-fix behaviour, since
       the cached CSVs were written before the loader-side resampler existed).
    B. through `getExperimentData`, which now applies the resampler.

For each ground-truth `up`/`down` segment, runs the barometer height-diff
predictor and compares Δh between the two paths. We expect them to be very
close (a few cm) — barometer Δh is a difference of edge-averaged altitudes,
which is essentially shift-invariant under linear resampling at the segment
edges.

This test is skipped if the experiment isn't materialised on disk (e.g. CI
without the dataset).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import getExperimentData
from src.data.loader.constants import STRUCTURED_DATA_DIR
from src.prediction.algorithms.barometer_only import (
    predict_height_difference_from_barometer,
)
from src.prediction.algorithms.configTypes import (
    BarometerHeightDiffConfig,
    PREDICT_ALGORITHM_CONFIG,
    PredictAlgorithm,
)


REFERENCE_EXP = "UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1"


def _exp_dir_exists(name: str) -> bool:
    return (Path(STRUCTURED_DATA_DIR) / name / "PRS.csv").exists()


@pytest.mark.skipif(
    not _exp_dir_exists(REFERENCE_EXP),
    reason=f"Reference experiment {REFERENCE_EXP} not on disk",
)
def test_barometer_predictions_unchanged_after_resampling():
    cfg = PREDICT_ALGORITHM_CONFIG(algorithm=PredictAlgorithm.BAROMETER_HEIGHT_DIFF)
    bcfg = BarometerHeightDiffConfig(**cfg.load_params())

    # ---- Path A: raw CSV from disk (pre-fix behaviour) ------------------
    raw_dir = Path(STRUCTURED_DATA_DIR) / REFERENCE_EXP
    prs_raw = pd.read_csv(raw_dir / "PRS.csv")
    gt = pd.read_csv(raw_dir / "gt.csv")

    # ---- Path B: through the loader (post-fix; resampled) ---------------
    sensors_new, gt_new, _ = getExperimentData(REFERENCE_EXP)
    prs_new = sensors_new["PRS"]

    # PRS in the cache was 25 Hz; after resampling we expect 50 Hz uniform.
    dt_new = np.diff(prs_new["timestamp_ms"].to_numpy())
    assert np.all(dt_new == 20), (
        f"resampler did not produce uniform 50 Hz PRS (dt range "
        f"{dt_new.min()}..{dt_new.max()})"
    )

    # Compare per-ride Δh between the two paths.
    rides = gt[gt["type"].isin(["up", "down"])]
    deltas = []
    for _, row in rides.iterrows():
        s, e = int(row["start_ms"]), int(row["end_ms"])

        seg_raw = prs_raw[(prs_raw["timestamp_ms"] >= s) & (prs_raw["timestamp_ms"] < e)]
        seg_new = prs_new[(prs_new["timestamp_ms"] >= s) & (prs_new["timestamp_ms"] < e)]

        if len(seg_raw) < 2 or len(seg_new) < 2:
            continue

        dh_raw = predict_height_difference_from_barometer(seg_raw, bcfg)
        dh_new = predict_height_difference_from_barometer(seg_new, bcfg)
        deltas.append(abs(dh_raw - dh_new))

    deltas = np.asarray(deltas)
    assert deltas.size > 0, "no rides covered — test is meaningless"
    # Median Δh deviation should be < 3 cm; max < 30 cm. This is loose
    # enough to allow for the edge-sample reshuffling caused by upsampling.
    assert np.median(deltas) < 0.03, f"median Δh shifted by {np.median(deltas):.3f} m"
    assert np.max(deltas) < 0.30, f"max Δh shifted by {np.max(deltas):.3f} m"


@pytest.mark.skipif(
    not _exp_dir_exists(REFERENCE_EXP),
    reason=f"Reference experiment {REFERENCE_EXP} not on disk",
)
def test_loader_returns_uniform_50hz_for_all_sensors():
    sensors, _, _ = getExperimentData(REFERENCE_EXP)
    for name, df in sensors.items():
        if df is None or df.empty or name == "GPS" or len(df) < 2:
            continue
        t = df["timestamp_ms"].to_numpy()
        dt = np.diff(t)
        # Allow gaps (>1 s) — but every other dt must be exactly 20 ms.
        non_gap = dt[dt <= 1000]
        assert np.all(non_gap == 20), (
            f"{name}: non-uniform sampling after loader "
            f"(unique dts ≤1 s: {sorted(set(non_gap.tolist()))[:5]})"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
