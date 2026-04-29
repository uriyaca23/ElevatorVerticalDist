"""Stress tests: corrupt a real measurement and verify the loader+algorithms
still produce sensible output.

Three deformations are applied to a clean experiment's sensor frames:

* **jitter** — perturb every timestamp by ±5 ms.
* **dropouts** — drop 30 % of rows at random.
* **dead_segments** — punch out 2 random ~3 s windows (sets nothing
  during them; pure data hole).
* **off-rate** — keep every other sample (effectively 25 Hz).

For each variant we:
  1. resample through the loader's pipeline and verify the output is
     uniform 50 Hz with no >1 s holes (apart from the deliberate ones).
  2. run the barometric Δh predictor per ride and check |Δh_corrupt − Δh_clean|
     stays bounded.

The tolerance is loose because the data has been mangled — but the
pipeline should NOT crash, NOT produce NaN, and should track the clean
prediction within a couple meters per ride.

This module is run by pytest; it is skipped if the reference experiment
isn't materialised on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import getExperimentData
from src.data.loader.constants import STRUCTURED_DATA_DIR
from src.data.loader.resampling import (
    is_uniform_at_hz,
    prepare_sensors_uniform_50hz,
)
from src.prediction.algorithms.barometer_only import (
    predict_height_difference_from_barometer,
)
from src.prediction.algorithms.configTypes import (
    BarometerHeightDiffConfig,
    PREDICT_ALGORITHM_CONFIG,
    PredictAlgorithm,
)


REFERENCE_EXP = "UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1"
SEED = 42


def _exp_dir_exists(name: str) -> bool:
    return (Path(STRUCTURED_DATA_DIR) / name / "PRS.csv").exists()


# --------------------------------------------------------------------------
# Deformations
# --------------------------------------------------------------------------

def _apply_jitter(df: pd.DataFrame, rng: np.random.Generator, max_ms: float = 5.0) -> pd.DataFrame:
    out = df.copy()
    delta = rng.uniform(-max_ms, max_ms, size=len(out)).round().astype("int64")
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64") + delta
    return out.sort_values("timestamp_ms", kind="stable").reset_index(drop=True)


def _apply_dropouts(df: pd.DataFrame, rng: np.random.Generator, frac: float = 0.30) -> pd.DataFrame:
    keep = rng.uniform(0, 1, size=len(df)) > frac
    return df[keep].reset_index(drop=True)


def _apply_dead_segments(
    df: pd.DataFrame, rng: np.random.Generator, n_holes: int = 2,
    hole_duration_ms: int = 3000,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ts = df["timestamp_ms"].to_numpy()
    span_lo, span_hi = int(ts[0]) + 5000, int(ts[-1]) - 5000
    if span_hi <= span_lo:
        return df.copy()
    keep_mask = np.ones(len(df), dtype=bool)
    for _ in range(n_holes):
        start = int(rng.integers(span_lo, span_hi))
        end = start + hole_duration_ms
        keep_mask &= ~((ts >= start) & (ts < end))
    return df[keep_mask].reset_index(drop=True)


def _apply_off_rate(df: pd.DataFrame) -> pd.DataFrame:
    # Keep every other row → halves the sample rate.
    return df.iloc[::2].reset_index(drop=True)


def _corrupt(sensors: dict[str, pd.DataFrame], mode: str, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    for name, df in sensors.items():
        if df is None or df.empty or name == "GPS":
            out[name] = df
            continue
        if mode == "jitter":
            out[name] = _apply_jitter(df, rng)
        elif mode == "dropouts":
            out[name] = _apply_dropouts(df, rng)
        elif mode == "dead":
            out[name] = _apply_dead_segments(df, rng)
        elif mode == "off-rate":
            out[name] = _apply_off_rate(df)
        else:
            raise ValueError(mode)
    return out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _baro_dh_per_ride(prs: pd.DataFrame, gt: pd.DataFrame, cfg: BarometerHeightDiffConfig
                     ) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    rides = gt[gt["type"].isin(["up", "down"])]
    for _, row in rides.iterrows():
        s, e = int(row["start_ms"]), int(row["end_ms"])
        seg = prs[(prs["timestamp_ms"] >= s) & (prs["timestamp_ms"] < e)]
        if len(seg) < 2:
            continue
        out[(s, e)] = float(predict_height_difference_from_barometer(seg, cfg))
    return out


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not _exp_dir_exists(REFERENCE_EXP),
    reason=f"Reference experiment {REFERENCE_EXP} not on disk",
)
@pytest.mark.parametrize("mode,tol_max_m", [
    ("jitter",   0.50),  # almost no effect — barometer Δh is well-conditioned
    ("dropouts", 1.50),  # 30 % loss; resampler interpolates
    ("dead",     2.00),  # 2× ~3 s holes; some rides may straddle them
    ("off-rate", 0.50),  # 25 Hz instead of 50 Hz; resampler upsamples
])
def test_corruption_then_resample_keeps_predictions_bounded(mode: str, tol_max_m: float):
    sensors_clean, gt, _ = getExperimentData(REFERENCE_EXP)
    cfg = PREDICT_ALGORITHM_CONFIG(algorithm=PredictAlgorithm.BAROMETER_HEIGHT_DIFF)
    bcfg = BarometerHeightDiffConfig(**cfg.load_params())

    dh_clean = _baro_dh_per_ride(sensors_clean["PRS"], gt, bcfg)
    assert dh_clean, "no rides on clean — test is meaningless"

    sensors_corrupt = _corrupt(sensors_clean, mode, seed=SEED)
    sensors_fixed = prepare_sensors_uniform_50hz(sensors_corrupt)

    # Loader contract: every non-skipped, non-empty sensor must come out
    # uniform 50 Hz (with original >1 s gaps preserved as sample-time gaps).
    for name, df in sensors_fixed.items():
        if df is None or df.empty or name == "GPS" or len(df) < 2:
            continue
        assert is_uniform_at_hz(df), f"{name} not uniform 50 Hz after resample (mode={mode})"
        assert df[["timestamp_ms"]].notna().all().all()

    # Predictions must be finite and reasonably close to clean.
    dh_corrupt = _baro_dh_per_ride(sensors_fixed["PRS"], gt, bcfg)
    common_keys = set(dh_clean) & set(dh_corrupt)
    # Allow up to 2 rides to be lost entirely if a hole punched the segment.
    assert len(dh_clean) - len(common_keys) <= 2, (
        f"{mode}: too many rides lost ({len(dh_clean) - len(common_keys)}/{len(dh_clean)})"
    )
    diffs = np.array([abs(dh_corrupt[k] - dh_clean[k]) for k in common_keys])
    assert np.all(np.isfinite(diffs)), f"{mode}: predictions returned non-finite values"
    assert np.max(diffs) <= tol_max_m, (
        f"{mode}: max |Δh_corrupt − Δh_clean| = {np.max(diffs):.2f} m exceeds tol {tol_max_m} m. "
        f"Per-ride: {sorted(diffs.tolist(), reverse=True)[:5]}"
    )


@pytest.mark.skipif(
    not _exp_dir_exists(REFERENCE_EXP),
    reason=f"Reference experiment {REFERENCE_EXP} not on disk",
)
def test_corruption_doesnt_crash_loader_path():
    """Even adversarial inputs must flow through prepare_sensors_uniform_50hz
    without throwing."""
    sensors_clean, _, _ = getExperimentData(REFERENCE_EXP)
    for mode in ("jitter", "dropouts", "dead", "off-rate"):
        corrupted = _corrupt(sensors_clean, mode, seed=SEED)
        # Must not raise.
        out = prepare_sensors_uniform_50hz(corrupted)
        assert isinstance(out, dict)
        for df in out.values():
            if df is not None and not df.empty:
                # Numeric columns must all be finite (resampler does not
                # introduce NaN; clean data has no NaN here).
                num = df.select_dtypes(include=[np.number])
                # exp_name is non-numeric so excluded; everything else must
                # be finite.
                assert num.notna().all().all(), f"{mode}: NaN appeared post-resample"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
