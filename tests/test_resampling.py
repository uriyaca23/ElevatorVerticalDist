"""Tests for src.data.loader.resampling.

These are pure-numpy/pandas tests — no real experiment data is required.
Run them from the project root with `python -m pytest tests/test_resampling.py`
or directly: `python tests/test_resampling.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader.resampling import (
    GAP_THRESHOLD_S,
    TARGET_HZ,
    _estimate_natural_hz,
    _resample_uniform,
    _split_on_gaps,
    is_uniform_at_hz,
    prepare_sensors_uniform_50hz,
    prepare_uniform_50hz,
)


# --------------------------------------------------------------------------
# _estimate_natural_hz
# --------------------------------------------------------------------------

class TestEstimateNaturalHz:
    def test_50hz_clean(self):
        t = np.arange(0, 1000, 20, dtype=float)  # 20 ms period → 50 Hz
        assert _estimate_natural_hz(t) == 50.0

    def test_25hz_clean(self):
        t = np.arange(0, 2000, 40, dtype=float)
        assert _estimate_natural_hz(t) == 25.0

    def test_100hz_clean(self):
        t = np.arange(0, 500, 10, dtype=float)
        assert _estimate_natural_hz(t) == 100.0

    def test_jittered_50hz_snaps_to_50(self):
        rng = np.random.default_rng(0)
        # ±2 ms jitter on a 20 ms period — within the 5 % snap window.
        t = np.cumsum(20.0 + rng.uniform(-2.0, 2.0, size=200))
        assert _estimate_natural_hz(t) == 50.0

    def test_unusual_rate_returns_raw(self):
        # 30 Hz isn't in _COMMON_RATES → returned as raw rounded value.
        t = np.arange(0, 1000, 1000.0 / 30.0)
        assert abs(_estimate_natural_hz(t) - 30.0) < 0.5

    def test_too_few_samples(self):
        assert _estimate_natural_hz(np.array([0.0])) == TARGET_HZ
        assert _estimate_natural_hz(np.array([])) == TARGET_HZ

    def test_zero_dt_safe(self):
        # All identical → should not divide by zero.
        assert _estimate_natural_hz(np.array([100.0, 100.0, 100.0])) == TARGET_HZ


# --------------------------------------------------------------------------
# _split_on_gaps
# --------------------------------------------------------------------------

class TestSplitOnGaps:
    def test_no_gaps(self):
        df = pd.DataFrame({"timestamp_ms": np.arange(0, 1000, 20), "x": 0.0})
        chunks = _split_on_gaps(df)
        assert len(chunks) == 1
        assert len(chunks[0]) == len(df)

    def test_single_big_gap(self):
        # 0..200 (50 Hz), then jump to 5000 (4.8 s gap), then 5000..5200.
        t = np.r_[np.arange(0, 200, 20), np.arange(5000, 5200, 20)]
        df = pd.DataFrame({"timestamp_ms": t, "x": 0.0})
        chunks = _split_on_gaps(df, gap_threshold_ms=1000.0)
        assert len(chunks) == 2
        assert chunks[0]["timestamp_ms"].iloc[-1] < chunks[1]["timestamp_ms"].iloc[0]
        assert int(chunks[0]["timestamp_ms"].max()) == 180
        assert int(chunks[1]["timestamp_ms"].min()) == 5000

    def test_multiple_gaps(self):
        t = np.r_[
            np.arange(0, 100, 20),       # chunk 0
            np.arange(2000, 2100, 20),   # chunk 1
            np.arange(4000, 4100, 20),   # chunk 2
        ]
        df = pd.DataFrame({"timestamp_ms": t, "x": 0.0})
        chunks = _split_on_gaps(df, gap_threshold_ms=1000.0)
        assert len(chunks) == 3
        assert sum(len(c) for c in chunks) == len(df)

    def test_just_under_threshold_doesnt_split(self):
        # 999 ms gap is below the 1 s threshold.
        t = np.array([0.0, 20.0, 40.0, 1039.0, 1059.0])
        df = pd.DataFrame({"timestamp_ms": t, "x": 0.0})
        chunks = _split_on_gaps(df, gap_threshold_ms=1000.0)
        assert len(chunks) == 1

    def test_empty(self):
        df = pd.DataFrame({"timestamp_ms": [], "x": []})
        assert _split_on_gaps(df) == []

    def test_single_sample(self):
        df = pd.DataFrame({"timestamp_ms": [123], "x": [1.0]})
        chunks = _split_on_gaps(df)
        assert len(chunks) == 1 and len(chunks[0]) == 1


# --------------------------------------------------------------------------
# _resample_uniform
# --------------------------------------------------------------------------

class TestResampleUniform:
    def test_identity_on_uniform_50hz(self):
        t = np.arange(0, 1000, 20, dtype=float)
        x = np.sin(2 * np.pi * 0.5 * t / 1000.0)
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": x})
        out = _resample_uniform(df, target_hz=50.0)
        # Same number of samples, same timestamps, near-identical values.
        assert len(out) == len(df)
        np.testing.assert_array_equal(out["timestamp_ms"].to_numpy(), t.astype("int64"))
        np.testing.assert_allclose(out["x"].to_numpy(), x, atol=1e-12)

    def test_upsample_25_to_50(self):
        # 25 Hz → 50 Hz: 1-second window with linear ramp.
        t = np.arange(0, 1000, 40, dtype=float)
        x = t / 1000.0
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": x})
        out = _resample_uniform(df, target_hz=50.0)
        # Should have ~ twice as many samples.
        assert len(out) == 1 + (int(t[-1] - t[0]) // 20)
        # Linear ramp interpolates to itself exactly.
        np.testing.assert_allclose(
            out["x"].to_numpy(),
            out["timestamp_ms"].to_numpy() / 1000.0,
            atol=1e-9,
        )

    def test_downsample_100_to_50(self):
        t = np.arange(0, 1000, 10, dtype=float)
        x = t / 1000.0
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": x})
        out = _resample_uniform(df, target_hz=50.0)
        # Linear ramp must come back exact.
        np.testing.assert_allclose(
            out["x"].to_numpy(),
            out["timestamp_ms"].to_numpy() / 1000.0,
            atol=1e-9,
        )
        # 50 Hz over the same window is half the samples.
        assert abs(len(out) - len(df) / 2) <= 1

    def test_preserves_non_numeric_columns(self):
        t = np.arange(0, 100, 20, dtype=float)
        df = pd.DataFrame({
            "timestamp_ms": t.astype("int64"),
            "x":            np.zeros_like(t),
            "exp_name":     ["foo"] * len(t),
        })
        out = _resample_uniform(df, target_hz=50.0)
        assert (out["exp_name"] == "foo").all()

    def test_handles_internal_nan(self):
        # NaN samples are skipped; interpolation should still produce finite values.
        t = np.arange(0, 100, 20, dtype=float)
        x = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": x})
        out = _resample_uniform(df, target_hz=50.0)
        assert np.all(np.isfinite(out["x"].to_numpy()))

    def test_too_few_samples_passthrough(self):
        df = pd.DataFrame({"timestamp_ms": [0], "x": [5.0]})
        out = _resample_uniform(df, target_hz=50.0)
        assert len(out) == 1 and out["x"].iloc[0] == 5.0


# --------------------------------------------------------------------------
# is_uniform_at_hz
# --------------------------------------------------------------------------

class TestIsUniformAtHz:
    def test_clean_50hz_is_uniform(self):
        t = np.arange(0, 1000, 20, dtype=float)
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": 0.0})
        assert is_uniform_at_hz(df)

    def test_jittered_50hz_is_not_uniform(self):
        rng = np.random.default_rng(0)
        t = np.cumsum(20.0 + rng.uniform(-2.0, 2.0, size=100))
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": 0.0})
        assert not is_uniform_at_hz(df)

    def test_25hz_is_not_50hz_uniform(self):
        t = np.arange(0, 1000, 40, dtype=float)
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": 0.0})
        assert not is_uniform_at_hz(df)

    def test_uniform_with_gaps_still_uniform(self):
        # Two perfectly uniform 50 Hz chunks separated by a >1 s gap.
        t = np.r_[np.arange(0, 200, 20), np.arange(5000, 5200, 20)]
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": 0.0})
        assert is_uniform_at_hz(df)


# --------------------------------------------------------------------------
# prepare_uniform_50hz — the orchestrator
# --------------------------------------------------------------------------

class TestPrepareUniform50hz:
    def test_clean_50hz_passthrough(self):
        t = np.arange(0, 1000, 20, dtype=float)
        x = np.sin(2 * np.pi * 0.5 * t / 1000.0)
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": x})
        out = prepare_uniform_50hz(df)
        # Idempotent: returns the input unchanged.
        pd.testing.assert_frame_equal(out, df)

    def test_jitter_fixed(self):
        rng = np.random.default_rng(0)
        t_clean = np.arange(0, 5000, 20, dtype=float)
        t = t_clean + rng.uniform(-3.0, 3.0, size=t_clean.size)
        t = np.sort(t)
        # A signal slow enough that ±3 ms jitter doesn't matter much.
        x = np.sin(2 * np.pi * 0.5 * t / 1000.0)
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": x})
        out = prepare_uniform_50hz(df)
        # Output is uniform 50 Hz.
        assert is_uniform_at_hz(out)
        # And reasonably close to the underlying clean signal.
        x_truth = np.sin(2 * np.pi * 0.5 * out["timestamp_ms"].to_numpy() / 1000.0)
        np.testing.assert_allclose(out["x"].to_numpy(), x_truth, atol=0.05)

    def test_25hz_input_upsamples(self):
        t = np.arange(0, 1000, 40, dtype=float)
        x = t / 1000.0
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": x})
        out = prepare_uniform_50hz(df)
        assert is_uniform_at_hz(out)
        # Output covers same span at 50 Hz → ~2× samples.
        assert abs(len(out) - 2 * len(df)) <= 2

    def test_gap_preserved(self):
        t = np.r_[np.arange(0, 1000, 25), np.arange(5000, 6000, 25)]
        x = np.zeros_like(t, dtype=float)
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": x})
        out = prepare_uniform_50hz(df)
        # The big gap must still be there.
        out_t = out["timestamp_ms"].to_numpy()
        max_gap = int(np.max(np.diff(out_t)))
        assert max_gap > 1000  # gap preserved
        # And both chunks are individually uniform 50 Hz.
        assert is_uniform_at_hz(out)

    def test_dead_segment_split(self):
        # Simulate "dead" (no samples) for 5 s in the middle of a 50 Hz feed.
        t = np.r_[np.arange(0, 2000, 20), np.arange(7000, 9000, 20)]
        x = np.zeros_like(t, dtype=float)
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": x})
        out = prepare_uniform_50hz(df)
        out_t = out["timestamp_ms"].to_numpy()
        # No interpolated samples landed inside the 5 s hole.
        in_hole = (out_t > 2000) & (out_t < 7000)
        assert in_hole.sum() == 0
        assert is_uniform_at_hz(out)

    def test_empty_passthrough(self):
        df = pd.DataFrame({"timestamp_ms": [], "x": []})
        out = prepare_uniform_50hz(df)
        assert out.empty

    def test_single_sample_passthrough(self):
        df = pd.DataFrame({"timestamp_ms": [100], "x": [1.5]})
        out = prepare_uniform_50hz(df)
        assert len(out) == 1
        assert out["x"].iloc[0] == 1.5

    def test_idempotent(self):
        rng = np.random.default_rng(1)
        t = np.cumsum(20.0 + rng.uniform(-2.0, 2.0, size=300))
        x = np.sin(2 * np.pi * 0.3 * t / 1000.0)
        df = pd.DataFrame({"timestamp_ms": t.astype("int64"), "x": x})
        once = prepare_uniform_50hz(df)
        twice = prepare_uniform_50hz(once)
        pd.testing.assert_frame_equal(once, twice)


# --------------------------------------------------------------------------
# prepare_sensors_uniform_50hz — dict orchestrator
# --------------------------------------------------------------------------

class TestPrepareSensorsUniform50hz:
    def test_skips_gps(self):
        # GPS at 1 sample / 80 s — must pass through untouched.
        gps = pd.DataFrame({
            "timestamp_ms": [0, 80_000],
            "lat":          [1.0, 2.0],
            "lon":          [3.0, 4.0],
            "alt":          [10.0, 20.0],
        })
        out = prepare_sensors_uniform_50hz({"GPS": gps})
        pd.testing.assert_frame_equal(out["GPS"], gps)

    def test_processes_acc(self):
        rng = np.random.default_rng(2)
        t = np.cumsum(20.0 + rng.uniform(-2.0, 2.0, size=200)).astype("int64")
        acc = pd.DataFrame({
            "timestamp_ms": t,
            "x":            np.zeros(200),
            "y":            np.zeros(200),
            "z":            np.full(200, 9.81),
        })
        out = prepare_sensors_uniform_50hz({"ACC": acc})
        assert is_uniform_at_hz(out["ACC"])

    def test_handles_empty_sensor(self):
        out = prepare_sensors_uniform_50hz({"ACC": pd.DataFrame()})
        assert out["ACC"].empty


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
