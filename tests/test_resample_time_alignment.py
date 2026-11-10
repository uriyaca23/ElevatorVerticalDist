"""Resampling must preserve the absolute time of signal features.

Regression test for a time-warp bug: ``_resample_sensor_to_hz`` rounded the
source rate to an integer before ``scipy.signal.resample_poly`` (e.g. a true
52.632 Hz recording was treated as 53 Hz). ``resample_poly`` then produced an
output whose implied rate (49.65 Hz) did not match the 50 Hz timestamp grid it
was stamped onto, warping time linearly — a sharp feature at 1150 s drifted to
1142 s. These tests pin every feature to its true time after resampling.
"""

import numpy as np
import pandas as pd

from src.data.loader.pipeline import _resample_sensor_to_hz


def _spiked_frame(true_hz: float, duration_s: float, spike_times_s):
    n = int(duration_s * true_hz)
    period_ms = 1000.0 / true_hz
    ts = np.round(np.arange(n) * period_ms).astype("int64")
    x = np.zeros(n)
    for st in spike_times_s:
        x[int(round(st * true_hz))] = 1.0
    return pd.DataFrame({"timestamp_ms": ts, "x": x})


def _recovered_spike_time_s(out: pd.DataFrame, near_s: float, window_s=25.0):
    to = out["timestamp_ms"].to_numpy(float)
    xo = out["x"].to_numpy(float)
    m = (to / 1000 > near_s - window_s) & (to / 1000 < near_s + window_s)
    return to[m][int(np.argmax(np.abs(xo)[m]))] / 1000.0


def test_noninteger_source_rate_does_not_warp_time():
    # 52.632 Hz (dt=19 ms) — the beitMansour S23 rate that triggered the bug.
    spikes = [100, 300, 600, 900, 1150]
    df = _spiked_frame(1000 / 19.0, 1200.0, spikes)
    out = _resample_sensor_to_hz(df, target_hz=50)
    for st in spikes:
        got = _recovered_spike_time_s(out, st)
        assert abs(got - st) < 0.5, f"spike at {st}s drifted to {got:.2f}s"


def test_integer_downsample_preserves_time():
    spikes = [50, 250, 500]
    df = _spiked_frame(100.0, 600.0, spikes)  # 100 -> 50 Hz, clean 2:1
    out = _resample_sensor_to_hz(df, target_hz=50)
    for st in spikes:
        got = _recovered_spike_time_s(out, st)
        assert abs(got - st) < 0.5, f"spike at {st}s drifted to {got:.2f}s"


def test_upsample_noninteger_preserves_time():
    spikes = [40, 200, 480]
    df = _spiked_frame(1000 / 37.0, 520.0, spikes)  # ~27 Hz -> 50 Hz upsample
    out = _resample_sensor_to_hz(df, target_hz=50)
    for st in spikes:
        got = _recovered_spike_time_s(out, st)
        assert abs(got - st) < 0.5, f"spike at {st}s drifted to {got:.2f}s"


def test_jittery_dt_does_not_warp_time():
    # Real phone ACC alternates dt (here 19/20 ms, like the Galaxy S23). A
    # sample-count resampler that ignores timestamps drifts features by many
    # seconds; a time-aware one keeps them put.
    rng_dt = np.resize([19, 20], 70000)  # alternating 19/20 ms ~ 51.3 Hz
    ts = np.concatenate([[0], np.cumsum(rng_dt)]).astype("int64")
    spikes_s = [100, 400, 800, 1200]
    x = np.zeros(len(ts))
    for st in spikes_s:
        x[int(np.argmin(np.abs(ts - st * 1000)))] = 1.0
    out = _resample_sensor_to_hz(pd.DataFrame({"timestamp_ms": ts, "x": x}),
                                 target_hz=50)
    for st in spikes_s:
        got = _recovered_spike_time_s(out, st, window_s=30.0)
        assert abs(got - st) < 0.5, f"spike at {st}s drifted to {got:.2f}s"
