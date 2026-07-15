"""The public API normalizes external data to a uniform 50 Hz grid by default.

External callers may send accelerometer data at any (even variable) sample
rate, so the entry points default to ``resample=True``. The normalization now
happens *inside* the stage dispatchers (``Segmenter`` / ``Predictor`` read a
``resample_hz`` config field) rather than in the wrapper — the façade only
chooses the cadence policy. These tests cover the resample helper and pin the
default path against "explicitly resample, then run with resample=False":

* Segmentation resamples the whole trace (the detector sees it whole), so the
  two paths are bit-identical.
* Prediction resamples each ride/pre/post window independently, so the two
  paths differ only by a sub-sample resample phase at the window boundaries —
  the predicted Δh agrees to within a few cm, not bit-for-bit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyramidElevatorDist.prediction import predictSegment
from pyramidElevatorDist.segmentor import findSegments
from pyramidElevatorDist._orchestration import RESAMPLE_TARGET_HZ, _resample_acc

from .conftest import deep_equal

_PERIOD_MS = 1000.0 / RESAMPLE_TARGET_HZ  # 20 ms


def _synthetic(rate_hz: float, dur_s: float = 30.0) -> pd.DataFrame:
    n = int(dur_s * rate_hz)
    ts = np.round(np.arange(n) * (1000.0 / rate_hz)).astype("int64")
    wobble = 0.2 * np.sin(np.arange(n) / 40.0)
    return pd.DataFrame(
        {"timestamp_ms": ts, "x": wobble, "y": wobble * 0.0, "z": 9.81 + wobble}
    )


def test_resample_normalizes_odd_rate_to_50hz():
    out = _resample_acc(_synthetic(33.0))  # 33 Hz in
    dt = float(np.median(np.diff(out["timestamp_ms"].to_numpy(float))))
    assert abs(dt - _PERIOD_MS) < 1.0  # lands on a ~20 ms grid


def test_resample_preserves_t0_and_span():
    df = _synthetic(37.0)
    out = _resample_acc(df)
    assert int(out["timestamp_ms"].iloc[0]) == int(df["timestamp_ms"].iloc[0])
    raw_span = int(df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0])
    out_span = int(out["timestamp_ms"].iloc[-1] - out["timestamp_ms"].iloc[0])
    assert abs(out_span - raw_span) <= _PERIOD_MS + 1


def test_resample_noop_on_unusable():
    assert _resample_acc(None) is None
    one = pd.DataFrame({"timestamp_ms": [0], "x": [0.0], "y": [0.0], "z": [9.8]})
    assert len(_resample_acc(one)) == 1  # <2 rows: returned unchanged
    nocol = pd.DataFrame({"x": [0.0, 1.0]})
    assert _resample_acc(nocol) is nocol  # missing timestamp_ms: unchanged


def test_findSegments_default_equals_preresampled(acc):
    """Default (resample=True) == explicitly resample then resample=False."""
    got = findSegments(acc)
    expected = findSegments(_resample_acc(acc), resample=False)
    assert len(got) == len(expected)
    deep_equal([g.model_dump() for g in got],
               [e.model_dump() for e in expected])


def test_findSegments_default_finds_rides(acc):
    assert findSegments(acc), "default resample path should still detect rides"


def test_predictSegment_default_equals_preresampled(acc):
    """Default (resample=True) ≈ explicitly resample then resample=False.

    ``Predictor`` normalizes each window's cadence itself, so the two paths
    differ only by the sub-sample resample phase at the window boundaries: same
    accept verdict and sign, Δh within a few cm (10 cm tolerance leaves ample
    headroom over the ~3 cm observed while still catching a real regression).
    """
    segs = findSegments(acc)
    assert segs
    seg = {
        "type": segs[0].ride_type,
        "start_s": segs[0].t_start_s,
        "end_s": segs[0].t_end_s,
    }
    got = predictSegment(acc, seg)
    expected = predictSegment(_resample_acc(acc), seg, resample=False)
    assert got.primary == expected.primary
    for aid in ("trap", "zupt"):
        g, e = got.row(aid), expected.row(aid)
        assert g.accepted == e.accepted, f"{aid}: accept verdict differs"
        gh, eh = float(g.delta_height_m), float(e.delta_height_m)
        if np.isnan(gh) or np.isnan(eh):
            assert np.isnan(gh) and np.isnan(eh), f"{aid}: one Δh is nan"
            continue
        assert np.sign(gh) == np.sign(eh), f"{aid}: Δh sign flip {gh} vs {eh}"
        assert abs(gh - eh) <= 0.10, f"{aid}: |Δh| gap {abs(gh - eh):.4f} m > 0.10"
