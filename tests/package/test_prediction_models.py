"""Typed prediction results on synthetic rides: happy path, coercion,
algorithm subsets, error paths, and behavior pins."""
from __future__ import annotations

import numpy as np
import pytest

import pyramidElevatorDist as P

from .conftest import RIDE, make_prs, nan_equal

_SEG = {"type": "up", "start_s": RIDE["t_c1"] - RIDE["W"],
        "end_s": RIDE["t_c2"] + RIDE["W"]}


def test_predict_detected_segment(ride_acc):
    segs = P.findSegments(ride_acc)
    s = segs[0]
    res = P.predictSegment(
        ride_acc, {"type": s.ride_type, "start_s": s.t_start_s,
                   "end_s": s.t_end_s})
    assert isinstance(res, P.PredictionResult)
    assert res.primary == "trap"
    assert isinstance(res.trap, P.PredictionRow)
    assert res.trap.accepted is True
    assert res.trap.delta_height_m > 0  # "up" ride
    assert res.trap.duration_s == pytest.approx(
        res.trap.end_s - res.trap.start_s)
    assert res.baro is None  # no prs supplied
    assert isinstance(res.trap.meta, dict)


def test_segmentspec_and_mapping_agree(ride_acc):
    spec = P.SegmentSpec(**_SEG)
    a = P.predictSegment(ride_acc, spec)
    b = P.predictSegment(ride_acc, dict(_SEG))
    assert nan_equal(a.model_dump(), b.model_dump())


def test_algorithms_subset(ride_acc):
    res = P.predictSegment(ride_acc, _SEG, algorithms=["trap"])
    assert res.trap is not None
    assert res.zupt is None
    assert res.row("zupt") is None
    with pytest.raises(P.UnknownAlgorithmError):
        res.row("nope")


def test_unknown_algorithm_raises(ride_acc):
    with pytest.raises(P.UnknownAlgorithmError) as ei:
        P.predictSegment(ride_acc, _SEG, algorithms=["bogus"])
    assert "trap" in str(ei.value) and "zupt" in str(ei.value)


@pytest.mark.parametrize("segment", [
    {"type": "up", "start_s": 30.0, "end_s": 20.0},          # end <= start
    {"type": "up", "start_s": float("nan"), "end_s": 20.0},  # NaN bound
    {"type": "up", "star_s": 10.0, "end_s": 20.0},           # typo key
    {"type": "sideways", "start_s": 10.0, "end_s": 20.0},    # bad type
])
def test_invalid_segment_raises(ride_acc, segment):
    with pytest.raises(P.InvalidSegmentError):
        P.predictSegment(ride_acc, segment)


def test_out_of_range_segment_is_reject_row(ride_acc):
    """Valid floats beyond the trace → 'empty_slice' reject row, not an
    exception (behavior pin)."""
    res = P.predictSegment(
        ride_acc, {"type": "up", "start_s": 1e6, "end_s": 1e6 + 10.0})
    assert res.trap is not None
    assert res.trap.accepted is False
    assert res.trap.reject_reason == "empty_slice"
    assert np.isnan(res.trap.delta_height_m)


def test_predict_by_parameters_override(ride_acc):
    params = P.TrapezoidParams(W=RIDE["W"], f=RIDE["f"], abs_A=RIDE["A"])
    res = P.predictByParameters(ride_acc, _SEG, params)
    override = res.trap.meta.get("trapezoid_override") or {}
    assert override.get("source") == "manual"
    base = P.predictSegment(ride_acc, _SEG)
    assert res.trap.delta_height_m == pytest.approx(
        base.trap.delta_height_m, rel=0.25)


@pytest.mark.parametrize("params", [
    {"W": -1.0, "f": 0.5, "abs_A": 1.0},
    {"W": 1.0, "f": 1.5, "abs_A": 1.0},
    {"W": 1.0, "f": 0.5, "abs_A": -0.1},
    {"W": 1.0, "f": 0.5, "abs_a": 1.0},  # typo key
])
def test_invalid_trapezoid_params(ride_acc, params):
    with pytest.raises(P.InvalidTrapezoidParamsError):
        P.predictByParameters(ride_acc, _SEG, params)


def test_baro_row_with_prs(ride_acc):
    res = P.predictSegment(ride_acc, _SEG, prs=make_prs(dh_m=3.0))
    assert isinstance(res.baro, P.PredictionRow)
    assert res.baro.delta_height_m == pytest.approx(3.0, abs=1.0)


def test_baro_row_empty_prs_pin(ride_acc):
    """Empty (0-row) prs is passed through → documented 'no_pressure' row
    (behavior pin)."""
    import pandas as pd
    empty = pd.DataFrame({"timestamp_ms": [], "pressure": []})
    res = P.predictSegment(ride_acc, _SEG, prs=empty)
    assert res.baro is not None
    assert res.baro.reject_reason == "no_pressure"
