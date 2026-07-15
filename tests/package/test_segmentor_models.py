"""Typed segmentation results on synthetic rides: detection, parameter
recovery, behavior pins, and model structure."""
from __future__ import annotations

import json

import numpy as np
import pydantic
import pytest

import pyramidElevatorDist as P

from .conftest import RIDE, make_ride_acc, nan_equal


def _matching_segment(segs, t_lo, t_hi):
    """The detected segment overlapping [t_lo, t_hi]."""
    for s in segs:
        if s.t_start_s <= t_hi and s.t_end_s >= t_lo:
            return s
    raise AssertionError(
        f"no segment overlaps [{t_lo}, {t_hi}]; got "
        f"{[(s.t_start_s, s.t_end_s) for s in segs]}"
    )


def test_finds_synthetic_up_ride(ride_acc):
    segs = P.findSegments(ride_acc)
    assert len(segs) >= 1
    assert all(isinstance(s, P.RideSegment) for s in segs)
    s = _matching_segment(segs, RIDE["t_c1"] - RIDE["W"],
                          RIDE["t_c2"] + RIDE["W"])
    assert s.ride_type == "up"
    assert abs(s.t_start_s - (RIDE["t_c1"] - RIDE["W"])) < 1.0
    assert abs(s.t_end_s - (RIDE["t_c2"] + RIDE["W"])) < 1.0
    assert s.duration_s == pytest.approx(s.t_end_s - s.t_start_s)
    assert s.lobe1.a_peak > 0 > s.lobe2.a_peak
    assert s.lobe1.half_width_s == s.lobe2.half_width_s
    assert 0.9 <= s.joint_r2_mean <= 1.0


def test_finds_down_ride_mirrored():
    acc = make_ride_acc(direction="down")
    segs = P.findSegments(acc)
    s = _matching_segment(segs, RIDE["t_c1"] - RIDE["W"],
                          RIDE["t_c2"] + RIDE["W"])
    assert s.ride_type == "down"
    assert s.lobe1.a_peak < 0 < s.lobe2.a_peak


def test_short_trace_returns_empty_list(quiet_acc):
    """No ride → empty list, no exception (behavior pin)."""
    assert P.findSegments(quiet_acc) == []


def test_segment_parameters_happy_path(ride_acc):
    res = P.findSegmentParameters(
        ride_acc, RIDE["t_c1"] - 2.0, RIDE["t_c2"] + 2.0)
    assert isinstance(res, P.SegmentDetail)
    assert res.heatmaps.lobe1.shape == (res.heatmaps.grid_w_s.size,
                                        res.heatmaps.grid_f.size)
    assert res.correlation.t.size == res.correlation.best_pos_r2.size
    # Parameter recovery ≈ synthesis (within a grid step / noise).
    assert res.lobe1.half_width_s == pytest.approx(RIDE["W"], abs=0.4)
    assert abs(res.lobe1.a_peak) == pytest.approx(RIDE["A"], abs=0.3)


def test_segment_parameters_quiet_window_none_or_detail(quiet_acc):
    """A quiet window either fits fresh or returns None — never raises
    (behavior pin)."""
    res = P.findSegmentParameters(quiet_acc, 5.0, 10.0)
    assert res is None or isinstance(res, P.SegmentDetail)


def test_detailed_matches_parameters(ride_acc):
    detailed = P.findSegmentsDetailed(ride_acc)
    assert len(detailed) >= 1
    for d in detailed:
        assert isinstance(d, P.DetailedRideSegment)
        if d.detail is None:
            continue
        res = P.findSegmentParameters(
            ride_acc, d.t_start_s, d.t_end_s, ride_type=d.ride_type)
        assert res is not None
        assert nan_equal(d.detail.model_dump(), res.model_dump())


def test_model_roundtrip_and_json(ride_acc):
    segs = P.findSegments(ride_acc)
    s = segs[0]
    assert P.RideSegment.model_validate(s.model_dump()).model_dump() \
        == s.model_dump()

    detail = P.findSegmentParameters(
        ride_acc, RIDE["t_c1"] - 2.0, RIDE["t_c2"] + 2.0)
    payload = detail.model_dump(mode="json")
    json.dumps(payload)  # ndarrays serialized to lists

    with pytest.raises(pydantic.ValidationError):
        P.RideSegment.model_validate({**s.model_dump(), "unknown_key": 1})
