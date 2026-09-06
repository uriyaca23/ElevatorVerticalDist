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


# --------------------------------------------------------------------------
# ε segment-padding — every segmentation entry point must describe a ride
# the same way. The pad exists so the Δh estimators (which re-fit / double-
# integrate inside the window they are given) see the full acceleration
# tails; an entry point that drops it silently hands callers a shorter ride.
# --------------------------------------------------------------------------

def _segment_pad_eps_s() -> float:
    """The live ε from the packaged detector config (not a hard-coded 0.25,
    so the invariant survives a retune)."""
    from pyramidElevatorDist.segmentation.algorithms.configTypes import (
        SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, TemplateMatchConfig,
    )
    params = SEGMENT_ALGORITHM_CONFIG(
        algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
    ).load_params()
    return float(TemplateMatchConfig(**params).segment_pad_eps_s)


def _assert_eps_padded(view, eps: float) -> None:
    """``[t_c1 - W - ε, t_c2 + W + ε]`` — the detector's emitted interval."""
    assert view.t_start_s == pytest.approx(
        view.lobe1.t_c - view.lobe1.half_width_s - eps)
    assert view.t_end_s == pytest.approx(
        view.lobe2.t_c + view.lobe2.half_width_s + eps)


def test_find_segments_bounds_carry_the_eps_pad(ride_acc):
    """Reference behavior pin: findSegments is the padded one."""
    eps = _segment_pad_eps_s()
    assert eps > 0.0, "fixture assumes a non-zero pad"
    for s in P.findSegments(ride_acc):
        _assert_eps_padded(s, eps)


def test_detail_bounds_match_their_own_ride(ride_acc):
    """``DetailedRideSegment.detail`` must not disagree with the ride it
    hangs off."""
    for d in P.findSegmentsDetailed(ride_acc):
        assert d.detail is not None
        assert d.detail.t_start_s == pytest.approx(d.t_start_s)
        assert d.detail.t_end_s == pytest.approx(d.t_end_s)


def test_segment_parameters_returns_the_detected_ride_bounds(ride_acc):
    """Asking for a detected ride's own window returns that ride's bounds."""
    seg = P.findSegments(ride_acc)[0]
    res = P.findSegmentParameters(
        ride_acc, seg.t_start_s, seg.t_end_s, ride_type=seg.ride_type)
    assert res is not None
    assert res.t_start_s == pytest.approx(seg.t_start_s)
    assert res.t_end_s == pytest.approx(seg.t_end_s)


def test_every_entry_point_describes_the_ride_identically(ride_acc):
    """findSegments / findSegmentsDetailed / .detail / findSegmentParameters
    are four views of one ride — they must agree on all of it."""
    plain = P.findSegments(ride_acc)
    detailed = P.findSegmentsDetailed(ride_acc)
    assert len(plain) == len(detailed) >= 1
    for p, d in zip(plain, detailed):
        params = P.findSegmentParameters(
            ride_acc, p.t_start_s, p.t_end_s, ride_type=p.ride_type)
        assert params is not None
        assert d.detail is not None
        for view in (d, d.detail, params):
            assert view.ride_type == p.ride_type
            assert view.t_start_s == pytest.approx(p.t_start_s)
            assert view.t_end_s == pytest.approx(p.t_end_s)
            assert nan_equal(view.lobe1.model_dump(), p.lobe1.model_dump())
            assert nan_equal(view.lobe2.model_dump(), p.lobe2.model_dump())
            assert view.joint_r2_mean == pytest.approx(p.joint_r2_mean)
            assert view.heatmap_energy == pytest.approx(p.heatmap_energy)


def test_fresh_fit_window_is_eps_padded_too(subthreshold_ride_acc):
    """The ``_fit_fresh`` branch — a window the detector proposed no ride for
    — pads the same way, so a hand-marked interval and a detected one are
    described on the same terms."""
    eps = _segment_pad_eps_s()
    segs = P.findSegments(subthreshold_ride_acc)
    assert not any(s.t_start_s <= 70.0 and s.t_end_s >= 50.0 for s in segs), \
        "fixture must leave [50, 70] s undetected to reach the fresh-fit branch"
    res = P.findSegmentParameters(subthreshold_ride_acc, 50.0, 70.0, "up")
    assert res is not None
    _assert_eps_padded(res, eps)
