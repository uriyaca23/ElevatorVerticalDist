"""findSegments / findSegmentParameters reproduce the current pipeline."""
from __future__ import annotations

import numpy as np

from pyramidElevatorDist._orchestration import segment as _core_segment

from pyramidElevatorDist.segmentor import findSegments, findSegmentParameters
from pyramidElevatorDist.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E501
    detect as _detect,
)
from pyramidElevatorDist.types.segmentation import RideSegment, SegmentDetail

from .conftest import deep_equal


def test_findSegments_matches_core(acc):
    # resample=False: pin the package façade to the shared core on the same
    # raw acc (the core does not resample by default). The resample path is
    # covered separately.
    got = findSegments(acc, resample=False)
    expected, _state, _t0 = _core_segment(acc)
    assert len(got) == len(expected) > 0
    deep_equal([g.model_dump() for g in got], expected)


def test_findSegments_schema(acc):
    segs = findSegments(acc)
    for s in segs:
        assert isinstance(s, RideSegment)
        assert s.ride_type in ("up", "down")
        assert s.duration_s == s.t_end_s - s.t_start_s
        for lobe in (s.lobe1, s.lobe2):
            for field in ("t_c", "a_peak", "half_width_s", "frac_flat",
                          "r2_local"):
                assert isinstance(getattr(lobe, field), float)


def test_findSegmentParameters_matches_detector_fit(acc):
    """For a detected ride, the returned lobe params and heatmaps must match
    the detector's own fit (what the editor displays)."""
    # resample=False throughout so segs, the detector state, and the param
    # fit all share the raw acc time base.
    segs = findSegments(acc, resample=False)
    assert segs, "fixture produced no segments"

    # Recompute the canonical detector state once, the same way the package
    # does internally (config.json-driven), to derive the expected heatmaps.
    from pyramidElevatorDist._orchestration import _segment_cfg
    cfg = _segment_cfg()
    state = _detect.detect(acc, cfg)
    t_arr = np.asarray(state["t"])

    for seg in segs[:5]:
        out = findSegmentParameters(acc, seg.t_start_s, seg.t_end_s,
                                    ride_type=seg.ride_type, resample=False)
        res = out.detail
        assert isinstance(res, SegmentDetail)
        # Lobe parameters identical to the detector's prediction.
        deep_equal(res.lobe1.model_dump(), seg.lobe1.model_dump())
        deep_equal(res.lobe2.model_dump(), seg.lobe2.model_dump())
        deep_equal(res.joint_r2_mean, seg.joint_r2_mean)
        deep_equal(res.heatmap_energy, seg.heatmap_energy)

        # Heatmaps equal detect.heatmap_at at the lobe centres.
        for lobe, heat in ((res.lobe1, res.heatmaps.lobe1),
                           (res.lobe2, res.heatmaps.lobe2)):
            i = int(np.clip(np.argmin(np.abs(t_arr - lobe.t_c)),
                            0, t_arr.size - 1))
            expected_heat = _detect.heatmap_at(
                state["a_smooth"], state["t"], i,
                state["grid_w_s"], state["grid_f"],
            )
            deep_equal(heat, expected_heat)

        # Heatmap grids match the detector grid.
        deep_equal(res.heatmaps.grid_w_s, np.asarray(state["grid_w_s"]))
        deep_equal(res.heatmaps.grid_f, np.asarray(state["grid_f"]))


def test_findSegmentParameters_no_overlap_returns_none_or_fit(acc):
    """A window in a quiet region either fits fresh or returns None — never
    raises."""
    # Far past the end of the trace → empty/extrema-less window.
    out = findSegmentParameters(acc, 1e9, 1e9 + 5.0)
    res = None if out is None else out.detail
    assert res is None or isinstance(res, SegmentDetail)
