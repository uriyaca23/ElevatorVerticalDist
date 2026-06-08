"""findSegments / findSegmentParameters reproduce the current pipeline."""
from __future__ import annotations

import numpy as np

from ui import api_client

from pyramidElevatorDist.segmentor import findSegments, findSegmentParameters
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E501
    detect as _detect,
)

from .conftest import deep_equal


def test_findSegments_matches_api_client(acc):
    # resample=False: pin the core detector to api_client on the same raw acc
    # (api_client does not resample). The resample path is covered separately.
    got = findSegments(acc, resample=False)
    expected, _state, _t0 = api_client.segment(acc)
    assert len(got) == len(expected) > 0
    deep_equal(got, expected)


def test_findSegments_schema(acc):
    segs = findSegments(acc)
    for s in segs:
        assert set(s) >= {
            "ride_type", "t_start_s", "t_end_s", "duration_s",
            "lobe1", "lobe2", "joint_r2_mean", "heatmap_energy",
        }
        assert s["ride_type"] in ("up", "down")
        for lobe in ("lobe1", "lobe2"):
            assert set(s[lobe]) >= {
                "t_c", "a_peak", "half_width_s", "frac_flat", "r2_local",
            }


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
        res = findSegmentParameters(acc, seg["t_start_s"], seg["t_end_s"],
                                    ride_type=seg["ride_type"], resample=False)
        assert res is not None
        # Lobe parameters identical to the detector's prediction.
        deep_equal(res["lobe1"], seg["lobe1"])
        deep_equal(res["lobe2"], seg["lobe2"])
        deep_equal(res["joint_r2_mean"], seg["joint_r2_mean"])
        deep_equal(res["heatmap_energy"], seg["heatmap_energy"])

        # Heatmaps equal detect.heatmap_at at the lobe centres.
        for lobe_key in ("lobe1", "lobe2"):
            t_c = res[lobe_key]["t_c"]
            i = int(np.clip(np.argmin(np.abs(t_arr - t_c)),
                            0, t_arr.size - 1))
            expected_heat = _detect.heatmap_at(
                state["a_smooth"], state["t"], i,
                state["grid_w_s"], state["grid_f"],
            )
            deep_equal(res["heatmaps"][lobe_key], expected_heat)

        # Heatmap grids match the detector grid.
        deep_equal(res["heatmaps"]["grid_w_s"], np.asarray(state["grid_w_s"]))
        deep_equal(res["heatmaps"]["grid_f"], np.asarray(state["grid_f"]))


def test_findSegmentParameters_no_overlap_returns_none_or_fit(acc):
    """A window in a quiet region either fits fresh or returns None — never
    raises."""
    # Far past the end of the trace → empty/extrema-less window.
    res = findSegmentParameters(acc, 1e9, 1e9 + 5.0)
    assert res is None or "lobe1" in res
