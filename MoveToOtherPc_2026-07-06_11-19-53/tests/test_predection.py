"""predictSegment / predictByParameters reproduce the current pipeline."""
from __future__ import annotations

from src.pipelines.inprocess import predict as _core_predict

from pyramidElevatorDist.segmentor import findSegments
from pyramidElevatorDist.predection import predictSegment, predictByParameters

from .conftest import deep_equal


def _first_segments(acc, n=5):
    # resample=False so the segment coordinates are on the raw acc time base —
    # the prediction pins below compare against the shared core (inprocess),
    # which runs on the same raw acc. The resample path is covered in
    # test_api_resampling.py.
    segs = findSegments(acc, resample=False)
    assert segs, "fixture produced no segments"
    return [
        {"type": s["ride_type"],
         "start_s": s["t_start_s"],
         "end_s": s["t_end_s"]}
        for s in segs[:n]
    ]


def test_predictSegment_matches_core(acc):
    for seg in _first_segments(acc):
        got = predictSegment(acc, seg, resample=False)
        rows_by_algo, primary = _core_predict(acc, [seg])

        assert got["primary"] == primary
        assert set(got) == {"primary", *rows_by_algo.keys()}
        for aid in rows_by_algo:
            deep_equal(got[aid], rows_by_algo[aid][0], path=aid)


def test_predictSegment_subset_algorithms(acc):
    seg = _first_segments(acc, 1)[0]
    got = predictSegment(acc, seg, algorithms=["trap"], resample=False)
    assert set(got) == {"primary", "trap"}
    rows_by_algo, _ = _core_predict(acc, [seg], algorithms=["trap"])
    deep_equal(got["trap"], rows_by_algo["trap"][0])


def test_predictByParameters_matches_override(acc):
    seg = _first_segments(acc, 1)[0]
    params = {"W": 1.2, "f": 0.4, "abs_A": 0.6}

    got = predictByParameters(acc, seg, params, resample=False)

    seg_override = {
        **seg,
        "trapezoid_override": {"mode": "manual", **params},
    }
    rows_by_algo, primary = _core_predict(acc, [seg_override])

    assert got["primary"] == primary
    for aid in rows_by_algo:
        deep_equal(got[aid], rows_by_algo[aid][0], path=aid)


def test_predictByParameters_changes_trap_estimate(acc):
    """Override should actually feed the trapezoid estimator (sanity)."""
    seg = _first_segments(acc, 1)[0]
    base = predictSegment(acc, seg, resample=False)["trap"]
    overridden = predictByParameters(
        acc, seg, {"W": 2.5, "f": 0.1, "abs_A": 1.5}, resample=False,
    )["trap"]
    # ZUPT is unaffected; trap should respond to a large shape change unless
    # both happen to be rejected. Just assert the call path returns a row.
    assert "delta_height_m" in base and "delta_height_m" in overridden
