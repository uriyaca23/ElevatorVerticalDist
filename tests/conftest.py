"""Shared fixtures + a tolerant deep-equality helper.

The regression tests pin the public ``pyramidElevatorDist`` façade to the
shared in-process boutique core (``pyramidElevatorDist._orchestration``) on a real
experiment, so any drift in output is caught.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# A stable training experiment that produces many ride segments.
_FIXTURE_EXPERIMENT = "RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026"


@pytest.fixture(scope="session")
def acc():
    """Raw accelerometer DataFrame for a real experiment."""
    try:
        from src.data.loader import getExperimentData, list_experiments
    except Exception as e:  # pragma: no cover
        pytest.skip(f"data loader unavailable: {e}")

    name = _FIXTURE_EXPERIMENT
    try:
        available = list_experiments(kind="train")
    except Exception:  # pragma: no cover
        available = []
    if name not in available:
        if not available:
            pytest.skip("no training experiments available for fixtures")
        name = available[0]

    sensors, _gt, _meta = getExperimentData(name)
    if "ACC" not in sensors or sensors["ACC"].empty:
        pytest.skip(f"experiment {name!r} has no ACC data")
    return sensors["ACC"]


def deep_equal(a, b, *, rtol=1e-9, atol=1e-9, path="") -> None:
    """Assert ``a`` and ``b`` are structurally equal.

    Handles nested dict/list/tuple, numpy arrays, and nan-aware float
    comparison with tolerance. Raises AssertionError with a path on mismatch.
    """
    # numpy arrays
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        assert aa.shape == bb.shape, f"{path}: shape {aa.shape} != {bb.shape}"
        assert np.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=True), (
            f"{path}: arrays differ (max abs diff "
            f"{np.nanmax(np.abs(aa - bb)) if aa.size else 0})"
        )
        return

    if isinstance(a, dict):
        assert isinstance(b, dict), f"{path}: type {type(a)} != {type(b)}"
        assert set(a.keys()) == set(b.keys()), (
            f"{path}: keys {sorted(a.keys())} != {sorted(b.keys())}"
        )
        for k in a:
            deep_equal(a[k], b[k], rtol=rtol, atol=atol, path=f"{path}.{k}")
        return

    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)), f"{path}: {type(a)} != {type(b)}"
        assert len(a) == len(b), f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            deep_equal(x, y, rtol=rtol, atol=atol, path=f"{path}[{i}]")
        return

    if isinstance(a, bool) or isinstance(b, bool):
        assert a == b, f"{path}: {a!r} != {b!r}"
        return

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        af, bf = float(a), float(b)
        if math.isnan(af) and math.isnan(bf):
            return
        assert math.isclose(af, bf, rel_tol=rtol, abs_tol=atol), (
            f"{path}: {af!r} != {bf!r}"
        )
        return

    assert a == b, f"{path}: {a!r} != {b!r}"
