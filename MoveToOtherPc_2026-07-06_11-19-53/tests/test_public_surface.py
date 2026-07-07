"""The package exposes exactly its five public functions + the reconstruct
method list — the only surface the UIs are allowed to call."""
from __future__ import annotations

import pyramidElevatorDist as pkg
from pyramidElevatorDist import segmentor, predection, signal


_PUBLIC_FUNCS = {
    "findSegments", "findSegmentParameters", "findSegmentsDetailed",
    "predictSegment", "predictByParameters",
    "reconstructedSignal", "barometricAltitude",
}


def test_top_level_exports():
    assert set(pkg.__all__) == _PUBLIC_FUNCS | {"RECONSTRUCT_CHOICES"}
    for name in _PUBLIC_FUNCS:
        assert callable(getattr(pkg, name))
    assert isinstance(pkg.RECONSTRUCT_CHOICES, list)
    assert pkg.RECONSTRUCT_CHOICES[0] == "none"


def test_submodule_exports():
    assert segmentor.__all__ == [
        "findSegments", "findSegmentParameters", "findSegmentsDetailed",
    ]
    assert predection.__all__ == ["predictSegment", "predictByParameters"]
    assert signal.__all__ == [
        "reconstructedSignal", "barometricAltitude", "RECONSTRUCT_CHOICES",
    ]


def test_documented_import_paths():
    from pyramidElevatorDist.segmentor import (  # noqa: F401
        findSegments, findSegmentParameters, findSegmentsDetailed,
    )
    from pyramidElevatorDist.predection import (  # noqa: F401
        predictSegment, predictByParameters,
    )
    from pyramidElevatorDist.signal import (  # noqa: F401
        reconstructedSignal, barometricAltitude, RECONSTRUCT_CHOICES,
    )
