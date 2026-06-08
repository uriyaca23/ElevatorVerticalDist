"""The package must expose exactly the four public functions."""
from __future__ import annotations

import pyramidElevatorDist as pkg
from pyramidElevatorDist import segmentor, predection


def test_top_level_exports():
    assert set(pkg.__all__) == {
        "findSegments", "findSegmentParameters",
        "predictSegment", "predictByParameters",
    }
    for name in pkg.__all__:
        assert callable(getattr(pkg, name))


def test_submodule_exports():
    assert segmentor.__all__ == ["findSegments", "findSegmentParameters"]
    assert predection.__all__ == ["predictSegment", "predictByParameters"]


def test_documented_import_paths():
    from pyramidElevatorDist.segmentor import (  # noqa: F401
        findSegments, findSegmentParameters,
    )
    from pyramidElevatorDist.predection import (  # noqa: F401
        predictSegment, predictByParameters,
    )
