"""Public surface sanity: every export resolves, versions and choices pin."""
from __future__ import annotations

import pyramidElevatorDist as P


def test_all_exports_resolve():
    for name in P.__all__:
        assert getattr(P, name) is not None, name
    assert P.__version__ == "0.2.0"


def test_reconstruct_choices():
    assert P.RECONSTRUCT_CHOICES[0] == "none"
    assert all(isinstance(c, str) for c in P.RECONSTRUCT_CHOICES)


def test_algorithm_choices():
    assert P.ALGORITHM_CHOICES == ("trap", "zupt")


def test_deprecated_predection_alias():
    from pyramidElevatorDist import predection, prediction

    assert predection.predictSegment is prediction.predictSegment
    assert predection.predictByParameters is prediction.predictByParameters
