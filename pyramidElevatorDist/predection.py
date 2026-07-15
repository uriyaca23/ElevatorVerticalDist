"""Deprecated alias (historical misspelling). Use
:mod:`pyramidElevatorDist.prediction` instead."""
from pyramidElevatorDist.prediction import predictByParameters, predictSegment  # noqa: F401

__all__ = ["predictSegment", "predictByParameters"]
