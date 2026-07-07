"""Deprecated shim — the boutique orchestration now lives in one place,
``src.pipelines.inprocess`` (the shared core behind every façade and UI).

This module used to hold a ~300-line copy of the "resample → segment → slice
ride/pre/post → predict-per-algorithm" logic. That duplication is gone; the
façade modules (:mod:`pyramidElevatorDist.segmentor`,
:mod:`pyramidElevatorDist.predection`) call the core directly. Only the names
still imported by the test-suite are re-exported here for backwards
compatibility.
"""
from __future__ import annotations

from src.pipelines.inprocess import (  # noqa: F401
    RESAMPLE_TARGET_HZ,
    find_matching_prediction,
    resample_acc as _resample_acc,
    segment_cfg as _segment_cfg,
)

__all__ = [
    "RESAMPLE_TARGET_HZ",
    "_resample_acc",
    "_segment_cfg",
    "find_matching_prediction",
]
