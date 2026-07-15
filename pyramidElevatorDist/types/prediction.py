"""Typed inputs/results of the public prediction API.

``PredictionRow`` mirrors the orchestration core's row dict one-to-one
(see ``_orchestration.predict``). ``meta`` deliberately stays a plain dict:
its keys differ per algorithm *and per fit mode* (pair / joined / override /
zupt-fallback), and consumers read it defensively via ``.get`` — typing it
would couple the public API to estimator internals.
"""
from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple

import numpy as np
from pydantic import model_validator

from ..exceptions import UnknownAlgorithmError
from .common import PyramidModel

__all__ = [
    "ALGORITHM_CHOICES",
    "SegmentSpec",
    "TrapezoidParams",
    "TrapezoidOverride",
    "PredictionRow",
    "PredictionResult",
    "DisplaySeries",
]

#: Accelerometer algorithm ids accepted by ``predictSegment(algorithms=...)``.
ALGORITHM_CHOICES: tuple[str, ...] = ("trap", "zupt")


class TrapezoidParams(PyramidModel):
    """Manually edited trapezoid pulse shape."""

    W: float          #: half-width, seconds (> 0)
    f: float          #: plateau fraction in [0, 1]
    abs_A: float      #: absolute peak amplitude, m/s² (>= 0)

    @model_validator(mode="after")
    def _finite_and_in_range(self) -> "TrapezoidParams":
        for name in ("W", "f", "abs_A"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v!r}")
        if self.W <= 0:
            raise ValueError(f"W must be > 0 (seconds), got {self.W!r}")
        if not 0.0 <= self.f <= 1.0:
            raise ValueError(f"f must be in [0, 1], got {self.f!r}")
        if self.abs_A < 0:
            raise ValueError(f"abs_A must be >= 0 (m/s²), got {self.abs_A!r}")
        return self


class TrapezoidOverride(TrapezoidParams):
    """A :class:`TrapezoidParams` tagged with its origin (always manual)."""

    mode: Literal["manual"] = "manual"


class SegmentSpec(PyramidModel):
    """One ride interval to predict, on the trace's own time axis."""

    type: Literal["up", "down"]
    start_s: float
    end_s: float
    trapezoid_override: TrapezoidOverride | None = None

    @model_validator(mode="after")
    def _bounds_valid(self) -> "SegmentSpec":
        if not (math.isfinite(self.start_s) and math.isfinite(self.end_s)):
            raise ValueError(
                f"start_s/end_s must be finite, got start_s={self.start_s!r}, "
                f"end_s={self.end_s!r}"
            )
        if self.end_s <= self.start_s:
            raise ValueError(
                f"end_s must be > start_s, got start_s={self.start_s!r}, "
                f"end_s={self.end_s!r}"
            )
        return self


class PredictionRow(PyramidModel):
    """One algorithm's Δh prediction for one segment."""

    segment: int               #: position of the segment in the request
    type: Literal["up", "down"]
    start_s: float
    end_s: float
    duration_s: float
    delta_height_m: float      #: signed Δh (up +, down -); NaN when rejected
    abs_height_m: float        #: |Δh|; NaN when rejected
    accepted: bool             #: passed the quality filter
    quality_score: float       #: 0 = excellent, higher = worse; NaN possible
    reject_reason: str         #: empty when accepted
    ci_half_width: float       #: 90 % CI half-width, m; NaN when rejected
    meta: dict[str, Any]       #: algorithm-specific extras (see module doc)


class PredictionResult(PyramidModel):
    """The per-algorithm rows for one segment.

    ``baro`` is present only when a pressure frame was supplied to
    ``predictSegment`` — it is the barometer (ground-truth) Δh, sign as
    measured.
    """

    primary: str
    trap: PredictionRow | None = None
    zupt: PredictionRow | None = None
    baro: PredictionRow | None = None

    def row(self, algo_id: str) -> PredictionRow | None:
        """Typed accessor: the row for ``algo_id`` (``None`` if not run)."""
        if algo_id not in ("trap", "zupt", "baro"):
            raise UnknownAlgorithmError(
                f"unknown algorithm id {algo_id!r}; choose from "
                f"{('trap', 'zupt', 'baro')}"
            )
        return getattr(self, algo_id)


class DisplaySeries(NamedTuple):
    """Return of ``displaySeries`` — unpacks as the historical 3-tuple."""

    values: np.ndarray
    label: str
    reconstructed: bool
