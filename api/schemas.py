"""Pydantic schemas for the API request bodies.

Responses are returned as already-serialised JSON (via
:mod:`api.encoding`) so they are not declared here — the response shape is
documented in the route docstrings.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class AccPayload(BaseModel):
    """Raw accelerometer samples, one column per array.

    All arrays must be the same length. ``timestamp_ms`` is the wall-clock
    timestamp in milliseconds (any monotonic origin works — the detector
    re-bases to ``t0_ms = timestamp_ms[0]``). ``x`` / ``y`` / ``z`` are the
    three accelerometer axes in m/s².
    """
    timestamp_ms: List[float]
    x: List[float]
    y: List[float]
    z: List[float]


class SegmentRequest(BaseModel):
    acc: AccPayload
    phone_model: str = Field(
        default="",
        description=(
            "Optional phone model string. When set, the detector's "
            "amplitude floors are tightened to the chip's noise σ "
            "(see src.utils.sensor_noise)."
        ),
    )
    include_state: bool = Field(
        default=True,
        description=(
            "Set to false to skip the heavy detector-state arrays in the "
            "response — useful for backend callers that only want the "
            "predictions list."
        ),
    )


class SegmentRow(BaseModel):
    """One ride interval as the prediction stage consumes it."""
    type: Literal["up", "down"]
    start_s: float
    end_s: float


class PredictRequest(BaseModel):
    acc: AccPayload
    segments: List[SegmentRow]
    phone_model: str = ""
    algorithms: Optional[List[Literal["trap", "zupt"]]] = Field(
        default=None,
        description=(
            "Subset of accelerometer-only algorithms to run. ``None`` "
            "(default) runs every algorithm exposed by the API."
        ),
    )
