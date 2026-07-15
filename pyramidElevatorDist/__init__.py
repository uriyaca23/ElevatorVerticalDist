"""pyramidElevatorDist — estimate elevator vertical travel from phone sensors.

Public API (the only supported entry points)::

    from pyramidElevatorDist.segmentor  import findSegments, findSegmentParameters
    from pyramidElevatorDist.predection import predictSegment, predictByParameters
    from pyramidElevatorDist.signal     import reconstructedSignal

* ``findSegments(acc)``                          — detect all ride segments.
* ``findSegmentParameters(acc, start_s, end_s)`` — fit the trapezoid + heatmaps
                                                   for one marked interval.
* ``predictSegment(acc, segment)``               — predict Δh for one segment.
* ``predictByParameters(acc, segment, params)``  — predict Δh with a manually
                                                   edited trapezoid.
* ``reconstructedSignal(acc, gyro, reconstruct)``— whole-trace vertical accel
                                                   for display (optionally gyro
                                                   orientation-reconstructed).
* ``barometricAltitude(prs)``                    — whole-trace barometric
                                                   altitude (ground-truth).

All take a pandas ``DataFrame`` with columns ``timestamp_ms``, ``x``, ``y``,
``z`` (raw accelerometer). Configuration is loaded internally.

Incoming data may be at any (even variable) sample rate: by default every
entry point first normalizes the trace onto a uniform 50 Hz grid — the cadence
the detector and Δh estimators are tuned for — via a gap-aware, time-correct
resample. Pass ``resample=False`` if your data is already at that cadence.
"""
from .segmentor import (
    findSegments, findSegmentParameters, findSegmentsDetailed,
)
from .predection import predictSegment, predictByParameters
from .signal import (
    reconstructedSignal, displaySeries, barometricAltitude, RECONSTRUCT_CHOICES,
)

__version__ = "0.1.0"

__all__ = [
    "findSegments",
    "findSegmentParameters",
    "findSegmentsDetailed",
    "predictSegment",
    "predictByParameters",
    "reconstructedSignal",
    "displaySeries",
    "barometricAltitude",
    "RECONSTRUCT_CHOICES",
]
