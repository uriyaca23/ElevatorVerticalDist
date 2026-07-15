"""pyramidElevatorDist — estimate elevator vertical travel from phone sensors.

Public API (the only supported entry points)::

    from pyramidElevatorDist.segmentor  import findSegments, findSegmentParameters
    from pyramidElevatorDist.prediction import predictSegment, predictByParameters
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
# `.prediction` MUST be imported before `.segmentor`/`.signal`: those two pull
# in `._orchestration`, which imports `pyramidElevatorDist.prediction.algorithms`
# and would re-enter a half-initialized `prediction/__init__.py` otherwise.
from .prediction import predictSegment, predictByParameters
from .segmentor import (
    findSegments, findSegmentParameters, findSegmentsDetailed,
)
from .signal import (
    reconstructedSignal, displaySeries, barometricAltitude, RECONSTRUCT_CHOICES,
)
from .types import (
    ALGORITHM_CHOICES,
    CalibrationSample,
    CorrelationCurves,
    DetailedRideSegment,
    DisplaySeries,
    Heatmaps,
    LobeFit,
    PredictionOutput,
    PredictionResult,
    PredictionRow,
    RideSegment,
    SegmentDetail,
    SegmentSpec,
    TrapezoidOverride,
    TrapezoidParams,
)
from .schemas import (
    ACC_SCHEMA,
    ALTITUDE_SCHEMA,
    FrameSchema,
    GYRO_SCHEMA,
    PRS_SCHEMA,
    RECONSTRUCTED_SIGNAL_SCHEMA,
    SEGMENTS_TABLE_SCHEMA,
)
from .exceptions import (
    BadDtypeError,
    CalibrationFileError,
    ConfigurationError,
    EmptyInputError,
    FrameValidationError,
    InputTypeError,
    InternalContractError,
    InvalidSegmentError,
    InvalidTrapezoidParamsError,
    MissingColumnsError,
    NaNValuesError,
    NonMonotonicTimestampsError,
    PyramidElevatorDistError,
    UnknownAlgorithmError,
    UnknownReconstructError,
)

__version__ = "0.2.0"

__all__ = [
    # entry points
    "findSegments",
    "findSegmentParameters",
    "findSegmentsDetailed",
    "predictSegment",
    "predictByParameters",
    "reconstructedSignal",
    "displaySeries",
    "barometricAltitude",
    # choice constants
    "RECONSTRUCT_CHOICES",
    "ALGORITHM_CHOICES",
    # typed models
    "LobeFit",
    "Heatmaps",
    "CorrelationCurves",
    "SegmentDetail",
    "RideSegment",
    "DetailedRideSegment",
    "SegmentSpec",
    "TrapezoidParams",
    "TrapezoidOverride",
    "PredictionRow",
    "PredictionResult",
    "DisplaySeries",
    "PredictionOutput",
    "CalibrationSample",
    # frame schemas
    "FrameSchema",
    "ACC_SCHEMA",
    "GYRO_SCHEMA",
    "PRS_SCHEMA",
    "RECONSTRUCTED_SIGNAL_SCHEMA",
    "ALTITUDE_SCHEMA",
    "SEGMENTS_TABLE_SCHEMA",
    # exceptions
    "PyramidElevatorDistError",
    "InputTypeError",
    "FrameValidationError",
    "MissingColumnsError",
    "EmptyInputError",
    "BadDtypeError",
    "NaNValuesError",
    "NonMonotonicTimestampsError",
    "InvalidSegmentError",
    "InvalidTrapezoidParamsError",
    "UnknownAlgorithmError",
    "UnknownReconstructError",
    "CalibrationFileError",
    "ConfigurationError",
    "InternalContractError",
]
