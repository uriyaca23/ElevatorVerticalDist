"""The package exposes exactly its public functions, choice constants,
typed models, frame schemas, and exceptions — the only surface the UIs are
allowed to call."""
from __future__ import annotations

import pyramidElevatorDist as pkg
from pyramidElevatorDist import segmentor, prediction, signal


_PUBLIC_FUNCS = {
    "findSegments", "findSegmentParameters", "findSegmentsDetailed",
    "predictSegment", "predictByParameters",
    "reconstructedSignal", "displaySeries", "barometricAltitude",
}

_CHOICE_CONSTANTS = {"RECONSTRUCT_CHOICES", "ALGORITHM_CHOICES"}

_MODELS = {
    "LobeFit", "Heatmaps", "CorrelationCurves", "SegmentDetail",
    "RideSegment", "DetailedRideSegment",
    "DetailedSegmentsResult", "SegmentParametersResult",
    "SegmentSpec", "TrapezoidParams", "TrapezoidOverride",
    "PredictionRow", "PredictionResult", "DisplaySeries",
    "PredictionOutput", "CalibrationSample",
}

_SCHEMAS = {
    "FrameSchema", "ACC_SCHEMA", "GYRO_SCHEMA", "PRS_SCHEMA",
    "RECONSTRUCTED_SIGNAL_SCHEMA", "ALTITUDE_SCHEMA", "SEGMENTS_TABLE_SCHEMA",
}

_EXCEPTIONS = {
    "PyramidElevatorDistError", "InputTypeError", "FrameValidationError",
    "MissingColumnsError", "EmptyInputError", "BadDtypeError",
    "NaNValuesError", "NonMonotonicTimestampsError",
    "InvalidSegmentError", "InvalidTrapezoidParamsError",
    "UnknownAlgorithmError", "UnknownReconstructError",
    "CalibrationFileError", "ConfigurationError", "InternalContractError",
}


def test_top_level_exports():
    assert set(pkg.__all__) == (
        _PUBLIC_FUNCS | _CHOICE_CONSTANTS | _MODELS | _SCHEMAS | _EXCEPTIONS
    )
    for name in _PUBLIC_FUNCS:
        assert callable(getattr(pkg, name))
    for name in pkg.__all__:
        assert getattr(pkg, name) is not None
    assert isinstance(pkg.RECONSTRUCT_CHOICES, list)
    assert pkg.RECONSTRUCT_CHOICES[0] == "none"
    assert pkg.ALGORITHM_CHOICES == ("trap", "zupt")
    assert pkg.__version__ == "0.2.0"


def test_submodule_exports():
    assert segmentor.__all__ == [
        "findSegments", "findSegmentParameters", "findSegmentsDetailed",
    ]
    assert prediction.__all__ == ["predictSegment", "predictByParameters"]
    assert signal.__all__ == [
        "reconstructedSignal", "barometricAltitude", "displaySeries",
        "RECONSTRUCT_CHOICES",
    ]


def test_exception_hierarchy():
    for name in _EXCEPTIONS:
        exc = getattr(pkg, name)
        assert issubclass(exc, pkg.PyramidElevatorDistError)
    assert issubclass(pkg.InputTypeError, TypeError)
    assert issubclass(pkg.FrameValidationError, ValueError)
    assert issubclass(pkg.InvalidSegmentError, ValueError)
    assert issubclass(pkg.InvalidTrapezoidParamsError, pkg.InvalidSegmentError)


def test_documented_import_paths():
    from pyramidElevatorDist.segmentor import (  # noqa: F401
        findSegments, findSegmentParameters, findSegmentsDetailed,
    )
    from pyramidElevatorDist.prediction import (  # noqa: F401
        predictSegment, predictByParameters,
    )
    from pyramidElevatorDist.signal import (  # noqa: F401
        reconstructedSignal, displaySeries, barometricAltitude,
        RECONSTRUCT_CHOICES,
    )


def test_deprecated_predection_alias():
    from pyramidElevatorDist import predection

    assert predection.__all__ == ["predictSegment", "predictByParameters"]
    assert predection.predictSegment is prediction.predictSegment
    assert predection.predictByParameters is prediction.predictByParameters
