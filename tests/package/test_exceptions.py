"""Exception hierarchy + message actionability."""
from __future__ import annotations

import pytest

import pyramidElevatorDist as P

from .conftest import make_ride_acc

_ALL_EXCEPTIONS = [
    P.InputTypeError, P.FrameValidationError, P.MissingColumnsError,
    P.EmptyInputError, P.BadDtypeError, P.NaNValuesError,
    P.NonMonotonicTimestampsError, P.InvalidSegmentError,
    P.InvalidTrapezoidParamsError, P.UnknownAlgorithmError,
    P.UnknownReconstructError, P.CalibrationFileError,
    P.ConfigurationError, P.InternalContractError,
]


def test_hierarchy():
    for exc in _ALL_EXCEPTIONS:
        assert issubclass(exc, P.PyramidElevatorDistError)
    # Frame errors keep working with legacy `except ValueError` handlers.
    for exc in (P.FrameValidationError, P.MissingColumnsError,
                P.EmptyInputError, P.BadDtypeError, P.NaNValuesError,
                P.NonMonotonicTimestampsError, P.InvalidSegmentError,
                P.InvalidTrapezoidParamsError, P.UnknownAlgorithmError,
                P.UnknownReconstructError):
        assert issubclass(exc, ValueError)
    assert issubclass(P.InputTypeError, TypeError)


def test_messages_are_actionable():
    """Each triggered error names the entry point, the parameter, and a
    remediation."""
    acc = make_ride_acc()

    with pytest.raises(P.MissingColumnsError) as ei:
        P.findSegments(acc.drop(columns=["x"]))
    msg = str(ei.value)
    assert "findSegments" in msg and "'acc'" in msg and "expected" in msg

    with pytest.raises(P.InvalidSegmentError) as ei:
        P.predictSegment(acc, {"type": "up", "start_s": 5.0, "end_s": 1.0})
    msg = str(ei.value)
    assert "predictSegment" in msg and "'segment'" in msg
    assert "end_s > start_s" in msg

    with pytest.raises(P.UnknownReconstructError) as ei:
        P.reconstructedSignal(acc, reconstruct="frobnicate")
    msg = str(ei.value)
    assert "reconstructedSignal" in msg and "'frobnicate'" in msg
    assert "none" in msg  # lists the valid choices

    one_exception_family = P.PyramidElevatorDistError
    with pytest.raises(one_exception_family):
        P.findSegments("not a frame")
