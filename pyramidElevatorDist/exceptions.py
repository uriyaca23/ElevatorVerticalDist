"""Exception taxonomy for the public pyramidElevatorDist API.

Every exception the package raises deliberately derives from
:class:`PyramidElevatorDistError`, so callers can catch the whole family with
one ``except``. Input-shape errors additionally derive from the matching
builtin (``ValueError`` / ``TypeError``) so legacy ``except ValueError``
handlers keep working.

Messages are written to be actionable: they name the entry point, the
offending parameter, what exactly is wrong, and how to fix it.
"""
from __future__ import annotations


class PyramidElevatorDistError(Exception):
    """Base class for every exception raised by pyramidElevatorDist."""


class InputTypeError(PyramidElevatorDistError, TypeError):
    """A parameter has the wrong Python type (e.g. a list where a
    ``pandas.DataFrame`` is required)."""


class FrameValidationError(PyramidElevatorDistError, ValueError):
    """Base class for DataFrame schema violations.

    Attributes ``func`` / ``param`` / ``schema`` identify the entry point,
    the parameter, and the schema that rejected the frame.
    """

    def __init__(self, message: str, *, func: str = "", param: str = "",
                 schema: str = "") -> None:
        super().__init__(message)
        self.func = func
        self.param = param
        self.schema = schema


class MissingColumnsError(FrameValidationError):
    """The frame lacks one or more required columns."""

    def __init__(self, message: str, *, missing: frozenset[str] = frozenset(),
                 expected: tuple[str, ...] = (), **kw) -> None:
        super().__init__(message, **kw)
        self.missing = missing
        self.expected = expected


class EmptyInputError(FrameValidationError):
    """The frame has zero rows where at least one sample is required."""


class BadDtypeError(FrameValidationError):
    """A required column is not numeric."""

    def __init__(self, message: str, *, column: str = "", dtype: str = "",
                 **kw) -> None:
        super().__init__(message, **kw)
        self.column = column
        self.dtype = dtype


class NaNValuesError(FrameValidationError):
    """A required column contains NaN values."""

    def __init__(self, message: str, *, column: str = "", count: int = 0,
                 first_index: int = -1, **kw) -> None:
        super().__init__(message, **kw)
        self.column = column
        self.count = count
        self.first_index = first_index


class NonMonotonicTimestampsError(FrameValidationError):
    """The timestamp column is not sorted in non-decreasing order."""

    def __init__(self, message: str, *, first_index: int = -1, **kw) -> None:
        super().__init__(message, **kw)
        self.first_index = first_index


class InvalidSegmentError(PyramidElevatorDistError, ValueError):
    """A segment specification is malformed (bad keys, non-finite bounds,
    ``end_s <= start_s``, unknown ride type, ...)."""


class InvalidTrapezoidParamsError(InvalidSegmentError):
    """A manual trapezoid override is malformed (``W <= 0``, ``f`` outside
    ``[0, 1]``, negative ``abs_A``, non-finite values, unknown keys)."""


class UnknownAlgorithmError(PyramidElevatorDistError, ValueError):
    """An algorithm id is not one of the supported choices."""


class UnknownReconstructError(PyramidElevatorDistError, ValueError):
    """A ``reconstruct`` method name is not one of ``RECONSTRUCT_CHOICES``."""


class CalibrationFileError(PyramidElevatorDistError, RuntimeError):
    """A conformal ``calibration.json`` exists but could not be read/parsed.

    A *missing* calibration file is not an error — the estimators fall back
    to their uncalibrated theoretical CI.
    """


class ConfigurationError(PyramidElevatorDistError, RuntimeError):
    """A packaged ``config.json`` is missing or corrupt. This indicates a
    broken installation — re-install the package."""


class InternalContractError(PyramidElevatorDistError, RuntimeError):
    """The package produced output violating its own schema. This is a bug
    in pyramidElevatorDist — please report it."""
