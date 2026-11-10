"""DataFrame schemas for every frame the public API accepts or returns.

A :class:`FrameSchema` declares the required columns, their dtype
expectations, and cheap structural sanity checks (non-empty, monotonic
timestamps). Wrappers validate *inputs* against these schemas at entry —
failures raise the specific :mod:`pyramidElevatorDist.exceptions` subclass —
and validate *outputs* at exit, where a failure raises
:class:`~pyramidElevatorDist.exceptions.InternalContractError` (a package
bug, not a caller error).

Validation is two vectorized O(n) passes (NaN scan + timestamp diff) plus
O(#columns) checks — negligible next to the matched-filter pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .exceptions import (
    BadDtypeError,
    EmptyInputError,
    InputTypeError,
    InternalContractError,
    MissingColumnsError,
    NaNValuesError,
    NonMonotonicTimestampsError,
)

__all__ = [
    "ColumnSpec",
    "FrameSchema",
    "ACC_SCHEMA",
    "GYRO_SCHEMA",
    "PRS_SCHEMA",
    "RECONSTRUCTED_SIGNAL_SCHEMA",
    "ALTITUDE_SCHEMA",
    "SEGMENTS_TABLE_SCHEMA",
]


@dataclass(frozen=True)
class ColumnSpec:
    """One required column of a :class:`FrameSchema`.

    ``kind``: ``"numeric"`` (must satisfy ``is_numeric_dtype``), ``"any"``
    (present, no dtype check), or ``"pair"`` (cells are 2-tuples, e.g. the
    ``(lo, hi)`` CI columns of the segments table).
    """

    name: str
    kind: str = "numeric"
    allow_nan: bool = False


@dataclass(frozen=True)
class FrameSchema:
    """Declarative schema + validator for one DataFrame shape."""

    name: str
    columns: tuple[ColumnSpec, ...]
    allow_empty: bool = False
    monotonic_col: str | None = "timestamp_ms"
    example: str = field(default="", compare=False)

    def validate(self, df: object, *, func: str, param: str,
                 internal: bool = False) -> pd.DataFrame:
        """Validate ``df`` and return it unchanged.

        ``func``/``param`` name the entry point and parameter for error
        messages. With ``internal=True`` any violation is re-raised as
        :class:`InternalContractError` (used on wrapper *outputs*).
        """
        try:
            return self._validate(df, func=func, param=param)
        except (InputTypeError, MissingColumnsError, EmptyInputError,
                BadDtypeError, NaNValuesError,
                NonMonotonicTimestampsError) as exc:
            if internal:
                raise InternalContractError(
                    f"{func}: the {self.name} it produced violates its own "
                    f"schema ({exc}). This is a bug in pyramidElevatorDist — "
                    f"please report it."
                ) from exc
            raise

    # -- individual checks, in order ------------------------------------

    def _validate(self, df: object, *, func: str, param: str) -> pd.DataFrame:
        ctx = dict(func=func, param=param, schema=self.name)

        if not isinstance(df, pd.DataFrame):
            hint = f" Build one with {self.example}." if self.example else ""
            raise InputTypeError(
                f"{func}: parameter '{param}' must be a pandas.DataFrame "
                f"({self.name}), got {type(df).__name__}.{hint}"
            )

        expected = tuple(c.name for c in self.columns)
        missing = frozenset(expected) - set(df.columns)
        if missing:
            raise MissingColumnsError(
                f"{func}: the {self.name} passed as '{param}' is missing "
                f"columns {sorted(missing)}; expected columns: "
                f"{', '.join(expected)}; got: "
                f"{', '.join(map(str, df.columns)) or '(none)'}.",
                missing=missing, expected=expected, **ctx,
            )

        if len(df) == 0 and not self.allow_empty:
            raise EmptyInputError(
                f"{func}: the {self.name} passed as '{param}' has 0 rows — "
                f"at least one sample is required.",
                **ctx,
            )

        for col in self.columns:
            if len(df) == 0:
                # An empty frame carries no dtype information (pandas gives
                # fresh empty columns dtype=object) — nothing to check.
                break
            if col.kind == "numeric":
                if not pd.api.types.is_numeric_dtype(df[col.name]):
                    raise BadDtypeError(
                        f"{func}: column '{col.name}' of the {self.name} "
                        f"passed as '{param}' has dtype "
                        f"{df[col.name].dtype} but must be numeric. Fix "
                        f"with: df['{col.name}'] = "
                        f"pd.to_numeric(df['{col.name}']).",
                        column=col.name, dtype=str(df[col.name].dtype), **ctx,
                    )
                if not col.allow_nan and len(df):
                    isna = df[col.name].isna().to_numpy()
                    if isna.any():
                        first = int(np.argmax(isna))
                        count = int(isna.sum())
                        raise NaNValuesError(
                            f"{func}: column '{col.name}' of the {self.name} "
                            f"passed as '{param}' contains {count} NaN "
                            f"value(s) (first at row {first}). Fix with: "
                            f"df = df.dropna(subset=['{col.name}']).",
                            column=col.name, count=count, first_index=first,
                            **ctx,
                        )
            elif col.kind == "pair" and len(df):
                first_valid = df[col.name].dropna()
                if len(first_valid):
                    cell = first_valid.iloc[0]
                    if not (isinstance(cell, tuple) and len(cell) == 2):
                        raise BadDtypeError(
                            f"{func}: column '{col.name}' of the {self.name} "
                            f"must hold (lo, hi) 2-tuples, got "
                            f"{type(cell).__name__}.",
                            column=col.name, dtype=str(type(cell).__name__),
                            **ctx,
                        )

        if self.monotonic_col is not None and len(df) > 1:
            ts = df[self.monotonic_col].to_numpy(dtype=np.float64)
            diffs = np.diff(ts)
            bad = diffs < 0
            if bad.any():
                first = int(np.argmax(bad)) + 1
                raise NonMonotonicTimestampsError(
                    f"{func}: column '{self.monotonic_col}' of the "
                    f"{self.name} passed as '{param}' is not sorted "
                    f"(first out-of-order sample at row {first}). Fix with: "
                    f"df = df.sort_values('{self.monotonic_col}', "
                    f"kind='stable').reset_index(drop=True).",
                    first_index=first, **ctx,
                )
        return df


_XYZ_EXAMPLE = ("pd.DataFrame({'timestamp_ms': ..., 'x': ..., "
                "'y': ..., 'z': ...})")

#: Raw accelerometer input — epoch-ms timestamps + device-frame x/y/z (m/s²).
ACC_SCHEMA = FrameSchema(
    name="accelerometer frame",
    columns=(
        ColumnSpec("timestamp_ms"),
        ColumnSpec("x"), ColumnSpec("y"), ColumnSpec("z"),
    ),
    example=_XYZ_EXAMPLE,
)

#: Raw gyroscope input — same clock as ACC, x/y/z in rad/s.
GYRO_SCHEMA = FrameSchema(
    name="gyroscope frame",
    columns=(
        ColumnSpec("timestamp_ms"),
        ColumnSpec("x"), ColumnSpec("y"), ColumnSpec("z"),
    ),
    example=_XYZ_EXAMPLE,
)

#: Barometer input — epoch-ms timestamps + pressure in hPa.
PRS_SCHEMA = FrameSchema(
    name="pressure frame",
    columns=(
        ColumnSpec("timestamp_ms"),
        ColumnSpec("pressure", allow_nan=True),
    ),
    example="pd.DataFrame({'timestamp_ms': ..., 'pressure': ...})",
)

#: Output of :func:`pyramidElevatorDist.signal.reconstructedSignal`.
RECONSTRUCTED_SIGNAL_SCHEMA = FrameSchema(
    name="reconstructed-signal frame",
    columns=(
        ColumnSpec("timestamp_ms"),
        ColumnSpec("a_vert", allow_nan=True),
        ColumnSpec("a_mag_g", allow_nan=True),
    ),
    allow_empty=True,
)

#: Output of :func:`pyramidElevatorDist.signal.barometricAltitude`.
ALTITUDE_SCHEMA = FrameSchema(
    name="altitude frame",
    columns=(
        ColumnSpec("timestamp_ms"),
        ColumnSpec("altitude_m", allow_nan=True),
    ),
    allow_empty=True,
)

#: Output of ``Segmenter.detect`` — one row per detected ride; the ``*_ci``
#: columns hold ``(lo, hi)`` tuples that collapse to zero width for
#: deterministic algorithms.
SEGMENTS_TABLE_SCHEMA = FrameSchema(
    name="segments table",
    columns=(
        ColumnSpec("start_ci", kind="pair"),
        ColumnSpec("end_ci", kind="pair"),
        ColumnSpec("duration"),
        ColumnSpec("type", kind="any"),
        ColumnSpec("probability_ci", kind="pair"),
    ),
    allow_empty=True,
    monotonic_col=None,
)
