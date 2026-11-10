"""Shared input-validation helpers for the public wrapper modules.

These run at every wrapper entry point, before any algorithm code. They
only *reject malformed input* — every behavior of the algorithms themselves
(empty results, ``None`` on no-fit, reject rows, ...) is unchanged.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd
from pydantic import ValidationError

from .exceptions import (
    InputTypeError,
    InvalidSegmentError,
    InvalidTrapezoidParamsError,
    UnknownAlgorithmError,
    UnknownReconstructError,
)
from .physics.reconstruct_az import RECONSTRUCT_CHOICES
from .schemas import ACC_SCHEMA, GYRO_SCHEMA, PRS_SCHEMA
from .types.prediction import ALGORITHM_CHOICES, SegmentSpec, TrapezoidParams


def validate_acc(acc: object, *, func: str) -> pd.DataFrame:
    return ACC_SCHEMA.validate(acc, func=func, param="acc")


def validate_gyro(gyro: object, *, func: str) -> pd.DataFrame | None:
    """Validate an optional gyro stream.

    ``None`` — and, deliberately, an *empty* frame — mean "no gyroscope":
    the reconstruction stages treat a missing/empty gyro as a no-op and the
    UIs pass ``sensors.get("GYR")`` which may be empty. Anything non-empty
    must satisfy the gyro schema.
    """
    if gyro is None:
        return None
    if isinstance(gyro, pd.DataFrame) and len(gyro) == 0:
        return None
    return GYRO_SCHEMA.validate(gyro, func=func, param="gyro")


def validate_prs(prs: object, *, func: str,
                 allow_empty: bool = False) -> pd.DataFrame | None:
    """Validate an optional pressure stream.

    With ``allow_empty=True`` (the ``predict*`` paths) an empty frame is
    passed through so the core keeps producing its documented
    ``"no_pressure"`` baro rows.
    """
    if prs is None:
        return None
    if (allow_empty and isinstance(prs, pd.DataFrame) and len(prs) == 0):
        return prs
    return PRS_SCHEMA.validate(prs, func=func, param="prs")


def check_reconstruct(reconstruct: object, *, func: str) -> str:
    """Validate a ``reconstruct`` method name against RECONSTRUCT_CHOICES.

    Always validated — even without a gyro — so a typo can never silently
    degrade to the no-reconstruction path.
    """
    if not isinstance(reconstruct, str):
        raise InputTypeError(
            f"{func}: parameter 'reconstruct' must be a str, got "
            f"{type(reconstruct).__name__}."
        )
    if reconstruct not in RECONSTRUCT_CHOICES:
        raise UnknownReconstructError(
            f"{func}: unknown reconstruct method {reconstruct!r}; choose "
            f"one of {tuple(RECONSTRUCT_CHOICES)!r} (names are "
            f"case-sensitive)."
        )
    return reconstruct


def check_algorithms(algorithms: object, *, func: str) -> list[str] | None:
    """Validate the optional accelerometer-algorithm subset."""
    if algorithms is None:
        return None
    if isinstance(algorithms, str) or not isinstance(algorithms, Iterable):
        raise InputTypeError(
            f"{func}: parameter 'algorithms' must be a list of algorithm "
            f"ids or None, got {type(algorithms).__name__}."
        )
    out = list(algorithms)
    for aid in out:
        if aid not in ALGORITHM_CHOICES:
            raise UnknownAlgorithmError(
                f"{func}: unknown algorithm id {aid!r}; choose from "
                f"{ALGORITHM_CHOICES!r}."
            )
    return out


def coerce_segment(segment: object, *, func: str) -> SegmentSpec:
    """Coerce a SegmentSpec-or-mapping into a validated SegmentSpec."""
    if isinstance(segment, SegmentSpec):
        return segment
    if not isinstance(segment, Mapping):
        raise InputTypeError(
            f"{func}: parameter 'segment' must be a SegmentSpec or a "
            f"mapping like {{'type': 'up', 'start_s': 12.0, "
            f"'end_s': 20.5}}, got {type(segment).__name__}."
        )
    try:
        return SegmentSpec.model_validate(dict(segment))
    except ValidationError as exc:
        raise InvalidSegmentError(
            f"{func}: invalid 'segment' — {_summarize(exc)}. Expected "
            f"{{'type': 'up'|'down', 'start_s': float, 'end_s': float}} "
            f"with finite end_s > start_s."
        ) from exc


def coerce_trapezoid(params: object, *, func: str) -> TrapezoidParams:
    """Coerce a TrapezoidParams-or-mapping into validated params."""
    if isinstance(params, TrapezoidParams):
        return params
    if not isinstance(params, Mapping):
        raise InputTypeError(
            f"{func}: parameter 'trapezoid_params' must be a "
            f"TrapezoidParams or a mapping like {{'W': 1.2, 'f': 0.5, "
            f"'abs_A': 0.8}}, got {type(params).__name__}."
        )
    try:
        return TrapezoidParams.model_validate(dict(params))
    except ValidationError as exc:
        raise InvalidTrapezoidParamsError(
            f"{func}: invalid 'trapezoid_params' — {_summarize(exc)}. "
            f"Expected {{'W': float > 0, 'f': float in [0, 1], "
            f"'abs_A': float >= 0}}."
        ) from exc


def _summarize(exc: ValidationError) -> str:
    """One-line human summary of a pydantic ValidationError."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "value"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def segment_to_core_dict(spec: SegmentSpec) -> dict[str, Any]:
    """The exact dict shape ``_orchestration.predict`` consumes."""
    seg: dict[str, Any] = {
        "type": spec.type,
        "start_s": spec.start_s,
        "end_s": spec.end_s,
    }
    if spec.trapezoid_override is not None:
        seg["trapezoid_override"] = spec.trapezoid_override.model_dump()
    return seg
