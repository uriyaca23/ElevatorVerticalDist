"""Shared pydantic base + numpy-array field types for the public models."""
from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer


class PyramidModel(BaseModel):
    """Base for every public result/spec model.

    * ``arbitrary_types_allowed`` — numpy arrays are first-class fields.
    * ``validate_assignment`` — mutating a field re-validates it.
    * ``extra="forbid"`` — unknown keys fail loudly (catches typos like
      ``"star_s"`` that a plain dict would swallow).
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )


def _as_1d_float64(v: object) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D float array, got shape {arr.shape}")
    return arr


def _as_2d_float64(v: object) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D float array, got shape {arr.shape}")
    return arr


def _array_to_list(v: np.ndarray) -> list:
    return v.tolist()


#: 1-D float64 ndarray field — coerced via ``np.asarray``; serialized to a
#: plain list only in ``model_dump(mode="json")``.
FloatArray1D = Annotated[
    np.ndarray,
    BeforeValidator(_as_1d_float64),
    PlainSerializer(_array_to_list, when_used="json"),
]

#: 2-D float64 ndarray field.
FloatArray2D = Annotated[
    np.ndarray,
    BeforeValidator(_as_2d_float64),
    PlainSerializer(_array_to_list, when_used="json"),
]
