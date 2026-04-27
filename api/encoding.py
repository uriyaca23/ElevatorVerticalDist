"""JSON-safety helpers for the API responses.

The detector's ``state`` dict and the predictor outputs hold numpy arrays,
numpy scalars, Pydantic configs, and ``NaN`` floats. ``json.dumps`` chokes
on the first three and silently emits the bare token ``NaN`` for the last
(which JS / strict parsers reject). :func:`jsonify` walks any structure and
produces something ``json.dumps`` can encode losslessly: arrays become
lists, numpy scalars become Python scalars, Pydantic models become dicts,
non-finite floats become ``None``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def _scalar(x: Any) -> Any:
    if isinstance(x, (np.floating,)):
        x = float(x)
    elif isinstance(x, (np.integer,)):
        return int(x)
    elif isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    return x


def jsonify(obj: Any) -> Any:
    """Recursively convert ``obj`` into a JSON-encodable Python tree."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, float, np.floating, np.integer, np.bool_)):
        return _scalar(obj)
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonify(v) for v in obj]
    # Pydantic v2 model
    if hasattr(obj, "model_dump"):
        try:
            return jsonify(obj.model_dump())
        except Exception:
            pass
    # plain dataclass (e.g. DetectConfig)
    if is_dataclass(obj):
        try:
            return jsonify(asdict(obj))
        except Exception:
            pass
    # last-ditch: stringify
    return str(obj)
