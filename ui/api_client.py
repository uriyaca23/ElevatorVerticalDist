"""Thin HTTP client for the boutique pipeline.

The Streamlit step modules used to import :func:`predict_intervals` and
:class:`Predictor` and call them in-process. They now go through this
client instead — the API service in ``api/`` owns the algorithm code.

The base URL is ``API_URL`` from the environment (default
``http://localhost:8000`` so local dev without docker-compose still
works). ``rehydrate_state`` walks the JSON-decoded detector state and
turns the array fields back into ``np.ndarray`` so the display helpers
that the UI still imports (``heatmap_at``, ``classify_peak``, …) get the
shapes they expect.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests


# Set in docker-compose; defaults to local-dev addr.
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

# Generous default — /segment on a 5-minute trace is mostly numeric work
# (~1 s on a laptop), but we don't want a slow first request to time out
# while uvicorn is still spinning up.
_DEFAULT_TIMEOUT_S = 120.0

# Detector state keys whose values are 1-D numeric arrays. Walked back to
# np.ndarray after JSON decode so display helpers behave identically to
# the in-process code path.
_STATE_ARRAY_KEYS = (
    "t", "a_vert", "a_smooth",
    "best_r2", "best_A", "best_W_idx", "best_f_idx",
    "best_pos_r2", "best_pos_A", "best_neg_r2", "best_neg_A",
    "best_r2_gated", "signs",
    "grid_w_s", "grid_f",
    "initial_peaks", "final_peaks",
)


def _acc_payload(acc: pd.DataFrame) -> dict[str, list[float]]:
    return {
        "timestamp_ms": acc["timestamp_ms"].astype(float).tolist(),
        "x": acc["x"].astype(float).tolist(),
        "y": acc["y"].astype(float).tolist(),
        "z": acc["z"].astype(float).tolist(),
    }


def rehydrate_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert JSON-decoded detector state back into the in-process shape.

    Two transforms:

    * Known array keys → ``np.ndarray``. Display helpers
      (`heatmap_at`, `classify_peak`, …) call ``np.where``, ``arr[mask]``
      etc. directly, so they need numpy.
    * ``state['config']`` → ``SimpleNamespace``. The detector serialises
      its dataclass config to a plain dict; the UI reads attributes off
      it (``cfg.r2_peak_thresh``, ``cfg.min_peak_abs_a``) so we wrap the
      dict to restore attribute access without re-importing the
      ``DetectConfig`` dataclass.
    """
    if state is None:
        return None
    for k in _STATE_ARRAY_KEYS:
        if k in state and isinstance(state[k], list):
            arr = np.asarray(
                [np.nan if v is None else v for v in state[k]],
                dtype=float if k not in ("best_W_idx", "best_f_idx",
                                         "initial_peaks", "final_peaks")
                else int,
            )
            state[k] = arr
    cfg = state.get("config")
    if isinstance(cfg, dict):
        state["config"] = SimpleNamespace(**cfg)
    return state


def segment(acc: pd.DataFrame, phone_model: str = "",
            include_state: bool = True) -> tuple[list[dict], dict | None, float]:
    """POST /segment.

    Returns ``(predictions, state, t0_ms)``. ``state`` is rehydrated to
    numpy and is ``None`` when the service couldn't produce a detection
    state (e.g. empty trace) or when ``include_state=False``.
    """
    body = {
        "acc": _acc_payload(acc),
        "phone_model": phone_model,
        "include_state": include_state,
    }
    r = requests.post(f"{API_URL}/segment", json=body, timeout=_DEFAULT_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    state = rehydrate_state(data.get("state"))
    return data["predictions"], state, data.get("t0_ms")


def predict(acc: pd.DataFrame,
            segments: Iterable[dict],
            phone_model: str = "",
            algorithms: list[str] | None = None) -> tuple[dict[str, list[dict]], str]:
    """POST /predict.

    ``segments`` is an iterable of dicts with ``type``, ``start_s``,
    ``end_s``. Returns ``(rows_by_algo, primary_algo_id)``.
    """
    body = {
        "acc": _acc_payload(acc),
        "segments": [
            {"type": str(s["type"]),
             "start_s": float(s["start_s"]),
             "end_s": float(s["end_s"])}
            for s in segments
        ],
        "phone_model": phone_model,
    }
    if algorithms is not None:
        body["algorithms"] = list(algorithms)
    r = requests.post(f"{API_URL}/predict", json=body, timeout=_DEFAULT_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    return data["rows_by_algo"], data["primary"]
