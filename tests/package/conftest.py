"""Synthetic-data fixtures for the package-only test suite.

Everything here imports ONLY pyramidElevatorDist + numpy/pandas/pytest —
no ``src``, no experiment corpus — so the suite passes on a clean machine
after ``pip install pyramidElevatorDist``.

The synthetic ride is built to be detectable by the template-match
defaults: A=1.0 m/s² clears the amplitude floors, W=1.2 s and f=0.5 sit
inside the (W, f) grid, the 12 s lobe gap is a legal ride duration, the
inter-lobe region is quiet, and the pulse IS the matched template so the
joint R² is ≈1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

FS = 50.0
T0_MS = 1_700_000_000_000  # arbitrary epoch anchor

# Canonical synthetic-ride shape parameters (shared by tests).
RIDE = dict(t_c1=20.0, t_c2=32.0, W=1.2, f=0.5, A=1.0)


def trapezoid_pulse(t: np.ndarray, t_c: float, W: float, f: float,
                    A: float) -> np.ndarray:
    """Symmetric trapezoid: plateau A for |t-t_c| <= f*W, linear ramps to 0
    at |t-t_c| = W."""
    y = np.zeros_like(t)
    dt = np.abs(t - t_c)
    flat = dt <= f * W
    ramp = (dt > f * W) & (dt <= W)
    y[flat] = A
    y[ramp] = A * (W - dt[ramp]) / (W * (1.0 - f) + 1e-12)
    return y


def make_ride_acc(dur_s: float = 60.0, direction: str = "up",
                  noise: float = 0.02, seed: int = 0,
                  **shape) -> pd.DataFrame:
    """A 50 Hz accelerometer trace containing exactly one synthetic ride."""
    p = {**RIDE, **shape}
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    sign = +1.0 if direction == "up" else -1.0
    ride = (trapezoid_pulse(t, p["t_c1"], p["W"], p["f"], sign * p["A"])
            + trapezoid_pulse(t, p["t_c2"], p["W"], p["f"], -sign * p["A"]))
    return pd.DataFrame({
        "timestamp_ms": (t * 1000).astype(np.int64) + T0_MS,
        "x": rng.normal(0.0, noise, n),
        "y": rng.normal(0.0, noise, n),
        "z": 9.81 + ride + rng.normal(0.0, noise, n),
    })


def make_gyro(dur_s: float = 60.0) -> pd.DataFrame:
    """A zero-rotation gyro stream on the same clock."""
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    return pd.DataFrame({
        "timestamp_ms": (t * 1000).astype(np.int64) + T0_MS,
        "x": np.zeros(n), "y": np.zeros(n), "z": np.zeros(n),
    })


def make_prs(dur_s: float = 60.0, dh_m: float = 3.0,
             t_c1: float = RIDE["t_c1"],
             t_c2: float = RIDE["t_c2"]) -> pd.DataFrame:
    """Pressure ramping down across the ride ≈ +dh_m of altitude
    (~0.12 hPa per metre near sea level)."""
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    frac = np.clip((t - t_c1) / (t_c2 - t_c1), 0.0, 1.0)
    pressure = 1013.25 - frac * dh_m * 0.12
    return pd.DataFrame({
        "timestamp_ms": (t * 1000).astype(np.int64) + T0_MS,
        "pressure": pressure,
    })


def nan_equal(a, b, rtol: float = 1e-9) -> bool:
    """Recursive equality tolerant to NaN and numpy arrays."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(nan_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            nan_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return a.shape == b.shape and bool(
            np.allclose(a, b, rtol=rtol, equal_nan=True))
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) or np.isnan(b):
            return np.isnan(a) and np.isnan(b)
        return bool(np.isclose(a, b, rtol=rtol))
    return a == b


@pytest.fixture(scope="session")
def ride_acc() -> pd.DataFrame:
    return make_ride_acc()


@pytest.fixture(scope="session")
def quiet_acc() -> pd.DataFrame:
    """50 Hz stationary trace with no ride in it."""
    n = int(30.0 * FS)
    rng = np.random.default_rng(7)
    t = np.arange(n) / FS
    return pd.DataFrame({
        "timestamp_ms": (t * 1000).astype(np.int64) + T0_MS,
        "x": rng.normal(0, 0.02, n),
        "y": rng.normal(0, 0.02, n),
        "z": 9.81 + rng.normal(0, 0.02, n),
    })
