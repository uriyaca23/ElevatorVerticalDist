"""Public display-signal API.

The UI plots the acceleration the pipeline actually operates on, but must not
reach into detector internals. :func:`reconstructedSignal` returns that
whole-trace series — optionally after gyro orientation reconstruction — as a
clean, documented frame, so the UI stays presentation-only.

``RECONSTRUCT_CHOICES`` is re-exported so the UI can offer the method picker
(``"none"`` + the accel+gyro filters) without importing the physics package.
"""
from __future__ import annotations

import pandas as pd

from src.physics.reconstruct_az import RECONSTRUCT_CHOICES

# Thin façade: the resample → reconstruct → vertical-accel computation lives in
# the shared boutique core; here we only pick the cadence policy and forward.
from src.pipelines.inprocess import (
    RESAMPLE_TARGET_HZ,
    barometric_altitude as _barometric_altitude,
    signal as _signal,
)

__all__ = ["reconstructedSignal", "barometricAltitude", "RECONSTRUCT_CHOICES"]


def reconstructedSignal(
    acc: pd.DataFrame,
    gyro: pd.DataFrame | None = None,
    reconstruct: str = "none",
    resample: bool = True,
) -> pd.DataFrame:
    """Whole-trace vertical-acceleration series for display.

    Parameters
    ----------
    acc:
        Raw accelerometer samples — columns ``timestamp_ms``, ``x``, ``y``,
        ``z``. May be at any (even variable) sample rate.
    gyro:
        Optional gyroscope stream (``timestamp_ms``, ``x``, ``y``, ``z``,
        rad/s). Required for a meaningful reconstruction; ignored when
        ``reconstruct == "none"``.
    reconstruct:
        One of :data:`RECONSTRUCT_CHOICES`. ``"none"`` (default) returns the
        plain vertical accel; any filter name returns the gyro
        orientation-corrected "reconstructed a_z".
    resample:
        When ``True`` (default), ``acc`` is first normalized onto the uniform
        50 Hz grid, matching :func:`findSegments`. Set ``False`` when ``acc``
        is already at the canonical cadence.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp_ms``, ``a_vert`` (gravity-projected vertical accel,
        m/s², post-reconstruction), ``a_mag_g`` (``|a| − g``). Empty frame with
        those columns when the input is empty.
    """
    return _signal(
        acc, gyro=gyro, reconstruct=reconstruct,
        resample_hz=(RESAMPLE_TARGET_HZ if resample else None),
    )


def barometricAltitude(
    prs: pd.DataFrame,
    temperature_c: float | None = None,
) -> pd.DataFrame:
    """Whole-trace barometric altitude — the ground-truth reference signal.

    Parameters
    ----------
    prs:
        Pressure samples — columns ``timestamp_ms``, ``pressure`` (hPa).
    temperature_c:
        Optional surface temperature for the ISA inversion; ``None`` uses the
        standard 15 °C.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp_ms``, ``altitude_m`` (metres above the ISA
        reference). Empty frame with those columns when ``prs`` is empty or
        lacks a ``pressure`` column.
    """
    return _barometric_altitude(prs, temperature_c=temperature_c)
