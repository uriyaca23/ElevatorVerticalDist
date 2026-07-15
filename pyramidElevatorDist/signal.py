"""Public display-signal API.

The UI plots the acceleration the pipeline actually operates on, but must not
reach into detector internals. :func:`reconstructedSignal` returns that
whole-trace series — optionally after gyro orientation reconstruction — as a
clean, documented frame, so the UI stays presentation-only.

``RECONSTRUCT_CHOICES`` is re-exported so the UI can offer the method picker
(``"none"`` + the accel+gyro filters) without importing the physics package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyramidElevatorDist.physics.reconstruct_az import RECONSTRUCT_CHOICES

# Thin façade: the resample → reconstruct → vertical-accel computation lives in
# the shared boutique core; here we only pick the cadence policy and forward.
from src.pipelines.inprocess import (
    RESAMPLE_TARGET_HZ,
    barometric_altitude as _barometric_altitude,
    signal as _signal,
)

__all__ = [
    "reconstructedSignal",
    "barometricAltitude",
    "displaySeries",
    "RECONSTRUCT_CHOICES",
]

#: Human labels for the two vertical-acceleration display modes.
AZ_LABEL_RECONSTRUCTED = "a_z reconstructed"
AZ_LABEL_FALLBACK = "|a|-g"


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


def displaySeries(
    sig: pd.DataFrame,
    has_gyro: bool = False,
    reconstruct: str = "none",
) -> tuple[np.ndarray, str, bool]:
    """Pick the vertical-acceleration trace to display, with its label.

    The UI shows the gyro-reconstructed vertical acceleration (``a_vert`` —
    the signed, world-frame a_z) whenever a reconstruction is actually
    *active*: a gyro stream is present **and** ``reconstruct != "none"``.
    Without a gyro (or with ``reconstruct == "none"``) there is nothing to
    reconstruct, so it falls back to the rotation-invariant ``|a| − g``
    residual the detector matches on. The plain fixed-gravity projection is
    intentionally never surfaced — the display is always either the
    reconstructed a_z or ``|a| − g``.

    Parameters
    ----------
    sig:
        A frame from :func:`reconstructedSignal` (columns ``a_vert``,
        ``a_mag_g``). ``None`` / empty is tolerated.
    has_gyro:
        Whether the experiment actually has a gyroscope stream.
    reconstruct:
        The selected reconstruction method (``"none"`` or a filter name).

    Returns
    -------
    (values, label, reconstructed):
        ``values`` — the chosen column as a float ndarray (empty when
        ``sig`` is empty); ``label`` — ``"a_z reconstructed"`` or
        ``"|a|-g"``; ``reconstructed`` — which branch was taken, so callers
        can drop ``|a| − g``-domain overlays (e.g. lobe amplitude markers)
        when the signed a_z is on screen.
    """
    reconstructed = bool(has_gyro) and reconstruct not in (None, "none")
    col = "a_vert" if reconstructed else "a_mag_g"
    label = AZ_LABEL_RECONSTRUCTED if reconstructed else AZ_LABEL_FALLBACK
    if sig is None or len(sig) == 0 or col not in getattr(sig, "columns", []):
        return np.empty(0, dtype=float), label, reconstructed
    return sig[col].to_numpy(dtype=float), label, reconstructed


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
