"""Reconstruct world-frame, gravity-removed acceleration from accel + gyro.

Five orientation-fusion algorithms recover the *real* vertical acceleration
``a_z`` (and ``a_x, a_y``) from a tilted, possibly rotating phone, by
estimating device orientation, rotating the measured specific force into the
world frame, and subtracting gravity. They share the
:class:`~pyramidElevatorDist.physics.reconstruct_az.base.AZReconstructor` interface and differ
only in the fusion rule:

* :class:`ComplementaryReconstructor` — gyro high-pass + accel low-pass.
* :class:`MahonyReconstructor` — nonlinear complementary on SO(3), PI + bias.
* :class:`MadgwickReconstructor` — gradient-descent correction.
* :class:`ValentiReconstructor` — algebraic quaternion + adaptive gain (AQUA).
* :class:`ESKFReconstructor` — error-state Kalman filter, attitude + bias.

Primary papers are in ``references/`` (see ``references/REFERENCES.md``).

Usage::

    from pyramidElevatorDist.physics.reconstruct_az import build, SensorChannel
    algo = build("Madgwick")
    rec = algo.reconstruct({SensorChannel.ACC: sig.acc,
                            SensorChannel.GYR: sig.gyr})
    a_z = rec.az          # real vertical acceleration (m/s²)
"""

from __future__ import annotations

import pandas as pd

from .base import AZReconstructor, Reconstruction, SensorChannel
from .complementary import ComplementaryReconstructor
from .eskf import ESKFReconstructor
from .madgwick import MadgwickReconstructor
from .mahony import MahonyReconstructor
from .valenti import ValentiReconstructor

# Display name → class. Order is the natural method-complexity progression
# and is what the viewer's dropdown shows.
RECONSTRUCTORS: dict[str, type[AZReconstructor]] = {
    "Complementary": ComplementaryReconstructor,
    "Mahony": MahonyReconstructor,
    "Madgwick": MadgwickReconstructor,
    "Valenti": ValentiReconstructor,
    "ESKF": ESKFReconstructor,
}


def build(name: str, **kwargs) -> AZReconstructor:
    """Instantiate a reconstructor by display name (see :data:`RECONSTRUCTORS`)."""
    try:
        cls = RECONSTRUCTORS[name]
    except KeyError:
        raise KeyError(
            f"Unknown reconstructor {name!r}. "
            f"Available: {sorted(RECONSTRUCTORS)}"
        ) from None
    return cls(**kwargs)


# Valid ``--reconstruct`` / ``config.reconstruct`` values: pass-through plus
# every registered filter. ``"none"`` is the identity (today's behavior).
RECONSTRUCT_CHOICES: list[str] = ["none", *RECONSTRUCTORS]


def reconstruct_az(
    type: str, acc: pd.DataFrame, gyro: pd.DataFrame | None,
) -> pd.DataFrame:
    """One-liner accel replacement used as the first step of predict/segment.

    Reconstructs device orientation from ``acc`` + ``gyro`` with the named
    filter and returns a **"virtually-flat-phone"** accelerometer frame: the
    world-frame acceleration with gravity kept on the vertical axis, in the
    same ``timestamp_ms, x, y, z`` schema as ``acc``. Downstream gravity
    projection then trivially recovers the true signed vertical acceleration.

    ``type="none"`` (or a missing/empty gyro) returns ``acc`` unchanged, so
    callers that don't opt in are byte-for-byte unaffected.

    Args:
        type: ``"none"`` or a key of :data:`RECONSTRUCTORS`
            (``"Complementary"``, ``"Mahony"``, ``"Madgwick"``, ``"Valenti"``,
            ``"ESKF"``).
        acc: accelerometer frame, columns ``timestamp_ms, x, y, z`` (m/s²).
        gyro: gyroscope frame, columns ``timestamp_ms, x, y, z`` (rad/s), on
            its own clock (resampled onto ``acc`` internally).

    Returns:
        A DataFrame with columns ``timestamp_ms, x, y, z`` — either ``acc``
        unchanged, or the reconstructed flat-phone accel (``x,y`` = world
        horizontal, ``z`` = world vertical + ``|g|``).

    Note:
        ``type`` shadows the builtin by design — the public call shape is
        ``reconstruct_az(type=..., acc=..., gyro=...)``.
    """
    if type in (None, "none") or gyro is None or getattr(gyro, "empty", True):
        return acc
    rec = build(type).reconstruct(
        {SensorChannel.ACC: acc, SensorChannel.GYR: gyro}
    )
    return pd.DataFrame({
        "timestamp_ms": rec.timestamp_ms,
        "x": rec.ax,
        "y": rec.ay,
        "z": rec.az + rec.g_mag,   # keep gravity on the vertical axis
    })


__all__ = [
    "AZReconstructor",
    "Reconstruction",
    "SensorChannel",
    "ComplementaryReconstructor",
    "MahonyReconstructor",
    "MadgwickReconstructor",
    "ValentiReconstructor",
    "ESKFReconstructor",
    "RECONSTRUCTORS",
    "RECONSTRUCT_CHOICES",
    "build",
    "reconstruct_az",
]
