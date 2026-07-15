"""Abstract interface for accelerometer+gyroscope vertical-acceleration
reconstruction.

The phone is held at an arbitrary, possibly time-varying tilt, so its raw
``z`` axis is not world-vertical. To recover the *real* vertical
acceleration ``a_z`` we must know the device orientation at every sample,
rotate the measured specific force into the world frame, and subtract
gravity. A gyroscope tracks orientation change accurately over the short
term but drifts; the accelerometer pins down the gravity direction over the
long term but is corrupted by the ride's own (linear) acceleration. Fusing
the two is exactly what the five concrete reconstructors do — they differ
only in *how* they fuse.

:class:`AZReconstructor` captures everything that is common:

* **dependencies()** — which sensor streams the algorithm needs.
* **reconstruct()** — a *template method* that

    1. validates the required channels are present,
    2. aligns every channel onto the accelerometer's timeline (the streams
       arrive on independent clocks; see ``src/data/loadFromDB.py``),
    3. seeds the initial orientation from the opening stationary window,
    4. delegates the actual filter to the subclass via
       :meth:`_estimate_orientation`,
    5. rotates the body-frame accel into the world frame and removes
       gravity, returning a :class:`Reconstruction`.

So a subclass implements *only* the orientation filter; the rotate /
gravity-removal arithmetic — the part that must stay numerically consistent
with the quaternion convention in :mod:`._quaternion` — lives here once.

Input contract (matches the structured-data ``LoadedSignal`` frames):

* ``acc`` — columns ``timestamp_ms, x, y, z`` in **m/s²**.
* ``gyr`` — columns ``timestamp_ms, x, y, z`` in **rad/s**.

Both already SI for the structured dataset; callers on other datasets
(e.g. USC-HAD in g / dps) convert at the edge before calling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pyramidElevatorDist.utils.accelerometer_utils import estimate_gravity_stationary

from . import _quaternion as quat

GRAVITY = 9.80665  # standard gravity (m/s²); fallback when the seed is degenerate


class SensorChannel(str, Enum):
    """Sensor streams a reconstructor may depend on (``LoadedSignal`` fields)."""
    ACC = "acc"
    GYR = "gyr"
    MAG = "mag"
    ORI = "ori"


@dataclass
class Reconstruction:
    """World-frame, gravity-removed acceleration for one signal.

    ``ax/ay/az`` are linear acceleration in the Earth frame (z = up) with
    gravity subtracted, so ``az`` is the *real vertical acceleration* that
    feeds the trapezoid / ZUPT predictors. ``quat`` is the per-sample
    body→world orientation (handy for plotting against the phone's own
    ``ori`` quaternion). Everything is on ``timestamp_ms`` (the accel clock).
    """
    timestamp_ms: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    quat: np.ndarray            # (N, 4) [w, x, y, z], body→world
    g_mag: float                # gravity magnitude removed (m/s²)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def a_vert(self) -> np.ndarray:
        """Alias for :pyattr:`az` — the real vertical acceleration."""
        return self.az


class AZReconstructor(ABC):
    """Base class: reconstruct world-frame acceleration from accel + gyro.

    Subclasses override :meth:`dependencies` and :meth:`_estimate_orientation`.
    ``reconstruct`` itself is concrete (template method) so the rotate /
    gravity-removal path is written once.

    Args:
        accel_trust_tol: half-width (as a fraction of g) of the band around
            ``|a| = g`` within which the accelerometer is fully trusted as a
            gravity reference. During the ride's accel/decel pulses ``|a|``
            departs from g, so the correction is smoothly down-weighted —
            this is what stops ride acceleration from being mistaken for tilt
            and is essential for a clean reconstructed trapezoid.
        seed_window_sec: length of the opening window assumed (quasi-)
            stationary, used to estimate the initial gravity direction.
    """

    def __init__(
        self,
        accel_trust_tol: float = 0.15,
        seed_window_sec: float = 0.5,
    ) -> None:
        self.accel_trust_tol = float(accel_trust_tol)
        self.seed_window_sec = float(seed_window_sec)

    # --------------------------------------------------------------- public ---
    @abstractmethod
    def dependencies(self) -> set[SensorChannel]:
        """Sensor channels this algorithm requires to run."""

    def reconstruct(
        self, sensors: Mapping[SensorChannel, pd.DataFrame],
    ) -> Reconstruction:
        """Reconstruct world-frame, gravity-removed acceleration.

        ``sensors`` maps :class:`SensorChannel` to its DataFrame (the
        caller typically passes ``{ACC: sig.acc, GYR: sig.gyr}``). Channels
        beyond :meth:`dependencies` are ignored.
        """
        acc_df, gyr_df = self._validate(sensors)
        ts, acc, gyr, dt = self._align(acc_df, gyr_df)
        if len(ts) < 2:
            return self._empty(ts)

        q0, g_mag = self._seed_orientation(acc, dt)
        q_seq = self._estimate_orientation(acc, gyr, dt, q0, g_mag)
        q_seq = q_seq / (np.linalg.norm(q_seq, axis=1, keepdims=True) + 1e-12)

        # Body → world, then subtract gravity along world +z.
        a_world = quat.rotate_batch(q_seq, acc)
        a_world[:, 2] -= g_mag

        return Reconstruction(
            timestamp_ms=ts,
            ax=a_world[:, 0], ay=a_world[:, 1], az=a_world[:, 2],
            quat=q_seq, g_mag=g_mag,
            meta={"algorithm": self.name, "n": int(len(ts)),
                  "fs_hz": float(1.0 / np.median(dt)) if len(dt) else float("nan")},
        )

    @property
    def name(self) -> str:
        """Short identifier (class name minus the ``Reconstructor`` suffix)."""
        return type(self).__name__.removesuffix("Reconstructor")

    # ------------------------------------------------------------- subclass ---
    @abstractmethod
    def _estimate_orientation(
        self,
        acc: np.ndarray,    # (N, 3) specific force, body frame, m/s²
        gyr: np.ndarray,    # (N, 3) angular rate, body frame, rad/s
        dt: np.ndarray,     # (N,) per-sample timestep, s
        q0: np.ndarray,     # (4,) seed orientation, body→world
        g_mag: float,       # estimated gravity magnitude, m/s²
    ) -> np.ndarray:
        """Return the per-sample body→world orientation as an ``(N, 4)`` array."""

    # --------------------------------------------------------------- shared ---
    def _accel_trust(self, acc_norm: np.ndarray, g_mag: float) -> np.ndarray:
        """Per-sample weight in [0, 1] for trusting the accelerometer as gravity.

        1 when ``|a| ≈ g``; decays smoothly (Gaussian in the normalised
        magnitude error) as the ride's linear acceleration pulls ``|a|`` away
        from g. ``accel_trust_tol`` sets the 1-σ width.
        """
        err = np.abs(acc_norm - g_mag) / max(g_mag, 1e-6)
        return np.exp(-0.5 * (err / max(self.accel_trust_tol, 1e-6)) ** 2)

    # --------------------------------------------------------------- private --
    def _validate(
        self, sensors: Mapping[SensorChannel, pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        need = self.dependencies()
        missing = [c for c in need if sensors.get(c) is None
                   or getattr(sensors.get(c), "empty", True)]
        if missing:
            raise ValueError(
                f"{self.name}: missing required sensor(s): "
                f"{sorted(c.value for c in missing)}"
            )
        return sensors[SensorChannel.ACC], sensors[SensorChannel.GYR]

    def _align(
        self, acc_df: pd.DataFrame, gyr_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Put accel and gyro on the accelerometer's timeline.

        The accel frame is already the project's clean, gap-resampled
        reference; the gyro arrives on its own clock, so each gyro axis is
        linearly interpolated onto the accel timestamps. Returns
        ``(timestamp_ms, acc[N,3], gyr[N,3], dt[N])``.
        """
        ts = acc_df["timestamp_ms"].to_numpy(dtype=float)
        acc = acc_df[["x", "y", "z"]].to_numpy(dtype=float)

        gt = gyr_df["timestamp_ms"].to_numpy(dtype=float)
        gyr = np.column_stack([
            np.interp(ts, gt, gyr_df["x"].to_numpy(dtype=float)),
            np.interp(ts, gt, gyr_df["y"].to_numpy(dtype=float)),
            np.interp(ts, gt, gyr_df["z"].to_numpy(dtype=float)),
        ])

        t_s = (ts - ts[0]) / 1000.0
        dt = np.diff(t_s, prepend=t_s[0])
        if len(dt) > 1:
            dt[0] = dt[1]
        # Guard against zero/negative steps from duplicate timestamps.
        med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.02
        dt[dt <= 0] = med
        return ts, acc, gyr, dt

    def _seed_orientation(
        self, acc: np.ndarray, dt: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Initial body→world quaternion + gravity magnitude from window start.

        Estimates the gravity vector from the opening (quasi-stationary)
        window and builds the rotation that carries that body-frame gravity
        direction onto world +z. Yaw is left arbitrary — irrelevant for the
        vertical channel.
        """
        fs = 1.0 / float(np.median(dt)) if len(dt) else 50.0
        n_seed = max(5, min(len(acc), int(self.seed_window_sec * fs)))
        gvec, g_mag, _ = estimate_gravity_stationary(
            acc[:n_seed, 0], acc[:n_seed, 1], acc[:n_seed, 2],
            fs=fs, window_sec=min(self.seed_window_sec, 0.25),
        )
        if not np.isfinite(g_mag) or g_mag < 1e-3:
            gvec, g_mag = np.array([0.0, 0.0, GRAVITY]), GRAVITY
        # q0 maps measured gravity direction (body) onto world up (+z).
        q0 = quat.from_two_vectors(gvec, quat.WORLD_UP)
        return quat.normalize(q0), float(g_mag)

    def _empty(self, ts: np.ndarray) -> Reconstruction:
        z = np.zeros(len(ts))
        q = np.tile([1.0, 0.0, 0.0, 0.0], (len(ts), 1)) if len(ts) else np.zeros((0, 4))
        return Reconstruction(ts, z, z, z, q, GRAVITY, meta={"algorithm": self.name})
