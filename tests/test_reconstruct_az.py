"""Synthetic-truth tests for the accel+gyro vertical-acceleration filters.

We build a signal with *known* world-vertical acceleration and a *known*
device orientation, synthesise the body-frame accelerometer + gyroscope a
real phone would report, and check each reconstructor recovers the vertical
acceleration:

* **Static tilt** — a tilted but non-rotating phone. Every filter must
  recover ``a_z`` near-exactly and drive the horizontal channels to ~0.
* **Rotating during the ride** — the phone turns while the elevator moves.
  Reconstruction (which tracks orientation with the gyro) must beat the
  project's current *fixed* gravity projection by a wide margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyramidElevatorDist.physics.reconstruct_az import RECONSTRUCTORS, SensorChannel, build
from pyramidElevatorDist.physics.reconstruct_az import _quaternion as quat

G = 9.80665
FS = 100.0


def _trapezoid_profile() -> np.ndarray:
    """1 s stationary, 6 s of ride-like vertical accel, 0.5 s stationary."""
    pre = np.zeros(int(1.0 * FS))
    t = np.arange(int(6.0 * FS)) / FS
    ride = 0.8 * np.sin(2.0 * np.pi * t / 4.0)
    post = np.zeros(int(0.5 * FS))
    return np.concatenate([pre, ride, post])


def _make_signal(q_true: np.ndarray, az_true: np.ndarray, gyro_body: np.ndarray):
    """Synthesise (acc_df, gyr_df) for a known orientation + vertical accel."""
    n = len(az_true)
    ts = np.arange(n) / FS * 1000.0
    f_world = np.column_stack([np.zeros(n), np.zeros(n), G + az_true])  # specific force
    acc = np.array([quat.rotate(quat.conjugate(q_true[i]), f_world[i])
                    for i in range(n)])                                 # world→body
    acc_df = pd.DataFrame({"timestamp_ms": ts,
                           "x": acc[:, 0], "y": acc[:, 1], "z": acc[:, 2]})
    gyr_df = pd.DataFrame({"timestamp_ms": ts,
                           "x": gyro_body[:, 0], "y": gyro_body[:, 1], "z": gyro_body[:, 2]})
    return acc_df, gyr_df


@pytest.mark.parametrize("name", list(RECONSTRUCTORS))
def test_static_tilt_recovers_az(name):
    """Tilted but non-rotating phone: a_z recovered, horizontals ~0."""
    az_true = _trapezoid_profile()
    n = len(az_true)
    # Constant 30° tilt about body-x, no rotation.
    tilt = quat.from_two_vectors(np.array([0.0, np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30))]),
                                 quat.WORLD_UP)
    q_true = np.tile(tilt, (n, 1))
    gyro = np.zeros((n, 3))
    acc_df, gyr_df = _make_signal(q_true, az_true, gyro)

    rec = build(name).reconstruct({SensorChannel.ACC: acc_df, SensorChannel.GYR: gyr_df})

    # Ignore the first 0.5 s (filter settle); compare the rest.
    s = int(0.5 * FS)
    assert np.mean(np.abs(rec.az[s:] - az_true[s:])) < 0.08, f"{name}: a_z error too large"
    assert np.mean(np.abs(rec.ax[s:])) < 0.15, f"{name}: a_x leak"
    assert np.mean(np.abs(rec.ay[s:])) < 0.15, f"{name}: a_y leak"
    # Quaternions stay unit norm.
    assert np.allclose(np.linalg.norm(rec.quat, axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", list(RECONSTRUCTORS))
def test_rotating_beats_fixed_projection(name):
    """Phone rotating mid-ride: gyro-tracked reconstruction beats fixed gravity."""
    az_true = _trapezoid_profile()
    n = len(az_true)
    dt = 1.0 / FS
    # Constant body-frame roll rate → integrate true orientation.
    w = np.array([0.3, 0.0, 0.0])
    gyro = np.tile(w, (n, 1))
    q_true = np.empty((n, 4))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for i in range(n):
        q = quat.from_omega(q, w, dt)
        q_true[i] = q
    acc_df, gyr_df = _make_signal(q_true, az_true, gyro)

    rec = build(name).reconstruct({SensorChannel.ACC: acc_df, SensorChannel.GYR: gyr_df})

    # Fixed-gravity baseline: project on the seed orientation (identity at t=0)
    # held constant for the whole ride — i.e. the current pipeline's assumption.
    acc = acc_df[["x", "y", "z"]].to_numpy()
    az_fixed = acc[:, 2] - G

    s = int(0.5 * FS)
    err_recon = np.mean(np.abs(rec.az[s:] - az_true[s:]))
    err_fixed = np.mean(np.abs(az_fixed[s:] - az_true[s:]))

    assert err_recon < 0.2, f"{name}: reconstruction a_z error {err_recon:.3f} too large"
    assert err_recon < 0.4 * err_fixed, (
        f"{name}: reconstruction ({err_recon:.3f}) not clearly better than "
        f"fixed projection ({err_fixed:.3f})"
    )


def test_build_and_dependencies():
    for name, cls in RECONSTRUCTORS.items():
        algo = build(name)
        assert isinstance(algo, cls)
        assert algo.dependencies() == {SensorChannel.ACC, SensorChannel.GYR}
    with pytest.raises(KeyError):
        build("NoSuchFilter")


def test_missing_sensor_raises():
    algo = build("Madgwick")
    acc_df = pd.DataFrame({"timestamp_ms": [0, 10], "x": [0, 0], "y": [0, 0], "z": [G, G]})
    with pytest.raises(ValueError):
        algo.reconstruct({SensorChannel.ACC: acc_df})  # no gyro


def test_reconstruct_az_none_and_missing_gyro_are_identity():
    """The one-liner is a pass-through for 'none' or when gyro is absent."""
    from pyramidElevatorDist.physics.reconstruct_az import reconstruct_az
    acc = pd.DataFrame({"timestamp_ms": [0, 10, 20], "x": [0.0, 0, 0],
                        "y": [0.0, 0, 0], "z": [G, G, G]})
    gyr = pd.DataFrame({"timestamp_ms": [0, 10, 20], "x": [0.0, 0, 0],
                        "y": [0.0, 0, 0], "z": [0.0, 0, 0]})
    assert reconstruct_az("none", acc, gyr) is acc
    assert reconstruct_az("Madgwick", acc, None) is acc


def test_reconstruct_az_flat_phone_keeps_gravity_on_z():
    """A filtered call returns the flat-phone accel: gravity on +z, |g| kept."""
    from pyramidElevatorDist.physics.reconstruct_az import reconstruct_az, RECONSTRUCT_CHOICES
    assert RECONSTRUCT_CHOICES == ["none", *RECONSTRUCTORS]

    az_true = _trapezoid_profile()
    n = len(az_true)
    tilt = quat.from_two_vectors(
        np.array([0.0, np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30))]),
        quat.WORLD_UP)
    q_true = np.tile(tilt, (n, 1))
    acc_df, gyr_df = _make_signal(q_true, az_true, np.zeros((n, 3)))

    out = reconstruct_az("Madgwick", acc_df, gyr_df)
    assert list(out.columns) == ["timestamp_ms", "x", "y", "z"]
    s = int(0.5 * FS)
    # z carries gravity + true vertical accel; horizontals collapse to ~0.
    assert np.mean(np.abs(out["z"].to_numpy()[s:] - (az_true[s:] + G))) < 0.08
    assert np.mean(np.abs(out["x"].to_numpy()[s:])) < 0.15
    assert np.mean(np.abs(out["y"].to_numpy()[s:])) < 0.15
