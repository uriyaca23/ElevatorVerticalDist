"""Pedagogical walkthrough figure for the Mahony orientation filter.

Drives the *shipped* :class:`~pyramidElevatorDist.physics.reconstruct_az.MahonyReconstructor`
on one controlled rotating-elevator example with **known ground truth**, logs
the per-sample internal state (learned gyro bias, correction error ``e``, trust
weight ``w``, tracked orientation), and renders the six-panel figure consumed by
the "Walkthrough: the Mahony filter step by step" section of
``docs/latex/main.tex``.

The example is synthetic on purpose: because we *inject* the true orientation,
the true gyro bias, and the true vertical acceleration, we can plot the filter's
estimate against the exact quantity it is trying to recover. To guarantee the
curves reflect the real algorithm and not a re-implementation, the instrumented
subclass only overrides ``_estimate_orientation`` (copied verbatim from
``mahony.py`` with logging added) and asserts its reconstructed ``a_z`` matches
the shipped filter's to floating-point tolerance.

Usage::

    venv/bin/python scripts/make_mahony_walkthrough.py
    venv/bin/python scripts/make_mahony_walkthrough.py --out /tmp/fig.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyramidElevatorDist.physics.reconstruct_az import SensorChannel, build
from pyramidElevatorDist.physics.reconstruct_az import _quaternion as quat
from pyramidElevatorDist.physics.reconstruct_az.mahony import MahonyReconstructor

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "latex" / "figures" / "gyro" / "mahony_walkthrough.png"

GRAVITY = 9.80665
FS = 100.0
DT = 1.0 / FS
TRUE_BIAS = np.array([0.030, -0.020, 0.015])  # rad/s, constant gyro offset to learn


# --------------------------------------------------------------------------
# Instrumented filter: identical loop to mahony.py, plus a per-sample trace.
# --------------------------------------------------------------------------
class _TracingMahony(MahonyReconstructor):
    """Mahony filter that records its internal state each iteration."""

    def _estimate_orientation(self, acc, gyr, dt, q0, g_mag):
        n = len(acc)
        out = np.empty((n, 4))
        acc_norm = np.linalg.norm(acc, axis=1)
        trust = self._accel_trust(acc_norm, g_mag)

        bias_log = np.zeros((n, 3))
        e_norm_log = np.zeros(n)
        gap_deg_log = np.zeros(n)  # angle between measured and predicted gravity

        q = q0.copy()
        bias = np.zeros(3)
        for i in range(n):
            omega = gyr[i].copy()
            if acc_norm[i] > 1e-6:
                a_hat = acc[i] / acc_norm[i]
                v_hat = quat.rotate(quat.conjugate(q), quat.WORLD_UP)
                e = np.cross(a_hat, v_hat) * trust[i]
                bias = bias - self.ki * e * dt[i]
                omega = omega - bias + self.kp * e
                e_norm_log[i] = np.linalg.norm(e)
                gap_deg_log[i] = np.degrees(
                    np.arccos(np.clip(float(np.dot(a_hat, v_hat)), -1.0, 1.0))
                )
            q = quat.from_omega(q, omega, dt[i])
            out[i] = q
            bias_log[i] = bias

        self.trace = {
            "bias": bias_log,
            "e_norm": e_norm_log,
            "gap_deg": gap_deg_log,
            "trust": trust,
        }
        return out


# --------------------------------------------------------------------------
# Ground-truth simulation
# --------------------------------------------------------------------------
def _axis_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    h = 0.5 * angle
    return np.array([np.cos(h), *(np.sin(h) * np.asarray(axis, dtype=float))])


def _euler_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body->world quaternion from intrinsic roll/pitch/yaw (Z*Y*X)."""
    return quat.mul(
        _axis_quat([0, 0, 1], yaw),
        quat.mul(_axis_quat([0, 1, 0], pitch), _axis_quat([1, 0, 0], roll)),
    )


def _raised_cosine(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Smooth 0->1 ramp over [t0, t1] (flat outside)."""
    x = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def _tilt_deg(q_seq: np.ndarray) -> np.ndarray:
    """Angle (deg) between the phone's body z-axis (in world) and world up."""
    z_world = quat.rotate_batch(q_seq, np.tile(quat.WORLD_UP, (len(q_seq), 1)))
    cos = np.clip(z_world[:, 2] / (np.linalg.norm(z_world, axis=1) + 1e-12), -1, 1)
    return np.degrees(np.arccos(cos))


def simulate() -> dict:
    """Synthesise a rotating up-elevator ride with known ground truth."""
    dur = 16.0
    t = np.arange(0.0, dur, DT)
    n = len(t)
    rng = np.random.default_rng(0)

    # --- true orientation: initial tilt, then a slow hand rotation 3.5-14.5 s ---
    ramp = _raised_cosine(t, 3.5, 14.5)
    roll = np.deg2rad(0.0) + np.deg2rad(-12.0) * ramp
    pitch = np.deg2rad(16.0) + np.deg2rad(14.0) * ramp
    yaw = np.deg2rad(0.0) + np.deg2rad(55.0) * ramp
    q_true = np.array([_euler_quat(roll[i], pitch[i], yaw[i]) for i in range(n)])

    # --- true world-frame vertical acceleration: an up-ride trapezoid pulse ---
    def pulse(t, lo, hi, amp):
        return amp * _raised_cosine(t, lo, lo + 0.6) * (1 - _raised_cosine(t, hi - 0.6, hi))

    az_true = pulse(t, 5.0, 7.0, 1.0) - pulse(t, 11.0, 13.0, 1.0)  # +accelerate, -decelerate
    f_world = np.column_stack([np.zeros(n), np.zeros(n), GRAVITY + az_true])

    # --- body-frame accelerometer: rotate specific force into body, add noise ---
    acc = np.array([quat.rotate(quat.conjugate(q_true[i]), f_world[i]) for i in range(n)])
    acc += rng.normal(0.0, 0.05, acc.shape)

    # --- body-frame gyro: true rate from q_true, plus constant bias + noise ---
    omega_true = np.zeros((n, 3))
    for i in range(n - 1):
        dq = (q_true[i + 1] - q_true[i]) / DT
        omega_true[i] = 2.0 * quat.mul(quat.conjugate(q_true[i]), dq)[1:4]
    omega_true[-1] = omega_true[-2]
    gyr = omega_true + TRUE_BIAS + rng.normal(0.0, 0.010, omega_true.shape)

    ts = (t * 1000.0)
    acc_df = pd.DataFrame({"timestamp_ms": ts, "x": acc[:, 0], "y": acc[:, 1], "z": acc[:, 2]})
    gyr_df = pd.DataFrame({"timestamp_ms": ts, "x": gyr[:, 0], "y": gyr[:, 1], "z": gyr[:, 2]})
    return {
        "t": t, "acc": acc, "gyr": gyr, "az_true": az_true,
        "q_true": q_true, "acc_df": acc_df, "gyr_df": gyr_df,
    }


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
RIDE_SPANS = [(5.0, 7.0), (11.0, 13.0)]  # accel / decel pulses


def _shade(ax, t):
    for a, b in RIDE_SPANS:
        ax.axvspan(a, b, color="0.85", alpha=0.6, lw=0)


def make_figure(out: Path) -> None:
    sim = simulate()
    t = sim["t"]
    sensors = {SensorChannel.ACC: sim["acc_df"], SensorChannel.GYR: sim["gyr_df"]}

    shipped = build("Mahony").reconstruct(sensors)   # the real filter
    traced = _TracingMahony()
    rec = traced.reconstruct(sensors)                # instrumented copy
    assert np.allclose(shipped.az, rec.az, atol=1e-9), "trace diverged from shipped filter"
    tr = traced.trace

    # raw "no reconstruction" vertical: project onto the FIXED seed gravity dir
    seed = int(0.5 * FS)
    g_hat0 = sim["acc"][:seed].mean(0)
    g_hat0 = g_hat0 / np.linalg.norm(g_hat0)
    a_vert_raw = sim["acc"] @ g_hat0 - rec.g_mag

    tilt_true = _tilt_deg(sim["q_true"])
    tilt_est = _tilt_deg(rec.quat)
    a_mag = np.linalg.norm(sim["acc"], axis=1)
    w_mag = np.linalg.norm(sim["gyr"], axis=1)

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.1), constrained_layout=True)

    # (1) Inputs -----------------------------------------------------------
    a = ax[0, 0]; _shade(a, t)
    a.plot(t, a_mag, color="#1f77b4", lw=1.2, label=r"$\|a\|$  accel")
    a.axhline(rec.g_mag, color="#1f77b4", ls=":", lw=0.8)
    a.set_ylabel(r"$\|a\|$  (m/s$^2$)", color="#1f77b4")
    a.tick_params(axis="y", colors="#1f77b4")
    a2 = a.twinx()
    a2.plot(t, w_mag, color="#d62728", lw=1.2, label=r"$\|\omega\|$  gyro")
    a2.set_ylabel(r"$\|\omega\|$  (rad/s)", color="#d62728")
    a2.tick_params(axis="y", colors="#d62728"); a2.grid(False)
    a.set_title("(1) Inputs: accel & gyro (body frame)")
    a.set_xlabel("time (s)")

    # (2) gravity-direction gap = what e measures --------------------------
    a = ax[0, 1]; _shade(a, t)
    a.plot(t, tr["gap_deg"], color="#6a3d9a", lw=1.2)
    a.set_title(r"(2) Gravity-direction error $\angle(\hat a,\hat v)$")
    a.set_ylabel("angle (deg)"); a.set_xlabel("time (s)")
    a.text(0.02, 0.92, r"$\|e\|\approx\sin(\cdot)$ drives the correction",
           transform=a.transAxes, va="top", fontsize=8, color="#6a3d9a")

    # (3) trust weight -----------------------------------------------------
    a = ax[0, 2]; _shade(a, t)
    a.plot(t, tr["trust"], color="#ff7f0e", lw=1.4)
    a.set_ylim(-0.05, 1.08)
    a.set_title(r"(3) Trust weight $w$ (accel gate)")
    a.set_ylabel(r"$w \in [0,1]$"); a.set_xlabel("time (s)")

    # (4) bias learning ----------------------------------------------------
    a = ax[1, 0]; _shade(a, t)
    cols = ["#1b9e77", "#7570b3", "#e7298a"]
    for k, (c, name) in enumerate(zip(cols, ["x", "y", "z"])):
        a.plot(t, tr["bias"][:, k], color=c, lw=1.3, label=fr"$\hat b_{name}$")
        a.axhline(TRUE_BIAS[k], color=c, ls="--", lw=0.9)
    a.set_title("(4) Learned gyro bias  (dashed = true)")
    a.set_ylabel("bias (rad/s)"); a.set_xlabel("time (s)")
    a.legend(ncol=3, fontsize=7.5, loc="lower right")

    # (5) orientation tracking --------------------------------------------
    a = ax[1, 1]; _shade(a, t)
    a.plot(t, tilt_true, color="k", lw=2.2, alpha=0.35, label="true tilt")
    a.plot(t, tilt_est, color="#2ca02c", lw=1.3, label="Mahony estimate")
    a.set_title("(5) Orientation tracking: phone tilt")
    a.set_ylabel("tilt vs. vertical (deg)"); a.set_xlabel("time (s)")
    a.legend(fontsize=7.5, loc="upper left")

    # (6) payoff -----------------------------------------------------------
    a = ax[1, 2]; _shade(a, t)
    a.plot(t, a_vert_raw, color="#999999", lw=1.2, label=r"raw $a_{\mathrm{vert}}$ (fixed frame)")
    a.plot(t, rec.az, color="#2ca02c", lw=1.5, label=r"Mahony $a_z$")
    a.plot(t, sim["az_true"], color="k", lw=1.4, ls="--", label=r"true $a_z$")
    a.set_title(r"(6) Payoff: recovered $a_z$")
    a.set_ylabel(r"$a_z$  (m/s$^2$)"); a.set_xlabel("time (s)")
    a.legend(fontsize=7.5, loc="upper right")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)

    # console summary (handy when regenerating)
    conv = np.abs(tr["bias"][-1] - TRUE_BIAS)
    raw_err = float(np.sqrt(np.mean((a_vert_raw - sim["az_true"]) ** 2)))
    rec_err = float(np.sqrt(np.mean((rec.az - sim["az_true"]) ** 2)))
    print(f"wrote {out}")
    print(f"  final bias error (rad/s): {conv}")
    print(f"  a_z RMSE vs truth  raw={raw_err:.3f}  Mahony={rec_err:.3f}  m/s^2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    make_figure(ap.parse_args().out)


if __name__ == "__main__":
    main()
