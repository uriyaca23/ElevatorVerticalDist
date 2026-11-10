"""Generate parity fixtures for web/lib/reconstruct.ts from the Python pipeline.

Run from the repo root with the project venv:

    venv/bin/python web/scripts/make_reconstruct_fixture.py

Writes web/tests/fixtures/reconstruct_parity.json containing
  - "synthetic": a small tilted-phone elevator ride (accel + gyro, committed)
    with Python's reconstructedSignal(..., "Mahony") output;
  - "experiments": every 25th sample of a_vert / a_mag_g for a few local
    structuredData experiments (the TS test reads their ACC/GYR.csv and skips
    when the corpus is absent).
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyramidElevatorDist import reconstructedSignal  # noqa: E402

DATA_DIR = ROOT / "src/data/structuredData/data"
OUT = ROOT / "web/tests/fixtures/reconstruct_parity.json"
EXPERIMENTS = [
    "eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2",
    "UriyaCohenEliya_beitMansour1_GooglePixel10_15-04-2026_exp5",
]
STRIDE = 25


def run(acc: pd.DataFrame, gyr: pd.DataFrame) -> pd.DataFrame:
    return reconstructedSignal(acc, gyr, reconstruct="Mahony")


def synthetic() -> dict:
    # 60 s at ~50 Hz with jitter: phone tilted 30° about x, a 12 s up-ride
    # (±1 m/s² trapezoid lobes) and a slow 20° yaw turn mid-way.
    rng = np.random.default_rng(3)
    n = 3000
    ts = 1_760_000_000_000 + np.cumsum(np.r_[0, rng.integers(18, 23, size=n - 1)])
    t = (ts - ts[0]) / 1000.0
    a_up = np.where((t > 20) & (t < 22), 1.0, 0.0) - np.where((t > 30) & (t < 32), 1.0, 0.0)
    tilt = np.deg2rad(30)
    g = 9.80665
    f_world = np.c_[np.zeros(n), np.zeros(n), g + a_up]
    # body = Rx(-tilt) · world
    c, s = np.cos(tilt), np.sin(tilt)
    acc = np.c_[f_world[:, 0], c * f_world[:, 1] + s * f_world[:, 2], -s * f_world[:, 1] + c * f_world[:, 2]]
    acc += rng.normal(0, 0.03, size=acc.shape)
    gyr = rng.normal(0, 0.002, size=(n, 3))
    gyr[(t > 40) & (t < 44), 2] += np.deg2rad(5)
    acc_df = pd.DataFrame({"timestamp_ms": ts.astype(np.int64), "x": acc[:, 0].round(6), "y": acc[:, 1].round(6), "z": acc[:, 2].round(6)})
    gyr_df = pd.DataFrame({"timestamp_ms": (ts + 3).astype(np.int64), "x": gyr[:, 0].round(6), "y": gyr[:, 1].round(6), "z": gyr[:, 2].round(6)})
    out = run(acc_df, gyr_df)
    return {
        "acc": {k: acc_df[k].tolist() for k in ("timestamp_ms", "x", "y", "z")},
        "gyr": {k: gyr_df[k].tolist() for k in ("timestamp_ms", "x", "y", "z")},
        "out": {k: out[k].astype(float).tolist() for k in ("timestamp_ms", "a_vert", "a_mag_g")},
    }


def resample_poly_cases() -> list[dict]:
    rng = np.random.default_rng(11)
    cases = []
    for up, down, n in [(19, 20, 157), (20, 21, 64), (5, 4, 33), (1, 2, 40)]:
        x = rng.normal(0, 1, size=n).round(6)
        cases.append({"x": x.tolist(), "up": up, "down": down, "y": resample_poly(x, up, down).tolist()})
    return cases


def main() -> None:
    experiments = {}
    for name in EXPERIMENTS:
        d = DATA_DIR / name
        if not (d / "ACC.csv").exists():
            print(f"skip {name}")
            continue
        acc = pd.read_csv(d / "ACC.csv")[["timestamp_ms", "x", "y", "z"]]
        gyr = pd.read_csv(d / "GYR.csv")[["timestamp_ms", "x", "y", "z"]]
        out = run(acc, gyr).iloc[::STRIDE]
        experiments[name] = {k: out[k].astype(float).round(6).tolist() for k in ("timestamp_ms", "a_vert", "a_mag_g")}
        print(f"{name}: {len(out)} samples")
    OUT.write_text(json.dumps({
        "synthetic": synthetic(),
        "experiments": experiments,
        "resample_poly": resample_poly_cases(),
        "limit_denominator": [
            [x, *(lambda f: (f.numerator, f.denominator))(Fraction(x).limit_denominator(1000))]
            for x in (50 / (1000 / 19), 50 / (1000 / 21.5), 50 / 52.0, 0.333333, 50 / 49.0, 1.0)
        ],
    }))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
