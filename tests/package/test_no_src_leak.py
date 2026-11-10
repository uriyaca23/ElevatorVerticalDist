"""The package must be importable and runnable without any `src` module.

Run in a subprocess so this test is meaningful even when other tests in the
session already imported `src.*`.
"""
from __future__ import annotations

import subprocess
import sys

_IMPORT_ONLY = (
    "import sys, pyramidElevatorDist;"
    "bad = [m for m in sys.modules if m == 'src' or m.startswith('src.')];"
    "assert not bad, bad;"
    "print('ok')"
)

_FULL_CALL = r"""
import sys
import numpy as np, pandas as pd
import pyramidElevatorDist as P

fs, n = 50, 60 * 50
t = np.arange(n) / fs
z = np.full(n, 9.81)
def pulse(tc, W, f, A):
    dt = np.abs(t - tc); y = np.zeros_like(t)
    y[dt <= f * W] = A
    ramp = (dt > f * W) & (dt <= W)
    y[ramp] = A * (W - dt[ramp]) / (W * (1 - f))
    return y
z = z + pulse(20.0, 1.2, 0.5, 1.0) - pulse(32.0, 1.2, 0.5, 1.0)
acc = pd.DataFrame({"timestamp_ms": (t * 1000).astype("int64"),
                    "x": np.zeros(n), "y": np.zeros(n), "z": z})
prs = pd.DataFrame({"timestamp_ms": (t * 1000).astype("int64"),
                    "pressure": 1013.25 - t * 0.001})
P.findSegments(acc)
P.predictSegment(acc, {"type": "up", "start_s": 19.0, "end_s": 33.0})
P.reconstructedSignal(acc)
P.barometricAltitude(prs)
bad = [m for m in sys.modules if m == 'src' or m.startswith('src.')]
assert not bad, bad
print('ok')
"""


def _run(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_import_does_not_load_src():
    assert _run(_IMPORT_ONLY) == "ok"


def test_full_call_does_not_load_src():
    assert _run(_FULL_CALL) == "ok"
