"""Clean-room smoke test for the pyramidElevatorDist wheel.

Builds the wheel from the root pyproject.toml, installs it into a FRESH
venv, and runs every public entry point on synthetic data with the repo off
``sys.path`` — proving the installed package is self-contained (no ``src``
leak, no file outside its install directory).

Usage:
    venv/bin/python scripts/package_smoke_test.py [--skip-build]

``--skip-build`` reuses the newest wheel already in dist/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "dist"
SMOKE_VENV = REPO / "build" / "smoke-venv"

SMOKE_DRIVER = r'''
import importlib.util, sys
assert importlib.util.find_spec("src") is None, "FAIL: a top-level 'src' package is importable"
import numpy as np, pandas as pd
import pyramidElevatorDist as P

fs, n = 50, 120*50
t = np.arange(n)/fs*1000.0
z = np.full(n, 9.81); z[30*fs:45*fs] += 0.6*np.hanning(15*fs)
acc = pd.DataFrame({"timestamp_ms": t, "x": np.zeros(n), "y": np.zeros(n), "z": z})
seg = {"type": "up", "start_s": 30.0, "end_s": 45.0}
prs = pd.DataFrame({"timestamp_ms": t[:100], "pressure": 1013.0 - np.arange(100)*0.001})

segs = P.findSegments(acc, resample=False)
assert all(isinstance(s, P.RideSegment) for s in segs)
P.findSegmentsDetailed(acc, resample=False)
P.findSegmentParameters(acc, 30.0, 45.0, resample=False)
res = P.predictSegment(acc, seg, resample=False)
assert isinstance(res, P.PredictionResult)
P.predictByParameters(acc, seg, {"W": 3.0, "f": 0.5, "abs_A": 0.3}, resample=False)
sig = P.reconstructedSignal(acc, resample=False)
P.displaySeries(sig)
P.barometricAltitude(prs)

# Deprecated alias still importable.
from pyramidElevatorDist.predection import predictSegment as _alias  # noqa: F401

# Typed error paths fire.
try:
    P.findSegments(acc.drop(columns=["z"]))
except P.MissingColumnsError:
    pass
else:
    raise AssertionError("MissingColumnsError did not fire")

bad = [m for m in sys.modules if m == "src" or m.startswith("src.")]
assert not bad, f"FAIL: src modules leaked into sys.modules: {bad}"
print(f"SMOKE OK: pyramidElevatorDist {P.__version__} at {P.__file__}")
'''


def log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def build_wheel() -> Path:
    log("building wheel via python -m build …")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(DIST)],
        check=True, cwd=str(REPO),
    )
    return newest_wheel()


def newest_wheel() -> Path:
    wheels = sorted(DIST.glob("pyramidelevatordist-*.whl"),
                    key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise SystemExit("no wheel found in dist/ — run without --skip-build")
    return wheels[-1]


def smoke_test(wheel: Path) -> None:
    if SMOKE_VENV.exists():
        shutil.rmtree(SMOKE_VENV)
    log(f"creating fresh smoke venv at {SMOKE_VENV} …")
    subprocess.run([sys.executable, "-m", "venv", str(SMOKE_VENV)], check=True)
    py = SMOKE_VENV / "bin" / "python"
    subprocess.run([str(py), "-m", "pip", "install", "--quiet",
                    "--upgrade", "pip"], check=True)
    log("installing wheel (+deps from index) into the fresh venv …")
    subprocess.run([str(py), "-m", "pip", "install", "--quiet", str(wheel)],
                   check=True)
    # cwd = the smoke venv so the repo root (and its src/) is NOT on sys.path
    subprocess.run([str(py), "-c", SMOKE_DRIVER], check=True,
                   cwd=str(SMOKE_VENV))
    log("clean-room smoke test PASSED")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true",
                    help="reuse the newest wheel already in dist/")
    args = ap.parse_args()
    wheel = newest_wheel() if args.skip_build else build_wheel()
    log(f"wheel: {wheel.name}")
    smoke_test(wheel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
