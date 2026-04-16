"""CLI entry point: ``python -m src.data.loader [name] [exp]``.

Loads (or reuses the cached) per-sensor DataFrames for a rawData experiment
and prints row counts and columns per sensor.
"""

from __future__ import annotations

import sys

from .legacy import loadDataWithGT


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "eyal"
    exp = sys.argv[2] if len(sys.argv) > 2 else 1
    data = loadDataWithGT(name, exp)
    for sensor, df in data.items():
        print(f"{sensor}: {len(df):>7} rows  cols={list(df.columns)}")


if __name__ == "__main__":
    main()
