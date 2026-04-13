"""Sensor log loader.

Loads `sensorLog_*.txt` files for a given experimenter (subfolder name under
`src/data/rawData/`) into pandas DataFrames.

File format (tab-separated, no header), one sample per line:
    <timestamp_ms>\t<SENSOR>\t<value1>\t<value2>...

Sensor schemas:
    ACC, GYR, MAG       -> x, y, z
    RAWGYR, RAWMAG      -> x, y, z, bias_x, bias_y, bias_z
    ORI                 -> w, x, y, z   (quaternion)
    PRS                 -> pressure
    GPS                 -> lat, lon, alt
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.physics import pressure_to_altitude

DATA_ROOT = Path(__file__).resolve().parent / "rawData"

SENSOR_COLUMNS: dict[str, list[str]] = {
    "ACC":    ["x", "y", "z"],
    "GYR":    ["x", "y", "z"],
    "MAG":    ["x", "y", "z"],
    "RAWGYR": ["x", "y", "z", "bias_x", "bias_y", "bias_z"],
    "RAWMAG": ["x", "y", "z", "bias_x", "bias_y", "bias_z"],
    "ORI":    ["w", "x", "y", "z"],
    "PRS":    ["pressure"],
    "GPS":    ["lat", "lon", "alt"],
}


def _list_experimenters(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        return []
    return sorted(p.name for p in data_root.iterdir() if p.is_dir())


def _find_sensor_log(experimenter_dir: Path, data_root: Path) -> Path:
    matches = sorted(experimenter_dir.glob("sensorLog_*.txt"))
    if not matches:
        available = _list_experimenters(data_root)
        available_str = ", ".join(available) if available else "(none)"
        raise FileNotFoundError(
            f"No sensorLog_*.txt found in {experimenter_dir}. "
            f"Available experimenters in {data_root}: {available_str}"
        )
    return matches[0]


def load_experimenter(
    experimenter: str,
    data_root: Path | str = DATA_ROOT,
) -> dict[str, pd.DataFrame]:
    """Load an experimenter's sensor log into per-sensor DataFrames.

    Returns a dict mapping sensor name (e.g. "ACC") to a DataFrame with a
    `timestamp_ms` column plus the sensor's value columns, sorted by time.
    """
    data_root = Path(data_root)
    exp_dir = data_root / experimenter
    if not exp_dir.is_dir():
        available = _list_experimenters(data_root)
        available_str = ", ".join(available) if available else "(none)"
        raise FileNotFoundError(
            f"Experimenter directory not found: {exp_dir}. "
            f"Available experimenters in {data_root}: {available_str}"
        )

    log_path = _find_sensor_log(exp_dir, data_root)

    buckets: dict[str, list[list[str]]] = {s: [] for s in SENSOR_COLUMNS}
    with log_path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            ts, sensor, *values = parts
            cols = SENSOR_COLUMNS.get(sensor)
            if cols is None or len(values) != len(cols):
                continue
            buckets[sensor].append([ts, *values])

    frames: dict[str, pd.DataFrame] = {}
    for sensor, rows in buckets.items():
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["timestamp_ms", *SENSOR_COLUMNS[sensor]])
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], downcast="integer")
        for col in SENSOR_COLUMNS[sensor]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames[sensor] = df.sort_values("timestamp_ms").reset_index(drop=True)

    if "PRS" in frames:
        frames["PRS"]["GT_height_m"] = pressure_to_altitude(frames["PRS"]["pressure"])

    return frames


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "oria"
    data = load_experimenter(name)
    for sensor, df in data.items():
        print(f"{sensor}: {len(df):>7} rows  cols={list(df.columns)}")
