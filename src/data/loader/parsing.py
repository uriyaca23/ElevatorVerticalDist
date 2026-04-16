"""Raw parsing: sensor logs, metadata files, filename timestamps.

These helpers are shared by both the legacy and the pipeline entry points.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.physics import pressure_to_altitude

from .constants import DATA_ROOT, SENSOR_COLUMNS


# Accept any `...YYYYMMDDTHHMMSS.txt` tail — some filenames carry extra tokens
# between `sensorLog_` and the ISO timestamp (e.g. `sensorLog_abc_xyz_<iso>.txt`).
_ISO_FILENAME_RE = re.compile(r"(\d{8}T\d{6})\.txt$")


def _normalize_exp(exp: int | str) -> str:
    if isinstance(exp, int):
        return f"exp{exp}"
    s = str(exp).strip()
    return s if s.startswith("exp") else f"exp{s}"


def _resolve_exp_dir(name: str, exp: int | str, data_root: Path | str = DATA_ROOT) -> Path:
    data_root = Path(data_root)
    exp_dir = data_root / name / _normalize_exp(exp)
    if not exp_dir.is_dir():
        name_dir = data_root / name
        available = (
            sorted(p.name for p in name_dir.iterdir() if p.is_dir())
            if name_dir.is_dir() else []
        )
        raise FileNotFoundError(
            f"Experiment directory not found: {exp_dir}. "
            f"Available exps under {name_dir}: {', '.join(available) or '(none)'}"
        )
    return exp_dir


def _find_sensor_log(exp_dir: Path) -> Path:
    matches = sorted(exp_dir.glob("sensorLog_*.txt"))
    if not matches:
        # forBarometer/ copies often keep macOS's "Copy of " prefix.
        matches = sorted(exp_dir.glob("Copy of sensorLog_*.txt"))
    if not matches:
        raise FileNotFoundError(f"No sensorLog_*.txt found in {exp_dir}")
    return matches[0]


def _parse_sensor_log(log_path: Path) -> dict[str, pd.DataFrame]:
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


def _parse_metadata_file(meta_path: Path) -> dict[str, str]:
    """Parse `Key: Value` line-oriented metadata.txt; returns {} if missing.

    Unknown/extra keys are kept verbatim. Lines without ':' are skipped.
    """
    if not meta_path.exists():
        return {}
    out: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        if key:
            out[key] = val.strip()
    return out


def _parse_iso_filename_to_ms(log_path: Path) -> int:
    """Extract wall-clock start ms (Unix epoch) from
    `sensorLog_YYYYMMDDTHHMMSS.txt`.

    The filename timestamp is interpreted in the machine's local timezone
    (matches the convention of `metadata.txt` `Date`/`Time` fields, which are
    also recorded in local time on the device).
    """
    m = _ISO_FILENAME_RE.search(log_path.name)
    if not m:
        raise ValueError(f"Cannot parse ISO timestamp from filename: {log_path.name}")
    dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
    return int(dt.timestamp() * 1000)


def _first_boot_ms_in_log(log_path: Path) -> int:
    """Return the smallest valid `timestamp_ms` in the raw sensorLog file.

    Scans the whole file (cheap — just reads first column) and returns the
    minimum, since lines from different sensors can interleave slightly out
    of order. Used to compute the boot→wall-clock offset.
    """
    from .constants import SENSOR_COLUMNS as _SC
    best: int | None = None
    with log_path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            ts, sensor, *values = parts
            cols = _SC.get(sensor)
            if cols is None or len(values) != len(cols):
                continue
            try:
                t = int(ts)
            except ValueError:
                continue
            if best is None or t < best:
                best = t
    if best is None:
        raise ValueError(f"No valid sensor lines found in {log_path}")
    return best
