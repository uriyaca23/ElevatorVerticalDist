"""Database-backed signal loader (stub).

Public surface is :func:`loadDataFromS3`. Real deployment fetches the
sensor traces from S3 (or whatever cloud store the phone app uploads
to) keyed by ``(phone_type, t_start, t_end)``.

For now the body is a local-disk fallback: it returns the sensor CSVs
of a fixed structured experiment so the rest of the app has something
real to chew on. ACC is always returned; PRS / GYR / MAG / ORI come
through whenever the experiment folder has them. Swap this out once
the DB/S3 backend is live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd


# Relative to this file: ../data/structuredData/data/<exp>/<sensor>.csv
_STRUCTURED_ROOT = Path(__file__).resolve().parent / "structuredData" / "data"

# Fallback experiment used while the DB backend is still mocked.
_DEFAULT_EXPERIMENT = "eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2"

# Canonical schema for each sensor file the mock S3 loader serves.
# `data_cols` are the non-time, non-meta columns kept on the returned
# frame; everything else (incl. the `exp_name` column the structured
# CSVs ship with, and PRS's derived `GT_height_m`) is stripped so the
# mock matches what a real S3 fetch would deliver — raw sensor data
# only, with derivations done downstream.
_SENSOR_SCHEMAS: dict[str, list[str]] = {
    "ACC": ["x", "y", "z"],
    "PRS": ["pressure"],
    "GYR": ["x", "y", "z"],
    "MAG": ["x", "y", "z"],
    "ORI": ["w", "x", "y", "z"],
}


class PhoneType(str, Enum):
    A = "Phone Type A"
    B = "Phone Type B"


@dataclass
class LoadedSignal:
    """Uniform result schema the data pipelines consume.

    ``acc`` is the primary, always-populated accelerometer frame.
    ``prs`` / ``gyr`` / ``mag`` / ``ori`` carry the optional sensor
    channels — ``None`` when the source backend had nothing for that
    sensor. Consumers that only need ACC (the Streamlit predictor)
    keep reading ``acc``; consumers that want the full bundle (the GT
    editor's auto-segmenter and save path) read each sensor field
    directly.
    """
    acc: pd.DataFrame                       # timestamp_ms, x, y, z
    source: str                             # human label for the report
    meta: dict[str, Any] = field(default_factory=dict)
    prs: pd.DataFrame | None = None         # timestamp_ms, pressure
    gyr: pd.DataFrame | None = None         # timestamp_ms, x, y, z
    mag: pd.DataFrame | None = None         # timestamp_ms, x, y, z
    ori: pd.DataFrame | None = None         # timestamp_ms, w, x, y, z


def _load_sensor_csv(exp_dir: Path, sensor: str) -> pd.DataFrame | None:
    """Read ``<exp>/<sensor>.csv`` keeping only the canonical columns,
    or return None when the file is absent. ACC is required upstream;
    every other sensor is optional and a missing file just means the
    mock backend has nothing to serve for it."""
    path = exp_dir / f"{sensor}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    schema = _SENSOR_SCHEMAS[sensor]
    required = {"timestamp_ms", *schema}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df[["timestamp_ms", *schema]].copy()


def loadDataFromS3(
    phone_type: PhoneType,
    phone_id: str,
    t_start: datetime,
    t_end: datetime,
    experiment: str | None = None,
) -> LoadedSignal:
    """Fetch a sensor bundle by phone + time window.

    Currently a stub: loads a fixed local experiment's sensor CSVs
    regardless of the inputs. ACC is always returned; PRS / GYR / MAG
    / ORI come through when the experiment folder has them. Inputs
    flow into ``meta`` so the downstream report shows what the user
    asked for.

    Args:
        phone_type: enum identifying the device model bucket.
        phone_id: user-facing device identifier (IMEI, serial, or label).
            Used by the real backend to disambiguate multiple devices of
            the same type; passes through to ``meta`` for the report.
        t_start / t_end: requested time window (passes through to meta).
        experiment: optional override of the fallback experiment name.
    """
    exp_name = experiment or _DEFAULT_EXPERIMENT
    exp_dir = _STRUCTURED_ROOT / exp_name
    if not exp_dir.exists():
        raise FileNotFoundError(
            f"Mock experiment folder not found: {exp_dir}"
        )

    acc = _load_sensor_csv(exp_dir, "ACC")
    if acc is None:
        raise FileNotFoundError(
            f"ACC.csv not found for experiment {exp_name!r}: {exp_dir}"
        )
    prs = _load_sensor_csv(exp_dir, "PRS")
    gyr = _load_sensor_csv(exp_dir, "GYR")
    mag = _load_sensor_csv(exp_dir, "MAG")
    ori = _load_sensor_csv(exp_dir, "ORI")

    n = len(acc)
    if n > 1:
        ts = acc["timestamp_ms"].to_numpy()
        dt_ms = float((ts[-1] - ts[0]) / max(n - 1, 1))
        fs_hz = 1000.0 / dt_ms if dt_ms > 0 else float("nan")
    else:
        fs_hz = float("nan")

    present = ["ACC"] + [
        n for n, df in (("PRS", prs), ("GYR", gyr), ("MAG", mag), ("ORI", ori))
        if df is not None
    ]
    return LoadedSignal(
        acc=acc,
        source=f"DB (stub) · {phone_type.value} · {phone_id or '—'} · {exp_name}",
        meta={
            "experiment":     exp_name,
            "phone_type":     phone_type.value,
            "phone_id":       phone_id,
            "t_start":        t_start.isoformat(timespec="seconds"),
            "t_end":          t_end.isoformat(timespec="seconds"),
            "samples":        int(n),
            "sample_rate":    f"{fs_hz:.1f} Hz" if fs_hz == fs_hz else "?",
            "backend":        "local_stub",
            "sensors":        present,
        },
        prs=prs,
        gyr=gyr,
        mag=mag,
        ori=ori,
    )
