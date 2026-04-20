"""Database-backed signal loader (stub).

Public surface is :func:`loadDataFromS3`. Real deployment fetches the
accelerometer trace from S3 (or whatever cloud store the phone app
uploads to) keyed by ``(phone_type, t_start, t_end)``.

For now the body is a local-disk fallback: it returns the ACC.csv of a
fixed structured experiment so the rest of the app has something real
to chew on. Swap this out once the DB/S3 backend is live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd


# Relative to this file: ../data/structuredData/data/<exp>/ACC.csv
_STRUCTURED_ROOT = Path(__file__).resolve().parent / "structuredData" / "data"

# Fallback experiment used while the DB backend is still mocked.
_DEFAULT_EXPERIMENT = "eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2"


class PhoneType(str, Enum):
    A = "Phone Type A"
    B = "Phone Type B"


@dataclass
class LoadedSignal:
    """Uniform result schema the Streamlit pipeline consumes."""
    acc: pd.DataFrame                       # timestamp_ms, x, y, z
    source: str                             # human label for the report
    meta: dict[str, Any] = field(default_factory=dict)


def _load_experiment_acc(exp_name: str = _DEFAULT_EXPERIMENT) -> pd.DataFrame:
    """Read ``structuredData/data/<exp>/ACC.csv`` and keep only the schema
    the detector expects."""
    path = _STRUCTURED_ROOT / exp_name / "ACC.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"ACC.csv not found for experiment {exp_name!r}: {path}"
        )
    df = pd.read_csv(path)
    required = {"timestamp_ms", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df[["timestamp_ms", "x", "y", "z"]].copy()


def loadDataFromS3(
    phone_type: PhoneType,
    phone_id: str,
    t_start: datetime,
    t_end: datetime,
    experiment: str | None = None,
) -> LoadedSignal:
    """Fetch an accelerometer trace by phone + time window.

    Currently a stub: loads a fixed local experiment's ACC.csv regardless
    of the inputs. The inputs are still captured into ``meta`` so the
    downstream report shows what the user asked for.

    Args:
        phone_type: enum identifying the device model bucket.
        phone_id: user-facing device identifier (IMEI, serial, or label).
            Used by the real backend to disambiguate multiple devices of
            the same type; passes through to ``meta`` for the report.
        t_start / t_end: requested time window (passes through to meta).
        experiment: optional override of the fallback experiment name.
    """
    exp_name = experiment or _DEFAULT_EXPERIMENT
    acc = _load_experiment_acc(exp_name)

    n = len(acc)
    if n > 1:
        ts = acc["timestamp_ms"].to_numpy()
        dt_ms = float((ts[-1] - ts[0]) / max(n - 1, 1))
        fs_hz = 1000.0 / dt_ms if dt_ms > 0 else float("nan")
    else:
        fs_hz = float("nan")

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
        },
    )
