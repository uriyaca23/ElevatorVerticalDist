"""CSV → canonical ACC parsing.

Pure data-layer helpers for turning a user-uploaded tabular file into
the canonical 4-column accelerometer schema (``timestamp_ms``, ``x``,
``y``, ``z``). UI surfaces (Streamlit, GT editor) call
:func:`parse_csv_to_acc` to get a clean DataFrame they can hand to
:func:`src.data.load_data.enrich_loaded`. They contain **no** logic —
all the heuristics live here so the same parsing runs regardless of
which UI ingested the file.

This module deliberately knows nothing about Streamlit or matplotlib
— it only depends on numpy / pandas so non-UI tools can reuse it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Canonical ACC schema, matching ``src/data/structuredData/data/<exp>/ACC.csv``.
CANONICAL_COLUMNS = ("timestamp_ms", "x", "y", "z")
MIN_SAMPLES = 10


@dataclass
class CsvParseInfo:
    """Diagnostic notes from a CSV parse — surfaced by UIs as captions."""

    time_format: str
    n_samples: int
    fs_hz: float
    n_dropped_nonnumeric: int
    n_sorted_out_of_order: int
    n_dedup_duplicate_ts: int

    def notes(self) -> list[str]:
        """User-facing one-liners describing what cleaning was done."""
        out: list[str] = []
        if self.n_dropped_nonnumeric:
            out.append(f"dropped {self.n_dropped_nonnumeric} non-numeric rows")
        if self.n_sorted_out_of_order:
            out.append(f"sorted {self.n_sorted_out_of_order} out-of-order samples")
        if self.n_dedup_duplicate_ts:
            out.append(f"removed {self.n_dedup_duplicate_ts} duplicate timestamps")
        return out


def detect_time_unit(ts: np.ndarray) -> tuple[str, np.ndarray]:
    """Identify how the user encoded time and convert to int64 ms.

    Heuristic, based on the magnitude of the largest timestamp:

    * ``> 1e12`` — Unix epoch milliseconds (a ms past year 2001 is
      already 13 digits, so anything this large is unambiguously ms).
    * ``> 1e9``  — Unix epoch seconds (10-digit values like
      ``1774373973``).
    * otherwise — relative time from the start of the recording. Span
      ``< 1e4`` is treated as seconds (a 5-minute capture spans ~300
      units); anything wider is already in ms.
    """
    if ts.size == 0:
        raise ValueError("Time column is empty after dropping non-numeric rows.")
    mx = float(np.nanmax(ts))
    mn = float(np.nanmin(ts))
    span = mx - mn
    if mx > 1e12:
        return "Unix epoch milliseconds", ts.astype("int64")
    if mx > 1e9:
        return "Unix epoch seconds", (ts * 1000.0).astype("int64")
    if span < 1e4:
        return "relative seconds (from t=0)", (ts * 1000.0).astype("int64")
    return "relative milliseconds (from t=0)", ts.astype("int64")


def parse_csv_to_acc(
    df: pd.DataFrame, mapping: dict[str, str],
) -> tuple[pd.DataFrame, CsvParseInfo]:
    """Reduce a user-provided CSV to the canonical ACC schema.

    ``mapping`` maps each canonical column (``timestamp_ms``, ``x``,
    ``y``, ``z``) to the source column the user picked. Returns the
    cleaned 4-column DataFrame and a :class:`CsvParseInfo` with parse
    diagnostics. The output is **not** yet resampled or split into
    valid intervals — pass it to
    :func:`src.data.load_data.enrich_loaded` to apply the same 50 Hz
    pipeline ``getExperimentData`` uses.

    Raises ``ValueError`` on any input the parser cannot recover from
    (duplicate column mappings, unmapped columns, too few numeric rows
    after cleaning, zero-span timestamps).
    """
    cols = {k: mapping[k] for k in CANONICAL_COLUMNS}
    if len(set(cols.values())) != len(cols):
        raise ValueError(
            "The same source column was mapped to more than one canonical "
            "column. Pick a different column for each of time / x / y / z."
        )
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in the CSV: {missing}")

    t_raw = pd.to_numeric(df[cols["timestamp_ms"]], errors="coerce").to_numpy(dtype=float)
    x_raw = pd.to_numeric(df[cols["x"]],            errors="coerce").to_numpy(dtype=float)
    y_raw = pd.to_numeric(df[cols["y"]],            errors="coerce").to_numpy(dtype=float)
    z_raw = pd.to_numeric(df[cols["z"]],            errors="coerce").to_numpy(dtype=float)

    good = (np.isfinite(t_raw) & np.isfinite(x_raw)
            & np.isfinite(y_raw) & np.isfinite(z_raw))
    n_dropped = int((~good).sum())
    t_raw = t_raw[good]; x_raw = x_raw[good]
    y_raw = y_raw[good]; z_raw = z_raw[good]

    if t_raw.size < MIN_SAMPLES:
        raise ValueError(
            f"Not enough numeric rows after cleaning "
            f"(need ≥{MIN_SAMPLES}, got {t_raw.size}). Check that the "
            "columns you mapped actually hold numbers and that the file "
            "isn't mostly blank."
        )

    time_label, ts_ms = detect_time_unit(t_raw)

    order = np.argsort(ts_ms, kind="stable")
    out_of_order = int(np.sum(np.diff(ts_ms) < 0))
    ts_ms = ts_ms[order]
    x_raw = x_raw[order]; y_raw = y_raw[order]; z_raw = z_raw[order]

    dup_mask = np.concatenate([[False], np.diff(ts_ms) == 0])
    n_dups = int(dup_mask.sum())
    if n_dups:
        keep = ~dup_mask
        ts_ms = ts_ms[keep]; x_raw = x_raw[keep]
        y_raw = y_raw[keep]; z_raw = z_raw[keep]

    if ts_ms.size < MIN_SAMPLES:
        raise ValueError(
            f"Only {ts_ms.size} samples left after deduping repeated "
            f"timestamps (need ≥{MIN_SAMPLES}). The time column may have "
            "many repeated values — check that you mapped the right column."
        )

    n = ts_ms.size
    span_s = float(ts_ms[-1] - ts_ms[0]) / 1000.0
    if span_s <= 0:
        raise ValueError(
            "Timestamps span zero seconds — every sample carries the same "
            "time value. Map the time column to a non-constant column."
        )
    fs_hz = (n - 1) / span_s

    acc = pd.DataFrame({
        "timestamp_ms": ts_ms,
        "x": x_raw, "y": y_raw, "z": z_raw,
    })
    info = CsvParseInfo(
        time_format=time_label,
        n_samples=int(n),
        fs_hz=float(fs_hz),
        n_dropped_nonnumeric=n_dropped,
        n_sorted_out_of_order=out_of_order,
        n_dedup_duplicate_ts=n_dups,
    )
    return acc, info
