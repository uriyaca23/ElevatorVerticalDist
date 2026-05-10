"""Unified signal loader with post-load gap detection.

Single entry point :func:`load_data` for every consumer that needs an
accelerometer trace plus its valid intervals (UI pipeline, GT editor,
segmentation editor). Picks the right underlying loader based on
``source`` and runs the gap-detection / parts-splitting post-process so
all callers see one shape:

    LoadedSignal(acc, source, meta, valid_intervals, acc_parts)

Lives under ``src/data/`` (not under ``src/pipelines/streamlit/``) so
non-UI tools — ``src/data/gt_editor.py`` and
``src/segmentation/algorithms/editor.py`` — can import the helpers
without reaching across the layering boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.data.loadFromDB import LoadedSignal as _BaseLoadedSignal, PhoneType, loadDataFromS3
from src.data.loader import (
    TARGET_SAMPLE_RATE_HZ,
    detect_valid_intervals,
    resample_sensor_with_gaps,
)


@dataclass
class LoadedSignal(_BaseLoadedSignal):
    """Loaded signal extended with post-load gap analysis.

    Subclass of the bare :class:`src.data.loadFromDB.LoadedSignal` so the
    DB stub can keep its minimal contract while every downstream stage
    sees the richer shape it actually wants.

    * ``valid_intervals`` — contiguous spans of ``acc.timestamp_ms``
      sampled densely enough (gap ≤ ``GAP_THRESHOLD_S``). Outside these
      spans the signal is treated as "no data": the UI overlays a band
      and segmenter / predictor must not run there.
    * ``acc_parts`` — one DataFrame per valid interval, mirroring the
      canonical 4-column schema of ``acc``. Stages iterate this list so
      each part is a clean, gap-free signal; ``acc`` itself is kept for
      legacy callers that still want a single frame.
    """
    valid_intervals: list[tuple[int, int]] = field(default_factory=list)
    acc_parts: list[pd.DataFrame] = field(default_factory=list)


def split_acc_into_parts(
    acc: pd.DataFrame, valid_intervals: list[tuple[int, int]],
) -> list[pd.DataFrame]:
    """Slice ``acc`` into one DataFrame per valid interval.

    Each part is a contiguous, gap-free chunk of the original signal so
    downstream stages do not see fabricated samples across a dropout.
    With no intervals defined falls back to a single-part list
    containing ``acc`` as-is so legacy callers keep working.
    """
    if acc is None or acc.empty:
        return []
    if not valid_intervals:
        return [acc.reset_index(drop=True)]
    ts = acc["timestamp_ms"].astype("int64").to_numpy()
    parts: list[pd.DataFrame] = []
    for s_ms, e_ms in valid_intervals:
        mask = (ts >= int(s_ms)) & (ts <= int(e_ms))
        chunk = acc.loc[mask].reset_index(drop=True)
        if not chunk.empty:
            parts.append(chunk)
    return parts


def enrich_loaded(
    base: _BaseLoadedSignal,
    *,
    resample: bool = True,
    target_hz: int = TARGET_SAMPLE_RATE_HZ,
) -> LoadedSignal:
    """Run the canonical post-load pipeline on a raw signal.

    Promotes the loader's plain :class:`LoadedSignal` to the enriched
    variant with ``valid_intervals`` / ``acc_parts`` populated.

    When ``resample`` is true (the default — matches what
    :func:`getExperimentData` does for structured experiments) the ACC
    frame is first split per gap and resampled to ``target_hz`` (50 Hz)
    inside each valid interval. This is what every downstream stage —
    segmenter, predictor, both UIs (editor + Streamlit boutique) — has
    historically expected: a clean uniform cadence with explicit holes
    where the raw recording was too sparse to recover. Set
    ``resample=False`` only if the caller has its own resampling
    pipeline already.

    Idempotent on already-resampled inputs (the resampler is a no-op
    when the cadence already matches ``target_hz``).
    """
    acc = base.acc
    if acc is not None and not acc.empty:
        if resample:
            acc, intervals = resample_sensor_with_gaps(acc, target_hz=target_hz)
        else:
            intervals = detect_valid_intervals(acc["timestamp_ms"].to_numpy())
    else:
        intervals = []
    parts = split_acc_into_parts(acc, intervals)
    meta = {**base.meta, "valid_intervals_count": len(intervals)}
    if resample:
        meta = {**meta, "resampled_hz": int(target_hz)}
    return LoadedSignal(
        acc=acc, source=base.source, meta=meta,
        prs=base.prs, gyr=base.gyr, mag=base.mag, ori=base.ori,
        valid_intervals=intervals, acc_parts=parts,
    )


def load_data(
    source: str,
    *,
    # DB args
    phone_type: PhoneType | None = None,
    phone_id: str | None = None,
    t_start: datetime | None = None,
    t_end: datetime | None = None,
    experiment: str | None = None,
    # File args
    acc: pd.DataFrame | None = None,
    source_label: str | None = None,
    meta: dict[str, Any] | None = None,
    # Pipeline args
    resample: bool = True,
    target_hz: int = TARGET_SAMPLE_RATE_HZ,
) -> LoadedSignal:
    """Dispatch to the right loader and apply the canonical post-load
    pipeline (resample to ``target_hz`` inside each valid interval, then
    split into gap-free parts).

    ``source``:

    * ``"db"`` — fetch via :func:`loadDataFromS3` (phone args required).
    * ``"file"`` — wrap a pre-cleaned ``acc`` frame (CSV ingestion via
      :func:`src.data.csv_ingest.parse_csv_to_acc` is upstream; this
      just packages it).

    Returns the enriched :class:`LoadedSignal` regardless of source so
    callers can treat all paths uniformly. ``resample=False`` skips the
    50 Hz step for callers that already manage their own cadence.
    """
    if source == "db":
        if phone_type is None or phone_id is None or t_start is None or t_end is None:
            raise ValueError(
                "load_data(source='db') requires phone_type, phone_id, "
                "t_start, t_end."
            )
        base = loadDataFromS3(
            phone_type, phone_id, t_start, t_end, experiment=experiment,
        )
    elif source == "file":
        if acc is None:
            raise ValueError("load_data(source='file') requires acc.")
        base = _BaseLoadedSignal(
            acc=acc,
            source=source_label or "File",
            meta=dict(meta or {}),
        )
    else:
        raise ValueError(
            f"Unknown source: {source!r}. Expected 'db' or 'file'."
        )
    return enrich_loaded(base, resample=resample, target_hz=target_hz)
