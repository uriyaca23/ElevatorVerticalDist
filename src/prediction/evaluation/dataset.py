"""Dataset helpers for the prediction evaluation.

We flatten every experiment's ``gt.csv`` into a list of
:class:`SegmentRecord` tuples that carry exactly what the predictor
needs: the in-segment ACC slice, short pre/post stationary windows,
the ground-truth Δh, the train/test split, the clean flag, and some
metadata (phone, experiment name, location) for the analysis figures.

Only the elevator segments (``type ∈ {up, down}``) are emitted. The
``outside`` intervals in ``gt.csv`` are used to extract the pre/post
windows but are otherwise discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.data.loader import (
    classify_experiment_type,
    getExperimentData,
    list_experiments,
)


@dataclass
class SegmentRecord:
    """One labelled elevator segment ready for prediction."""
    exp_name: str
    experimenter: str
    phone: str
    location: str
    seg_idx: int                     # index within the experiment's gt.csv
    ride_type: str                   # 'up' or 'down'
    start_ms: int
    end_ms: int
    true_dh: float
    signal_clear: bool
    experiment_type: str             # 'train' or 'test'

    # Data slices
    acc: pd.DataFrame = field(repr=False)
    pre_acc: pd.DataFrame = field(repr=False)
    post_acc: pd.DataFrame = field(repr=False)

    # Derived
    duration_sec: float = 0.0


def _slice_acc(acc: pd.DataFrame, s: int, e: int) -> pd.DataFrame:
    return acc[(acc["timestamp_ms"] >= s) & (acc["timestamp_ms"] < e)].reset_index(drop=True)


def build_segment_records(
    exp_name: str,
    pre_ms: int = 3000,
    post_ms: int = 3000,
    min_segment_samples: int = 20,
    verbose: bool = False,
) -> list[SegmentRecord]:
    """Flatten one experiment's elevator intervals into ``SegmentRecord``s.

    ``pre_ms`` and ``post_ms`` are the stationary windows before / after
    the segment used for gravity calibration. They are clipped to the
    available data if the experiment starts / ends inside an elevator
    ride. Segments with fewer than ``min_segment_samples`` ACC rows are
    dropped.
    """
    try:
        sensors, gt, meta = getExperimentData(exp_name)
    except Exception as e:
        if verbose:
            print(f"  [skip] {exp_name}: {type(e).__name__}: {e}")
        return []

    if "ACC" not in sensors or sensors["ACC"].empty:
        return []

    acc = sensors["ACC"]
    exp_type = classify_experiment_type(exp_name)

    out: list[SegmentRecord] = []
    for idx, row in gt.iterrows():
        if row["type"] not in ("up", "down"):
            continue
        s = int(row["start_ms"]); e = int(row["end_ms"])
        ride = _slice_acc(acc, s, e)
        if len(ride) < min_segment_samples:
            continue
        pre = _slice_acc(acc, s - pre_ms, s)
        post = _slice_acc(acc, e, e + post_ms)

        rec = SegmentRecord(
            exp_name=exp_name,
            experimenter=str(meta.get("experimenter", "")),
            phone=str(meta.get("phone", "")),
            location=str(meta.get("location", "")),
            seg_idx=int(idx),
            ride_type=str(row["type"]),
            start_ms=s, end_ms=e,
            true_dh=float(row["height_diff_m"]) if row["height_diff_m"] is not None else float("nan"),
            signal_clear=bool(row["signalClearRecording"]),
            experiment_type=exp_type,
            acc=ride, pre_acc=pre, post_acc=post,
            duration_sec=float((e - s) / 1000.0),
        )
        out.append(rec)
    return out


def load_all_segments(
    experiments: Optional[Iterable[str]] = None,
    verbose: bool = False,
    **kwargs,
) -> list[SegmentRecord]:
    """Load + flatten every experiment (or a subset)."""
    if experiments is None:
        experiments = list_experiments()
    records: list[SegmentRecord] = []
    for name in experiments:
        recs = build_segment_records(name, verbose=verbose, **kwargs)
        if verbose:
            print(f"  {name}: +{len(recs)} segments")
        records.extend(recs)
    return records
