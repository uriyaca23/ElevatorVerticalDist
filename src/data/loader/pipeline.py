"""Pipeline loader: the main entry points that downstream code uses.

`getExperimentRawParsed` returns a dict of per-sensor DataFrames (with
forBarometer alignment if applicable). `getExperimentPipelineData` extends
that with barometer-derived GT intervals wrapped in an `ExperimentPipeline`
container that yields `(data_slice, gt_row, metaData)` tuples when iterated.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter,
)
import pandas as pd

from .alignment import _merge_secondary_prs
from .constants import (
    FOR_BAROMETER_PLOT_FILENAME,
    FOR_BAROMETER_SUBDIR,
    METADATA_FILENAME,
    PIPELINE_CACHE_FILENAME,
)
from .parsing import (
    _find_sensor_log,
    _parse_metadata_file,
    _parse_sensor_log,
)


def getExperimentRawParsed(exp_path: Path | str) -> dict[str, pd.DataFrame]:
    """Parse a structuredData experiment's sensorLog into per-sensor DataFrames.

    If `exp_path/forBarometer/` exists and has a `sensorLog_*.txt`, the primary's
    PRS frame is replaced with the secondary device's PRS re-timestamped onto the
    primary's uptime timebase (filename ISO offset + ACC cross-correlation).
    A diagnostic `forBarometer_alignment.png` is written to `exp_path`.
    """
    exp_path = Path(exp_path)
    primary_log = _find_sensor_log(exp_path)
    frames = _parse_sensor_log(primary_log)

    fb_dir = exp_path / FOR_BAROMETER_SUBDIR
    if fb_dir.is_dir():
        try:
            secondary_log = _find_sensor_log(fb_dir)
        except FileNotFoundError:
            print(f"[loader] forBarometer/ exists but has no sensorLog: {fb_dir}")
        else:
            plot_path = exp_path / FOR_BAROMETER_PLOT_FILENAME
            frames = _merge_secondary_prs(frames, primary_log, secondary_log, plot_path)

    return frames


def _segments_to_full_gt(
    segments: pd.DataFrame, t0_ms: int, t_end_ms: int,
) -> pd.DataFrame:
    """Convert segmenter output to alternating `[start_ms, end_ms, type]` rows
    covering `[t0_ms, t_end_ms]` with 'outside' filler between rides."""
    rides: list[dict] = []
    for _, row in segments.iterrows():
        s_lo, _ = row["start_ci"]
        _, e_hi = row["end_ci"]
        rides.append({
            "start_ms": int(t0_ms + float(s_lo) * 1000),
            "end_ms": int(t0_ms + float(e_hi) * 1000),
            "type": str(row["type"]),
        })
    rides.sort(key=lambda r: r["start_ms"])

    for r in rides:
        r["start_ms"] = max(r["start_ms"], t0_ms)
        r["end_ms"] = min(r["end_ms"], t_end_ms)
    rides = [r for r in rides if r["end_ms"] > r["start_ms"]]

    out: list[dict] = []
    cursor = t0_ms
    for r in rides:
        if r["start_ms"] < cursor:
            r["start_ms"] = cursor
            if r["end_ms"] <= cursor:
                continue
        if r["start_ms"] > cursor:
            out.append({"start_ms": cursor, "end_ms": r["start_ms"], "type": "outside"})
        out.append(r)
        cursor = r["end_ms"]

    if cursor < t_end_ms:
        out.append({"start_ms": cursor, "end_ms": t_end_ms, "type": "outside"})
    if not out:
        out.append({"start_ms": t0_ms, "end_ms": t_end_ms, "type": "outside"})

    return pd.DataFrame(out, columns=["start_ms", "end_ms", "type"])


@dataclass
class ExperimentPipeline:
    """Container for a fully preprocessed experiment.

    Attributes:
        data: per-sensor DataFrames covering the whole experiment.
        gt: alternating intervals with columns `start_ms`, `end_ms`, `type`
            ('up' | 'down' | 'outside'). Covers the full timeline with no gaps.
        metaData: parsed `metadata.txt` key/value pairs.

    Iterating yields `(data_slice_dict, gt_row, metaData)` per interval, where
    `data_slice_dict[sensor]` is that sensor's frame sliced to the interval.
    """
    data: dict[str, pd.DataFrame]
    gt: pd.DataFrame
    metaData: dict[str, str]

    def __iter__(self) -> Iterator[tuple[dict[str, pd.DataFrame], pd.Series, dict[str, str]]]:
        for _, row in self.gt.iterrows():
            s, e = int(row["start_ms"]), int(row["end_ms"])
            slice_dict = {
                name: df[(df["timestamp_ms"] >= s) & (df["timestamp_ms"] < e)]
                      .reset_index(drop=True)
                for name, df in self.data.items()
            }
            yield slice_dict, row, self.metaData

    def __len__(self) -> int:
        return len(self.gt)


def getExperimentPipelineData(
    exp_path: Path | str, use_cache: bool = True,
) -> ExperimentPipeline:
    """Build (or load cached) ExperimentPipeline for an experiment folder.

    Writes `pipeline_data.pkl` inside `exp_path` on first build. On corrupt
    cache falls back to rebuild.
    """
    exp_path = Path(exp_path)
    cache_path = exp_path / PIPELINE_CACHE_FILENAME

    if use_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, ExperimentPipeline):
                return loaded
            print(f"[loader] cache at {cache_path} is not an ExperimentPipeline; rebuilding")
        except (pickle.UnpicklingError, EOFError, AttributeError,
                ModuleNotFoundError, ImportError) as e:
            print(f"[loader] cache load failed ({type(e).__name__}: {e}); rebuilding")

    data = getExperimentRawParsed(exp_path)

    if "PRS" in data and not data["PRS"].empty:
        prs = data["PRS"]
        t0_ms = int(prs["timestamp_ms"].iloc[0])
        t_end_ms = int(prs["timestamp_ms"].iloc[-1])

        t_sec = (prs["timestamp_ms"].to_numpy(dtype=float) - t0_ms) / 1000.0
        h = prs["GT_height_m"].to_numpy(dtype=float)
        h_smooth = (pd.Series(h).rolling(window=51, center=True, min_periods=1)
                                 .median().to_numpy())
        height_frame = pd.DataFrame({"time": t_sec, "height": h_smooth})

        cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)
        segments = Segmenter(cfg).detect(height_frame)
        gt = _segments_to_full_gt(segments, t0_ms, t_end_ms)
    else:
        if "ACC" not in data or data["ACC"].empty:
            raise ValueError(f"No PRS or ACC data in {exp_path}; cannot build pipeline")
        acc = data["ACC"]
        t0_ms = int(acc["timestamp_ms"].iloc[0])
        t_end_ms = int(acc["timestamp_ms"].iloc[-1])
        gt = pd.DataFrame(
            [{"start_ms": t0_ms, "end_ms": t_end_ms, "type": "outside"}],
            columns=["start_ms", "end_ms", "type"],
        )

    metadata = _parse_metadata_file(exp_path / METADATA_FILENAME)
    pipeline = ExperimentPipeline(data=data, gt=gt, metaData=metadata)

    try:
        with cache_path.open("wb") as f:
            pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"[loader] cache write failed: {type(e).__name__}: {e}")

    return pipeline
