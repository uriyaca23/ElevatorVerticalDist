"""Constants shared by the loader submodules.

File layout and filenames used by both the legacy `rawData/` flow and the
newer `structuredData/` pipeline live here so the other modules stay lean.
"""

from __future__ import annotations

from pathlib import Path

# Parent is `src/data/` (this file lives at `src/data/loader/constants.py`).
DATA_ROOT = Path(__file__).resolve().parents[1] / "rawData"
STRUCTURED_ROOT = Path(__file__).resolve().parents[1] / "structuredData"

# Filenames produced / consumed by the pipeline loader.
PIPELINE_CACHE_FILENAME = "pipeline_data.pkl"
FOR_BAROMETER_SUBDIR = "forBarometer"
FOR_BAROMETER_PLOT_FILENAME = "forBarometer_alignment.png"
METADATA_FILENAME = "metadata.txt"

# Filenames produced by the legacy GT flow.
GT_FILENAME = "data_with_gt.xlsx"
GT_PLOT_FILENAME = "gt_plot.png"

# Schema for each sensor line in a sensorLog_*.txt. Rows whose column count
# doesn't match are dropped silently during parsing.
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
