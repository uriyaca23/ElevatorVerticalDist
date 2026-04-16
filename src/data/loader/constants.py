"""Constants shared by the loader submodules.

Project layout:

    src/data/
    ├── rawData/                        (raw sensor logs, grouped per experiment)
    │   └── <exp_name>/
    │       ├── sensorLog_*.txt
    │       ├── metadata.txt
    │       └── forBarometer/           (optional: secondary device for barometer)
    ├── structuredData/                 (processed, CSV-based artifacts)
    │   ├── metadata.csv                (index of all experiments)
    │   └── data/<exp_name>/
    │       ├── <SENSOR>.csv            (one per sensor present in the raw log)
    │       ├── gt.csv                  (ground-truth intervals)
    │       ├── metadata.csv            (single row, same schema as the index)
    │       ├── baramoshka.csv          (floor → height map, populated later)
    │       └── forBarometer_alignment.png  (diagnostic plot if applicable)
    └── (achive)/                       (legacy experimenter folders, left as-is)
"""

from __future__ import annotations

from pathlib import Path


# Parent is `src/data/` (this file lives at `src/data/loader/constants.py`).
_DATA_DIR = Path(__file__).resolve().parents[1]

# Raw sensor logs (1 folder per experiment, sensorLog_*.txt + metadata.txt).
RAW_DATA_ROOT = _DATA_DIR / "rawData"

# Processed CSV artifacts.
STRUCTURED_ROOT = _DATA_DIR / "structuredData"
STRUCTURED_DATA_DIR = STRUCTURED_ROOT / "data"
STRUCTURED_INDEX_CSV = STRUCTURED_ROOT / "metadata.csv"

# Legacy rawData path (renamed by the user). Kept pointing to something sane so
# `load_experimenter('eyal')` etc. don't crash outright; callers will still get
# FileNotFoundError if that directory doesn't exist.
DATA_ROOT = _DATA_DIR / "(achive)"

# Per-experiment filenames (raw side).
METADATA_FILENAME = "metadata.txt"
FOR_BAROMETER_SUBDIR = "forBarometer"

# Per-experiment filenames (structured side).
GT_CSV = "gt.csv"
METADATA_CSV = "metadata.csv"
BAROMOSHKA_CSV = "baramoshka.csv"
FOR_BAROMETER_PLOT_FILENAME = "forBarometer_alignment.png"

# Legacy filenames produced by `loadDataWithGT` — kept so legacy code still
# knows where to write, even though the new flow doesn't use them.
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

# CSV schemas.
METADATA_COLUMNS = [
    "exp_name", "experimenter", "phone", "location",
    "description", "date", "time",
]
BAROMOSHKA_COLUMNS = ["floor", "height"]
GT_COLUMNS = ["start_ms", "end_ms", "type", "description", "signalClearRecording"]
