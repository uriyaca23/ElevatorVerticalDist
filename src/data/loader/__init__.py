"""Sensor log loader.

Two entry-point families live here:

1. Legacy `<DATA_ROOT>/<name>/<expN>/` flow — see :mod:`.legacy`:
    * :func:`loadBasicData(name, exp)` → per-sensor DataFrames.
    * :func:`loadDataWithGT(name, exp)` → same, with `gt_label` on PRS,
      cached to ``<exp>/data_with_gt.xlsx``.
    * :func:`load_experimenter(name)` → first-available experiment, used by
      most scripts.

2. Pipeline CSV flow — see :mod:`.pipeline`:
    * :func:`getExperimentRawParsed(exp)` → per-sensor DataFrames from a raw
      sensorLog (with `forBarometer/` aligned onto the primary's timebase if
      present).
    * :func:`getExperimentData(exp)` → ``(sensors, gt, metadata)`` tuple.
      Reads `structuredData/data/<name>/` CSVs when they exist; otherwise
      parses the raw log and materialises the CSVs. Wrap into
      :class:`ExperimentPipeline` when you want iteration.
    * :func:`saveExperimentData(name, sensors, gt, metadata)` → persist the
      three components back to the structured directory.

Raw sensorLog file format (tab-separated, no header), one sample per line:
    <timestamp_ms>\\t<SENSOR>\\t<value1>\\t<value2>...
"""

from __future__ import annotations

# Public API re-exports.
from .constants import (
    BAROMOSHKA_COLUMNS,
    BAROMOSHKA_CSV,
    DATA_ROOT,
    EXPERIMENT_TYPE_TEST,
    EXPERIMENT_TYPE_TRAIN,
    EXPERIMENT_TYPES,
    FOR_BAROMETER_PLOT_FILENAME,
    FOR_BAROMETER_SUBDIR,
    GAP_THRESHOLD_S,
    GT_COLUMNS,
    GT_CSV,
    GT_FILENAME,
    GT_PLOT_FILENAME,
    METADATA_COLUMNS,
    METADATA_CSV,
    METADATA_FILENAME,
    RAW_DATA_ROOT,
    SENSOR_COLUMNS,
    SOURCE_EXPERIMENT,
    SOURCE_IDO,
    SOURCE_REAL_WORLD,
    STRUCTURED_DATA_DIR,
    STRUCTURED_INDEX_CSV,
    STRUCTURED_ROOT,
    THRESHOLD_FREQUENCY_HZ,
    VALID_SOURCES,
)
from .legacy import (
    loadBasicData,
    loadDataWithGT,
    load_experimenter,
)
from .parsing import (
    # Re-exported private helpers — used by tests and some downstream scripts.
    _find_sensor_log,
    _parse_metadata_file,
    _parse_sensor_log,
)
from .pipeline import (
    ExperimentPipeline,
    _detect_valid_intervals as detect_valid_intervals,
    classify_experiment_type,
    getExperimentData,
    getExperimentRawParsed,
    list_experiments,
    list_structured_experiments,
    rebuild_metadata_index,
    saveExperimentData,
)

__all__ = [
    # Constants
    "DATA_ROOT",
    "RAW_DATA_ROOT",
    "STRUCTURED_ROOT",
    "STRUCTURED_DATA_DIR",
    "STRUCTURED_INDEX_CSV",
    "FOR_BAROMETER_SUBDIR",
    "FOR_BAROMETER_PLOT_FILENAME",
    "METADATA_FILENAME",
    "METADATA_CSV",
    "METADATA_COLUMNS",
    "GT_CSV",
    "GT_COLUMNS",
    "BAROMOSHKA_CSV",
    "BAROMOSHKA_COLUMNS",
    "GT_FILENAME",
    "GT_PLOT_FILENAME",
    "SENSOR_COLUMNS",
    "EXPERIMENT_TYPE_TEST",
    "EXPERIMENT_TYPE_TRAIN",
    "EXPERIMENT_TYPES",
    "SOURCE_EXPERIMENT",
    "SOURCE_IDO",
    "SOURCE_REAL_WORLD",
    "VALID_SOURCES",
    "THRESHOLD_FREQUENCY_HZ",
    "GAP_THRESHOLD_S",
    # Pipeline
    "ExperimentPipeline",
    "classify_experiment_type",
    "detect_valid_intervals",
    "getExperimentRawParsed",
    "getExperimentData",
    "saveExperimentData",
    "list_experiments",
    "list_structured_experiments",
    "rebuild_metadata_index",
    # Legacy
    "loadBasicData",
    "loadDataWithGT",
    "load_experimenter",
]
