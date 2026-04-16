"""Sensor log loader.

Two entry-point families live here:

1. Legacy `rawData/<name>/<expN>/` flow — see :mod:`.legacy`:
    * :func:`loadBasicData(name, exp)` → per-sensor DataFrames.
    * :func:`loadDataWithGT(name, exp)` → same, with `gt_label` on PRS,
      cached to ``<exp>/data_with_gt.xlsx``.
    * :func:`load_experimenter(name)` → first-available experiment, used by
      most scripts.

2. Pipeline `structuredData/<folder>/` flow — see :mod:`.pipeline`:
    * :func:`getExperimentRawParsed(exp_path)` → per-sensor DataFrames,
      with `forBarometer/` aligned onto the primary's timebase if present.
    * :func:`getExperimentPipelineData(exp_path)` → :class:`ExperimentPipeline`
      with `.data`, `.gt`, `.metaData`. Iterable as
      ``for data_slice, gt_row, meta in pipeline:``.

Raw sensorLog file format (tab-separated, no header), one sample per line:
    <timestamp_ms>\\t<SENSOR>\\t<value1>\\t<value2>...
"""

from __future__ import annotations

# Public API re-exports.
from .constants import (
    DATA_ROOT,
    FOR_BAROMETER_PLOT_FILENAME,
    FOR_BAROMETER_SUBDIR,
    GT_FILENAME,
    GT_PLOT_FILENAME,
    METADATA_FILENAME,
    PIPELINE_CACHE_FILENAME,
    SENSOR_COLUMNS,
    STRUCTURED_ROOT,
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
    getExperimentPipelineData,
    getExperimentRawParsed,
)

__all__ = [
    # Constants
    "DATA_ROOT",
    "STRUCTURED_ROOT",
    "PIPELINE_CACHE_FILENAME",
    "FOR_BAROMETER_SUBDIR",
    "FOR_BAROMETER_PLOT_FILENAME",
    "METADATA_FILENAME",
    "GT_FILENAME",
    "GT_PLOT_FILENAME",
    "SENSOR_COLUMNS",
    # Pipeline (new)
    "ExperimentPipeline",
    "getExperimentRawParsed",
    "getExperimentPipelineData",
    # Legacy
    "loadBasicData",
    "loadDataWithGT",
    "load_experimenter",
]
