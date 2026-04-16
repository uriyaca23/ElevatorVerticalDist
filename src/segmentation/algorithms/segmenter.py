"""Segmentation algorithm dispatch.

:class:`Segmenter` is configured via :class:`SEGMENT_ALGORITHM_CONFIG`
(see ``class.py``) which selects an algorithm and loads its hyperparameters
from ``config.json``.
"""

from __future__ import annotations

import importlib

import pandas as pd

from .barometer_only import detect_elevator_segments_from_height
from .accelerometer_only import detect_elevator_segments_from_acc
from .template_match import detect_elevator_segments_from_template_match

_config_mod = importlib.import_module(__package__ + ".class")
SEGMENT_ALGORITHM_CONFIG = _config_mod.SEGMENT_ALGORITHM_CONFIG
SegmentAlgorithm = _config_mod.SegmentAlgorithm
PressureFilterConfig = _config_mod.PressureFilterConfig
AccOnlyConfig = _config_mod.AccOnlyConfig
TemplateMatchConfig = _config_mod.TemplateMatchConfig


class Segmenter:
    def __init__(self, config: SEGMENT_ALGORITHM_CONFIG):
        self.config = config
        self.params = config.load_params()

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.config.algorithm is SegmentAlgorithm.PRESSURE_FILTER:
            algo_config = PressureFilterConfig(**self.params)
            return detect_elevator_segments_from_height(data, algo_config)
        if self.config.algorithm is SegmentAlgorithm.ACC_ONLY:
            algo_config = AccOnlyConfig(**self.params)
            return detect_elevator_segments_from_acc(data, algo_config)
        if self.config.algorithm is SegmentAlgorithm.ACC_TEMPLATE_MATCH:
            algo_config = TemplateMatchConfig(**self.params)
            return detect_elevator_segments_from_template_match(data, algo_config)
        raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
