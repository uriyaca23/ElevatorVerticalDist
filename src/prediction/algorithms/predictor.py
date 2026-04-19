"""Prediction algorithm dispatch.

:class:`Predictor` is configured via :class:`PREDICT_ALGORITHM_CONFIG`
(see ``class.py``) which selects an algorithm and loads its hyperparameters
from ``config.json``. The ``forward`` method takes a segment DataFrame and
returns the predicted height difference (meters).
"""

from __future__ import annotations

import importlib

import pandas as pd

from .barometer_only import predict_height_difference_from_barometer

_config_mod = importlib.import_module(__package__ + ".class")
PREDICT_ALGORITHM_CONFIG = _config_mod.PREDICT_ALGORITHM_CONFIG
PredictAlgorithm = _config_mod.PredictAlgorithm
BarometerHeightDiffConfig = _config_mod.BarometerHeightDiffConfig


class Predictor:
    def __init__(self, config: PREDICT_ALGORITHM_CONFIG):
        self.config = config
        self.params = config.load_params()

    def forward(self, data: pd.DataFrame) -> float:
        if self.config.algorithm is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
            algo_config = BarometerHeightDiffConfig(**self.params)
            return predict_height_difference_from_barometer(data, algo_config)
        raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
