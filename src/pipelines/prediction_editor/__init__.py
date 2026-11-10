"""Presentation-only Tk prediction editor.

Launch with ``python -m src.pipelines.prediction_editor [exp]``. All
computation is delegated to the ``pyramidElevatorDist`` package; the modules
here own only Tk / matplotlib / numpy layout and data loading.
"""
from __future__ import annotations

from src.pipelines.prediction_editor.app import PredictionEditor, main

__all__ = ["PredictionEditor", "main"]
