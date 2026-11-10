"""Evaluation harness for the prediction algorithms.

``run_evaluation.py`` at the repo root is the legacy pipeline-v4
entry point. This module hosts the new harness tailored to the
:class:`Predictor` class, train/test split via the experiment
metadata, per-segment conformal calibration, and a blind-test
convention where the test pass is produced only once at the end.
"""

from .dataset import (
    SegmentRecord,
    build_segment_records,
    load_all_segments,
)
from .metrics import compute_metrics, MetricsBundle
from .runner import run_predictions, collect_predictions

__all__ = [
    "SegmentRecord",
    "build_segment_records",
    "load_all_segments",
    "compute_metrics",
    "MetricsBundle",
    "run_predictions",
    "collect_predictions",
]
