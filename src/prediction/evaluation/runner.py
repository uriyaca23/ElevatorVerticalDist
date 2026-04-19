"""Run a ``Predictor`` over a list of ``SegmentRecord``s and collect
:class:`PredictionOutput` + :class:`CalibrationSample` entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from ..algorithms import CalibrationSample, PredictionOutput, Predictor
from .dataset import SegmentRecord


@dataclass
class RecordPrediction:
    """One (record, prediction) pair + the per-row metrics we want to
    sort and filter on later.
    """
    record: SegmentRecord
    output: PredictionOutput

    @property
    def abs_error(self) -> float:
        return abs(self.output.height_diff - self.record.true_dh)

    @property
    def covered(self) -> bool:
        return self.abs_error <= self.output.ci_half_width


def run_predictions(
    predictor: Predictor,
    records: Iterable[SegmentRecord],
    verbose: bool = False,
) -> list[RecordPrediction]:
    preds: list[RecordPrediction] = []
    for i, rec in enumerate(records):
        try:
            out = predictor.predict(
                rec.acc,
                phone_model=rec.phone,
                pre=rec.pre_acc,
                post=rec.post_acc,
            )
        except Exception as e:
            out = PredictionOutput(
                height_diff=0.0, ci_half_width=float("inf"),
                theoretical_sigma=float("inf"), accepted=False,
                quality_score=10.0,
                reject_reason=f"exception_{type(e).__name__}",
                meta={"error": str(e)},
            )
        preds.append(RecordPrediction(record=rec, output=out))
        if verbose and (i + 1) % 50 == 0:
            print(f"    predicted {i + 1}/{len(list(records)) if hasattr(records, '__len__') else '?'}")
    return preds


def collect_predictions(preds: list[RecordPrediction]) -> pd.DataFrame:
    """Flatten a list of ``RecordPrediction`` into a tidy DataFrame.

    The DataFrame is *the* shape we hand to the metric functions and
    the plot functions. One row per segment.
    """
    rows = []
    for rp in preds:
        rec = rp.record; out = rp.output
        rows.append({
            "exp_name": rec.exp_name,
            "experimenter": rec.experimenter,
            "phone": rec.phone,
            "location": rec.location,
            "seg_idx": rec.seg_idx,
            "ride_type": rec.ride_type,
            "start_ms": rec.start_ms, "end_ms": rec.end_ms,
            "duration_sec": rec.duration_sec,
            "true_dh": rec.true_dh,
            "signal_clear": rec.signal_clear,
            "experiment_type": rec.experiment_type,
            "pred_dh": out.height_diff,
            "ci_half_width": out.ci_half_width,
            "theoretical_sigma": out.theoretical_sigma,
            "accepted": out.accepted,
            "quality_score": out.quality_score,
            "reject_reason": out.reject_reason,
            "abs_error": abs(out.height_diff - rec.true_dh),
            "covered": abs(out.height_diff - rec.true_dh) <= out.ci_half_width,
        })
    return pd.DataFrame(rows)


def to_calibration_samples(preds: list[RecordPrediction]) -> list[CalibrationSample]:
    return [
        CalibrationSample(
            predicted_dh=rp.output.height_diff,
            true_dh=rp.record.true_dh,
            theoretical_sigma=rp.output.theoretical_sigma,
            accepted=rp.output.accepted,
            quality_score=rp.output.quality_score,
            signal_clear=rp.record.signal_clear,
            exp_name=rp.record.exp_name,
            segment_idx=rp.record.seg_idx,
        )
        for rp in preds
    ]
