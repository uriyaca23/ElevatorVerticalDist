"""Typed results of the public segmentation API.

Field sets mirror the detector core's dicts one-to-one (see
``pair_filter.predict_pairs`` and ``segmentor._detail_from_state``) — the
wrappers convert at the boundary, the algorithms are untouched.
"""
from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from .common import FloatArray1D, FloatArray2D, PyramidModel

__all__ = [
    "LobeFit",
    "Heatmaps",
    "CorrelationCurves",
    "SegmentDetail",
    "RideSegment",
    "DetailedRideSegment",
]


class LobeFit(PyramidModel):
    """Best-matching trapezoid for one acceleration lobe."""

    t_c: float             #: lobe centre, seconds (trace-relative)
    a_peak: float          #: SIGNED peak amplitude, m/s²
    half_width_s: float    #: trapezoid half-width W, seconds
    frac_flat: float       #: plateau fraction f in [0, 1]
    r2_local: float        #: local R² over the ±W window (NaN when unknown)


class Heatmaps(PyramidModel):
    """Per-lobe R² heatmaps over the (W, f) template grid."""

    lobe1: FloatArray2D
    lobe2: FloatArray2D
    grid_w_s: FloatArray1D
    grid_f: FloatArray1D

    @model_validator(mode="after")
    def _shapes_match(self) -> "Heatmaps":
        want = (self.grid_w_s.size, self.grid_f.size)
        for tag in ("lobe1", "lobe2"):
            got = getattr(self, tag).shape
            if got != want:
                raise ValueError(
                    f"heatmaps.{tag} has shape {got} but the grids imply "
                    f"{want} (len(grid_w_s), len(grid_f))"
                )
        return self


class CorrelationCurves(PyramidModel):
    """Whole-trace best positive/negative template R² curves."""

    t: FloatArray1D
    best_pos_r2: FloatArray1D
    best_neg_r2: FloatArray1D

    @model_validator(mode="after")
    def _lengths_match(self) -> "CorrelationCurves":
        n = self.t.size
        if self.best_pos_r2.size != n or self.best_neg_r2.size != n:
            raise ValueError(
                f"correlation arrays must share one length, got "
                f"t={n}, best_pos_r2={self.best_pos_r2.size}, "
                f"best_neg_r2={self.best_neg_r2.size}"
            )
        return self


class SegmentDetail(PyramidModel):
    """Full diagnostics for one marked interval — what the editor shows."""

    ride_type: Literal["up", "down"]
    t_start_s: float
    t_end_s: float
    lobe1: LobeFit
    lobe2: LobeFit
    joint_r2_mean: float       #: NaN when the detector had no joint fit
    heatmap_energy: float      #: NaN when the detector had no joint fit
    heatmaps: Heatmaps
    correlation: CorrelationCurves


class RideSegment(PyramidModel):
    """One detected elevator ride."""

    index: int                 #: position in the returned list
    ride_type: Literal["up", "down"]
    t_start_s: float           #: seconds, relative to acc[0]
    t_end_s: float
    duration_s: float
    lobe1: LobeFit
    lobe2: LobeFit
    joint_r2_mean: float       #: shared-shape mean R² across both lobes
    heatmap_energy: float      #: grid support of the match


class DetailedRideSegment(RideSegment):
    """A :class:`RideSegment` plus its precomputed editor detail.

    ``detail`` is ``None`` for the rare ride whose window yields no usable
    +/- lobe pair.
    """

    detail: SegmentDetail | None = None
