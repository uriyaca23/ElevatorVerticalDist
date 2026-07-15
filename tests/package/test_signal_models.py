"""reconstructedSignal / displaySeries / barometricAltitude contracts."""
from __future__ import annotations

import numpy as np
import pytest

import pyramidElevatorDist as P

from .conftest import make_gyro, make_prs, nan_equal


def test_reconstructed_signal_schema(ride_acc):
    sig = P.reconstructedSignal(ride_acc)
    assert list(sig.columns) == ["timestamp_ms", "a_vert", "a_mag_g"]
    assert len(sig) > 0
    ts = sig["timestamp_ms"].to_numpy(dtype=float)
    assert np.all(np.diff(ts) >= 0)


def test_display_series_namedtuple(ride_acc):
    sig = P.reconstructedSignal(ride_acc)
    out = P.displaySeries(sig)
    values, label, reconstructed = out      # 3-tuple unpacking preserved
    assert values is out.values and label == out.label
    assert label == "|a|-g" and reconstructed is False
    assert values.size == len(sig)

    out2 = P.displaySeries(sig, has_gyro=True,
                           reconstruct=P.RECONSTRUCT_CHOICES[-1])
    assert out2.label == "a_z reconstructed" and out2.reconstructed is True


def test_display_series_tolerates_none_and_empty(ride_acc):
    """None/empty sig is documented tolerated behavior (pin)."""
    for sig in (None, P.reconstructedSignal(ride_acc).iloc[0:0]):
        out = P.displaySeries(sig)
        assert out.values.size == 0
        assert out.label == "|a|-g"


def test_unknown_reconstruct_raises_even_without_gyro(ride_acc):
    """A typo'd method name can no longer silently no-op."""
    with pytest.raises(P.UnknownReconstructError) as ei:
        P.reconstructedSignal(ride_acc, gyro=None, reconstruct="madgwick")
    assert "case-sensitive" in str(ei.value)
    with pytest.raises(P.UnknownReconstructError):
        P.findSegments(ride_acc, reconstruct="bogus")


def test_valid_reconstruct_without_gyro_is_noop(ride_acc):
    """A real method name with no gyro stays a no-op (behavior pin the
    Streamlit UX depends on)."""
    method = P.RECONSTRUCT_CHOICES[-1]
    assert method != "none"
    a = P.reconstructedSignal(ride_acc, gyro=None, reconstruct=method)
    b = P.reconstructedSignal(ride_acc, reconstruct="none")
    assert nan_equal(a["a_vert"].to_numpy(), b["a_vert"].to_numpy())


def test_reconstruct_with_gyro_runs(ride_acc):
    method = P.RECONSTRUCT_CHOICES[-1]
    sig = P.reconstructedSignal(ride_acc, gyro=make_gyro(),
                                reconstruct=method)
    assert list(sig.columns) == ["timestamp_ms", "a_vert", "a_mag_g"]
    assert len(sig) > 0


def test_barometric_altitude(ride_acc):
    alt = P.barometricAltitude(make_prs(dh_m=3.0))
    assert list(alt.columns) == ["timestamp_ms", "altitude_m"]
    h = alt["altitude_m"].to_numpy()
    # Pressure drops across the ride → altitude rises (sign pin).
    assert h[-1] - h[0] == pytest.approx(3.0, abs=1.0)
