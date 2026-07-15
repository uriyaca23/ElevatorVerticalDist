"""Every malformed-input path raises its specific, actionable exception."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyramidElevatorDist as P

from .conftest import make_gyro, make_ride_acc


def test_not_a_dataframe():
    with pytest.raises(P.InputTypeError) as ei:
        P.findSegments([1, 2, 3])
    assert "findSegments" in str(ei.value)
    assert "'acc'" in str(ei.value)
    assert isinstance(ei.value, P.PyramidElevatorDistError)
    assert isinstance(ei.value, TypeError)


def test_missing_column():
    acc = make_ride_acc().drop(columns=["z"])
    with pytest.raises(P.MissingColumnsError) as ei:
        P.findSegments(acc)
    msg = str(ei.value)
    assert "'z'" in msg and "timestamp_ms, x, y, z" in msg
    assert ei.value.missing == frozenset({"z"})


@pytest.mark.parametrize("call", [
    lambda acc: P.findSegments(acc),
    lambda acc: P.findSegmentsDetailed(acc),
    lambda acc: P.findSegmentParameters(acc, 1.0, 2.0),
    lambda acc: P.predictSegment(
        acc, {"type": "up", "start_s": 1.0, "end_s": 2.0}),
    lambda acc: P.reconstructedSignal(acc),
])
def test_empty_acc_raises(call):
    with pytest.raises(P.EmptyInputError):
        call(make_ride_acc().iloc[0:0])


def test_empty_prs_barometric_altitude():
    with pytest.raises(P.EmptyInputError):
        P.barometricAltitude(pd.DataFrame(
            {"timestamp_ms": [], "pressure": []}))


def test_bad_dtype():
    acc = make_ride_acc()
    acc["x"] = acc["x"].astype(str)
    with pytest.raises(P.BadDtypeError) as ei:
        P.findSegments(acc)
    assert ei.value.column == "x"
    assert "pd.to_numeric" in str(ei.value)


def test_nan_in_required_column():
    acc = make_ride_acc()
    acc.loc[5:9, "y"] = np.nan
    with pytest.raises(P.NaNValuesError) as ei:
        P.findSegments(acc)
    assert ei.value.column == "y"
    assert ei.value.count == 5
    assert "dropna" in str(ei.value)


def test_unsorted_timestamps():
    acc = make_ride_acc().iloc[::-1].reset_index(drop=True)
    with pytest.raises(P.NonMonotonicTimestampsError) as ei:
        P.findSegments(acc)
    assert "sort_values" in str(ei.value)


def test_gyro_empty_is_tolerated(ride_acc):
    """Empty gyro frame ≡ no gyro (the editor passes sensors.get('GYR'))."""
    empty_gyro = make_gyro(0.0)
    a = P.findSegments(ride_acc, gyro=empty_gyro)
    b = P.findSegments(ride_acc, gyro=None)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]


def test_gyro_missing_columns_raises(ride_acc):
    bad_gyro = make_gyro().drop(columns=["z"])
    with pytest.raises(P.MissingColumnsError) as ei:
        P.findSegments(ride_acc, gyro=bad_gyro)
    assert "'gyro'" in str(ei.value)
