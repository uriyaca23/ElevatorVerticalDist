"""Stay↔movement switches parsed out of a GPS stay segmentation.

Synthetic and package-only: this suite never imports the GPS stay detector,
and :func:`test_parser_does_not_load_gps_stay` enforces that the parser does
not either. The fixtures reproduce two committed detector outputs — a plain
stay/movement/stay transit, and a track whose last two stays sit side by side
with no movement between them.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from types import SimpleNamespace

import pandas as pd
import pytest

from pyramidElevatorDist.utils.findPossibleHouseEntrance import (
    MOVE_TO_STAY,
    SAME_PLACE_RADIUS_FACTOR,
    STAY_TO_MOVE,
    HouseEntranceCandidates,
    findPossibleHouseEntrance,
    haversine_m,
)

T0 = datetime(2024, 3, 1, 8, 0, 0)   # naive, holding a UTC instant
CADENCE = timedelta(seconds=10)
T0_MS = 1_709_280_000_000            # 2024-03-01T08:00:00Z


def to_ms(moment: datetime) -> int:
    return int(moment.replace(tzinfo=timezone.utc).timestamp() * 1000)


def seg(label, i0, i1, *, lat=32.080000, lon=34.780000, probability=0.97,
        radius_m=50.0):
    """One segment in the detector's own field shape."""
    return SimpleNamespace(
        startTime=T0 + i0 * CADENCE, endTime=T0 + i1 * CADENCE,
        classification=label, centerLat=lat, centerLon=lon,
        radiusMeters=radius_m, probability=probability,
        pointCount=i1 - i0 + 1, startIndex=i0, endIndex=i1,
    )


def as_dict(segment) -> dict:
    return dict(vars(segment))


@pytest.fixture
def transit():
    """stay → movement → stay, the shape of a plain trip between two places."""
    return [
        seg("stay", 0, 79, lon=34.780000, probability=0.971699),
        seg("movement", 80, 119, lon=34.790616, probability=0.832058),
        seg("stay", 120, 199, lon=34.801227, probability=0.971699),
    ]


@pytest.fixture
def adjacent_stays():
    """Two stays 292 m apart with NO movement segment between them.

    Reproduces a committed detector output. The travel was never observed, so
    a label-change-only rule would emit nothing here — yet it is exactly the
    arrival worth running the elevator pipeline around.
    """
    return [
        seg("stay", 0, 98, lon=34.780002, probability=0.980),
        seg("movement", 99, 118, lon=34.781093, probability=0.609),
        seg("stay", 119, 149, lon=34.783097, probability=0.878),
        seg("stay", 150, 249, lon=34.780000, probability=0.980),
    ]


# --------------------------------------------------------------- core walk

def test_transit_yields_one_departure_and_one_arrival(transit):
    result = findPossibleHouseEntrance(transit)

    assert [t.direction for t in result.transitions] == [STAY_TO_MOVE, MOVE_TO_STAY]
    # The switch instant is the first fix of the new state.
    assert result.timestamps == [T0_MS + 800_000, T0_MS + 1_200_000]
    assert result.timestamps == [to_ms(T0 + 80 * CADENCE), to_ms(T0 + 120 * CADENCE)]


def test_switch_is_bracketed_by_the_previous_last_fix(transit):
    depart, arrive = findPossibleHouseEntrance(transit).transitions

    # endTime is the previous segment's LAST fix, so the bracket is one
    # inter-fix interval wide — the switch happened somewhere inside it.
    assert depart.prev_end_timestamp_ms == to_ms(T0 + 79 * CADENCE)
    assert arrive.prev_end_timestamp_ms == to_ms(T0 + 119 * CADENCE)
    for t in (depart, arrive):
        assert t.timestamp_ms - t.prev_end_timestamp_ms == 10_000


def test_each_transition_is_anchored_on_its_stay(transit):
    depart, arrive = findPossibleHouseEntrance(transit).transitions

    # Never the movement centroid, which sits midway along the trip.
    assert (depart.lat, depart.lon) == (32.080000, 34.780000)
    assert (arrive.lat, arrive.lon) == (32.080000, 34.801227)
    assert depart.stay_probability == arrive.stay_probability == 0.971699
    assert depart.stay_duration_s == arrive.stay_duration_s == 790.0


def test_timestamps_track_the_transitions(transit):
    result = findPossibleHouseEntrance(transit)

    assert result.timestamps == [t.timestamp_ms for t in result.transitions]
    assert result.timestamps == sorted(result.timestamps)


def test_empty_and_single_segment_inputs_are_empty(transit):
    assert findPossibleHouseEntrance([]) == HouseEntranceCandidates([], [])
    assert findPossibleHouseEntrance(transit[:1]).timestamps == []


# ------------------------------------------------- adjacent same-label stays

def test_adjacent_stays_far_apart_emit_a_departure_and_an_arrival(adjacent_stays):
    result = findPossibleHouseEntrance(adjacent_stays)

    # stay→movement, movement→stay, then the unobserved relocation: the
    # committed output's last two stays are ~292 m apart with no movement
    # between them, and that relocation is a real leave-and-arrive.
    assert [t.direction for t in result.transitions] == [
        STAY_TO_MOVE, MOVE_TO_STAY, STAY_TO_MOVE, MOVE_TO_STAY,
    ]
    relocation_out, relocation_in = result.transitions[2:]
    assert relocation_out.crossed_unknown and relocation_in.crossed_unknown
    assert relocation_out.lon == 34.783097     # left this stay
    assert relocation_in.lon == 34.780000      # reached this one
    assert relocation_in.timestamp_ms == to_ms(T0 + 150 * CADENCE)


def test_adjacent_stays_at_the_same_place_emit_nothing():
    # 2 * radiusMeters = 100 m is the detector's own "same place" scale.
    near = [seg("stay", 0, 79), seg("stay", 120, 199, lat=32.08054)]
    far = [seg("stay", 0, 79), seg("stay", 120, 199, lat=32.08135)]

    assert haversine_m(32.080000, 34.78, 32.08054, 34.78) < 100.0
    assert haversine_m(32.080000, 34.78, 32.08135, 34.78) > 100.0
    assert findPossibleHouseEntrance(near).timestamps == []
    assert len(findPossibleHouseEntrance(far).transitions) == 2


def test_place_tolerance_is_overridable():
    stays = [seg("stay", 0, 79), seg("stay", 120, 199, lat=32.08135)]  # ~150 m

    assert findPossibleHouseEntrance(stays, place_tolerance_m=500.0).timestamps == []
    assert findPossibleHouseEntrance(stays, place_tolerance_m=10.0).transitions


def test_consecutive_movement_segments_emit_nothing():
    runs = [seg("movement", 0, 79), seg("movement", 80, 119, lat=32.1)]

    assert findPossibleHouseEntrance(runs).timestamps == []


def test_same_place_scale_follows_the_segment_radius():
    # A detector reporting a wider decision radius widens "same place" with it.
    wide = [seg("stay", 0, 79, radius_m=500.0),
            seg("stay", 120, 199, lat=32.08135, radius_m=500.0)]

    assert SAME_PLACE_RADIUS_FACTOR == 2.0
    assert findPossibleHouseEntrance(wide).timestamps == []


# ------------------------------------------------------------------ unknown

def test_unknown_is_transparent_by_default_and_flags_the_gap():
    seq = [seg("stay", 0, 79),
           seg("unknown", 80, 89, probability=0.0),
           seg("movement", 90, 119, lat=32.005)]

    result = findPossibleHouseEntrance(seq)

    assert [t.direction for t in result.transitions] == [STAY_TO_MOVE]
    assert result.transitions[0].crossed_unknown is True
    # The bracket widens to the whole unobserved hole.
    assert result.transitions[0].prev_end_timestamp_ms == to_ms(T0 + 79 * CADENCE)
    assert result.transitions[0].timestamp_ms == to_ms(T0 + 90 * CADENCE)


def test_unknown_break_refuses_to_pair_across_the_gap():
    seq = [seg("stay", 0, 79),
           seg("unknown", 80, 89, probability=0.0),
           seg("movement", 90, 119, lat=32.005)]

    assert findPossibleHouseEntrance(seq, unknown="break").timestamps == []


def test_empty_index_range_filler_is_not_rejected():
    # A clipped hole carries a deliberately empty row range and an endTime
    # that precedes its startTime. Indices are ignored; this must not raise.
    filler = SimpleNamespace(
        startTime=T0 + 80 * CADENCE, endTime=T0 + 79 * CADENCE,
        classification="unknown", centerLat=32.08, centerLon=34.78,
        radiusMeters=50.0, probability=0.0,
        pointCount=0, startIndex=80, endIndex=79,
    )

    result = findPossibleHouseEntrance(
        [seg("stay", 0, 79), filler, seg("movement", 80, 119, lat=32.005)]
    )

    assert [t.direction for t in result.transitions] == [STAY_TO_MOVE]


def test_all_unknown_input_is_empty():
    assert findPossibleHouseEntrance(
        [seg("unknown", 0, 79, probability=0.0),
         seg("unknown", 80, 119, probability=0.0)]
    ).timestamps == []


# ------------------------------------------------------------ time handling

def test_naive_timestamps_are_read_as_utc_not_local(monkeypatch):
    """A naive datetime holds a UTC instant; reading it as local time shifts
    every result by the machine's offset."""
    if not hasattr(time, "tzset"):
        pytest.skip("TZ manipulation needs time.tzset()")
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        result = findPossibleHouseEntrance(
            [seg("stay", 0, 79), seg("movement", 80, 119, lat=32.005)]
        )
        assert result.timestamps == [T0_MS + 800_000]
    finally:
        monkeypatch.undo()
        time.tzset()


@pytest.mark.parametrize("render", [
    pytest.param(lambda d: d, id="naive-datetime"),
    pytest.param(lambda d: d.replace(tzinfo=timezone.utc), id="aware-utc"),
    pytest.param(lambda d: d.replace(tzinfo=timezone(timedelta(hours=3)))
                 + timedelta(hours=3), id="aware-plus-three"),
    pytest.param(lambda d: pd.Timestamp(d), id="pandas-timestamp"),
    pytest.param(lambda d: d.isoformat() + "Z", id="iso-z-suffix"),
    pytest.param(lambda d: int(d.replace(tzinfo=timezone.utc).timestamp() * 1000),
                 id="epoch-ms"),
])
def test_every_accepted_time_spelling_gives_the_same_instant(transit, render):
    retimed = [SimpleNamespace(**{**as_dict(s),
                                  "startTime": render(s.startTime),
                                  "endTime": render(s.endTime)})
               for s in transit]

    assert findPossibleHouseEntrance(retimed).timestamps == [
        T0_MS + 800_000, T0_MS + 1_200_000,
    ]


def test_epoch_seconds_are_refused_rather_than_read_as_1970(transit):
    in_seconds = [SimpleNamespace(**{**as_dict(s),
                                     "startTime": to_ms(s.startTime) // 1000,
                                     "endTime": to_ms(s.endTime) // 1000})
                  for s in transit]

    with pytest.raises(ValueError, match="epoch seconds"):
        findPossibleHouseEntrance(in_seconds)


def test_missing_timestamp_raises_rather_than_wrapping_around(transit):
    broken = SimpleNamespace(**{**as_dict(transit[0]), "endTime": pd.NaT})

    with pytest.raises(ValueError, match="NaT"):
        findPossibleHouseEntrance([broken] + transit[1:])


# --------------------------------------------------------------- input shape

def test_objects_mappings_and_frames_agree(transit):
    near_home = dict(center=(32.080000, 34.780000), radius_m=5000.0)
    rows = [as_dict(s) for s in transit]

    from_objects = findPossibleHouseEntrance(transit, **near_home)
    from_mappings = findPossibleHouseEntrance(rows, **near_home)
    from_frame = findPossibleHouseEntrance(pd.DataFrame(rows), **near_home)

    assert from_objects == from_mappings == from_frame


def test_snake_case_spellings_are_accepted(transit):
    renamed = [{"start_time": s.startTime, "end_time": s.endTime,
                "classification": s.classification, "center_lat": s.centerLat,
                "center_lon": s.centerLon, "radius_meters": s.radiusMeters,
                "probability": s.probability}
               for s in transit]

    assert findPossibleHouseEntrance(renamed).timestamps == [
        T0_MS + 800_000, T0_MS + 1_200_000,
    ]


def test_a_string_enum_label_is_accepted(transit):
    class Classification(StrEnum):
        MOVEMENT = "movement"
        STAY = "stay"

    enumerated = [SimpleNamespace(
        **{**as_dict(s), "classification": Classification(s.classification)})
        for s in transit]

    assert findPossibleHouseEntrance(enumerated).timestamps == [
        T0_MS + 800_000, T0_MS + 1_200_000,
    ]


def test_optional_fields_may_be_absent(transit):
    sparse = [{"startTime": s.startTime, "endTime": s.endTime,
               "classification": s.classification,
               "centerLat": s.centerLat, "centerLon": s.centerLon}
              for s in transit]

    assert findPossibleHouseEntrance(sparse).timestamps == [
        T0_MS + 800_000, T0_MS + 1_200_000,
    ]


def test_unsorted_input_is_sorted(transit):
    assert findPossibleHouseEntrance(list(reversed(transit))).timestamps == [
        T0_MS + 800_000, T0_MS + 1_200_000,
    ]


def test_results_are_immutable(transit):
    from dataclasses import FrozenInstanceError

    transition = findPossibleHouseEntrance(transit).transitions[0]
    with pytest.raises(FrozenInstanceError):
        transition.timestamp_ms = 0


# ---------------------------------------------------------------- proximity

def test_center_filters_to_switches_near_that_place(transit):
    home = (32.080000, 34.780000)          # the first stay

    within_km = findPossibleHouseEntrance(transit, center=home)
    within_100m = findPossibleHouseEntrance(transit, center=home, radius_m=100.0)
    everything = findPossibleHouseEntrance(transit, center=home, radius_m=5000.0)

    # The far stay is ~2 km away: kept only when the radius reaches it.
    assert within_km.timestamps == [T0_MS + 800_000]
    assert within_100m.timestamps == [T0_MS + 800_000]
    assert len(everything.transitions) == 2
    assert everything.transitions[0].distance_from_center_m == pytest.approx(0.0)
    assert everything.transitions[1].distance_from_center_m == pytest.approx(
        2000.0, abs=50.0)


def test_distance_is_nan_without_a_center(transit):
    import math

    for t in findPossibleHouseEntrance(transit).transitions:
        assert math.isnan(t.distance_from_center_m)


def test_direction_selects_arrivals_or_departures(transit):
    arrivals = findPossibleHouseEntrance(transit, direction="arrive")
    departures = findPossibleHouseEntrance(transit, direction="depart")

    assert [t.direction for t in arrivals.transitions] == [MOVE_TO_STAY]
    assert [t.direction for t in departures.transitions] == [STAY_TO_MOVE]


# ------------------------------------------------------------------ haversine

def test_haversine_matches_known_distances():
    assert haversine_m(32.0, 34.0, 32.0, 34.0) == 0.0
    assert haversine_m(32.0, 34.0, 33.0, 34.0) == pytest.approx(111_194.9, abs=0.1)


def test_haversine_crosses_the_antimeridian():
    # A projection that subtracts longitudes would report ~40 000 km here.
    assert haversine_m(0.0, 179.999, 0.0, -179.999) == pytest.approx(222.0, abs=1.0)


# ------------------------------------------------------------- bad arguments

@pytest.mark.parametrize("kwargs, message", [
    (dict(direction="arrivals"), "direction"),
    (dict(unknown="ignore"), "unknown"),
    (dict(center=(32.0, 34.0), radius_m=-1.0), "radius_m"),
    (dict(center=(95.0, 34.0)), "latitude"),
    (dict(center=(32.0, 200.0)), "longitude"),
    (dict(place_tolerance_m=-5.0), "place_tolerance_m"),
])
def test_bad_arguments_name_the_problem(transit, kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        findPossibleHouseEntrance(transit, **kwargs)


def test_center_must_be_a_pair(transit):
    with pytest.raises(TypeError, match="lat, lon"):
        findPossibleHouseEntrance(transit, center=32.0)


def test_non_iterable_segments_are_rejected():
    with pytest.raises(TypeError, match="segments"):
        findPossibleHouseEntrance(None)
    with pytest.raises(TypeError, match="segments"):
        findPossibleHouseEntrance("stay")


def test_a_missing_required_field_names_it(transit):
    incomplete = [{k: v for k, v in as_dict(s).items() if k != "centerLat"}
                  for s in transit]

    with pytest.raises(ValueError, match="centerLat"):
        findPossibleHouseEntrance(incomplete)


def test_an_unrecognized_label_is_refused(transit):
    mislabelled = [SimpleNamespace(**{**as_dict(transit[0]),
                                      "classification": "semi_stay"})]

    with pytest.raises(ValueError, match="semi_stay"):
        findPossibleHouseEntrance(mislabelled)


# ------------------------------------------------------------- independence

_NO_GPS_STAY = r"""
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from pyramidElevatorDist.utils.findPossibleHouseEntrance import (
    findPossibleHouseEntrance,
)

t0, cadence = datetime(2024, 3, 1, 8, 0, 0), timedelta(seconds=10)
def seg(label, i0, i1, lon):
    return SimpleNamespace(
        startTime=t0 + i0 * cadence, endTime=t0 + i1 * cadence,
        classification=label, centerLat=32.08, centerLon=lon,
        radiusMeters=50.0, probability=0.97,
    )

result = findPossibleHouseEntrance(
    [seg("stay", 0, 79, 34.78), seg("movement", 80, 119, 34.79),
     seg("stay", 120, 199, 34.80)],
    center=(32.08, 34.78),
)
assert result.timestamps, result
leaked = [m for m in sys.modules if m == "gps_stay" or m.startswith("gps_stay.")]
assert not leaked, leaked
print("ok")
"""


def test_parser_does_not_load_gps_stay():
    """The package must never depend on the stay detector it parses.

    Run in a subprocess so the assertion holds regardless of what this test
    session imported earlier.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _NO_GPS_STAY],
        capture_output=True, text=True, check=True,
    )

    assert completed.stdout.strip() == "ok"
