"""Candidate house-entrance moments, parsed out of GPS stay detection.

Running the elevator pipeline over a whole trace is expensive; it only needs
to run where a ride could have happened — around the moments the user reached
or left a building. A GPS stay detector marks those cheaply, and this parses
its output into the switch timestamps.

Combined use with the GPS stay package::

    import pandas as pd
    from gps_stay import ProbabilisticStayDetector
    from pyramidElevatorDist.utils.findPossibleHouseEntrance import (
        findPossibleHouseEntrance,
    )

    df = get_gps_data()                    # timestamp_ms, lat, lon, ...
    gps = pd.DataFrame({                   # the detector's own frame
        "timestamp": pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True),
        "lat": df["lat"].astype(float),
        "lon": df["lon"].astype(float),
    }).sort_values("timestamp", ignore_index=True)   # detect() will not sort

    stay_move_segments = ProbabilisticStayDetector().detect(gps)
    house_entrance = findPossibleHouseEntrance(
        stay_move_segments, center=(32.19258, 34.88399))   # lat/lon of home

    house_entrance.timestamps              # epoch ms — where to run findSegments

Nothing here imports the stay detector; it is not a dependency of this
package. Segments are read structurally (attributes, mapping keys or
DataFrame columns) through :data:`_FIELDS`, the one place to update if those
upstream names move. Three properties of that output drive the logic:

* ``endTime`` is a segment's **last fix**, so a switch is bracketed by
  ``[prev_end_timestamp_ms, timestamp_ms]`` rather than instantaneous, and
  ``stay_duration_s`` runs one sample short of a tiled duration.
* Times are **naive datetimes holding UTC instants**. A naive value is
  localized, never converted; ``datetime.timestamp()`` would read it as local
  time and shift every result by the machine's offset.
* Row indices are ignored — a clipped hole carries an intentionally empty
  range (``endIndex == startIndex - 1``), so ordering on them is wrong.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0
DEFAULT_RADIUS_M = 1000.0
DEFAULT_STAY_RADIUS_M = 50.0
#: Two stay centres within this many radii are one place — the detector's own
#: ``merge_radius_factor`` default, so places split where it would have merged.
SAME_PLACE_RADIUS_FACTOR = 2.0
#: Below this (1973) a "millisecond" value is really epoch seconds.
MIN_PLAUSIBLE_EPOCH_MS = 100_000_000_000

STAY, MOVEMENT, UNKNOWN = "stay", "movement", "unknown"
MOVE_TO_STAY, STAY_TO_MOVE = "move_to_stay", "stay_to_move"

_LABELS = (STAY, MOVEMENT, UNKNOWN)
_UNKNOWN_MODES = ("skip", "break")
_DIRECTIONS = {"both": None, "arrive": MOVE_TO_STAY, "depart": STAY_TO_MOVE}
#: field -> (accepted spellings, default). The detector's camelCase is first.
_FIELDS = {
    "start_ms": (("startTime", "start_time"), None),
    "end_ms": (("endTime", "end_time"), None),
    "label": (("classification", "label"), None),
    "lat": (("centerLat", "center_lat"), None),
    "lon": (("centerLon", "center_lon"), None),
    "radius_m": (("radiusMeters", "radius_meters", "radius_m"), math.nan),
    "probability": (("probability",), 1.0),
}
_MISSING = object()


def _err(msg: str) -> str:
    return f"findPossibleHouseEntrance: {msg}"


@dataclass(frozen=True)
class HouseEntranceTransition:
    """One stay↔movement switch — a moment the user reached or left a place."""

    timestamp_ms: int              #: epoch ms, first fix of the new state
    direction: str                 #: MOVE_TO_STAY (arrived) | STAY_TO_MOVE (left)
    lat: float                     #: the stay's centre, degrees
    lon: float
    stay_duration_s: float
    stay_probability: float
    prev_end_timestamp_ms: int     #: epoch ms, last fix of the old state
    crossed_unknown: bool          #: the switch itself was never observed
    distance_from_center_m: float  #: NaN without a ``center``


@dataclass(frozen=True)
class HouseEntranceCandidates:
    timestamps: list[int]                        #: ascending epoch ms
    transitions: list[HouseEntranceTransition]   #: same order, with context


@dataclass(frozen=True)
class _Seg:
    """One normalized segment: epoch ms, lowercase label, plain floats."""

    start_ms: int
    end_ms: int
    label: str
    lat: float
    lon: float
    radius_m: float
    probability: float


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between lat/lon points (degrees).

    Numpy-vectorized. Needs no projection origin and stays correct across the
    antimeridian, unlike subtracting locally projected coordinates.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    d_lam = np.radians(np.subtract(lon2, lon1))
    a = (np.sin((p2 - p1) / 2.0) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(d_lam / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _to_epoch_ms(value, field: str) -> int:
    """Epoch ms from a datetime / Timestamp / ISO string; naive means UTC."""
    if isinstance(value, (int, float, np.integer, np.floating)) \
            and not isinstance(value, bool):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(_err(f"{field} is {value!r}, not a finite epoch-ms value"))
        if abs(v) < MIN_PLAUSIBLE_EPOCH_MS:
            raise ValueError(_err(
                f"{field} is {value!r}: too small for epoch milliseconds — this "
                f"looks like epoch seconds. Multiply by 1000, or pass a datetime."))
        return int(round(v))
    # utc=True localizes a naive value and converts an aware one.
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(_err(f"{field} is missing or unreadable (NaT): {value!r}"))
    return int(ts.value // 1_000_000)


def _to_float(value, field: str, default: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = math.nan
    if math.isfinite(out):
        return out
    if default is not None:
        return default
    raise ValueError(_err(f"{field} is {value!r}, expected a finite number"))


def _to_label(value) -> str:
    """The label set is closed, so an unrecognized one is an error: bucketing
    it would fabricate transitions, skipping it would drop real ones."""
    label = str(getattr(value, "value", value)).strip().lower()
    if label not in _LABELS:
        raise ValueError(_err(f"unrecognized classification {value!r}; "
                              f"expected one of {list(_LABELS)}"))
    return label


def _read(row, name: str):
    """One field off a segment, trying each accepted spelling in turn."""
    names, default = _FIELDS[name]
    for key in names:
        got = (row.get(key, _MISSING) if isinstance(row, Mapping)
               else getattr(row, key, _MISSING))
        if got is not _MISSING:
            return got
    if default is not None:
        return default
    raise ValueError(_err(f"a segment is missing {names[0]} (looked for "
                          f"{list(names)}). Pass the detector's output, a list "
                          f"of its model_dump() dicts, or a DataFrame."))


def _coerce_segments(segments) -> list[_Seg]:
    """Normalize detector output into ascending :class:`_Seg` records."""
    if isinstance(segments, pd.DataFrame):
        rows = segments.to_dict("records")
    elif segments is None or isinstance(segments, (str, bytes)):
        raise TypeError(_err(
            f"'segments' must be the detector's output — an iterable of "
            f"segments or mappings, or a DataFrame — got "
            f"{type(segments).__name__}"))
    else:
        try:
            rows = list(segments)
        except TypeError:
            raise TypeError(_err(f"'segments' is not iterable (got "
                                 f"{type(segments).__name__})")) from None

    out = []
    for row in rows:
        start = _to_epoch_ms(_read(row, "start_ms"), "startTime")
        out.append(_Seg(
            start_ms=start,
            # A clipped hole's end precedes its start; never go negative.
            end_ms=max(start, _to_epoch_ms(_read(row, "end_ms"), "endTime")),
            label=_to_label(_read(row, "label")),
            lat=_to_float(_read(row, "lat"), "centerLat"),
            lon=_to_float(_read(row, "lon"), "centerLon"),
            radius_m=_to_float(_read(row, "radius_m"), "radiusMeters", math.nan),
            probability=_to_float(_read(row, "probability"), "probability", math.nan),
        ))
    out.sort(key=lambda s: (s.start_ms, s.end_ms))
    return out


def _relocated(a: _Seg, b: _Seg, override: float | None) -> bool:
    """Whether two adjacent stays sit at genuinely different places."""
    tolerance = override
    if tolerance is None:
        widest = max(a.radius_m, b.radius_m)
        if not math.isfinite(widest) or widest <= 0.0:
            widest = DEFAULT_STAY_RADIUS_M
        tolerance = SAME_PLACE_RADIUS_FACTOR * widest
    return float(haversine_m(a.lat, a.lon, b.lat, b.lon)) > tolerance


def _walk(segs, unknown, place_tolerance_m, center):
    """One transition per observed stay↔movement switch."""
    def make(ts, direction, anchor, prev_end, gap):
        return HouseEntranceTransition(
            timestamp_ms=ts, direction=direction, lat=anchor.lat, lon=anchor.lon,
            stay_duration_s=(anchor.end_ms - anchor.start_ms) / 1000.0,
            stay_probability=anchor.probability, prev_end_timestamp_ms=prev_end,
            crossed_unknown=gap,
            distance_from_center_m=(math.nan if center is None else float(
                haversine_m(center[0], center[1], anchor.lat, anchor.lon))))

    # Drop UNKNOWN, remembering that one sat before the next survivor. "break"
    # instead cuts the chain, so nothing pairs across the hole.
    chain, gap = [], False
    for seg in segs:
        if seg.label == UNKNOWN:
            if unknown == "break":
                chain.append(None)
            gap = unknown != "break"
            continue
        chain.append((seg, gap))
        gap = False

    out = []
    for before, after in zip(chain, chain[1:]):
        if before is None or after is None:
            continue
        (a, _), (b, crossed) = before, after
        if a.label != b.label:      # UNKNOWN is gone, so this is stay<->movement
            leaving = a.label == STAY
            out.append(make(b.start_ms, STAY_TO_MOVE if leaving else MOVE_TO_STAY,
                            a if leaving else b, a.end_ms, crossed))
        elif a.label == STAY and _relocated(a, b, place_tolerance_m):
            # Adjacent stays at different places: the travel between them was
            # never observed, so the last fix at the old place is the best
            # instant available for the departure.
            out.append(make(a.end_ms, STAY_TO_MOVE, a, a.end_ms, True))
            out.append(make(b.start_ms, MOVE_TO_STAY, b, a.end_ms, True))
    return out


def _check_center(center):
    if center is None:
        return None
    try:
        lat, lon = center
    except (TypeError, ValueError):
        raise TypeError(_err(f"'center' must be a (lat, lon) degree pair or "
                             f"None, got {center!r}")) from None
    lat = _to_float(lat, "center latitude")
    lon = _to_float(lon, "center longitude")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(_err(f"'center' latitude must be in [-90, 90], got {lat!r}"))
    if not -180.0 <= lon <= 180.0:
        raise ValueError(_err(f"'center' longitude must be in [-180, 180], got {lon!r}"))
    return lat, lon


def findPossibleHouseEntrance(segments, *, center=None,
                              radius_m: float = DEFAULT_RADIUS_M,
                              direction: str = "both", unknown: str = "skip",
                              place_tolerance_m: float | None = None):
    """Find the stay↔movement switches in a GPS stay segmentation.

    Each is a moment the user plausibly walked into or out of a building, and
    so worth running the elevator pipeline around. Module docstring has the
    combined call with the detector.

    ``segments`` is the detector's output: segment objects, mappings, or a
    DataFrame; read structurally and sorted internally. ``center`` (lat, lon
    degrees) keeps only switches whose stay lies within ``radius_m`` of it
    (default 1 km) and fills ``distance_from_center_m``. ``direction`` is
    ``"both"`` / ``"arrive"`` / ``"depart"``. ``unknown="skip"`` makes the
    detector's abstentions transparent, flagging the switch
    ``crossed_unknown``; ``"break"`` emits nothing across one.
    ``place_tolerance_m`` overrides how far adjacent stay centres may sit and
    still be one place (default: :data:`SAME_PLACE_RADIUS_FACTOR` × radius).

    Returns :class:`HouseEntranceCandidates` — ``timestamps`` is the plain
    ascending epoch-ms array, ``transitions`` the same moments with context.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(_err(f"'direction' must be one of {list(_DIRECTIONS)}, "
                              f"got {direction!r}"))
    if unknown not in _UNKNOWN_MODES:
        raise ValueError(_err(f"'unknown' must be one of {list(_UNKNOWN_MODES)}, "
                              f"got {unknown!r}"))
    radius = _to_float(radius_m, "radius_m")
    if radius < 0.0:
        raise ValueError(_err(f"'radius_m' must be >= 0, got {radius_m!r}"))
    if place_tolerance_m is not None:
        place_tolerance_m = _to_float(place_tolerance_m, "place_tolerance_m")
        if place_tolerance_m < 0.0:
            raise ValueError(_err(f"'place_tolerance_m' must be >= 0 or None, "
                                  f"got {place_tolerance_m!r}"))
    center = _check_center(center)

    wanted = _DIRECTIONS[direction]
    found = [t for t in _walk(_coerce_segments(segments), unknown,
                              place_tolerance_m, center)
             if (wanted is None or t.direction == wanted)
             and (center is None or t.distance_from_center_m <= radius)]
    found.sort(key=lambda t: t.timestamp_ms)
    return HouseEntranceCandidates([t.timestamp_ms for t in found], found)
