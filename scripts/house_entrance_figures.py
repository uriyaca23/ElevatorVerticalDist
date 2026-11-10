"""Per-experiment figure: GPS track on a Google basemap, nahori indoor/outdoor
prediction, GPS stay segmentation + house-entrance candidates, and NMEA signal.

For every Shaft Lab session (``shaft_sessions`` in Supabase) that uploaded both
``loc`` and ``nmea`` files, this:

1. downloads the two files from Firebase Storage (cached under ``--out-dir``),
2. parses them with nahori's own Skydex parsers (``data.parsed``),
3. runs nahori's production indoor/outdoor detector (``--strategy``, default
   ``lean``) on the NMEA features,
4. runs the ``gps_stay`` detector on the best fix per second and feeds its
   segments to :func:`findPossibleHouseEntrance`,
5. writes ``<out-dir>/session_<id>.png``.

nahori and gps_stay are sibling repos, not dependencies of this project; they
are put on ``sys.path`` from ``--nahori-src`` / ``--gps-stay``. Credentials come
from the web app's ``web/.env.local`` (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
FIREBASE_SERVICE_ACCOUNT, FIREBASE_BUCKET). Needs ``pyyaml`` (nahori's strategy
configs) and ``google-cloud-storage`` (nahori's ``data`` package imports it).

    venv/bin/python scripts/house_entrance_figures.py              # all sessions
    venv/bin/python scripts/house_entrance_figures.py --sessions 38 41
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
from bidi.algorithm import get_display

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GITHUB = _REPO_ROOT.parent

TILE_SERVERS = {
    "google": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",  # satellite + labels
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
}

# Reference data-viz palette: a one-hue blue ramp for time, categorical slots
# 2/3/7 for indoor/outdoor/stay (slot 1 is taken by the time ramp), inks and a
# neutral grey for movement.
TIME_CMAP = LinearSegmentedColormap.from_list(
    "time_blue", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
INDOOR, OUTDOOR = "#eb6834", "#1baf7a"
INK, INK_2, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#9a9993", "#e4e3df"
CNR30 = "#d55181"
STAY, MOVE_FILL, MOVE_DOT = "#6250d6", "#d7d6d0", "#f2f1ec"
# Hebrew class names in visual order (matplotlib lays text out left-to-right).
STAY_HE, MOVE_HE, UNKNOWN_HE = (get_display(w) for w in ("שהייה", "תנועה", "לא ידוע"))

STRONG_CNR_DBHZ = 30.0
HDOP_CAP = 10.0


# --------------------------------------------------------------------------- data

def _read_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
        if m:
            v = m.group(2)
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            env[m.group(1)] = v
    return env


def _sessions(env: dict[str, str], include_deleted: bool) -> list[dict]:
    key = env["SUPABASE_SERVICE_ROLE_KEY"]
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    base = env["SUPABASE_URL"].rstrip("/") + "/rest/v1"
    rows = requests.get(f"{base}/shaft_sessions?select=*&order=id", headers=headers, timeout=30)
    rows.raise_for_status()
    rows = rows.json()
    if not include_deleted:
        rev = requests.get(f"{base}/shaft_session_review?select=session_id,deleted_at",
                           headers=headers, timeout=30)
        deleted = {r["session_id"] for r in rev.json() if r.get("deleted_at")} if rev.ok else set()
        rows = [r for r in rows if r["id"] not in deleted]
    return [r for r in rows if {"loc", "nmea"} <= set(r.get("files") or {})]


def _download(env: dict[str, str], session: dict, cache: Path, refresh: bool) -> tuple[bytes, bytes]:
    d = cache / str(session["id"])
    d.mkdir(parents=True, exist_ok=True)
    paths = {"loc": d / "loc.csv", "nmea": d / "nmea.txt"}
    if refresh or not all(p.exists() for p in paths.values()):
        from google.cloud import storage
        from google.oauth2 import service_account

        info = json.loads(env["FIREBASE_SERVICE_ACCOUNT"])
        creds = service_account.Credentials.from_service_account_info(info)
        bucket = storage.Client(project=info["project_id"], credentials=creds).bucket(env["FIREBASE_BUCKET"])
        for k, p in paths.items():
            p.write_bytes(bucket.blob(session["files"][k]).download_as_bytes())
    return paths["loc"].read_bytes(), paths["nmea"].read_bytes()


def _best_fix_per_second(gps: pd.DataFrame) -> pd.DataFrame:
    """One fix per device-local second, the most accurate one (providers overlap)."""
    g = gps.dropna(subset=["lat", "lon"]).copy()
    g["sec"] = g["t_ms"] // 1000
    g = g.sort_values(["sec", "accuracy"], na_position="last").drop_duplicates("sec")
    return g.sort_values("t_ms").reset_index(drop=True)


def _local(ms, tz: str | None = None) -> pd.DatetimeIndex:
    """Naive local datetimes. Without ``tz`` the ms are nahori's device-local
    wall clock stored as UTC; with ``tz`` they are true UTC instants."""
    t = pd.to_datetime(np.asarray(ms, dtype="int64"), unit="ms", utc=tz is not None)
    return t.tz_convert(tz).tz_localize(None) if tz else t


def _label(seg) -> str:
    """A gps_stay segment's class as a lowercase string (stay / movement / unknown)."""
    return str(getattr(seg.classification, "value", seg.classification)).lower()


def analyse(loc_bytes: bytes, nmea_bytes: bytes, strategy: str, tz: str) -> dict:
    from data.parsed import parse_gps_csv, parse_nmea_txt
    from detector.dependencies import Dependencies
    from detector.strategies import get_strategy
    from gps_stay import ProbabilisticStayDetector
    from pyramidElevatorDist.utils.findPossibleHouseEntrance import findPossibleHouseEntrance

    gps = parse_gps_csv(loc_bytes)
    parsed = parse_nmea_txt(nmea_bytes)
    agg = parsed.aggregate.sort_values("t_ms").reset_index(drop=True)
    sat = parsed.per_satellite

    model = get_strategy(strategy)
    result = model.forward(Dependencies(
        nmea=agg if not agg.empty else None,
        gps=gps if not gps.empty else None,
        sat=sat if not sat.empty else None,
    ))
    io_segments = model.to_segments(result)

    fixes = _best_fix_per_second(gps) if not gps.empty else gps.assign(stay_class=pd.Series(dtype=str))
    fixes["stay_class"] = "unknown"
    stay_segments, entrances = [], None
    if len(fixes):
        # The device clock, not timeUTC: on several phones the `gps` provider's
        # timeUTC carries the GPS week-rollover bug (1024 weeks early, 2007).
        utc = (pd.to_datetime(fixes["t_ms"].astype("int64"), unit="ms")
               .dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC"))
        track = pd.DataFrame({
            "timestamp": utc,
            "lat": fixes["lat"].astype(float),
            "lon": fixes["lon"].astype(float),
            "row": fixes.index,
        }).dropna(subset=["timestamp"]).sort_values("timestamp", ignore_index=True)
        stay_segments = ProbabilisticStayDetector().detect(track[["timestamp", "lat", "lon"]])
        # Segment row ranges index `track`; carry each fix's class back to `fixes`.
        for seg in stay_segments:
            rows = track["row"].iloc[seg.startIndex:seg.endIndex + 1]
            fixes.loc[rows, "stay_class"] = _label(seg)
        entrances = findPossibleHouseEntrance(stay_segments)

    strong = (sat[sat["snr"].astype(float) >= STRONG_CNR_DBHZ].groupby("t_ms").size()
              if not sat.empty else pd.Series(dtype="int64"))
    agg["num_cnr30"] = agg["t_ms"].map(strong).fillna(0).astype(int)
    return {"fixes": fixes, "nmea": agg, "verdicts": result.verdicts,
            "io_segments": io_segments, "stays": stay_segments, "entrances": entrances}


# --------------------------------------------------------------------------- plot

def _basemap(fixes: pd.DataFrame, tile_server: str, cache: Path):
    from data.cache import FileCache
    from data.geo.tiles import basemap_image
    from pyramidElevatorDist.utils.findPossibleHouseEntrance import haversine_m
    from util.projection import zoom_for_radius

    lat_lo, lat_hi = fixes["lat"].min(), fixes["lat"].max()
    lon_lo, lon_hi = fixes["lon"].min(), fixes["lon"].max()
    lat_c, lon_c = (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2
    half_m = max(float(haversine_m(lat_lo, lon_c, lat_hi, lon_c)),
                 float(haversine_m(lat_c, lon_lo, lat_c, lon_hi))) / 2
    radius = max(half_m * 1.15, 120.0)
    zoom = zoom_for_radius(radius, lat_c, target_px=2048, max_zoom=19)
    bm = basemap_image(lat_c, lon_c, radius, zoom=zoom, tile_server=tile_server,
                       cache=FileCache(cache / "tiles"))
    return bm, (lat_c, lon_c, radius)


def _shade(ax, spans, color, alpha):
    for s, e in spans:
        ax.axvspan(s, e, color=color, alpha=alpha, lw=0)


def _runs(times: pd.DatetimeIndex, mask: np.ndarray, step=pd.Timedelta(seconds=1)):
    """Contiguous True runs of ``mask`` as (start, end) datetimes."""
    out, start = [], None
    for t, m in zip(times, mask):
        if m and start is None:
            start = t
        elif not m and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, times[-1] + step))
    return out


def _style_time_ax(ax, label):
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK_2)
    ax.yaxis.set_label_coords(-0.03, 0.5)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _timezone(session: dict) -> str:
    return (session.get("recording_settings") or {}).get("timezone") or "Asia/Jerusalem"


def _draw_map_base(ax, bm, view, res: dict, entrances: list) -> None:
    """Washed basemap, stay circles, entrance markers, scale bar and limits."""
    lat_c, lon_c, radius = view
    min_lon, min_lat, max_lon, max_lat = bm.extent
    ax.imshow(bm.image, extent=(min_lon, max_lon, min_lat, max_lat), origin="upper", zorder=0)
    ax.add_patch(plt.Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                               color="white", alpha=0.28, lw=0, zorder=1))

    m_per_deg = 111_320.0
    for s in res["stays"]:
        if _label(s) != "stay":
            continue
        w = 2 * s.radiusMeters / (m_per_deg * np.cos(np.radians(s.centerLat)))
        h = 2 * s.radiusMeters / m_per_deg
        for lw, col in ((3.2, "white"), (1.4, INK)):
            ax.add_patch(Ellipse((s.centerLon, s.centerLat), w, h, fill=False, lw=lw,
                                 ec=col, ls=(0, (4, 2)) if col == INK else "-", zorder=4))

    by_place: dict[tuple[float, float], list[str]] = {}
    for i, e in enumerate(entrances, 1):
        by_place.setdefault((round(e.lat, 5), round(e.lon, 5)), []).append(str(i))
    for (la, lo), ids in by_place.items():
        ax.annotate(",".join(ids), (lo, la), xytext=(9, 9), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=INK, lw=0.8),
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
        ax.plot(lo, la, marker="o", ms=6, mfc="white", mec=INK, mew=1.4, zorder=6)

    half_lat = radius / m_per_deg
    half_lon = radius / (m_per_deg * np.cos(np.radians(lat_c)))
    ax.set_xlim(lon_c - half_lon, lon_c + half_lon)
    ax.set_ylim(lat_c - half_lat, lat_c + half_lat)
    ax.set_aspect(1 / np.cos(np.radians(lat_c)))
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    bar_m = 10 ** np.floor(np.log10(radius * 0.6))
    bx = lon_c - half_lon * 0.92
    by = lat_c - half_lat * 0.94
    bar = [bx, bx + bar_m / (m_per_deg * np.cos(np.radians(lat_c)))]
    ax.plot(bar, [by, by], color="white", lw=5, solid_capstyle="butt", zorder=7)
    ax.plot(bar, [by, by], color=INK, lw=2.5, solid_capstyle="butt", zorder=8)
    ax.text(bx, by + half_lat * 0.03, f"{bar_m:.0f} m", fontsize=9, color=INK, zorder=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


_STAY_RING = Line2D([], [], color=INK, ls=(0, (4, 2)), lw=1.4, label="gps_stay stay area (50 m)")
_ENTRANCE_MARK = Line2D([], [], marker="o", ls="", ms=6, mfc="white", mec=INK, mew=1.4,
                        label="entrance candidate #")


def plot_session(session: dict, res: dict, out_path: Path, tile_server: str, cache: Path) -> None:
    tz = _timezone(session)
    fixes, nmea = res["fixes"], res["nmea"]
    fix_t = _local(fixes["t_ms"])
    nmea_t = _local(nmea["t_ms"])
    t0 = min(fix_t.min(), nmea_t.min()) if len(nmea_t) else fix_t.min()
    t1 = max(fix_t.max(), nmea_t.max()) if len(nmea_t) else fix_t.max()
    norm = Normalize(mdates.date2num(t0), mdates.date2num(t1))

    entrances = res["entrances"].transitions if res["entrances"] else []
    ent_t = _local([e.timestamp_ms for e in entrances], tz)

    fig = plt.figure(figsize=(22, 21), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.08,
                             left=0.07, right=0.985, top=0.93, bottom=0.04)
    top = outer[0].subgridspec(1, 2, wspace=0.03)
    bottom = outer[1].subgridspec(7, 1, height_ratios=[1.5, 0.55, 0.3, 1.6, 1.25, 1.05, 0.3], hspace=0.3)

    # ---- maps: by time | by gps_stay class
    bm, view = _basemap(fixes, tile_server, cache)
    ax_map, ax_cls = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    for ax in (ax_map, ax_cls):
        _draw_map_base(ax, bm, view, res, entrances)

    ax_map.scatter(fixes["lon"], fixes["lat"], c=mdates.date2num(fix_t), cmap=TIME_CMAP, norm=norm,
                   s=13, linewidths=0.35, edgecolors="white", zorder=3)
    ax_map.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=6, mfc=TIME_CMAP(0.15), mec="white", label="GPS fix, earliest"),
        Line2D([], [], marker="o", ls="", ms=6, mfc=TIME_CMAP(0.95), mec="white", label="GPS fix, latest"),
        _STAY_RING, _ENTRANCE_MARK,
    ], loc="upper left", fontsize=8.5, framealpha=0.92, edgecolor=GRID)
    ax_map.set_title("GPS fixes (best per second), coloured by time", fontsize=11, color=INK, loc="left")

    cls = fixes["stay_class"]
    styles = {  # drawn in this order, so stays sit on top
        "unknown": dict(s=9, fc="none", ec=INK_MUTED, lw=0.6),
        "movement": dict(s=9, fc=MOVE_DOT, ec=INK_2, lw=0.4),
        "stay": dict(s=24, fc=STAY, ec="white", lw=0.5),
    }
    for lab, st in styles.items():
        sel = cls == lab
        if sel.any():
            ax_cls.scatter(fixes.loc[sel, "lon"], fixes.loc[sel, "lat"], zorder=3, **st)
    counts = cls.value_counts()
    ax_cls.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=6, mfc=STAY, mec="white",
               label=f"{STAY_HE}  stay ({counts.get('stay', 0):,} fixes)"),
        Line2D([], [], marker="o", ls="", ms=5, mfc=MOVE_DOT, mec=INK_2,
               label=f"{MOVE_HE}  movement ({counts.get('movement', 0):,})"),
        *([Line2D([], [], marker="o", ls="", ms=5, mfc="none", mec=INK_MUTED,
                  label=f"{UNKNOWN_HE}  unknown ({counts['unknown']:,})")] if counts.get("unknown") else []),
        _STAY_RING, _ENTRANCE_MARK,
    ], loc="upper left", fontsize=8.5, framealpha=0.92, edgecolor=GRID)
    ax_cls.set_title(f"GPS segmentation (gps_stay, input to findPossibleHouseEntrance): "
                     f"{get_display('שהייה / תנועה')}", fontsize=11, color=INK, loc="left")

    # ---- time panels
    axes = [fig.add_subplot(bottom[i]) for i in range(7)]
    for a in axes[1:]:
        a.sharex(axes[0])
    ax_io, ax_stay, ax_time, ax_sats, ax_cnr, ax_hdop, ax_fix = axes

    # nahori indoor/outdoor
    for s in res["io_segments"]:
        ax_io.axvspan(_local([s.start_ms])[0], _local([s.end_ms])[0],
                      color=INDOOR if s.label == "indoor" else OUTDOOR, alpha=0.32, lw=0)
    if res["verdicts"]:
        ax_io.plot(_local([v.t_ms for v in res["verdicts"]]), [v.p_indoor for v in res["verdicts"]],
                   color=INK, lw=1.2)
    ax_io.set_ylim(-0.02, 1.02)
    ax_io.set_yticks([0, 0.5, 1])
    _style_time_ax(ax_io, "nahori\nP(indoor)")
    ax_io.legend(handles=[Patch(color=INDOOR, alpha=0.5, label="nahori indoor"),
                          Patch(color=OUTDOOR, alpha=0.5, label="nahori outdoor"),
                          Line2D([], [], color=INK, lw=1.2, label="P(indoor), HMM-smoothed"),
                          Patch(color=STAY, label=f"{STAY_HE} stay"),
                          Patch(color=MOVE_FILL, label=f"{MOVE_HE} movement"),
                          Patch(fc="none", ec=INK_MUTED, hatch="////", label=f"{UNKNOWN_HE} unknown"),
                          Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)), label="entrance candidate")],
                 loc="lower left", ncol=7, fontsize=8.5, frameon=False, bbox_to_anchor=(0, 1.02),
                 handlelength=1.6, columnspacing=1.4)

    # gps_stay segmentation, each segment labelled with its class and probability
    span_days = norm.vmax - norm.vmin
    for s in res["stays"]:
        lab = _label(s)
        st, en = _local([int(pd.Timestamp(s.startTime, tz="UTC").value // 1_000_000),
                         int(pd.Timestamp(s.endTime, tz="UTC").value // 1_000_000)], tz)
        if lab == "unknown":
            ax_stay.axvspan(st, en, fc="none", ec=INK_MUTED, hatch="////", lw=0)
        else:
            ax_stay.axvspan(st, en, color=STAY if lab == "stay" else MOVE_FILL, lw=0)
        if (mdates.date2num(en) - mdates.date2num(st)) > 0.07 * span_days:
            name = {"stay": STAY_HE, "movement": MOVE_HE}.get(lab, UNKNOWN_HE)
            ax_stay.text(st + (en - st) / 2, 0.5, f"{name}  p={s.probability:.2f}",
                         transform=ax_stay.get_xaxis_transform(), ha="center", va="center",
                         fontsize=9, color="white" if lab == "stay" else INK)
    ax_stay.set_yticks([])
    _style_time_ax(ax_stay, f"gps_stay\n{get_display('שהייה / תנועה')}")
    ax_stay.grid(False)

    # time colour key, same mapping as the time map
    grid = np.linspace(norm.vmin, norm.vmax, 600)
    ax_time.imshow(grid[None, :], aspect="auto", cmap=TIME_CMAP, norm=norm,
                   extent=(norm.vmin, norm.vmax, 0, 1))
    ax_time.set_yticks([])
    _style_time_ax(ax_time, "map colour")
    ax_time.grid(False)

    # NMEA: satellites
    ax_sats.plot(nmea_t, nmea["sats_seen"], color=INK_MUTED, lw=1, label="satellites seen")
    ax_sats.plot(nmea_t, nmea["sats_used"], color=INK, lw=1, label="satellites used in fix")
    ax_sats.plot(nmea_t, nmea["num_cnr30"], color=CNR30, lw=1, label=f"satellites ≥ {STRONG_CNR_DBHZ:.0f} dB-Hz")
    ax_sats.set_ylim(bottom=0)
    _style_time_ax(ax_sats, "NMEA\nsatellites")
    ax_sats.legend(loc="upper left", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0, 1.2))

    # NMEA: mean C/N0
    ax_cnr.plot(nmea_t, nmea["mean_snr"], color=INK, lw=1)
    ax_cnr.axhline(STRONG_CNR_DBHZ, color=CNR30, lw=1, ls=(0, (4, 3)))
    ax_cnr.text(0.998, STRONG_CNR_DBHZ, f"{STRONG_CNR_DBHZ:.0f} dB-Hz", transform=ax_cnr.get_yaxis_transform(),
                fontsize=8, color=CNR30, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    _style_time_ax(ax_cnr, "NMEA mean\nC/N0 (dB-Hz)")

    # NMEA: HDOP (no-fix seconds are left as gaps)
    hdop = nmea["hdop"].astype(float).where(nmea["fix_quality"].fillna(0).astype(float) > 0)
    ax_hdop.plot(nmea_t, hdop.clip(upper=HDOP_CAP), color=INK, lw=1)
    ax_hdop.set_ylim(0, HDOP_CAP * 1.05)
    _style_time_ax(ax_hdop, f"NMEA HDOP\n(capped {HDOP_CAP:.0f})")

    # NMEA: fix / no fix
    has_fix = nmea["fix_quality"].fillna(0).astype(float).to_numpy() > 0
    _shade(ax_fix, _runs(nmea_t, has_fix), INK, 0.85)
    ax_fix.set_yticks([])
    _style_time_ax(ax_fix, "GNSS fix")
    ax_fix.grid(False)

    for a in axes:
        for t in ent_t:
            a.axvline(t, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=5)
    for i, (t, e) in enumerate(zip(ent_t, entrances), 1):
        what = "arrived" if e.direction == "move_to_stay" else "left"
        ax_io.annotate(f"{i} {what}", (t, 1.0), xytext=(0, -2), textcoords="offset points",
                       fontsize=8.5, color=INK, ha="center", va="top", fontweight="bold", zorder=6,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=INK, lw=0.6))

    for a in axes[:-1]:
        a.tick_params(labelbottom=False)
    ax_fix.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_fix.set_xlim(t0, t1)
    ax_fix.set_xlabel(f"local time ({tz})", fontsize=9, color=INK_2)

    io = res["io_segments"]
    total = sum(s.end_ms - s.start_ms for s in io) or 1
    indoor_pct = 100 * sum(s.end_ms - s.start_ms for s in io if s.label == "indoor") / total
    n_stays = sum(1 for s in res["stays"] if _label(s) == "stay")
    device = " ".join(x for x in (session.get("device_brand"), session.get("device_model")) if x)
    fig.text(0.03, 0.975, f"Session {session['id']} · {session['username']} · {device}",
             fontsize=16, fontweight="bold", color=INK)
    fig.text(0.03, 0.96,
             f"{t0:%d %b %Y  %H:%M}–{t1:%H:%M} local · {len(fixes):,} GPS fixes · "
             f"nahori lean: {indoor_pct:.0f}% indoor · gps_stay: {n_stays} stays · "
             f"findPossibleHouseEntrance: {len(entrances)} candidates",
             fontsize=11, color=INK_2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=115)
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sessions", type=int, nargs="*", help="session ids (default: all with loc + nmea)")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "elevator_reports" / "house_entrance")
    ap.add_argument("--env", type=Path, default=_REPO_ROOT / "web" / ".env.local")
    ap.add_argument("--nahori-src", type=Path, default=_GITHUB / "nahori" / "src" / "algorithm" / "src")
    ap.add_argument("--gps-stay", type=Path, default=_GITHUB / "gps_stay")
    ap.add_argument("--strategy", default="lean", help="nahori detector strategy (lean, lean_live, nmea, ...)")
    ap.add_argument("--basemap", choices=sorted(TILE_SERVERS), default="google")
    ap.add_argument("--include-deleted", action="store_true", help="also plot sessions hidden on the site")
    ap.add_argument("--refresh", action="store_true", help="re-download loc/nmea files")
    args = ap.parse_args(argv)

    for p in (str(_REPO_ROOT), str(args.nahori_src), str(args.gps_stay)):
        if p not in sys.path:
            sys.path.insert(0, p)

    env = _read_env(args.env)
    sessions = _sessions(env, args.include_deleted)
    if args.sessions:
        sessions = [s for s in sessions if s["id"] in set(args.sessions)]
    cache = args.out_dir / "cache"
    for s in sessions:
        loc, nmea = _download(env, s, cache, args.refresh)
        res = analyse(loc, nmea, args.strategy, _timezone(s))
        if res["fixes"].empty or res["nmea"].empty:
            print(f"session {s['id']}: skipped (no GPS fixes or no NMEA)")
            continue
        out = args.out_dir / f"session_{s['id']}.png"
        plot_session(s, res, out, TILE_SERVERS[args.basemap], cache)
        ents = res["entrances"].transitions if res["entrances"] else []
        print(f"session {s['id']}: {len(ents)} entrance candidates -> {out.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
