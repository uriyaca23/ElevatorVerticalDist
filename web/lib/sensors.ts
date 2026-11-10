import "server-only";
import { analyzeBarometer } from "@/lib/barometer";
import { bucket } from "@/lib/clients/firebase";
import { parseLocCsv, parseNmea, skydexUtcOffsetMin, zoneOffsetMin, type GpsTrack, type NmeaSeries } from "@/lib/gnss";
import { detectStays } from "@/lib/gpsStay";
import { reconstructVertical } from "@/lib/reconstruct";
import type { SensorPayload, SessionRow } from "@/lib/types";

// Above this many samples a sensor is min/max-decimated for plotting.
// The barometer analysis always runs on full-resolution data.
const MAX_PLOT_POINTS = 60_000;

interface Table {
  t_ms: Float64Array;
  cols: Record<string, Float64Array>;
}

// Skydex rows look like `"2026-10-18T13:43:05.120+0300","1008.49","79.1"`:
// device-local ISO time with a `+HHMM` offset, which Date.parse won't take.
function parseTimestamp(s: string): number {
  return Date.parse(s.replace(/([+-]\d\d)(\d\d)$/, "$1:$2"));
}

/** Parse a Skydex sensor CSV into numeric columns, sorted by time. */
export function parseSensorCsv(text: string, columns: string[]): Table {
  const lines = text.split(/\r?\n/);
  const header = lines[0].split(",").map((h) => h.replace(/"/g, "").trim());
  const tIdx = header.indexOf("timestamp");
  const idx = columns.map((c) => header.indexOf(c));
  if (tIdx < 0 || idx.some((i) => i < 0)) {
    throw new Error(`expected columns timestamp,${columns.join(",")}; got ${header.join(",")}`);
  }
  const rows: number[][] = [];
  for (let k = 1; k < lines.length; k++) {
    const line = lines[k];
    if (!line) continue;
    const f = line.split(",").map((v) => v.replace(/"/g, ""));
    const row = [parseTimestamp(f[tIdx]), ...idx.map((i) => Number(f[i]))];
    if (row.every(Number.isFinite)) rows.push(row);
  }
  rows.sort((a, b) => a[0] - b[0]);
  const t_ms = Float64Array.from(rows, (r) => r[0]);
  const cols: Record<string, Float64Array> = {};
  columns.forEach((c, j) => (cols[c] = Float64Array.from(rows, (r) => r[j + 1])));
  return { t_ms, cols };
}

async function download(path: string): Promise<string> {
  const [buf] = await bucket().file(path).download();
  return buf.toString("utf8");
}

/**
 * Keep the min and max of `key` in each bucket so peaks survive decimation;
 * returns the kept sample indices (ascending, shared by all columns).
 */
function minMaxIndices(key: Float64Array, maxPoints: number): number[] {
  const n = key.length;
  if (n <= maxPoints) return Array.from({ length: n }, (_, i) => i);
  const buckets = Math.floor(maxPoints / 2);
  const out: number[] = [];
  for (let b = 0; b < buckets; b++) {
    const lo = Math.floor((b * n) / buckets);
    const hi = Math.floor(((b + 1) * n) / buckets);
    let iMin = lo;
    let iMax = lo;
    for (let i = lo; i < hi; i++) {
      if (key[i] < key[iMin]) iMin = i;
      if (key[i] > key[iMax]) iMax = i;
    }
    out.push(Math.min(iMin, iMax));
    if (iMin !== iMax) out.push(Math.max(iMin, iMax));
  }
  return out;
}

const round = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d;

function toSeries(t_ms: Float64Array, cols: Record<string, ArrayLike<number>>, keep: number[], t0: number) {
  const out: Record<string, number[]> = { t: keep.map((i) => round((t_ms[i] - t0) / 1000, 3)) };
  for (const [name, v] of Object.entries(cols)) out[name] = keep.map((i) => round(v[i], 4));
  return out;
}

function xyzSeries(table: Table, t0: number) {
  const { x, y, z } = table.cols;
  const norm = Float64Array.from(x, (_, i) => Math.hypot(x[i], y[i], z[i]));
  const keep = minMaxIndices(norm, MAX_PLOT_POINTS);
  return { series: toSeries(table.t_ms, { x, y, z }, keep, t0), from: x.length, to: keep.length };
}

/**
 * Accelerometer panel: the 50 Hz signal the algorithm operates on — resampled
 * x/y/z plus the Mahony-reconstructed vertical acceleration (a_rec) and |a|−g.
 */
function accelSeries(accel: Table, gyro: Table | null, t0: number) {
  const xyz = (t: Table) => ({ t: t.t_ms, x: t.cols.x, y: t.cols.y, z: t.cols.z });
  const v = reconstructVertical(xyz(accel), gyro && gyro.t_ms.length ? xyz(gyro) : null);
  const keep = minMaxIndices(v.a_rec, MAX_PLOT_POINTS);
  const series = toSeries(v.t, { x: v.x, y: v.y, z: v.z, a_rec: v.a_rec, a_mag_g: v.a_mag_g }, keep, t0);
  return { series, from: v.t.length, to: keep.length };
}

/**
 * NMEA line prefixes are device-local wall clock with no zone. The loc CSV's
 * `+HHMM` is the device's own offset, so prefer it; otherwise fall back to
 * the session's timezone (two passes so a DST switch lands on the right side).
 */
function nmeaOffset(locText: string | null, timeZone: string): number | ((wallMs: number) => number) {
  const fromLoc = locText ? skydexUtcOffsetMin(locText) : null;
  if (fromLoc != null) return fromLoc;
  const byHour = new Map<number, number>();
  return (wall) => {
    const h = Math.floor(wall / 3_600_000);
    let off = byHour.get(h);
    if (off == null) {
      off = zoneOffsetMin(wall - zoneOffsetMin(wall, timeZone) * 60000, timeZone);
      byHour.set(h, off);
    }
    return off;
  };
}

/** GPS fixes for the map + time panel, and their stay / movement segmentation. */
function gpsPayload(track: GpsTrack, t0: number) {
  const r = (v: number, d: number) => (Number.isFinite(v) ? round(v, d) : null);
  const n = track.t_ms.length;
  const series = {
    t: Array.from(track.t_ms, (t) => round((t - t0) / 1000, 3)),
    // 1e-7° ≈ 1 cm; the generic 4-decimal rounding would be ~10 m.
    lat: Array.from(track.lat, (v) => round(v, 7)),
    lon: Array.from(track.lon, (v) => round(v, 7)),
    accuracy: Array.from(track.accuracy, (v) => r(v, 1)),
  };
  return { series: series satisfies SensorPayload["gps"], stays: n ? detectStays(track.t_ms, track.lat, track.lon) : [] };
}

function nmeaPayload(nmea: NmeaSeries, t0: number) {
  return {
    t: Array.from(nmea.t_ms, (t) => round((t - t0) / 1000, 3)),
    snr: Array.from(nmea.snr, (v) => round(v, 2)),
  } satisfies SensorPayload["nmea"];
}

/** Download, parse and analyse the plotted sensors of one session. */
export async function loadSensors(session: SessionRow): Promise<SensorPayload> {
  const files = session.files ?? {};
  const errors: Record<string, string> = {};
  const fail = (key: string, e: unknown) => (errors[key] = e instanceof Error ? e.message : String(e));
  const load = async (key: string, columns: string[]) => {
    if (!files[key]) return null;
    try {
      return parseSensorCsv(await download(files[key]), columns);
    } catch (e) {
      fail(key, e);
      return null;
    }
  };
  const text = async (key: string) => {
    if (!files[key]) return null;
    try {
      return await download(files[key]);
    } catch (e) {
      fail(key, e);
      return null;
    }
  };
  const [baro, accel, gyro, locText, nmeaText] = await Promise.all([
    load("barometer", ["pressure"]),
    load("accel", ["x", "y", "z"]),
    load("gyro", ["x", "y", "z"]),
    text("loc"),
    text("nmea"),
  ]);

  let track: GpsTrack | null = null;
  let nmea: NmeaSeries | null = null;
  try {
    track = locText ? parseLocCsv(locText) : null;
  } catch (e) {
    fail("loc", e);
  }
  try {
    const tz = session.recording_settings?.timezone ?? "Asia/Jerusalem";
    nmea = nmeaText ? parseNmea(nmeaText, nmeaOffset(locText, tz)) : null;
  } catch (e) {
    fail("nmea", e);
  }

  const spans = [baro, accel, gyro, track, nmea]
    .filter((s): s is Table | GpsTrack | NmeaSeries => !!s && s.t_ms.length > 0)
    .map((s) => [s.t_ms[0], s.t_ms[s.t_ms.length - 1]]);
  const t0 = spans.length ? Math.min(...spans.map((s) => s[0])) : Date.parse(session.start_time ?? "") || 0;
  const t1 = spans.length ? Math.max(...spans.map((s) => s[1])) : t0;

  const payload: SensorPayload = {
    t0_ms: t0,
    t1_ms: t1,
    barometer: null,
    accel: null,
    gyro: null,
    gps: null,
    stays: [],
    nmea: null,
    segments: [],
    decimated: {},
    errors,
  };

  if (baro && baro.t_ms.length) {
    const analysis = analyzeBarometer(baro.t_ms, baro.cols.pressure);
    const all = Array.from({ length: baro.t_ms.length }, (_, i) => i);
    payload.barometer = toSeries(
      baro.t_ms,
      { height: analysis.height, height_smooth: analysis.heightSmooth },
      all,
      t0,
    ) as SensorPayload["barometer"];
    payload.segments = analysis.segments;
  }
  if (accel && accel.t_ms.length > 1) {
    const { series, from, to } = accelSeries(accel, gyro, t0);
    payload.accel = series as SensorPayload["accel"];
    if (to < from) payload.decimated.accel = { from, to };
  }
  if (gyro && gyro.t_ms.length) {
    const { series, from, to } = xyzSeries(gyro, t0);
    payload.gyro = series as SensorPayload["gyro"];
    if (to < from) payload.decimated.gyro = { from, to };
  }
  if (track && track.t_ms.length) {
    try {
      const { series, stays } = gpsPayload(track, t0);
      payload.gps = series;
      payload.stays = stays;
    } catch (e) {
      fail("loc", e);
    }
  }
  if (nmea && nmea.t_ms.length) payload.nmea = nmeaPayload(nmea, t0);
  return payload;
}
