// GNSS inputs of a Skydex session: the `loc` CSV (Android Location fixes from
// every provider) and the raw `nmea` log. Pure parsers — no I/O — so the
// parity tests can run them on cached files.

/** One fix per second, the most accurate one (gps / fused / network overlap). */
export interface GpsTrack {
  t_ms: Float64Array;
  lat: Float64Array;
  lon: Float64Array;
  /** Horizontal accuracy (m, 68 %); NaN when the provider left it empty. */
  accuracy: Float64Array;
  /** Reported ground speed (m/s); NaN when empty. */
  speed: Float64Array;
}

/** Per-second NMEA aggregate. Only seconds that carried a GSV sentence. */
export interface NmeaSeries {
  t_ms: Float64Array;
  /**
   * Mean C/N0 (dB-Hz) over every GSV satellite report with SNR > 0 that
   * second — nahori's `mean_snr`. A second whose GSV reports tracked no
   * satellite at all is 0 ("no signal"), not missing: indoors that is the
   * common case, and dropping it would hide the strongest indoor evidence.
   */
  snr: Float64Array;
  /** Distinct satellites with SNR > 0 that second. */
  sats: Float64Array;
}

const num = (s: string | undefined) => (s == null || s.trim() === "" ? NaN : Number(s));

/** Skydex CSV time `2026-10-18T13:43:05.120+0300` → epoch ms (honours the offset). */
export function parseSkydexTime(s: string): number {
  return Date.parse(s.replace(/([+-]\d\d)(\d\d)$/, "$1:$2"));
}

/** UTC offset (minutes) of the first Skydex timestamp in a CSV, or null. */
export function skydexUtcOffsetMin(text: string): number | null {
  const m = /"?\d{4}-\d\d-\d\dT[\d:.]+([+-])(\d\d):?(\d\d)"?/.exec(text.slice(0, 4096));
  if (!m) return null;
  return (m[1] === "-" ? -1 : 1) * (Number(m[2]) * 60 + Number(m[3]));
}

/** UTC offset (minutes) of `timeZone` at instant `ms`, via Intl. */
export function zoneOffsetMin(ms: number, timeZone: string): number {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    hourCycle: "h23",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).formatToParts(ms);
  const get = (t: string) => Number(parts.find((p) => p.type === t)?.value);
  const wall = Date.UTC(get("year"), get("month") - 1, get("day"), get("hour"), get("minute"), get("second"));
  return Math.round((wall - Math.floor(ms / 1000) * 1000) / 60000);
}

function splitCsvLine(line: string): string[] {
  return line.split(",").map((v) => v.replace(/"/g, ""));
}

/**
 * Parse `loc.csv` and keep the most accurate fix of each second (ties: the
 * first in file order). Rows without a parseable time or lat/lon are skipped
 * ("Location services were off" placeholders, empty fields).
 */
export function parseLocCsv(text: string): GpsTrack {
  const lines = text.split(/\r?\n/);
  const header = splitCsvLine(lines[0] ?? "").map((h) => h.trim());
  const col = (name: string) => header.indexOf(name);
  const [iT, iLat, iLon, iAcc, iSpd] = [col("timestamp"), col("latitude"), col("longitude"), col("accuracy"), col("speed")];
  if (iT < 0 || iLat < 0 || iLon < 0) {
    throw new Error(`loc: expected timestamp, latitude, longitude columns; got ${header.join(",")}`);
  }
  const best = new Map<number, { t: number; lat: number; lon: number; acc: number; spd: number }>();
  for (let k = 1; k < lines.length; k++) {
    if (!lines[k]) continue;
    const f = splitCsvLine(lines[k]);
    const t = parseSkydexTime(f[iT] ?? "");
    const lat = num(f[iLat]);
    const lon = num(f[iLon]);
    if (!Number.isFinite(t) || !Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    if (Math.abs(lat) > 90 || Math.abs(lon) > 180 || (lat === 0 && lon === 0)) continue;
    const acc = iAcc >= 0 ? num(f[iAcc]) : NaN;
    const spd = iSpd >= 0 ? num(f[iSpd]) : NaN;
    const sec = Math.floor(t / 1000);
    const prev = best.get(sec);
    const rank = (a: number) => (Number.isFinite(a) ? a : Infinity);
    if (!prev || rank(acc) < rank(prev.acc)) best.set(sec, { t, lat, lon, acc, spd });
  }
  const rows = [...best.values()].sort((a, b) => a.t - b.t);
  return {
    t_ms: Float64Array.from(rows, (r) => r.t),
    lat: Float64Array.from(rows, (r) => r.lat),
    lon: Float64Array.from(rows, (r) => r.lon),
    accuracy: Float64Array.from(rows, (r) => r.acc),
    speed: Float64Array.from(rows, (r) => r.spd),
  };
}

/**
 * Parse a Skydex NMEA log (`yyyyMMddHHmmssSSS|$GPGSV,...` per line, after one
 * header line) into per-second mean C/N0. Mirrors nahori's `parse_nmea_txt`
 * aggregate (data/parsed.py) for the SNR columns.
 *
 * The line prefix is the device's local wall clock with no zone, so
 * `utcOffsetMin` (the device's offset, e.g. from the loc CSV's `+0300`)
 * converts it to epoch ms — the same clock every other panel is on.
 */
export function parseNmea(text: string, utcOffsetMin: number | ((wallMs: number) => number)): NmeaSeries {
  const offset = typeof utcOffsetMin === "number" ? () => utcOffsetMin : utcOffsetMin;
  const acc = new Map<number, { sum: number; n: number; prns: Set<string> }>();
  const lines = text.split("\n");
  for (let k = 1; k < lines.length; k++) {
    const raw = lines[k].replace(/\r$/, "");
    const bar = raw.indexOf("|");
    if (bar < 14) continue;
    const sentence = raw.slice(bar + 1);
    if (sentence[0] !== "$" || sentence.length < 6 || sentence.slice(3, 6) !== "GSV") continue;
    const c = raw.slice(0, 14);
    const wall = Date.UTC(+c.slice(0, 4), +c.slice(4, 6) - 1, +c.slice(6, 8), +c.slice(8, 10), +c.slice(10, 12), +c.slice(12, 14));
    if (!Number.isFinite(wall)) continue;
    let b = acc.get(wall);
    if (!b) acc.set(wall, (b = { sum: 0, n: 0, prns: new Set() }));
    const f = sentence.split("*")[0].split(",");
    const talker = sentence.slice(1, 3);
    for (let i = 4; i + 3 < f.length; i += 4) {
      const prn = Number.parseInt(f[i], 10);
      const snr = Number.parseInt(f[i + 3], 10);
      if (!Number.isFinite(prn) || !(snr > 0)) continue;
      b.sum += snr;
      b.n += 1;
      b.prns.add(`${talker}${prn}`);
    }
  }
  const walls = [...acc.keys()].sort((a, b) => a - b);
  return {
    t_ms: Float64Array.from(walls, (w) => w - offset(w) * 60000),
    snr: Float64Array.from(walls, (w) => {
      const b = acc.get(w)!;
      return b.n ? b.sum / b.n : 0;
    }),
    sats: Float64Array.from(walls, (w) => acc.get(w)!.prns.size),
  };
}
