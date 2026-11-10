// Parity between lib/gpsStay.ts and gps_stay's ProbabilisticStayDetector, and
// between lib/gnss.ts parseNmea and nahori's parse_nmea_txt. Fixtures come
// from scripts/make_gps_stay_fixture.py; regenerate after changing either side.
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { parseLocCsv, parseNmea, skydexUtcOffsetMin, zoneOffsetMin } from "@/lib/gnss";
import { detectStays, type StaySegment } from "@/lib/gpsStay";

type Expected = Omit<StaySegment, "radius_m">;
const fixture: {
  synthetic: Record<string, { t_ms: number[]; lat: number[]; lon: number[]; segments: Expected[] }>;
  sessions: Record<
    string,
    { n_fixes: number; segments: Expected[]; nmea?: { count: number; sum: number; samples: [number, number][] } }
  >;
} = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures/gps_stay_parity.json"), "utf8"));

const CACHE = path.resolve(__dirname, "../../elevator_reports/house_entrance/cache");

function expectSameSegments(actual: StaySegment[], expected: Expected[]) {
  expect(actual.map((s) => s.classification)).toEqual(expected.map((s) => s.classification));
  actual.forEach((s, i) => {
    const e = expected[i];
    expect([s.start_index, s.end_index, s.point_count]).toEqual([e.start_index, e.end_index, e.point_count]);
    expect([s.start_ms, s.end_ms]).toEqual([e.start_ms, e.end_ms]);
    expect(s.center_lat).toBeCloseTo(e.center_lat, 9);
    expect(s.center_lon).toBeCloseTo(e.center_lon, 9);
    expect(s.probability).toBeCloseTo(e.probability, 9);
  });
}

describe("detectStays matches gps_stay ProbabilisticStayDetector.detect", () => {
  for (const [name, c] of Object.entries(fixture.synthetic)) {
    it(`synthetic ${name}`, () => expectSameSegments(detectStays(c.t_ms, c.lat, c.lon), c.segments));
  }

  it("degenerate tracks", () => {
    expect(detectStays([], [], [])).toEqual([]);
    const [one] = detectStays([1_792_000_000_000, 1_792_000_020_000], [32, 32.0001], [34, 34]);
    expect(one).toMatchObject({ classification: "unknown", point_count: 2, probability: 0 });
    expect(one.center_lat).toBeCloseTo(32.00005, 10);
  });

  for (const [name, s] of Object.entries(fixture.sessions)) {
    const loc = path.join(CACHE, name, "loc.csv");
    it.skipIf(!fs.existsSync(loc))(`cached session ${name}`, () => {
      const track = parseLocCsv(fs.readFileSync(loc, "utf8"));
      expect(track.t_ms.length).toBe(s.n_fixes);
      expectSameSegments(detectStays(track.t_ms, track.lat, track.lon), s.segments);
    });
  }
});

describe("parseNmea matches nahori parse_nmea_txt mean_snr", () => {
  for (const [name, s] of Object.entries(fixture.sessions)) {
    const file = path.join(CACHE, name, "nmea.txt");
    it.skipIf(!s.nmea || !fs.existsSync(file))(`cached session ${name}`, () => {
      // Offset 0 keeps the wall-clock-as-UTC ms nahori reports.
      const p = parseNmea(fs.readFileSync(file, "utf8"), 0);
      const tracked = Array.from(p.t_ms).flatMap((t, i) => (p.snr[i] > 0 ? [[t, p.snr[i]]] : []));
      expect(tracked.length).toBe(s.nmea!.count);
      expect(tracked.reduce((a, [, v]) => a + v, 0)).toBeCloseTo(s.nmea!.sum, 6);
      const byT = new Map(tracked as [number, number][]);
      for (const [t, v] of s.nmea!.samples) expect(byT.get(t)).toBeCloseTo(v, 9);
    });
  }

  it("zero-fills seconds whose GSV tracked nothing and drops seconds without GSV", () => {
    const text = [
      "header",
      "20261021212602880|$GPGSV,1,1,02,25,78,031,30,29,55,307,40*70",
      "20261021212602890|$GLGSV,1,1,01,74,81,036,20*66",
      "20261021212603100|$GPGSV,1,1,01,25,78,031,*70",
      "20261021212604100|$GPGGA,182602.37,,,,,0,00,999.9,,M,,M,,*6D",
    ].join("\n");
    const p = parseNmea(text, 180);
    expect(Array.from(p.t_ms)).toEqual([Date.UTC(2026, 9, 21, 18, 26, 2), Date.UTC(2026, 9, 21, 18, 26, 3)]);
    expect(Array.from(p.snr)).toEqual([30, 0]);
    expect(Array.from(p.sats)).toEqual([3, 0]);
  });
});

describe("time helpers", () => {
  it("reads the device offset from a Skydex CSV", () => {
    expect(skydexUtcOffsetMin('"timestamp","x"\n"2026-10-21T21:26:14.982+0300","1"')).toBe(180);
    expect(skydexUtcOffsetMin('"2026-01-02T03:04:05.000-0530"')).toBe(-330);
    expect(skydexUtcOffsetMin("no times here")).toBeNull();
  });
  it("zone offsets follow DST", () => {
    expect(zoneOffsetMin(Date.UTC(2026, 6, 1), "Asia/Jerusalem")).toBe(180);
    expect(zoneOffsetMin(Date.UTC(2026, 11, 1), "Asia/Jerusalem")).toBe(120);
  });
  it("keeps the most accurate fix per second", () => {
    const csv = [
      '"timestamp","timeUTC","provider","longitude","latitude","altitude","speed","bearing","accuracy"',
      '"2026-10-21T21:26:14.100+0300","1","gps","34.1","31.1","0","0.5","0","20.0"',
      '"2026-10-21T21:26:14.900+0300","1","fused","34.2","31.2","0","","0","8.0"',
      '"2026-10-21T21:26:15.000+0300","1","network","34.3","31.3","0","0","0",""',
      '"2026-10-21T21:26:16.000+0300","1","gps","","","0","0","0","3.0"',
    ].join("\n");
    const t = parseLocCsv(csv);
    expect(Array.from(t.lon)).toEqual([34.2, 34.3]);
    expect(Array.from(t.t_ms)).toEqual([Date.UTC(2026, 9, 21, 18, 26, 14, 900), Date.UTC(2026, 9, 21, 18, 26, 15)]);
    expect(Number.isNaN(t.speed[0])).toBe(true);
  });
});
