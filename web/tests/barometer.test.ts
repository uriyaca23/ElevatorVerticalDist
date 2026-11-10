// Parity between lib/barometer.ts and the Python pressure-filter pipeline.
// Fixtures come from scripts/make_baro_fixture.py; regenerate after changing
// either implementation.
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { analyzeBarometer, boxcarSame, rollingMeanCentered, rollingMedianCentered } from "@/lib/barometer";

interface Ride { start_ms: number; end_ms: number; type: string }
const fixture: {
  synthetic: { timestamp_ms: number[]; pressure: number[]; rides: Ride[] };
  experiments: Record<string, Ride[]>;
} = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures/baro_parity.json"), "utf8"));

const DATA_DIR = path.resolve(__dirname, "../../src/data/structuredData/data");

function expectSameRides(actual: Ride[], expected: Ride[]) {
  expect(actual.map((r) => r.type)).toEqual(expected.map((r) => r.type));
  actual.forEach((r, i) => {
    // 1 ms of slack for float → int truncation differences.
    expect(Math.abs(r.start_ms - expected[i].start_ms)).toBeLessThanOrEqual(1);
    expect(Math.abs(r.end_ms - expected[i].end_ms)).toBeLessThanOrEqual(1);
  });
}

describe("window helpers match pandas / numpy alignment", () => {
  const x = [0, 1, 2, 3, 4, 5];
  it("rolling mean, even window, min_periods=1", () => {
    // pandas: Series(range(6)).rolling(4, center=True, min_periods=1).mean()
    expect(Array.from(rollingMeanCentered(x, 4))).toEqual([0.5, 1, 1.5, 2.5, 3.5, 4]);
  });
  it("rolling median, odd window", () => {
    expect(Array.from(rollingMedianCentered([5, 1, 4, 2, 3], 3))).toEqual([3, 4, 2, 3, 2.5]);
  });
  it("np.convolve same, even kernel, zero-padded", () => {
    const impulse = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
    expect(Array.from(boxcarSame(impulse, 4), (v) => v * 4)).toEqual([0, 0, 0, 0, 1, 1, 1, 1, 0, 0]);
  });
});

describe("analyzeBarometer matches Python _derive_gt_from_prs", () => {
  it("synthetic trace", () => {
    const { timestamp_ms, pressure, rides } = fixture.synthetic;
    expectSameRides(analyzeBarometer(timestamp_ms, pressure).segments, rides);
  });

  for (const [name, rides] of Object.entries(fixture.experiments)) {
    const csv = path.join(DATA_DIR, name, "PRS.csv");
    it.skipIf(!fs.existsSync(csv))(`experiment ${name}`, () => {
      const lines = fs.readFileSync(csv, "utf8").trim().split("\n").slice(1);
      const ts: number[] = [];
      const p: number[] = [];
      for (const line of lines) {
        const [t, pr] = line.split(",");
        ts.push(Number(t));
        p.push(Number(pr));
      }
      expectSameRides(analyzeBarometer(ts, p).segments, rides);
    });
  }
});
