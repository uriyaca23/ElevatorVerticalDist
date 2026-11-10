// Parity between lib/reconstruct.ts and pyramidElevatorDist.reconstructedSignal
// (reconstruct="Mahony"). Fixtures come from scripts/make_reconstruct_fixture.py.
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { limitDenominator, reconstructVertical, resamplePoly, type Xyz } from "@/lib/reconstruct";

interface Cols { timestamp_ms: number[]; x: number[]; y: number[]; z: number[] }
interface Out { timestamp_ms: number[]; a_vert: number[]; a_mag_g: number[] }
const fixture: {
  synthetic: { acc: Cols; gyr: Cols; out: Out };
  experiments: Record<string, Out>;
  resample_poly: { x: number[]; up: number; down: number; y: number[] }[];
  limit_denominator: [number, number, number][];
} = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures/reconstruct_parity.json"), "utf8"));

const DATA_DIR = path.resolve(__dirname, "../../src/data/structuredData/data");
const xyz = (c: Cols): Xyz => ({ t: c.timestamp_ms, x: c.x, y: c.y, z: c.z });

function readXyz(file: string): Xyz {
  const lines = fs.readFileSync(file, "utf8").trim().split("\n").slice(1);
  const c: Cols = { timestamp_ms: [], x: [], y: [], z: [] };
  for (const l of lines) {
    const [t, x, y, z] = l.split(",");
    c.timestamp_ms.push(Number(t));
    c.x.push(Number(x));
    c.y.push(Number(y));
    c.z.push(Number(z));
  }
  return xyz(c);
}

/** Compare TS output against Python samples matched by timestamp. */
function expectClose(ours: ReturnType<typeof reconstructVertical>, expected: Out, tol: number) {
  const index = new Map<number, number>();
  ours.t.forEach((t, i) => index.set(t, i));
  let worstRec = 0;
  let worstMag = 0;
  for (let k = 0; k < expected.timestamp_ms.length; k++) {
    const i = index.get(expected.timestamp_ms[k]);
    expect(i, `timestamp ${expected.timestamp_ms[k]} missing`).toBeDefined();
    worstRec = Math.max(worstRec, Math.abs(ours.a_rec[i!] - expected.a_vert[k]));
    worstMag = Math.max(worstMag, Math.abs(ours.a_mag_g[i!] - expected.a_mag_g[k]));
  }
  expect(worstRec).toBeLessThan(tol);
  expect(worstMag).toBeLessThan(tol);
}

describe("reconstructVertical matches Python reconstructedSignal(Mahony)", () => {
  it("synthetic tilted ride", () => {
    const { acc, gyr, out } = fixture.synthetic;
    const ours = reconstructVertical(xyz(acc), xyz(gyr));
    expect(ours.t.length).toBe(out.timestamp_ms.length);
    expectClose(ours, out, 1e-6);
  });

  for (const [name, out] of Object.entries(fixture.experiments)) {
    const dir = path.join(DATA_DIR, name);
    it.skipIf(!fs.existsSync(path.join(dir, "ACC.csv")))(`experiment ${name}`, () => {
      const ours = reconstructVertical(readXyz(path.join(dir, "ACC.csv")), readXyz(path.join(dir, "GYR.csv")));
      expectClose(ours, out, 1e-4);
    });
  }
});

describe("scipy / fractions building blocks", () => {
  it.each(fixture.limit_denominator)("Fraction(%f).limit_denominator(1000) = %i/%i", (x, n, d) => {
    expect(limitDenominator(x, 1000)).toEqual([n, d]);
  });
  it.each(fixture.resample_poly.map((c) => [c.up, c.down, c] as const))("resample_poly up=%i down=%i", (_u, _d, c) => {
    const y = resamplePoly(c.x, c.up, c.down);
    expect(y.length).toBe(c.y.length);
    y.forEach((v, i) => expect(v).toBeCloseTo(c.y[i], 10));
  });
});
