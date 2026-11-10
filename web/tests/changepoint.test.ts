// PELT (lib/changepoint.ts) must return the exact optimal-partitioning answer,
// minimum segment length included.
import { describe, expect, it } from "vitest";
import { pelt, snrChangepoints } from "@/lib/changepoint";

/** O(n²) optimal partitioning: the reference PELT has to match. */
function bruteForce(y: number[], pen: number, minSize: number): number {
  const n = y.length;
  const sse = (a: number, b: number) => {
    const s = y.slice(a, b);
    const m = s.reduce((p, v) => p + v, 0) / s.length;
    return s.reduce((p, v) => p + (v - m) ** 2, 0);
  };
  const F = new Array(n + 1).fill(Infinity);
  F[0] = -pen;
  for (let t = minSize; t <= n; t++) {
    for (let s = 0; s <= t - minSize; s++) {
      if (s !== 0 && s < minSize) continue;
      F[t] = Math.min(F[t], F[s] + sse(s, t) + pen);
    }
  }
  return F[n];
}

function objective(y: number[], segs: { start: number; end: number; mean: number }[], pen: number) {
  return segs.reduce((acc, s) => acc + y.slice(s.start, s.end).reduce((p, v) => p + (v - s.mean) ** 2, 0), 0) + pen * (segs.length - 1);
}

/** Deterministic LCG so failures reproduce. */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

describe("pelt", () => {
  it("finds clean steps", () => {
    const y = [...Array(40).fill(30), ...Array(25).fill(0), ...Array(35).fill(28)];
    const segs = pelt(y, 50, 10);
    expect(segs.map((s) => [s.start, s.end])).toEqual([
      [0, 40],
      [40, 65],
      [65, 100],
    ]);
    expect(segs.map((s) => s.mean)).toEqual([30, 0, 28]);
  });

  it("matches brute-force optimal partitioning", () => {
    const r = rng(42);
    for (let trial = 0; trial < 60; trial++) {
      const n = 20 + Math.floor(r() * 120);
      const y: number[] = [];
      let level = 25;
      for (let i = 0; i < n; i++) {
        if (r() < 0.05) level = r() < 0.3 ? 0 : 15 + r() * 20;
        y.push(level + (r() - 0.5) * 8);
      }
      const minSize = 1 + Math.floor(r() * 12);
      const pen = 5 + r() * 400;
      const segs = pelt(y, pen, minSize);
      expect(segs[0].start).toBe(0);
      expect(segs[segs.length - 1].end).toBe(n);
      for (const s of segs) expect(s.end - s.start).toBeGreaterThanOrEqual(Math.min(minSize, n));
      expect(objective(y, segs, pen)).toBeCloseTo(bruteForce(y, pen, minSize), 6);
    }
  });

  it("returns one segment when no split is admissible", () => {
    expect(pelt([1, 2, 3, 4, 5], 0, 3)).toEqual([{ start: 0, end: 5, mean: 3 }]);
    expect(pelt([], 1, 1)).toEqual([]);
  });
});

describe("snrChangepoints", () => {
  it("finds the drop and tiles time", () => {
    const n = 300;
    const t = Array.from({ length: n }, (_, i) => 1_000_000 + i * 1000);
    const snr = t.map((_, i) => (i >= 100 && i < 200 ? (i % 3 ? 0 : 12) : 30 + (i % 5)));
    const { segments, fit } = snrChangepoints(t, snr);
    expect(segments).toHaveLength(3);
    expect(segments[0].start_ms).toBe(t[0]);
    expect(segments[1].start_ms).toBe(t[100]);
    expect(segments[0].end_ms).toBe(segments[1].start_ms);
    expect(segments[2].end_ms).toBe(t[n - 1] + 1000);
    expect(fit).toHaveLength(n);
    expect(segments[1].mean_snr).toBeLessThan(10);
  });
});
