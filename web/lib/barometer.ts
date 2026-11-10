// Barometer → height → elevator segments.
//
// Faithful TypeScript port of the Python pressure-filter pipeline:
//   - ISA inversion:   pyramidElevatorDist/physics/barometric.py
//   - pre-smoothing:   src/data/loader/pipeline.py::_derive_gt_from_prs (51-sample median)
//   - segmentation:    pyramidElevatorDist/segmentation/algorithms/barometer_only/height_segmentation.py
// Parity is checked by tests/barometer.test.ts against fixtures produced by
// scripts/make_baro_fixture.py. Pure functions only — safe on server and client.

export type RideType = "up" | "down";

export interface Segment {
  start_ms: number;
  end_ms: number;
  type: RideType;
  height_diff_m: number;
}

export interface PressureFilterConfig {
  velocity_threshold: number;
  smooth_window_sec: number;
  height_lowpass_sec: number;
  min_duration_sec: number;
  min_height_diff_m: number;
  merge_gap_sec: number;
  pad_sec: number;
}

// Mirrors pyramidElevatorDist/segmentation/algorithms/config.json.
export const DEFAULT_PRESSURE_FILTER: PressureFilterConfig = {
  velocity_threshold: 0.15,
  smooth_window_sec: 3.0,
  height_lowpass_sec: 8.0,
  min_duration_sec: 3.0,
  min_height_diff_m: 2.0,
  merge_gap_sec: 6.0,
  pad_sec: 1.0,
};

export const MEDIAN_PRESMOOTH_SAMPLES = 51;

const P0_HPA = 1013.25;
const T0_STD_K = 288.15;
const LAPSE_RATE_K_PER_M = 0.0065;

/** ISA pressure (hPa) → altitude (m), standard atmosphere when temperature is null. */
export function pressureToHeight(pressureHpa: number, temperatureC: number | null = null): number {
  const t0 = temperatureC == null ? T0_STD_K : temperatureC + 273.15;
  return (t0 / LAPSE_RATE_K_PER_M) * (1 - Math.pow(pressureHpa / P0_HPA, 1 / 5.255));
}

// pandas `rolling(window=w, center=True)` and `np.convolve(mode="same")` both
// place sample i's window at [i - floor(w/2), i + floor((w-1)/2)].
function windowBounds(i: number, w: number, n: number): [number, number] {
  return [Math.max(0, i - Math.floor(w / 2)), Math.min(n - 1, i + Math.floor((w - 1) / 2))];
}

/** pandas `Series.rolling(w, center=True, min_periods=1).mean()`. */
export function rollingMeanCentered(x: ArrayLike<number>, w: number): Float64Array {
  const n = x.length;
  const out = new Float64Array(n);
  const prefix = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) prefix[i + 1] = prefix[i] + x[i];
  for (let i = 0; i < n; i++) {
    const [lo, hi] = windowBounds(i, w, n);
    out[i] = (prefix[hi + 1] - prefix[lo]) / (hi - lo + 1);
  }
  return out;
}

/** pandas `Series.rolling(w, center=True, min_periods=1).median()`. */
export function rollingMedianCentered(x: ArrayLike<number>, w: number): Float64Array {
  const n = x.length;
  const out = new Float64Array(n);
  const buf: number[] = [];
  for (let i = 0; i < n; i++) {
    const [lo, hi] = windowBounds(i, w, n);
    buf.length = 0;
    for (let j = lo; j <= hi; j++) buf.push(x[j]);
    buf.sort((a, b) => a - b);
    const m = buf.length;
    out[i] = m % 2 ? buf[(m - 1) / 2] : (buf[m / 2 - 1] + buf[m / 2]) / 2;
  }
  return out;
}

/** `np.convolve(x, ones(w)/w, mode="same")` — zero-padded boxcar. */
export function boxcarSame(x: ArrayLike<number>, w: number): Float64Array {
  const n = x.length;
  const out = new Float64Array(n);
  const prefix = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) prefix[i + 1] = prefix[i] + x[i];
  for (let i = 0; i < n; i++) {
    const [lo, hi] = windowBounds(i, w, n);
    out[i] = (prefix[hi + 1] - prefix[lo]) / w;
  }
  return out;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length;
  return m % 2 ? s[(m - 1) / 2] : (s[m / 2 - 1] + s[m / 2]) / 2;
}

function sampleRate(t: ArrayLike<number>): number {
  const pos: number[] = [];
  for (let i = 1; i < t.length; i++) {
    const d = t[i] - t[i - 1];
    if (d > 0) pos.push(d);
  }
  return pos.length ? 1 / median(pos) : 1;
}

interface RawSegment {
  start_sec: number;
  end_sec: number;
  type: RideType;
}

/** Port of HeightSegmenter.segment. `t` in seconds, `z` height in metres. */
export function segmentHeight(
  t: ArrayLike<number>,
  z: ArrayLike<number>,
  config: PressureFilterConfig = DEFAULT_PRESSURE_FILTER,
): RawSegment[] {
  const n = t.length;
  if (n < 2) return [];
  const fs = sampleRate(t);

  const lpWin = Math.max(1, Math.round(config.height_lowpass_sec * fs));
  const zLp = rollingMeanCentered(z, lpWin);
  const vz = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    const dt = t[i] - t[i - 1];
    vz[i] = dt > 0 ? (zLp[i] - zLp[i - 1]) / dt : 0;
  }

  const win = Math.max(1, Math.round(config.smooth_window_sec * fs));
  const vzSmooth = boxcarSame(vz, win);

  const moving = Array.from(vzSmooth, (v) => Math.abs(v) > config.velocity_threshold);
  const starts: number[] = [];
  const ends: number[] = [];
  if (moving[0]) starts.push(0);
  for (let i = 1; i < n; i++) {
    if (moving[i] && !moving[i - 1]) starts.push(i);
    else if (!moving[i] && moving[i - 1]) ends.push(i);
  }
  if (moving[n - 1]) ends.push(n - 1);

  const raw: [number, number, number][] = [];
  for (let k = 0; k < Math.min(starts.length, ends.length); k++) {
    const s = starts[k];
    const e = ends[k];
    let sum = 0;
    for (let i = s; i <= e; i++) sum += vzSmooth[i];
    raw.push([s, e, sum / (e - s + 1) >= 0 ? 1 : -1]);
  }

  const merged: [number, number, number][] = [];
  let lastSign: number | null = null;
  for (const [s, e, sign] of raw) {
    const prev = merged[merged.length - 1];
    if (prev && sign === lastSign && t[s] - t[prev[1]] < config.merge_gap_sec) {
      prev[1] = e;
    } else {
      merged.push([s, e, sign]);
      lastSign = sign;
    }
  }

  const out: RawSegment[] = [];
  for (const [s, e, sign] of merged) {
    if (t[e] - t[s] <= config.min_duration_sec) continue;
    if (Math.abs(z[e] - z[s]) <= config.min_height_diff_m) continue;
    out.push({
      start_sec: Math.max(t[0], t[s] - config.pad_sec),
      end_sec: Math.min(t[n - 1], t[e] + config.pad_sec),
      type: sign > 0 ? "up" : "down",
    });
  }
  return out;
}

export interface BarometerAnalysis {
  /** Raw ISA height (m) per sample. */
  height: Float64Array;
  /** 51-sample rolling-median height — the signal the segmenter sees. */
  heightSmooth: Float64Array;
  segments: Segment[];
}

/**
 * Full pipeline from raw barometer samples, matching `_derive_gt_from_prs`:
 * ISA → 51-sample median → HeightSegmenter, with segment times converted back
 * to absolute epoch ms (truncated like Python's `int()`) and clipped to the recording.
 */
export function analyzeBarometer(
  timestampMs: ArrayLike<number>,
  pressureHpa: ArrayLike<number>,
  config: PressureFilterConfig = DEFAULT_PRESSURE_FILTER,
): BarometerAnalysis {
  const n = timestampMs.length;
  const height = Float64Array.from(pressureHpa as ArrayLike<number>, (p) => pressureToHeight(p));
  const heightSmooth = rollingMedianCentered(height, MEDIAN_PRESMOOTH_SAMPLES);
  if (n < 2) return { height, heightSmooth, segments: [] };

  const t0 = timestampMs[0];
  const tEnd = timestampMs[n - 1];
  const tSec = Float64Array.from(timestampMs as ArrayLike<number>, (ms) => (ms - t0) / 1000);
  const segments: Segment[] = [];
  let cursor = t0;
  for (const r of segmentHeight(tSec, heightSmooth, config)) {
    // Padding can make neighbours overlap; like _segments_to_full_gt, the
    // later ride starts where the earlier one ends.
    const start_ms = Math.max(cursor, Math.trunc(t0 + r.start_sec * 1000));
    const end_ms = Math.min(tEnd, Math.trunc(t0 + r.end_sec * 1000));
    if (end_ms <= start_ms) continue;
    cursor = end_ms;
    segments.push({
      start_ms,
      end_ms,
      type: r.type,
      height_diff_m: heightDiff(timestampMs, heightSmooth, start_ms, end_ms),
    });
  }
  return { height, heightSmooth, segments };
}

/** Height change across [startMs, endMs], read off the nearest samples of `h`. */
export function heightDiff(
  timestampMs: ArrayLike<number>,
  h: ArrayLike<number>,
  startMs: number,
  endMs: number,
): number {
  if (timestampMs.length === 0) return 0;
  return h[nearestIndex(timestampMs, endMs)] - h[nearestIndex(timestampMs, startMs)];
}

/** Index of the sample closest to `ms` in a sorted timestamp array. */
export function nearestIndex(timestampMs: ArrayLike<number>, ms: number): number {
  let lo = 0;
  let hi = timestampMs.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (timestampMs[mid] < ms) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(timestampMs[lo - 1] - ms) <= Math.abs(timestampMs[lo] - ms)) return lo - 1;
  return lo;
}
