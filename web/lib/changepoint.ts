// Offline changepoint detection for the NMEA mean-C/N0 trace: PELT (Killick,
// Fearnhead & Eckley 2012) on a Gaussian mean-shift (L2) cost, with a
// minimum segment length. Exact: it returns the same partition as optimal
// partitioning (tests/changepoint.test.ts checks it against brute force).

export interface CpSegment {
  /** Sample range [start, end). */
  start: number;
  end: number;
  mean: number;
}

/**
 * Minimise Σ_segments SSE(segment) + penalty · (#segments − 1), every segment
 * at least `minSize` samples long. Returns the segments in order.
 */
export function pelt(y: ArrayLike<number>, penalty: number, minSize = 1): CpSegment[] {
  const n = y.length;
  if (n === 0) return [];
  minSize = Math.max(1, Math.floor(minSize));
  const S = new Float64Array(n + 1);
  const S2 = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    S[i + 1] = S[i] + y[i];
    S2[i + 1] = S2[i] + y[i] * y[i];
  }
  const cost = (a: number, b: number) => {
    const s = S[b] - S[a];
    return Math.max(S2[b] - S2[a] - (s * s) / (b - a), 0);
  };
  const mean = (a: number, b: number) => (S[b] - S[a]) / (b - a);
  if (n < 2 * minSize) return [{ start: 0, end: n, mean: mean(0, n) }];

  const F = new Float64Array(n + 1).fill(Infinity);
  F[0] = -penalty;
  const last = new Int32Array(n + 1);
  let R: number[] = [0];

  for (let t = minSize; t <= n; t++) {
    // Prune with u = t − minSize rather than t: a start s that loses to u
    // there loses to it for every end ≥ t, where [u, end) is long enough to
    // be a segment. Pruning at t itself would drop starts still needed for
    // ends in (t, t + minSize), where t is not yet an admissible start.
    const u = t - minSize;
    if (u >= minSize) R = R.filter((s) => s >= u || F[s] + cost(s, u) <= F[u]);
    let best = Infinity;
    let arg = 0;
    for (const s of R) {
      if (t - s < minSize) continue;
      const v = F[s] + cost(s, t) + penalty;
      if (v < best) {
        best = v;
        arg = s;
      }
    }
    F[t] = best;
    last[t] = arg;
    R.push(t);
  }

  const bounds: number[] = [];
  for (let t = n; t > 0; t = last[t]) bounds.push(t);
  bounds.push(0);
  bounds.reverse();
  return bounds.slice(1).map((end, k) => ({ start: bounds[k], end, mean: mean(bounds[k], end) }));
}

/** Noise scale of the per-second mean C/N0 within one environment, dB-Hz. */
export const SNR_SIGMA_DBHZ = 3.0;
/** Shortest segment, in samples (≈ seconds at the 1 Hz NMEA rate). */
export const SNR_MIN_SEGMENT = 30;
/** Default penalty multiplier β in β·σ²·ln n (BIC is β = 1–2; the series is autocorrelated). */
export const SNR_DEFAULT_BETA = 5;

export interface SnrSegment {
  start_ms: number;
  end_ms: number;
  mean_snr: number;
}

export interface SnrChangepoints {
  segments: SnrSegment[];
  /** Piecewise-constant fit, one value per input sample. */
  fit: number[];
  penalty: number;
}

/** Changepoints of a per-second mean-C/N0 series (`t_ms` ascending). */
export function snrChangepoints(tMs: ArrayLike<number>, snr: ArrayLike<number>, beta = SNR_DEFAULT_BETA): SnrChangepoints {
  const n = snr.length;
  const penalty = beta * SNR_SIGMA_DBHZ ** 2 * Math.log(Math.max(n, 2));
  const segs = pelt(snr, penalty, SNR_MIN_SEGMENT);
  const fit = new Array<number>(n);
  const segments = segs.map((s, k) => {
    for (let i = s.start; i < s.end; i++) fit[i] = s.mean;
    // Tile the time axis: a segment runs to the next one's first sample.
    const end = k + 1 < segs.length ? tMs[segs[k + 1].start] : tMs[s.end - 1] + 1000;
    return { start_ms: tMs[s.start], end_ms: end, mean_snr: s.mean };
  });
  return { segments, fit, penalty };
}
