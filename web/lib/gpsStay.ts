// Port of gps_stay's ProbabilisticStayDetector.detect (the sibling repo
// ../gps_stay, gps_stay/algorithm/model.py) — STAY / MOVEMENT / UNKNOWN
// segmentation of a GPS track by exact dynamic programming over Student-t
// stay, constant-velocity / step move and diffuse background costs.
//
// Kept line-for-line with the Python so the two stay in parity
// (tests/gpsStay.test.ts, fixture from scripts/make_gps_stay_fixture.py).
// Only the batch defaults are ported: no semi_stay post-process and the
// projection origin is always the track's centroid. Read model.py for the
// reasoning behind each constant; the comments here only flag where the
// TypeScript differs in shape.

export type StayClass = "stay" | "movement" | "unknown";

export interface StaySegment {
  /** First / last fix of the segment, epoch ms. */
  start_ms: number;
  end_ms: number;
  classification: StayClass;
  center_lat: number;
  center_lon: number;
  radius_m: number;
  point_count: number;
  /** Inclusive row range into the input arrays. */
  start_index: number;
  end_index: number;
  /** Calibrated P(label correct); 0 for unknown. */
  probability: number;
}

/** The pinned "same place" radius (gps_stay.STAY_RADIUS_M) — not a knob. */
export const STAY_RADIUS_M = 50.0;
const EARTH_RADIUS_M = 6_371_000.0;
const EPS = 1e-9;
const STAY = 0;
const MOVE = 1;
const UNKNOWN = 2;
const CLASS: StayClass[] = ["stay", "movement", "unknown"];

/** ProbabilisticStayDetector.Config defaults (gps_stay d9c05d9). */
export const STAY_CONFIG = {
  min_duration_seconds: 300.0,
  min_move_speed_mps: 1.0,
  stay_max_spread_factor: 0.55,
  context_spread_factor: 1.0,
  context_max_lateral_accel: 3.0,
  evidence_ref_seconds: 10.0,
  evidence_cap_seconds: 60.0,
  evidence_floor_seconds: 0.5,
  move_speed_scale_mps: 10.0,
  move_speed_rel_sd: 0.3,
  path_noise_floor_meters: 45.0,
  nu: 1.5,
  sigma_meters: 6.0,
  lambda_penalty: 60.0,
  confidence_threshold: 0.5,
  outlier_scale: 9.0,
  outlier_weight_cutoff: 0.1,
  gap_max_seconds: 1800.0,
  stay_gap_max_seconds: 14400.0,
  merge_excursion_seconds: 120.0,
  merge_radius_factor: 2.0,
  prob_w0: -6.02891,
  prob_w_move: -0.79848,
  prob_w_logn: 1.20033,
  prob_w_logd: 2.51444,
  dp_window_seconds: 93600.0,
  dp_window_min_points: 1024,
  dp_max_window_seconds: 259200.0 as number | null,
  target_cadence_seconds: 10.0 as number | null,
};
export type StayConfig = typeof STAY_CONFIG;

// np.radians / np.degrees multiply by these constants; keep the same op.
const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;
const clip = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);

interface Pre {
  w: Float64Array;
  wx: Float64Array;
  wy: Float64Array;
  wxx: Float64Array;
  wyy: Float64Array;
  wxy: Float64Array;
  raw_t: Float64Array;
  raw_w: Float64Array;
  raw_x: Float64Array;
  raw_y: Float64Array;
}

interface PreStep {
  nll: Float64Array;
  d_net: Float64Array;
  g: Float64Array;
}

/** Cumulative sum with a leading zero (np.concatenate([[0], np.cumsum(v)])). */
function cumsum0(v: ArrayLike<number>): Float64Array {
  const out = new Float64Array(v.length + 1);
  for (let i = 0; i < v.length; i++) out[i + 1] = out[i] + v[i];
  return out;
}

/** First index with a[idx] >= x (np.searchsorted side="left"); a ascending. */
function searchLeft(a: Float64Array, x: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

class Detector {
  constructor(private readonly c: StayConfig) {}

  detect(tMs: ArrayLike<number>, lat: ArrayLike<number>, lon: ArrayLike<number>): StaySegment[] {
    const R = STAY_RADIUS_M;
    const nOrig = tMs.length;
    if (nOrig === 0) return [];

    const keep = this.downsampleIndices(nOrig, tMs);
    const n = keep.length;
    if (n < 3) return [this.toStay(0, nOrig, UNKNOWN, tMs, lat, lon, null)];

    let refLat = 0;
    let refLon = 0;
    for (const k of keep) {
      refLat += lat[k];
      refLon += lon[k];
    }
    refLat /= n;
    refLon /= n;
    const cos0 = Math.cos(refLat * DEG);
    const x = new Float64Array(n);
    const y = new Float64Array(n);
    const secs = new Float64Array(n);
    const t0 = tMs[keep[0]];
    for (let q = 0; q < n; q++) {
      const k = keep[q];
      x[q] = (lon[k] - refLon) * DEG * cos0 * EARTH_RADIUS_M;
      y[q] = (lat[k] - refLat) * DEG * EARTH_RADIUS_M;
      secs[q] = (tMs[k] - t0) / 1000;
    }

    const wOut = this.outlierWeights(x, y, secs);
    const ev = this.timeEvidence(secs);
    const w = Float64Array.from(wOut, (v, i) => Math.min(v, 1.0) * ev[i]);
    const pre = this.prefixSums(x, y, secs, w);
    const preStep = this.stepPrefixSums(x, y, secs, wOut);

    const { labels, starts } = this.segment(pre, preStep, secs, R);
    const bounds = [...starts, n];

    let segs: [number, number, number][] = starts.map((i, k) => [
      i,
      bounds[k + 1],
      this.confidenceLabel(pre, preStep, secs, i, bounds[k + 1], labels[k], R),
    ]);
    segs = this.merge(segs, pre, secs, R);

    return segs.map(([i, j, lab]) => {
      const [cx, cy] = this.weightedCentre(pre, i, j);
      const centre = {
        lat: refLat + (cy / EARTH_RADIUS_M) * RAD,
        lon: refLon + (cx / (EARTH_RADIUS_M * cos0)) * RAD,
      };
      return this.toStay(keep[i], keep[j - 1] + 1, lab, tMs, lat, lon, centre);
    });
  }

  // ------------------------------------------------------------ pre-processing

  /** First fix of each target_cadence_seconds bucket of epoch time. */
  private downsampleIndices(n: number, tMs: ArrayLike<number>): number[] {
    const target = this.c.target_cadence_seconds;
    if (target == null) return Array.from({ length: n }, (_, i) => i);
    const keep: number[] = [];
    let prev = NaN;
    for (let i = 0; i < n; i++) {
      const b = Math.floor(tMs[i] / 1000 / target);
      if (i === 0 || b !== prev) keep.push(i);
      prev = b;
    }
    return keep;
  }

  private outlierWeights(x: Float64Array, y: Float64Array, secs: Float64Array): Float64Array {
    const n = x.length;
    if (n < 3) return new Float64Array(n).fill(1);
    const w = this.contextWeight(x, y, secs, 1);
    if (n >= 5) {
      const w2 = this.contextWeight(x, y, secs, 2);
      for (let i = 0; i < n; i++) w[i] = Math.min(w[i], w2[i]);
    }
    for (let i = 0; i < n; i++) if (w[i] < this.c.outlier_weight_cutoff) w[i] = 0;
    return w;
  }

  private contextWeight(x: Float64Array, y: Float64Array, secs: Float64Array, k: number): Float64Array {
    const c = this.c;
    const n = x.length;
    const base = c.outlier_scale * c.sigma_meters ** 2;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let r2 = 0;
      let slack = 0;
      if (i >= k && i < n - k) {
        const dtL = secs[i] - secs[i - k];
        const dtR = secs[i + k] - secs[i];
        const f = dtL / Math.max(dtL + dtR, EPS);
        const px = x[i - k] + (x[i + k] - x[i - k]) * f;
        const py = y[i - k] + (y[i + k] - y[i - k]) * f;
        r2 = (x[i] - px) ** 2 + (y[i] - py) ** 2;
        const chord = Math.hypot(x[i + k] - x[i - k], y[i + k] - y[i - k]);
        const phys = 0.5 * c.context_max_lateral_accel * dtL * dtR;
        slack = Math.min(c.context_spread_factor * chord, phys);
      }
      const s2 = base + slack ** 2;
      out[i] = (c.nu + 2.0) / (c.nu + r2 / s2);
    }
    return out;
  }

  private timeEvidence(secs: Float64Array): Float64Array {
    const c = this.c;
    const n = secs.length;
    if (n < 2) return new Float64Array(n).fill(1);
    const gap = new Float64Array(n - 1);
    for (let i = 0; i < n - 1; i++) gap[i] = clip(secs[i + 1] - secs[i], c.evidence_floor_seconds, c.evidence_cap_seconds);
    const t = new Float64Array(n);
    t[0] = gap[0];
    t[n - 1] = gap[n - 2];
    for (let i = 1; i < n - 1; i++) t[i] = 0.5 * (gap[i - 1] + gap[i]);
    for (let i = 0; i < n; i++) t[i] /= c.evidence_ref_seconds;
    return t;
  }

  private stepPrefixSums(x: Float64Array, y: Float64Array, secs: Float64Array, wOut: Float64Array): PreStep {
    const c = this.c;
    const n = x.length;
    if (n < 2) return { nll: new Float64Array(1), d_net: new Float64Array(1), g: new Float64Array(1) };
    const nll = new Float64Array(n - 1);
    const dNet = new Float64Array(n - 1);
    const gs = new Float64Array(n - 1);
    const twoPi = 2.0 * Math.PI;
    for (let k = 0; k < n - 1; k++) {
      const g = Math.max(secs[k + 1] - secs[k], EPS);
      const d = Math.hypot(x[k + 1] - x[k], y[k + 1] - y[k]);
      const mu = c.move_speed_scale_mps * g;
      const sRing = c.move_speed_rel_sd * Math.max(mu, d);
      const s2 = sRing ** 2 + 2.0 * c.sigma_meters ** 2;
      const v = (d - mu) ** 2 / (2.0 * s2) + 0.5 * Math.log(twoPi * s2) + Math.log(twoPi * Math.max(d, 1.0));
      const tEv = clip(g, c.evidence_floor_seconds, c.evidence_cap_seconds) / c.evidence_ref_seconds;
      const wStep = Math.min(Math.min(wOut[k], wOut[k + 1]), 1.0);
      nll[k] = tEv * v;
      dNet[k] = wStep * Math.max(d - c.path_noise_floor_meters, 0.0);
      gs[k] = g;
    }
    return { nll: cumsum0(nll), d_net: cumsum0(dNet), g: cumsum0(gs) };
  }

  private prefixSums(x: Float64Array, y: Float64Array, t: Float64Array, w: Float64Array): Pre {
    const n = w.length;
    const wx = new Float64Array(n);
    const wy = new Float64Array(n);
    const wxx = new Float64Array(n);
    const wyy = new Float64Array(n);
    const wxy = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      wx[i] = w[i] * x[i];
      wy[i] = w[i] * y[i];
      wxx[i] = w[i] * x[i] * x[i];
      wyy[i] = w[i] * y[i] * y[i];
      wxy[i] = w[i] * x[i] * y[i];
    }
    return {
      w: cumsum0(w),
      wx: cumsum0(wx),
      wy: cumsum0(wy),
      wxx: cumsum0(wxx),
      wyy: cumsum0(wyy),
      wxy: cumsum0(wxy),
      raw_t: t,
      raw_w: w,
      raw_x: x,
      raw_y: y,
    };
  }

  // ------------------------------------------------------------ segment costs

  private majorStd(pre: Pre, i: number, j: number): number {
    const Sw = pre.w[j] - pre.w[i] + EPS;
    const mx = (pre.wx[j] - pre.wx[i]) / Sw;
    const my = (pre.wy[j] - pre.wy[i]) / Sw;
    const cxx = (pre.wxx[j] - pre.wxx[i]) / Sw - mx * mx;
    const cyy = (pre.wyy[j] - pre.wyy[i]) / Sw - my * my;
    const cxy = (pre.wxy[j] - pre.wxy[i]) / Sw - mx * my;
    const half = (cxx + cyy) / 2.0;
    const radius = Math.sqrt(Math.max(((cxx - cyy) / 2.0) ** 2 + cxy * cxy, 0.0));
    return Math.sqrt(Math.max(half + radius, 0.0));
  }

  private stayCost(pre: Pre, i: number, j: number, R: number): number {
    const Sw = pre.w[j] - pre.w[i] + EPS;
    const Swx = pre.wx[j] - pre.wx[i];
    const Swy = pre.wy[j] - pre.wy[i];
    const Swxx = pre.wxx[j] - pre.wxx[i];
    const Swyy = pre.wyy[j] - pre.wyy[i];
    const sse = Swxx + Swyy - (Swx * Swx + Swy * Swy) / Sw;
    const varS = this.c.sigma_meters ** 2 + (R * R) / 4.0;
    const kappa = Math.log(2 * Math.PI * varS);
    return Math.max(sse, 0.0) / (2 * varS) + Sw * kappa;
  }

  /**
   * Line-fit MOVE cost and speed of [i, j), given the tail sums of w·τ, w·τ²,
   * w·τ·x, w·τ·y over [i, j) on a clock re-zeroed at fix j-1 (model.py
   * _move_cost; the caller accumulates the tails so a scan stays O(window)).
   */
  private moveCost(pre: Pre, i: number, j: number, Swt: number, Swtt: number, Swtx: number, Swty: number): [number, number] {
    const Sw = pre.w[j] - pre.w[i];
    const Swx = pre.wx[j] - pre.wx[i];
    const Swy = pre.wy[j] - pre.wy[i];
    const Swxx = pre.wxx[j] - pre.wxx[i];
    const Swyy = pre.wyy[j] - pre.wyy[i];
    const det = Sw * Swtt - Swt * Swt;
    const detSafe = Math.abs(det) < EPS ? NaN : det;
    const bx = (Sw * Swtx - Swt * Swx) / detSafe;
    const by = (Sw * Swty - Swt * Swy) / detSafe;
    const ax = (Swtt * Swx - Swt * Swtx) / detSafe;
    const ay = (Swtt * Swy - Swt * Swty) / detSafe;
    const sse = Swxx - ax * Swx - bx * Swtx + (Swyy - ay * Swy - by * Swty);
    const varV = this.c.sigma_meters ** 2;
    const kappa = Math.log(2 * Math.PI * varV);
    // Math.max(NaN, 0) is NaN, like np.maximum.
    return [Math.max(sse, 0.0) / (2 * varV) + Sw * kappa, Math.hypot(bx, by)];
  }

  /** Tail sums of the line-fit time moments for a single start i (no scan). */
  private moveCostAt(pre: Pre, i: number, j: number): [number, number] {
    let Swt = 0;
    let Swtt = 0;
    let Swtx = 0;
    let Swty = 0;
    const tEnd = pre.raw_t[j - 1];
    for (let p = j - 1; p >= i; p--) {
      const tau = pre.raw_t[p] - tEnd;
      const wt = pre.raw_w[p] * tau;
      Swt += wt;
      Swtt += wt * tau;
      Swtx += wt * pre.raw_x[p];
      Swty += wt * pre.raw_y[p];
    }
    return this.moveCost(pre, i, j, Swt, Swtt, Swtx, Swty);
  }

  // ------------------------------------------------------------ dynamic program

  private segment(pre: Pre, preStep: PreStep, secs: Float64Array, R: number): { labels: number[]; starts: number[] } {
    const c = this.c;
    const n = secs.length;
    const logArea = Math.log(50_000.0 ** 2);
    const lam = c.lambda_penalty;
    const F = new Float64Array(n + 1).fill(Infinity);
    F[0] = 0.0;
    const backI = new Int32Array(n + 1).fill(-1);
    const backLab = new Int8Array(n + 1).fill(UNKNOWN);

    const gapCum = new Int32Array(n);
    const stayGapCum = new Int32Array(n);
    for (let i = 1; i < n; i++) {
      const dt = secs[i] - secs[i - 1];
      gapCum[i] = gapCum[i - 1] + (dt > c.gap_max_seconds ? 1 : 0);
      stayGapCum[i] = stayGapCum[i - 1] + (dt > c.stay_gap_max_seconds ? 1 : 0);
    }
    // _window_starts / _window_limits
    const loArr = new Int32Array(n);
    const limArr = new Int32Array(n);
    for (let q = 0; q < n; q++) {
      const lo = Math.max(Math.min(searchLeft(secs, secs[q] - c.dp_window_seconds), q + 1 - c.dp_window_min_points), 0);
      loArr[q] = lo;
      limArr[q] = c.dp_max_window_seconds == null ? 0 : Math.min(searchLeft(secs, secs[q] - c.dp_max_window_seconds), lo);
    }
    const spreadMax = c.stay_max_spread_factor * R;

    /** Best (cost, start, label) over starts [lo, hi) for a segment ending at j. */
    const scan = (lo: number, hi: number, j: number): [number, number, number] => {
      let best = Infinity;
      let bestI = -1;
      let bestLab = UNKNOWN;
      // Tail sums for the line-fit cost run over [i, j), so accumulate from the
      // segment end down — the order np.cumsum(m[:, ::-1]) adds in.
      let Swt = 0;
      let Swtt = 0;
      let Swtx = 0;
      let Swty = 0;
      const tEnd = pre.raw_t[j - 1];
      for (let i = j - 1; i >= lo; i--) {
        const tau = pre.raw_t[i] - tEnd;
        const wt = pre.raw_w[i] * tau;
        Swt += wt;
        Swtt += wt * tau;
        Swtx += wt * pre.raw_x[i];
        Swty += wt * pre.raw_y[i];
        if (i >= hi) continue;

        const noGap = gapCum[j - 1] - gapCum[i] === 0;
        const noStayGap = stayGapCum[j - 1] - stayGapCum[i] === 0;
        const dur = secs[j - 1] - secs[i];
        const nij = j - i;

        let s = Infinity;
        if (noStayGap && dur >= c.min_duration_seconds && nij >= 3 && this.majorStd(pre, i, j) <= spreadMax) {
          s = this.stayCost(pre, i, j, R);
        }
        let m = Infinity;
        let u = Infinity;
        if (noGap) {
          u = (pre.w[j] - pre.w[i]) * logArea;
          if (nij >= 3) {
            const [cLine, speed] = this.moveCost(pre, i, j, Swt, Swtt, Swtx, Swty);
            const cStep = preStep.nll[j - 1] - preStep.nll[i];
            const cMove = Number.isFinite(cLine) ? Math.min(cLine, cStep) : cStep;
            const d = preStep.d_net[j - 1] - preStep.d_net[i];
            const g = preStep.g[j - 1] - preStep.g[i];
            const gate = Math.max(Number.isFinite(speed) ? speed : 0.0, d / Math.max(g, EPS));
            if (gate >= c.min_move_speed_mps && Number.isFinite(cMove)) m = cMove;
          }
        }
        const total = F[i] + Math.min(Math.min(s, m), u) + lam;
        // <= while walking down keeps the earliest start on ties (np.argmin).
        if (total <= best) {
          best = total;
          bestI = i;
          bestLab = s <= m && s <= u ? STAY : m <= u ? MOVE : UNKNOWN;
        }
      }
      return [best, bestI, bestLab];
    };

    let reach = 0;
    for (let j = 1; j <= n; j++) {
      const lim = limArr[j - 1];
      let lo = Math.max(lim, Math.min(loArr[j - 1], j - reach));
      let [cost, start, lab] = scan(lo, j, j);
      while (lo > lim && (start === lo || (j - start) * 2 > j - lo)) {
        reach = Math.max((j - lo) * 2, reach * 2, 2);
        const newLo = Math.max(lim, j - reach);
        const [c2, s2, l2] = scan(newLo, lo, j);
        if (c2 <= cost) [cost, start, lab] = [c2, s2, l2];
        lo = newLo;
      }
      if ((j - start) * 4 < j - lo) reach = Math.floor(reach / 2);
      F[j] = cost;
      backI[j] = start;
      backLab[j] = lab;
    }

    const starts: number[] = [];
    const labels: number[] = [];
    for (let j = n; j > 0; j = backI[j]) {
      starts.push(backI[j]);
      labels.push(backLab[j]);
    }
    return { starts: starts.reverse(), labels: labels.reverse() };
  }

  // ------------------------------------------------------------ output

  /** Reject option: keep the DP label unless the background explains [i, j) as well. */
  private confidenceLabel(pre: Pre, preStep: PreStep, secs: Float64Array, i: number, j: number, lab: number, R: number): number {
    if (lab === UNKNOWN) return UNKNOWN;
    let cLab: number;
    if (lab === STAY) cLab = this.stayCost(pre, i, j, R);
    else {
      const [cm] = this.moveCostAt(pre, i, j);
      const cs = preStep.nll[j - 1] - preStep.nll[i];
      cLab = Number.isFinite(cm) ? Math.min(cm, cs) : cs;
    }
    const sw = pre.w[j] - pre.w[i];
    const cUnk = sw * Math.log(50_000.0 ** 2);
    const margin = (cUnk - cLab) / Math.max(sw, EPS);
    const vsBg = 1.0 / (1.0 + Math.exp(-clip(margin, -60.0, 60.0)));
    return vsBg < this.c.confidence_threshold ? UNKNOWN : lab;
  }

  private weightedCentre(pre: Pre, i: number, j: number): [number, number] {
    const Sw = pre.w[j] - pre.w[i] + EPS;
    return [(pre.wx[j] - pre.wx[i]) / Sw, (pre.wy[j] - pre.wy[i]) / Sw];
  }

  private samePlace(pre: Pre, i1: number, j1: number, i2: number, j2: number, R: number): boolean {
    const [ax, ay] = this.weightedCentre(pre, i1, j1);
    const [bx, by] = this.weightedCentre(pre, i2, j2);
    return Math.hypot(ax - bx, ay - by) <= this.c.merge_radius_factor * R;
  }

  /** Same-class neighbours, and same-place stays across a brief excursion (model.py _merge). */
  private merge(segs: [number, number, number][], pre: Pre, secs: Float64Array, R: number): [number, number, number][] {
    const c = this.c;
    const spreadMax = c.stay_max_spread_factor * R;
    const out: [number, number, number][] = [];
    for (const [i, j, lab] of segs) {
      if (out.length) {
        const [pi, pj, plab] = out[out.length - 1];
        if (lab === plab && (lab !== STAY || (this.samePlace(pre, pi, pj, i, j, R) && this.majorStd(pre, pi, j) <= spreadMax))) {
          out[out.length - 1] = [pi, j, lab];
          continue;
        }
      }
      if (lab === STAY) {
        let k = -1;
        for (let q = out.length - 1; q >= 0; q--) {
          if (out[q][2] === STAY) {
            k = q;
            break;
          }
        }
        if (k >= 0) {
          const between = out.slice(k + 1);
          let bdur = 0;
          for (const b of between) bdur += secs[b[1] - 1] - secs[b[0]];
          if (
            !between.some((b) => b[2] === STAY) &&
            bdur <= c.merge_excursion_seconds &&
            this.samePlace(pre, out[k][0], out[k][1], i, j, R) &&
            this.majorStd(pre, out[k][0], j) <= spreadMax
          ) {
            out[k] = [out[k][0], j, STAY];
            out.length = k + 1;
            continue;
          }
        }
      }
      out.push([i, j, lab]);
    }
    return out;
  }

  private labelProbability(lab: number, durationS: number): number {
    if (lab === UNKNOWN) return 0.0;
    const c = this.c;
    const nEff = Math.max(durationS, 0.0) / c.evidence_ref_seconds;
    const z =
      c.prob_w0 +
      c.prob_w_move * (lab === MOVE ? 1.0 : 0.0) +
      c.prob_w_logn * Math.log10(Math.max(nEff, 1.0)) +
      c.prob_w_logd * Math.log10(Math.max(durationS, 0.0) + 1.0);
    return 1.0 / (1.0 + Math.exp(-clip(z, -60.0, 60.0)));
  }

  private toStay(
    oStart: number,
    oEndExcl: number,
    lab: number,
    tMs: ArrayLike<number>,
    lat: ArrayLike<number>,
    lon: ArrayLike<number>,
    centre: { lat: number; lon: number } | null,
  ): StaySegment {
    if (!centre) {
      // Fallback: plain mean of the covered original points.
      let a = 0;
      let b = 0;
      for (let i = oStart; i < oEndExcl; i++) {
        a += lat[i];
        b += lon[i];
      }
      centre = { lat: a / (oEndExcl - oStart), lon: b / (oEndExcl - oStart) };
    }
    const start = tMs[oStart];
    const end = tMs[oEndExcl - 1];
    return {
      start_ms: start,
      end_ms: end,
      classification: CLASS[lab],
      center_lat: centre.lat,
      center_lon: centre.lon,
      radius_m: STAY_RADIUS_M,
      point_count: oEndExcl - oStart,
      start_index: oStart,
      end_index: oEndExcl - 1,
      probability: clip(this.labelProbability(lab, (end - start) / 1000), 0, 1),
    };
  }
}

/**
 * Segment a GPS track into stay / movement / unknown. `tMs` must be ascending
 * epoch ms (one fix per instant); the arrays are parallel.
 */
export function detectStays(
  tMs: ArrayLike<number>,
  lat: ArrayLike<number>,
  lon: ArrayLike<number>,
  config: Partial<StayConfig> = {},
): StaySegment[] {
  return new Detector({ ...STAY_CONFIG, ...config }).detect(tMs, lat, lon);
}
