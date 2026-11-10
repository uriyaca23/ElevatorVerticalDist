// Accelerometer + gyroscope → reconstructed vertical acceleration (a_rec).
//
// TypeScript port of the signal the Python UIs draw:
//   pyramidElevatorDist.reconstructedSignal(acc, gyro, reconstruct="Mahony")
//     = _orchestration.signal: resample_acc (50 Hz, gap-aware)
//       → reconstruct_az("Mahony")  (physics/reconstruct_az/{base,mahony,_quaternion}.py)
//       → _vertical_accel / _a_mag_minus_g  (template_match/.../common.py)
// Parity is checked by tests/reconstruct.test.ts against
// scripts/make_reconstruct_fixture.py. Pure functions — no I/O.
//
// One deliberate difference: for *uniformly* sampled input Python resamples
// with scipy.signal.resample_poly; here we always interpolate linearly. Phone
// streams are jittery enough to take Python's np.interp branch anyway.

import { rollingMeanCentered } from "@/lib/barometer";

export const RESAMPLE_HZ = 50;
const GAP_MS = 1000;
const GRAVITY = 9.80665;
const DETREND_SEC = 8;

type Vec3 = [number, number, number];
type Quat = [number, number, number, number]; // [w, x, y, z], body → world

export interface Xyz {
  t: ArrayLike<number>; // epoch ms, ascending
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  z: ArrayLike<number>;
}

export interface VerticalSignal {
  t: Float64Array; // epoch ms on the 50 Hz grid (with holes at gaps)
  x: Float64Array; // resampled raw accel — what the algorithm sees
  y: Float64Array;
  z: Float64Array;
  a_rec: Float64Array; // Mahony-reconstructed vertical acceleration, m/s²
  a_mag_g: Float64Array; // |a| − g, rotation invariant
}

// ---------- numpy-ish helpers ----------

function median(xs: ArrayLike<number>): number {
  const s = Array.from(xs).sort((a, b) => a - b);
  const m = s.length;
  if (!m) return NaN;
  return m % 2 ? s[(m - 1) / 2] : (s[m / 2 - 1] + s[m / 2]) / 2;
}

/** np.interp(xNew, xp, fp): linear, clamped to the end values. */
export function interp(xNew: ArrayLike<number>, xp: ArrayLike<number>, fp: ArrayLike<number>): Float64Array {
  const out = new Float64Array(xNew.length);
  const n = xp.length;
  let j = 0;
  for (let i = 0; i < xNew.length; i++) {
    const x = xNew[i];
    if (x <= xp[0]) {
      out[i] = fp[0];
      continue;
    }
    if (x >= xp[n - 1]) {
      out[i] = fp[n - 1];
      continue;
    }
    while (j < n - 2 && xp[j + 1] < x) j++;
    while (j > 0 && xp[j] > x) j--;
    const x0 = xp[j];
    const x1 = xp[j + 1];
    out[i] = x1 === x0 ? fp[j] : fp[j] + ((fp[j + 1] - fp[j]) * (x - x0)) / (x1 - x0);
  }
  return out;
}

// ---------- scipy.signal.resample_poly ----------

/** Exact Fraction(x).limit_denominator(maxDen) for a JS double (CPython algorithm). */
export function limitDenominator(x: number, maxDen: number): [number, number] {
  // Exact rational value of the double.
  const buf = new DataView(new ArrayBuffer(8));
  buf.setFloat64(0, x);
  const bits = buf.getBigUint64(0);
  const exp = Number((bits >> 52n) & 0x7ffn);
  let mant = bits & ((1n << 52n) - 1n);
  if (exp) mant |= 1n << 52n;
  const e = (exp || 1) - 1075;
  let num = x < 0 ? -mant : mant;
  let den = 1n;
  if (e > 0) num <<= BigInt(e);
  else den <<= BigInt(-e);
  const gcd = (a: bigint, b: bigint): bigint => (b === 0n ? (a < 0n ? -a : a) : gcd(b, a % b));
  const g = gcd(num, den);
  num /= g;
  den /= g;
  const M = BigInt(maxDen);
  if (den <= M) return [Number(num), Number(den)];
  let [p0, q0, p1, q1] = [0n, 1n, 1n, 0n];
  let [n, d] = [num, den];
  for (;;) {
    const a = n / d;
    const q2 = q0 + a * q1;
    if (q2 > M) break;
    [p0, q0, p1, q1] = [p1, q1, p0 + a * p1, q2];
    [n, d] = [d, n - a * d];
  }
  const k = (M - q0) / q1;
  // Compare |p1/q1 − x| vs |(p0+k·p1)/(q0+k·q1) − x| exactly.
  const b1n = p0 + k * p1;
  const b1d = q0 + k * q1;
  const abs = (v: bigint) => (v < 0n ? -v : v);
  const err2 = abs(p1 * den - num * q1) * b1d; // scaled by q1·b1d·den
  const err1 = abs(b1n * den - num * b1d) * q1;
  return err2 <= err1 ? [Number(p1), Number(q1)] : [Number(b1n), Number(b1d)];
}

/** Modified Bessel function of the first kind, order 0 (series). */
function besselI0(x: number): number {
  let sum = 1;
  let term = 1;
  const q = (x * x) / 4;
  for (let k = 1; k < 500; k++) {
    term *= q / (k * k);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  return sum;
}

/** scipy.signal.firwin(numtaps, cutoff, window=("kaiser", beta)) — lowpass, scaled. */
function firwinKaiser(numtaps: number, cutoff: number, beta: number): Float64Array {
  const alpha = 0.5 * (numtaps - 1);
  const h = new Float64Array(numtaps);
  const i0b = besselI0(beta);
  let s = 0;
  for (let i = 0; i < numtaps; i++) {
    const m = i - alpha;
    const x = cutoff * m;
    const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    const r = (2 * i) / (numtaps - 1) - 1;
    const w = besselI0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / i0b;
    h[i] = cutoff * sinc * w;
    s += h[i];
  }
  for (let i = 0; i < numtaps; i++) h[i] /= s;
  return h;
}

/** scipy.signal.resample_poly(x, up, down) with the default Kaiser(5) window, zero padding. */
export function resamplePoly(x: ArrayLike<number>, up0: number, down0: number): Float64Array {
  const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
  const g = gcd(up0, down0);
  const up = up0 / g;
  const down = down0 / g;
  const nIn = x.length;
  if (up === 1 && down === 1) return Float64Array.from(x);
  const nOutTotal = nIn * up;
  const nOut = Math.floor(nOutTotal / down) + (nOutTotal % down ? 1 : 0);
  const maxRate = Math.max(up, down);
  const halfLen = 10 * maxRate;
  const core = firwinKaiser(2 * halfLen + 1, 1 / maxRate, 5.0);
  const nPrePad = down - (halfLen % down);
  const nPreRemove = Math.floor((halfLen + nPrePad) / down);
  const outputLen = (lenH: number) => Math.floor(((nIn - 1) * up + lenH - 1) / down) + 1;
  let nPostPad = 0;
  while (outputLen(core.length + nPrePad + nPostPad) < nOut + nPreRemove) nPostPad++;
  const h = new Float64Array(nPrePad + core.length + nPostPad);
  for (let i = 0; i < core.length; i++) h[nPrePad + i] = core[i] * up;

  // upfirdn: y[k] = Σ_j h[j] · xUp[k·down − j], xUp[m] = x[m/up] when up | m.
  const y = new Float64Array(nOut);
  for (let o = 0; o < nOut; o++) {
    const k = o + nPreRemove;
    const m = k * down; // index into the upsampled signal
    // j ≡ m (mod up) so that (m − j) is a multiple of up.
    let j = m % up;
    let acc = 0;
    for (; j < h.length; j += up) {
      const src = (m - j) / up;
      if (src < 0) break;
      if (src < nIn) acc += h[j] * x[src];
    }
    y[o] = acc;
  }
  return y;
}

function isUniform(ts: ArrayLike<number>, relTol = 0.05): boolean {
  if (ts.length < 3) return true;
  const d = new Float64Array(ts.length - 1);
  for (let i = 1; i < ts.length; i++) d[i - 1] = ts[i] - ts[i - 1];
  const md = median(d);
  if (md <= 0) return false;
  let mean = 0;
  for (const v of d) mean += v;
  mean /= d.length;
  let v2 = 0;
  for (const v of d) v2 += (v - mean) ** 2;
  return Math.sqrt(v2 / d.length) / md <= relTol;
}

/** utils/resampling.py::_resample_sensor_to_hz for one contiguous chunk. */
function resampleChunk(ct: number[], cols: number[][], grid: number[], hz: number): Float64Array[] {
  if (!isUniform(ct)) return cols.map((c) => interp(grid, ct, c));
  const d = new Float64Array(ct.length - 1);
  for (let i = 1; i < ct.length; i++) d[i - 1] = ct[i] - ct[i - 1];
  const srcHz = 1000 / median(d);
  let [up, down] = limitDenominator(hz / srcHz, 1000);
  if (up < 1 || down < 1) {
    const isrc = Math.max(1, Math.round(srcHz));
    const g = ((a: number, b: number): number => { while (b) [a, b] = [b, a % b]; return a; })(hz, isrc);
    [up, down] = [hz / g, isrc / g];
  }
  const srcIdx = ct.map((_, i) => i);
  return cols.map((c) => {
    const y = resamplePoly(c, up, down);
    const tOfY = interp(Array.from({ length: y.length }, (_, i) => i * (down / up)), srcIdx, ct);
    return interp(grid, tOfY, y);
  });
}

// ---------- resampling (utils/resampling.py) ----------

/** Sorted, de-duplicated, gap-aware resample of x/y/z onto a uniform grid. */
export function resampleXyz(s: Xyz, hz = RESAMPLE_HZ): Xyz & { t: Float64Array } {
  const order = Array.from({ length: s.t.length }, (_, i) => i).sort((a, b) => s.t[a] - s.t[b]);
  const ts: number[] = [];
  const cols: number[][] = [[], [], []];
  for (const i of order) {
    if (ts.length && ts[ts.length - 1] === s.t[i]) continue; // drop_duplicates keeps the first
    ts.push(s.t[i]);
    cols[0].push(s.x[i]);
    cols[1].push(s.y[i]);
    cols[2].push(s.z[i]);
  }
  const outT: number[] = [];
  const out: number[][] = [[], [], []];
  const period = 1000 / hz;
  let start = 0;
  for (let k = 1; k <= ts.length; k++) {
    if (k < ts.length && ts[k] - ts[k - 1] <= GAP_MS) continue;
    // Chunk [start, k).
    const ct = ts.slice(start, k);
    const t0 = ct[0];
    const n = Math.floor((ct[ct.length - 1] - t0) / period) + 1;
    if (ct.length < 2 || n < 2) {
      // Python returns such chunks unresampled.
      for (let i = start; i < k; i++) {
        outT.push(ts[i]);
        for (let c = 0; c < 3; c++) out[c].push(cols[c][i]);
      }
    } else {
      const grid = Array.from({ length: n }, (_, i) => Math.round(t0 + i * period));
      const res = resampleChunk(ct, cols.map((c) => c.slice(start, k)), grid, hz);
      for (let c = 0; c < 3; c++) for (const v of res[c]) out[c].push(v);
      for (const t of grid) outT.push(t);
    }
    start = k;
  }
  return { t: Float64Array.from(outT), x: Float64Array.from(out[0]), y: Float64Array.from(out[1]), z: Float64Array.from(out[2]) };
}

// ---------- gravity (utils/accelerometer_utils.py) ----------

function estimateGravityStationary(
  ax: ArrayLike<number>,
  ay: ArrayLike<number>,
  az: ArrayLike<number>,
  n: number,
  fs: number,
  windowSec: number,
): { g: Vec3; mag: number } {
  if (n < 10) return { g: [0, 0, 9.81], mag: 9.81 };
  const win = Math.max(10, Math.trunc(fs * windowSec));
  const nWin = Math.max(1, Math.floor(n / win));
  const means: number[][] = [[], [], []];
  for (let w = 0; w < nWin; w++) {
    const s = w * win;
    const e = Math.min(s + win, n);
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (let i = s; i < e; i++) {
      sx += ax[i];
      sy += ay[i];
      sz += az[i];
    }
    means[0].push(sx / (e - s));
    means[1].push(sy / (e - s));
    means[2].push(sz / (e - s));
  }
  const g: Vec3 = [median(means[0]), median(means[1]), median(means[2])];
  return { g, mag: Math.hypot(...g) };
}

// ---------- quaternions (physics/reconstruct_az/_quaternion.py) ----------

const UP: Vec3 = [0, 0, 1];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

function qNormalize(q: Quat): Quat {
  const n = Math.hypot(...q);
  return n < 1e-12 ? [1, 0, 0, 0] : [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function qMul(a: Quat, b: Quat): Quat {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

function qRotate(q: Quat, v: Vec3): Vec3 {
  const qv: Vec3 = [q[1], q[2], q[3]];
  const c = cross(qv, v);
  const t: Vec3 = [2 * c[0], 2 * c[1], 2 * c[2]];
  const c2 = cross(qv, t);
  return [v[0] + q[0] * t[0] + c2[0], v[1] + q[0] * t[1] + c2[1], v[2] + q[0] * t[2] + c2[2]];
}

function qFromOmega(q: Quat, w: Vec3, dt: number): Quat {
  const d = qMul(q, [0, w[0], w[1], w[2]]);
  return qNormalize([q[0] + 0.5 * d[0] * dt, q[1] + 0.5 * d[1] * dt, q[2] + 0.5 * d[2] * dt, q[3] + 0.5 * d[3] * dt]);
}

function qFromTwoVectors(a0: Vec3, b0: Vec3): Quat {
  const na = Math.hypot(...a0) + 1e-12;
  const nb = Math.hypot(...b0) + 1e-12;
  const a: Vec3 = [a0[0] / na, a0[1] / na, a0[2] / na];
  const b: Vec3 = [b0[0] / nb, b0[1] / nb, b0[2] / nb];
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  if (d > 0.999999) return [1, 0, 0, 0];
  if (d < -0.999999) {
    let axis = cross(a, [1, 0, 0]);
    if (Math.hypot(...axis) < 1e-6) axis = cross(a, [0, 1, 0]);
    const n = Math.hypot(...axis) + 1e-12;
    return [0, axis[0] / n, axis[1] / n, axis[2] / n];
  }
  const axis = cross(a, b);
  return qNormalize([1 + d, axis[0], axis[1], axis[2]]);
}

// ---------- Mahony reconstruction (base.py + mahony.py) ----------

/**
 * World-frame accel with gravity kept on +z ("virtually flat phone"), on the
 * accelerometer's timeline — reconstruct_az("Mahony", acc, gyro).
 */
export function mahonyFlatten(acc: Xyz, gyr: Xyz, kp = 1.0, ki = 0.3, trustTol = 0.15, seedSec = 0.5): Xyz & { t: Float64Array } {
  const n = acc.t.length;
  const ts = Float64Array.from(acc.t);
  if (n < 2 || gyr.t.length < 1) return { t: ts, x: Float64Array.from(acc.x), y: Float64Array.from(acc.y), z: Float64Array.from(acc.z) };

  const gx = interp(ts, gyr.t, gyr.x);
  const gy = interp(ts, gyr.t, gyr.y);
  const gz = interp(ts, gyr.t, gyr.z);

  const dt = new Float64Array(n);
  for (let i = 1; i < n; i++) dt[i] = (ts[i] - ts[i - 1]) / 1000;
  dt[0] = dt[1];
  const pos = Array.from(dt).filter((d) => d > 0);
  const med = pos.length ? median(pos) : 0.02;
  for (let i = 0; i < n; i++) if (dt[i] <= 0) dt[i] = med;

  // Seed orientation from the opening window.
  const fs = 1 / median(dt);
  const nSeed = Math.max(5, Math.min(n, Math.trunc(seedSec * fs)));
  let { g: gvec, mag: gMag } = estimateGravityStationary(acc.x, acc.y, acc.z, nSeed, fs, Math.min(seedSec, 0.25));
  if (!Number.isFinite(gMag) || gMag < 1e-3) {
    gvec = [0, 0, GRAVITY];
    gMag = GRAVITY;
  }
  let q = qNormalize(qFromTwoVectors(gvec, UP));

  const out = { t: ts, x: new Float64Array(n), y: new Float64Array(n), z: new Float64Array(n) };
  let bias: Vec3 = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const a: Vec3 = [acc.x[i], acc.y[i], acc.z[i]];
    const norm = Math.hypot(...a);
    let omega: Vec3 = [gx[i], gy[i], gz[i]];
    if (norm > 1e-6) {
      const err = Math.abs(norm - gMag) / Math.max(gMag, 1e-6);
      const trust = Math.exp(-0.5 * (err / Math.max(trustTol, 1e-6)) ** 2);
      const aHat: Vec3 = [a[0] / norm, a[1] / norm, a[2] / norm];
      const vHat = qRotate([q[0], -q[1], -q[2], -q[3]], UP);
      const c = cross(aHat, vHat);
      const e: Vec3 = [c[0] * trust, c[1] * trust, c[2] * trust];
      bias = [bias[0] - ki * e[0] * dt[i], bias[1] - ki * e[1] * dt[i], bias[2] - ki * e[2] * dt[i]];
      omega = [omega[0] - bias[0] + kp * e[0], omega[1] - bias[1] + kp * e[1], omega[2] - bias[2] + kp * e[2]];
    }
    q = qFromOmega(q, omega, dt[i]);
    // base.reconstruct rotates body → world and removes g on z;
    // reconstruct_az adds it straight back, so z keeps gravity.
    const w = qRotate(qNormalize(q), a);
    out.x[i] = w[0];
    out.y[i] = w[1];
    out.z[i] = w[2];
  }
  return out;
}

// ---------- vertical channels (fit_elevator_parameters/common.py) ----------

function fsFromMs(ts: ArrayLike<number>): number {
  if (ts.length < 2) return 100;
  const d = new Float64Array(ts.length - 1);
  for (let i = 1; i < ts.length; i++) d[i - 1] = ts[i] - ts[i - 1];
  const m = median(d);
  return m <= 0 ? 100 : 1000 / m;
}

function detrend(a: Float64Array, fs: number): Float64Array {
  const w = Math.max(3, Math.round(DETREND_SEC * fs));
  const dc = rollingMeanCentered(a, w);
  return a.map((v, i) => v - dc[i]);
}

/** _vertical_accel: project onto the whole-trace gravity estimate, 8 s DC-detrend. */
export function verticalAccel(s: Xyz, fs: number): Float64Array {
  const n = s.t.length;
  const { g, mag } = estimateGravityStationary(s.x, s.y, s.z, n, fs, 0.5);
  const a = new Float64Array(n);
  if (!Number.isFinite(mag) || mag < 1e-6) {
    const mags = Array.from({ length: n }, (_, i) => Math.hypot(s.x[i], s.y[i], s.z[i]));
    const gm = median(mags);
    mags.forEach((m, i) => (a[i] = m - gm));
  } else {
    const h = [g[0] / mag, g[1] / mag, g[2] / mag];
    for (let i = 0; i < n; i++) a[i] = s.x[i] * h[0] + s.y[i] * h[1] + s.z[i] * h[2] - mag;
  }
  return detrend(a, fs);
}

/** _a_mag_minus_g: |a| − |ĝ|, 8 s DC-detrend. */
export function aMagMinusG(s: Xyz, fs: number): Float64Array {
  const n = s.t.length;
  const { mag } = estimateGravityStationary(s.x, s.y, s.z, n, fs, 0.5);
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.hypot(s.x[i], s.y[i], s.z[i]) - mag;
  return detrend(a, fs);
}

/** reconstructedSignal(acc, gyro, "Mahony"): the a_rec the algorithm operates on. */
export function reconstructVertical(acc: Xyz, gyr: Xyz | null): VerticalSignal {
  const rs = resampleXyz(acc);
  const flat = gyr && gyr.t.length ? mahonyFlatten(rs, gyr) : rs;
  const fs = fsFromMs(flat.t);
  return {
    t: rs.t,
    x: Float64Array.from(rs.x),
    y: Float64Array.from(rs.y),
    z: Float64Array.from(rs.z),
    a_rec: verticalAccel(flat, fs),
    a_mag_g: aMagMinusG(flat, fs),
  };
}
