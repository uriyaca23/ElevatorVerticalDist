/**
 * Unit tests — run with `node --test` from this directory (Node >= 22.6,
 * which strips the type annotations natively; no build step).
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  calculateBinomialCI,
  formatExplanation,
  formatPercent,
  standardError,
  waldInterval,
  wilsonInterval,
  zScore,
  type BinomialCIResult,
} from './binomialCI.ts';

/** Assert a proportion equals `expectedPercent`, to within `tol` percent. */
function assertPercent(actual: number, expectedPercent: number, tol = 0.01): void {
  const got = actual * 100;
  assert.ok(
    Math.abs(got - expectedPercent) <= tol,
    `expected ${expectedPercent}% ± ${tol}, got ${got.toFixed(4)}%`,
  );
}

// The reference case: n = 202 + 88 + 59 + 17 = 366 trials at p̂ = 0.88.
const N = 202 + 88 + 59 + 17;
const P = 0.88;

test('the reference n is 366', () => {
  assert.equal(N, 366);
});

test('90% Wald ≈ [85.21%, 90.79%]', () => {
  const { bounds } = waldInterval(P, N, 0.9);
  assertPercent(bounds[0], 85.21);
  assertPercent(bounds[1], 90.79);
});

test('95% Wald ≈ [84.67%, 91.33%]', () => {
  const { bounds } = waldInterval(P, N, 0.95);
  assertPercent(bounds[0], 84.67);
  assertPercent(bounds[1], 91.33);
});

test('90% Wilson ≈ [84.92%, 90.52%]', () => {
  const { bounds } = wilsonInterval(P, N, 0.9);
  assertPercent(bounds[0], 84.92);
  assertPercent(bounds[1], 90.52);
});

test('95% Wilson ≈ [84.27%, 90.94%]', () => {
  const { bounds } = wilsonInterval(P, N, 0.95);
  assertPercent(bounds[0], 84.27);
  assertPercent(bounds[1], 90.94);
});

test('percentBounds render the same numbers as strings', () => {
  assert.deepEqual(wilsonInterval(P, N, 0.95).percentBounds, ['84.27%', '90.94%']);
  assert.deepEqual(waldInterval(P, N, 0.9).percentBounds, ['85.21%', '90.79%']);
});

test('defaults are 95% and both methods', () => {
  const r = calculateBinomialCI({ totalCount: N, successRate: P });
  assert.equal(r.confidenceLevel, 0.95);
  assert.equal(r.method, 'both');
  assert.equal(r.z, 1.959964);
  assert.ok(r.wald && r.wilson);
  assertPercent(r.wald.lower, 84.67);
  assertPercent(r.wilson.upper, 90.94);
});

test('method selects which intervals are returned', () => {
  const wald = calculateBinomialCI({ totalCount: N, successRate: P, method: 'wald' });
  assert.ok(wald.wald);
  assert.equal(wald.wilson, undefined);

  const wilson = calculateBinomialCI({ totalCount: N, successRate: P, method: 'wilson' });
  assert.ok(wilson.wilson);
  assert.equal(wilson.wald, undefined);
});

test('standard error matches sqrt(p(1-p)/n)', () => {
  assert.ok(Math.abs(standardError(P, N) - 0.0169860) < 1e-6);
  // SE shrinks as 1/sqrt(n): quadrupling n halves it.
  assert.ok(Math.abs(standardError(P, 4 * N) - standardError(P, N) / 2) < 1e-12);
});

test('z-score lookup', () => {
  assert.equal(zScore(0.9), 1.644853);
  assert.equal(zScore(0.95), 1.959964);
  assert.equal(zScore(), 1.959964);
});

test('95% is strictly wider than 90%, for both methods', () => {
  for (const fn of [waldInterval, wilsonInterval]) {
    const narrow = fn(P, N, 0.9);
    const wide = fn(P, N, 0.95);
    assert.ok(wide.lower < narrow.lower, `${fn.name}: lower bound must drop`);
    assert.ok(wide.upper > narrow.upper, `${fn.name}: upper bound must rise`);
  }
});

test('more trials narrow the interval', () => {
  const few = wilsonInterval(P, 40, 0.95);
  const many = wilsonInterval(P, 4000, 0.95);
  assert.ok(many.upper - many.lower < few.upper - few.lower);
});

test('Wald is symmetric about p̂; Wilson is pulled toward 50%', () => {
  const wald = waldInterval(P, N, 0.95);
  assert.ok(Math.abs((wald.lower + wald.upper) / 2 - P) < 1e-12);

  const wilson = wilsonInterval(P, N, 0.95);
  assert.ok(wilson.center < P, 'centre shifts down toward 0.5 for p̂ > 0.5');
  assert.ok(wilsonInterval(0.12, N, 0.95).center > 0.12, 'and up for p̂ < 0.5');
});

test('bounds stay inside [0, 1] at the boundaries', () => {
  for (const p of [0, 1]) {
    for (const m of [waldInterval, wilsonInterval]) {
      const { lower, upper } = m(p, 25, 0.95);
      assert.ok(lower >= 0 && lower <= 1, `${m.name}(${p}) lower out of range`);
      assert.ok(upper >= 0 && upper <= 1, `${m.name}(${p}) upper out of range`);
    }
  }
  // Wald degenerates at p̂ = 1 (SE = 0); Wilson still reports uncertainty.
  assert.deepEqual(waldInterval(1, 25, 0.95).bounds, [1, 1]);
  assert.ok(wilsonInterval(1, 25, 0.95).lower < 1);
});

test('invalid input is rejected', () => {
  assert.throws(() => calculateBinomialCI({ totalCount: 0, successRate: 0.5 }), RangeError);
  assert.throws(() => calculateBinomialCI({ totalCount: -5, successRate: 0.5 }), RangeError);
  assert.throws(() => calculateBinomialCI({ totalCount: 12.5, successRate: 0.5 }), RangeError);
  assert.throws(() => calculateBinomialCI({ totalCount: 10, successRate: 1.4 }), RangeError);
  assert.throws(() => calculateBinomialCI({ totalCount: 10, successRate: -0.1 }), RangeError);
  assert.throws(
    () => calculateBinomialCI({
      totalCount: 10, successRate: 0.5,
      confidenceLevel: 0.99 as unknown as 0.95,
    }),
    RangeError,
  );
  assert.throws(
    () => calculateBinomialCI({
      totalCount: 10, successRate: 0.5,
      method: 'jeffreys' as unknown as 'wald',
    }),
    RangeError,
  );
});

test('formatPercent clamps and rounds', () => {
  assert.equal(formatPercent(0.8467), '84.67%');
  assert.equal(formatPercent(1.4), '100.00%');
  assert.equal(formatPercent(-0.2), '0.00%');
  assert.equal(formatPercent(0.8467, 1), '84.7%');
});

test('formatExplanation reports z, the widening and the exact bounds', () => {
  const result: BinomialCIResult = calculateBinomialCI({
    totalCount: N, successRate: P, confidenceLevel: 0.95,
  });
  const text = formatExplanation(result);
  const flat = text.replace(/\s+/g, ' ');   // prose may wrap anywhere

  assert.match(flat, /n = 366 trials/);
  assert.match(flat, /p̂ = 88\.00%/);
  assert.match(flat, /z = 1\.959964/);
  assert.match(flat, /standard-normal quantile/);
  assert.match(flat, /wider/);            // why more confidence costs width
  assert.match(flat, /1\.644853/);        // the 90% z it is compared against
  assert.match(flat, /\[84\.67%, 91\.33%\]/);  // Wald bounds
  assert.match(flat, /\[84\.27%, 90\.94%\]/);  // Wilson bounds
});

test('formatExplanation covers a single-method result', () => {
  const text = formatExplanation(
    calculateBinomialCI({ totalCount: N, successRate: P, confidenceLevel: 0.9, method: 'wilson' }),
  );
  assert.match(text, /z = 1\.644853/);
  assert.match(text, /narrow|wider/);
  assert.match(text, /\[84\.92%, 90\.52%\]/);
  assert.doesNotMatch(text, /Wald/);
});
