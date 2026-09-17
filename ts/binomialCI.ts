/**
 * Binomial proportion confidence intervals for Bernoulli trial results.
 *
 * A rate measured over a finite number of trials — an inclusion rate, an
 * accept rate, a detection rate — is an *estimate* of a probability, and
 * carries sampling error of its own. Two bins both reading 88% support
 * very different claims when one holds 366 trials and the other holds 17.
 *
 * Two constructions:
 *
 *  - **Wald**: `p̂ ± z·SE`. Symmetric and easy to read, but it can run
 *    past [0, 1] and its true coverage collapses for small `n` or for `p̂`
 *    near the boundaries.
 *  - **Wilson score**: obtained by inverting the score test. Pulled
 *    toward 1/2, always inside [0, 1], and holds its nominal coverage far
 *    better at small `n`. Prefer it for anything reported.
 *
 * Mirrors `pyramidElevatorDist/utils/binomial_ci.py`; keep the two in step.
 */

export type ConfidenceLevel = 0.9 | 0.95;
export type CIMethod = 'wald' | 'wilson' | 'both';
export type IntervalMethod = Exclude<CIMethod, 'both'>;

/** Two-sided standard-normal quantiles z_{1-α/2} for the supported levels. */
export const Z_SCORES: Readonly<Record<ConfidenceLevel, number>> = {
  0.9: 1.644853,
  0.95: 1.959964,
};

export interface BinomialCIInput {
  /** Number of Bernoulli trials, n. A positive integer. */
  totalCount: number;
  /** Observed success proportion p̂, in [0, 1] (0.88 = 88%). */
  successRate: number;
  /** Two-sided confidence level. Defaults to 0.95. */
  confidenceLevel?: ConfidenceLevel;
  /** Which construction(s) to compute. Defaults to 'both'. */
  method?: CIMethod;
}

export interface Interval {
  method: IntervalMethod;
  /** Lower bound, clamped to [0, 1]. */
  lower: number;
  /** Upper bound, clamped to [0, 1]. */
  upper: number;
  /** `[lower, upper]`, clamped to [0, 1]. */
  bounds: [number, number];
  /** Point the interval is built around (p̂ for Wald, shrunk for Wilson). */
  center: number;
  /** Half-width before clamping. */
  margin: number;
  /** Bounds rendered as percentages, e.g. `['84.67%', '91.33%']`. */
  percentBounds: [string, string];
}

export interface BinomialCIResult {
  totalCount: number;
  successRate: number;
  /** n·p̂ rounded to the nearest whole trial, for readability. */
  successCount: number;
  confidenceLevel: ConfidenceLevel;
  method: CIMethod;
  /** The z quantile used. */
  z: number;
  /** `sqrt(p̂(1-p̂)/n)`. */
  standardError: number;
  wald?: Interval;
  wilson?: Interval;
}

const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));

/** Greedy word wrap to `width` columns, each line prefixed by `indent`. */
function wrapText(text: string, width = 72, indent = '  '): string[] {
  const lines: string[] = [];
  let line = '';
  for (const word of text.split(/\s+/).filter(Boolean)) {
    if (line === '') {
      line = word;
    } else if (`${line} ${word}`.length <= width) {
      line += ` ${word}`;
    } else {
      lines.push(indent + line);
      line = word;
    }
  }
  if (line !== '') lines.push(indent + line);
  return lines;
}

/** Format a proportion as a bounded percentage string, e.g. `84.67%`. */
export function formatPercent(p: number, digits = 2): string {
  return `${(clamp01(p) * 100).toFixed(digits)}%`;
}

/** The two-sided z quantile for `confidence`. Throws on unsupported levels. */
export function zScore(confidence: ConfidenceLevel = 0.95): number {
  const z = Z_SCORES[confidence];
  if (z === undefined) {
    const supported = Object.keys(Z_SCORES).join(', ');
    throw new RangeError(
      `unsupported confidence level ${confidence}; expected one of ${supported}`,
    );
  }
  return z;
}

/** Bernoulli standard error `sqrt(p̂(1-p̂)/n)`. */
export function standardError(successRate: number, totalCount: number): number {
  return Math.sqrt((successRate * (1 - successRate)) / totalCount);
}

function validate(input: BinomialCIInput): Required<BinomialCIInput> {
  const { totalCount, successRate } = input;
  const confidenceLevel = input.confidenceLevel ?? 0.95;
  const method = input.method ?? 'both';

  if (!Number.isFinite(totalCount) || !Number.isInteger(totalCount) || totalCount <= 0) {
    throw new RangeError(
      `totalCount must be a positive integer, received ${totalCount}`,
    );
  }
  if (!Number.isFinite(successRate) || successRate < 0 || successRate > 1) {
    throw new RangeError(
      `successRate must be a proportion in [0, 1], received ${successRate}`,
    );
  }
  if (method !== 'wald' && method !== 'wilson' && method !== 'both') {
    throw new RangeError(
      `method must be 'wald', 'wilson' or 'both', received ${String(method)}`,
    );
  }
  zScore(confidenceLevel); // throws on an unsupported level

  return { totalCount, successRate, confidenceLevel, method };
}

/** Standard (Wald) interval: `p̂ ± z·SE`. */
export function waldInterval(
  successRate: number,
  totalCount: number,
  confidenceLevel: ConfidenceLevel = 0.95,
): Interval {
  const z = zScore(confidenceLevel);
  const margin = z * standardError(successRate, totalCount);
  const lower = clamp01(successRate - margin);
  const upper = clamp01(successRate + margin);
  return {
    method: 'wald',
    lower,
    upper,
    bounds: [lower, upper],
    center: successRate,
    margin,
    percentBounds: [formatPercent(lower), formatPercent(upper)],
  };
}

/**
 * Wilson score interval.
 *
 *   center = (p̂ + z²/2n) / (1 + z²/n)
 *   margin = z/(1 + z²/n) · sqrt(p̂(1-p̂)/n + z²/4n²)
 */
export function wilsonInterval(
  successRate: number,
  totalCount: number,
  confidenceLevel: ConfidenceLevel = 0.95,
): Interval {
  const n = totalCount;
  const p = successRate;
  const z = zScore(confidenceLevel);
  const z2 = z * z;

  const denominator = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denominator;
  const margin =
    (z / denominator) * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n));

  const lower = clamp01(center - margin);
  const upper = clamp01(center + margin);
  return {
    method: 'wilson',
    lower,
    upper,
    bounds: [lower, upper],
    center,
    margin,
    percentBounds: [formatPercent(lower), formatPercent(upper)],
  };
}

/**
 * Confidence interval(s) for a Bernoulli success rate.
 *
 * ```ts
 * calculateBinomialCI({ totalCount: 366, successRate: 0.88 });
 * // → 95% Wald [84.67%, 91.33%], Wilson [84.27%, 90.94%]
 * ```
 */
export function calculateBinomialCI(input: BinomialCIInput): BinomialCIResult {
  const { totalCount, successRate, confidenceLevel, method } = validate(input);

  const result: BinomialCIResult = {
    totalCount,
    successRate,
    successCount: Math.round(totalCount * successRate),
    confidenceLevel,
    method,
    z: zScore(confidenceLevel),
    standardError: standardError(successRate, totalCount),
  };
  if (method === 'wald' || method === 'both') {
    result.wald = waldInterval(successRate, totalCount, confidenceLevel);
  }
  if (method === 'wilson' || method === 'both') {
    result.wilson = wilsonInterval(successRate, totalCount, confidenceLevel);
  }
  return result;
}

/** Human-readable breakdown of a result: the z, the widening, the bounds. */
export function formatExplanation(result: BinomialCIResult): string {
  const {
    totalCount: n,
    successRate: p,
    successCount,
    confidenceLevel,
    z,
    standardError: se,
    wald,
    wilson,
  } = result;

  const pct = (x: number) => formatPercent(x);
  const level = `${(confidenceLevel * 100).toFixed(0)}%`;
  const other: ConfidenceLevel = confidenceLevel === 0.95 ? 0.9 : 0.95;
  const zOther = zScore(other);
  const ratio = zOther / z;
  const direction = zOther > z ? 'widen' : 'narrow';
  const otherLevel = `${(other * 100).toFixed(0)}%`;

  const rationale =
    `z is the two-sided standard-normal quantile: ${level} of the sampling ` +
    `distribution of p̂ falls within ±${z} standard errors of the true ` +
    `probability. The margin is proportional to z, so moving to ${otherLevel} ` +
    `(z = ${zOther}) would ${direction} it by a factor of ${ratio.toFixed(3)} — ` +
    `more confidence is bought with a wider interval, never with a sharper ` +
    `estimate. Only more trials narrow it: SE falls as 1/sqrt(n).`;

  const lines: string[] = [
    `Binomial proportion CI — n = ${n} trials, ` +
      `p̂ = ${pct(p)} (≈${successCount} successes)`,
    '',
    `Confidence level ${level} → z = ${z}`,
    ...wrapText(rationale),
    '',
    `Standard error  SE = sqrt(p̂(1-p̂)/n) = ${se.toFixed(6)}`,
  ];

  if (wald) {
    lines.push(
      '',
      `Wald    p̂ ± z·SE = ${pct(p)} ± ${pct(wald.margin)}`,
      `        → [${wald.percentBounds[0]}, ${wald.percentBounds[1]}]`,
    );
  }
  if (wilson) {
    lines.push(
      '',
      `Wilson  centre ${pct(wilson.center)} (pulled toward 50% by z²/2n), ` +
        `margin ${pct(wilson.margin)}`,
      `        → [${wilson.percentBounds[0]}, ${wilson.percentBounds[1]}]`,
    );
  }
  if (wald && wilson) {
    lines.push(
      '',
      'Quote Wilson: it stays inside [0, 1] and holds its nominal coverage',
      'at small n, where Wald is optimistic.',
    );
  }
  return lines.join('\n');
}
