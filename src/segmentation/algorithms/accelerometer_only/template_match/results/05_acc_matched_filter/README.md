# 05 — Amplitude-preserving (template-energy-normalized) matched filter

## Motivation

Folder `04_pulse_detection/` used standard NCC, which divides by the *local* signal RMS. That strips amplitude and makes any hump-shaped wiggle look like a real ride. Hypothesis: use **template-energy-only** normalization so that the score is proportional to the actual amplitude of the matching ride.

## Method

- Same 5 trapezoid + 5 parabola velocity templates as folder 04 (k-means representatives of visually-labeled rides).
- Session signal: `cumsum(|a| − mean|a|)` → 4th-order Butterworth LPF @ 0.3 Hz (same as `run_results/.../velocity.png`).
- Score formula:
  ```
  score[i] = Σ_j  signal[i+j] · template[j]  /  ||template||²
  ```
  This is the matched filter **without local signal normalization**. At exact shape match with amplitude `k`, score ≈ `k`. Template energies are normalized so scores are comparable across different template lengths.
- Threshold: 90th percentile of the session's smoothed score (detection budget ≈10% of session, matching GT density).

## Files

- `acc_pulse_templates.png` — the 5+5 templates (velocity shape + its derivative).
- `acc_pulse_confidence_{name}.png` — separate trapezoid/parabola score traces with GT yellow overlay.
- `acc_pulse_detections_{name}.png` — combined max score + threshold + detections.

## Results

| Experimenter | det | TP | FP | GT hit |
|---|---|---|---|---|
| uriya | 16 | 2 | 14 | 2/44 |
| roy_turgeman | 16 | 3 | 13 | 3/39 |

**The matched filter still does not localize rides reliably.** The unnormalized score shows peaks above the walking baseline, but those peaks don't land inside GT windows consistently. Scores are dominated by integration artifacts and wide walking-induced humps, which survive both the amplitude normalization fix and the template-energy fix.

## Why it doesn't work

1. **Session velocity has structural drift.** `cumsum(a − mean(a))` over 1000+ s accumulates any bias into slow wander. A 0.3 Hz LPF doesn't remove ride-length drift, which correlates with ride-length templates.
2. **Walking produces humps of comparable velocity amplitude.** When someone walks to the elevator, stops, then walks away, the integrated velocity across that window traces a shape eerily similar to a ride (because integration of a bipolar acc pattern always looks bumpy).
3. **Templates are ≈10 s long.** Over that scale, too many things look like templates.

## Conclusion

Matched filtering on the velocity signal — normalized *or* amplitude-preserving — is not sufficient to detect elevator rides. The richness of information needed ("stillness before/during, moving after; bipolar acc spike at boundaries; real displacement") isn't captured by a single shape-match score.

See root `results/README.md` for next-step recommendations. The most promising path is the **stillness gate** used in the existing `accelerometer_only` pipeline (rolling variance of `|a|` → within-block integration → displacement threshold), with matched filter as a *ranking* layer on top of candidate windows rather than a primary detector.

**Script:** `../../scripts/acc_pulse_detect.py`
