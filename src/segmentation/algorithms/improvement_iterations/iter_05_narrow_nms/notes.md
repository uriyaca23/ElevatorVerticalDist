# Iteration: iter_05_narrow_nms

**What changed:** `nms_radius_s` 2.0 → 1.0. Hypothesis: narrower local
NMS admits more peaks in the same time window, recovering the 103
"one-sign-only" missed GTs from iter_04.

## Metrics

| metric | iter_04 | iter_05 | Δ |
|---|---|---|---|
| clean | 334 | 334 | 0 |
| missed | 148 | 148 | 0 |
| fp | 50 | 54 | +4 |
| gt_merged | 15 | 16 | +1 |
| mistakes_total | 214 | 218 | +4 |
| f1_like | 0.750 | 0.747 | −0.003 |
| iou_f1@0.5 | 0.510 | 0.508 | −0.002 |

## Analysis — hypothesis rejected

Narrowing the NMS radius did nothing for the "one-sign-only" missed GTs.
The reason: `_peak_pick` operates on **unsigned** R². When a +sample and
a −sample are both local maxima within a window, the higher R² sample
wins regardless of sign. Narrower NMS only helps when a same-sign sample
was suppressed by a stronger same-sign sample — which isn't the failure
pattern.

The actual failure mode is: at time t there's a +peak with R²=0.95 and
at time t+δ (δ < NMS radius) there's a −peak with R²=0.85. Unsigned
peak-pick keeps only the +peak. The −peak disappears from `final_peaks`
and the pair filter has nothing to pair with.

## Next iteration hypothesis

Apply `_peak_pick` **per sign** — run it twice, once on
`best_r2` restricted to samples with `A > 0` and once restricted to
`A < 0`. Merge the two peak sets. This guarantees every ±peak that
passes the amplitude + R² gates gets considered.

**iter_06 plan:** change `detect.detect` to call `_peak_pick` twice per
sign. Keep `nms_radius_s = 1.0` (narrower is still reasonable).
