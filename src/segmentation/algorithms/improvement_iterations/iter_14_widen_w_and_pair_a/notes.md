# Iteration: iter_14_widen_w_and_pair_a

**What changed:** Two concurrent adjustments to `DetectConfig`:
- `w_min_s` 0.4 → **0.3** (narrower-lobe templates, matching prediction-side `pulse_pair.py`)
- `min_pair_abs_a` 0.30 → **0.22** (admit lower-amplitude valid rides)

Hypothesis: iter_13 diagnostics showed 33/86 missed rides had candidate pairs pinned at `w_min=0.4` or rejected by `pair_A<0.30`. Relaxing both floors should recover missed rides while the joint-R²=0.90 gate + quiet-middle=0.5 filter contain FPs.

## Metrics (vs iter_13 / iter_07)

| metric | iter_07 | iter_13 | iter_14 | Δ vs iter_13 |
|---|---|---|---|---|
| clean | 403 | 403 | **394** | **−9** |
| missed | 88 | 86 | 84 | −2 |
| gt_merged | 6 | 8 | 10 | +2 |
| gt_split | 1 | 1 | **10** | **+9** |
| fp | 59 | 60 | 51 | −9 |
| **mistakes_total** | **154** | 155 | 155 | 0 |
| f1_like | 0.835 | 0.834 | 0.817 | −0.017 |
| iou_f1@0.5 | 0.601 | 0.703 | 0.691 | −0.012 |

## Observations

- **Composite mistakes flat at 155, but distribution is worse:** FPs dropped (good, −9) and missed dropped (good, −2), but 9 rides flipped from `clean` → `gt_split`. Net clean loss = 9.
- **Over-segmentation pattern is exp-localized.** `beitMansour1` across 4 phones all gained 2 `gt_split` apiece — the narrower W templates detect sub-pulses within legitimate multi-floor rides (e.g. a floor-to-floor sub-cruise gets its own pair).
- **BarIlan2Herzelia_Pixel10 gained 1 clean** (12→9 missed) — the widened grid + lowered amplitude helped this problem experiment slightly.
- **FP drop is real (60→51)** — suggests that some of the sub-pulse pairs "absorbed" the peaks that previously formed FP pairs under the greedy conflict resolver, moving them from `fp` → `gt_split`. That's not a real FP improvement, it's a re-labeling.
- **Conclusion:** `min_pair_abs_a=0.22` is too permissive — low-amplitude sub-pulses within long rides pass the gate and split real rides. `w_min_s=0.3` effect is confounded and will be isolated in iter_15.

## Next iteration hypothesis (iter_15)

Keep `w_min_s=0.3`. Revert `min_pair_abs_a` → 0.30. If gt_split returns to ~1 and missed stays close to iter_13 levels, the W widening alone was safe; otherwise iter_16 reverts both and tackles a different axis (per-phone amplitude calibration, same-sign gap).
