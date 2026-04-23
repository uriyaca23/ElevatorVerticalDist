# Iteration: iter_15_w_min_only — **new best, beats iter_07**

**What changed:** Keep iter_13's triangle row. Keep iter_14's `w_min_s=0.3` (narrower W templates). **Revert** `min_pair_abs_a` → 0.30 to stop iter_14's gt_split blowup.

## Metrics (vs iter_07 / iter_13 / iter_14)

| metric | iter_07 | iter_13 | iter_14 | **iter_15** | Δ vs iter_07 |
|---|---|---|---|---|---|
| clean | 403 | 403 | 394 | **400** | −3 |
| missed | 88 | 86 | 84 | 87 | −1 |
| gt_merged | 6 | 8 | 10 | 10 | +4 |
| gt_split | 1 | 1 | 10 | 1 | 0 |
| fp | **59** | 60 | 51 | **49** | **−10** |
| **mistakes_total** | **154** | 155 | 155 | **147** | **−7** |
| f1_like | 0.835 | 0.834 | 0.817 | **0.839** | **+0.004** |
| iou_f1@0.5 | 0.601 | 0.703 | 0.691 | 0.702 | +0.101 |

## Observations

- **New best. Beats deployed iter_07 baseline by 7 mistakes (−4.5%) and +0.004 on f1_like.** Combination: triangle row (iter_13) + narrower W grid (0.3–3.0s, 30 cells) + original amplitude gates.
- **FPs drop 10 vs iter_07** (59→49). Narrower W templates appear to match walking/transient artifacts less readily than wider ones. Unexpected bonus.
- **Missed stays at 87 (essentially the iter_13 number).** The W floor wasn't the dominant miss driver — most missed rides lack candidate peaks upstream.
- **Reject flag breakdown (38 missed with candidates):**
  - `pair |A| < 0.30` continues to reject ~8 (pair_A values 0.10–0.20 m/s²).
  - `joint R² < 0.90` rejects ~8.
  - `pair_W` now at 0.30 floor for 18/38 — grid wants even narrower templates for some rides.
- **49/87 missed rides have no candidate pair at all** — peak-pick gates (`r2_peak_thresh=0.40`, `min_peak_abs_a=0.25`) are still the upstream bottleneck.
- **Known data-ceiling:** Pixel10_exp2 contributes 31 of 87 missed rides (damped-chip memory). Removing this fixed cost: 87−31 = 56 genuinely recoverable misses.
- **`noise_sigma_multiplier` investigation:** confirmed it never triggers for the phones in our dataset (all chip σ values at 100 Hz put the σ-based floor below 0.10 m/s², well under the 0.25/0.30 default). Not a worthwhile knob to tune for this dataset.

## Next iteration hypothesis (iter_16)

Attack the 49 missed rides without candidate pairs by lowering `min_peak_abs_a` 0.25 → 0.20. iter_08 tried this plus `r2_peak_thresh` 0.40→0.35 and regressed by +1 — this time we change only the amplitude gate, and the quiet-middle + `pair_A=0.30` filters should hold against any new FPs. Risk to watch: FP climb on noisy Samsung/Xiaomi milleniumHotel experiments.
