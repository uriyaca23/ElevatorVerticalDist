# Iteration: iter_16_lower_peak_a

**What changed:** Lower `min_peak_abs_a` 0.25 → **0.20** on top of iter_15 baseline (triangle row + `w_min_s=0.3`). Rationale: iter_15 diagnostics showed 49 missed rides had no candidate pair; their best-peak amplitudes reported ~0.14-0.20, right at the gate.

## Metrics (vs iter_07 / iter_15)

| metric | iter_07 | iter_15 | **iter_16** | Δ vs iter_15 | Δ vs iter_07 |
|---|---|---|---|---|---|
| clean | 403 | 400 | **400** | 0 | −3 |
| missed | 88 | 87 | 87 | 0 | −1 |
| gt_merged | 6 | 10 | 10 | 0 | +4 |
| gt_split | 1 | 1 | 1 | 0 | 0 |
| fp | 59 | 49 | **46** | **−3** | **−13** |
| **mistakes_total** | 154 | 147 | **144** | **−3** | **−10** |
| f1_like | 0.835 | 0.839 | **0.841** | +0.002 | +0.006 |
| iou_f1@0.5 | 0.601 | 0.702 | 0.705 | +0.003 | +0.104 |

## Observations

- **Small but real win: 147→144 (−3).** Comes entirely from FP reduction (49→46). Missed count unchanged.
- **The admitted lower-amplitude peaks didn't unlock missed rides** — they paired with noise-adjacent peaks that were previously FPs, replacing them with valid rides. Good: the quiet-middle + joint_r2=0.90 gates caught the walking FPs that might have leaked through.
- **Deep-dive on the 87 remaining missed:**
  - **50/87 (57%) have no candidate pair.** But their best-peak signed R² is EXCELLENT (pos_r2 median 0.95, neg_r2 median 0.94). Just the amplitude is weak: pos_A median 0.14, |neg_A| median 0.16. These are real rides that look like elevator rides but have low peak acceleration — the detector *sees* them, it's the 0.20 gate that rejects.
  - **37/87 have a candidate pair that fails filters.** Only 9 of those fail *solely* on `pair_A<0.30`. The rest also fail joint_r2 or heatmap_energy.
  - Of the 9 pair_A-only rejects: raising the floor to 0.24 would recover 1, 0.22 would recover 1, 0.18 would recover 4. Limited ceiling on this axis alone.
- **Problem experiments** (non-Pixel10_exp2): milleniumHotel_SamsungSM-A235F_exp1 (10 missed), beitYitzchakiRaanana_Xiaomi_exp6 (10), milleniumHotel_Xiaomi_exp1 (8), milleniumHotel_SS911B_exp1 (7). These are damped/noisy-phone rides where the signal is real but the amplitude is genuinely low. Recovering them requires gate lowering.

## Next iteration hypothesis (iter_17)

Lower both gates more aggressively: `min_peak_abs_a` 0.20 → **0.12** and `min_pair_abs_a` 0.30 → **0.20**. This should admit most of the 50 no-candidate rides AND most of the 9 pair_A-only rejects. FP risk is high — quiet_middle, heatmap_energy=0.40 and joint_r2=0.90 must hold. If FPs explode I'll revert and pursue a per-phone noise-scaled floor instead.
