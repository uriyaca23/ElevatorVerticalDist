# Iteration: iter_13_triangle_shape_row

**What changed:** Prepend `f=0` (pure triangle) row to `DetectConfig.grid_f()` so `joint_pair_score` argmax can pick trapezoid-vs-triangle per pair based on shared-shape joint R². Also restored the offline-fitter grid constants in `fit_elevator_parameters/common.py` which had been corrupted to a degenerate `linspace(0.01, 0.0, 15)`.

## Metrics (vs iter_07 baseline)

| metric | iter_07 | iter_13 | Δ |
|---|---|---|---|
| n_gt | 498 | 498 | — |
| clean | 403 | 403 | 0 |
| missed | 88 | 86 | **−2** |
| gt_merged | 6 | 8 | +2 |
| gt_split | 1 | 1 | 0 |
| fp | 59 | 60 | +1 |
| **mistakes_total** | **154** | **155** | +1 |
| f1_like | 0.835 | 0.834 | −0.001 |
| **iou_f1@0.5** | **0.601** | **0.703** | **+0.102** |

## Observations

- **Composite mistake count is essentially flat (+1).** Triangle row recovered 2 missed rides but introduced 2 merges + 1 FP.
- **IoU improved substantially (+0.102 on iou_f1@0.5, +0.10 mean IoU).** Pairs that were already clean are now fitted more tightly around the ride — triangle lobes pull endpoints in where cruise-phase plateau doesn't exist.
- **Triangle usage stats:** 33/335 clean pairs (10%) prefer `f=0`; 9/33 missed-but-candidate pairs (27%) prefer `f=0`. Triangle disproportionately picked on weak/marginal rides — not a pure win.
- **Missed-ride diagnostics (critical):**
  - **53/86 (62%) of missed rides have NO candidate pair** (failed upstream peak-picking before pair filter ran). Triangle template doesn't help these — the gate `r2_peak_thresh=0.40` or `min_peak_abs_a=0.25` killed their candidates.
  - Of the 33 with candidate pairs: `pair_A_abs` median = 0.27 (below `min_pair_abs_a=0.30` floor), and `pair_W` median = 0.40 (pinned at `w_min_s` floor). Grid wants narrower templates than allowed.
  - 30 of 86 missed rides come from `UriyaCohenEliya_milleniumHotel_GooglePixel10_exp2` alone — known damped-accelerometer data-quality ceiling (see memory).
- **Misses are NOT short rides.** Duration histogram: 20 misses in 8-12s, 54 in 12-20s, 12 in 20-60s. Zero under 8s. The triangle premise ("one-floor ride as triangle") is real physically but isn't the dominant failure mode here — misses are mostly normal-duration rides on noisy/damped phones.

## Next iteration hypothesis (iter_14)

Lower the W-grid floor from 0.40s to 0.30s so narrow lobes can match, and lower `min_pair_abs_a` from 0.30 to 0.22 to admit lower-amplitude lobes whose grid-fit is otherwise good. These directly attack the two dominant reject flags observed above. Quiet-middle filter (iter_04) plus joint_r2≥0.90 gate should contain FP blowback from the amplitude relaxation. Risk axis: FPs. If they explode, revert `min_pair_abs_a` only and keep the wider W grid.
