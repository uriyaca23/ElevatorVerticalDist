# Iteration: iter_03_middle_heatmap

**What changed:** `heatmap_energy_thresh` 0.20 → 0.40. Keeps `min_pair_abs_a
=0.40` and `same_sign_min_gap_s=5.0`.

## Metrics

| metric | iter_00 | iter_01 | iter_02 | iter_03 | Δ(02→03) |
|---|---|---|---|---|---|
| clean | 264 | 352 | 344 | **336** | −8 |
| missed | 232 | 136 | 146 | 154 | +8 |
| fp | 26 | 146 | 145 | **99** | **−46** |
| gt_merged | 2 | 10 | 8 | 8 | 0 |
| mistakes_total | 260 | 292 | 299 | **261** | **−38** |
| f1_like | 0.669 | 0.703 | 0.694 | **0.717** | +0.023 |
| iou_f1@0.5 | 0.406 | 0.454 | 0.442 | **0.470** | +0.028 |

vs baseline: clean +72, missed −78, fp +73, mistakes ≈0.

## Analysis

The heatmap_energy threshold was indeed the FP driver. 0.20 → 0.40 cuts FPs
by 46 at the cost of only 8 clean. iou_f1 and f1_like both beat the
baseline now. Still need to reduce mistakes further.

## Failure-mode breakdown of the 154 missed

| sub-category | count |
|---|---|
| `pair_reject_flags=""` (pair passes OR only one sign in window) | 91 |
|   ↳ with valid pair data (pair would be accepted) | 14 |
|   ↳ with only one sign detected (77) | 77 |
| `low_pair_A` alone | 26 |
| `low_joint_r2 + low_heatmap_energy` | 21 |
| `low_joint_r2 + low_heatmap_energy + low_pair_A` | 13 |
| other | 3 |

- **77 cases**: only one sign has any peak inside the GT window. Detector
  NMS or peak-pick killed the matching-sign peak entirely.
- **14 cases**: both peaks detected, pair clears all filters, but no ride
  emitted — stolen by greedy resolver (a super-pair or same-sign NMS
  took one of the lobes).
- **26 cases**: pair A barely below 0.40 (0.40 floor).
- **34 cases**: joint_r2 < 0.90 AND / OR heatmap_energy < 0.40.

## Worst experiments (still many missed)

| exp | missed |
|---|---|
| UriyaCohenEliya_milleniumHotel_GooglePixel10_exp2 | 31 / 31 (0% clean) |
| RoyTurgeman_Haari3 | 19 / 30 |
| UriyaCohenEliya_BarIlan2Herzelia_Pixel10 | 15 / 20 |
| All 4 milleniumHotel `_exp1` variants | 10 / 10 each |

`milleniumHotel_GooglePixel10_exp2` is eerie — 31 rides, every single one
missed. Likely a recording issue specific to that file (time
misalignment? pocket vs hand shift?). Worth checking separately.

## Next iteration hypothesis

Two paths forward:

1. **Raise `heatmap_energy_thresh` further to 0.50** — try to cut another
   ~30 FPs. Expected cost: ~10 more clean lost.
2. **Reduce `nms_radius_s` from 2.0s to 1.0s** — may admit more peaks in
   dense ride sequences, addressing the 77 "one-sign-only" GTs.

iter_04 will attempt #1 (safer, smaller change).
