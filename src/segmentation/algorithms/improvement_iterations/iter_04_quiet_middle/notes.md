# Iteration: iter_04_quiet_middle

**What changed:** NEW algorithmic filter. In `pair_filter.py::_try_pair`,
compute `mid_rms = RMS(a_smooth[t1+W : t2-W])` — the inter-lobe plateau.
Reject the pair if `mid_rms > quiet_middle_ratio * pair_A_abs`. Real rides
cruise at constant velocity → mid_rms ≈ 0; walking FPs have continuous
motion → mid_rms ≈ A_abs.

Added `DetectConfig.quiet_middle_ratio = 0.5` (and mirrored in
`config.json`). Values ≥ 1.0 disable the filter; 0.5 is aggressive but
gives real-rides huge headroom.

Simultaneously reverted `min_pair_abs_a` 0.40 → 0.30 — the new quiet-middle
filter catches the low-amplitude walking FPs that this threshold was
indirectly guarding against.

## Metrics

| metric | iter_00 | iter_03 | iter_04 | Δ(03→04) |
|---|---|---|---|---|
| clean | 264 | 336 | 334 | −2 |
| missed | 232 | 154 | 148 | −6 |
| fp | 26 | 99 | **50** | **−49** |
| gt_merged | 2 | 8 | 15 | +7 |
| gt_split | 0 | 0 | 1 | +1 |
| **mistakes_total** | 260 | 261 | **214** | **−47** |
| f1_like | 0.669 | 0.717 | **0.750** | +0.033 |
| iou_f1@0.5 | 0.406 | 0.470 | **0.510** | +0.040 |

**Progress toward target (≤ 78 mistakes): 46 / 182 reduction achieved (25 %).**

## Analysis

The quiet-middle filter is extremely effective: killed 49 walking FPs
while costing only 2 clean. The revert of `min_pair_abs_a` added those
borderline rides back at essentially no FP cost.

The +7 merged GTs are a side-effect of the lower amplitude floor: more
candidate pairs survive the pair filter, some greedy-accepted as
super-pairs.

## Failure-mode breakdown of the 148 missed

| sub-category | count |
|---|---|
| `no_flags` (pair passes OR only one sign in window) | **103** |
| `low_r2 + low_heat` | 22 |
| `low_A` alone | 9 |
| `low_r2 + low_heat + low_A` | 9 |
| `low_r2` alone | 4 |
| `low_heat` alone | 1 |

**103 / 148 = 70 % of remaining missed are upstream detector loss** — the
detector didn't pick a matching ± peak inside the GT window. Causes:

1. Local NMS (`nms_radius_s=2.0 s`) killed a weaker-sign peak because a
   nearby higher-R² peak of the other sign dominated.
2. `r2_peak_thresh=0.55` or `min_peak_abs_a=0.4` too strict for short /
   low-amplitude rides.
3. `same_sign_min_gap_s=5.0` dropped a same-sign peak in dense sequences.

## Worst experiments (≥ 8 missed)

| exp | missed | note |
|---|---|---|
| UriyaCohenEliya_milleniumHotel_GooglePixel10_exp2 | 31 / 31 | still 0 % clean — investigate |
| RoyTurgeman_Haari3 | 16 / 30 | Samsung ZFlip6 new phone, noisy |
| UriyaCohenEliya_BarIlan2Herzelia_Pixel10 | 15 / 20 | new building |
| 4× milleniumHotel `_exp1` | 10 / 10 each | short/low-amp rides |
| 4× acroBuilding | 7–8 / 12 each | |

## Next iteration hypothesis

Address upstream peak-picking to unlock the 103 "one-sign-only" missed
GTs. Cheapest, least risky change: `nms_radius_s` 2.0 → 1.0 s. Admits
more peaks per ± sign. The quiet-middle filter should mop up any spurious
pairs that result.

**iter_05 plan:** `nms_radius_s` 2.0 → 1.0.
