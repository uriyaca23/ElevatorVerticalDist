# Iteration: iter_07_lower_peak_gates

**What changed:** Lower peak-pick gates in `DetectConfig`:
- `r2_peak_thresh` 0.55 → 0.40
- `min_peak_abs_a` 0.40 → 0.25

Strategy: admit more borderline lobes into the candidate set; rely on the
four strict pair-filter gates (`joint_r2 ≥ 0.90`, `heatmap_energy ≥ 0.40`,
`min_pair_abs_a ≥ 0.30`, `quiet_middle_ratio ≤ 0.5`) to reject noise.

## Metrics

| metric | iter_06 | iter_07 | Δ | vs iter_00 |
|---|---|---|---|---|
| clean | 334 | **403** | **+69** | **+139** |
| missed | 148 | 88 | −60 | −144 |
| fp | 54 | 59 | +5 | +33 |
| gt_merged | 16 | 6 | −10 | +4 |
| **mistakes_total** | 218 | **154** | **−64** | **−106 (−41 %)** |
| f1_like | 0.747 | **0.835** | +0.088 | +0.166 |
| iou_f1@0.5 | 0.508 | **0.601** | +0.093 | +0.195 |

**Progress toward target (≤ 78 mistakes): 106 / 182 reduction achieved (58 %).**

## Analysis

Lowering the pre-filter gates was massively effective:
- Recovered 60 missed GTs.
- RoyTurgeman_Haari3 improved from 19 missed → 1 missed! The noisy
  Samsung ZFlip6 lobes sit at R²≈0.45, previously killed by the 0.55
  gate.
- Merged GTs dropped from 16 to 6 — more real rides means greedy resolver
  has fewer super-pair shortcuts.
- FPs only grew by 5 (+10 %). The pair filter's four strict gates held.

## Remaining failures

- 88 missed (55 no_flags, 33 with reject flags)
- 6 merged, 1 split, 59 fp

Worst experiments:
| exp | missed |
|---|---|
| UriyaCohenEliya_milleniumHotel_GooglePixel10_exp2 | 30 / 31 |
| UriyaCohenEliya_BarIlan2Herzelia_Pixel10 | 14 / 20 |
| eyalyakir_beitYitzchakiRaanana_Xiaomi_exp6 | 10 |
| eyalyakir_milleniumHotel_SamsungSM-A235F_exp1 | 10 |
| UriyaCohenEliya_milleniumHotel_Xiaomi_exp1 | 8 |

GooglePixel10_exp2 is systematically broken: 50 Hz OK, 31 rides spread
across 15 min, durations 11.5–32.9 s. 2 rides exceed `max_ride_s=30`.
Worth investigating separately.

## Distribution check

24 / 498 GTs have duration > 30 s (4.8 %). All in beitYitzchaki exp6 and
various milleniumHotel exp1/exp2. They hit `max_ride_s=30` — the pair
filter rejects with gap out of bounds. Raising to 40 s unblocks them.

## Next iteration hypothesis

**iter_08 plan:**
- Raise `max_ride_s` 30.0 → 40.0 — unblocks 24 long GTs.
- Push peak gates one more notch: `r2_peak_thresh` 0.40 → 0.35,
  `min_peak_abs_a` 0.25 → 0.20.

Expected: another 20–30 missed recovered, FPs may climb by ~15. Net
should still be a reduction.
