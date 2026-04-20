# Iteration: 07_min_r2

**What changed:** Min R² acceptance: min(r2_1, r2_2) ≥ joint_r2_thresh instead of mean ≥ thresh. Forces both lobes to agree with the shared shape.

**Variant kwargs:** `{'require_min_r2': True}`

## Metrics

| metric | value |
|---|---|
| clean | 28 / 415 |
| missed | 123 |
| gt_merged | 212 |
| gt_split | 52 |
| pred_merged | 123 |
| fp | 36 |
| **f1_like** | **0.092** |
| **IoU-F1 @ 0.5** | **0.082** |
| recall | 0.067 |
| precision | 0.146 |
| mean IoU (matched) | 0.608 |

## Per-exp breakdown

| exp | gt | pred | clean | miss | merged | split | fp |
|---|---|---|---|---|---|---|---|
| RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026 | 39 | 17 | 5 | 12 | 7 | 0 | 5 |
| UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026 | 44 | 6 | 3 | 37 | 1 | 0 | 2 |
| UriyaCohenEliya_acroBuilding_GooglePixel10_15-04-2026_exp4 | 12 | 8 | 3 | 0 | 5 | 3 | 0 |
| UriyaCohenEliya_acroBuilding_SamsungSM-A235F_15-04-2026_exp4 | 12 | 7 | 2 | 0 | 5 | 2 | 0 |
| UriyaCohenEliya_beitMansour1_GooglePixel10_15-04-2026_exp5 | 18 | 11 | 3 | 1 | 7 | 3 | 0 |
| UriyaCohenEliya_beitMansour1_SamsungSM-A235F_15-04-2026_exp5 | 18 | 11 | 2 | 1 | 8 | 5 | 1 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1 | 10 | 5 | 0 | 10 | 0 | 0 | 5 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2 | 31 | 6 | 1 | 26 | 1 | 0 | 4 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp3 | 12 | 6 | 0 | 2 | 5 | 2 | 1 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp2 | 31 | 16 | 2 | 0 | 13 | 8 | 0 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp3 | 12 | 4 | 0 | 1 | 3 | 2 | 0 |
| UriyaCohenEliya_milleniumHotel_Xiaomi22101320I_15-04-2026_exp1 | 10 | 3 | 0 | 7 | 1 | 0 | 2 |
| eyalyakir_acroBuilding_SamsungSM-S911B_15-04-2026_exp4 | 12 | 8 | 1 | 0 | 6 | 3 | 1 |
| eyalyakir_acroBuilding_Xiaomi22101320I_15-04-2026_exp4 | 12 | 6 | 2 | 0 | 4 | 1 | 0 |
| eyalyakir_beitMansour1_SamsungSM-S911B_15-04-2026_exp5 | 18 | 10 | 2 | 0 | 8 | 1 | 0 |
| eyalyakir_beitMansour1_Xiaomi22101320I_15-04-2026_exp5 | 18 | 10 | 0 | 2 | 8 | 2 | 1 |
| eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp1 | 10 | 9 | 0 | 10 | 0 | 0 | 9 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2 | 31 | 15 | 1 | 1 | 13 | 5 | 1 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp3 | 12 | 7 | 0 | 1 | 7 | 4 | 0 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp2 | 31 | 19 | 1 | 1 | 14 | 7 | 3 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp3 | 12 | 8 | 0 | 1 | 7 | 4 | 1 |

## Diagnostic plots

- `errors_bar.png` — per-exp stacked breakdown of clean / missed / merged / split / fp.
- `timeline_eyalyakir_milleniumHotel_SamsungSM-S911B…png` — GT (top row) vs. pred (bottom row) intervals for the exp with the worst merge count, to inspect the swallowing pattern.
