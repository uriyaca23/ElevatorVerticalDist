# Iteration: 10_combined_final

**What changed:** Kitchen sink: mutual nearest + quiet middle + min R² + λ=0.003 duration penalty.

**Variant kwargs:** `{'require_mutual_nearest': True, 'require_quiet_middle': True, 'quiet_middle_ratio': 0.5, 'require_min_r2': True, 'duration_penalty_lambda': 0.003}`

## Metrics

| metric | value |
|---|---|
| clean | 30 / 415 |
| missed | 275 |
| gt_merged | 68 |
| gt_split | 42 |
| pred_merged | 76 |
| fp | 0 |
| **f1_like** | **0.115** |
| **IoU-F1 @ 0.5** | **0.115** |
| recall | 0.072 |
| precision | 0.283 |
| mean IoU (matched) | 0.614 |

## Per-exp breakdown

| exp | gt | pred | clean | miss | merged | split | fp |
|---|---|---|---|---|---|---|---|
| RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026 | 39 | 0 | 0 | 39 | 0 | 0 | 0 |
| UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026 | 44 | 0 | 0 | 44 | 0 | 0 | 0 |
| UriyaCohenEliya_acroBuilding_GooglePixel10_15-04-2026_exp4 | 12 | 4 | 4 | 8 | 0 | 0 | 0 |
| UriyaCohenEliya_acroBuilding_SamsungSM-A235F_15-04-2026_exp4 | 12 | 4 | 4 | 8 | 0 | 0 | 0 |
| UriyaCohenEliya_beitMansour1_GooglePixel10_15-04-2026_exp5 | 18 | 3 | 0 | 13 | 3 | 1 | 0 |
| UriyaCohenEliya_beitMansour1_SamsungSM-A235F_15-04-2026_exp5 | 18 | 3 | 2 | 14 | 1 | 0 | 0 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2 | 31 | 0 | 0 | 31 | 0 | 0 | 0 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp3 | 12 | 7 | 1 | 2 | 6 | 3 | 0 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp2 | 31 | 9 | 0 | 16 | 9 | 3 | 0 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp3 | 12 | 7 | 1 | 2 | 6 | 3 | 0 |
| UriyaCohenEliya_milleniumHotel_Xiaomi22101320I_15-04-2026_exp1 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| eyalyakir_acroBuilding_SamsungSM-S911B_15-04-2026_exp4 | 12 | 4 | 4 | 8 | 0 | 0 | 0 |
| eyalyakir_acroBuilding_Xiaomi22101320I_15-04-2026_exp4 | 12 | 4 | 4 | 8 | 0 | 0 | 0 |
| eyalyakir_beitMansour1_SamsungSM-S911B_15-04-2026_exp5 | 18 | 4 | 2 | 12 | 2 | 0 | 0 |
| eyalyakir_beitMansour1_Xiaomi22101320I_15-04-2026_exp5 | 18 | 1 | 0 | 16 | 1 | 0 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp1 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2 | 31 | 21 | 3 | 5 | 18 | 13 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp3 | 12 | 7 | 1 | 2 | 6 | 3 | 0 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp2 | 31 | 21 | 3 | 5 | 18 | 13 | 0 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp3 | 12 | 7 | 1 | 2 | 6 | 3 | 0 |

## Diagnostic plots

- `errors_bar.png` — per-exp stacked breakdown of clean / missed / merged / split / fp.
- `timeline_eyalyakir_milleniumHotel_SamsungSM-S911B…png` — GT (top row) vs. pred (bottom row) intervals for the exp with the worst merge count, to inspect the swallowing pattern.
