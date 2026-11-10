# Iteration: 04_dur_penalty_heavy

**What changed:** Duration penalty λ=0.01: aggressive — a 100 s pair must beat a 10 s pair by ≥ 0.9 R² to outrank it.

**Variant kwargs:** `{'duration_penalty_lambda': 0.01}`

## Metrics

| metric | value |
|---|---|
| clean | 81 / 415 |
| missed | 133 |
| gt_merged | 106 |
| gt_split | 95 |
| pred_merged | 136 |
| fp | 59 |
| **f1_like** | **0.228** |
| **IoU-F1 @ 0.5** | **0.169** |
| recall | 0.195 |
| precision | 0.276 |
| mean IoU (matched) | 0.625 |

## Per-exp breakdown

| exp | gt | pred | clean | miss | merged | split | fp |
|---|---|---|---|---|---|---|---|
| RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026 | 39 | 19 | 11 | 21 | 3 | 0 | 5 |
| UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026 | 44 | 8 | 3 | 41 | 0 | 0 | 5 |
| UriyaCohenEliya_acroBuilding_GooglePixel10_15-04-2026_exp4 | 12 | 11 | 6 | 0 | 4 | 3 | 0 |
| UriyaCohenEliya_acroBuilding_SamsungSM-A235F_15-04-2026_exp4 | 12 | 9 | 5 | 0 | 4 | 1 | 0 |
| UriyaCohenEliya_beitMansour1_GooglePixel10_15-04-2026_exp5 | 18 | 15 | 7 | 0 | 6 | 3 | 0 |
| UriyaCohenEliya_beitMansour1_SamsungSM-A235F_15-04-2026_exp5 | 18 | 17 | 6 | 0 | 8 | 5 | 2 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1 | 10 | 8 | 0 | 10 | 0 | 0 | 8 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2 | 31 | 6 | 0 | 31 | 0 | 0 | 6 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp3 | 12 | 10 | 1 | 1 | 7 | 5 | 1 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp2 | 31 | 28 | 5 | 0 | 21 | 18 | 0 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp3 | 12 | 9 | 1 | 0 | 7 | 4 | 0 |
| UriyaCohenEliya_milleniumHotel_Xiaomi22101320I_15-04-2026_exp1 | 10 | 3 | 0 | 10 | 0 | 0 | 3 |
| eyalyakir_acroBuilding_SamsungSM-S911B_15-04-2026_exp4 | 12 | 12 | 8 | 2 | 1 | 1 | 2 |
| eyalyakir_acroBuilding_Xiaomi22101320I_15-04-2026_exp4 | 12 | 15 | 6 | 2 | 2 | 3 | 4 |
| eyalyakir_beitMansour1_SamsungSM-S911B_15-04-2026_exp5 | 18 | 16 | 6 | 0 | 8 | 6 | 0 |
| eyalyakir_beitMansour1_Xiaomi22101320I_15-04-2026_exp5 | 18 | 17 | 6 | 0 | 8 | 6 | 1 |
| eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp1 | 10 | 18 | 0 | 3 | 1 | 0 | 17 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2 | 31 | 27 | 4 | 1 | 21 | 17 | 1 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp3 | 12 | 8 | 1 | 0 | 7 | 3 | 0 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp2 | 31 | 29 | 4 | 1 | 21 | 17 | 3 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp3 | 12 | 9 | 1 | 0 | 7 | 3 | 1 |

## Diagnostic plots

- `errors_bar.png` — per-exp stacked breakdown of clean / missed / merged / split / fp.
- `timeline_eyalyakir_milleniumHotel_SamsungSM-S911B…png` — GT (top row) vs. pred (bottom row) intervals for the exp with the worst merge count, to inspect the swallowing pattern.
