# Iteration: 02_dur_penalty_light

**What changed:** Duration penalty λ=0.001: rank by score - 0.001·Δt to nudge short pairs ahead of long ones at similar scores.

**Variant kwargs:** `{'duration_penalty_lambda': 0.001}`

## Metrics

| metric | value |
|---|---|
| clean | 65 / 415 |
| missed | 136 |
| gt_merged | 118 |
| gt_split | 96 |
| pred_merged | 148 |
| fp | 54 |
| **f1_like** | **0.188** |
| **IoU-F1 @ 0.5** | **0.156** |
| recall | 0.157 |
| precision | 0.234 |
| mean IoU (matched) | 0.618 |

## Per-exp breakdown

| exp | gt | pred | clean | miss | merged | split | fp |
|---|---|---|---|---|---|---|---|
| RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026 | 39 | 19 | 8 | 19 | 5 | 0 | 6 |
| UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026 | 44 | 6 | 3 | 41 | 0 | 0 | 3 |
| UriyaCohenEliya_acroBuilding_GooglePixel10_15-04-2026_exp4 | 12 | 9 | 5 | 0 | 4 | 2 | 0 |
| UriyaCohenEliya_acroBuilding_SamsungSM-A235F_15-04-2026_exp4 | 12 | 9 | 5 | 0 | 4 | 1 | 0 |
| UriyaCohenEliya_beitMansour1_GooglePixel10_15-04-2026_exp5 | 18 | 15 | 4 | 1 | 9 | 6 | 1 |
| UriyaCohenEliya_beitMansour1_SamsungSM-A235F_15-04-2026_exp5 | 18 | 16 | 4 | 1 | 9 | 6 | 2 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1 | 10 | 8 | 0 | 10 | 0 | 0 | 8 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2 | 31 | 6 | 0 | 31 | 0 | 0 | 6 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp3 | 12 | 10 | 1 | 0 | 7 | 4 | 1 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp2 | 31 | 27 | 3 | 0 | 22 | 18 | 0 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp3 | 12 | 8 | 1 | 0 | 7 | 3 | 0 |
| UriyaCohenEliya_milleniumHotel_Xiaomi22101320I_15-04-2026_exp1 | 10 | 3 | 0 | 10 | 0 | 0 | 3 |
| eyalyakir_acroBuilding_SamsungSM-S911B_15-04-2026_exp4 | 12 | 12 | 5 | 0 | 4 | 2 | 2 |
| eyalyakir_acroBuilding_Xiaomi22101320I_15-04-2026_exp4 | 12 | 13 | 5 | 0 | 4 | 2 | 3 |
| eyalyakir_beitMansour1_SamsungSM-S911B_15-04-2026_exp5 | 18 | 16 | 6 | 1 | 8 | 6 | 1 |
| eyalyakir_beitMansour1_Xiaomi22101320I_15-04-2026_exp5 | 18 | 16 | 5 | 2 | 8 | 6 | 2 |
| eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp1 | 10 | 13 | 1 | 9 | 0 | 0 | 12 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2 | 31 | 27 | 4 | 1 | 21 | 17 | 1 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp3 | 12 | 8 | 1 | 0 | 7 | 3 | 0 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp2 | 31 | 28 | 3 | 0 | 22 | 17 | 2 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp3 | 12 | 9 | 1 | 0 | 7 | 3 | 1 |

## Diagnostic plots

- `errors_bar.png` — per-exp stacked breakdown of clean / missed / merged / split / fp.
- `timeline_eyalyakir_milleniumHotel_Xiaomi22101320I…png` — GT (top row) vs. pred (bottom row) intervals for the exp with the worst merge count, to inspect the swallowing pattern.
