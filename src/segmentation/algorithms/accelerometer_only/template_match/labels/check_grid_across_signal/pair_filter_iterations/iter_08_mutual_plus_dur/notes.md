# Iteration: 08_mutual_plus_dur

**What changed:** Combine 05 + 03: mutual nearest pairing, then duration penalty λ=0.003 as tiebreaker.

**Variant kwargs:** `{'require_mutual_nearest': True, 'duration_penalty_lambda': 0.003}`

## Metrics

| metric | value |
|---|---|
| clean | 68 / 415 |
| missed | 188 |
| gt_merged | 83 |
| gt_split | 76 |
| pred_merged | 114 |
| fp | 36 |
| **f1_like** | **0.213** |
| **IoU-F1 @ 0.5** | **0.147** |
| recall | 0.164 |
| precision | 0.304 |
| mean IoU (matched) | 0.617 |

## Per-exp breakdown

| exp | gt | pred | clean | miss | merged | split | fp |
|---|---|---|---|---|---|---|---|
| RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026 | 39 | 7 | 4 | 30 | 2 | 0 | 1 |
| UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026 | 44 | 6 | 3 | 41 | 0 | 0 | 3 |
| UriyaCohenEliya_acroBuilding_GooglePixel10_15-04-2026_exp4 | 12 | 9 | 7 | 2 | 2 | 1 | 0 |
| UriyaCohenEliya_acroBuilding_SamsungSM-A235F_15-04-2026_exp4 | 12 | 6 | 5 | 5 | 1 | 0 | 0 |
| UriyaCohenEliya_beitMansour1_GooglePixel10_15-04-2026_exp5 | 18 | 13 | 9 | 2 | 4 | 1 | 0 |
| UriyaCohenEliya_beitMansour1_SamsungSM-A235F_15-04-2026_exp5 | 18 | 15 | 4 | 1 | 8 | 5 | 1 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1 | 10 | 4 | 0 | 10 | 0 | 0 | 4 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2 | 31 | 6 | 0 | 31 | 0 | 0 | 6 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp3 | 12 | 9 | 1 | 1 | 7 | 4 | 1 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp2 | 31 | 24 | 3 | 2 | 21 | 16 | 0 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp3 | 12 | 9 | 2 | 0 | 7 | 4 | 0 |
| UriyaCohenEliya_milleniumHotel_Xiaomi22101320I_15-04-2026_exp1 | 10 | 3 | 0 | 10 | 0 | 0 | 3 |
| eyalyakir_acroBuilding_SamsungSM-S911B_15-04-2026_exp4 | 12 | 8 | 7 | 5 | 0 | 0 | 1 |
| eyalyakir_acroBuilding_Xiaomi22101320I_15-04-2026_exp4 | 12 | 12 | 8 | 3 | 0 | 1 | 2 |
| eyalyakir_beitMansour1_SamsungSM-S911B_15-04-2026_exp5 | 18 | 10 | 5 | 6 | 4 | 2 | 0 |
| eyalyakir_beitMansour1_Xiaomi22101320I_15-04-2026_exp5 | 18 | 5 | 2 | 13 | 2 | 1 | 1 |
| eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp1 | 10 | 11 | 0 | 10 | 0 | 0 | 11 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2 | 31 | 26 | 3 | 2 | 21 | 17 | 1 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp3 | 12 | 8 | 1 | 1 | 7 | 4 | 0 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp2 | 31 | 25 | 3 | 2 | 21 | 16 | 1 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp3 | 12 | 8 | 1 | 1 | 7 | 4 | 0 |

## Diagnostic plots

- `errors_bar.png` — per-exp stacked breakdown of clean / missed / merged / split / fp.
- `timeline_UriyaCohenEliya_milleniumHotel_SamsungSM…png` — GT (top row) vs. pred (bottom row) intervals for the exp with the worst merge count, to inspect the swallowing pattern.
