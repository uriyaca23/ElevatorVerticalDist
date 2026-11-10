# Iteration: 06_quiet_middle

**What changed:** Quiet-middle constraint: reject if |a_smooth| between lobes ever exceeds 0.5·A_abs. A real ride's middle is cruise — nearly zero acceleration.

**Variant kwargs:** `{'require_quiet_middle': True, 'quiet_middle_ratio': 0.5}`

## Metrics

| metric | value |
|---|---|
| clean | 42 / 415 |
| missed | 252 |
| gt_merged | 79 |
| gt_split | 42 |
| pred_merged | 79 |
| fp | 0 |
| **f1_like** | **0.157** |
| **IoU-F1 @ 0.5** | **0.157** |
| recall | 0.101 |
| precision | 0.347 |
| mean IoU (matched) | 0.625 |

## Per-exp breakdown

| exp | gt | pred | clean | miss | merged | split | fp |
|---|---|---|---|---|---|---|---|
| RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026 | 39 | 0 | 0 | 39 | 0 | 0 | 0 |
| UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026 | 44 | 0 | 0 | 44 | 0 | 0 | 0 |
| UriyaCohenEliya_acroBuilding_GooglePixel10_15-04-2026_exp4 | 12 | 4 | 4 | 8 | 0 | 0 | 0 |
| UriyaCohenEliya_acroBuilding_SamsungSM-A235F_15-04-2026_exp4 | 12 | 5 | 5 | 7 | 0 | 0 | 0 |
| UriyaCohenEliya_beitMansour1_GooglePixel10_15-04-2026_exp5 | 18 | 3 | 0 | 13 | 3 | 1 | 0 |
| UriyaCohenEliya_beitMansour1_SamsungSM-A235F_15-04-2026_exp5 | 18 | 5 | 4 | 12 | 1 | 0 | 0 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2 | 31 | 0 | 0 | 31 | 0 | 0 | 0 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp3 | 12 | 7 | 1 | 2 | 6 | 3 | 0 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp2 | 31 | 10 | 1 | 15 | 9 | 3 | 0 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp3 | 12 | 7 | 1 | 2 | 6 | 3 | 0 |
| UriyaCohenEliya_milleniumHotel_Xiaomi22101320I_15-04-2026_exp1 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| eyalyakir_acroBuilding_SamsungSM-S911B_15-04-2026_exp4 | 12 | 5 | 5 | 7 | 0 | 0 | 0 |
| eyalyakir_acroBuilding_Xiaomi22101320I_15-04-2026_exp4 | 12 | 5 | 5 | 7 | 0 | 0 | 0 |
| eyalyakir_beitMansour1_SamsungSM-S911B_15-04-2026_exp5 | 18 | 6 | 4 | 10 | 2 | 0 | 0 |
| eyalyakir_beitMansour1_Xiaomi22101320I_15-04-2026_exp5 | 18 | 6 | 4 | 10 | 2 | 0 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp1 | 10 | 1 | 0 | 3 | 1 | 0 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2 | 31 | 21 | 4 | 5 | 17 | 12 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp3 | 12 | 7 | 1 | 2 | 6 | 3 | 0 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp2 | 31 | 22 | 2 | 3 | 20 | 14 | 0 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp3 | 12 | 7 | 1 | 2 | 6 | 3 | 0 |

## Diagnostic plots

- `errors_bar.png` — per-exp stacked breakdown of clean / missed / merged / split / fp.
- `timeline_eyalyakir_milleniumHotel_Xiaomi22101320I…png` — GT (top row) vs. pred (bottom row) intervals for the exp with the worst merge count, to inspect the swallowing pattern.
