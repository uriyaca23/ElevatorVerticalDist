# Iteration: 01_baseline

**What changed:** Baseline — current pair_filter.predict_pairs (greedy by mean R²).

**Variant kwargs:** `{}`

## Metrics

| metric | value |
|---|---|
| clean | 30 / 415 |
| missed | 121 |
| gt_merged | 208 |
| gt_split | 56 |
| pred_merged | 123 |
| fp | 40 |
| **f1_like** | **0.097** |
| **IoU-F1 @ 0.5** | **0.081** |
| recall | 0.072 |
| precision | 0.149 |
| mean IoU (matched) | 0.608 |

## Per-exp breakdown

| exp | gt | pred | clean | miss | merged | split | fp |
|---|---|---|---|---|---|---|---|
| RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026 | 39 | 17 | 5 | 12 | 7 | 0 | 5 |
| UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026 | 44 | 6 | 3 | 37 | 1 | 0 | 2 |
| UriyaCohenEliya_acroBuilding_GooglePixel10_15-04-2026_exp4 | 12 | 8 | 3 | 0 | 5 | 3 | 0 |
| UriyaCohenEliya_acroBuilding_SamsungSM-A235F_15-04-2026_exp4 | 12 | 7 | 2 | 0 | 5 | 2 | 0 |
| UriyaCohenEliya_beitMansour1_GooglePixel10_15-04-2026_exp5 | 18 | 11 | 3 | 1 | 7 | 3 | 0 |
| UriyaCohenEliya_beitMansour1_SamsungSM-A235F_15-04-2026_exp5 | 18 | 12 | 2 | 1 | 8 | 5 | 2 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1 | 10 | 7 | 0 | 10 | 0 | 0 | 7 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2 | 31 | 6 | 1 | 26 | 1 | 0 | 4 |
| UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp3 | 12 | 7 | 1 | 1 | 5 | 2 | 1 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp2 | 31 | 17 | 2 | 0 | 13 | 9 | 0 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp3 | 12 | 4 | 0 | 1 | 3 | 2 | 0 |
| UriyaCohenEliya_milleniumHotel_Xiaomi22101320I_15-04-2026_exp1 | 10 | 3 | 0 | 7 | 1 | 0 | 2 |
| eyalyakir_acroBuilding_SamsungSM-S911B_15-04-2026_exp4 | 12 | 9 | 1 | 0 | 6 | 4 | 1 |
| eyalyakir_acroBuilding_Xiaomi22101320I_15-04-2026_exp4 | 12 | 8 | 2 | 0 | 4 | 2 | 1 |
| eyalyakir_beitMansour1_SamsungSM-S911B_15-04-2026_exp5 | 18 | 11 | 2 | 0 | 8 | 2 | 0 |
| eyalyakir_beitMansour1_Xiaomi22101320I_15-04-2026_exp5 | 18 | 11 | 1 | 1 | 8 | 2 | 1 |
| eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp1 | 10 | 9 | 0 | 10 | 0 | 0 | 9 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2 | 31 | 15 | 1 | 1 | 13 | 5 | 1 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp3 | 12 | 7 | 0 | 1 | 7 | 4 | 0 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp2 | 31 | 19 | 1 | 1 | 14 | 7 | 3 |
| eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp3 | 12 | 8 | 0 | 1 | 7 | 4 | 1 |

## Diagnostic plots

- `errors_bar.png` — per-exp stacked breakdown of clean / missed / merged / split / fp.
- `timeline_eyalyakir_milleniumHotel_SamsungSM-S911B…png` — GT (top row) vs. pred (bottom row) intervals for the exp with the worst merge count, to inspect the swallowing pattern.
