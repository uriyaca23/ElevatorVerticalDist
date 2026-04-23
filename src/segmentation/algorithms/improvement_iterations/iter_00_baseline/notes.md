# Iteration: iter_00_baseline

**What changed:** Baseline — production pair_filter.py as of commit c4d2a9f.
Config: `r2_peak_thresh=0.55`, `min_peak_abs_a=0.4`, `nms_radius_s=2.0`,
`same_sign_min_gap_s=10.0`, `min_ride_s=0.0`, `max_ride_s=30.0`,
`joint_r2_thresh=0.90`, `min_pair_abs_a=0.5`, `heatmap_energy_thresh=0.60`.
Pair-filter duration penalty `λ=0.01` s⁻¹ in `pair_filter.py`.

## Headline metrics (26 experiments, 498 GTs)

| metric | value |
|---|---|
| n_gt | 498 |
| clean | 264 (53%) |
| **missed** | **232 (46.6%)** |
| gt_merged | 2 |
| gt_split | 0 |
| pred_merged | 1 |
| fp | 26 |
| **mistakes_total** | **260** |
| **target (≥70% reduction)** | **≤ 78 mistakes** |
| f1_like | 0.669 |
| iou_f1@0.5 | 0.406 |
| mean IoU (matched) | 0.666 |

## Failure-mode breakdown (from `per_gt.csv`)

### Where does `missed` come from?

| sub-category | count | % of missed |
|---|---|---|
| `pair_reject_flags=""` (pair would clear filters, but GT still missed → peak killed upstream by NMS / gates) | 101 | 44 % |
| `low_pair_A` alone (pair `|A|` < `min_pair_abs_a=0.5`) | 61 | 26 % |
| `low_joint_r2 + low_heatmap_energy` | 35 | 15 % |
| `low_joint_r2 + low_heatmap_energy + low_pair_A` | 16 | 7 % |
| `low_heatmap_energy + low_pair_A` | 15 | 6 % |
| other combos | 4 | 2 % |

- **Peaks totally absent from GT window:** 94 / 232 = 41 %. Detector gates
  (`r2_peak_thresh`, `min_peak_abs_a`) or the local/same-sign NMS dropped
  every sample inside.
- **Peaks present, pair would theoretically clear filters, but no pred:** 101.
  The `diagnose_window` best ± samples pass all pair-fit thresholds, yet the
  detector didn't emit a ride. Three possible causes:
  1. The ± sample was dropped by `_peak_pick`'s local-NMS (`nms_radius_s=2.0`)
     because a nearby higher-R² sample won but that sample then failed pair
     conditions with anything.
  2. The ± sample was dropped by `_same_sign_nms` (`same_sign_min_gap_s=10`)
     — a neighbouring ride's same-sign peak had higher R². This is the most
     plausible driver for dense ride sequences (milleniumHotel exp2 has 31
     rides packed into a short window).
  3. Greedy duration-penalty resolver picked a competing pair that claimed
     one lobe.

### Peak-fit stats for the 138 missed GTs where both peaks DO exist

| field | median | 25 % | 75 % |
|---|---|---|---|
| `pos_r2` | 0.951 | 0.895 | 0.964 |
| `neg_r2` | 0.951 | 0.861 | 0.970 |
| `pair_joint_r2` | 0.938 | 0.729 | 0.956 |
| `pair_heatmap_energy` | 0.605 | 0.174 | 0.707 |

Observations:
- Per-lobe R² is excellent (median 0.95). The shape is there.
- Joint R² median is 0.938 — above the 0.90 threshold. So 50 %+ of these
  missed GTs have a pair that clears `joint_r2_thresh`. The **next** gates
  (heatmap / pair_A) are what kill them.
- Heatmap energy: median 0.60, with the 25th percentile at 0.17. This
  threshold is acting as a narrow gate — half the valid pairs sit below it.

## Per-experiment patterns

Best (≥90 % clean):
- `eyalyakir_milleniumHotel_SamsungSM-S911B_exp3` 12/12
- `eyalyakir_milleniumHotel_Xiaomi_exp3` 12/12
- `UriyaCohenEliya_milleniumHotel_GooglePixel10_exp3` 11/12
- `eyalyakir_milleniumHotel_SamsungSM-S911B_exp2` 29/31
- `eyalyakir_milleniumHotel_Xiaomi_exp2` 30/31
- `UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_exp2` 30/31

Worst (0 clean):
- `UriyaCohenEliya_milleniumHotel_GooglePixel10_exp1` 0/10
- `UriyaCohenEliya_milleniumHotel_GooglePixel10_exp2` 0/31
- `UriyaCohenEliya_milleniumHotel_Xiaomi_exp1` 0/10
- `eyalyakir_milleniumHotel_SamsungSM-A235F_exp1` 0/10
- `eyalyakir_milleniumHotel_SamsungSM-S911B_exp1` 0/10
- `RoyTurgeman_Haari3_SamsungGalaxyZFlip6` 1/30 — new experimenter, noisy
- `UriyaCohenEliya_BarIlan2Herzelia_Pixel10` 0/20 — new building

Same building + different phone produces very different detection rates
(milleniumHotel_SamsungSM-S911B_exp2 = 29/31 vs exp1 = 0/10). The failure
isn't phone-specific — it's tied to ride characteristics in each exp. exp1
recordings tend to have short / low-amplitude rides.

## Sample mistake PNGs inspected

`UriyaCohenEliya_milleniumHotel_GooglePixel10_exp1__gt00__missed.png`:
- Clear ±1 m/s² lobe structure inside the 15 s GT window.
- `+ lobe R²=0.58 OK`, `− lobe R²=0.73 OK`, `pair R²=0.889` (just below 0.90),
  `heatmap_energy=0.089` (FAR below 0.60) → **rejected for heatmap energy**.
- Fitted `W=0.40s` (grid minimum), `f=0.77` (almost pure flat). Short pulse
  shape — grid breadth can't cover wide W support.

## Next iteration hypothesis

The three cheapest wins, by root cause:

1. **Loosen `heatmap_energy_thresh` from 0.60 → 0.20**. This filter is the
   single biggest source of false rejections. Short or amplitude-modest
   rides naturally produce narrow-grid-support pairs. Expected wins: 50 +
   missed rides. Risk: may admit spurious single-cell matches — watch FP
   count.
2. **Loosen `min_pair_abs_a` from 0.5 → 0.30**. 61 missed GTs fail on this
   alone; 93 fail on it in combination. Many rides have modest shared
   amplitude — the lobes are there but combined `|A|` < 0.5. Risk: low-amp
   FPs on noisy sessions (RoyTurgeman, BarIlan).
3. **Tighten `same_sign_min_gap_s` from 10.0 → 5.0**. Dense ride
   sequences (milleniumHotel exp2 — 31 rides in a few minutes) drop peaks
   because consecutive same-sign take-offs are < 10 s apart. Risk: may
   admit ringing tails as spurious extra peaks.

**iter_01 plan:** apply all three together (aggressive). If FP rate
explodes, iter_02 backs each off individually to find the sweet spot.
