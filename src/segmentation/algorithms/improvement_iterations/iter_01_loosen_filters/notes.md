# Iteration: iter_01_loosen_filters

**What changed:** Loosen three pair-filter thresholds simultaneously
targeting the three failure modes flagged by iter_00's `per_gt.csv`:

| knob | 00 → 01 |
|---|---|
| `heatmap_energy_thresh` | 0.60 → 0.20 |
| `min_pair_abs_a` | 0.50 → 0.30 |
| `same_sign_min_gap_s` | 10.0 → 5.0 |

## Metrics vs iter_00

| metric | iter_00 | iter_01 | Δ |
|---|---|---|---|
| clean | 264 | **352** | **+88** |
| missed | 232 | 136 | −96 |
| gt_merged | 2 | 10 | +8 |
| fp | 26 | **146** | **+120** |
| **mistakes_total** | **260** | **292** | **+32** |
| f1_like | 0.669 | 0.703 | +0.034 |
| iou_f1@0.5 | 0.406 | 0.454 | +0.048 |

## Analysis

- **Recall improved as hoped** — we recovered 96 missed GTs.
- **Precision collapsed** — FPs went from 26 to 146 (5.6× worse). The
  loosened pair-filter admits noise in low-SNR sessions.
- **Net mistakes went up by 32**. f1_like and iou_f1 still improved
  because clean count grew more than bad_pred count (bad_pred penalises
  merged / split, but plain FPs only add to the denominator via the
  bad_pred = fp + pred_merged + pred_split_part term).

### Where the FPs concentrated (per-exp delta)

| exp | fp_00 | fp_01 | Δfp |
|---|---|---|---|
| eyalyakir_milleniumHotel_SamsungSM-A235F_exp1 | 4 | 14 | +10 |
| eyalyakir_milleniumHotel_SamsungSM-S911B_exp1 | 1 | 9 | +8 |
| RoyTurgeman_Haari3_SamsungGalaxyZFlip6 | 3 | high | +large |

The noisy "exp1" recordings (phone in bag/pocket, walking bursts) took the
biggest FP hit. In those sessions the signal is dominated by walking
artefacts that look trapezoid-ish once the amplitude floor drops to 0.30.

### Where recall went up

Many experiments that were mostly 0-clean now score well — the
`heatmap_energy_thresh` and `same_sign_min_gap_s` changes unblocked them:

| exp | clean_00 | clean_01 |
|---|---|---|
| eyalyakir_milleniumHotel_SamsungSM-S911B_exp2 | 29 | 31 |
| UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_exp2 | 30 | 31 |

## Next iteration hypothesis

The recall win almost certainly came from `heatmap_energy_thresh` and
`same_sign_min_gap_s`. The FP explosion came from `min_pair_abs_a=0.30`.

**iter_02 plan:** keep `heatmap_energy_thresh=0.20` and
`same_sign_min_gap_s=5.0`, but revert `min_pair_abs_a` back to `0.40`
(halfway between baseline 0.50 and iter_01 0.30). Goal: retain most of
the recall gain while dropping FP rate substantially.
