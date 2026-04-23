# Iteration: iter_02_tighten_pair_A

**What changed:** partial revert of iter_01 — only `min_pair_abs_a` 0.30 → 0.40.
`heatmap_energy_thresh=0.20` and `same_sign_min_gap_s=5.0` unchanged.

## Metrics vs iter_01

| metric | iter_00 | iter_01 | iter_02 | Δ(01→02) |
|---|---|---|---|---|
| clean | 264 | 352 | 344 | −8 |
| missed | 232 | 136 | 146 | +10 |
| gt_merged | 2 | 10 | 8 | −2 |
| fp | 26 | 146 | 145 | **−1 (!)** |
| mistakes_total | 260 | 292 | 299 | +7 |
| f1_like | 0.669 | 0.703 | 0.694 | −0.009 |
| iou_f1@0.5 | 0.406 | 0.454 | 0.442 | −0.012 |

## Analysis — the pair-A floor is NOT the FP driver

Going 0.30 → 0.40 eliminated exactly **1 FP** while losing **8 clean**. So
the FP explosion from iter_01 is not driven by the amplitude floor — it
must come from `heatmap_energy_thresh=0.20`, which admits narrow-grid
single-cell lucky matches on noisy signals.

`same_sign_min_gap_s=5.0` can't directly create FP pairs (the pair filter
looks across +/− peaks, not same-sign), but it may produce more candidate
pairs which the greedy resolver later commits; that's likely a secondary
contributor.

## Next iteration hypothesis

Focus on `heatmap_energy_thresh`. Baseline 0.60 let through 264 clean at
the cost of 232 missed. iter_01 0.20 let through 352 clean but admitted
146 FPs. Try a middle value.

**iter_03 plan:** `heatmap_energy_thresh` 0.20 → 0.40, keep
`min_pair_abs_a=0.40` and `same_sign_min_gap_s=5.0`. Expected FP drop ~50–80
at the cost of some clean.
