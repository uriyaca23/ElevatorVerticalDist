# Iteration: iter_06_per_sign_pick

**What changed:** Per-sign local NMS in `detect.detect`. Run `_peak_pick`
twice — once on the `+sign` subset of `best_r2_gated`, once on the
`-sign` subset — then merge the two peak sets. Fixes the case where a
+peak's high R² suppressed a valid −peak within `nms_radius_s`.

## Metrics

| metric | iter_04 | iter_05 | iter_06 | Δ(05→06) |
|---|---|---|---|---|
| clean | 334 | 334 | 334 | 0 |
| missed | 148 | 148 | 148 | 0 |
| fp | 50 | 54 | 54 | 0 |
| gt_merged | 15 | 16 | 16 | 0 |
| mistakes_total | 214 | 218 | 218 | 0 |

## Analysis — hypothesis rejected, root cause identified

Per-sign peak picking made zero difference. Inspecting individual
"no_flags" missed GTs via a debug script shows the real problem:

```
RoyTurgeman_Haari3 gt4 [109.7, 123.8]
  initial_peaks in ±5 s window: 0
  final_peaks in window: 0
```

```
RoyTurgeman_Haari3 gt5 [132.9, 146.8]
  initial_peaks: 1  (t=136.4, A=-0.40, r2=0.99)
  final_peaks:   1  (same)
```

**The problem is upstream of local NMS: the peak-pick gates
(`r2_peak_thresh=0.55`, `min_peak_abs_a=0.4`) filter out valid lobes.**
`diagnose_window` reports a theoretical pair because it uses
`_find_extrema_in_window` which ignores the gates — it reports the
argmax of `best_A` in the window even when the R² at that sample is
below `r2_peak_thresh`. Those weak samples never enter `initial_peaks`,
so the pair filter can't use them.

Per-sign vs unsigned NMS doesn't matter when the samples never got past
the gate in the first place.

## Next iteration hypothesis

Lower the peak-pick gates so more borderline lobes enter the candidate
set. The pair filter has four independent strict gates
(`joint_r2_thresh=0.90`, `min_pair_abs_a=0.30`, `heatmap_energy=0.40`,
`quiet_middle_ratio=0.5`) that will filter out noise.

**iter_07 plan:**
- `r2_peak_thresh` 0.55 → 0.40
- `min_peak_abs_a` 0.40 → 0.25

Retain the per-sign NMS from this iter (doesn't hurt, may help once the
gates admit more candidates).
