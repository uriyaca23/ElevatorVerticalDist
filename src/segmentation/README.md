# Segmentation — the problem & the accelerometer signature

## The problem

Slice a continuous, gravity-removed accelerometer stream from a phone (held by
a passenger) into discrete **elevator-ride intervals**, each tagged
`up` / `down` / `outside`. The detector must work on phones that have **no
barometer**, so we can't lean on altitude — only on `ACC` (and, when present,
`GYR` / `MAG` / `ORI`). A barometer-derived label, where available, is the
ground truth we evaluate against (see `src/data/loader/pipeline.py` and
`src/segmentation/algorithms/barometer_only/`).

## What an elevator ride looks like in `|a|`

Modern passenger elevators are commanded by a **jerk-limited motion profile**,
saturating the constraints

```
|jerk|        ≤ j_max   (~1.0–1.6 m/s³)
|acceleration|≤ a_max   (~0.8–1.5 m/s²)
|velocity|    ≤ v_max   (~1.0–2.5 m/s)
```

Saturating each in turn yields the canonical **seven-segment S-curve**:

```
  jerk-up · cruise(+a_max) · jerk-down · cruise(v_max) · jerk-down · cruise(-a_max) · jerk-up
```

In the gravity-removed accelerometer this comes out as **two opposite-sign
trapezoidal lobes** separated by a quiet plateau (the `v_max` cruise):

```
   |a|
    │       ___                        ___
    │      /   \                      /   \
    │     /     \____________________/     \
    │____/                                  \____  → t
       ramp-up   v_max cruise   ramp-down (mirror, opposite sign)
```

Three facts the detector exploits:

1. **Jerk phases are linear in `a(t)`** — slope `±j_max`. Each lobe is a
   trapezoid (triangle on short hops where `v_max` is never reached).
2. **Signed area under each lobe = peak velocity**:
   `v_peak = a_max · (T_j + T_a)`, and the two lobes cancel
   (`∫a dt = 0` over the whole ride — the cabin starts and ends at rest).
   This is the basis of ZUPT.
3. **Inter-lobe spacing ≈ cruise time `T_v ≈ H / v_max`**, so the gap between
   the two acceleration lobes is a direct, scale-free proxy for floor-to-floor
   height `H`.

A ride is therefore fully described by four parameters:
`θ = (j_max, a_max, v_max, H)`. The first three are properties of the
elevator/building (constant within a building); only `H` varies per ride.

## Visual reference

- Single ride going **up** —
  [`elevator_kinematics_up_example.png`](../../docs/latex/figures/elevator_kinematics_up_example.png):
  positive lobe at takeoff, negative lobe at landing, GT label shaded.
- Single ride going **down** —
  [`elevator_kinematics_down_example.png`](../../docs/latex/figures/elevator_kinematics_down_example.png):
  lobes mirrored; the velocity panel shows the trapezoidal `v(t)`.
- **All rides** from one experiment on a common layout —
  [`elevator_kinematics_all_rides.png`](../../docs/latex/figures/elevator_kinematics_all_rides.png):
  the two-lobe `+a` then `−a` structure repeats; lobe spacing scales with `H`,
  lobe height saturates at the building's `a_max`.

(Source: `eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp3`.)

## How this shapes the detectors here

| Family | What it leans on | Code |
|---|---|---|
| **Barometer-only** (GT source) | `v_z` thresholding on smoothed altitude | `algorithms/barometer_only/` |
| **State machine / pulse detect** | `a_max` threshold + lobe-duration floor `≈ T_j` | `algorithms/accelerometer_only/template_match/scripts/{acc_,}pulse_detect.py` |
| **Matched-filter template matching (NCC)** | Synthetic `a_θ(t)` family from the four parameters above, searched by normalized cross-correlation with closed-form LS amplitude | `algorithms/accelerometer_only/template_match/` |
| **Velocity-integral / ZUPT** | `∫a dt = 0` and `∫|v| dt ≳ H_min` | (planned) |

## Pointers for deeper context

- Full physics derivation, equations, and references: `docs/latex/main.tex`,
  Section 3.1 “Background: Elevator Kinematics”.
- Ground-truth pipeline & alignment: `src/data/README.md`.
- Per-experiment GT visualisation/edits: `src/data/gt_editor.py`.

---

## Tuning round — 2026-04 (iter_07 promotion)

A 12-iteration sweep (`algorithms/improvement_iterations/iter_00_baseline`
… `iter_11_quiet_edges`) reduced segmentation mistakes from **260 → 154
on 498 GT rides (−40.8 %)**. The winning configuration, `iter_07_lower_peak_gates`, is what sits on `development` now.

### Headline delta (all 26 experiments, 498 GT rides)

| | before | after | Δ |
|---|---|---|---|
| clean (one-to-one matches) | 264 (53 %) | **403 (81 %)** | +139 |
| missed | 232 | 88 | −144 |
| fp | 26 | 59 | +33 |
| gt_merged / gt_split | 2 / 0 | 6 / 1 | +5 |
| **mistakes_total** | **260** | **154** | **−106** |
| f1_like | 0.669 | 0.835 | +0.166 |
| iou_f1 @ 0.5 | 0.406 | 0.601 | +0.195 |

### Algorithmic changes

1. **New `quiet_middle_ratio` filter in `pair_filter.predict_pairs`.**
   A ride's cabin cruises at constant velocity between the two
   acceleration lobes, so `a_vert ≈ 0` in the plateau. Walking false
   positives have continuous motion there. For each candidate pair we
   compute `mid_rms = RMS(a_smooth[t1+W : t2-W])` and reject the pair
   if `mid_rms > quiet_middle_ratio × pair_A_abs`. Single biggest FP
   cut of the round (FPs 99 → 50 in iter_04).
2. **Per-sign local NMS in `detect.detect`.** `_peak_pick` used to run
   on the unsigned `best_r2` array, so a strong +peak could suppress a
   valid −peak within `nms_radius_s`. It now runs once per sign
   (pos and neg gated separately) and the two peak sets are merged. No
   metric impact at the current gates but architecturally removes a
   failure mode that showed up whenever the tuning loosened thresholds.

### Hyperparameter changes (`config.json` + mirrored dataclass/pydantic defaults)

| knob | before | after | why |
|---|---|---|---|
| `r2_peak_thresh` | 0.55 | **0.40** | Admit borderline lobes on noisy recordings. Unlocked Roy Turgeman Haari (19 missed → 1). |
| `min_peak_abs_a` | 0.40 | **0.25** | Same motivation — short / low-amp rides had lobes below the old floor. |
| `nms_radius_s` | 2.0 | **1.0** | Minor; kept for consistency with the per-sign NMS change. |
| `same_sign_min_gap_s` | 10.0 | **5.0** | Dense ride sequences (e.g. milleniumHotel exp2, 31 rides in 15 min) were losing same-sign peaks. |
| `min_pair_abs_a` | 0.50 | **0.30** | Catches modest-amplitude pairs. Safe because `quiet_middle_ratio` now gates walking FPs. |
| `heatmap_energy_thresh` | 0.60 | **0.40** | Was the biggest source of false rejections — short/narrow-grid-support rides were hitting it. 0.40 is the empirical sweet spot (0.20 admits too much noise, 0.60 rejects too many real rides). |
| **NEW** `quiet_middle_ratio` | — | **0.5** | See algorithmic change #1. |

### What was tried and rolled back

Iterations 08 – 11 explored further tightening in several directions
(push peak gates harder, widen `max_ride_s`, tighten `quiet_middle_ratio`
or `joint_r2_thresh`, add `min_lobe_r2`, add a quiet-edges filter). All
either no-opped or regressed. The rejected ideas and their metric
deltas are documented per iteration in
`algorithms/improvement_iterations/iter_NN_*/notes.md` and summarised
in `algorithms/improvement_iterations/README.md`.

### Why not a bigger win

≈ 30 of the remaining 88 missed rides come from a single recording
(`UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2`) where
every lobe has `|A| ≈ 0.10 – 0.20 m/s²`. That is a data-quality issue
(phone damped or stationary in a bag), not an algorithm issue. Further
gains likely need session-adaptive noise floors, a dedicated low-amp
fallback detector, or fixing that recording.

### Branches kept for follow-up

All 12 per-iteration branches (`iter_00_baseline` … `iter_11_quiet_edges`)
are preserved locally. Each branch has its own
`algorithms/improvement_iterations/iter_NN_<slug>/` folder with
`metrics.json`, `per_gt.csv`, `mistakes/*.png`, `per_exp_summary.png`
and `notes.md`. Rerun the whole evaluation on any branch with:

```bash
venv/bin/python -m src.segmentation.algorithms.improvement_iterations._iter_runner \
    --iter NN --slug <slug> --kind all --what "<one-line description>"
```
