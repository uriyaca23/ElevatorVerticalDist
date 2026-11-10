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

## Tuning round — 2026-04 (iter_16 promotion)

A continuing iteration sweep (`algorithms/improvement_iterations/iter_00_baseline`
… `iter_19_pair_a_26`) further reduced segmentation mistakes.
Latest best: **iter_16_lower_peak_a — 144 mistakes / 498 GT rides (−44.6 % vs
iter_00 baseline; −6.5 % vs the previously deployed iter_07)**. The
iter_13 → iter_16 tranche added explicit triangle-shape template support
for one-floor rides, narrowed the W grid, and lowered the peak-amplitude
gate in lockstep.

### Headline delta (all 26 experiments, 498 GT rides)

| | iter_00 baseline | iter_07 | **iter_16 (new best)** | Δ vs baseline | Δ vs iter_07 |
|---|---|---|---|---|---|
| clean (one-to-one matches) | 264 (53 %) | 403 (81 %) | **400 (80 %)** | +136 | −3 |
| missed | 232 | 88 | 87 | −145 | −1 |
| fp | 26 | 59 | **46** | +20 | **−13** |
| gt_merged / gt_split | 2 / 0 | 6 / 1 | 10 / 1 | +9 | +4 |
| **mistakes_total** | 260 | 154 | **144** | **−116 (−44.6 %)** | **−10 (−6.5 %)** |
| f1_like | 0.669 | 0.835 | **0.841** | +0.172 | +0.006 |
| **iou_f1 @ 0.5** | 0.406 | 0.601 | **0.705** | **+0.299** | **+0.104** |
| mean IoU (matched) | — | — | 0.629 | — | — |

### Algorithmic changes

1. **New `quiet_middle_ratio` filter in `pair_filter.predict_pairs`** (iter_04).
   A ride's cabin cruises at constant velocity between the two
   acceleration lobes, so `a_vert ≈ 0` in the plateau. Walking false
   positives have continuous motion there. For each candidate pair we
   compute `mid_rms = RMS(a_smooth[t1+W : t2-W])` and reject the pair
   if `mid_rms > quiet_middle_ratio × pair_A_abs`. Single biggest FP
   cut of the round (FPs 99 → 50 in iter_04).
2. **Per-sign local NMS in `detect.detect`** (iter_06). `_peak_pick` used to run
   on the unsigned `best_r2` array, so a strong +peak could suppress a
   valid −peak within `nms_radius_s`. It now runs once per sign
   (pos and neg gated separately) and the two peak sets are merged. No
   metric impact at the current gates but architecturally removes a
   failure mode that showed up whenever the tuning loosened thresholds.
3. **Triangle-shape template row in `DetectConfig.grid_f()`** (iter_13).
   A one-floor (short) ride has no cruise phase, so each acceleration
   lobe collapses to a pure triangle — a trapezoid kernel with `f=0`.
   The grid now prepends `f=0` to the 15-point trapezoid `linspace`;
   the existing per-pair joint-R² argmax in `pair_filter.joint_pair_score`
   picks trapezoid-vs-triangle per pair based on shared-shape R². This
   did not move the mistake count but drove `iou_f1@0.5` from 0.601 →
   0.703 (+17 %) — fitted endpoints snap tighter when the true lobe
   has no plateau.

### Hyperparameter changes (`DetectConfig` defaults in `check_grid_across_signal/detect.py`)

| knob | iter_00 | iter_07 | **iter_16** | why (iter_16 additions highlighted) |
|---|---|---|---|---|
| `r2_peak_thresh` | 0.55 | **0.40** | 0.40 | Unlocked borderline lobes on noisy recordings (iter_07). |
| `min_peak_abs_a` | 0.40 | 0.25 | **0.20** | Further lowered in iter_16 — admits weaker peaks so the pair filter can evaluate them; FPs dropped 49 → 46. |
| `w_min_s` (grid W floor) | 0.40 | 0.40 | **0.30** | Narrower W templates (iter_15) — match short-lobe rides better, reject walking artifacts more aggressively (FPs 59 → 49). |
| `nms_radius_s` | 2.0 | **1.0** | 1.0 | Minor; kept for consistency with per-sign NMS. |
| `same_sign_min_gap_s` | 10.0 | **5.0** | 5.0 | Dense ride sequences were losing same-sign peaks (iter_07). |
| `min_pair_abs_a` | 0.50 | **0.30** | 0.30 | Catches modest-amplitude pairs; `quiet_middle_ratio` gates walking FPs. iter_14/19 tried lower values (0.22, 0.26) — both caused gt_split over-segmentation. 0.30 is the boundary. |
| `heatmap_energy_thresh` | 0.60 | **0.40** | 0.40 | Biggest source of false rejections at 0.60 (iter_03). |
| `quiet_middle_ratio` | — | **0.5** | 0.5 | Pair-filter plateau check. Single biggest iteration win (iter_04, FPs 99 → 50). |
| **NEW** `include_triangle_row` | — | — | **True** | Prepends f=0 (pure triangle) to `grid_f()`. No composite-mistake effect but +0.10 on IoU_F1 (iter_13). |

### What was tried and rolled back

Iterations 08 – 11 explored further tightening in several directions
(push peak gates harder, widen `max_ride_s`, tighten `quiet_middle_ratio`
or `joint_r2_thresh`, add `min_lobe_r2`, add a quiet-edges filter). All
either no-opped or regressed.

Iterations 14, 17, 18, 19 explored loosening `min_pair_abs_a` below 0.30
and tuning the duration penalty `λ` in the greedy resolver. Results:
gt_split over-segmentation every time. The pair-filter greedy resolver
prefers higher-R² pairs, so once sub-pulse pairs pass the amplitude gate
they win over longer multi-floor pairs. A scalar λ cannot simultaneously
prevent nested sub-pulses (needs λ < 0, rewarding long pairs) and break
super-pairs (needs λ > 0, penalising long pairs). Further gains on this
axis require a nesting-aware resolver — a structural change outside the
scope of the threshold-sweep iterations.

The rejected ideas and their metric deltas are documented per iteration
in `algorithms/improvement_iterations/iter_NN_*/notes.md` and summarised
in `algorithms/improvement_iterations/README.md`.

### Why not a bigger win

Of the remaining 87 missed rides (iter_16): 31 come from a single damped-
recording experiment (`UriyaCohenEliya_milleniumHotel_GooglePixel10_
15-04-2026_exp2`), and another ~25 come from 4 other damped/noisy-phone
recordings (Samsung A235F_exp1, Xiaomi_exp1, Xiaomi_exp6, SS911B_exp1 at
the same milleniumHotel location). In all of these the true lobes have
`|A| ≈ 0.10 – 0.20 m/s²` — below the amplitude floors that keep FPs
sane on the rest of the dataset. Analysis of iter_16's `per_gt.csv`:
50 of the 87 missed rides have **no candidate pair** (peak-amplitude
gate cut them), and the remaining 37 have candidate pairs that fail
on `pair_A < 0.30` (9 rides) or `joint_R² < 0.90` (~20) or both.

Closing this gap in a single-scalar threshold sweep appears infeasible:
lowering `min_peak_abs_a` further causes gt_split (iter_17: +89 mistakes,
86 of which are split cases). The durable path is **session-adaptive
noise floors** — estimate per-session `σ_a` from `outside` windows and
use it to scale `min_peak_abs_a` / `min_pair_abs_a` downward on clean
recordings while keeping them tight on noisy ones.

### Branches kept for follow-up

All per-iteration branches (`iter_00_baseline` … `iter_19_pair_a_26`)
are preserved locally. Each branch has its own
`algorithms/improvement_iterations/iter_NN_<slug>/` folder with
`metrics.json`, `per_gt.csv`, `mistakes/*.png`, `per_exp_summary.png`
and `notes.md`. Rerun the whole evaluation on any branch with:

```bash
venv/bin/python -m src.segmentation.algorithms.improvement_iterations._iter_runner \
    --iter NN --slug <slug> --kind all --what "<one-line description>"
```
