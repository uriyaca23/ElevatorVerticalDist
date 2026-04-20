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
