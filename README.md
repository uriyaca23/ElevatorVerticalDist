# ElevatorVerticalDist — Codebase Map

This document is the single source of truth for where things live in this
repository. It exists so that an LLM (or a human) dropped into any corner of
the codebase can orient quickly: what problem we're solving, which folder owns
which stage, and where to look for deeper context.

> **Project idea.** Estimate the vertical distance travelled by a passenger
> during an elevator ride using only the sensors on their phone
> (accelerometer always; barometer when available). The pipeline runs in two
> stages: **(1) segmentation** — slice a full sensor session into discrete
> `up` / `down` / `outside` ride intervals; **(2) prediction** — for each
> detected ride, predict the signed Δh (meters) with a calibrated 90 %
> confidence interval.

---

## Top-level layout

```
ElevatorVerticalDist/
├── pyramidElevatorDist/      # THE pip-installable algorithm library (self-contained)
├── src/                      # application layer: data I/O, pipelines/apps, evaluation
├── scripts/                  # one-shot runners (build reports, evaluations)
├── tests/                    # pytest suite (tests/package/ = synthetic, package-only)
├── docs/                     # LaTeX report + figures + reference PDFs
├── papers/                   # reference papers (elevator motion profiles, etc.)
├── pyproject.toml            # packaging config for pyramidElevatorDist (v0.2.0)
├── requirements.txt          # package runtime deps (mirrors pyproject)
├── requirements-ui.txt       # full dev environment (UIs, reports, tooling)
├── README.md                 # ← you are here
└── CLAUDE.md                 # LLM onboarding notes (project idea + setup)
```

**The dependency rule of this repo (one-way):** everything algorithmic lives
in `pyramidElevatorDist/` and never imports `src.*`; everything in `src/`
(data loading, UIs, evaluation harnesses, research tooling) imports FROM the
package. `python -m build` produces a distributable wheel from the root
`pyproject.toml`; `scripts/package_smoke_test.py` proves self-containment in
a clean venv.

---

## `pyramidElevatorDist/` — the algorithm library

```
pyramidElevatorDist/
├── __init__.py            # public API (functions + typed models + schemas + exceptions)
├── segmentor.py           # findSegments / findSegmentParameters / findSegmentsDetailed
├── prediction/            # predictSegment / predictByParameters (wrapper = __init__.py)
│   └── algorithms/        # Predictor dispatcher, configTypes + config.json,
│                          #   barometer_only/, zupt_accel/ + calibration.json,
│                          #   trapezoid_accel/ + calibration.json, common/types.py
├── predection.py          # deprecated alias of .prediction (historical misspelling)
├── signal.py              # reconstructedSignal / displaySeries / barometricAltitude
├── _orchestration.py      # shared segment+predict core (ex src/pipelines/inprocess.py)
├── exceptions.py          # PyramidElevatorDistError taxonomy (actionable messages)
├── schemas.py             # FrameSchema + ACC/GYRO/PRS/… DataFrame schemas
├── _validation.py         # wrapper-entry validation helpers
├── types/                 # pydantic models: RideSegment, SegmentDetail, SegmentSpec,
│                          #   TrapezoidParams, PredictionRow, PredictionResult, …
├── utils/                 # pure numpy/scipy helpers (accelerometer_utils, conformal,
│                          #   sensor_noise, signal_processing, trapezoid_template,
│                          #   trapezoid_fast, resampling)
├── physics/               # barometric.py (ISA inversion) + reconstruct_az/ (accel+gyro
│                          #   world-frame a_z filters: Complementary, Mahony, Madgwick,
│                          #   Valenti, ESKF)
└── segmentation/
    └── algorithms/        # Segmenter dispatcher, configTypes + config.json, metrics/,
                           #   barometer_only/, template_match core (templates, matcher,
                           #   check_grid_across_signal/, pure fit_elevator_parameters
                           #   primitives)
```

The wrappers accept schema-validated DataFrames and return **typed pydantic
models — never plain dicts**. Malformed input raises a specific
`pyramidElevatorDist.exceptions` subclass whose message names the function,
the parameter, the problem, and the fix. The per-algorithm `config.json` and
committed conformal `calibration.json` files travel inside the wheel and are
resolved via `Path(__file__).with_name(...)`.

---

## `src/` — the application layer

```
src/
├── data/              # sensor I/O, ground truth, dataset cleanup, GT editor
├── evaluation/
│   ├── segmentation/  # stage-1 evaluation harness (ex src/segmentation/evaluate)
│   └── prediction/    # stage-2 evaluation & reporting (ex src/prediction/evaluation)
├── segmentation/      # research tooling only: offline template fitters,
│   │                  #   label/plot scripts, labels/ data, run_detector.py CLI
├── pipelines/         # end-to-end apps (streamlit wizard, Tk prediction editor,
│                      #   boutique pipeline, pipeline-level evaluate/)
├── plotting/          # shared plotting helpers (experiment overviews)
└── (archive)/         # old code kept for reference; not imported by anything
```

The algorithm stages themselves (`Segmenter`, `Predictor`, `utils`,
`physics`) live in the `pyramidElevatorDist` package — see the section
above. `src/` imports them from the package.

### `src/data/` — sensor I/O and ground truth

> See `src/data/README.md` for the full schema.

- **`rawData/`** — raw per-experiment sensor logs (input only, never mutated).
- **`structuredData/`** — processed CSV artifacts (one folder per experiment;
  per-sensor CSVs + `gt.csv` + `metadata.csv`).
- **`loader/`** — the public loader package. `getExperimentData(name)` and
  `list_experiments(kind=...)` are the two entry points every downstream
  stage calls. `pipeline.py` is the modern CSV flow; `legacy.py` is the old
  Excel-cached flow retained for a few callers.
- **`gramushka/`** — barometer-derived calibration reference data.
- **`dataset_cleanup/`** — one-shot scripts for curating the dataset
  (time-calibration, noise tagging, residual calibration, etc.). Read their
  docstrings; they're not part of the runtime path.
- **`gt_editor.py`** — Tkinter GUI for hand-editing `gt.csv`.

### `pyramidElevatorDist/physics/`

- **`barometric.py`** — ISA pressure → altitude inversion (`pressure_to_altitude`).
  Used by the barometer-only segmenter and predictor.
- **`reconstruct_az/`** — accel+gyro orientation-fusion filters producing the
  world-frame vertical acceleration (`reconstruct_az`, `RECONSTRUCT_CHOICES`).

### `pyramidElevatorDist/utils/` — reusable, stage-agnostic helpers

Plain-numpy utilities that segmentation, prediction, and dataset-cleanup
share. If it has no dependency on either stage's data types, it lives here.

- **`accelerometer_utils.py`** — gravity estimation
  (`estimate_gravity_stationary`), vertical-accel projection
  (`vertical_accel_projected`, `vertical_accel_magnitude`), convenience
  `compute_a_vert`, `compute_velocity`, ZUPT double-integration
  (`zupt_integrate`), and a ride-band low-pass (`lowpass`, cutoff 0.3 Hz).
- **`signal_processing.py`** — general-purpose `butter_lowpass` (filtfilt,
  order-2, 3 Hz default).
- **`conformal.py`** — `ConformalCalibrator`: split-conformal multiplier on
  `|err|/σ` scores; used by the accelerometer predictors to calibrate their
  theoretical σ into a 90 %-coverage CI.
- **`sensor_noise.py`** — phone-model → accelerometer-chip noise σ
  (`get_phone_accel_noise_sigma`, `resolve_phone_to_chip`).
- **`trapezoid_template.py` / `trapezoid_fast.py`** — trapezoid kernel +
  matched-filter primitives (single-template and FFT-batched bank).
- **`resampling.py`** — gap-aware 50 Hz resampler
  (`resample_sensor_with_gaps`; `src.data.loader` re-exports it).

### Segmentation — stage 1

> See `src/segmentation/README.md` for the physics of elevator motion and
> why we use a trapezoid pulse-pair matched filter.

Algorithm core — `pyramidElevatorDist/segmentation/algorithms/`:

```
segmentation/algorithms/
├── configTypes.py        # Pydantic configs: SEGMENT_ALGORITHM_CONFIG,
│                         # SegmentAlgorithm enum, PressureFilterConfig,
│                         # TemplateMatchConfig
├── config.json           # per-algorithm hyperparameters loaded by configTypes
├── segmenter.py          # public Segmenter class + .detect(data) dispatcher
├── metrics/              # IntervalPredictionMetrics, SegmentationMetrics
├── barometer_only/
│   └── height_segmentation.py    # HeightSegmenter (pressure-filter)
└── accelerometer_only/
    └── template_match/           # the trapezoid pulse-pair detector
        ├── templates.py          # per-experimenter template fit
        ├── matcher.py            # sliding-NCC detector (legacy entry point)
        ├── fit_elevator_parameters/common.py  # pure fit primitives
        └── check_grid_across_signal/
            ├── detect.py         # stage 1–4: R² + |A| peak-pick,
            │                     # same-sign NMS — THIS is the active
            │                     # detector. Accepts optional
            │                     # phone_model for chip-spec-aware
            │                     # amplitude floors.
            └── pair_filter.py    # stage 5–6: shared-shape joint fit,
                                  # greedy pair resolver
```

Application side: `src/evaluation/segmentation/` (evaluator, plots,
`python -m src.evaluation.segmentation ...` CLI) and `src/segmentation/`
(offline template-parameter fitters, label building/plot scripts, the
`labels/` data tree, and `run_detector.py` — the detector sanity CLI).

**Public API**: `Segmenter(config).detect(data)` → DataFrame with
`start_ci`, `end_ci`, `duration`, `type`, `probability_ci` (exit-validated
against `SEGMENTS_TABLE_SCHEMA`). The `data` schema depends on the
algorithm (see the docstring on `Segmenter.detect`).

### Prediction — stage 2

Algorithm core — `pyramidElevatorDist/prediction/algorithms/`:

```
prediction/algorithms/
├── configTypes.py        # PREDICT_ALGORITHM_CONFIG, PredictAlgorithm,
│                         # BarometerHeightDiffConfig, ZuptAccelConfig,
│                         # TrapezoidAccelConfig
├── config.json           # per-algorithm hyperparameters
├── predictor.py          # public Predictor class + .predict(data, pre,
│                         # post, phone_model) dispatcher
├── common/
│   └── types.py          # PredictionOutput, CalibrationSample
├── barometer_only/
│   └── height_difference.py
└── accelerometer_only/
    ├── zupt_accel/
    │   ├── estimator.py     # ZuptAccelEstimator (+ calibration.json)
    │   ├── quality.py       # quality filter (gravity drift, peaks, ...)
    │   └── theoretical_ci.py  # σ_pos = σ_a · dt² · √(N³/12) noise model
    └── trapezoid_accel/
        ├── estimator.py     # TrapezoidAccelEstimator (+ calibration.json)
        ├── pulse_pair.py    # shared-shape trapezoid pulse-pair fitter
        └── quality.py
```

Application side: `src/evaluation/prediction/` — per-experiment inference
loop (`runner.py`), GT dataset iterator (`dataset.py`), metrics, figures,
report assembly, and the `python -m src.evaluation.prediction.evaluateOnData`
CLI (with `--calibrate` to refresh the committed conformal calibrations).

**Public API**: `Predictor(config).predict(data, pre, post, phone_model)` →
`PredictionOutput` with `height_diff`, `ci_half_width`, `theoretical_sigma`,
`accepted`, `quality_score`, `reject_reason`, `meta`. Accelerometer
algorithms need `pre`/`post` (stationary windows around the ride) for
gravity calibration; the barometer algorithm ignores them.

### `src/pipelines/`

- **`boutique_pipeline.py`** — end-to-end: loads an experiment → runs
  segmentation → runs prediction per detected ride → writes diagnostic
  figures. Use this as the integration-test entry point.
- **`prediction_editor/`** — presentation-only Tk/matplotlib diagnostic UI
  (one module per screen area). Loads raw experiment data and delegates all
  computation to the `pyramidElevatorDist` package. Run with
  `python -m src.pipelines.prediction_editor [exp]`.

### `src/plotting/`

Shared plotting helpers (experiment-overview figures, etc). Stage-specific
plots live inside each stage's `evaluate/` or `evaluation/` module.

### `src/(archive)/`

Pre-refactor code kept for reference. **Nothing in `src/` imports from
here.** Scripts in this folder reference paths like `src/algorithms/` that
no longer exist.

---

## `scripts/` — top-level runners

- **`package_smoke_test.py`** — build the wheel and prove self-containment
  in a fresh venv (all entry points on synthetic data, no `src` leak).
- **`build_prediction_report_assets.py`** — build figures for the LaTeX
  report under `docs/latex/`.

(The full prediction evaluation is
`python -m src.evaluation.prediction.evaluateOnData`.)

---

## Conventions worth knowing

### Config modules are named `configTypes.py`

`segmentation/algorithms/configTypes.py` and
`prediction/algorithms/configTypes.py` hold the Pydantic config models
(was previously `class.py`, renamed because `class` is a reserved keyword
and forced `importlib.import_module` hacks). Each stage has one top-level
config (`SEGMENT_ALGORITHM_CONFIG` / `PREDICT_ALGORITHM_CONFIG`) that
selects an algorithm enum and holds hyperparameter overrides merged on top
of `config.json`.

### Dispatcher pattern

Both stages have a single public dispatcher class:
- `pyramidElevatorDist/segmentation/algorithms/segmenter.py` → `Segmenter.detect(data)`
- `pyramidElevatorDist/prediction/algorithms/predictor.py` → `Predictor.predict(data, ...)`

The dispatcher reads its config's `algorithm` enum and builds the right
algorithm implementation. Callers never instantiate algorithm classes
directly.

### Algorithm-specific vs. reusable

Inside each stage, algorithms are grouped by the sensor modality they
depend on: `barometer_only/` for pressure-based algorithms,
`accelerometer_only/` for accelerometer-based ones. Anything that *isn't*
specific to one algorithm — signal processing, gravity math, conformal
calibration, phone-chip noise — lives in `pyramidElevatorDist/utils/`.

### One-way imports

`pyramidElevatorDist` never imports `src.*` — enforced by the grep gate in
the verification checklist and by `scripts/package_smoke_test.py`. Code in
`src/` freely imports the package.

### Outputs are CI-valued

Segmentation emits `(lo, hi)` tuples for `start_ci`, `end_ci`,
`probability_ci`. Prediction emits a scalar `height_diff` plus a scalar
`ci_half_width`. When the algorithm is deterministic, the CIs collapse to
zero-width tuples — downstream code must not assume the interval is
non-empty.

---

## Entry points cheat-sheet

| Need to… | Use |
|---|---|
| Detect rides / predict Δh (public API) | `pyramidElevatorDist.findSegments(acc)` / `predictSegment(acc, segment)` |
| Load an experiment | `src.data.loader.getExperimentData(name)` |
| List experiments | `src.data.loader.list_experiments(kind='train'/'test'/'all')` |
| Segment a session (dispatcher) | `pyramidElevatorDist.segmentation.algorithms.Segmenter(config).detect(data)` |
| Predict Δh of a ride (dispatcher) | `pyramidElevatorDist.prediction.algorithms.Predictor(config).predict(data, pre, post, phone_model)` |
| Sweep segmentation hyperparams | `src.evaluation.segmentation.sweep_hyperparameters(...)` |
| Evaluate a segmentation config | `src.evaluation.segmentation.evaluate_algorithm(...)` |
| Run the prediction evaluation | `python -m src.evaluation.prediction.evaluateOnData` |
| Run the detector sanity CLI | `python -m src.segmentation.run_detector --only <exp>` |
| Run the end-to-end pipeline | `src.pipelines.boutique_pipeline` |
| Build + verify the wheel | `python scripts/package_smoke_test.py` |

---

## Where to look for deeper docs

- `src/segmentation/README.md` — physics of elevator motion, why trapezoids,
  and the full segmentation problem statement.
- `src/segmentation/algorithms/accelerometer_only/template_match/README.md`
  — the template-match detector internals.
- `pyramidElevatorDist/segmentation/algorithms/accelerometer_only/template_match/check_grid_across_signal/README.md`
  — the active detector's state-dict schema.
- `src/data/README.md` — data schemas, folder layout, loader conventions.
- `src/data/dataset_cleanup/README.md` — dataset curation workflow.
- `pyramidElevatorDist/segmentation/algorithms/metrics/METRICS.md` — metric definitions.

---

## Triangle vs Trapezoid rides — the short-ride extension

The shared-shape pair-fit described above covers long rides, where the
cabin reaches `v_max` and the acceleration trace has two clearly
separated lobes. Short rides (typically `|Δh| ≲ 3 m`) never reach
`v_max`, so the velocity profile is triangular and the two acceleration
lobes **touch** at the peak-velocity instant. Fitting those rides with
the unconstrained pair-fit is ill-posed (the centre-to-centre spacing
`Δt_c` becomes degenerate at `Δt_c = 2W`), which historically showed up
as either silent under-coverage in the 0-3 m bin or a blanket
`lobes_overlapping` rejection that threw away legitimate data.

### The fix: dual fit with a joined-pulse branch

For every segment we run both of

1. the **pair fit** (`fit_shared_shape_pair`, 5 free parameters,
   `Δt_c` free),
2. the **joined-pulse fit** (`fit_joined_pulse`, 4 free parameters,
   `Δt_c = 2W` forced, bipolar template of width `4W`).

and pick between them on joint R² with an Occam tiebreak in the
overlap regime (`Δt_c / 2W < 1.15`). The joined regime has a
constraint-aware height formula and σ — see
[`trapezoid_accel/pulse_pair.py`](pyramidElevatorDist/prediction/algorithms/accelerometer_only/trapezoid_accel/pulse_pair.py):

```
Δh = 2·s·A·W²·(1+f)

σ_Δh² = (2W²(1+f))²·σ_A² + (4AW(1+f))²·σ_W² + (2AW²)²·σ_f²

σ_A² = σ_a_eff² / ⟨τ_joined, τ_joined⟩   where ⟨τ_joined, τ_joined⟩ = 2·⟨τ, τ⟩
```

The time-of-arrival parameter `t_mid` drops out of the σ sum because
`Δh` is shift-invariant in the joined model. No drift multiplier, no
`k_rel|Δh|` term — same design decisions as the unconstrained pair
σ model (see the "Why the trapezoid CI is not the ZUPT CI"
subsection in the paper).

When both fits end up with R² below `zupt_fallback_r2 = 0.20`, the
estimator falls back to an inline ZUPT double-integration
displacement with a `zupt_fallback_both_fits_failed` reject label —
the segment is recorded but excluded from conformal calibration.

Result on the 0-3 m bin: from 7 → **23 accepted segments at 100 %
coverage** on train (18 of them fit via the joined branch). Test-set
median CI half-width on filter-accepted segments drops from ±3.14 m
to **±1.16 m**. See
[`docs/latex/figures/triangle_vs_trapezoid.png`](docs/latex/figures/triangle_vs_trapezoid.png)
for a juxtaposed pair of real rides; the per-segment dumps live in
[`docs/examples/triangles/`](docs/examples/triangles/) and
[`docs/examples/trapezoids/`](docs/examples/trapezoids/).

### Generalising the distinction

**Detection.** The whole-signal detector currently scans a single-lobe
template bank and uses a pair-filter to assemble rides from detected
lobes. A short-ride produces either zero or two noisy lobe hits that
the pair-filter has to rescue. Extending the bank with a second
family — the joined bipolar template
`τ_{W,f}(t-W) − τ_{W,f}(t+W)` (already shipping as `joined_kernel`
in `pulse_pair.py`) — lets the detector emit each short ride as a
single candidate that doesn't need pair filtering. The two banks
share the `(W,f)` grid, so runtime cost is one extra convolution
per template pair; expected recall gain on short rides is significant.

**Segmentation.** On sessions without a barometer, the segmenter has
no notion of ride shape today. Running both a pair-fit score and a
joined-fit score per candidate segment, and labelling each segment
with its regime, gives

* a physically-grounded duration floor `T_ride ≥ 4·W_min_physical`
  for triangle rides (replacing the hand-tuned `min_duration_sec`),
* a binary regime label that the prediction stage consumes directly
  instead of re-deciding.

The two-stage pipeline becomes regime-aware end-to-end rather than
only at the prediction stage.

All the code, derivations, evaluation numbers, and visual examples
live alongside the main pipeline — no separate extension module.
