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
├── src/                      # all library code
├── scripts/                  # one-shot runners (build reports, evaluations)
├── docs/                     # LaTeX report + figures
├── metadata/                 # experiment-level metadata artefacts
├── papers/                   # reference papers (elevator motion profiles, etc.)
├── mistake/                  # scratch / dumps (ignore)
├── requirements.txt
├── README.md                 # ← you are here
└── CLAUDE.md                 # LLM onboarding notes (project idea + setup)
```

Everything that ships is under `src/`. Scripts import from `src/`; nothing in
the root is part of the library itself.

---

## `src/` — the library

```
src/
├── data/              # sensor I/O, ground truth, dataset cleanup, GT editor
├── physics/           # pressure → altitude (ISA inversion)
├── utils/             # reusable helpers (accelerometer, conformal, chip noise)
├── segmentation/      # stage 1 — detect elevator rides in a session
├── prediction/        # stage 2 — Δh per ride, with CI
├── pipelines/         # end-to-end orchestration across both stages
├── plotting/          # shared plotting helpers (experiment overviews)
└── (archive)/         # old code kept for reference; not imported by anything
```

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

### `src/physics/`

- **`barometric.py`** — ISA pressure → altitude inversion (`pressure_to_altitude`).
  Used by the barometer-only segmenter and predictor.

### `src/utils/` — reusable, stage-agnostic helpers

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

### `src/segmentation/` — stage 1

> See `src/segmentation/README.md` for the physics of elevator motion and
> why we use a trapezoid pulse-pair matched filter.

```
segmentation/
├── algorithms/
│   ├── configTypes.py        # Pydantic configs: SEGMENT_ALGORITHM_CONFIG,
│   │                         # SegmentAlgorithm enum, PressureFilterConfig,
│   │                         # TemplateMatchConfig
│   ├── config.json           # per-algorithm hyperparameters loaded by configTypes
│   ├── segmenter.py          # public Segmenter class + .detect(data)
│   │                         # dispatcher
│   ├── metrics/              # IntervalPredictionMetrics, SegmentationMetrics
│   ├── barometer_only/
│   │   └── height_segmentation.py    # HeightSegmenter (pressure-filter)
│   └── accelerometer_only/
│       └── template_match/           # the trapezoid pulse-pair detector
│           ├── templates.py          # per-experimenter template fit
│           ├── matcher.py            # sliding-NCC detector (legacy entry point)
│           ├── fit_elevator_parameters/  # offline template-parameter fitter
│           └── check_grid_across_signal/
│               ├── detect.py         # stage 1–4: R² + |A| peak-pick,
│               │                     # same-sign NMS — THIS is the active
│               │                     # detector. Accepts optional
│               │                     # phone_model for chip-spec-aware
│               │                     # amplitude floors.
│               ├── pair_filter.py    # stage 5–6: shared-shape joint fit,
│               │                     # greedy pair resolver
│               └── editor.py         # Tk/matplotlib diagnostic UI
└── evaluate/                 # generic, algorithm-agnostic evaluation harness
    ├── evaluator.py          # sweep_hyperparameters + evaluate_algorithm
    ├── plots.py              # CDFs (IoU, start/end residual, duration err)
    └── __main__.py           # python -m src.segmentation.evaluate ...
```

**Public API**: `Segmenter(config).detect(data)` → DataFrame with
`start_ci`, `end_ci`, `duration`, `type`, `probability_ci`. The `data`
schema depends on the algorithm (see the docstring on `Segmenter.detect`).

### `src/prediction/` — stage 2

```
prediction/
├── algorithms/
│   ├── configTypes.py        # PREDICT_ALGORITHM_CONFIG, PredictAlgorithm,
│   │                         # BarometerHeightDiffConfig, ZuptAccelConfig,
│   │                         # TrapezoidAccelConfig
│   ├── config.json           # per-algorithm hyperparameters
│   ├── predictor.py          # public Predictor class + .predict(data, pre,
│   │                         # post, phone_model) dispatcher
│   ├── common/
│   │   └── types.py          # PredictionOutput, CalibrationSample
│   ├── barometer_only/
│   │   └── height_difference.py
│   └── accelerometer_only/
│       ├── zupt_accel/
│       │   ├── estimator.py     # ZuptAccelEstimator
│       │   ├── quality.py       # quality filter (gravity drift, peaks, ...)
│       │   └── theoretical_ci.py  # σ_pos = σ_a · dt² · √(N³/12) noise model
│       └── trapezoid_accel/
│           ├── estimator.py     # TrapezoidAccelEstimator
│           ├── pulse_pair.py    # shared-shape trapezoid pulse-pair fitter
│           └── quality.py
└── evaluation/               # prediction-specific evaluation & reporting
    ├── runner.py             # per-experiment inference loop
    ├── dataset.py            # iterator over GT-intervals + sensors
    ├── metrics.py            # coverage, CI-width, abs-error statistics
    ├── figures.py            # reliability, coverage, error scatter
    └── report.py             # assembles a full evaluation report
```

**Public API**: `Predictor(config).predict(data, pre, post, phone_model)` →
`PredictionOutput` with `height_diff`, `ci_half_width`, `theoretical_sigma`,
`accepted`, `quality_score`, `reject_reason`, `meta`. Accelerometer
algorithms need `pre`/`post` (stationary windows around the ride) for
gravity calibration; the barometer algorithm ignores them.

### `src/pipelines/`

- **`boutique_pipeline.py`** — end-to-end: loads an experiment → runs
  segmentation → runs prediction per detected ride → writes diagnostic
  figures. Use this as the integration-test entry point.

### `src/plotting/`

Shared plotting helpers (experiment-overview figures, etc). Stage-specific
plots live inside each stage's `evaluate/` or `evaluation/` module.

### `src/(archive)/`

Pre-refactor code kept for reference. **Nothing in `src/` imports from
here.** Scripts in this folder reference paths like `src/algorithms/` that
no longer exist.

---

## `scripts/` — top-level runners

- **`run_prediction_evaluation.py`** — run the full prediction evaluation
  (trains conformal, runs on test split, writes report).
- **`build_prediction_report_assets.py`** — build figures for the LaTeX
  report under `docs/latex/`.

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
- `segmentation/algorithms/segmenter.py` → `Segmenter.detect(data)`
- `prediction/algorithms/predictor.py` → `Predictor.predict(data, ...)`

The dispatcher reads its config's `algorithm` enum and builds the right
algorithm implementation. Callers never instantiate algorithm classes
directly.

### Algorithm-specific vs. reusable

Inside each stage, algorithms are grouped by the sensor modality they
depend on: `barometer_only/` for pressure-based algorithms,
`accelerometer_only/` for accelerometer-based ones. Anything that *isn't*
specific to one algorithm — signal processing, gravity math, conformal
calibration, phone-chip noise — lives in `src/utils/`.

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
| Load an experiment | `src.data.loader.getExperimentData(name)` |
| List experiments | `src.data.loader.list_experiments(kind='train'/'test'/'all')` |
| Segment a session | `src.segmentation.algorithms.Segmenter(config).detect(data)` |
| Predict Δh of a ride | `src.prediction.algorithms.Predictor(config).predict(data, pre, post, phone_model)` |
| Sweep segmentation hyperparams | `src.segmentation.evaluate.sweep_hyperparameters(...)` |
| Evaluate a segmentation config | `src.segmentation.evaluate.evaluate_algorithm(...)` |
| Run the end-to-end pipeline | `src.pipelines.boutique_pipeline` |

---

## Where to look for deeper docs

- `src/segmentation/README.md` — physics of elevator motion, why trapezoids,
  and the full segmentation problem statement.
- `src/segmentation/algorithms/accelerometer_only/template_match/README.md`
  — the template-match detector internals.
- `src/segmentation/algorithms/accelerometer_only/template_match/check_grid_across_signal/README.md`
  — the active detector's state-dict schema.
- `src/data/README.md` — data schemas, folder layout, loader conventions.
- `src/data/dataset_cleanup/README.md` — dataset curation workflow.
- `src/segmentation/algorithms/metrics/METRICS.md` — metric definitions.
