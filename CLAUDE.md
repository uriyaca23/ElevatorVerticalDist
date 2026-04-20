# CLAUDE.md — Working notes for the assistant

## Read this first

The canonical codebase map is **[README.md](./README.md)** at the project
root. Consult it before suggesting any file-layout change or writing new
code. Sub-directories have their own READMEs (listed at the bottom of the
root README); prefer those when you need deep context on one stage.

## What this project is

This is a research codebase for **estimating how far an elevator travelled
vertically during a ride, using only the sensors on a passenger's phone**.
End-to-end it is two stages:

1. **Segmentation** (`src/segmentation/`) — take a continuous sensor
   session and slice it into discrete ride intervals tagged `up` / `down` /
   `outside`. The active detector is a matched-filter over trapezoid
   pulse-pair templates (accelerometer-only). A pressure-filter fallback
   runs on phones with a barometer, mostly as a ground-truth source.
2. **Prediction** (`src/prediction/`) — for each detected ride, predict the
   signed Δh in meters with a calibrated 90 % confidence interval. Three
   algorithms: barometric-height-difference (ISA inversion), ZUPT
   double-integration, and a trapezoid-pulse-pair accelerometer fit.

Both stages share the same shape: a Pydantic config selects one of several
algorithms, and a single dispatcher class (`Segmenter` / `Predictor`)
exposes the stage's public method (`.detect` / `.predict`). Reusable,
stage-agnostic helpers live in `src/utils/`.

Ground truth comes from the barometer when the phone has one (PRS sensor →
ISA altitude → labelled ride intervals). Experiments without a barometer
use hand-labelled intervals edited through `src/data/gt_editor.py`.

## Setup

```bash
# Python 3.10+ is assumed (the code uses `|` union syntax and `dataclass`
# keyword-only features).
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The venv is already present as `venv/` in the repo root; if it's stale,
rebuild it with the commands above. `venv-old-apple-py39/` is an archived
Python 3.9 environment kept for reference — do not use it.

Core dependencies (`requirements.txt`):

- `numpy>=1.24`, `pandas>=2.0`, `scipy>=1.10` — the numerics stack.
- `matplotlib>=3.7` — plotting (always `matplotlib.use("Agg")` in headless
  paths; see `src/segmentation/evaluate/plots.py` for the pattern).
- `pydantic>=2.0` — config models in both `configTypes.py` modules. Use
  `model_copy(update={...})` to clone a config with overrides (this is
  what `sweep_hyperparameters` does).
- `scikit-learn>=1.3` — only used for small bits inside the pipelines.
- `openpyxl>=3.1` — reading/writing the legacy Excel GT caches.
- `python-docx>=1.0` — report generation.
- `pytest>=7.0` — test runner.

## How to run things

All commands assume `cwd = project root` and `source venv/bin/activate`.

| Task | Command |
|---|---|
| Load experiments (CLI inspection) | `python -m src.data.loader` |
| Run segmentation evaluation | `python -m src.segmentation.evaluate --algorithm pressure_filter --out-dir elevator_reports/seg_eval` |
| Sweep segmentation hyperparams | `python -m src.segmentation.evaluate --sweep grid.json --out-csv sweep.csv` |
| Run active template-match detector | `python src/segmentation/algorithms/accelerometer_only/template_match/check_grid_across_signal/detect.py --only <exp>` |
| End-to-end pipeline | `python -m src.pipelines.boutique_pipeline` |
| Prediction evaluation | `python scripts/run_prediction_evaluation.py` |

## House rules when editing this repo

- **Never import from `src/(archive)/`.** It's frozen; several paths it
  references (`src/algorithms/`, `src/pipeline`) no longer exist. Treat it
  as documentation, not code.
- **Never put reusable signal processing in a stage folder.** If something
  works on bare numpy arrays and has no dependency on prediction/segmentation
  types, it belongs in `src/utils/`.
- **Respect the dispatcher pattern.** New algorithms get a new enum value
  in the stage's `configTypes.py` + a new implementation module, wired
  through the `Segmenter` / `Predictor` class. Callers should not import
  algorithm classes directly unless they're writing stage-internal code.
- **Config files are `configTypes.py`, not `class.py`.** `class` is a
  reserved keyword and forced `importlib.import_module` hacks; the rename
  is intentional.
- **Algorithm-specific data types stay with the algorithm.** For example
  `PredictionOutput` stays in `src/prediction/algorithms/common/types.py`
  because its schema is prediction-API contract. `ConformalCalibrator` does
  *not* — split conformal is general statistics and lives in `src/utils/`.
- **Do not write docs that aren't asked for.** This project already has
  several sub-READMEs and a LaTeX report under `docs/`. Don't add new
  markdown files unless the user explicitly asks.
- **`pre` and `post` on `Predictor.predict` are stationary windows around
  a ride** (not padding). They are used to calibrate the phone's gravity
  vector; without them the accelerometer algorithms fall back to noisy
  magnitude-based estimates. Never pass ride samples there.
