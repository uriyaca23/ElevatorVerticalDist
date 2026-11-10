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

1. **Segmentation** (`pyramidElevatorDist/segmentation/`) — take a
   continuous sensor session and slice it into discrete ride intervals
   tagged `up` / `down` / `outside`. The active detector is a
   matched-filter over trapezoid pulse-pair templates (accelerometer-only).
   A pressure-filter fallback runs on phones with a barometer, mostly as a
   ground-truth source.
2. **Prediction** (`pyramidElevatorDist/prediction/`) — for each detected
   ride, predict the signed Δh in meters with a calibrated 90 % confidence
   interval. Three algorithms: barometric-height-difference (ISA
   inversion), ZUPT double-integration, and a trapezoid-pulse-pair
   accelerometer fit.

Both stages share the same shape: a Pydantic config selects one of several
algorithms, and a single dispatcher class (`Segmenter` / `Predictor`)
exposes the stage's public method (`.detect` / `.predict`). Reusable,
stage-agnostic helpers live in `pyramidElevatorDist/utils/`.

**Layout rule (one-way imports):** all algorithm code lives in the
pip-installable `pyramidElevatorDist/` package, which must never import
`src.*`. Everything in `src/` (data I/O, pipeline apps, evaluation
harnesses in `src/evaluation/`, research tooling in `src/segmentation/`)
imports FROM the package. The wheel is built from the root
`pyproject.toml` (`python -m build`); `scripts/package_smoke_test.py`
verifies self-containment in a clean venv.

Ground truth comes from the barometer when the phone has one (PRS sensor →
ISA altitude → labelled ride intervals). Experiments without a barometer
use hand-labelled intervals edited through `src/data/gt_editor.py`.

## Setup

```bash
# Python 3.10+ is assumed (the code uses `|` union syntax and `dataclass`
# keyword-only features).
python -m venv venv
source venv/bin/activate
pip install -r requirements-ui.txt   # full dev env (UIs, plotting, tooling)
```

The venv is already present as `venv/` in the repo root; if it's stale,
rebuild it with the commands above. `venv-old-apple-py39/` is an archived
Python 3.9 environment kept for reference — do not use it.

`requirements.txt` holds ONLY the package runtime deps (mirrors
`[project.dependencies]` in `pyproject.toml`): `numpy>=1.24`,
`pandas>=2.0`, `scipy>=1.10`, `pydantic>=2.0`. Everything else the repo
uses comes from `requirements-ui.txt`:

- `matplotlib>=3.7` — plotting (always `matplotlib.use("Agg")` in headless
  paths; see `src/evaluation/segmentation/plots.py` for the pattern).
  Deliberately NOT a package dependency — nothing in
  `pyramidElevatorDist/` imports it.
- `pydantic>=2.0` — config models in both `configTypes.py` modules and the
  public typed API. Use `model_copy(update={...})` to clone a config with
  overrides (this is what `sweep_hyperparameters` does).
- `scikit-learn>=1.3` — only used for small bits inside the pipelines.
- `openpyxl>=3.1` — reading/writing the legacy Excel GT caches.
- `python-docx>=1.0` — report generation.
- `pytest>=7.0` — test runner.

## How to run things

All commands assume `cwd = project root` and `source venv/bin/activate`.

| Task | Command |
|---|---|
| Load experiments (CLI inspection) | `python -m src.data.loader` |
| Run segmentation evaluation | `python -m src.evaluation.segmentation --algorithm pressure_filter --out-dir elevator_reports/seg_eval` |
| Sweep segmentation hyperparams | `python -m src.evaluation.segmentation --sweep grid.json --out-csv sweep.csv` |
| Run active template-match detector | `python -m src.segmentation.run_detector --only <exp>` |
| End-to-end pipeline | `python -m src.pipelines.boutique_pipeline` |
| Prediction evaluation | `python -m src.evaluation.prediction.evaluateOnData` |
| Run the test suite | `pytest` (tests/package/ runs without the data corpus) |
| Build + clean-venv verify the wheel | `python scripts/package_smoke_test.py` |

## House rules when editing this repo

- **Never import from `src/(archive)/`.** It's frozen; several paths it
  references (`src/algorithms/`, `src/pipeline`) no longer exist. Treat it
  as documentation, not code.
- **Never import `src.*` from inside `pyramidElevatorDist/`.** The package
  must stay self-contained (grep-gated; the smoke test asserts it). `src/`
  may import the package freely.
- **Never put reusable signal processing in a stage folder.** If something
  works on bare numpy arrays and has no dependency on prediction/segmentation
  types, it belongs in `pyramidElevatorDist/utils/`.
- **Respect the dispatcher pattern.** New algorithms get a new enum value
  in the stage's `configTypes.py` + a new implementation module, wired
  through the `Segmenter` / `Predictor` class. Callers should not import
  algorithm classes directly unless they're writing stage-internal code.
- **Config files are `configTypes.py`, not `class.py`.** `class` is a
  reserved keyword and forced `importlib.import_module` hacks; the rename
  is intentional.
- **Algorithm-specific data types stay with the algorithm.** For example
  `PredictionOutput` stays in
  `pyramidElevatorDist/prediction/algorithms/common/types.py` because its
  schema is prediction-API contract. `ConformalCalibrator` does *not* —
  split conformal is general statistics and lives in
  `pyramidElevatorDist/utils/`.
- **The public wrappers are hard-typed.** `findSegments`/`predictSegment`
  et al. validate their DataFrame inputs against
  `pyramidElevatorDist.schemas` and return pydantic models from
  `pyramidElevatorDist.types` — never plain dicts or unvalidated frames.
  New wrapper outputs need a model + tests in `tests/package/`.
- **Do not write docs that aren't asked for.** This project already has
  several sub-READMEs and a LaTeX report under `docs/`. Don't add new
  markdown files unless the user explicitly asks.
- **`pre` and `post` on `Predictor.predict` are stationary windows around
  a ride** (not padding). They are used to calibrate the phone's gravity
  vector; without them the accelerometer algorithms fall back to noisy
  magnitude-based estimates. Never pass ride samples there.
