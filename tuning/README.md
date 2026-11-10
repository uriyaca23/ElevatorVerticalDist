# Hyperparameter tuning (`scripts/tune_hyperparameters.py`)

## TL;DR — tuning on new data

1. Load + label the data (barometer GT auto, else `python -m src.data.gt_editor`); check with `python -m src.data.loader`.
2. `python scripts/tune_hyperparameters.py --stage both` (add `--dry-run` to preview).
3. Done — the API and all three `evaluateOnData` use the new params automatically.
4. Calibration CI: prediction eval refits itself; for API/pipeline point them at `tuning/calibration_<algo>.json`.

---

This folder holds everything the hyperparameter tuner reads and writes. The
tuner finds the best detector / predictor hyperparameters and **writes them
into the project's live `config.json` files**, which every stage reads at call
time — so once you tune, the API and all three `evaluateOnData` scripts use the
new params automatically, with no code change, rebuild, or restart.

---

## What it does

`scripts/tune_hyperparameters.py`:

1. **Sweeps a grid** of hyperparameters (edit `tuning/grid_segmentation.json` /
   `tuning/grid_prediction.json`; defaults are written on first run).
2. **Scores each candidate** with **building-grouped k-fold cross-validation**
   on **clear** ride segments only — a config is never validated on a building
   it was scored against (mirrors the cross-building test we care about).
3. **Writes the winners** into the live config files (timestamped backup +
   printed diff):
   - `src/segmentation/algorithms/config.json` — key `acc_template_match`
   - `src/prediction/algorithms/config.json`   — keys `zupt_accel`, `trapezoid_accel`
4. **Refits conformal calibration** for the winning prediction configs to
   `tuning/calibration_<algo>.json` (see the calibration caveat below).
5. At the **start of the run**, writes one **QC plot per experiment** to
   `tuning/segment_qc/<exp>.png` (|a|−g with GT ride bands) so you can confirm
   the timing/alignment before trusting the scores.

Ranked results are written to `tuning/results_segmentation.csv` and
`tuning/results_prediction_<algo>.csv`.

> A winner is only persisted if it actually beats the current defaults
> (segmentation: higher f1-like) / meets the feasibility floors (prediction:
> coverage ≥ `--min-coverage`, accept-rate ≥ `--min-accept-rate`). Otherwise
> `config.json` is left untouched and the run says so.

---

## How to run

```bash
source venv/bin/activate

# preview only — rank + print the diffs, but DON'T modify config.json:
python scripts/tune_hyperparameters.py --stage both --dry-run

# tune both stages and write the winners into config.json (backup + diff):
python scripts/tune_hyperparameters.py --stage both

# one stage only:
python scripts/tune_hyperparameters.py --stage segmentation
python scripts/tune_hyperparameters.py --stage prediction

# skip the per-experiment QC plots:
python scripts/tune_hyperparameters.py --stage both --no-qc-plots
```

Useful knobs: `--folds N`, `--grid-seg/--grid-pred <path>`, `--min-coverage`,
`--min-accept-rate`, `--max-configs N`, `--include-corrupted`, `--out-dir`.

---

## Does it affect the API functions? **Yes.**

The tuned params live in `config.json`, which is **the single source of truth**.
Every stage loads it via `load_params()` *on each call*, so the change takes
effect immediately on the next call — nothing is cached at import time, and the
installed `pyramidElevatorDist` package reads the same files.

| Consumer | Reads `config.json`? | Auto-uses tuned params? |
|---|---|---|
| `pyramidElevatorDist.findSegments` / `findSegmentParameters` | seg → `_segment_cfg()` | ✅ segmentation params |
| `pyramidElevatorDist.predictSegment` / `predictByParameters` | pred → `Predictor.load_params()` | ✅ prediction params |
| `src.segmentation.evaluate.evaluateOnData` | seg | ✅ |
| `src.prediction.evaluation.evaluateOnData` | pred | ✅ |
| `src.pipelines.evaluate.evaluateOnData` | seg + pred | ✅ |
| Boutique / Streamlit app (`ui/api_client`) | seg + pred | ✅ |

So the intended workflow works exactly as expected:

1. Run the tuner → it writes the best params into `config.json`.
2. The `pyramidElevatorDist` API **instantly** starts predicting with the new
   params (next call).
3. Later, run any of the three `evaluateOnData` scripts → they also use the new
   params, so you can measure the improvement.

> Before this was wired up, the API/boutique segmentation read **hardcoded
> detector defaults** and ignored `config.json`. That gap is now closed — the
> API segmentation goes through `_segment_cfg()`, which reads `config.json`
> (same as the `Segmenter` the evaluators use).

---

## Caveats (read these)

1. **Conformal CI calibration is a separate artifact from the hyperparameters.**
   - The tuner refits a calibrator to `tuning/calibration_<algo>.json`.
   - The **`alpha`** knob (which scales the CI) **is** a tuned hyperparameter and
     **does** propagate via `config.json` — so point predictions, accept/reject,
     and the alpha-scaled CI all update everywhere.
   - But the **refit calibrator file itself is not auto-loaded** by the API
     (`pyramidElevatorDist` uses the built-in conformal default) or by the
     pipeline evaluator (which loads a fixed checkpoint at
     `src/data/.../test_results/prediction/train/calibration_trapezoid.json`
     unless you pass `--calibration-path`). The **prediction** `evaluateOnData`
     **refits its own** calibration each run, so it is automatically consistent.
   - If you want the API / pipeline to use the tuner's refit CI, point them at
     `tuning/calibration_<algo>.json` (or copy it to their expected path).

2. **Segmentation input-signal policy differs across the project.** The tuner
   and the **segmentation/pipeline** `evaluateOnData` score the matched filter on
   `a_vert` (the `config.json` default), while the **API + boutique** score on
   `a_mag_minus_g`. The tuned *threshold* params propagate to both, but they were
   tuned against `a_vert`. If you want them tuned for the deployed API signal,
   that's a follow-up (unify the `input_signal` first).

3. **Backups + dry-run.** Every write first copies `config.json` to
   `config.json.bak-<timestamp>`. Use `--dry-run` to see the diff without
   touching anything.

---

## Files in this folder

| File | What it is |
|---|---|
| `grid_segmentation.json` / `grid_prediction.json` | the editable search grids |
| `results_segmentation.csv` / `results_prediction_<algo>.csv` | every candidate ranked |
| `calibration_<algo>.json` | refit conformal calibration for the winners |
| `segment_qc/<exp>.png` | per-experiment |a|−g + GT timing QC plots |
