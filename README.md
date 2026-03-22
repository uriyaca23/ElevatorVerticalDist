# Elevator Vertical Distance & Confidence Analyzer

This repository contains the algorithms and dataset tools needed to evaluate ZUPT-based height estimations for elevator rides, including empirical Confidence Interval generation.

## Work Dataset Prep Workflow

To prepare for analyzing your real "work dataset", we have provided a toolset that trains an empirical conformal prediction model using 90% confidence bounding over a theoretical ZUPT integration error model.

### 1. Generating a Synthetic Dataset
To validate the capabilities or mock your real dataset format, use the generator to create `example/work_dataset/train` and `test` directories.
```powershell
python run_work_dataset_analysis.py generate --dataset_dir example/work_dataset
```

### 2. Training the Conformal Predictor
Runs ZUPT on the `train` dataset (which contains `gt_height_meters`), computes the theoretical confidence bounding, and dynamically calibrates the sigma margins (via conformal prediction) so that the empirical coverage over the training set reaches exactly 90%.
```powershell
python run_work_dataset_analysis.py train --dataset_dir example/work_dataset
```
_This will result in a `conformal_params.json` file being saved for subsequent inference._

### 3. Predicting with Confidence Bounding
Runs the analysis on the `test` dataset (or your actual unseen dataset). It assesses whether each sample should be accepted or rejected (due to excessive simulated shaking, impacts, or unrealistically long active windows) and outputs the height estimation ± the 90% Confidence Interval margin.
```powershell
python run_work_dataset_analysis.py predict --dataset_dir example/work_dataset
```

## Testing
Comprehensive pytest coverage exists under `tests/`.
```powershell
python -m pytest tests/test_zupt_confidence.py tests/test_dataset_generation.py
```
