# Elevator Vertical Distance Estimator

Accelerometer-only elevator height estimation pipeline with conformal prediction intervals.

Three-stage pipeline: **Detection** → **Quality Filter** → **Height Estimation** with 90% conformal coverage.

## Quick Start

```python
from src.pipeline import ElevatorHeightPipeline

# Load pre-calibrated pipeline
pipeline = ElevatorHeightPipeline.load("model/")

# Process raw 3-axis accelerometer data (numpy arrays, m/s²)
results = pipeline.process(acc_x, acc_y, acc_z, fs=100)

# Print accepted rides with time frames
for r in results:
    if r['accepted']:
        ci = f" ± {r['confidence_interval_90']:.2f}m" if r['confidence_interval_90'] else ""
        print(f"  [{r['start_time']:.1f}s – {r['end_time']:.1f}s]  "
              f"Height: {r['height_estimate']:+.2f}m{ci}")
```

## Visual Output

Generate a figure showing all detected rides and their height estimates:

```python
# process_plot returns (results_dict, matplotlib_figure)
results, fig = pipeline.process_plot(acc_x, acc_y, acc_z, fs=100,
                                     save_path="output.png")
```

This produces a 2-panel figure:
- **Top**: Accelerometer magnitude with detected segments highlighted (green=accepted, red=rejected)
- **Bottom**: Height estimate bar chart with 90% confidence interval whiskers

## CLI Usage

```bash
# Run on a CSV file (columns: acc_x, acc_y, acc_z)
python run_inference.py --input data.csv --output results.json --verbose

# Only accepted rides
python run_inference.py --input data.csv --accepted-only -v
```

## Input Format

CSV with columns (case-insensitive):
| Column | Description |
|--------|-------------|
| `acc_x` | X-axis acceleration (m/s²) |
| `acc_y` | Y-axis acceleration (m/s²) |
| `acc_z` | Z-axis acceleration (m/s²) |
| `time` (optional) | Timestamp in seconds |

## Output Format

JSON array of detected rides:
```json
[
  {
    "start_time": 85.9,
    "end_time": 108.0,
    "height_estimate": 18.4,
    "confidence_interval_90": 3.62,
    "method": "gravity_proj",
    "accepted": true,
    "reject_reason": ""
  }
]
```

## Pipeline Stages

1. **Detection**: Finds elevator rides via accelerometer magnitude variance analysis and velocity zero-crossing segmentation.
2. **Quality Filter**: Accelerometer-only assessment of orientation stability, gravity drift, impact peaks, and noise. Rejects unreliable segments with clear reasons.
3. **Height Estimation**: Gravity-projected ZUPT integration with drift-corrected magnitude fallback. Three estimation methods with automatic selection based on signal quality.
4. **Conformal Prediction**: LOO conformal provides 90% coverage intervals calibrated on the Bar-Ilan dataset.

## Re-Calibration

```python
from src.pipeline import ElevatorHeightPipeline

pipeline = ElevatorHeightPipeline(fs=100)
stats = pipeline.calibrate(rides_with_gt=[
    {'acc_x': ax, 'acc_y': ay, 'acc_z': az, 'true_height': 6.0},
    # ... more labeled rides
])
pipeline.save("model/")
```


## Evaluation

```bash
# Run full evaluation pipeline on Bar-Ilan dataset with figures
python run_evaluation.py
```

## Project Structure

```
├── run_inference.py          # CLI inference entry point
├── run_evaluation.py         # Full evaluation pipeline with figures
├── src/
│   ├── pipeline.py           # Production pipeline class
│   ├── algorithms/           # Core algorithms (quality_filter, ZUPT, etc.)
│   └── legacy/               # Historical algorithm files
├── model/                    # Saved pipeline parameters
├── datasets/                 # ADVIO, Bar-Ilan, synthetic data
├── docs/                     # Reports, figures
├── tests/                    # Test suite
└── prompt/                   # Project prompts history
```

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `python-docx`

## Performance (Bar-Ilan Dataset)

| Metric | Value |
|--------|-------|
| Detection | 28/33 rides matched (85%) |
| Acceptance rate | 58% (19/33) |
| Accepted MAE | 1.16m |
| Accepted Median Error | 0.80m |
| LOO Conformal Coverage | 94.7% (≥90% target) |
| LOO Conformal Interval | ±3.98m |
