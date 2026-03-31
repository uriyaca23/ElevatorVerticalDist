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

---

## Using Pre-Computed Segments

If you already have detection/segmentation results (e.g. from your own algorithm or manual annotation), you can skip the detection stage and go directly to quality filtering + height estimation.

### Python API

```python
from src.pipeline import ElevatorHeightPipeline

pipeline = ElevatorHeightPipeline.load("model/")

# Define your segments as time intervals within the recording
segments = [
    {"start_time": 10.5, "end_time": 25.3},
    {"start_time": 45.0, "end_time": 62.1},
    {"start_time": 120.0, "end_time": 145.8},
]

# Process with pre-computed segments (skips detection entirely)
results = pipeline.process_segments(acc_x, acc_y, acc_z, segments, fs=100)

# Same output format as process()
for r in results:
    status = "✓" if r['accepted'] else "✗"
    ci = f" ± {r['confidence_interval_90']:.2f}m" if r['confidence_interval_90'] else ""
    print(f"  {status} [{r['start_time']:.1f}s – {r['end_time']:.1f}s]  "
          f"Height: {r['height_estimate']:+.2f}m{ci}")

# Visual output
results, fig = pipeline.process_segments_plot(
    acc_x, acc_y, acc_z, segments, fs=100,
    save_path="segments_output.png"
)
```

### CLI with Segments

```bash
# Create a segments JSON file
echo [{"start_time": 10.5, "end_time": 25.3}, {"start_time": 45.0, "end_time": 62.1}] > segments.json

# Run inference with pre-computed segments (skips detection)
python run_inference.py --input data.csv --segments segments.json --verbose

# Only accepted rides
python run_inference.py --input data.csv --segments segments.json --accepted-only -v
```

**Segments JSON format:**
```json
[
  {"start_time": 10.5, "end_time": 25.3},
  {"start_time": 45.0, "end_time": 62.1}
]
```

Each segment specifies a `start_time` and `end_time` in seconds within the accelerometer recording.

---

## Custom Dataset Evaluation

Evaluate the pipeline on your own tagged dataset with comprehensive figures, metrics, and quality analysis. This is designed for **test-set evaluation** — the pre-calibrated model is used as-is (no re-calibration on your data).

### Dataset CSV Format

Create a CSV index file with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `segment_id` | Yes | Unique integer ID for each ride segment |
| `acc_data_path` | Yes | Path to the accelerometer CSV file |
| `start_time` | Yes | Segment start time in seconds |
| `end_time` | Yes | Segment end time in seconds |
| `true_height` | Yes | Ground-truth height difference in meters (signed: positive=up, negative=down) |
| `phone_position` | No | Phone position during ride (e.g. `hand`, `pocket`) — used for analysis breakdowns |
| `fs` | No | Sampling rate in Hz (defaults to auto-detect or 100 Hz) |

**Example CSV:**
```csv
segment_id,acc_data_path,start_time,end_time,true_height,phone_position
1,data/recording1.csv,85.9,108.0,18.4,hand
2,data/recording1.csv,150.3,172.1,-12.6,hand
3,data/recording2.csv,22.0,45.5,6.1,pocket
```

**Notes:**
- Multiple segments can reference the same accelerometer file
- The `acc_data_path` must point to a CSV with accelerometer data. Supported formats:
  - Standard: `acc_x, acc_y, acc_z` columns (with optional `time` column)
  - Headerless 4-column: `time_ms, x, y, z` (auto-detected)
- `start_time` and `end_time` are in seconds from the start of the recording
- `true_height` is signed: positive = upward movement, negative = downward

### Evaluation Modes

| Mode | What it tests | When to use |
|------|--------------|-------------|
| `segments_only` | Quality filter + height estimation only | You have reliable segment boundaries and want to test estimation accuracy |
| `full` | Detection + segmentation + quality + estimation | You want to test the entire pipeline end-to-end |

### Running Evaluation

```bash
# Segments-only evaluation (recommended for testing estimation quality)
python run_custom_evaluation.py --dataset my_dataset.csv --mode segments_only -v

# Full pipeline evaluation (tests detection + estimation)
python run_custom_evaluation.py --dataset my_dataset.csv --mode full -v

# Custom output directory
python run_custom_evaluation.py --dataset my_dataset.csv --mode segments_only \
    --output-dir my_results/ -v
```

### Output Files

All outputs are saved to the `--output-dir` directory (default: `evaluation_output/`):

| File | Description |
|------|-------------|
| `results.json` | Per-ride detailed results including estimated height, error, method, accept/reject status, quality features |
| `summary.json` | Aggregate metrics: MAE, median error, acceptance rate, conformal coverage |

### Output Figures

| Figure | Description |
|--------|-------------|
| `fig_scatter.png` | True vs Estimated height scatter plot (accepted + rejected) |
| `fig_per_ride_errors.png` | Per-ride absolute error bar chart with segment IDs |
| `fig_error_histogram.png` | Error distribution histogram |
| `fig_error_cdf.png` | Cumulative error distribution (accepted vs all) |
| `fig_conformal_coverage.png` | Pre-calibrated conformal interval coverage check |
| `fig_quality_vs_error.png` | Quality score vs height error relationship |
| `fig_rejection_analysis.png` | Rejection reasons breakdown + error comparison |
| `fig_error_vs_height.png` | Error vs ride height magnitude |
| `fig_method_breakdown.png` | Estimation method usage and per-method accuracy |
| `fig_individual_rides.png` | Best and worst ride height curves |
| `fig_phone_positions.png` | Error by phone position (if applicable) |
| `fig_summary_dashboard.png` | 4-panel summary with key metrics table |
| `fig_detection_*.png` | Detection timeline (full mode only) |
| `fig_iou_distribution.png` | Detection IoU distribution (full mode only) |

### Reading the Results

**`summary.json`** contains the key metrics:
```json
{
  "n_total": 33,
  "n_accepted": 12,
  "n_rejected": 21,
  "acceptance_rate": 36.4,
  "accepted_mae": 0.996,
  "accepted_median": 0.579,
  "accepted_max_err": 3.881,
  "accepted_within_1m": 7,
  "conformal_interval": 3.62,
  "conformal_coverage": 91.7
}
```

Key things to check:
1. **`conformal_coverage`** ≥ 90%: The pre-calibrated conformal interval achieves the target coverage on your data
2. **`accepted_mae`**: Mean absolute error on accepted rides should be low (~1m or less)
3. **`acceptance_rate`**: What fraction of rides pass quality filtering
4. **Rejection quality** (in console output): Most rejected rides should genuinely have high error

**`results.json`** contains per-ride details:
```json
[
  {
    "segment_id": 1,
    "true_dh": 18.4,
    "est_dh": 16.56,
    "err": 1.84,
    "method": "gravity_proj",
    "accepted": true,
    "reject_reason": "",
    "quality_score": 1.23,
    "confidence_interval_90": 3.62
  }
]
```

### Example: Bar-Ilan Dataset

Generate and run the example evaluation dataset:

```bash
# Generate example CSV from Bar-Ilan dataset
python scripts/generate_example_eval_csv.py

# Run segments-only evaluation
python run_custom_evaluation.py --dataset datasets/bar_ilan_eval_example.csv \
    --mode segments_only --output-dir evaluation_output/ -v
```

---

## CLI Usage

```bash
# Run on a CSV file (columns: acc_x, acc_y, acc_z)
python run_inference.py --input data.csv --output results.json --verbose

# Only accepted rides
python run_inference.py --input data.csv --accepted-only -v

# With pre-computed segments
python run_inference.py --input data.csv --segments segments.json -v
```

## Input Format

CSV with columns (case-insensitive):
| Column | Description |
|--------|-------------|
| `acc_x` | X-axis acceleration (m/s²) |
| `acc_y` | Y-axis acceleration (m/s²) |
| `acc_z` | Z-axis acceleration (m/s²) |
| `time` (optional) | Timestamp in seconds |

Also supports headerless 4-column CSVs (`time_ms, x, y, z`).

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

# Run evaluation on custom dataset
python run_custom_evaluation.py --dataset my_data.csv --mode segments_only -v
```

## Project Structure

```
├── run_inference.py              # CLI inference entry point
├── run_evaluation.py             # Bar-Ilan evaluation pipeline with figures
├── run_custom_evaluation.py      # Custom dataset evaluation (test-set)
├── src/
│   ├── pipeline.py               # Production pipeline class
│   ├── algorithms/               # Core algorithms (quality_filter, ZUPT, etc.)
│   └── legacy/                   # Historical algorithm files
├── model/                        # Saved pipeline parameters
├── datasets/                     # ADVIO, Bar-Ilan, synthetic data
│   └── bar_ilan_eval_example.csv # Example evaluation dataset CSV
├── scripts/
│   └── generate_example_eval_csv.py  # Generate example from Bar-Ilan
├── docs/                         # Reports, figures
├── tests/                        # Test suite
└── prompt/                       # Project prompts history
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
