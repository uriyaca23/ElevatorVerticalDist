# pyramidElevatorDist

Estimate how far an elevator travelled vertically during a ride, using only the
sensors on a passenger's phone. Two stages — **segmentation** (slice a continuous
accelerometer session into `up`/`down` ride intervals) and **prediction** (for each
ride, a signed Δh in metres with a calibrated 90 % confidence interval).

## Install

```bash
pip install pyramidElevatorDist
```

## Public API

Every function takes a pandas `DataFrame` with columns `timestamp_ms, x, y, z` (raw
accelerometer). Input may be at any (even variable) sample rate; by default each entry
point first normalizes onto a uniform 50 Hz grid (pass `resample=False` if already at
that cadence).

```python
import pandas as pd
from pyramidElevatorDist import (
    findSegments, findSegmentParameters, findSegmentsDetailed,
    predictSegment, predictByParameters,
    reconstructedSignal, barometricAltitude,
)

acc = pd.DataFrame(...)                      # timestamp_ms, x, y, z

segments = findSegments(acc)                 # -> list of {type, start_s, end_s}
result   = predictSegment(acc, segments[0])  # -> {"primary": "trap", "trap": {...}, "zupt": {...}}
disp     = reconstructedSignal(acc)          # -> DataFrame(timestamp_ms, a_vert, a_mag_g) for plotting
```

- `findSegments(acc)` — detect all ride segments.
- `findSegmentParameters(acc, start_s, end_s)` — fit the trapezoid pulse-pair for one interval.
- `predictSegment(acc, segment)` — predict Δh (+ CI) for one ride.
- `predictByParameters(acc, segment, params)` — predict Δh with a manually edited trapezoid.
- `reconstructedSignal(acc, gyro=None, reconstruct="none")` — whole-trace vertical accel
  for display (optionally gyro orientation-reconstructed).
- `barometricAltitude(prs)` — whole-trace barometric altitude (ground-truth reference).

Configuration and per-algorithm conformal calibrations are bundled and loaded
internally — the caller never passes hyperparameters.

> Note: the experiment-loading helpers (`list_experiments`, `getExperimentData`) require
> the research repo's data corpus and are **not** functional in a standalone install —
> the supported surface is the DataFrame-in API above.
