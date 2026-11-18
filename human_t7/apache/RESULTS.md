# Apache T7 — Experimental Results

## Step 1: Remember

Requirements: APACHE-01 and APACHE-02.

### Performance Summary

| Metric | Value |
|---|---:|
| Configurations | 192 |
| Minimum performance | 840 |
| Median performance | 1215 |
| Mean performance | 1427.66 |
| Maximum performance | 2640 |

Lower `Performance-` values are preferable.

### Best Historical Configurations

| CSV row | Performance |
|---|---:|
| 125 | 840 |
| 151 | 840 |
| 73 | 870 |
| 91 | 870 |
| 103 | 870 |

### Favorable Group

Definition: lowest 25% of performance values, including ties.

- Cutoff: 1050
- Favorable configurations: 52
- Remaining configurations: 140

Largest observed setting-frequency differences:

| Setting | Favorable | Remaining |
|---|---:|---:|
| keepAlive=1 | 7.7% | 65.7% |
| hostnameLookups=1 | 86.5% | 36.4% |
| inMemory=1 | 21.2% | 37.9% |

These associations do not establish causation.

### Verification

- 7/7 behavioral unit tests passed.
- The Step 1 demonstration executed successfully.
- Full independent semantic certification is not claimed.

## Incremental LOC

Harness-compatible LOC measurements will be calculated
from the cumulative Git checkpoints.

## Step 2: Guess

Requirements: APACHE-03 and APACHE-04.

### Prediction Method

- Deterministic shuffled train/test split (seed 42).
- Training records: 154.
- Held-out records: 38.
- Five-nearest-neighbor regression.
- Normalized Hamming distance over eight adjustable settings.
- Each prediction is the mean performance of five nearest historical records.
- No held-out performance value is used during prediction.

### APACHE-03: Prediction Accuracy

| Metric | MAE |
|---|---:|
| 5-nearest-neighbor model | 174.79 |
| Historical-mean baseline | 484.05 |

The model reduced MAE by approximately 63.9% relative to
the historical-mean baseline on this held-out split.

This is an exploratory result from one deterministic split,
not a statistical significance claim.

### APACHE-04: One-Setting Comparison

- Eligible held-out configuration pairs: 31.
- Configuration A: row 2.
- Configuration B: row 24.
- Differing setting: accessLog (0 -> 1).
- Predicted performance A: 1794.
- Predicted performance B: 1584.
- Predicted difference (B - A): -210.
- Observed held-out difference (B - A): -270.
- Neighbor standard deviations: 494.04 and 399.91.

The neighbor ranges and standard deviations are descriptive
uncertainty indicators, not calibrated prediction intervals.

The observed association does not establish causation.

### Verification

- Prediction calculations, distance metric, baseline MAE,
  deterministic splitting, and leakage safeguards tested.
- APACHE-04 pair validation and uncertainty outputs checked.
- Step 2 demonstration executed successfully.
