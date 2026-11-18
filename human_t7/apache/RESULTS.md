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

## Step 3: Certify

Requirements: APACHE-05 and APACHE-06.

### Statistical Method

Configurations were divided by the value of each binary setting.

For each comparison, we calculated:
- Group counts, means, and medians.
- Cohen's d (signed standardized mean difference).
- Cliff's delta (signed dominance effect).
- Two-sample KS statistic.
- Two-sided permutation-test p-value using 2,000 seeded
  permutations (seed 42).

Differences are observational and do not establish causation.

### APACHE-05: KeepAlive Comparison

| Metric | Disabled | Enabled |
|---|---:|---:|
| Configurations | 96 | 96 |
| Mean performance | 1087.19 | 1768.12 |
| Median performance | 1065 | 1890 |

Statistical results:

- Mean difference (enabled - disabled): +680.94
- Cohen's d: +2.0316
- Cliff's delta: +0.8130
- KS statistic: 0.6771
- Permutation p-value: 0.0005

KeepAlive-disabled configurations have substantially lower
recorded performance values.

### APACHE-06: Access Logging Comparison

| Metric | Disabled | Enabled |
|---|---:|---:|
| Configurations | 96 | 96 |
| Mean performance | 1501.25 | 1354.06 |
| Median performance | 1245 | 1200 |

Statistical results:

- Mean difference (enabled - disabled): -147.19
- Cohen's d: -0.3110
- Cliff's delta: -0.2161
- KS statistic: 0.1979
- Permutation p-value: 0.0370

Access-logging-enabled configurations have somewhat lower
recorded performance values.

The effect is more modest than for KeepAlive, and the
permutation p-value is unadjusted for multiple comparisons.

### Limitations

- Observational comparisons cannot establish causality.
- Server settings may be associated with other settings.
- Permutation tests here are exploratory and do not
  control for possible confounding.
- Multiple comparisons require caution.
- No repeated measurements of identical configurations
  are available to directly estimate benchmark noise.

### Verification

- The complete Apache test suite passed 27/27 tests.
- Step 3 executed successfully.
- Tests cover grouping, effect-size calculations,
  distribution comparisons, and deterministic permutation tests.

## Step 4: Flow

Requirements: APACHE-07 and APACHE-08.

### Streaming Setup

- Initial historical configurations: 154.
- Incoming benchmark records: 38.
- Final configurations: 192.
- Split reused from Guess (seed 42).
- Arrival order follows the held-out records.
- Alert threshold: historical nearest-rank 90th percentile.
- An alert is triggered when performance strictly exceeds
  the threshold computed before the arrival.
- Lower Performance- values are preferable.

### APACHE-07: Streaming Alerts

Eight incoming records triggered unusually poor performance alerts.

| Arrival | CSV row | Performance | Prior P90 |
|---:|---:|---:|---:|
| 4 | 9 | 2160 | 2100 |
| 5 | 23 | 2340 | 2130 |
| 7 | 27 | 2190 | 2130 |
| 23 | 109 | 2370 | 2130 |
| 27 | 144 | 2220 | 2130 |
| 33 | 167 | 2190 | 2130 |
| 34 | 174 | 2280 | 2160 |
| 36 | 180 | 2190 | 2160 |

Each incoming record was assessed against earlier observations
before being added to the historical inventory.

### APACHE-08: Final Updated Summary

| Metric | Result |
|---|---:|
| Final configurations | 192 |
| Minimum performance | 840 |
| Median performance | 1215 |
| Mean performance | 1427.66 |
| Maximum performance | 2640 |

Final top-five configurations:

| CSV row | Performance |
|---:|---:|
| 125 | 840 |
| 151 | 840 |
| 73 | 870 |
| 91 | 870 |
| 103 | 870 |

### Verification

- 40/40 total Apache tests passed through Step 4.
- Streaming and full-data batch summaries agree.
- Final top-five shortlist matches batch ranking.
- No future records are used in arrival-time decisions.
- Duplicate arrivals and nonfinite performance values
  are rejected by the streaming monitor.
- The stream represents simulated arrivals rather than
  actual timestamped benchmark observations.

### Limitations

- The 90th-percentile alert threshold is a documented
  implementation choice.
- The simulated arrival order is not chronological.
- An alert indicates unusual performance relative to
  previous observations, not a confirmed server defect.

## Step 5: Blame

Requirements: APACHE-09 and APACHE-10.

### Method

- Poor performers: highest 25% of recorded Performance-
  values, including ties at the cutoff.
- Poor-performance cutoff: 1890.
- Poor-performing configurations: 49.
- Remaining configurations: 143.
- Individual binary settings and two-setting combinations
  compared using between-group prevalence differences.
- Pair patterns require at least five poor-group records.
- The constant `base` is excluded.

### APACHE-09: Individual Settings

| Setting/value | Poor group | Others | Difference |
|---|---:|---:|---:|
| keepAlive=1 | 100.0% | 32.9% | +67.1 pp |
| handle=0 | 100.0% | 55.2% | +44.8 pp |
| inMemory=1 | 59.2% | 24.5% | +34.7 pp |
| accessLog=0 | 63.3% | 45.5% | +17.8 pp |
| extendedStatus=0 | 57.1% | 47.6% | +9.6 pp |
| enableSendfile=1 | 55.1% | 48.3% | +6.9 pp |
| hostnameLookups=0 | 55.1% | 48.3% | +6.9 pp |
| followSymLinks=1 | 53.1% | 49.0% | +4.1 pp |

All 49 poor-performing configurations have keepAlive=1
and handle=0.

### APACHE-10: Setting Combinations

| Combination | Poor group | Others | Difference | Poor count |
|---|---:|---:|---:|---:|
| keepAlive=1, handle=0 | 100.0% | 10.5% | +89.5 pp | 49 |
| keepAlive=1, inMemory=1 | 59.2% | 2.1% | +57.1 pp | 29 |
| keepAlive=1, accessLog=0 | 63.3% | 11.9% | +51.4 pp | 31 |
| keepAlive=1, extendedStatus=0 | 57.1% | 14.0% | +43.2 pp | 28 |
| hostnameLookups=0, keepAlive=1 | 55.1% | 14.7% | +40.4 pp | 27 |

The comparisons use group-normalized frequencies, not
absolute counts alone.

### Verification

- Complete Apache suite: 50/50 tests passed.
- Group selection, prevalence, pair enumeration,
  minimum support, and real-data invariants tested.
- Step 5 demonstration executed successfully.

### Limitations

- These are observational associations, not causal effects.
- Correlated settings can produce overlapping patterns.
- Pairwise patterns are descriptive and were not subjected
  to formal significance or multiplicity corrections.
- The highest-quarter threshold is an operational choice.

## Step 6: Justify

Requirements: APACHE-11 and APACHE-12.

### Method

- Reused the deterministic 154/38 train/test split from Guess.
- Used five-nearest-neighbor regression over eight adjustable settings.
- Selected the held-out candidate with the lowest predicted performance.
- Explained predictions using recorded outcomes and setting differences
  from the five nearest training configurations.
- Revealed held-out measured values only after making predictions
  and selecting the recommended configuration.
- Explanations describe model evidence, not causal effects.

### APACHE-11: Recommended Configuration

Recommended CSV row: 71.

| Metric | Value |
|---|---:|
| Predicted performance | 948 |
| Measured performance | 960 |
| Prediction error (predicted - measured) | -12 |
| Absolute prediction error | 12 |
| Historical neighbor SD | 62.21 |
| Historical neighbor range | 870 to 1020 |

Recommended configuration settings:

| Setting | Value |
|---|---:|
| hostnameLookups | 1 |
| keepAlive | 0 |
| enableSendfile | 0 |
| followSymLinks | 0 |
| accessLog | 0 |
| extendedStatus | 1 |
| inMemory | 0 |
| handle | 1 |

Supporting historical configurations:

| CSV row | Performance | Hamming distance | Differing setting |
|---|---:|---:|---|
| 70 | 990 | 0.125 | extendedStatus |
| 73 | 870 | 0.125 | accessLog |
| 76 | 960 | 0.125 | followSymLinks |
| 80 | 1020 | 0.125 | keepAlive |
| 122 | 900 | 0.125 | handle |

All five neighbors differ in exactly one adjustable setting.
Their mean recorded performance is 948.

### APACHE-12: Configuration Comparison

Configurations A and B are held-out historical records.

| Metric | A | B |
|---|---:|---:|
| CSV row | 2 | 24 |
| accessLog | 0 | 1 |
| Predicted performance | 1794 | 1584 |
| Historical neighbor SD | 494.04 | 399.91 |

- Differing setting: accessLog (0 -> 1).
- Predicted performance difference (B - A): -210.
- Recorded performance difference (B - A): -270.
- Difference between predicted and observed changes: 60.

The two predicted values are supported by different sets
of five historical nearest neighbors.

### Verification

- Complete Apache test suite: 62/62 tests passed.
- Explanations were checked against historical evidence.
- Candidate targets were excluded from recommendation decisions.
- Changing held-out target values did not change the recommendation.
- Fabricated historical evidence was rejected by semantic tests.

### Limitations

- Local explanations describe kNN behavior, not causal mechanisms.
- Neighbor ranges and standard deviations are descriptive,
  not calibrated confidence or prediction intervals.
- Good prediction accuracy for one selected configuration
  does not establish overall reliability.
- Observed performance differences cannot be attributed
  causally to individual settings.

## Step 7: Choose

Requirements: APACHE-13 and APACHE-14.

### Method

- Initial training history: 154 configurations.
- Held-out candidate pool: 38 configurations.
- Predictive model: five-nearest-neighbor regression.
- Ranking: lowest predicted Performance- first.
- Ranking uses only configuration settings and
  historical training outcomes.
- Candidate measured outcomes are revealed only after inspection.
- Fixed inspection plan: no retraining after each benchmark.
- All candidates are historically observed configurations.

### APACHE-14: Recommended Benchmarking Order

| Priority | CSV row | Predicted | Measured |
|---:|---:|---:|---:|
| 1 | 71 | 948 | 960 |
| 2 | 152 | 972 | 900 |
| 3 | 72 | 1008 | 930 |
| 4 | 88 | 1026 | 960 |
| 5 | 8 | 1062 | 1110 |
| 6 | 151 | 1062 | 840 |
| 7 | 87 | 1068 | 990 |
| 8 | 40 | 1104 | 1200 |
| 9 | 51 | 1128 | 1110 |
| 10 | 115 | 1140 | 1080 |

The measured values are shown retrospectively.
They were not available to the ranking algorithm.

### APACHE-13: Benchmark Budget Results

| Budget | Best CSV row | Best measured performance |
|---:|---:|---:|
| 5 | 152 | 900 |
| 10 | 151 | 840 |
| 20 | 151 | 840 |

The full held-out candidate pool contains a minimum
measured performance of 840 at CSV row 151.

This configuration was discovered at inspection 6,
after examining 6 of the 38 candidate configurations
(approximately 15.8% of the pool).

### Verification

- Complete Apache suite: 74/74 tests passed.
- Predictions ranked candidates independently of
  their measured outcomes.
- Changing candidate targets did not affect rankings.
- No overlap between training and candidate identifiers.
- Inspections respected the specified evaluation budgets.
- Best-so-far performance never increased with budget.
- The full-pool optimum was used only for retrospective analysis.

### Limitations

- No random-selection baseline has yet been evaluated;
  this is reserved for Step 10: Race.
- The model uses a fixed prediction-based ranking and
  does not update its plan after observing new benchmarks.
- Results depend on the selected train/test split.
- Finding the full-pool optimum early in one simulation
  does not establish statistically reliable superiority.
- The benchmark simulation reveals existing measurements;
  no new Apache server benchmarks were executed.

## Step 8: Fix

Requirements: APACHE-15 and APACHE-16.

### Method

- Reused the deterministic 154/38 training and held-out split.
- Selected the worst measured held-out configuration as
  the source requiring repair.
- Searched historically observed Apache configurations
  as potential valid alternatives.
- Used five-nearest-neighbor regression to estimate
  performance for source and alternative configurations.
- Excluded a historical row from its own prediction.
- Never modified the constant `base` setting.
- Did not use alternative held-out outcomes in predictions.
- Restricted changes to eight adjustable binary settings.

### Source Configuration

| Metric | Value |
|---|---:|
| CSV row | 109 |
| Recorded performance | 2370 |
| Predicted performance | 2088 |

### APACHE-15: Small Predicted Repair

Maximum permitted setting changes: 2.

Qualifying predicted-improvement alternatives: 31.

| Metric | Result |
|---|---:|
| Recommended CSV row | 63 |
| Number of setting changes | 1 |
| Changed setting | handle: 0 -> 1 |
| Alternative predicted performance | 1188 |
| Predicted improvement | 900 |
| Alternative neighbor SD | 124.78 |

The recommended alternative is an observed configuration,
not an artificially generated combination.

The predicted improvement is approximately 43.1% relative
to the source predicted performance.

### APACHE-16: Minimum Changes to Target

Specified demonstration target: Performance- <= 1200.

Qualifying predicted-to-meet-target alternatives: 81.

| Metric | Result |
|---|---:|
| Recommended CSV row | 63 |
| Number of setting changes | 1 |
| Changed setting | handle: 0 -> 1 |
| Source predicted performance | 2088 |
| Alternative predicted performance | 1188 |
| Predicted improvement | 900 |

A single setting change reaches the specified predicted
target. Since the source prediction exceeds the target,
one is the minimum possible positive number of edits.

### Verification

- Complete Apache test suite: 86/86 tests passed.
- Valid historically observed alternatives were verified.
- Identical configurations and the source itself were
  excluded from repair alternatives.
- Constant `base` modifications were prohibited.
- Self-prediction was excluded for training records.
- Repair recommendations were invariant to changes in
  held-out alternative target values.
- Minimum edit counts and target satisfaction were tested.
- Step 8 demonstration executed successfully.

### Limitations

- The 1200 performance target is an illustrative choice.
- Predicted improvements are not performance guarantees.
- A fresh benchmark is necessary to confirm an improvement.
- The source is chosen using its known measured performance.
- Repair recommendations depend on the fixed historical
  prediction model and the available observed configurations.

## Step 9: Spot

Requirements: APACHE-17 and APACHE-18.

### Method

- Reused the deterministic 154/38 training and held-out split.
- Used five-nearest-neighbor regression and normalized
  Hamming distance over eight adjustable settings.
- Calibrated both anomaly thresholds from training-only
  leave-one-out scores.
- Thresholds use nearest-rank 90th percentiles.
- A score must strictly exceed its threshold to be flagged.
- Test records are never included in their own neighbor sets.

### APACHE-17: Unexpected Performance

Performance anomaly score:

    absolute residual = abs(measured - predicted)

Training-calibrated absolute residual threshold: 294.

Flagged held-out records: 9 of 38.

Five largest absolute residuals:

| CSV row | Measured | Predicted | Residual | Absolute error |
|---:|---:|---:|---:|---:|
| 144 | 2220 | 1614 | +606 | 606 |
| 130 | 2130 | 1656 | +474 | 474 |
| 29 | 2100 | 1752 | +348 | 348 |
| 174 | 2280 | 1944 | +336 | 336 |
| 176 | 2160 | 1824 | +336 | 336 |

All five configurations exceed the training-derived
absolute residual threshold.

Measured outcomes are used only to calculate residuals
after predictions have been generated.

### APACHE-18: Unusual Configuration Settings

Configuration isolation score:

    mean normalized Hamming distance to five nearest
    historical configurations

Training-calibrated isolation threshold: 0.1500.

Flagged held-out records: 1 of 38.

Five highest isolation scores:

| CSV row | Isolation score | Nearest distance | Flagged |
|---:|---:|---:|---|
| 9 | 0.1750 | 0.1250 | Yes |
| 23 | 0.1500 | 0.1250 | No |
| 174 | 0.1500 | 0.1250 | No |
| 2 | 0.1250 | 0.1250 | No |
| 7 | 0.1250 | 0.1250 | No |

The configuration anomaly detector uses only adjustable
settings and never uses the measured performance outcome.

### Detector Comparison

| Metric | Count |
|---|---:|
| Outcome anomalies | 9 |
| Configuration anomalies | 1 |
| Flagged by both detectors | 0 |

The two methods detect different forms of unusual behavior.

### Verification

- Complete Apache test suite: 98/98 tests passed.
- Outcome residual calculations verified.
- Configuration-distance calculations verified.
- Evaluated records excluded from their own neighbor sets.
- Thresholds computed exclusively from training records.
- Changing test performance values did not affect
  configuration anomaly scores or threshold calibration.
- Synthetic examples verified that unexpected performance
  and unusual settings can be detected independently.

### Limitations

- Thresholds are exploratory 90th-percentile screening rules.
- A training-based 90th-percentile threshold does not
  guarantee a 10% anomaly rate on held-out configurations.
- Binary settings produce discrete isolation scores and ties.
- Anomaly flags do not establish a server fault.
- No repeated measurements of identical configurations
  are available to directly estimate benchmark noise.

## Step 10: Race

Requirements: APACHE-19 and APACHE-20.

### Experimental Setup

- Historical training configurations: 154.
- Held-out evaluation configurations: 38.
- Deterministic train/test split: seed 42.
- Candidate selection: fixed five-nearest-neighbor prediction plan.
- Random baseline: 30 seeded random inspection orders.
- Inspection budgets: 5, 10, and 20.
- Both selection strategies use the same held-out candidate pool.
- Lower Performance- values are better.

### APACHE-19: Model-Guided vs Random Selection

| Budget | Guided best | Random mean | Random median | Random SD | Guided W/T/L |
|---:|---:|---:|---:|---:|---|
| 5 | 900 | 1022 | 960 | 214.31 | 21/5/4 |
| 10 | 840 | 930 | 930 | 81.49 | 23/7/0 |
| 20 | 840 | 877 | 840 | 42.92 | 14/16/0 |

W/T/L counts the 30 trials in which the guided result
was better than, tied with, or worse than random selection.

Random-selection ranges:

| Budget | Random minimum | Random maximum |
|---:|---:|---:|
| 5 | 840 | 1980 |
| 10 | 840 | 1170 |
| 20 | 840 | 960 |

The full held-out candidate pool's minimum is 840.
The model-guided plan discovered this value after six
configuration inspections.

The model-guided strategy achieved a better result than
the mean random-selection outcome at every evaluated budget.

This does not establish superiority in every random trial
or on independently sampled datasets.

### APACHE-20: Prediction Baseline Comparison

| Metric | Result |
|---|---:|
| Held-out observations | 38 |
| Five-nearest-neighbor MAE | 174.79 |
| Historical-mean baseline MAE | 484.05 |
| Absolute MAE improvement | 309.26 |
| Relative MAE improvement | 63.9% |
| kNN better on individual records | 34 |
| Equal absolute errors | 0 |
| Historical-mean baseline better | 4 |

Exploratory paired-bootstrap interval for the difference
in mean absolute error (baseline minus kNN):

    [+242.94, +376.06]

The paired bootstrap used 2,000 seeded resamples of the
38 per-record error differences.

### Verification

- Complete Apache test suite: 113/113 tests passed.
- Random trials were reproducible.
- Both selection strategies used identical candidate pools
  and inspection budgets.
- Random results at different budgets were derived from
  the same inspection order within each trial.
- At a full-pool budget, both approaches reach the same
  retrospective optimum.
- Win/tie/loss totals were checked against trial counts.
- Paired error calculations and bootstrap reproducibility
  were tested.
- Prediction MAEs reconciled with the earlier Guess results.

### Limitations

- Only one deterministic training/test split was evaluated.
- Thirty random inspection orders are not thirty
  independent learning-experiment replications.
- Random-trial variability measures selection-order
  variability for this fixed candidate pool.
- Bootstrap uncertainty does not account for different
  training samples, datasets, or deployment conditions.
- The five-nearest-neighbor strategy was not tuned through
  an independent hyperparameter-selection procedure.
- Prediction and selection improvements are empirical
  findings for the available historical measurements.

## Final Cumulative LOC Audit

The read-only audit compared filtered implementation-source
snapshots at Git tags t7-apache-step1 through t7-apache-step10.

| Step | Skill | New LOC | Deleted LOC | Cumulative LOC |
|---:|---|---:|---:|---:|
| 1 | Remember | 162 | 0 | 162 |
| 2 | Guess | 210 | 0 | 372 |
| 3 | Certify | 133 | 0 | 505 |
| 4 | Flow | 173 | 0 | 678 |
| 5 | Blame | 168 | 0 | 846 |
| 6 | Justify | 216 | 0 | 1062 |
| 7 | Choose | 182 | 0 | 1244 |
| 8 | Fix | 182 | 0 | 1426 |
| 9 | Spot | 205 | 0 | 1631 |
| 10 | Race | 244 | 0 | 1875 |

### Primary Research Metric

**New implementation LOC, Steps 2-10: 1713.**

Additional measurements:

- Total new implementation LOC, Steps 1-10: 1875.
- Deleted implementation LOC: 0.
- Separately counted new test LOC: 1662.
- Final cumulative implementation LOC: 1875.

The final cumulative LOC was independently verified by
counting filtered lines across the ten implementation modules.

The original tagged checkpoints remain unchanged.

Absolute added LOC did not consistently shrink across skills.
Relative new LOC as a percentage of the cumulative
implementation generally declined.

See EXPERIMENT.md for the counting method, reproduction
instructions, and study limitations.
