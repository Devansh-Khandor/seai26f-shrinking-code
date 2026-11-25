# Apache T7 — Experimental Protocol and Reproducibility

**Course:** CSC 591 — AI & Software Engineering  
**Project:** The Shrinking-Code Experiment  
**Treatment:** T7 — Cumulative human-directed, AI-assisted implementation  
**Domain:** Apache Server Configuration Optimization  
**Frozen requirements:** `docs/requirements_apache.md`  
**Dataset:** `data/raw/Apache_AllMeasurements.csv`

## 1. Research Objective

This implementation serves as one cumulative T7 reference for
the shrinking-code experiment.

It implements ten skills addressing twenty frozen customer
requirements, APACHE-01 through APACHE-20.

Each successive step extends the implementation and may reuse
code from previous skills.

The experiment measures how much additional implementation code
is introduced as capabilities accumulate.

This is one T7 implementation sequence, not ten independent
randomized repetitions.

## 2. Dataset

The Apache dataset contains 192 observed server configurations.

Each row includes:

- Eight adjustable binary configuration settings:
  hostnameLookups, keepAlive, enableSendfile, followSymLinks,
  accessLog, extendedStatus, inMemory, and handle.
- One constant setting: base.
- One optimization target: Performance-.

Performance- must be minimized.

The dataset does not establish the physical units of
Performance-, so none are assumed.

Only historically observed configuration combinations are
treated as valid alternatives for recommendations.

There are no repeated measurements of identical configurations
for directly estimating benchmark noise.

## 3. Train/Test Protocol

For prediction-based tasks, a deterministic shuffled 80/20
partition is created with Python's random.Random(42).

- Training history: 154 configurations.
- Held-out evaluation set: 38 configurations.
- Prediction model: five-nearest-neighbor regression.
- Distance: normalized Hamming distance across eight settings.
- Prediction: mean Performance- of the five nearest
  historical observations.
- Distance ties: resolved by CSV row number.
- Baseline predictor: training-set mean Performance-.

Held-out Performance- values are not used to train the model
or prioritize candidate evaluations.

They become available only when a simulated benchmark is
inspected or when retrospective evaluation is performed.

The same partition is reused for the applicable skills.

This fixed split does not measure variation across
independently sampled training/test partitions.

## 4. Skills and Requirements

| Step | Skill | Requirements | Implementation |
|---:|---|---|---|
| 1 | Remember | APACHE-01/02 | remember.py |
| 2 | Guess | APACHE-03/04 | guess.py |
| 3 | Certify | APACHE-05/06 | certify.py |
| 4 | Flow | APACHE-07/08 | flow.py |
| 5 | Blame | APACHE-09/10 | blame.py |
| 6 | Justify | APACHE-11/12 | justify.py |
| 7 | Choose | APACHE-13/14 | choose.py |
| 8 | Fix | APACHE-15/16 | fix.py |
| 9 | Spot | APACHE-17/18 | spot.py |
| 10 | Race | APACHE-19/20 | race.py |

Each module has its own executable main() demonstration.

## 5. Main Implementation Choices

### Remember

Reports descriptive performance statistics and the best/worst
historical configurations.

Defines favorable records as the lowest 25% of Performance-
values, including ties.

### Guess

Uses five-nearest-neighbor regression to estimate performance
and evaluate one-setting configuration comparisons.

Neighbor standard deviations and ranges are descriptive
uncertainty indicators, not calibrated prediction intervals.

### Certify

Compares keepAlive and accessLog groups using descriptive
statistics, Cohen's d, Cliff's delta, a KS statistic, and a
seeded 2,000-permutation test of the difference in means.

These tests are exploratory observational comparisons.
They do not establish the independent causal impact of a
server setting.

These statistics are separate from ezr's same() and
bestRanks() treatment-ranking methodology.

### Flow

Simulates 38 incoming records after an initial history of 154.

Flags an arrival when its performance strictly exceeds the
90th percentile of the previously available history.

Updates summaries and the top-five configurations incrementally.
Alert thresholds are calculated before inserting each arrival.

### Blame

Defines poor performers as the highest 25% of recorded
Performance- values, including ties.

Ranks individual setting values and two-setting combinations
by differences in prevalence between poor and remaining
configurations.

Pair patterns require at least five poor-group observations.

### Justify

Explains a recommended held-out configuration using
five comparable historical observations.

Also explains predicted and recorded differences between
two held-out configurations with one differing setting.

### Choose

Constructs a fixed model-guided inspection plan, ordered by
lowest predicted performance.

Evaluates the best measured performance discovered under
inspection budgets of 5, 10, and 20.

The plan is not retrained after each inspection.

### Fix

Searches historically observed valid configurations for
alternatives with lower predicted performance.

For training-set alternatives, the candidate itself is
excluded from its kNN neighbor set.

The demonstration target for APACHE-16 is Performance- <= 1200.

Recommendations are predictions, not guarantees.

### Spot

Uses two separate anomaly scores:

1. Absolute held-out prediction residual.
2. Mean normalized Hamming distance to five training neighbors.

Both thresholds are calibrated using training-only
leave-one-out 90th-percentile scores.

The configuration-isolation score does not use Performance-.

### Race

Compares the fixed guided inspection plan with 30 seeded
random inspection orders of the same held-out candidate pool.

Budgets: 5, 10, and 20.

Also compares paired kNN and historical-mean absolute errors
on the same 38 held-out records.

The MAE-improvement interval uses 2,000 seeded paired
bootstrap resamples.

The random orders are not independent learning-experiment runs.

## 6. Verification

The final Apache implementation passed 113 automated unittest
cases across ten test modules.

Tests cover, among other behaviors:

- Numerical summaries and ranking direction.
- Threshold handling and tied performance values.
- Deterministic train/test separation.
- Exclusion of held-out outcomes from model predictions.
- Nearest-neighbor calculations and deterministic tie-breaking.
- Group statistics and permutation-test reproducibility.
- Streaming alerts based on prior history only.
- Incremental summary and shortlist reconciliation.
- Setting and pairwise prevalence calculations.
- Explanation evidence tied to historical observations.
- Budget accounting and prediction-only candidate rankings.
- Valid repair alternatives and minimum edit counts.
- Separate outcome and input-configuration anomalies.
- Fair random-baseline budgets and paired prediction errors.

Passing tests is evidence for the checked behaviors, not
proof of correctness for every possible input or every
interpretation of the customer requirements.

The T7 test suite should not be treated as equivalent to
the common cross-treatment semantic acceptance suite unless
the same tests are actually run against the other treatments.

## 7. Code-Growth Measurement

The measurement replicates the existing NASA T7 auditor and
the treatment-harness snapshot-difference convention.

At each successive Git checkpoint:

1. Read committed Python source files using git show.
2. Remove blank lines and comment-only lines.
3. Compare filtered old/new source with
   difflib.SequenceMatcher(autojunk=False).
4. Count inserts and replacement lines as additions.
5. Count deletions and replacement lines as deletions.
6. Report test additions separately.
7. Exclude tests from primary implementation LOC.

The method is a filtered source-line diff rather than
an abstract syntax tree or token-based complexity metric.

| Step | Skill | New implementation LOC | Cumulative LOC |
|---:|---|---:|---:|
| 1 | Remember | 162 | 162 |
| 2 | Guess | 210 | 372 |
| 3 | Certify | 133 | 505 |
| 4 | Flow | 173 | 678 |
| 5 | Blame | 168 | 846 |
| 6 | Justify | 216 | 1062 |
| 7 | Choose | 182 | 1244 |
| 8 | Fix | 182 | 1426 |
| 9 | Spot | 205 | 1631 |
| 10 | Race | 244 | 1875 |

Measured totals:

- New implementation LOC, Steps 1-10: 1875.
- Primary new implementation LOC, Steps 2-10: 1713.
- Deleted implementation LOC: 0.
- New test LOC, tracked separately: 1662.
- Final cumulative implementation LOC: 1875.

The audit was independently cross-checked by counting
filtered source lines in all ten final implementation modules.
Both methods returned 1875.

The percentage of new code relative to cumulative code
generally decreased, but absolute new LOC did not
consistently decline across steps.

## 8. Git Checkpoints

The ten implementation snapshots are:

- t7-apache-step1
- t7-apache-step2
- t7-apache-step3
- t7-apache-step4
- t7-apache-step5
- t7-apache-step6
- t7-apache-step7
- t7-apache-step8
- t7-apache-step9
- t7-apache-step10

The final implementation checkpoint is:

    t7-apache-step10
    0c6db76

Development branch:

    human-t7-apache

Later audit and documentation commits do not change the
tagged implementation checkpoints.

## 9. Reproduction

Run the commands from the repository root.

All demonstrations:

    for skill in remember guess certify flow blame justify choose fix spot race; do
        python3 "human_t7/apache/$skill.py" || break
    done

Complete unit-test suite:

    python3 -m unittest discover \
        -s human_t7/apache/tests \
        -p 'test_*.py' -v

Read-only code-growth audit:

    python3 human_t7/apache/audit_apache_harness_loc.py

List immutable implementation references:

    git tag --list 't7-apache-step*' | sort -V

All ten implementation modules use the Python standard library.

No GPU is required for these local analyses.

## 10. Interpretation and Limitations

T7 is a single cumulative, human-directed, AI-assisted
implementation. AI assistance was used during development;
this is not an unaided-human programming baseline.

The observed results are specific to this dataset, fixed split,
frozen requirements, and implementation choices.

Key limitations:

- Only one cumulative T7 implementation sequence.
- No within-T7 variance estimate.
- One deterministic train/test partition.
- Observational data cannot establish causal setting effects.
- No direct repeated-measurement noise estimate.
- Several thresholds are documented operational choices.
- The kNN model and hyperparameters were not independently tuned.
- Thirty random inspection orders are not thirty independent
  repetitions of model training and code generation.
- Bootstrap intervals are conditional on the fixed sample.
- Unit tests do not constitute universal semantic certification.
- Different domain requirements can affect code size.

Treatment-level statistical ranking using ezr's same()
and bestRanks() should be performed as a separate
cross-treatment analysis once repeated treatment outputs
and a common validation protocol are available.

This T7 experiment should be compared using the same
source-filtering, LOC-counting, and requirement-evaluation
definitions as the other treatments.
