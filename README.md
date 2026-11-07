# The Shrinking-Code Experiment

CSC 591 — AI & Software Engineering graduate project. We investigate whether small reusable symbolic components reduce the new code needed for successive AI capabilities across three domains. The [frozen requirements](docs/requirements.md) and [pre-registered hypotheses](docs/hypotheses.md) define the baseline; no experimental verdict is available yet.

The selected domains and original datasets are:

- Cars / Independent Used-Car Dealership: [auto93.csv](data/raw/auto93.csv).
- NASA / Software Engineering Project Manager: [nasa93dem.csv](data/raw/nasa93dem.csv).
- Apache / Apache Web Server Administrator: [Apache_AllMeasurements.csv](data/raw/Apache_AllMeasurements.csv).

## Current milestones: Remember through Race on Cars

**Step 1 — Remember is complete:** CAR-01 (inventory goal summaries) and CAR-02 (ranking recorded multi-goal outcomes). **Step 2 — Guess is complete after validation:** CAR-03 (fuel-efficiency estimates) and CAR-04 (weight estimates with descriptive uncertainty). **Step 3 — Certify is complete after validation:** CAR-05 (cylinder-group MPG comparisons) and CAR-06 (fixed model-year-group performance comparisons). **Step 4 — Flow is complete after validation:** CAR-07 (arrival-time assessment) and CAR-08 (automatic inventory summary and shortlist updates). **Step 5 — Blame is complete after validation:** CAR-09 (poor-MPG group diagnosis) and CAR-10 (unfavorable overall-performance group diagnosis). **Step 6 — Justify is complete after validation:** CAR-11 (recommendation comparisons) and CAR-12 (pairwise goal tradeoffs). **Step 7 — Choose implements** CAR-13 (discover a promising car through limited inspections) and CAR-14 (prioritize inspections and report budget checkpoints). **Step 8 — Fix implements** CAR-15 (similar historical alternatives with better recorded outcomes) and CAR-16 (alternatives meeting more targets with fewer X differences). **Step 9 — Spot implements** CAR-17 (X isolation) and CAR-18 (held-out MPG and weight residual review). **Step 10 — Race implements** CAR-19 (equal-budget random selection comparison) and CAR-20 (prediction versus the training mean). This completes the Cars ten-skill ladder, not the full graduate project. NASA and Apache demonstrations and the seven-treatment experiment harness are not implemented.

From the repository root, using Python 3.8+ and only its standard library:

```sh
python3 -m shrinking_code.demo --domain cars --skill remember
python3 -m shrinking_code.demo --domain cars --skill guess
python3 -m shrinking_code.demo --domain cars --skill certify
python3 -m shrinking_code.demo --domain cars --skill flow
python3 -m shrinking_code.demo --domain cars --skill blame
python3 -m shrinking_code.demo --domain cars --skill justify
python3 -m shrinking_code.demo --domain cars --skill choose
python3 -m shrinking_code.demo --domain cars --skill fix
python3 -m shrinking_code.demo --domain cars --skill spot
python3 -m shrinking_code.demo --domain cars --skill race
```

The Remember demo reads the actual CSV, reports record counts and column roles, summarizes all three goals, and shows the top five cars with their original data-row numbers, independent characteristics, recorded goals, and scores. Unsupported domains or skills produce an explicit error and a nonzero exit status.

Run all automated tests:

```sh
python3 -m unittest discover -s tests -v
```

Tests cover real data summaries and roles, categorical values, missing data, hand-calculated synthetic distances, constant goals, ties, an independent raw-CSV calculation of the rankings, changes in arbitrary input data, and CLI behavior.

## Shared architecture

- `shrinking_code/data.py`: immutable `Column`, `Row`, and `Table` representations and MOOT CSV loading. Names and column order are preserved; row numbers start at 1 after the header.
- `shrinking_code/stats.py`: numerical count, minimum, maximum, mean, median, and normalization; symbolic count, frequencies, and mode.
- `shrinking_code/ranking.py`: equal-weight distance to heaven and deterministic historical ranking, with no Cars-specific names or values.
- `shrinking_code/distance.py`: immutable training-fitted mixed-type X distance, reusing column roles, summaries, and normalization.
- `shrinking_code/prediction.py`: immutable k-nearest-neighbors regressor for any numerical goal and descriptive neighbor uncertainty.
- `shrinking_code/evaluation.py`: deterministic row split, MAE, and a training-mean baseline for this holdout evaluation.
- `shrinking_code/certification.py`: domain-independent numeric distribution comparison with sample summaries and diagnostic thresholds.
- `shrinking_code/streaming.py`: validates and accepts sequential records, reusing existing summaries and ranking to maintain exact inventory state.
- `shrinking_code/diagnosis.py`: fixed outcome-quartile selection and structured numeric/categorical X-only diagnostic evidence.
- `shrinking_code/explanation.py`: structured historical score contributions, X neighbors, and pairwise tradeoff evidence using existing ranking and distance.
- `shrinking_code/selection.py`: masked candidate oracle, sequential predicted-d2h selection, and inspection/checkpoint records.
- `shrinking_code/alternatives.py`: observed-alternative searches, exact X differences, and direction-aware target satisfaction using existing explanations.
- `shrinking_code/anomalies.py`: X-only nearest-neighbor isolation and separately ranked held-out target residuals with structured review flags.
- `shrinking_code/baselines.py`: seeded random inspection, fair selection checkpoint summaries, and paired prediction-error comparisons using existing certification.
- `shrinking_code/demo.py`: Cars Remember through Race demonstrations; domain configuration stays here.

Uppercase initial letters denote numeric columns; lowercase initials denote symbolic columns. Goal suffixes `+` and `-` denote numeric goals to maximize and minimize. Suffix `X` excludes a column from inputs and goals while retaining its values for inspection. Remaining columns are inputs. Thus `origin` remains a string category and `HpX` is ignored. The six `?` horsepower entries load as `None`.

Cell whitespace is trimmed in memory. `?` is the supported missing marker; empty cells, malformed row widths, duplicate headers, and invalid or nonfinite numerical values are rejected. Files are never rewritten. Summaries omit missing values. An empty summary has count zero and undefined statistics (`None`); tied symbolic modes use first occurrence.

## Ranking policy and limits

For each goal, normalize a known value using its observed minimum and maximum in the entire evaluation table:

```text
normalized = (value - minimum) / (maximum - minimum)
ideal = 0 for minimize goals, 1 for maximize goals
d2h = sqrt(sum((normalized - ideal)^2) / number_of_goals)
```

All goals have equal weight. Lower `d2h` is better under this policy. A constant goal normalizes to 0.5, giving every eligible row the same squared contribution (0.25); it remains in the denominator. If every goal is constant, all eligible rows score 0.5. This neutral convention avoids division by zero without claiming that a constant measurement is ideal.

Rows missing **any** goal are excluded from ranking rather than imputed. The demo reports the excluded count. Observed goal values in otherwise incomplete rows still contribute to that goal's range. If no rows have complete goals, the ranking is empty; a table without goals raises an error. Ties use original data-row numbers. Normalization and ranking are intended for observed rows in the evaluation table, not extrapolation to unseen values.

For Cars, the policy minimizes `Lbs-` and maximizes `Acc+` and `Mpg+`. `Acc+` units and physical interpretation remain uncertain. This ranks historical outcomes; it predicts neither unseen vehicle performance nor purchase price, and it does not establish faster physical acceleration. Equal weighting is an initial policy, not a learned customer preference. Rankings depend on the evaluation table's ranges, and recorded MPG values are coarse. Batch summaries retain data in memory; Flow reuses them for exact updates.

## Guess prediction policy and evaluation

`KNNRegressor.fit(training, k=5, p=2)` fits X normalization once using only the training table. `predict(query, target)` supports any numerical goal by exact column name, including both `Mpg+` and `Lbs-`. No goal, ignored column, or row number enters feature distance. The same fitted model is used for both targets. Positive integer k is configurable; no k tuning uses test data.

For each available X column, numeric distance is the absolute difference between values normalized by the training range; categorical distance is zero for equal strings and one otherwise. Distance is `(sum(difference**p) / available_column_count)**(1/p)`, with finite `p >= 1`. If either row is missing an input, that column is skipped for that pair. Numeric inputs with no training observations are also skipped. No comparable inputs gives infinite distance, making that candidate ineligible. Constant training inputs normalize to 0.5 through the existing summary, hence contribute zero distance even for a new value. Nonconstant inputs outside the training range are not clipped; distances may exceed one.

Neighbors must have an observed requested target and finite X distance. They are ordered by distance rounded to 12 decimal places, then original row number (rounding affects ordering only, stabilizing floating-point ties); up to k are used. Prediction is their arithmetic mean. Minimum, maximum, and **population** standard deviation describe neighbor outcomes (one neighbor has zero standard deviation). These are **not calibrated prediction intervals**; zero dispersion does not imply certainty. No eligible neighbors produces an explicit error rather than a fabricated estimate.

Queries must use the training schema and original source identities. Prediction rejects a query whose row number appears in training, including a copy with masked goals. New records need unused row numbers. Row numbers only guard overlap and break ties. The evaluation checks matching schemas and disjoint train/test identities before scoring.

The fixed split sorts original row identities, shuffles them with a local `Random(42)`, assigns the first `floor(0.8 * 398) = 318` rows to training and the remaining 80 to testing, then restores row order within each partition. No goal values determine assignment. Training-only normalization remains frozen during evaluation. Both targets and the historical-training-mean baseline use the same observed-target test records. Missing test targets are omitted from both MAEs; an unscorable observed target causes evaluation to fail explicitly rather than silently reduce coverage.

Validated single-split results (k=5, p=2; 80 observed test targets each):

- `Mpg+`: predictor MAE **3.300000**; training-mean baseline MAE **7.807390** (training mean **23.899371**).
- `Lbs-`: predictor MAE **227.337500**; training-mean baseline MAE **758.529481** (training mean **2965.446541**).

Both MAEs are lower than the baseline on this split. This is descriptive holdout evaluation, not a significance claim, the full Race skill, or a seven-treatment experiment. The Guess demo prints five held-out predictions per target with actual values, errors, and neighbor dispersion. These are historical records, not live vehicle evaluations.

Limitations: **29 of 80 test rows share X combinations with training**, so this split does not establish generalization to entirely new configurations. MPG values are coarse, and the small historical dataset may not represent new inventory. Equal feature weighting and k=5 are initial fixed choices. Pairwise missing-input omission can favor neighbors compared on fewer features; constant features cannot detect novelty. Neighbor dispersion does not measure all prediction error. Exact neighbor search scans all training rows.

Step 2 validation covers training-only normalization, identity separation, deterministic splits/ties, missing inputs and targets, independent synthetic distances/predictions/MAEs, baseline correctness, reproducible CLI output, and Step 1 regression tests. At Step 2, the suite had 42 passing tests. Independent raw-CSV verification caught a floating-point tie-ordering discrepancy; ordering now rounds distances to 12 decimal places before breaking ties by row identity, with a regression test. The reported MAEs above use this corrected policy.

Step 1 refactoring: only CLI dispatch/help changed to add Guess; the Remember output body and shared data, statistics, and ranking modules remain unchanged. The old unsupported-Guess test now checks unsupported Certify instead. New prediction/evaluation logic is separate from these existing components.

## Certify: reference semantics and limits

The implementation follows [`ezr.lua` at revision 4ce98345](https://github.com/txt/seai26f/blob/4ce98345eb2632d07bd6c950322a96649e2aba0e/src/ezr-lua/ezr.lua#L613) and its [`ezr-eg6.lua --same` example](https://github.com/txt/seai26f/blob/4ce98345eb2632d07bd6c950322a96649e2aba0e/src/ezr-lua/ezr-eg6.lua). The source combines absolute pooled-sample-SD Cohen effect size, absolute Cliff's rank imbalance, and scaled two-sample KS. Its rule is:

```text
same = (cohen <= 0.35) AND (cliffs <= 0.195) AND (KS <= 1.36)
cohen = abs(mean_left - mean_right) / (pooled_sample_SD + 1e-32)
D = max absolute difference between the empirical CDFs
KS = D / sqrt((n + m) / (n * m))
cliffs = abs(number_of_left_wins - number_of_right_wins) / (n * m)
```

Thus `same=True` means all screens are within thresholds; it does **not** prove equivalence. `same=False` means at least one threshold is exceeded. It does **not** require all three to indicate a difference. Effect sizes alone do not establish statistical significance. The source's short-circuit evaluation computes only needed screens; our diagnostics compute all three. Sample variance uses Python's standard library rather than Lua's incremental accumulator, so floating-point boundary differences remain possible.

`compare()` returns both numerical summaries, diagnostics, the `same` decision, and exceeded screen names. `same()` returns the decision alone. Both accept threshold overrides in source order (Cohen, KS, Cliffs); demos use the source defaults. Missing `None` values are omitted, nonfinite values rejected, and input is sorted internally. Fewer than two observations in either group yields `same=None` (insufficient), an explicit safety policy absent from Lua. Equal constants give zero effects; unequal constants retain the source's epsilon-regularized large Cohen value. Tied observations enter empirical CDFs together and contribute no wins/losses to Cliff's delta.

Validation used the actual pinned Lua source and installed Lua runtime, not a second Python translation. Seven reference cases are stored in `tests/fixtures/same_reference.json` with source revision and SHA-256: identical, separated, tied, unequal-size, equal-constant, effect-only, and shape-only samples. The standard-library tests read those fixtures and need neither Lua nor network access. All seven real Cars comparisons also matched Lua metrics (within 1e-12 tolerance) and decisions. The actual eg6 `--same` function body and its `bag`/`again` helpers were executed with seed 42: the jittered sample returned true and the shifted sample false. This does not claim exhaustive numerical equivalence.

CAR-05 discovers cylinder groups from the data and reports every pair. CAR-06 fixes ranges at 70–76 and 77–82, scores all records once using existing `rank_rows()` and full-historical-table bounds, then groups those scores. No separate group normalization or Step 1 score change occurs. Missing goal values are excluded by the existing ranker; missing grouping values and years outside the fixed ranges are not assigned to a group.

Observed group summaries (mean, median):

- MPG, 3 cylinders: n=4, 20.000000, 20.000000.
- MPG, 4 cylinders: n=204, 29.509804, 30.000000.
- MPG, 6 cylinders: n=87, 21.264368, 20.000000.
- MPG, 8 cylinders: n=103, 14.951456, 10.000000.
- d2h, years 70–76: n=216, 0.616387, 0.589882.
- d2h, years 77–82: n=182, 0.483939, 0.465962.

All pairwise results, expressed as exceeded screens:

- 3 vs 4 cylinders: Cohen, Cliff's, KS (`same=False`).
- 3 vs 6 cylinders: none (`same=True`; not distinguished, not proven equivalent).
- 3 vs 8 cylinders: Cohen and Cliff's only (`same=False`; KS does not exceed its threshold).
- 4 vs 6, 4 vs 8, and 6 vs 8 cylinders: all three (`same=False` for each).
- Years 70–76 vs 77–82: all three (`same=False`); later records have lower mean and median d2h.

These procedure-based distinctions are exploratory historical associations. Six cylinder comparisons are unadjusted for multiplicity; no family-wise significance claim is made. The KS cutoff is an asymptotic rule, not an exact tied-data or small-sample test or a computed p-value. In particular, the 3-cylinder group has only four records, and MPG is coarse with many ties. Independence and representativeness are not established by the CSV. Neither cylinder count nor model-year code is proven to cause an outcome. Practical inventory decisions need more evidence; d2h retains the unresolved physical interpretation of `Acc+`.

Step 3 validation: 56 tests pass. Remember and Guess complete outputs remain byte-for-byte unchanged. Existing modules were reused without refactoring; only CLI dispatch/help was extended and the unsupported-Certify test now uses Flow. No statistical runtime error or failing unit test was found. The initial guessed reference filename `eg6.lua` was corrected to the repository's actual `ezr-eg6.lua` before implementation.

## Flow: arrival assessment and inventory updates

`Inventory(initial_table)` accepts an empty table with a valid goal-bearing schema or an existing inventory. `accept(cells, number=None)` takes values in original schema order, using `Column.parse()` for the existing types and missing-value policy (`?` or Python `None`). Explicit positive integer row numbers preserve source identities; otherwise the next identity is `max(existing IDs)+1`. Duplicate IDs, wrong widths, invalid numeric values, and malformed cells are rejected before state changes. Repeated characteristics with distinct IDs remain valid observations.

Each arrival is staged with the current accepted inventory in a temporary `Table`. `rank_rows()` computes common bounds from that inventory **plus the arrival**, re-scores every complete-goal candidate, and determines the arrival's 1-based rank and top-five membership. No future record enters this assessment. If bounds expand, all candidates use the expanded bounds together. The immutable `Arrival` result preserves its arrival-time decision even if later records change rankings. This evaluates observed outcomes; it does not predict missing outcomes.

Records with incomplete goals are accepted but receive `rank=None`, `score=None`, and `top_five=False`. Their observed values still update summaries and goal bounds, following Step 1 semantics. If there are fewer than five complete-goal records, all eligible records enter the shortlist. Ties use original row identity. A first complete record in an empty inventory has constant-goal midpoint normalization, not a claim of ideal performance.

After validation and assessment, the component commits the new table, all-column summaries, shortlist, and accepted-arrival count together. Initial records are not included in `accepted_count`; inventory size is `len(inventory.table.rows)`. Rows are retained in original identity order, which also makes symbolic mode ties consistent with full-batch source order. Numerical means, medians, minima/maxima and categorical frequencies/modes are recomputed using `summarize()`; there is no second statistical or ranking implementation.

The demo reuses seed-42 partitioning: **318 initial records and 80 incoming records**, with the incoming partition processed in original CSV row order. This is synthetic ordering, not actual chronology. Step 2's held-out partition is reused only as an independent Flow simulation; no predictor is trained or refitted, and Guess's frozen evaluation and MAEs remain unchanged. Every arrival decision is retained, with the first three and last three printed. Only after all arrivals are processed does the demo compare its final table, every column summary, and shortlist against independently computed full-batch results; it prints PASS only if all match, otherwise raises an error.

Observed final result: **80 accepted arrivals, 398 total records**, with shortlist rows **395, 394, 393, 396, 377**, matching Remember. The last arrival (row 395) is rank 1 with d2h 0.167460 at arrival. All three goal summaries and categorical frequencies match the full inventory.

This is a simple sequential interface, not a constant-time statistical update algorithm. It retains all records and recomputes exact summaries (including median sorting) and rankings at each acceptance, roughly O(columns × n log n) work per arrival and O(columns × n) retained data. It is single-process, not a concurrent, persistent, or real-time deployment system. Arrival decisions depend on the synthetic initial inventory and order. Existing d2h weighting and uncertain `Acc+` interpretation still apply.

Step 4 validation: **71 tests pass**, including independently calculated synthetic updates and re-ranking, missing-goal handling, rejection without mutation, no future influence, full-batch equality, and saved pre-Flow demo-output hashes for all three prior skills. The fixture records the `step3` source commit; it is regression evidence, not experimental data. No test failures or implementation errors surfaced during validation. Only CLI dispatch/help and its unsupported-skill test needed adaptation; the earlier core algorithms remain unchanged.

## Blame: historical group diagnosis

CAR-09 compares bottom and top MPG quartiles. CAR-10 compares worst and best d2h quartiles using unchanged `rank_rows()` and full-table bounds. The selection policy is fixed before feature comparison: sort eligible records ascending by `(outcome, original row number)`, take the first and last `floor(n/4)`, and discard the middle. Lower MPG is poor; higher d2h is poor. With 398 eligible records, each group contains **99** cars and 200 middle records are unused. The upper tail selects higher row IDs at a tied boundary; the lower tail selects lower IDs. The groups remain disjoint even if every outcome is tied, in which case the labels carry no actual outcome contrast.

`outcome_groups()` accepts a numerical outcome column (explicit `higher_better=False` for a minimization outcome), or uses Step 1 d2h by default. `select_quartiles()` also accepts an existing row-ID-to-score mapping. Missing outcomes are excluded; fewer than four observed outcomes raises an insufficient-data error. MPG grouping requires only observed MPG, while d2h requires all goals as before. No groups are adjusted after viewing X separation.

`compare_groups()` accepts disjoint tables with the same schema and returns structured `Finding` objects for **X columns only**. Goals, ignored fields, and row IDs are not explanatory inputs. It reuses `summarize()` and the existing certification `compare()`; no parser, normalization, ranking, or statistical test is reimplemented. Findings retain column identity, observed group summaries, total group sizes, direction, descriptive score, categorical evidence, numeric certification diagnostics, and limitations.

- **Numerical X:** show mean and median in both groups; score by the existing absolute Cliff's delta, the absolute imbalance of cross-group greater-than versus less-than pairs. Zero means no directional rank imbalance; one means complete directional separation. Mean direction is explicitly described as higher, lower, or equal, and is not a causal effect. The existing `same` result is supplementary and keeps its thresholds and semantics; `same=False` can result from an effect-size screen alone.
- **Categorical X:** show each category's counts and proportions in both groups, including zero counts for categories absent in one group. Score by total variation, `0.5 * sum(abs(poor_proportion - favorable_proportion))`. Zero means identical category proportions; one means disjoint category support. Positive per-category differences indicate overrepresentation in the poor group. Numeric-looking codes remain strings; no numeric subtraction is applied to them.

Features are ranked **separately within numerical and categorical types**, descending by their respective score, with original column order breaking ties. These are different descriptive measures and must not be read as a common cross-type importance scale. Findings with fewer than two observed X values in either group retain summaries and explicit insufficient-evidence notes but have no separation score and sort last within type. Missing values are omitted per feature, with observed counts reported. Small nonzero category counts below five are flagged as a descriptive caution, not a statistical decision threshold. Constant features have zero separation when sufficient observations exist.

Actual Cars results (poor/favorable):

- **CAR-09:** MPG means **14.646465 / 33.737374**, medians **10 / 30**, n=99/99. Numeric order: `Volume` delta **0.996429** (means 335.787879 / 101.888889; medians 350 / 97), `Clndrs` **0.994286** (7.717172 / 4.080808; medians 8 / 4), `Model` **0.853076** (73.383838 / 79.212121; medians 73 / 80). `origin` TV **0.717172**; category counts 1: **96/25**, 2: **3/25**, 3: **0/49**.
- **CAR-10:** d2h means **0.782366 / 0.363853**, medians **0.797106 / 0.385321**, n=99/99. `Clndrs` and `Volume` tie at delta **0.999694**, with column order placing `Clndrs` first. Cylinder means are 7.979798 / 4.060606 (medians 8 / 4); Volume means 346.333333 / 100.479798 (medians 350 / 97). `Model` delta **0.615856**, means 73.696970 / 77.929293 (medians 73 / 79). `origin` TV **0.747475**; category counts 1: **99/25**, 2: **0/27**, 3: **0/47**.

These show historical group associations, not individual causes, predictions, or repair advice. Coarse tied MPG makes quartile membership sensitive to row-ID tie-breaking. The selected extremes omit the middle population. Correlated characteristics may describe the same pattern; these are univariate comparisons without adjustment for confounding or multiple feature comparisons. Small counts, missingness, and selective historical sampling limit generalization. No physical meanings are assigned to origin codes, and d2h retains the uncertain `Acc+` interpretation. Large descriptive separation alone is not a significance claim.

Step 5 validation: **87 tests pass**. Independent raw-CSV calculations verified both group selections, means, medians, pairwise Cliff's deltas, and categorical TV scores. Pre-Blame hashes verify unchanged outputs for all four earlier demos. The previous core algorithms and frozen evaluation/streaming policies remain unchanged; only CLI dispatch/help and the unsupported-skill test were adapted. No unit-test failures or implementation errors surfaced during validation.

## Justify: explain observed scores and tradeoffs

`Explainer(reference_table)` uses existing `rank_rows()`, `summarize()`, and `XDistance.fit()` against one explicit historical inventory. `explain(row_number)` returns the recorded row, existing rank/score, and goal contributions: name, raw value, minimize/maximize direction, reference bounds, normalized value, ideal, and squared deviation from ideal. Averaging the squared deviations and taking the square root reconstructs d2h with unchanged equal weighting. These are contributions to a scoring formula, not causal feature attributions.

Constants normalize to 0.5 and contribute 0.25, exactly as before. Missing goals have undefined normalized values/contributions; the overall score and rank remain `None`. Known values in incomplete rows still inform reference bounds. Unknown row IDs are rejected. This component explains only records in its declared historical reference; it does not explain held-out predictions or reveal unavailable outcomes.

`neighbors(row_number, k=3)` selects up to three other rows by the existing mixed-type X distance fitted on that same reference. Goals and ignored fields do not influence similarity; numeric-coded categories remain categorical. Self is excluded by original row ID. As in Guess, distances rounded to 12 decimals determine ordering, then original row IDs break ties; displayed distances are unrounded internally. Missing X follows the existing pairwise omission policy; infinite-distance candidates are excluded. Neighbors with missing goals may be shown but remain unranked. Similarity in X is separate from desirability in Y.

`compare(a, b)` reports goals favoring each record, equal goals, unknown comparisons, and the strict lower-d2h winner when both scores exist. Equal scores have no score winner even though the historical ranking uses IDs to order ties. `tradeoff_pair()` scans nested existing ranking order: first A from best to worst, then each later B. It returns the first pair with strictly lower d2h for A and at least one recorded goal favoring each side. Incomplete rows cannot qualify; if no pair exists, it returns `None` and the demo explicitly reports that. No pair, score, or narrative outcome is hardcoded.

Actual CAR-11 result: **row 395**, d2h **0.167459891**, with Clndrs=4, Volume=97, Model=82, origin category=2, ignored HpX=52. Recorded goals are Lbs-=2130, Acc+=24.6, Mpg+=40. Squared contributions are respectively **0.021486722**, **0.000141723**, and **0.062500000**. Their average is **0.028042815**, whose square root gives the score (displayed figures rounded).

The three nearest X neighbors are **368, 361, 347**, at distances **0.010336, 0.041847, 0.042930**. Row 368 is lighter (1980 versus 2130), has lower recorded Acc+ (15.3 versus 24.6), and equal MPG (40). Rows 361 and 347 have less favorable recorded values on all three goals under the header directions. These are measured historical values, not predictions.

Actual CAR-12 pair: **395 versus 393**. Row 393 has goals 1985, 21.5, 40 and d2h **0.193398157**. Its squared contributions are **0.011124358**, **0.038584184**, **0.062500000**, averaging **0.037402847**. It wins Lbs-, while row 395 wins Acc+, and MPG ties. Row 395's smaller average squared deviation explains its better overall score. A customer who prioritizes weight could prefer row 393; the implementation does not alter weights or claim the default represents every customer's preferences.

The demo reuses Blame's top numerical overall-group finding only as explicitly labeled population context. Cliff's delta is not an individual attribution or cause of the recommendation. No causal, purchase-price, mechanical-reliability, or faster-acceleration claims are made; `Acc+` remains physically uncertain.

Limitations: explanations depend on reference bounds and the equal-weight policy, not statistical confidence. Sparse/missing X can weaken neighbor comparability. Tradeoff search may examine quadratically many pairs. The core is domain-independent for MOOT tables with numerical goals and stable unique row IDs; domain adapters still need to supply trustworthy meanings, units, and reference scope. No NASA or Apache demo, held-out-prediction explanation, or later skill is implemented.

Step 6 validation: **102 tests pass**. Tests reconstruct every Cars score from contributions and cover constants, missing goals, common bounds, neighbors, tradeoffs, no-pair cases, and all five previous demo-output hashes. One initial synthetic-test failure was corrected: a column named `X` was correctly ignored by the existing MOOT suffix rule, so the intended input was renamed `Size`. Production conventions were unchanged. No earlier core refactoring was needed; only CLI dispatch/help and the unsupported-skill test were extended.

## Choose: budget-constrained candidate inspection

CAR-13 and CAR-14 use the same seed-42 split as Guess: **318 known historical rows and 80 candidate rows**. This is a separate offline simulation of inspections using held-out historical records, not live optimization or a revised Guess evaluation. An inspection reveals all available goals for exactly one candidate and costs one; initial historical records and prediction computations are not charged as inspections. Checkpoints at 10, 20, and 40 are prefixes of one 40-inspection run, with no restart or independent tuning.

`HistoricalOracle` owns hidden candidate Y. Its immutable `visible` table retains original IDs, schema, and X cells but masks both goals and ignored fields with `None`. `select_candidates(training, visible, inspect, budget, k)` receives this masked table and a narrow callback returning only the selected record's goals. It rejects unmasked candidates, duplicate IDs, overlapping train/candidate identities, and mismatched schemas. The oracle rejects unknown IDs, repeat inspections, and calls beyond its budget, and exposes a read-only tuple of inspected IDs. This is an explicit software API boundary, not protection against malicious Python introspection. The demo orchestration loads the CSV; the selection policy never receives the unmasked candidate table.

Each round reuses `KNNRegressor` with k=5 and its existing mixed-type `XDistance` (p=2) to predict every goal for each remaining candidate. It reuses `distance_to_heaven()` with equal goal weights to score predicted goal tuples, selects the lowest score with original row ID breaking exact score ties, then calls the oracle. Actual revealed values receive the same score; the best inspected complete-goal row is retained (score, then ID). The newly observed record joins training for the next round. Predictions saved in each immutable `Inspection` are those at selection time, not later refitted predictions. The result includes per-goal prediction count and exact prefix checkpoints.

Goal summaries are fitted **once, only on the initial training rows**, using `summarize()`. The Cars training bounds happen to match the full CSV bounds: Lbs- 1613..5140, Acc+ 8..24.8, Mpg+ 10..50. That coincidence is not used by the algorithm; synthetic tests use different candidate ranges. X normalization is refitted on the growing observed training pool. Existing normalization is preserved without clipping: values outside nonconstant training bounds extrapolate, and squared distance penalizes overshooting even the favorable ideal. A constant goal always normalizes to 0.5, including a newly revealed different value. Bounds never expand after an inspection. These policies preserve prior scoring semantics but limit the desirability interpretation outside the reference range.

Budgets must be nonnegative integers (booleans rejected). Zero makes no inspections; one permits one; an oversized budget stops at pool exhaustion. Empty candidate pools return empty results. Training must be nonempty with at least one observed value for every goal. Missing revealed goals still cost one and contribute available targets to future training, but the incomplete row cannot be recommended. Until a complete row is inspected, best row/score are `None`. If a remaining candidate has no comparable observed neighbors for a goal, selection raises before that round's inspection rather than assigning an invented score. Earlier inspections remain charged in the oracle. Malformed oracle responses raise; a real inspection already performed is not refunded. Checkpoints beyond completed inspections are rejected, and checkpoint zero returns `None`.

Actual run: **40 distinct inspections**, **7,260 per-goal predictions** (2,420 candidate scoring computations). Inspection order:

```text
358, 380, 347, 364, 361, 352, 386, 390, 182, 389,
303, 322, 378, 383, 387, 333, 259, 310, 184, 215,
328, 367, 395, 317, 341, 296, 309, 360, 302, 230,
233, 275, 141, 288, 280, 283, 217, 136, 195, 175
```

At budgets **10 and 20**, best inspected row is **390**, d2h **0.253947221**. At budget **40**, it is **395**, d2h **0.167459891**, first inspected at cost 23. Its X values are Clndrs=4, Volume=97, Model=82, origin category=2; revealed goals are Lbs-=2130, Acc+=24.6, Mpg+=40. The CLI prints every decision's predicted goals/score and subsequently revealed goals/score, followed by the checkpoints and final recommendation. These are computed simulation results, not experimental treatment comparisons.

Tests use hand-calculated synthetic decisions, direction checks, ties, missing goals, constants, extrapolation, and changing next decisions after an observation. Adversarial tests alter hidden Y while preserving X and verify unchanged initial decisions, and alter a later outcome while preserving the earlier prefix. A guarded model verifies that its training IDs contain only initial and previously inspected records and that every query has masked Y/ignored cells. CLI tests check deterministic complete output, inspection counts, and independently recovered prefix minima; saved pre-Choose hashes protect all six prior demos.

Limitations: greedy selection has no exploration or global-optimum guarantee; inspection efficiency has not been established against a baseline. Predictions and rankings depend on the initial split, k, observed neighbors, equal weights, and normalization policy. The historical split contains shared X combinations and is not evidence about exclusively novel vehicles. Missing X reduces comparability; uncertain physical interpretation of Acc+ remains. All remaining candidates are predicted each round, so computation is not restricted by the inspection budget. No random baseline, Fix, Spot, Race, other-domain adapter, treatment harness, or independent experimental repetition is included.

Step 7 validation: **123 tests pass, 0 fail**, including 21 new Choose tests. All seven demos execute successfully; the six previous outputs match their pre-change SHA-256 hashes, and repeated Choose output is identical. No implementation bugs or test failures surfaced during validation. Raw datasets, frozen documentation, HEAD, project branches, and milestone tags remain unchanged.

## Fix: better existing historical alternatives

For Cars, Fix means selecting another observed vehicle record. It does not synthesize a configuration, predict an invented outcome, or describe a feasible mechanical modification. CAR-15 and CAR-16 search the full historical inventory independently of Choose; the Step 7 oracle and selection policy are unchanged. Historical presence does not establish current commercial availability.

`Alternatives(reference)` reuses Step 6's `Explainer`: existing `rank_rows()`, common full-reference goal summaries, mixed-type `XDistance`, and pairwise goal explanations. No d2h, distance, normalization, or goal-comparison algorithm is duplicated. Structured `Alternative` results carry the existing source/alternative explanations and row identities, observed goal comparisons, known changed X values, unknown X names, distance, optional target statuses, ordering key, and rationale. `improvement` is source minus alternative d2h; `pareto_dominates` requires at least one improved goal, no worsened goal, and no unknown comparison. CAR-15 has no target policy, so its target statuses are `None`.

The demo source is the worst complete-goal historical record by existing d2h, breaking equal worst scores by **smallest original row ID**. `better(source, limit=5)` excludes self, incomplete-goal alternatives, and scores equal to or worse than the source. Results sort by **raw existing X distance, lower d2h, then row ID**. No new rounding tolerance is applied to distance ties. Both records use the same full-inventory normalization bounds and unchanged equal goal weights and constant midpoint semantics. Known goals in incomplete records still inform those common bounds.

`target_status(row, columns, targets)` accepts a nonempty mapping of numerical goal names to finite thresholds (subsets are allowed). Minimize goals satisfy `value <= threshold`; maximize goals satisfy `value >= threshold`. It returns satisfied, failed, and unknown goal identities plus the satisfied count, in schema order. Missing outcomes are unknown, not satisfied or asserted to fail. Unknown columns, non-goals, booleans, and nonfinite thresholds are rejected. Cars targets are computed from full historical medians: **Lbs- <= 2803.5, Acc+ >= 15.5, Mpg+ >= 20**. This is a reproducible initial policy, not customer-specified thresholds.

`meet_more(source, targets)` requires strictly more satisfied targets and sorts lexicographically by **fewest changed X fields, smallest raw X distance, highest satisfied-target count, lowest d2h, then original ID**. It returns the best structured alternative or `None` for no solution. It does not additionally require better d2h or Pareto dominance: more targets met can coexist with worse overall score. Both source and alternatives require complete goals because d2h participates in the comparison. An unknown source ID or incomplete source raises an explicit error; `worst()` returns `None` when no complete source exists.

Only active X columns count as characteristics. Numerical changes use exact recorded inequality; categorical changes use category inequality, including numeric-looking category codes. Ignored fields never affect distance or change counts. No physical tolerance or modification-cost model is invented. For CAR-15, missing X follows existing pairwise distance omission, known changes are counted, and unknown differences are listed separately; no comparable X excludes the pair. For CAR-16, any unknown X difference excludes the pair, avoiding a falsely low change count. This conservative missing-data policy can produce no solution even when partially observed alternatives exist.

Actual source: **row 15**, d2h **0.927451502**, with Clndrs=8, Volume=400, Model=71, origin=1, and goals Lbs-=5140, Acc+=12, Mpg+=10. CAR-15 alternatives in order:

- **40:** d2h **0.871807071**, improvement **0.055644431**, X distance **0**, **0** changed X fields. Lbs- improves, Acc+ worsens, Mpg+ ties.
- **14:** d2h **0.888736665**, improvement **0.038714837**, X distance **0**, **0** changed X fields. Lbs- improves; Acc+ and Mpg+ tie. Pareto-dominates the source.
- **8:** d2h **0.917348054**, improvement **0.010103448**, X distance **0.021963824**, **1** changed field (Volume 400→383). Lbs- improves, Acc+ worsens, Mpg+ ties.
- **58:** d2h **0.766550976**, improvement **0.160900526**, X distance **0.041666667**, **1** changed field (Model 71→70). Lbs- and Mpg+ improve; Acc+ worsens.
- **18:** d2h **0.850554725**, improvement **0.076896777**, X distance **0.041666667**, **1** changed field (Model 71→72). Lbs- and Acc+ improve; Mpg+ ties. Pareto-dominates the source.

CAR-16 recommends **row 58**: the target count rises from **0/3 to 1/3**, satisfying Mpg+ only. Weight changes **5140→3761** (−1379), Acc+ **12→9.5** (−2.5), and MPG **10→20** (+10). Weight and Acc+ targets remain unmet. The single changed X field is the recorded Model code, not a proposed action on the source vehicle. Identical recorded X with different Y (rows 40 and 14) also demonstrates that these characteristics do not determine outcomes or establish causal explanations.

Limitations: recommendations depend on the reference inventory, recorded fields, equal weights, and median-target policy. Changed-field count treats fields equally and says nothing about feasibility, effort, or cost. More targets and lower d2h are different criteria; neither implies every goal improves. Missing inputs weaken comparisons; exact-value matching can count small differences. Acc+ remains physically uncertain and follows its header direction. No causal improvement, commercial availability, Spot/Race implementation, treatment comparison, or independent experimental trial is claimed.

Step 8 validation: **144 tests pass, 0 fail**, including 21 new Fix tests. All eight demos execute successfully; all seven previous outputs match saved pre-Fix hashes. Tests independently verify the Cars search ordering and synthetic lexicographic priorities, target directions, missing/constant data, exact changes, and goal tradeoffs. The Step 7 oracle module is byte-for-byte unchanged and its leakage tests still pass. No test failures or implementation bugs surfaced; no earlier core refactoring was needed. Raw data, frozen documents, HEAD, branches, and milestone tags remain unchanged.

## Spot: historical anomaly review

CAR-17 and CAR-18 provide two separate descriptive review lists. `input_isolation(reference, k=5, top=5)` scores each record by the mean distance to up to five other eligible records, using the existing `XDistance` fitted on **all historical X**. Goals and ignored fields never enter distance or normalization. Self is excluded by original ID; duplicate X combinations with distinct IDs remain neighbors. Neighbor selection follows existing kNN ordering (distance rounded to 12 decimals, then ID); the mean uses unrounded distances and `NumericSummary`. Anomaly ranks use descending raw mean distance, then original ID. This is descriptive inventory analysis, not a held-out model evaluation.

The existing distance policies apply: numeric columns use reference bounds, categories (including numeric-looking codes) compare by equality, constant numeric columns contribute zero, and missing X is omitted pairwise. Fewer than k eligible neighbors uses all available neighbors. No comparable other records, including singletons, yield `score=None`, no neighbors, and no flag; empty input yields no findings. Duplicate IDs and invalid k/top (nonpositive integers, booleans, or nonintegers) are rejected. Structured `Isolation` findings retain the row, score, neighbor IDs/distances, flag, and reason. Returned results include unflagged and unscored records, with unscored records last.

`outcome_residuals(training, test, target, k=5, top=5)` reuses `KNNRegressor` unchanged. Cars uses Guess's **seed-42 split: 318 training and 80 held-out records**, separately for Mpg+ and Lbs-. Numerical X normalization and all prediction neighbors use training records only. Before prediction, every query's Y and ignored cells are masked with `None`; only after prediction is its actual target used for the signed residual `actual - predicted` and absolute residual score. There is no held-out tuning of k, model fitting, or residual scaling. Results rank by descending absolute residual, then original ID. Raw MPG and weight residuals are never combined.

Structured `Residual` findings retain the target, actual value, existing `Prediction` (mean, neighbor range, population standard deviation, and identities), signed/absolute residual, and flag/reason. Schema mismatch, duplicate test IDs, train/test overlap, unavailable targets, and absent training target observations are rejected. Missing test targets remain unscored and unflagged with no fabricated prediction; they do not enter the review ranking. As in Guess, an observed target with no comparable observed training neighbors raises an explicit error instead of silently inventing a score. Empty test sets return no findings when training and target are valid.

The **top-five policy is fixed in advance**, separately for X isolation, MPG residuals, and weight residuals. Up to five scored records are flagged even when scores are zero or only mildly unusual; this is not a learned cutoff, significance test, calibrated anomaly probability, or assertion that the flagged records contain mistakes. Neighbor ranges and standard deviations describe observed dispersion, not calibrated prediction intervals.

Actual CAR-17 flags (mean X distances displayed to 12 decimals):

- Row **155**: **0.184120948940**; neighbors 142, 193, 189, 200, 221.
- Row **142**: **0.150201119972**; neighbors 193, 155, 342, 222, 226.
- Row **193**: **0.134944217027**; neighbors 142, 342, 226, 155, 243.
- Row **243**: **0.130941369609**; neighbors 226, 342, 193, 335, 142.
- Row **226**: **0.130269904498**; neighbors 243, 342, 193, 328, 344.

Actual CAR-18 MPG flags, shown as actual → predicted; signed residual; absolute residual:

- Row **182**: 20 → 32; **−12; 12**.
- Row **215**: 20 → 32; **−12; 12**.
- Row **184**: 20 → 30; **−10; 10**.
- Row **141**: 20 → 28; **−8; 8**.
- Row **195**: 20 → 28; **−8; 8**.

Actual CAR-18 weight flags in recorded Lbs- units:

- Row **141**: 3270 → 2394.6; **+875.4; 875.4**.
- Row **182**: 2720 → 1914; **+806; 806**.
- Row **50**: 4638 → 4013.2; **+624.8; 624.8**.
- Row **126**: 3205 → 3821; **−616; 616**.
- Row **13**: 4499 → 3931; **+568; 568**.

The CLI computes these results and also shows recorded X, prediction neighbor IDs, ranges, and standard deviations. Tests use independent synthetic distances/residuals, verify prediction equality with Guess, perturb Y without changing X isolation, and change held-out labels without changing predictions. A guarded predictor verifies masked queries and training-only normalization.

Limitations: isolation depends on k, reference scaling, missing-data comparability, and inventory composition; uncommon combinations may be valid. Residuals can reflect predictor limitations, sparse neighbors, unrecorded factors, or genuine variation. Coarse MPG and shared X combinations limit interpretation. Top-five lists provide no error-rate control and require independent review before calling a record erroneous. This is offline historical analysis, not a deployed fraud or data-quality detector. No Race, other-domain demonstration, or experimental treatment comparison is implemented.

Step 9 validation: **161 tests pass, 0 fail**, including 17 new Spot tests. All nine demos execute successfully; the previous eight outputs match saved pre-Spot SHA-256 hashes. Repeated Spot output is identical, and both synthetic and real-data checks preserve Guess prediction semantics. No bugs or test failures surfaced. Earlier core modules, raw data, frozen documents, HEAD, branches, and milestone tags remain unchanged.

## Race: customer-facing baseline comparisons

CAR-19 compares model-guided selection with random selection; CAR-20 compares kNN prediction with the historical training mean. These complete the Cars skill ladder. They do not compare the seven software-development treatments, test the graduate hypotheses, establish cross-domain transfer, or constitute independent LLM experimental repetitions. The frozen CAR-19 wording remains authoritative; the task's paraphrase did not change it.

`selection_race()` reuses the fixed seed-42 split (318 known historical rows, 80 candidates) and runs **one unchanged deterministic Choose trajectory**. It also runs **30 random trajectories with predefined seeds 0..29**, using a local Python `random.Random(seed)` to shuffle candidate rows sorted by original ID. Random selection takes a prefix without replacement and makes no predictions. The CLI records every seed's inspection sequence and checkpoint scores. Reordering the input table does not change the random sequence. Reproduction depends on this Python shuffle procedure; it is not a cross-language RNG specification.

Every run starts with the same history and candidate pool, has a fresh existing `HistoricalOracle`, and spends 40 distinct inspections; budgets 10 and 20 are prefixes of that same run. The random policy receives only masked candidates and the inspection callback, never the unmasked table. It calls existing `distance_to_heaven()` after each reveal using `summarize()` bounds fitted on initial training goals only. It retains Step 7's no-clipping and constant-midpoint policies, exact score/ID best-row tie-break, and missing-goal cost (charged but not eligible as best). Random decisions never depend on outcomes, including previously inspected ones. No strategy is tuned from comparison results.

Both methods pay the same **inspection** budget, not the same computational cost. At budgets 10/20/40, the model makes **2,265 / 4,230 / 7,260 per-goal predictions**; each random run makes zero. Total final inspections across this demonstration are 40 model-guided plus 30×40 random, or **1,240**. The initial 318 known observations are not charged. Structured results retain training/candidate IDs, the original model result, each random seed and inspection history, and checkpoint summaries. Model prediction counts at prefixes follow the number of remaining candidates times goal count and are checked against the existing final counter.

Observed CAR-19 results (lower d2h is better):

- **Budget 10:** model **0.253947221**; random mean **0.298924116**, median **0.263846607**, range **0.167459891..0.521572283**. Random matches or beats the model in **7/30 (23.33%)** runs.
- **Budget 20:** model **0.253947221**; random mean **0.265182737**, median **0.258316258**, range **0.167459891..0.382321778**. Random matches or beats the model in **10/30 (33.33%)** runs.
- **Budget 40:** model **0.167459891**; random mean **0.220614149**, median **0.253947221**, range **0.167459891..0.325585691**. Random matches or beats the model in **13/30 (43.33%)** runs.

All 30 runs have defined scores at each checkpoint. Generic summaries omit undefined best scores and explicitly report valid/total trial counts; match-or-beat uses only valid trials and is undefined when none exist or the model has no score. Direct random selection supports zero budgets and stops at pool exhaustion; race checkpoints must be positive, unique, and within the pool, with a nonempty unique seed list. No `same` test compares the single model result with the random distribution. Thirty randomized sequences on one dataset are not thirty independently sampled datasets, and a deterministic trajectory is not replicated into thirty observations.

`prediction_race(training, test, target)` calls existing `KNNRegressor.fit()` and `evaluate()` with unchanged k=5, training-only X normalization, and the existing training-mean baseline. It evaluates exactly the same observed-target test rows for both methods. The predictor's distance and neighbor selection use X only; test goals enter error calculations, never prediction fitting or neighbor choice. Missing test targets are omitted identically; no observed evaluable targets raises the existing evaluation error. A single scored record has descriptive MAEs but `same=None` (insufficient observations).

Each structured paired result retains original row ID, both absolute errors, and **kNN error minus mean error**, where negative favors kNN. The CLI prints every pair, win/loss/tie counts, MAEs, and reduction **baseline MAE minus kNN MAE**. Percentage reduction is `100 * reduction / baseline MAE`; it is undefined when baseline MAE is zero. Counts use exact error comparisons, with no new tolerance.

Observed CAR-20 results, each with 318 training and 80 scored held-out records:

- **Mpg+:** kNN MAE **3.300000000**, historical-mean MAE **7.807389937**; reduction **4.507389937 (57.732353%)**. kNN wins **64**, mean wins **16**, ties **0**.
- **Lbs-:** kNN MAE **227.337500000**, historical-mean MAE **758.529481132**; reduction **531.191981132 (70.029181%)**. kNN wins **66**, mean wins **14**, ties **0**.

Existing course `compare()` screens the two absolute-error distributions descriptively. For both targets, **same=False**, with Cohen, Cliff's, and KS thresholds exceeded. MPG: Cohen **1.126147124**, Cliff's **0.573437500**, scaled KS **3.715676251**. Weight: Cohen **1.527917887**, Cliff's **0.710312500**, scaled KS **3.952847075**. This procedure does not account for the pairing of errors; per-record differences are preserved rather than calling the distribution comparison a paired significance test. No formal statistical superiority is asserted; `same=True` would not establish equivalence. These are observed improvements and course-screen decisions, not p-values or independent-replication evidence.

Limitations: results depend on one historical split, its shared X combinations, fixed k, default equal goal weights, and uncertain Acc+ meaning. Random-seed variation describes this finite pool only; model performance across different datasets/splits is not measured. The simulation uses historical inspections, not live deployment or cost measurements. The model trajectory is computed once per Race invocation, not once per random seed. One measured development run of the complete Race demo took **5.28 seconds**; runtime is machine/load dependent and not an experimental treatment metric. No earlier algorithm was refactored or optimized.

Step 10 validation: **174 tests pass, 0 fail**, including 13 new Race tests. All ten demos execute successfully; all nine previous outputs match saved pre-Race hashes. Earlier core modules, raw datasets, frozen documents, HEAD, branches, and milestone tags are unchanged.

Step 10 test correction: one synthetic expected win count was initially wrong. Errors 0 and 80 both beat baseline errors 16.6667 and 83.3333, so the expected count was corrected from one win/one loss to two wins. The production calculation needed no change.

## Reproducibility and disclosure

Development proceeds incrementally to preserve accurate code-history measurements. Review each working milestone before manually committing and tagging it; this implementation does not create commits, tags, or pushes. Raw datasets, frozen requirements, and hypotheses remain unchanged.

The Step 1 through Step 10 code, tests, and README changes were generated with LLM assistance. This development session is not an independent experimental trial and must not be labeled as human-written treatment code. No LOC measurements, treatment comparisons, or success claims about the research hypotheses are supplied here.

Course references: [project specification](https://github.com/txt/seai26f/blob/main/docs/submit/uproj.md) and [graduate scope](https://github.com/txt/seai26f/blob/main/docs/submit/gproj.md).
