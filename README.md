# The Shrinking-Code Experiment

CSC 591 — AI & Software Engineering graduate project. We investigate whether small reusable symbolic components reduce the new code needed for successive AI capabilities across three domains. The [frozen requirements](docs/requirements.md) and [pre-registered hypotheses](docs/hypotheses.md) define the baseline; no experimental verdict is available yet.

The selected domains and original datasets are:

- Cars / Independent Used-Car Dealership: [auto93.csv](data/raw/auto93.csv).
- NASA / Software Engineering Project Manager: [nasa93dem.csv](data/raw/nasa93dem.csv).
- Apache / Apache Web Server Administrator: [Apache_AllMeasurements.csv](data/raw/Apache_AllMeasurements.csv).

## Current milestones: Remember, Guess, Certify, Flow, and Blame on Cars

**Step 1 — Remember is complete:** CAR-01 (inventory goal summaries) and CAR-02 (ranking recorded multi-goal outcomes). **Step 2 — Guess is complete after validation:** CAR-03 (fuel-efficiency estimates) and CAR-04 (weight estimates with descriptive uncertainty). **Step 3 — Certify is complete after validation:** CAR-05 (cylinder-group MPG comparisons) and CAR-06 (fixed model-year-group performance comparisons). **Step 4 — Flow is complete after validation:** CAR-07 (arrival-time assessment) and CAR-08 (automatic inventory summary and shortlist updates). **Step 5 — Blame is complete after validation:** CAR-09 (poor-MPG group diagnosis) and CAR-10 (unfavorable overall-performance group diagnosis). The remaining five skills—Justify, Choose, Fix, Spot, and Race—and the NASA and Apache demonstrations are not yet implemented.

From the repository root, using Python 3.8+ and only its standard library:

```sh
python3 -m shrinking_code.demo --domain cars --skill remember
python3 -m shrinking_code.demo --domain cars --skill guess
python3 -m shrinking_code.demo --domain cars --skill certify
python3 -m shrinking_code.demo --domain cars --skill flow
python3 -m shrinking_code.demo --domain cars --skill blame
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
- `shrinking_code/demo.py`: Cars Remember, Guess, Certify, Flow, and Blame demonstrations; domain configuration stays here.

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

## Reproducibility and disclosure

Development proceeds incrementally to preserve accurate code-history measurements. Review each working milestone before manually committing and tagging it; this implementation does not create commits, tags, or pushes. Raw datasets, frozen requirements, and hypotheses remain unchanged.

The Step 1 through Step 5 code, tests, and README changes were generated with LLM assistance. This development session is not an independent experimental trial and must not be labeled as human-written treatment code. No LOC measurements, treatment comparisons, or success claims about the research hypotheses are supplied here.

Course references: [project specification](https://github.com/txt/seai26f/blob/main/docs/submit/uproj.md) and [graduate scope](https://github.com/txt/seai26f/blob/main/docs/submit/gproj.md).
