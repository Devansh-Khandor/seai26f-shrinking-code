# The Shrinking-Code Experiment

CSC 591 — AI & Software Engineering graduate project. We investigate whether small reusable symbolic components reduce the new code needed for successive AI capabilities across three domains. The [frozen requirements](docs/requirements.md) and [pre-registered hypotheses](docs/hypotheses.md) define the baseline; no experimental verdict is available yet.

The selected domains and original datasets are:

- Cars / Independent Used-Car Dealership: [auto93.csv](data/raw/auto93.csv).
- NASA / Software Engineering Project Manager: [nasa93dem.csv](data/raw/nasa93dem.csv).
- Apache / Apache Web Server Administrator: [Apache_AllMeasurements.csv](data/raw/Apache_AllMeasurements.csv).

## Current milestone: Step 1 — Remember on Cars

Only **CAR-01** (inventory goal summaries) and **CAR-02** (ranking recorded multi-goal outcomes) are implemented. The other nine skills—Guess, Certify, Flow, Blame, Justify, Choose, Fix, Spot, and Race—and the NASA and Apache demonstrations are not yet implemented.

From the repository root, using Python 3.8+ and only its standard library:

```sh
python3 -m shrinking_code.demo --domain cars --skill remember
```

The demo reads the actual CSV, reports record counts and column roles, summarizes all three goals, and shows the top five cars with their original data-row numbers, independent characteristics, recorded goals, and scores. Unsupported domains or skills produce an explicit error and a nonzero exit status.

Run all automated tests:

```sh
python3 -m unittest discover -s tests -v
```

Tests cover real data summaries and roles, categorical values, missing data, hand-calculated synthetic distances, constant goals, ties, an independent raw-CSV calculation of the rankings, changes in arbitrary input data, and CLI behavior.

## Shared architecture

- `shrinking_code/data.py`: immutable `Column`, `Row`, and `Table` representations and MOOT CSV loading. Names and column order are preserved; row numbers start at 1 after the header.
- `shrinking_code/stats.py`: numerical count, minimum, maximum, mean, median, and normalization; symbolic count, frequencies, and mode.
- `shrinking_code/ranking.py`: equal-weight distance to heaven and deterministic historical ranking, with no Cars-specific names or values.
- `shrinking_code/demo.py`: the Cars Remember command-line demonstration.

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

For Cars, the policy minimizes `Lbs-` and maximizes `Acc+` and `Mpg+`. `Acc+` units and physical interpretation remain uncertain. This ranks historical outcomes; it predicts neither unseen vehicle performance nor purchase price, and it does not establish faster physical acceleration. Equal weighting is an initial policy, not a learned customer preference. Rankings depend on the evaluation table's ranges, and recorded MPG values are coarse. Batch summaries retain data in memory; no streaming or prediction skill is implemented.

## Reproducibility and disclosure

Development proceeds incrementally to preserve accurate code-history measurements. Review each working milestone before manually committing and tagging it; this implementation does not create commits, tags, or pushes. Raw datasets, frozen requirements, and hypotheses remain unchanged.

The Step 1 package, tests, and README were generated with LLM assistance. This development session is not an independent experimental trial and must not be labeled as human-written treatment code. No LOC measurements, treatment comparisons, or success claims about the research hypotheses are supplied here.

Course references: [project specification](https://github.com/txt/seai26f/blob/main/docs/submit/uproj.md) and [graduate scope](https://github.com/txt/seai26f/blob/main/docs/submit/gproj.md).
