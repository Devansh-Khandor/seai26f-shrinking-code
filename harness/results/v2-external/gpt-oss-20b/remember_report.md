# External metric, phase 5E: Remember — are the recommended records really the desirable ones?

**Status: post-observation analysis.** Added after the Task Talk; not part of the pre-registered plan in `docs/hypotheses.md`.

**Ground truth.** For Cars and NASA, a row is desirable when it is on the Pareto front: no other recorded row is at least as good on every goal and better on one, using the declared goal directions. For Apache, a configuration is desirable when its recorded `Performance-` is in the best quartile (the frozen checker's 25% cut, ties included). These follow from the declared goals and recorded values only. They are **not** verified customer preferences, prices or satisfaction.

**-01 is not an external metric.** CAR-01, NASA-01 and APACHE-01 ask for a summary of the data. Their score here is *summary fidelity*: the share of per-goal mean, median, min, max, best and worst values that are correct. It says whether the description is right, not whether a recommendation is useful.

**Integrity-valid** = ran and passed the frozen format check (which already rejects duplicate and out-of-range rows). The -02 answers cite only row numbers and computed shares, so there are no cited cell values to verify. The shuffled rerun is recorded but does not gate.

**Success** uses the frozen checker's own -02 rules: Cars/NASA, at least 8 of the 10 rows Pareto-optimal or in the best 25% by any overall score, and the list is not just the top rows of one goal; Apache, at least 80% of the listed rows in the best quartile, every listed setting share correct (±0.011), and the first setting among the two most distinguishing.

## Base rates

| domain | request | truth set | size | rows | base rate | best possible recall with k = 10 |
|---|---|---|---:|---:|---:|---:|
| cars | CAR-02 | Pareto front | 13 | 398 | 3% | 0.77 |
| nasa | NASA-02 | Pareto front | 47 | 93 | 51% | 0.21 |
| apache | APACHE-02 | best Performance- quartile | 52 | 192 | 27% | 0.19 |

A random 10-row list would score the base rate in precision on average. NASA's front holds about half of all projects, so NASA precision says little; read it with the single-goal column.

## CAR-01 (`summary`)

| T | attempted | executable | schema valid | fully correct summaries | median fidelity |
|---|---:|---:|---:|---:|---:|
| T1 | 3 | 3 | 3 | 3 | 1.000 |
| T2 | 3 | 3 | 3 | 3 | 1.000 |
| T3 | 3 | 3 | 3 | 3 | 1.000 |
| T4 | 3 | 3 | 3 | 2 | 1.000 |
| T5 | 3 | 3 | 3 | 3 | 1.000 |
| T6 | 3 | 1 | 1 | 1 | 1.000 |

## CAR-02 (`top`)

| T | attempted | executable | usable | k | median precision | median recall | median recall (attainable) | median d2h percentile | single-goal lists | successful | delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 3 | 3 | 3 | 10 | 0.80 | 0.62 | 0.80 | 0.011 | 0 | 3 | 100% |
| T2 | 3 | 3 | 3 | 10 | 0.80 | 0.62 | 0.80 | 0.011 | 1 | 2 | 67% |
| T3 | 3 | 3 | 3 | 10 | 0.80 | 0.62 | 0.80 | 0.011 | 0 | 3 | 100% |
| T4 | 3 | 3 | 3 | 10 | 0.80 | 0.62 | 0.80 | 0.011 | 0 | 3 | 100% |
| T5 | 3 | 3 | 3 | 10 | 0.80 | 0.62 | 0.80 | 0.011 | 0 | 3 | 100% |
| T6 | 3 | 1 | 1 | 10 | 0.80 | 0.62 | 0.80 | 0.011 | 0 | 1 | 33% |

## NASA-01 (`summary`)

| T | attempted | executable | schema valid | fully correct summaries | median fidelity |
|---|---:|---:|---:|---:|---:|
| T1 | 3 | 3 | 3 | 3 | 1.000 |
| T2 | 3 | 3 | 2 | 2 | 1.000 |
| T3 | 3 | 3 | 3 | 3 | 1.000 |
| T4 | 3 | 3 | 3 | 3 | 1.000 |
| T5 | 3 | 2 | 2 | 1 | 0.889 |
| T6 | 3 | 2 | 2 | 2 | 1.000 |

## NASA-02 (`top`)

| T | attempted | executable | usable | k | median precision | median recall | median recall (attainable) | median d2h percentile | single-goal lists | successful | delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 3 | 3 | 3 | 10 | 0.90 | 0.19 | 0.90 | 0.134 | 0 | 3 | 100% |
| T2 | 3 | 3 | 2 | 10 | 0.90 | 0.19 | 0.90 | 0.484 | 0 | 2 | 67% |
| T3 | 3 | 3 | 3 | 10 | 0.90 | 0.19 | 0.90 | 0.134 | 1 | 2 | 67% |
| T4 | 3 | 3 | 3 | 10 | 0.60 | 0.13 | 0.60 | 0.048 | 2 | 1 | 33% |
| T5 | 3 | 2 | 2 | 10 | 0.75 | 0.16 | 0.75 | 0.091 | 1 | 1 | 33% |
| T6 | 3 | 2 | 1 | 10 | 0.90 | 0.19 | 0.90 | 0.134 | 0 | 1 | 33% |

## APACHE-01 (`summary`)

| T | attempted | executable | schema valid | fully correct summaries | median fidelity |
|---|---:|---:|---:|---:|---:|
| T1 | 3 | 3 | 3 | 3 | 1.000 |
| T2 | 3 | 3 | 3 | 3 | 1.000 |
| T3 | 3 | 3 | 3 | 3 | 1.000 |
| T4 | 3 | 1 | 1 | 1 | 1.000 |
| T5 | 3 | 3 | 3 | 3 | 1.000 |
| T6 | 3 | 2 | 2 | 2 | 1.000 |

## APACHE-02 (`top_settings`)

| T | attempted | executable | usable | median k | median precision | median recall | median precision@10 | median performance percentile | median settings accuracy | successful | delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 3 | 3 | 2 | 14 | 1.00 | 0.27 | 1.00 | 0.021 | 0.56 | 1 | 33% |
| T2 | 3 | 3 | 2 | 20 | 1.00 | 0.38 | 1.00 | 0.031 | 1.00 | 2 | 67% |
| T3 | 3 | 3 | 3 | 10 | 1.00 | 0.19 | 1.00 | 0.031 | 1.00 | 3 | 100% |
| T4 | 3 | 1 | 1 | 20 | 1.00 | 0.38 | 1.00 | 0.031 | 1.00 | 1 | 33% |
| T5 | 3 | 3 | 3 | 19 | 1.00 | 0.37 | 1.00 | 0.031 | 1.00 | 3 | 100% |
| T6 | 3 | 2 | 2 | 14 | 1.00 | 0.28 | 1.00 | 0.021 | 1.00 | 2 | 67% |

## By domain and request

| domain | request | attempted | executable | schema valid | integrity valid | successful | delivery rate | conditional success | shuffled failures (valid) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cars | CAR-01 | 18 | 16 | 16 | 16 | 15 | 83% | 94% | 1 |
| cars | CAR-02 | 18 | 16 | 16 | 16 | 15 | 83% | 94% | 1 |
| nasa | NASA-01 | 18 | 16 | 15 | 15 | 14 | 78% | 93% | 1 |
| nasa | NASA-02 | 18 | 16 | 14 | 14 | 10 | 56% | 71% | 5 |
| apache | APACHE-01 | 18 | 15 | 15 | 15 | 15 | 83% | 100% | 0 |
| apache | APACHE-02 | 18 | 15 | 13 | 13 | 12 | 67% | 92% | 1 |

## Limitations

- **Pareto is not desirability.** Non-domination ignores price, condition, reliability and customer taste, none of which are in the data. A row can be non-dominated only because it is extreme on one goal (for example the largest NASA project), which is why the single-goal rule matters.
- **Base rates differ hugely** (Cars 13 of 398 rows on the front, NASA 47 of 93, Apache quartile 52 of 192), so precision is not comparable across domains.
- **CAR-02 and NASA-02 ask for exactly 10 rows**, so k is fixed; recall is capped at k / front size when the front is larger than 10.
- **The Apache quartile cut** is the frozen checker's 25%; it is an evaluation choice, not a customer target.
- **-01 measures fidelity**, not usefulness.
- **Three trials per cell, often fewer usable.** No statistical test is applied.
- **T7 is not included.** NASA and Apache Remember adapters exist (`t7_validation/adapters/{nasa,apache}/remember.py`); the Apache adapter reports a ten-row shortlist of its 52-row favourable group (`t7_validation/README.md`). Scoring them would need an approved adapter run. Cars T7 has no Remember adapter.

## Provenance

- Archive: `harness/results/v2/gpt-oss-20b-3trials.zip` (sha256 `d1a51a3fa3711829ede726dd08e1f77da070fc3e3064fdcd1971d24974eb7683`)
- Code commit: `6f580c8eaaf527c4292a03985ec17032268dc418` on `analysis/external-metrics` (uncommitted changes present)
- Command: `python -m harness.external`
