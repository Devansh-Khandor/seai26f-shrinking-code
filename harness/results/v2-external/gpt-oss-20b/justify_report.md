# External metric, phase 5D: Justify — are the explanations truthful?

**Status: post-observation analysis.** Added after the Task Talk; not part of the pre-registered plan in `docs/hypotheses.md`. Success rules are analysis choices unless they repeat a frozen checker rule.

**What this measures.** Whether the answer's claims about recorded rows are true: which goals the non-preferred row really wins (trade-offs), whether the recommended row is really good, whether the cited neighbours are really similar, and whether cited values exist in the file. It is **not** a human rating of how useful an explanation is, and a truthful comparison of two recorded rows is not evidence that changing one into the other would cause an improvement.

**Integrity-valid** = ran, passed the frozen format check, and every cited cell value matches the file. The programs may see all goal values (retrospective task), except NASA-11's held-out project, for which no mask rerun exists.

## Metric definitions

| request | kind | metrics | success |
|---|---|---|---|
| CAR-12, NASA-12, APACHE-12 | `justify_pair` | true trade-offs T = goals where the non-preferred row is strictly better (ties are not trade-offs); precision = \|claimed ∩ T\| / \|claimed\|, recall = \|claimed ∩ T\| / \|T\| (blank when the denominator is 0); exact-set match; false claims and omissions counted separately; whether the preferred row is dominated; whether the program's scores rank it higher | exact trade-offs, preferred not dominated, cited differences real and really different, scores consistent; CAR-12 also needs T non-empty ("different strengths", frozen) |
| CAR-11, APACHE-11 | `justify_neighbors` | recommended row's d2h and percentile (share of rows strictly better; 0 = best); Pareto / good-row membership (frozen); neighbours among the 25% most similar (frozen X-space) and their mean distance percentile | recommended row good and every neighbour similar |
| NASA-11 | `justify_estimate` | single held-out project: estimate error and the training-mean error on that row (reported only: no leakage rerun exists); neighbours similar and not held out | project held out, neighbours not held out and similar, estimate within the frozen neighbour range |

## CAR-11 (`justify_neighbors`)

| domain | T | attempted | executable | schema valid | integrity valid | successful | delivery rate | recommended good | median d2h percentile | median similar share | median neighbour distance percentile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cars | T1 | 3 | 3 | 3 | 3 | 2 | 67% | 2 | 0.000 | 1.00 | 0.085 |
| cars | T2 | 3 | 3 | 3 | 3 | 2 | 67% | 3 | 0.000 | 1.00 | 0.059 |
| cars | T3 | 3 | 3 | 3 | 3 | 2 | 67% | 2 | 0.000 | 1.00 | 0.059 |
| cars | T4 | 3 | 3 | 3 | 3 | 2 | 67% | 3 | 0.000 | 1.00 | 0.023 |
| cars | T5 | 3 | 2 | 2 | 2 | 0 | 0% | 0 | 0.859 | 1.00 | 0.033 |
| cars | T6 | 3 | 0 | 0 | 0 | 0 | 0% | 0 | — | — | — |

## CAR-12 (`justify_pair`)

| domain | T | attempted | executable | schema valid | integrity valid | successful | delivery rate | exact trade-offs | non-trivial pairs | false claims | omissions | median precision | median recall | preferred dominated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cars | T1 | 3 | 3 | 3 | 3 | 1 | 33% | 3 | 1 | 0 | 0 | 1.00 | 1.00 | 0 |
| cars | T2 | 3 | 3 | 3 | 3 | 1 | 33% | 3 | 1 | 0 | 0 | 1.00 | 1.00 | 0 |
| cars | T3 | 3 | 3 | 3 | 3 | 1 | 33% | 3 | 1 | 0 | 0 | 1.00 | 1.00 | 0 |
| cars | T4 | 3 | 3 | 3 | 3 | 0 | 0% | 2 | 1 | 0 | 1 | — | 0.00 | 0 |
| cars | T5 | 3 | 2 | 2 | 2 | 1 | 33% | 2 | 1 | 0 | 0 | 1.00 | 1.00 | 0 |
| cars | T6 | 3 | 0 | 0 | 0 | 0 | 0% | 0 | 0 | 0 | 0 | — | — | 0 |

## NASA-11 (`justify_estimate`)

| domain | T | attempted | executable | schema valid | integrity valid | successful | delivery rate | median similar share | median estimate error ÷ mean error (one row) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nasa | T1 | 3 | 1 | 1 | 1 | 0 | 0% | 0.67 | 0.27 |
| nasa | T2 | 3 | 3 | 2 | 2 | 1 | 33% | 1.00 | 0.75 |
| nasa | T3 | 3 | 3 | 2 | 2 | 2 | 67% | 1.00 | 0.07 |
| nasa | T4 | 3 | 2 | 1 | 1 | 0 | 0% | 0.67 | 0.32 |
| nasa | T5 | 3 | 2 | 2 | 2 | 1 | 33% | 0.50 | 0.59 |
| nasa | T6 | 3 | 2 | 2 | 1 | 1 | 33% | 1.00 | 0.12 |

## NASA-12 (`justify_pair`)

| domain | T | attempted | executable | schema valid | integrity valid | successful | delivery rate | exact trade-offs | non-trivial pairs | false claims | omissions | median precision | median recall | preferred dominated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nasa | T1 | 3 | 1 | 1 | 1 | 1 | 33% | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 0 |
| nasa | T2 | 3 | 3 | 2 | 2 | 1 | 33% | 2 | 1 | 0 | 0 | 1.00 | 1.00 | 0 |
| nasa | T3 | 3 | 3 | 2 | 2 | 0 | 0% | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 0 |
| nasa | T4 | 3 | 2 | 1 | 1 | 0 | 0% | 0 | 1 | 2 | 1 | 0.00 | 0.00 | 0 |
| nasa | T5 | 3 | 2 | 2 | 2 | 2 | 67% | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 0 |
| nasa | T6 | 3 | 2 | 2 | 2 | 0 | 0% | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 0 |

## APACHE-11 (`justify_neighbors`)

| domain | T | attempted | executable | schema valid | integrity valid | successful | delivery rate | recommended good | median d2h percentile | median similar share | median neighbour distance percentile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| apache | T1 | 3 | 3 | 3 | 3 | 3 | 100% | 3 | 0.000 | 1.00 | 0.000 |
| apache | T2 | 3 | 3 | 2 | 2 | 2 | 67% | 2 | 0.000 | 1.00 | 0.000 |
| apache | T3 | 3 | 2 | 2 | 2 | 2 | 67% | 2 | 0.000 | 1.00 | 0.000 |
| apache | T4 | 3 | 3 | 3 | 3 | 3 | 100% | 3 | 0.000 | 1.00 | 0.000 |
| apache | T5 | 3 | 3 | 3 | 3 | 3 | 100% | 3 | 0.000 | 1.00 | 0.000 |
| apache | T6 | 3 | 1 | 1 | 1 | 1 | 33% | 1 | 0.000 | 1.00 | 0.000 |

## APACHE-12 (`justify_pair`)

| domain | T | attempted | executable | schema valid | integrity valid | successful | delivery rate | exact trade-offs | non-trivial pairs | false claims | omissions | median precision | median recall | preferred dominated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| apache | T1 | 3 | 3 | 2 | 2 | 2 | 67% | 2 | 0 | 0 | 0 | — | — | 0 |
| apache | T2 | 3 | 3 | 2 | 2 | 1 | 33% | 1 | 0 | 1 | 0 | 0.00 | — | 0 |
| apache | T3 | 3 | 2 | 2 | 2 | 0 | 0% | 1 | 0 | 1 | 0 | 0.00 | — | 0 |
| apache | T4 | 3 | 3 | 2 | 2 | 2 | 67% | 2 | 0 | 0 | 0 | — | — | 0 |
| apache | T5 | 3 | 3 | 3 | 3 | 2 | 67% | 2 | 0 | 1 | 0 | 0.00 | — | 0 |
| apache | T6 | 3 | 1 | 1 | 1 | 1 | 33% | 1 | 0 | 0 | 0 | — | — | 0 |

## By domain and request

| domain | request | attempted | executable | schema valid | integrity valid | successful | delivery rate | conditional success | fabricated values | shuffled failures (valid) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cars | CAR-11 | 18 | 14 | 14 | 14 | 8 | 44% | 57% | 0 | 4 |
| cars | CAR-12 | 18 | 14 | 14 | 14 | 4 | 22% | 29% | 0 | 12 |
| nasa | NASA-11 | 18 | 13 | 10 | 9 | 5 | 28% | 56% | 1 | 3 |
| nasa | NASA-12 | 18 | 13 | 10 | 10 | 4 | 22% | 40% | 0 | 5 |
| apache | APACHE-11 | 18 | 15 | 14 | 14 | 14 | 78% | 100% | 0 | 0 |
| apache | APACHE-12 | 18 | 15 | 12 | 12 | 8 | 44% | 67% | 0 | 4 |

## Limitations

- **Truthfulness, not usefulness.** These metrics check claims against recorded values and the frozen similarity definition; nobody rated whether an explanation helps a customer.
- **Trivial pairs.** If the preferred row dominates the other, the true trade-off set is empty and an empty claim is "exact" by default. `nontrivial_pairs_n` shows how many pairs really had trade-offs; recall is blank for the rest.
- **Apache has one goal**, so its trade-off set is either empty or {Performance-}.
- **The program picks the pair and the recommended row.** An easy choice makes truthfulness easier.
- **d2h is one overall score**; the frozen checker accepts any of three aggregates for "good".
- **NASA-11 uses one held-out project** with no leakage rerun; its error is reported but not used for success.
- **Three trials per cell, often fewer usable.** No statistical test is applied.
- **T7 is not included.** NASA and Apache T7 have Justify adapters (`t7_validation/adapters/{nasa,apache}/justify.py`) that answer these requests under the common protocol; Cars T7 has none. Scoring them would need a separate, approved run of those adapters, as in phase 4B.

## Provenance

- Archive: `harness/results/v2/gpt-oss-20b-3trials.zip` (sha256 `d1a51a3fa3711829ede726dd08e1f77da070fc3e3064fdcd1971d24974eb7683`)
- Code commit: `884ae8ad6bbc7d33bcfc853d620b58e9b6f6ccdd` on `analysis/external-metrics` (uncommitted changes present)
- Command: `python -m harness.external`
