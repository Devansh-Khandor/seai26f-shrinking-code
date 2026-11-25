# External metric, phase 5C: Fix — do the suggested alternatives really do better?

**Status: post-observation analysis.** Added after the Task Talk; not part of the pre-registered plan in `docs/hypotheses.md`. Every success rule below is an analysis choice unless it repeats a frozen checker rule.

**What is measured — and what is not.** All four measurable requests answer with an **existing row** of the CSV as the alternative. The improvement is therefore the difference between two *recorded* vehicles or configurations. It shows that a better observed option with the stated differences exists; it is **not** evidence that changing a real vehicle or server in that way would produce the improvement.

**Visibility.** Fix is retrospective: the program may see every goal value. There is no hidden target, so no leakage gate applies; the shuffled rerun is recorded but does not gate. Integrity here means the answer parses, passes the frozen format check, and cites only values that are really in the file.

## Metric definitions

| request | frozen task | improvement | success (post-hoc unless noted) |
|---|---|---|---|
| CAR-15 | similar vehicles with better recorded performance | d2h(row) − d2h(best listed alternative); lower d2h is better | row in the worse half by d2h, and ≥ 1 listed alternative is better by d2h **and** among the 25% most similar rows (the frozen similarity rule) |
| CAR-16 | closest alternative meeting more preferred targets, fewest changes | targets met (row → alternative) under the program's own thresholds; d2h change shown for context | thresholds in the observed range, row misses ≥ 1, alternative meets more, and it changes no more input columns than the minimum over all rows that meet more (computed from the CSV) |
| APACHE-15 | a small setting change with lower performance | Performance-(row) − Performance-(alternative); share of the best improvement available among rows 1–2 settings away | row in the worse half, alternative 1–2 settings away, improvement > 0 |
| APACHE-16 | fewest setting changes toward a performance target | \|row − target\| − \|alternative − target\| (program's own target) | target better than the row and in range, alternative closer, and no more changed settings than the minimum over all rows closer to the target |
| NASA-15/16 | changes that reduce *predicted* effort/defects | — | **not independently measurable**: the answer is a hypothetical change with a predicted outcome; no observed outcome exists to check it |

## Results by domain, treatment and request

Each cell has 3 attempted trials. Medians are over integrity-valid trials. "—" = no value.

| domain | T | request | attempted | executable | schema valid | integrity valid | successful | delivery rate | median improvement | median excess changes | median share of best captured |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cars | T1 | CAR-15 | 3 | 3 | 3 | 3 | 3 | 100% | 0.119 | — | — |
| cars | T2 | CAR-15 | 3 | 1 | 1 | 1 | 1 | 33% | 0.119 | — | — |
| cars | T3 | CAR-15 | 3 | 2 | 2 | 2 | 2 | 67% | 0.025 | — | — |
| cars | T4 | CAR-15 | 3 | 2 | 1 | 1 | 1 | 33% | 0.030 | — | — |
| cars | T5 | CAR-15 | 3 | 3 | 2 | 2 | 2 | 67% | 0.025 | — | — |
| cars | T6 | CAR-15 | 3 | 0 | 0 | 0 | 0 | 0% | — | — | — |
| cars | T1 | CAR-16 | 3 | 3 | 3 | 3 | 1 | 33% | 0.381 | 1.0 | — |
| cars | T2 | CAR-16 | 3 | 1 | 1 | 1 | 1 | 33% | 0.354 | 0.0 | — |
| cars | T3 | CAR-16 | 3 | 2 | 2 | 2 | 2 | 67% | 0.278 | 0.0 | — |
| cars | T4 | CAR-16 | 3 | 2 | 2 | 2 | 1 | 33% | 0.397 | 1.5 | — |
| cars | T5 | CAR-16 | 3 | 3 | 3 | 3 | 1 | 33% | 0.486 | 1.0 | — |
| cars | T6 | CAR-16 | 3 | 0 | 0 | 0 | 0 | 0% | — | — | — |
| nasa | T1 | NASA-15 | 3 | 2 | 1 | — | not measurable | — | — | — | — |
| nasa | T2 | NASA-15 | 3 | 3 | 1 | — | not measurable | — | — | — | — |
| nasa | T3 | NASA-15 | 3 | 2 | 1 | — | not measurable | — | — | — | — |
| nasa | T4 | NASA-15 | 3 | 1 | 0 | — | not measurable | — | — | — | — |
| nasa | T5 | NASA-15 | 3 | 2 | 1 | — | not measurable | — | — | — | — |
| nasa | T6 | NASA-15 | 3 | 0 | 0 | — | not measurable | — | — | — | — |
| nasa | T1 | NASA-16 | 3 | 2 | 1 | — | not measurable | — | — | — | — |
| nasa | T2 | NASA-16 | 3 | 3 | 1 | — | not measurable | — | — | — | — |
| nasa | T3 | NASA-16 | 3 | 2 | 1 | — | not measurable | — | — | — | — |
| nasa | T4 | NASA-16 | 3 | 1 | 0 | — | not measurable | — | — | — | — |
| nasa | T5 | NASA-16 | 3 | 2 | 1 | — | not measurable | — | — | — | — |
| nasa | T6 | NASA-16 | 3 | 0 | 0 | — | not measurable | — | — | — | — |
| apache | T1 | APACHE-15 | 3 | 3 | 2 | 2 | 2 | 67% | 1500.0 | — | 1.00 |
| apache | T2 | APACHE-15 | 3 | 3 | 3 | 3 | 3 | 100% | 1500.0 | — | 1.00 |
| apache | T3 | APACHE-15 | 3 | 3 | 3 | 3 | 3 | 100% | 1500.0 | — | 1.00 |
| apache | T4 | APACHE-15 | 3 | 3 | 2 | 2 | 2 | 67% | 1230.0 | — | 0.82 |
| apache | T5 | APACHE-15 | 3 | 3 | 3 | 3 | 3 | 100% | 1500.0 | — | 1.00 |
| apache | T6 | APACHE-15 | 3 | 1 | 1 | 1 | 1 | 33% | 1170.0 | — | 0.78 |
| apache | T1 | APACHE-16 | 3 | 3 | 2 | 2 | 0 | 0% | 1800.0 | 5.0 | — |
| apache | T2 | APACHE-16 | 3 | 3 | 3 | 3 | 2 | 67% | 1170.0 | 0.0 | — |
| apache | T3 | APACHE-16 | 3 | 3 | 3 | 3 | 3 | 100% | 1170.0 | 0.0 | — |
| apache | T4 | APACHE-16 | 3 | 3 | 2 | 2 | 2 | 67% | 1170.0 | 0.0 | — |
| apache | T5 | APACHE-16 | 3 | 3 | 3 | 3 | 3 | 100% | 1170.0 | 0.0 | — |
| apache | T6 | APACHE-16 | 3 | 1 | 1 | 1 | 1 | 33% | 1170.0 | 0.0 | — |

Improvement units: CAR-15: Δd2h; CAR-16: Δd2h (context); APACHE-15: Δ Performance-; APACHE-16: gap reduction.

## By domain and request

| domain | request | attempted | executable | schema valid | integrity valid | successful | delivery rate | conditional success | fabricated values | shuffled failures (valid) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cars | CAR-15 | 18 | 11 | 9 | 9 | 9 | 50% | 100% | 0 | 0 |
| cars | CAR-16 | 18 | 11 | 11 | 11 | 6 | 33% | 55% | 0 | 5 |
| nasa | NASA-15 | 18 | 10 | 4 | — | not measurable | — | — | 0 | 0 |
| nasa | NASA-16 | 18 | 10 | 4 | — | not measurable | — | — | 0 | 1 |
| apache | APACHE-15 | 18 | 16 | 14 | 14 | 14 | 78% | 100% | 0 | 0 |
| apache | APACHE-16 | 18 | 16 | 14 | 14 | 11 | 61% | 79% | 0 | 3 |

## Per-trial values

| domain | T | trial | request | outcome | reference row | candidate row | improvement | changes | minimum | success |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| cars | T1 | 1 | CAR-15 | included | 11 | 38 | 0.119 | 1 | — | True |
| cars | T1 | 2 | CAR-15 | included | 11 | 5 | 0.030 | 2 | — | True |
| cars | T1 | 3 | CAR-15 | included | 11 | 38 | 0.119 | 1 | — | True |
| cars | T2 | 1 | CAR-15 | generation_failed | — | — | — | — | — | — |
| cars | T2 | 2 | CAR-15 | included | 11 | 38 | 0.119 | 1 | — | True |
| cars | T2 | 3 | CAR-15 | execution_failure | — | — | — | — | — | — |
| cars | T3 | 1 | CAR-15 | included | 11 | 5 | 0.030 | 2 | — | True |
| cars | T3 | 2 | CAR-15 | included | 11 | 25 | 0.020 | 1 | — | True |
| cars | T3 | 3 | CAR-15 | generation_failed | — | — | — | — | — | — |
| cars | T4 | 1 | CAR-15 | execution_failure | — | — | — | — | — | — |
| cars | T4 | 2 | CAR-15 | included | 11 | 5 | 0.030 | 2 | — | True |
| cars | T4 | 3 | CAR-15 | answer_schema_error | — | — | — | — | — | — |
| cars | T5 | 1 | CAR-15 | answer_schema_error | — | — | — | — | — | — |
| cars | T5 | 2 | CAR-15 | included | 11 | 5 | 0.030 | 2 | — | True |
| cars | T5 | 3 | CAR-15 | included | 11 | 25 | 0.020 | 1 | — | True |
| cars | T6 | 1 | CAR-15 | execution_failure | — | — | — | — | — | — |
| cars | T6 | 2 | CAR-15 | execution_failure | — | — | — | — | — | — |
| cars | T6 | 3 | CAR-15 | execution_failure | — | — | — | — | — | — |
| cars | T1 | 1 | CAR-16 | included | 11 | 1 | 0.125 | 2 | 2 | True |
| cars | T1 | 2 | CAR-16 | included | 1 | 230 | 0.381 | 3 | 2 | False |
| cars | T1 | 3 | CAR-16 | included | 1 | 230 | 0.381 | 3 | 2 | False |
| cars | T2 | 1 | CAR-16 | generation_failed | — | — | — | — | — | — |
| cars | T2 | 2 | CAR-16 | included | 11 | 116 | 0.354 | 2 | 2 | True |
| cars | T2 | 3 | CAR-16 | execution_failure | — | — | — | — | — | — |
| cars | T3 | 1 | CAR-16 | included | 1 | 172 | 0.249 | 2 | 2 | True |
| cars | T3 | 2 | CAR-16 | included | 1 | 264 | 0.308 | 2 | 2 | True |
| cars | T3 | 3 | CAR-16 | generation_failed | — | — | — | — | — | — |
| cars | T4 | 1 | CAR-16 | execution_failure | — | — | — | — | — | — |
| cars | T4 | 2 | CAR-16 | included | 2 | 1 | 0.044 | 1 | 1 | True |
| cars | T4 | 3 | CAR-16 | included | 35 | 395 | 0.750 | 4 | 1 | False |
| cars | T5 | 1 | CAR-16 | included | 15 | 276 | 0.540 | 4 | 3 | False |
| cars | T5 | 2 | CAR-16 | included | 11 | 78 | 0.117 | 1 | 1 | True |
| cars | T5 | 3 | CAR-16 | included | 2 | 233 | 0.486 | 3 | 1 | False |
| cars | T6 | 1 | CAR-16 | execution_failure | — | — | — | — | — | — |
| cars | T6 | 2 | CAR-16 | execution_failure | — | — | — | — | — | — |
| cars | T6 | 3 | CAR-16 | execution_failure | — | — | — | — | — | — |
| apache | T1 | 1 | APACHE-15 | included | 18 | 21 | 1500.000 | 2 | — | True |
| apache | T1 | 2 | APACHE-15 | answer_schema_error | — | — | — | — | — | — |
| apache | T1 | 3 | APACHE-15 | included | 18 | 21 | 1500.000 | 2 | — | True |
| apache | T2 | 1 | APACHE-15 | included | 18 | 21 | 1500.000 | 2 | — | True |
| apache | T2 | 2 | APACHE-15 | included | 2 | 137 | 1140.000 | 2 | — | True |
| apache | T2 | 3 | APACHE-15 | included | 18 | 21 | 1500.000 | 2 | — | True |
| apache | T3 | 1 | APACHE-15 | included | 18 | 17 | 1170.000 | 1 | — | True |
| apache | T3 | 2 | APACHE-15 | included | 18 | 21 | 1500.000 | 2 | — | True |
| apache | T3 | 3 | APACHE-15 | included | 18 | 21 | 1500.000 | 2 | — | True |
| apache | T4 | 1 | APACHE-15 | included | 18 | 17 | 1170.000 | 1 | — | True |
| apache | T4 | 2 | APACHE-15 | included | 18 | 107 | 1290.000 | 2 | — | True |
| apache | T4 | 3 | APACHE-15 | answer_schema_error | — | — | — | — | — | — |
| apache | T5 | 1 | APACHE-15 | included | 18 | 21 | 1500.000 | 2 | — | True |
| apache | T5 | 2 | APACHE-15 | included | 18 | 21 | 1500.000 | 2 | — | True |
| apache | T5 | 3 | APACHE-15 | included | 18 | 21 | 1500.000 | 2 | — | True |
| apache | T6 | 1 | APACHE-15 | included | 18 | 17 | 1170.000 | 1 | — | True |
| apache | T6 | 2 | APACHE-15 | generation_failed | — | — | — | — | — | — |
| apache | T6 | 3 | APACHE-15 | execution_failure | — | — | — | — | — | — |
| apache | T1 | 1 | APACHE-16 | included | 18 | 151 | 1800.000 | 6 | 1 | False |
| apache | T1 | 2 | APACHE-16 | answer_schema_error | — | — | — | — | — | — |
| apache | T1 | 3 | APACHE-16 | included | 18 | 151 | 1800.000 | 6 | 1 | False |
| apache | T2 | 1 | APACHE-16 | included | 18 | 17 | 1170.000 | 1 | 1 | True |
| apache | T2 | 2 | APACHE-16 | included | 2 | 182 | 900.000 | 1 | 1 | True |
| apache | T2 | 3 | APACHE-16 | included | 18 | 17 | 1170.000 | 1 | 1 | False |
| apache | T3 | 1 | APACHE-16 | included | 18 | 17 | 1170.000 | 1 | 1 | True |
| apache | T3 | 2 | APACHE-16 | included | 18 | 17 | 1170.000 | 1 | 1 | True |
| apache | T3 | 3 | APACHE-16 | included | 18 | 17 | 1170.000 | 1 | 1 | True |
| apache | T4 | 1 | APACHE-16 | included | 18 | 17 | 1170.000 | 1 | 1 | True |
| apache | T4 | 2 | APACHE-16 | included | 18 | 17 | 1170.000 | 1 | 1 | True |
| apache | T4 | 3 | APACHE-16 | answer_missing | — | — | — | — | — | — |
| apache | T5 | 1 | APACHE-16 | included | 18 | 17 | 1170.000 | 1 | 1 | True |
| apache | T5 | 2 | APACHE-16 | included | 18 | 17 | 1170.000 | 1 | 1 | True |
| apache | T5 | 3 | APACHE-16 | included | 18 | 17 | 1170.000 | 1 | 1 | True |
| apache | T6 | 1 | APACHE-16 | included | 18 | 17 | 1170.000 | 1 | 1 | True |
| apache | T6 | 2 | APACHE-16 | generation_failed | — | — | — | — | — | — |
| apache | T6 | 3 | APACHE-16 | execution_failure | — | — | — | — | — | — |

## Limitations

- **Observed alternatives, not interventions.** A better recorded row that differs in a few columns is association, not the causal effect of making those changes.
- **Self-chosen targets.** In CAR-16 and APACHE-16 the program sets its own thresholds or target (frozen format). Easy targets make success easier; the target values are in `fix_quality.csv`.
- **The program picks the reference row.** A program can choose an easy poor row; `reference_poor` and the reference values are recorded so this is visible.
- **d2h is one overall score.** The frozen checker also accepts z-score and rank sums for CAR-15.
- **Three trials per cell, often fewer usable.** No statistical test is applied.
- **T7 is not included.** No Cars adapter exists for Fix, and the APACHE-16 T7 adapter optimizes a different objective from the original T7 code; it must not be pooled with T1–T6.

## Provenance

- Archive: `harness/results/v2/gpt-oss-20b-3trials.zip` (sha256 `d1a51a3fa3711829ede726dd08e1f77da070fc3e3064fdcd1971d24974eb7683`)
- Code commit: `6f580c8eaaf527c4292a03985ec17032268dc418` on `analysis/external-metrics` (uncommitted changes present)
- Command: `python -m harness.external`
