# External metric, phase 1: prediction MAE vs the historical-mean baseline

**Status: post-observation analysis.** This metric was added after the Task Talk, after the v2 T1–T6 results had been seen. It is not part of the pre-registered plan in `docs/hypotheses.md` and does not change its win conditions.

## What is measured

- **Requests:** CAR-03 (`Mpg+`), NASA-03 (`Effort-`), APACHE-03 (`Performance-`): the Guess step (skill 2) of every v2 trial of T1–T6, model `gpt-oss-20b`.
- **Held-out rows:** the frozen protocol (`harness.oracle.held_out`): every row whose number is divisible by 5. Cars 79 rows, NASA 18, Apache 38.
- **Model MAE:** recomputed from the program's recorded held-out predictions against the CSV. The program's own reported `mae` is not used (it is kept in `reported_mae` for reference).
- **Baseline MAE:** the mean target value of the non-held-out rows, scored on the same held-out rows. This is the baseline the frozen checker uses for "beats the historical mean".
- **MAE ratio** = model MAE / baseline MAE. Below 1 means the program beats the baseline. MAE is an error, so lower is better whatever the goal's direction.
- **Source:** archived base-run output only. No generated program was rerun or retrained, and no LLM was called.

## Inclusion rules

Every attempted trial is a row in `prediction_mae.csv`. A row enters the primary metric only if:

1. the step produced a program and it ran on the real CSV,
2. it printed an `ANSWER` line for the request that the frozen checker accepts (predictions for exactly the held-out rows), and
3. the archived `mask` rerun shows the predictions did not move when only the held-out target values were changed (`leakage_gate = pass`).

A failed leakage check excludes the row (`leakage_failed`). So does a mask rerun that crashed, timed out or gave no valid answer (`leakage_unverified`): leakage could not be ruled out. This second rule is a choice made for this analysis. Schema-valid rows excluded by the gate still show their MAE in the CSV for transparency, but never count in the summaries.

The shuffled-rows rerun is recorded (`shuffled_gate`) but does not gate this metric.

Two rates are reported, and they answer different questions:

- **Conditional accuracy:** the median MAE ratio over primary rows only. It says how good the predictions are *when* a treatment delivers them.
- **Delivery rate:** primary rows that beat the baseline ÷ all attempted trials. Failed, missing and excluded attempts count as not delivered, so a treatment cannot look better by crashing on hard trials.

## Results by domain and treatment

n = trials. Each cell has 3 attempted trials. "—" means no primary value.

| domain | treatment | attempted | gen. failed | exec. failed | no / bad answer | schema valid | leakage failed | leakage unverified | primary n | median MAE ratio | min–max ratio | beats baseline | delivery rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| cars | T1 | 3 | 0 | 2 | 0 | 1 | 0 | 0 | 1 | 0.502 | 0.502–0.502 | 1/1 | 33% |
| cars | T2 | 3 | 0 | 1 | 0 | 2 | 0 | 0 | 2 | 0.635 | 0.502–0.769 | 2/2 | 67% |
| cars | T3 | 3 | 0 | 1 | 0 | 2 | 0 | 0 | 2 | 0.563 | 0.559–0.567 | 2/2 | 67% |
| cars | T4 | 3 | 0 | 1 | 0 | 2 | 1 | 0 | 1 | 0.555 | 0.555–0.555 | 1/1 | 33% |
| cars | T5 | 3 | 0 | 1 | 0 | 2 | 0 | 0 | 2 | 1.453 | 0.559–2.348 | 1/2 | 33% |
| cars | T6 | 3 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | — | — | 0/0 | 0% |
| nasa | T1 | 3 | 0 | 1 | 0 | 2 | 0 | 0 | 2 | 0.814 | 0.801–0.827 | 2/2 | 67% |
| nasa | T2 | 3 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | — | — | 0/0 | 0% |
| nasa | T3 | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | — | — | 0/0 | 0% |
| nasa | T4 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 3 | 0.827 | 0.712–0.832 | 3/3 | 100% |
| nasa | T5 | 3 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0.827 | 0.827–0.827 | 1/1 | 33% |
| nasa | T6 | 3 | 0 | 1 | 0 | 2 | 0 | 0 | 2 | 0.883 | 0.765–1.000 | 1/2 | 33% |
| apache | T1 | 3 | 0 | 0 | 0 | 3 | 1 | 0 | 2 | 0.689 | 0.451–0.927 | 2/2 | 67% |
| apache | T2 | 3 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1.000 | 1.000–1.000 | 0/1 | 0% |
| apache | T3 | 3 | 0 | 0 | 0 | 3 | 1 | 0 | 2 | 0.451 | 0.451–0.451 | 2/2 | 67% |
| apache | T4 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 3 | 0.451 | 0.451–0.451 | 3/3 | 100% |
| apache | T5 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 3 | 0.484 | 0.451–0.603 | 3/3 | 100% |
| apache | T6 | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | — | — | 0/0 | 0% |

## By domain

| domain | request | baseline MAE | attempted | primary n | median MAE ratio | median model MAE | delivery rate |
|---|---|---:|---:|---:|---:|---:|---:|
| cars | CAR-03 | 6.988 | 18 | 8 | 0.559 | 3.905 | 39% |
| nasa | NASA-03 | 885.089 | 18 | 8 | 0.827 | 731.770 | 39% |
| apache | APACHE-03 | 464.590 | 18 | 11 | 0.451 | 209.715 | 56% |

## By treatment, all three domains

MAE ratios are unitless, so they can be listed together, but the three domains have different targets and sample sizes. Treat this table as descriptive.

| treatment | attempted | primary n | median MAE ratio | beats baseline | delivery rate |
|---|---:|---:|---:|---:|---:|
| T1 | 9 | 5 | 0.801 | 5/5 | 56% |
| T2 | 9 | 3 | 0.769 | 2/3 | 22% |
| T3 | 9 | 4 | 0.505 | 4/4 | 44% |
| T4 | 9 | 7 | 0.555 | 7/7 | 78% |
| T5 | 9 | 6 | 0.581 | 5/6 | 56% |
| T6 | 9 | 2 | 0.883 | 1/2 | 11% |

## Per-trial values

| domain | treatment | trial | exclusion reason | model MAE | baseline MAE | MAE ratio | leakage | shuffled |
|---|---|---:|---|---:|---:|---:|---|---|
| cars | T1 | 1 | execution_failure | — | 6.988 | — | not_run | not_run |
| cars | T1 | 2 | included | 3.508 | 6.988 | 0.502 | pass | pass |
| cars | T1 | 3 | execution_failure | — | 6.988 | — | not_run | not_run |
| cars | T2 | 1 | execution_failure | — | 6.988 | — | not_run | not_run |
| cars | T2 | 2 | included | 3.508 | 6.988 | 0.502 | pass | pass |
| cars | T2 | 3 | included | 5.371 | 6.988 | 0.769 | pass | pass |
| cars | T3 | 1 | included | 3.964 | 6.988 | 0.567 | pass | pass |
| cars | T3 | 2 | included | 3.905 | 6.988 | 0.559 | pass | pass |
| cars | T3 | 3 | execution_failure | — | 6.988 | — | not_run | not_run |
| cars | T4 | 1 | included | 3.881 | 6.988 | 0.555 | pass | pass |
| cars | T4 | 2 | leakage_failed | 3.481 | 6.988 | 0.498 | fail | pass |
| cars | T4 | 3 | execution_failure | — | 6.988 | — | not_run | not_run |
| cars | T5 | 1 | execution_failure | — | 6.988 | — | not_run | not_run |
| cars | T5 | 2 | included | 16.406 | 6.988 | 2.348 | pass | fail |
| cars | T5 | 3 | included | 3.905 | 6.988 | 0.559 | pass | pass |
| cars | T6 | 1 | execution_failure | — | 6.988 | — | not_run | not_run |
| cars | T6 | 2 | answer_schema_error | — | 6.988 | — | not_run | not_run |
| cars | T6 | 3 | execution_failure | — | 6.988 | — | not_run | not_run |
| nasa | T1 | 1 | execution_failure | — | 885.089 | — | not_run | not_run |
| nasa | T1 | 2 | included | 709.112 | 885.089 | 0.801 | pass | pass |
| nasa | T1 | 3 | included | 731.770 | 885.089 | 0.827 | pass | pass |
| nasa | T2 | 1 | generation_failed | — | 885.089 | — | not_run | not_run |
| nasa | T2 | 2 | execution_failure | — | 885.089 | — | not_run | not_run |
| nasa | T2 | 3 | answer_invalid_json | — | 885.089 | — | not_run | not_run |
| nasa | T3 | 1 | execution_failure | — | 885.089 | — | not_run | not_run |
| nasa | T3 | 2 | execution_failure | — | 885.089 | — | not_run | not_run |
| nasa | T3 | 3 | execution_failure | — | 885.089 | — | not_run | not_run |
| nasa | T4 | 1 | included | 731.770 | 885.089 | 0.827 | pass | pass |
| nasa | T4 | 2 | included | 736.207 | 885.089 | 0.832 | pass | fail |
| nasa | T4 | 3 | included | 630.268 | 885.089 | 0.712 | pass | pass |
| nasa | T5 | 1 | included | 731.770 | 885.089 | 0.827 | pass | pass |
| nasa | T5 | 2 | answer_schema_error | — | 885.089 | — | not_run | not_run |
| nasa | T5 | 3 | execution_failure | — | 885.089 | — | not_run | not_run |
| nasa | T6 | 1 | included | 677.172 | 885.089 | 0.765 | pass | pass |
| nasa | T6 | 2 | execution_failure | — | 885.089 | — | not_run | not_run |
| nasa | T6 | 3 | included | 885.089 | 885.089 | 1.000 | pass | fail |
| apache | T1 | 1 | leakage_failed | 194.398 | 464.590 | 0.418 | fail | pass |
| apache | T1 | 2 | included | 430.490 | 464.590 | 0.927 | pass | pass |
| apache | T1 | 3 | included | 209.711 | 464.590 | 0.451 | pass | pass |
| apache | T2 | 1 | included | 464.590 | 464.590 | 1.000 | pass | fail |
| apache | T2 | 2 | answer_schema_error | — | 464.590 | — | not_run | not_run |
| apache | T2 | 3 | execution_failure | — | 464.590 | — | not_run | not_run |
| apache | T3 | 1 | included | 209.715 | 464.590 | 0.451 | pass | pass |
| apache | T3 | 2 | leakage_failed | 20.309 | 464.590 | 0.044 | fail | pass |
| apache | T3 | 3 | included | 209.715 | 464.590 | 0.451 | pass | pass |
| apache | T4 | 1 | included | 209.715 | 464.590 | 0.451 | pass | pass |
| apache | T4 | 2 | included | 209.715 | 464.590 | 0.451 | pass | pass |
| apache | T4 | 3 | included | 209.715 | 464.590 | 0.451 | pass | pass |
| apache | T5 | 1 | included | 225.000 | 464.590 | 0.484 | pass | pass |
| apache | T5 | 2 | included | 280.263 | 464.590 | 0.603 | pass | pass |
| apache | T5 | 3 | included | 209.715 | 464.590 | 0.451 | pass | pass |
| apache | T6 | 1 | generation_failed | — | 464.590 | — | not_run | not_run |
| apache | T6 | 2 | execution_failure | — | 464.590 | — | not_run | not_run |
| apache | T6 | 3 | execution_failure | — | 464.590 | — | not_run | not_run |

## Limitations

- **Three trials per cell, often fewer usable.** No statistical comparison between treatments is made in this phase; with n ≤ 3, `same()` would have very little power.
- **One fixed split.** The every-fifth-row hold-out is the frozen protocol shared by all treatments, so the comparison is fair, but there is no resampling of the split.
- **Baseline is per domain, not per trial.** It depends only on the CSV, so it is the same for every treatment in a domain.
- **Skill 2 only.** The Race request (CAR-20, NASA-19, APACHE-20) also reports MAE; it comes from the same trials and is not pooled with this one.
- **T7 is not included** in this phase.

## Provenance

- Archive: `harness/results/v2/gpt-oss-20b-3trials.zip` (sha256 `d1a51a3fa3711829ede726dd08e1f77da070fc3e3064fdcd1971d24974eb7683`)
- Code commit: `1318303c4841a311877ae6b38dfeef558bcec05f` on `analysis/external-metrics` (uncommitted changes present)
- Command: `python -m harness.external`
- Full input and output hashes: `manifest.json`.
