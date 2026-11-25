# External metric, phase 2: selection quality under a fixed inspection budget

**Status: post-observation analysis.** This metric was added after the Task Talk, after the v2 T1–T6 results had been seen. It is not part of the pre-registered plan in `docs/hypotheses.md` and does not change its win conditions.

## What is measured

- **Requests:** CAR-14, NASA-14, APACHE-14 (Choose, skill 7): "recommend which rows should be evaluated first" under a limited inspection budget. Every v2 trial of T1–T6, model `gpt-oss-20b`.
- **Recorded answer:** `{"order": [row numbers]}`, the rows the program would inspect, first one first. The frozen checker requires 10 to `budget` distinct rows, numbered 1..n in file order.
- **Budget and candidate pool:** the frozen `harness.oracle.budget(n)` = min(30, n // 5): Cars 30 of 398 rows, NASA 18 of 93, Apache 30 of 192. Every row of the CSV is a candidate.
- **Score:** the frozen `harness.oracle.d2h` (distance to heaven over all goals, goal direction from the column suffix; lower is better). It only grades rows the program already listed; it never builds or changes an order.
- **Best selected d2h:** the lowest d2h among the rows in the order.
- **Regret:** best selected d2h − best d2h in the whole pool. **Normalized regret:** regret ÷ (worst − best d2h in the pool); 0 = found the best row, 1 = found only the worst.
- **Advantage over random:** the exact expected best d2h of the same number of rows drawn uniformly at random, minus the best selected d2h. Positive = better than random. It is computed in closed form, so no random draws are simulated and none are counted as trials.
- **P(random ≥ as good):** the probability that the same number of random rows would include one at least as good as the program's best.
- **At 10 rows:** the same metrics for the first 10 rows of every order. Programs chose different order lengths (10 to the budget), and budgets differ by domain; this column compares everyone at the same inspection count.

## Inclusion rules

Every attempted trial is a row in `selection_quality.csv`. A row enters the primary metric only if the program ran, its `ANSWER` passed the frozen checker's format, and the archived **honeypot** rerun passed. The honeypot copy plants six rows with ideal goals but middling inputs; a program that puts them early in its order is reading goal values it has not paid to inspect (budget peeking).

- `honeypot_failed`: the frozen check "respects the budget" failed. Excluded.
- `honeypot_unverified`: the honeypot rerun crashed, timed out or gave no valid answer. Excluded (analysis choice: peeking could not be ruled out).
- The shuffled-rows rerun is recorded (`shuffled_gate`) but does not gate this metric.

**Delivery rate** = primary trials whose selection beats random (advantage > 0) ÷ all 3 attempted trials. "Beats random" (advantage > 0) is a threshold chosen for this analysis.

## Results by domain and treatment

Each cell has 3 attempted trials. "—" means no primary value. Medians are over primary trials only.

| domain | treatment | budget | attempted | exec. failed | no / bad answer | schema valid | honeypot failed | honeypot unverified | primary n | median best d2h | median regret | median normalized regret | median advantage vs random | median advantage at 10 | delivery rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cars | T1 | 30 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | — | — | — | — | — | 0% |
| cars | T2 | 30 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | — | — | — | — | — | 0% |
| cars | T3 | 30 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | — | — | — | — | — | 0% |
| cars | T4 | 30 | 3 | 1 | 0 | 2 | 1 | 0 | 1 | 0.310 | 0.143 | 0.188 | -0.044 | +0.002 | 0% |
| cars | T5 | 30 | 3 | 1 | 0 | 2 | 2 | 0 | 0 | — | — | — | — | — | 0% |
| cars | T6 | 30 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | — | — | — | — | — | 0% |
| nasa | T1 | 18 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | — | — | — | — | — | 0% |
| nasa | T2 | 18 | 3 | 0 | 0 | 3 | 2 | 0 | 1 | 0.387 | 0.000 | 0.000 | +0.037 | -0.086 | 33% |
| nasa | T3 | 18 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | — | — | — | — | — | 0% |
| nasa | T4 | 18 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | — | — | — | — | — | 0% |
| nasa | T5 | 18 | 3 | 1 | 0 | 2 | 1 | 0 | 1 | 0.520 | 0.133 | 0.386 | -0.096 | -0.093 | 0% |
| nasa | T6 | 18 | 3 | 1 | 0 | 2 | 1 | 0 | 1 | 0.520 | 0.133 | 0.386 | -0.096 | -0.093 | 0% |
| apache | T1 | 30 | 3 | 0 | 0 | 3 | 1 | 0 | 2 | 0.067 | 0.067 | 0.067 | -0.046 | -0.034 | 0% |
| apache | T2 | 30 | 3 | 0 | 0 | 3 | 1 | 0 | 2 | 0.067 | 0.067 | 0.067 | -0.046 | -0.034 | 0% |
| apache | T3 | 30 | 3 | 0 | 0 | 3 | 0 | 0 | 3 | 0.000 | 0.000 | 0.000 | +0.021 | +0.049 | 100% |
| apache | T4 | 30 | 3 | 0 | 0 | 3 | 0 | 0 | 3 | 0.000 | 0.000 | 0.000 | +0.021 | +0.049 | 100% |
| apache | T5 | 30 | 3 | 1 | 0 | 2 | 0 | 0 | 2 | 0.033 | 0.033 | 0.033 | -0.013 | -0.042 | 33% |
| apache | T6 | 30 | 3 | 0 | 0 | 3 | 0 | 0 | 3 | 0.000 | 0.000 | 0.000 | +0.021 | -0.034 | 100% |

"exec. failed" includes steps that produced no program (`parse_failed`).

## By domain

| domain | request | budget | pool best d2h | attempted | schema valid | honeypot failed | primary n | median normalized regret | median advantage vs random | delivery rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cars | CAR-14 | 30 | 0.167 | 18 | 13 | 12 | 1 | 0.188 | -0.044 | 0% |
| nasa | NASA-14 | 18 | 0.387 | 18 | 16 | 13 | 3 | 0.386 | -0.096 | 6% |
| apache | APACHE-14 | 30 | 0.000 | 18 | 17 | 2 | 15 | 0.000 | +0.021 | 56% |

## By treatment, all three domains

Normalized regret and the advantage are on the d2h scale of each domain; the domains differ in size, goals and budget. Treat this table as descriptive.

| treatment | attempted | schema valid | honeypot failed | primary n | median normalized regret | median advantage at 10 | delivery rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 9 | 9 | 7 | 2 | 0.067 | -0.034 | 0% |
| T2 | 9 | 9 | 6 | 3 | 0.067 | -0.034 | 11% |
| T3 | 9 | 9 | 6 | 3 | 0.000 | +0.049 | 33% |
| T4 | 9 | 8 | 4 | 4 | 0.000 | +0.049 | 33% |
| T5 | 9 | 6 | 3 | 3 | 0.067 | -0.051 | 11% |
| T6 | 9 | 5 | 1 | 4 | 0.000 | -0.034 | 33% |

## Budget peeking

27 of the 46 schema-valid orders failed the honeypot check. Their selections are kept in the CSV (`primary = false`) but excluded from every summary above. For context only, the table compares their scores on the real data with the primary orders. A program that reads unpaid goal values can look better than an honest one, so the excluded values do not measure selection skill.

| domain | schema valid | honeypot failed | median normalized regret, excluded | median normalized regret, primary | median advantage vs random, excluded | median advantage vs random, primary |
|---|---:|---:|---:|---:|---:|---:|
| cars | 13 | 12 | 0.000 | 0.188 | +0.099 | -0.044 |
| nasa | 16 | 13 | 0.498 | 0.386 | -0.135 | -0.096 |
| apache | 17 | 2 | 0.000 | 0.000 | +0.021 | +0.021 |

## Per-trial values

| domain | treatment | trial | exclusion reason | inspected | best row | best d2h | regret | normalized regret | expected random best | advantage | P(random ≥ as good) | honeypot |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cars | T1 | 1 | honeypot_failed | 30 | 395 | 0.167 | 0.000 | 0.000 | 0.266 | +0.099 | 0.08 | fail |
| cars | T1 | 2 | honeypot_failed | 30 | 395 | 0.167 | 0.000 | 0.000 | 0.266 | +0.099 | 0.08 | fail |
| cars | T1 | 3 | honeypot_failed | 30 | 395 | 0.167 | 0.000 | 0.000 | 0.266 | +0.099 | 0.08 | fail |
| cars | T2 | 1 | honeypot_failed | 30 | 395 | 0.167 | 0.000 | 0.000 | 0.266 | +0.099 | 0.08 | fail |
| cars | T2 | 2 | honeypot_failed | 30 | 395 | 0.167 | 0.000 | 0.000 | 0.266 | +0.099 | 0.08 | fail |
| cars | T2 | 3 | honeypot_failed | 30 | 386 | 0.258 | 0.091 | 0.120 | 0.266 | +0.008 | 0.47 | fail |
| cars | T3 | 1 | honeypot_failed | 30 | 395 | 0.167 | 0.000 | 0.000 | 0.266 | +0.099 | 0.08 | fail |
| cars | T3 | 2 | honeypot_failed | 30 | 395 | 0.167 | 0.000 | 0.000 | 0.266 | +0.099 | 0.08 | fail |
| cars | T3 | 3 | honeypot_failed | 30 | 395 | 0.167 | 0.000 | 0.000 | 0.266 | +0.099 | 0.08 | fail |
| cars | T4 | 1 | included | 30 | 298 | 0.310 | 0.143 | 0.188 | 0.266 | -0.044 | 0.74 | pass |
| cars | T4 | 2 | honeypot_failed | 30 | 395 | 0.167 | 0.000 | 0.000 | 0.266 | +0.099 | 0.08 | fail |
| cars | T4 | 3 | execution_failure | — | — | — | — | — | — | — | — | not_run |
| cars | T5 | 1 | honeypot_failed | 30 | 395 | 0.167 | 0.000 | 0.000 | 0.266 | +0.099 | 0.08 | fail |
| cars | T5 | 2 | honeypot_failed | 30 | 395 | 0.167 | 0.000 | 0.000 | 0.266 | +0.099 | 0.08 | fail |
| cars | T5 | 3 | execution_failure | — | — | — | — | — | — | — | — | not_run |
| cars | T6 | 1 | execution_failure | — | — | — | — | — | — | — | — | not_run |
| cars | T6 | 2 | execution_failure | — | — | — | — | — | — | — | — | not_run |
| cars | T6 | 3 | execution_failure | — | — | — | — | — | — | — | — | not_run |
| nasa | T1 | 1 | honeypot_failed | 18 | 60 | 0.387 | 0.000 | 0.000 | 0.425 | +0.037 | 0.19 | fail |
| nasa | T1 | 2 | honeypot_failed | 18 | 19 | 0.569 | 0.182 | 0.526 | 0.425 | -0.145 | 1.00 | fail |
| nasa | T1 | 3 | honeypot_failed | 18 | 60 | 0.387 | 0.000 | 0.000 | 0.425 | +0.037 | 0.19 | fail |
| nasa | T2 | 1 | included | 18 | 60 | 0.387 | 0.000 | 0.000 | 0.425 | +0.037 | 0.19 | pass |
| nasa | T2 | 2 | honeypot_failed | 18 | 60 | 0.387 | 0.000 | 0.000 | 0.425 | +0.037 | 0.19 | fail |
| nasa | T2 | 3 | honeypot_failed | 18 | 19 | 0.569 | 0.182 | 0.526 | 0.425 | -0.145 | 1.00 | fail |
| nasa | T3 | 1 | honeypot_failed | 18 | 19 | 0.569 | 0.182 | 0.526 | 0.425 | -0.145 | 1.00 | fail |
| nasa | T3 | 2 | honeypot_failed | 18 | 19 | 0.569 | 0.182 | 0.526 | 0.425 | -0.145 | 1.00 | fail |
| nasa | T3 | 3 | honeypot_failed | 18 | 60 | 0.387 | 0.000 | 0.000 | 0.425 | +0.037 | 0.19 | fail |
| nasa | T4 | 1 | honeypot_failed | 18 | 60 | 0.387 | 0.000 | 0.000 | 0.425 | +0.037 | 0.19 | fail |
| nasa | T4 | 2 | honeypot_failed | 18 | 19 | 0.569 | 0.182 | 0.526 | 0.425 | -0.145 | 1.00 | fail |
| nasa | T4 | 3 | honeypot_failed | 18 | 18 | 0.559 | 0.172 | 0.498 | 0.425 | -0.135 | 1.00 | fail |
| nasa | T5 | 1 | execution_failure | — | — | — | — | — | — | — | — | not_run |
| nasa | T5 | 2 | honeypot_failed | 18 | 19 | 0.569 | 0.182 | 0.526 | 0.425 | -0.145 | 1.00 | fail |
| nasa | T5 | 3 | included | 18 | 12 | 0.520 | 0.133 | 0.386 | 0.425 | -0.096 | 1.00 | pass |
| nasa | T6 | 1 | execution_failure | — | — | — | — | — | — | — | — | not_run |
| nasa | T6 | 2 | honeypot_failed | 18 | 60 | 0.387 | 0.000 | 0.000 | 0.425 | +0.037 | 0.19 | fail |
| nasa | T6 | 3 | included | 18 | 12 | 0.520 | 0.133 | 0.386 | 0.425 | -0.096 | 1.00 | pass |
| apache | T1 | 1 | included | 30 | 14 | 0.067 | 0.067 | 0.067 | 0.021 | -0.046 | 0.99 | pass |
| apache | T1 | 2 | honeypot_failed | 30 | 125 | 0.000 | 0.000 | 0.000 | 0.021 | +0.021 | 0.29 | fail |
| apache | T1 | 3 | included | 30 | 14 | 0.067 | 0.067 | 0.067 | 0.021 | -0.046 | 0.99 | pass |
| apache | T2 | 1 | honeypot_failed | 30 | 125 | 0.000 | 0.000 | 0.000 | 0.021 | +0.021 | 0.29 | fail |
| apache | T2 | 2 | included | 30 | 14 | 0.067 | 0.067 | 0.067 | 0.021 | -0.046 | 0.99 | pass |
| apache | T2 | 3 | included | 30 | 14 | 0.067 | 0.067 | 0.067 | 0.021 | -0.046 | 0.99 | pass |
| apache | T3 | 1 | included | 30 | 125 | 0.000 | 0.000 | 0.000 | 0.021 | +0.021 | 0.29 | pass |
| apache | T3 | 2 | included | 30 | 125 | 0.000 | 0.000 | 0.000 | 0.021 | +0.021 | 0.29 | pass |
| apache | T3 | 3 | included | 30 | 125 | 0.000 | 0.000 | 0.000 | 0.021 | +0.021 | 0.29 | pass |
| apache | T4 | 1 | included | 30 | 151 | 0.000 | 0.000 | 0.000 | 0.021 | +0.021 | 0.29 | pass |
| apache | T4 | 2 | included | 30 | 125 | 0.000 | 0.000 | 0.000 | 0.021 | +0.021 | 0.29 | pass |
| apache | T4 | 3 | included | 30 | 125 | 0.000 | 0.000 | 0.000 | 0.021 | +0.021 | 0.29 | pass |
| apache | T5 | 1 | included | 30 | 125 | 0.000 | 0.000 | 0.000 | 0.021 | +0.021 | 0.29 | pass |
| apache | T5 | 2 | execution_failure | — | — | — | — | — | — | — | — | not_run |
| apache | T5 | 3 | included | 30 | 14 | 0.067 | 0.067 | 0.067 | 0.021 | -0.046 | 0.99 | pass |
| apache | T6 | 1 | included | 30 | 125 | 0.000 | 0.000 | 0.000 | 0.021 | +0.021 | 0.29 | pass |
| apache | T6 | 2 | included | 30 | 125 | 0.000 | 0.000 | 0.000 | 0.021 | +0.021 | 0.29 | pass |
| apache | T6 | 3 | included | 30 | 125 | 0.000 | 0.000 | 0.000 | 0.021 | +0.021 | 0.29 | pass |

## Limitations

- **Three trials per cell, often fewer usable.** No statistical comparison between treatments is made; with n ≤ 3, `same()` would have very little power.
- **Budgets differ by domain** (30, 18, 30 rows) and pools differ in size (398, 93, 192), so a regret of the same size is not equally hard in every domain. The at-10 columns and normalized regret reduce, but do not remove, this difference.
- **The order is scored, not replayed.** The archive records only the final order. Whether each choice used only goals of rows already placed is known only through the honeypot gate.
- **The honeypot gate is a test, not a proof.** It checks the first budget(n − 6) rows of the order on a planted copy of the data. Passing it does not prove the program never peeked.
- **d2h is one overall score.** The frozen checker also accepts z-score sums and rank sums for "overall"; on NASA, whose goals conflict, they can disagree. This metric uses d2h only.
- **The random baseline is analytical.** It is the exact expectation for uniform sampling without replacement, not a simulation, and it says nothing about variance between LLM trials.
- **T7 is not included** in this phase.

## Provenance

- Archive: `harness/results/v2/gpt-oss-20b-3trials.zip` (sha256 `d1a51a3fa3711829ede726dd08e1f77da070fc3e3064fdcd1971d24974eb7683`)
- Code commit: `1318303c4841a311877ae6b38dfeef558bcec05f` on `analysis/external-metrics` (uncommitted changes present)
- Command: `python -m harness.external`
- Full input and output hashes: `manifest.json`.
