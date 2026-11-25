# External metric, phase 3: Race — guided vs random selection, model vs historical-mean prediction

**Status: post-observation analysis.** These metrics were added after the Task Talk, after the v2 T1–T6 results had been seen. They are not part of the pre-registered plan in `docs/hypotheses.md` and do not change its win conditions.

## Requests

Race is skill 10. Each attempted step holds two requests, and NASA numbers them the other way round:

| domain | selection race | prediction race |
|---|---|---|
| cars | CAR-19 | CAR-20 |
| nasa | NASA-20 | NASA-19 |
| apache | APACHE-19 | APACHE-20 |

The prediction race is kept separate from the phase 1 Guess metric (CAR-03, NASA-03, APACHE-03), even where the target and the trial are the same.

## A. Guided vs random selection

**Recorded answer:** `{"budget", "repeats", "ours": [best row per repeat], "random": [best of `budget` random rows per repeat], "verdict"}`. Every row is scored with the frozen `harness.oracle.d2h` (lower is better). Two comparisons are reported, and they are different kinds of evidence:

- **Empirical (archived repeats):** median d2h of each arm over the program's own repeats (20 in every valid answer); *median advantage* = median random − median guided (positive = guided better); *win fraction* = share of repeats where the guided row has lower d2h than the random row of the same repeat index. This relies on the program's random arm, so it is used only when the frozen check "random baseline is really random" passed.
- **Analytical:** the exact expected best d2h of `budget` rows drawn uniformly at random from the whole CSV, computed in closed form; *advantage vs analytical* = that expectation − the median guided d2h. This does not depend on the program's random arm.

Repeats are summarised within each trial; they are never counted as LLM trials. The archive records only the best row of each repeat, not the rows each arm evaluated, so it cannot be checked that those rows really are the best each arm saw.

**Primary:** the program ran, the answer passed the frozen format check, the honeypot rerun passed (no budget peeking) and the reported budget equals the frozen budget. **Delivered** = primary and advantage vs analytical > 0 (analysis choice), over all 3 attempted trials.

| domain | T | budget | attempted | exec. failed | bad answer | schema valid | honeypot failed | honeypot unverified | primary n | median guided d2h | analytical random best | median advantage vs analytical | empirical n | median random-arm d2h | median advantage (empirical) | median win fraction | delivery rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cars | T1 | 30 | 3 | 1 | 1 | 1 | 1 | 0 | 0 | — | 0.266 | — | 0 | — | — | — | 0% |
| cars | T2 | 30 | 3 | 1 | 0 | 2 | 2 | 0 | 0 | — | 0.266 | — | 0 | — | — | — | 0% |
| cars | T3 | 30 | 3 | 2 | 0 | 1 | 1 | 0 | 0 | — | 0.266 | — | 0 | — | — | — | 0% |
| cars | T4 | 30 | 3 | 2 | 0 | 1 | 0 | 0 | 1 | 0.167 | 0.266 | +0.099 | 1 | 0.271 | +0.104 | 0.95 | 33% |
| cars | T5 | 30 | 3 | 1 | 0 | 2 | 2 | 0 | 0 | — | 0.266 | — | 0 | — | — | — | 0% |
| cars | T6 | 30 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | — | 0.266 | — | 0 | — | — | — | 0% |
| nasa | T1 | 18 | 3 | 1 | 0 | 2 | 2 | 0 | 0 | — | 0.425 | — | 0 | — | — | — | 0% |
| nasa | T2 | 18 | 3 | 1 | 1 | 1 | 1 | 0 | 0 | — | 0.425 | — | 0 | — | — | — | 0% |
| nasa | T3 | 18 | 3 | 2 | 0 | 1 | 1 | 0 | 0 | — | 0.425 | — | 0 | — | — | — | 0% |
| nasa | T4 | 18 | 3 | 2 | 0 | 1 | 0 | 1 | 0 | — | 0.425 | — | 0 | — | — | — | 0% |
| nasa | T5 | 18 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | — | 0.425 | — | 0 | — | — | — | 0% |
| nasa | T6 | 18 | 3 | 1 | 0 | 2 | 2 | 0 | 0 | — | 0.425 | — | 0 | — | — | — | 0% |
| apache | T1 | 30 | 3 | 1 | 0 | 2 | 2 | 0 | 0 | — | 0.021 | — | 0 | — | — | — | 0% |
| apache | T2 | 30 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | — | 0.021 | — | 0 | — | — | — | 0% |
| apache | T3 | 30 | 3 | 1 | 0 | 2 | 1 | 0 | 1 | 0.000 | 0.021 | +0.021 | 1 | 0.033 | +0.033 | 0.80 | 33% |
| apache | T4 | 30 | 3 | 0 | 0 | 3 | 1 | 0 | 2 | 0.000 | 0.021 | +0.021 | 2 | 0.025 | +0.025 | 0.70 | 67% |
| apache | T5 | 30 | 3 | 2 | 0 | 1 | 1 | 0 | 0 | — | 0.021 | — | 0 | — | — | — | 0% |
| apache | T6 | 30 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | — | 0.021 | — | 0 | — | — | — | 0% |

"exec. failed" includes steps that produced no program (`parse_failed`).

### By domain

| domain | request | attempted | schema valid | honeypot failed | primary n | median advantage vs analytical | delivery rate |
|---|---|---:|---:|---:|---:|---:|---:|
| cars | CAR-19 | 18 | 7 | 6 | 1 | +0.099 | 6% |
| nasa | NASA-20 | 18 | 10 | 9 | 0 | — | 0% |
| apache | APACHE-19 | 18 | 11 | 8 | 3 | +0.021 | 17% |

### Budget peeking

23 of the 28 schema-valid selection answers failed the honeypot check and are excluded. For context only (these values do not measure honest selection):

| domain | schema valid | honeypot failed | median advantage vs analytical, excluded | median advantage vs analytical, primary |
|---|---:|---:|---:|---:|
| cars | 7 | 6 | +0.045 | +0.099 |
| nasa | 10 | 9 | +0.033 | — |
| apache | 11 | 8 | +0.021 | +0.021 |

### Is the programs' random arm really random?

1 of the 28 schema-valid answers failed the frozen check "random baseline is really random". The table compares each program's random arm (mean d2h of its recorded best-of-budget rows) with the exact expectation for uniform random sampling. A gap far from 0 means the reported random arm is not plain best-of-budget random sampling scored by d2h.

| domain | schema valid | analytical random best | median of the programs' random-arm mean d2h | median gap | min–max gap |
|---|---:|---:|---:|---:|---|
| cars | 7 | 0.266 | 0.286 | +0.020 | -0.013–+0.099 |
| nasa | 10 | 0.425 | 0.484 | +0.059 | +0.022–+0.150 |
| apache | 11 | 0.021 | 0.027 | +0.006 | -0.004–+0.304 |

## B. Model vs historical-mean prediction

Same definitions as phase 1: the every-fifth-row hold-out, the training-mean baseline, `mae_ratio` = model MAE ÷ baseline MAE (below 1 = better than the baseline). Both MAEs are recomputed from the recorded predictions and the CSV; the program's own `baseline`, `mae_model` and `mae_baseline` are compared but never used. **Primary:** ran, schema valid, mask leakage rerun passed.

| domain | T | attempted | exec. failed | bad answer | schema valid | leakage failed | leakage unverified | primary n | median MAE ratio | min–max ratio | median MAE advantage | beats baseline | delivery rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| cars | T1 | 3 | 1 | 0 | 2 | 0 | 0 | 2 | 0.922 | 0.559–1.285 | 0.547 | 1/2 | 33% |
| cars | T2 | 3 | 1 | 0 | 2 | 0 | 0 | 2 | 0.635 | 0.502–0.769 | 2.549 | 2/2 | 67% |
| cars | T3 | 3 | 2 | 0 | 1 | 0 | 0 | 1 | 0.559 | 0.559–0.559 | 3.084 | 1/1 | 33% |
| cars | T4 | 3 | 2 | 0 | 1 | 1 | 0 | 0 | — | — | — | 0/0 | 0% |
| cars | T5 | 3 | 1 | 0 | 2 | 0 | 0 | 2 | 0.564 | 0.559–0.569 | 3.047 | 2/2 | 67% |
| cars | T6 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | — | — | — | 0/0 | 0% |
| nasa | T1 | 3 | 1 | 0 | 2 | 0 | 0 | 2 | 0.827 | 0.827–0.827 | 153.319 | 2/2 | 67% |
| nasa | T2 | 3 | 1 | 0 | 2 | 0 | 0 | 2 | 1.005 | 0.988–1.022 | -4.584 | 1/2 | 33% |
| nasa | T3 | 3 | 2 | 1 | 0 | 0 | 0 | 0 | — | — | — | 0/0 | 0% |
| nasa | T4 | 3 | 2 | 0 | 1 | 0 | 0 | 1 | 0.712 | 0.712–0.712 | 254.821 | 1/1 | 33% |
| nasa | T5 | 3 | 0 | 0 | 3 | 0 | 0 | 3 | 0.827 | 0.568–1.023 | 153.319 | 2/3 | 67% |
| nasa | T6 | 3 | 1 | 0 | 2 | 0 | 0 | 2 | 0.763 | 0.761–0.765 | 209.860 | 2/2 | 67% |
| apache | T1 | 3 | 1 | 0 | 2 | 0 | 0 | 2 | 0.802 | 0.603–1.000 | 92.163 | 1/2 | 33% |
| apache | T2 | 3 | 0 | 1 | 2 | 0 | 0 | 2 | 1.000 | 1.000–1.000 | -0.000 | 0/2 | 0% |
| apache | T3 | 3 | 1 | 0 | 2 | 0 | 0 | 2 | 0.294 | 0.137–0.451 | 327.889 | 2/2 | 67% |
| apache | T4 | 3 | 0 | 0 | 3 | 0 | 0 | 3 | 0.451 | 0.451–0.451 | 254.875 | 3/3 | 100% |
| apache | T5 | 3 | 2 | 0 | 1 | 0 | 0 | 1 | 3.295 | 3.295–3.295 | -1066.200 | 0/1 | 0% |
| apache | T6 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | — | — | — | 0/0 | 0% |

### By domain

| domain | request | baseline MAE | attempted | primary n | median MAE ratio | delivery rate |
|---|---|---:|---:|---:|---:|---:|
| cars | CAR-20 | 6.988 | 18 | 7 | 0.559 | 33% |
| nasa | NASA-19 | 885.089 | 18 | 10 | 0.827 | 44% |
| apache | APACHE-20 | 464.590 | 18 | 10 | 0.527 | 33% |

### Reported vs recomputed values

Among 28 schema-valid prediction answers, the program's reported value disagreed with the recomputed one for: baseline mean 1, model MAE 0, baseline MAE 3. Only recomputed values are used above.

## Per-trial values

| domain | T | trial | selection | guided d2h | advantage vs analytical | win fraction | prediction | MAE ratio |
|---|---|---:|---|---:|---:|---:|---|---:|
| cars | T1 | 1 | answer_schema_error | — | — | — | included | 0.559 |
| cars | T1 | 2 | execution_failure | — | — | — | execution_failure | — |
| cars | T1 | 3 | honeypot_failed | 0.167 | +0.099 | 0.95 | included | 1.285 |
| cars | T2 | 1 | execution_failure | — | — | — | execution_failure | — |
| cars | T2 | 2 | honeypot_failed | 0.167 | +0.099 | 1.00 | included | 0.502 |
| cars | T2 | 3 | honeypot_failed | 0.275 | -0.008 | 0.80 | included | 0.769 |
| cars | T3 | 1 | honeypot_failed | 0.393 | -0.126 | 0.00 | included | 0.559 |
| cars | T3 | 2 | execution_failure | — | — | — | execution_failure | — |
| cars | T3 | 3 | execution_failure | — | — | — | execution_failure | — |
| cars | T4 | 1 | execution_failure | — | — | — | execution_failure | — |
| cars | T4 | 2 | included | 0.167 | +0.099 | 0.95 | leakage_failed | 0.498 |
| cars | T4 | 3 | execution_failure | — | — | — | execution_failure | — |
| cars | T5 | 1 | execution_failure | — | — | — | execution_failure | — |
| cars | T5 | 2 | honeypot_failed | 0.167 | +0.099 | 1.00 | included | 0.569 |
| cars | T5 | 3 | honeypot_failed | 0.335 | -0.069 | 0.00 | included | 0.559 |
| cars | T6 | 1 | execution_failure | — | — | — | execution_failure | — |
| cars | T6 | 2 | generation_failed | — | — | — | generation_failed | — |
| cars | T6 | 3 | execution_failure | — | — | — | execution_failure | — |
| nasa | T1 | 1 | execution_failure | — | — | — | execution_failure | — |
| nasa | T1 | 2 | honeypot_failed | 0.577 | -0.153 | 0.00 | included | 0.827 |
| nasa | T1 | 3 | honeypot_failed | 0.391 | +0.033 | 0.75 | included | 0.827 |
| nasa | T2 | 1 | execution_failure | — | — | — | execution_failure | — |
| nasa | T2 | 2 | honeypot_failed | 0.391 | +0.033 | 0.75 | included | 0.988 |
| nasa | T2 | 3 | answer_schema_error | — | — | — | included | 1.022 |
| nasa | T3 | 1 | honeypot_failed | 0.391 | +0.033 | 0.75 | answer_schema_error | — |
| nasa | T3 | 2 | execution_failure | — | — | — | execution_failure | — |
| nasa | T3 | 3 | execution_failure | — | — | — | execution_failure | — |
| nasa | T4 | 1 | generation_failed | — | — | — | generation_failed | — |
| nasa | T4 | 2 | generation_failed | — | — | — | generation_failed | — |
| nasa | T4 | 3 | honeypot_unverified | 0.559 | -0.135 | 0.20 | included | 0.712 |
| nasa | T5 | 1 | honeypot_failed | 0.391 | +0.033 | 0.55 | included | 0.827 |
| nasa | T5 | 2 | honeypot_failed | 0.574 | -0.150 | 0.65 | included | 1.023 |
| nasa | T5 | 3 | honeypot_failed | 0.526 | -0.101 | 0.30 | included | 0.568 |
| nasa | T6 | 1 | execution_failure | — | — | — | execution_failure | — |
| nasa | T6 | 2 | honeypot_failed | 0.391 | +0.033 | 0.70 | included | 0.765 |
| nasa | T6 | 3 | honeypot_failed | 0.526 | -0.101 | 0.30 | included | 0.761 |
| apache | T1 | 1 | honeypot_failed | 0.067 | -0.046 | 1.00 | included | 1.000 |
| apache | T1 | 2 | honeypot_failed | 0.067 | -0.046 | 0.00 | included | 0.603 |
| apache | T1 | 3 | execution_failure | — | — | — | execution_failure | — |
| apache | T2 | 1 | honeypot_failed | 0.000 | +0.021 | 0.80 | included | 1.000 |
| apache | T2 | 2 | honeypot_failed | 0.000 | +0.021 | 0.55 | included | 1.000 |
| apache | T2 | 3 | honeypot_failed | 0.067 | -0.046 | 0.00 | answer_schema_error | — |
| apache | T3 | 1 | honeypot_failed | 0.000 | +0.021 | 0.75 | included | 0.451 |
| apache | T3 | 2 | included | 0.000 | +0.021 | 0.80 | included | 0.137 |
| apache | T3 | 3 | execution_failure | — | — | — | execution_failure | — |
| apache | T4 | 1 | included | 0.000 | +0.021 | 0.80 | included | 0.451 |
| apache | T4 | 2 | honeypot_failed | 0.000 | +0.021 | 0.80 | included | 0.451 |
| apache | T4 | 3 | included | 0.000 | +0.021 | 0.60 | included | 0.451 |
| apache | T5 | 1 | execution_failure | — | — | — | execution_failure | — |
| apache | T5 | 2 | execution_failure | — | — | — | execution_failure | — |
| apache | T5 | 3 | honeypot_failed | 0.000 | +0.021 | 0.75 | included | 3.295 |
| apache | T6 | 1 | execution_failure | — | — | — | execution_failure | — |
| apache | T6 | 2 | execution_failure | — | — | — | execution_failure | — |
| apache | T6 | 3 | execution_failure | — | — | — | execution_failure | — |

## Limitations

- **Three trials per cell, often fewer usable.** No statistical comparison between treatments is made.
- **Repeats are not trials.** The 20 repeats inside one answer come from one program run; the win fraction describes that program, not variation between LLM trials.
- **Pairing is as recorded.** Repeat i of `ours` is compared with repeat i of `random`; the archive does not show whether the program used the same seed or candidate pool for both arms.
- **The arms' evaluated rows are not archived**, so the reported best rows cannot be checked against what each arm actually saw. The analytical baseline avoids relying on the program's random arm.
- **"Best" is judged by d2h here.** The frozen checker also accepts z-score sums and rank sums; the programs may have used another overall score to pick their best rows.
- **Budgets differ by domain** (Cars 30, NASA 18, Apache 30) and so do pool sizes.
- **The prediction race uses the same split and target as Guess** for Cars and Apache (and NASA's Effort); it comes from a later step of the same trials, so it is not independent evidence.
- **T7 is not included** in this phase.

## Provenance

- Archive: `harness/results/v2/gpt-oss-20b-3trials.zip` (sha256 `d1a51a3fa3711829ede726dd08e1f77da070fc3e3064fdcd1971d24974eb7683`)
- Code commit: `b986524c3fa050ce9b4c7c808687847804dbafa4` on `analysis/external-metrics` (uncommitted changes present)
- Command: `python -m harness.external`
- Full input and output hashes: `manifest.json`.
