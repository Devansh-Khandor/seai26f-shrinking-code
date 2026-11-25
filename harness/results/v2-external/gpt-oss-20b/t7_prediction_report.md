# External metric, phases 4B–4C: T7 prediction (Cars, NASA, Apache) beside T1–T6

**Status: post-observation analysis.** Added after the Task Talk; not part of the pre-registered plan in `docs/hypotheses.md`.

**What T7 is here.** One human-directed implementation per domain, run through its existing `t7_validation` adapter. It is deterministic: rerunning it gives the same numbers. Each row below is that one implementation's result, **not** an LLM trial, and the four requests are not a sample of repetitions. No `same()`, `bestRanks()` or other significance test is applied to T7.

**How the two sides were built differs.** The T7 adapters were written and refined against the common semantic checker. The T1–T6 programs were generated one-shot by `gpt-oss-20b` without access to it; their failed and crashed attempts are counted, while T7's development iterations are not.

**What the T7 code is built on differs by domain.** NASA and Apache T7 are standalone human-directed modules. Cars T7 is a set of human-directed scripts that call the shared `shrinking_code/` library, which was developed with LLM assistance (README, "Disclosure"); the Cars predictions therefore come from that library, which is not independently human-written. The Cars adapters (`t7_validation/adapters/cars/`, added in phase 4C) only re-split the data to the common protocol and print the common answer format; they add no modelling logic, and their lines are not T7 implementation LOC.

## Protocol alignment

Both sides are scored by the same code (`harness.external.baseline`, `model_mae`, `mae_ratio`) on the same CSV, the same held-out rows (`harness.oracle.held_out`: row numbers divisible by 5), the same target and the same training-mean baseline. MAE is recomputed from the recorded predictions; reported values are only compared.

| request | domain | target | rows | training | held out | held-out rows match | baseline mean | baseline MAE |
|---|---|---|---:|---:|---:|---|---:|---:|
| CAR-03 | cars | `Mpg+` | 398 | 319 | 79 | True | 23.824 | 6.988 |
| CAR-20 | cars | `Mpg+` | 398 | 319 | 79 | True | 23.824 | 6.988 |
| NASA-03 | nasa | `Effort-` | 93 | 75 | 18 | True | 541.300 | 885.089 |
| NASA-19 | nasa | `Effort-` | 93 | 75 | 18 | True | 541.300 | 885.089 |
| APACHE-03 | apache | `Performance-` | 192 | 154 | 38 | True | 1402.208 | 464.590 |
| APACHE-20 | apache | `Performance-` | 192 | 154 | 38 | True | 1402.208 | 464.590 |

## T7 results

| request | model MAE | baseline MAE | MAE ratio | beats baseline | held-out rows better than baseline | semantic verdict | leakage (mask) | shuffled | reruns | included |
|---|---:|---:|---:|---|---:|---|---|---|---|---|
| CAR-03 | 4.228 | 6.988 | 0.605 | True | 51/79 | pass | pass | pass | base=ran;mask=ran;shuffled=ran | yes |
| CAR-20 | 4.228 | 6.988 | 0.605 | True | 51/79 | pass | pass | pass | base=ran;mask=ran;shuffled=ran | yes |
| NASA-03 | 724.967 | 885.089 | 0.819 | True | 10/18 | pass | pass | pass | base=ran;mask Defects-=ran;mask Effort-=ran;shuffled=ran | yes |
| NASA-19 | 724.967 | 885.089 | 0.819 | True | 10/18 | pass | pass | pass | base=ran;honeypot=ran;mask=ran;shuffled=ran | yes |
| APACHE-03 | 209.053 | 464.590 | 0.450 | True | 34/38 | pass | pass | pass | base=ran;mask=ran;shuffled=ran | yes |
| APACHE-20 | 209.053 | 464.590 | 0.450 | True | 34/38 | pass | pass | pass | base=ran;honeypot=ran;mask=ran;shuffled=ran | yes |

A passed mask rerun shows the predictions did not move when only held-out targets changed. It is a test, not a proof that no information leaked.

## CAR-03: T7 beside T1–T6 (Phase 1 Guess)

| treatment | usable / attempted | median MAE ratio | min–max ratio | beats baseline |
|---|---:|---:|---|---:|
| T1 | 1 / 3 | 0.502 | 0.502–0.502 | 1/1 |
| T2 | 2 / 3 | 0.635 | 0.502–0.769 | 2/2 |
| T3 | 2 / 3 | 0.563 | 0.559–0.567 | 2/2 |
| T4 | 1 / 3 | 0.555 | 0.555–0.555 | 1/1 |
| T5 | 2 / 3 | 1.453 | 0.559–2.348 | 1/2 |
| T6 | 0 / 3 | — | — | 0/0 |
| **T7** (n = 1, not a trial) | 1 / 1 | 0.605 | — | 1/1 |

Position only: T7's ratio is lower than 2 of the 8 usable T1–T6 trials and equal to 0 (all treatments pooled). The closest usable T1–T6 trial (T3 trial 1, ratio 0.5672) differs by +0.0378 in ratio, +0.264 in MAE. This is a description, not a test.

Original T7 write-up (`human_t7/cars/guess.py`), a **different protocol** (seeded shuffled 80/20 split (train_test_split, seed 42)): no MAE is recorded in the repository; the script prints it when run. Not comparable with the numbers above.

## CAR-20: T7 beside T1–T6 (Phase 3 Race)

| treatment | usable / attempted | median MAE ratio | min–max ratio | beats baseline |
|---|---:|---:|---|---:|
| T1 | 2 / 3 | 0.922 | 0.559–1.285 | 1/2 |
| T2 | 2 / 3 | 0.635 | 0.502–0.769 | 2/2 |
| T3 | 1 / 3 | 0.559 | 0.559–0.559 | 1/1 |
| T4 | 0 / 3 | — | — | 0/0 |
| T5 | 2 / 3 | 0.564 | 0.559–0.569 | 2/2 |
| T6 | 0 / 3 | — | — | 0/0 |
| **T7** (n = 1, not a trial) | 1 / 1 | 0.605 | — | 1/1 |

Position only: T7's ratio is lower than 2 of the 7 usable T1–T6 trials and equal to 0 (all treatments pooled). The closest usable T1–T6 trial (T5 trial 2, ratio 0.5692) differs by +0.0357 in ratio, +0.250 in MAE. This is a description, not a test.

Original T7 write-up (`human_t7/cars/race.py`), a **different protocol** (seeded shuffled 80/20 split (train_test_split, seed 42)): no MAE is recorded in the repository; the script prints it when run. Not comparable with the numbers above.

## NASA-03: T7 beside T1–T6 (Phase 1 Guess)

| treatment | usable / attempted | median MAE ratio | min–max ratio | beats baseline |
|---|---:|---:|---|---:|
| T1 | 2 / 3 | 0.814 | 0.801–0.827 | 2/2 |
| T2 | 0 / 3 | — | — | 0/0 |
| T3 | 0 / 3 | — | — | 0/0 |
| T4 | 3 / 3 | 0.827 | 0.712–0.832 | 3/3 |
| T5 | 1 / 3 | 0.827 | 0.827–0.827 | 1/1 |
| T6 | 2 / 3 | 0.883 | 0.765–1.000 | 1/2 |
| **T7** (n = 1, not a trial) | 1 / 1 | 0.819 | — | 1/1 |

Position only: T7's ratio is lower than 5 of the 8 usable T1–T6 trials and equal to 0 (all treatments pooled). The closest usable T1–T6 trial (T1 trial 3, ratio 0.8268) differs by -0.0077 in ratio, -6.803 in MAE. This is a description, not a test.

Original T7 write-up (`human_t7/nasa/RESULTS.md`), a **different protocol** (first 74 rows train, last 19 held out; the Guess section defers to the Race section for its MAE comparison): kNN 1,092.17 vs historical mean 1,031.84 (effort). Not comparable with the numbers above.

## NASA-19: T7 beside T1–T6 (Phase 3 Race)

| treatment | usable / attempted | median MAE ratio | min–max ratio | beats baseline |
|---|---:|---:|---|---:|
| T1 | 2 / 3 | 0.827 | 0.827–0.827 | 2/2 |
| T2 | 2 / 3 | 1.005 | 0.988–1.022 | 1/2 |
| T3 | 0 / 3 | — | — | 0/0 |
| T4 | 1 / 3 | 0.712 | 0.712–0.712 | 1/1 |
| T5 | 3 / 3 | 0.827 | 0.568–1.023 | 2/3 |
| T6 | 2 / 3 | 0.763 | 0.761–0.765 | 2/2 |
| **T7** (n = 1, not a trial) | 1 / 1 | 0.819 | — | 1/1 |

Position only: T7's ratio is lower than 6 of the 10 usable T1–T6 trials and equal to 0 (all treatments pooled). The closest usable T1–T6 trial (T1 trial 2, ratio 0.8268) differs by -0.0077 in ratio, -6.803 in MAE. This is a description, not a test.

Original T7 write-up (`human_t7/nasa/RESULTS.md`), a **different protocol** (first 74 rows train, last 19 held out): kNN 1,092.17 vs historical mean 1,031.84 (effort). Not comparable with the numbers above.

## APACHE-03: T7 beside T1–T6 (Phase 1 Guess)

| treatment | usable / attempted | median MAE ratio | min–max ratio | beats baseline |
|---|---:|---:|---|---:|
| T1 | 2 / 3 | 0.689 | 0.451–0.927 | 2/2 |
| T2 | 1 / 3 | 1.000 | 1.000–1.000 | 0/1 |
| T3 | 2 / 3 | 0.451 | 0.451–0.451 | 2/2 |
| T4 | 3 / 3 | 0.451 | 0.451–0.451 | 3/3 |
| T5 | 3 / 3 | 0.484 | 0.451–0.603 | 3/3 |
| T6 | 0 / 3 | — | — | 0/0 |
| **T7** (n = 1, not a trial) | 1 / 1 | 0.450 | — | 1/1 |

Position only: T7's ratio is lower than 11 of the 11 usable T1–T6 trials and equal to 0 (all treatments pooled). The closest usable T1–T6 trial (T1 trial 3, ratio 0.4514) differs by -0.0014 in ratio, -0.658 in MAE. This is a description, not a test.

Original T7 write-up (`human_t7/apache/RESULTS.md`), a **different protocol** (seeded shuffled 154/38 split (seed 42)): kNN 174.79 vs historical mean 484.05. Not comparable with the numbers above.

## APACHE-20: T7 beside T1–T6 (Phase 3 Race)

| treatment | usable / attempted | median MAE ratio | min–max ratio | beats baseline |
|---|---:|---:|---|---:|
| T1 | 2 / 3 | 0.802 | 0.603–1.000 | 1/2 |
| T2 | 2 / 3 | 1.000 | 1.000–1.000 | 0/2 |
| T3 | 2 / 3 | 0.294 | 0.137–0.451 | 2/2 |
| T4 | 3 / 3 | 0.451 | 0.451–0.451 | 3/3 |
| T5 | 1 / 3 | 3.295 | 3.295–3.295 | 0/1 |
| T6 | 0 / 3 | — | — | 0/0 |
| **T7** (n = 1, not a trial) | 1 / 1 | 0.450 | — | 1/1 |

Position only: T7's ratio is lower than 9 of the 10 usable T1–T6 trials and equal to 0 (all treatments pooled). The closest usable T1–T6 trial (T3 trial 1, ratio 0.4514) differs by -0.0014 in ratio, -0.663 in MAE. This is a description, not a test.

Original T7 write-up (`human_t7/apache/RESULTS.md`), a **different protocol** (seeded shuffled 154/38 split reused from Guess): kNN 174.79 vs historical mean 484.05. Not comparable with the numbers above.

## Limitations

- **n = 1 per domain and request.** T7 can only be placed beside the T1–T6 distributions, not tested against them.
- **Guess and Race prediction share data.** In every domain (Cars `Mpg+`, NASA `Effort-`, Apache `Performance-`) the Guess and Race requests use the same target, split and model in T7, so each pair of rows is one result shown twice, not two pieces of evidence.
- **Few usable T1–T6 trials.** Several T1–T6 cells have 0–1 usable trials.
- **Development process differs** (see above). A T7 result that matches or beats T1–T6 here reflects an implementation refined against the checker.
- **Cars adapters judge only the prediction request.** CAR-04 and CAR-19 are not implemented, so the semantic verdict shown for CAR-03 and CAR-20 is the request's own verdict (`evaluated_requests`), not a step verdict. NASA and Apache adapters are judged on both requests of their step, as their runners do.
- **The Cars CAR-20 verdict label** is derived from the shared library's own paired comparison (`compare().same`) and the median errors; it does not affect the MAE values.

## Provenance

- Code commit: `6ec26a85031cf212fca12284e855c9a745e2e3a8` on `analysis/external-metrics` (uncommitted changes present)
- Command: `python -m harness.external`
- Adapter, implementation, dataset, stdout and evidence hashes: `t7_prediction.csv`; all output hashes: `manifest.json`.
