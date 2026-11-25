# External metric, phase 5G: Flow — are streaming decisions, predictions and shortlists useful?

**Status: post-observation analysis.** Added after the Task Talk; not part of the pre-registered plan in `docs/hypotheses.md`. Arrival order, checkpoints and thresholds are the frozen ones; every reference set at arrival r uses rows 1..r only (and r-1 for baselines and thresholds). Scores are recomputed from the archived ANSWER lines and the recorded CSV values; no self-reported statistic is used.

**Frozen request kinds.** CAR-07 labels each arriving vehicle desirable or not. NASA-07 predicts each arrival's effort and defects and raises alerts. APACHE-07 raises alerts on observed performance. CAR-08, NASA-08 and APACHE-08 all report summaries and a 5-row shortlist at checkpoints after 0, k/2 and k arrivals (NASA-08 has no alerts).

**Prediction versus rule adherence.** Only NASA-07 is prospective: its alerts are meant to come from predictions made before the outcome is known. CAR-07 classifies vehicles whose goals are already in the file, and APACHE-07 alerts on an observed benchmark result; those measure correct use of a rule on observed data, not prediction.

**References (post-hoc applications of frozen definitions).** CAR-07: the oracle's 'best seen' set (Pareto-optimal or best 25% under any overall score among rows 1..r, as the frozen checkpoint checker uses), and a strict reference, the Pareto front of rows 1..r. NASA-07 alerts: recorded Effort- or Defects- above the median of rows 1..r-1 (the frozen alert-justification threshold applied to actual outcomes). APACHE-07: Performance- worse than the median of rows 1..r-1 (frozen justification); worse than every earlier row is the frozen required alert. NASA-07 baseline: mean Effort- of rows 1..r-1.

**Integrity.** -07 answers need the archived future-arrivals rerun to pass (unverified reruns are excluded). -08 answers must not shortlist rows that have not arrived. *Success* = the frozen checker's base checks all pass; *delivery* = successes / all attempts.

## Base rates (fixed by the data)

| request | reference | positives / arrivals or base rate |
|---|---|---|
| CAR-07 | best-seen set | 89% of arrivals |
| CAR-08 | best-seen set / Pareto front of visible rows | 26% / 2% of visible rows (checkpoint mean) |
| NASA-07 | high outcome | 89% of arrivals |
| NASA-08 | best-seen set / Pareto front of visible rows | 67% / 54% of visible rows (checkpoint mean) |
| APACHE-07 | worse than median so far | 74% of arrivals |
| APACHE-08 | best-seen set / Pareto front of visible rows | 27% / 1% of visible rows (checkpoint mean) |

With base rates this high, 'accept or alert on everything' already scores precision equal to the base rate; precision must be read against it.

## CAR-07 (`stream_desirable`)

| T | attempted | executable | valid | median accepted | median precision | median recall | median strict precision | median strict recall | all accepted | successful | delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 3 | 3 | 3 | 16 | 1.00 | 0.23 | 0.62 | 0.83 | 0 | 3 | 100% |
| T2 | 3 | 1 | 1 | 12 | 1.00 | 0.17 | 0.83 | 0.83 | 0 | 1 | 33% |
| T3 | 3 | 2 | 2 | 16 | 1.00 | 0.23 | 0.62 | 0.83 | 0 | 2 | 67% |
| T4 | 3 | 3 | 3 | 18 | 1.00 | 0.25 | 0.41 | 0.83 | 0 | 3 | 100% |
| T5 | 3 | 2 | 2 | 16 | 1.00 | 0.23 | 0.62 | 0.83 | 0 | 2 | 67% |
| T6 | 3 | 1 | 1 | 3 | 1.00 | 0.04 | 1.00 | 0.25 | 0 | 1 | 33% |

## CAR-08 (`stream_summary`)

| T | attempted | executable | schema valid | valid | median best-seen precision | median front precision | median d2h percentile | median arrivals shortlisted | median summary fidelity | future rows | successful | delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 3 | 3 | 2 | 2 | 1.00 | 0.67 | 0.006 | 6 | 1.000 | 0 | 2 | 67% |
| T2 | 3 | 1 | 1 | 1 | 1.00 | 0.80 | 0.006 | 6 | 1.000 | 0 | 1 | 33% |
| T3 | 3 | 2 | 2 | 2 | 1.00 | 0.67 | 0.006 | 6 | 1.000 | 0 | 2 | 67% |
| T4 | 3 | 3 | 3 | 3 | 1.00 | 0.67 | 0.006 | 6 | 1.000 | 0 | 2 | 67% |
| T5 | 3 | 2 | 2 | 2 | 1.00 | 0.67 | 0.006 | 6 | 1.000 | 0 | 2 | 67% |
| T6 | 3 | 1 | 1 | 1 | 1.00 | 0.53 | 0.011 | 4 | 1.000 | 0 | 1 | 33% |

## NASA-07 (`stream_alert_pred`)

| T | attempted | executable | valid | median MAE ratio | beats baseline | median share better | median defects MAE ratio | median self-match | median alerts | median alert precision | median alert recall | successful | delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 3 | 2 | 2 | 1.01 | 1 | 0.42 | 0.22 | 0.00 | 0 | — | 0.00 | 2 | 67% |
| T2 | 3 | 3 | 1 | 1.00 | 0 | 0.32 | 0.97 | 0.00 | 0 | — | 0.00 | 0 | 0% |
| T3 | 3 | 0 | 0 | — | 0 | — | — | — | — | — | — | 0 | 0% |
| T4 | 3 | 2 | 1 | 1.20 | 0 | 0.47 | 1.11 | 0.00 | 8 | 1.00 | 0.47 | 1 | 33% |
| T5 | 3 | 1 | 1 | 0.99 | 1 | 0.42 | 0.22 | 0.00 | 0 | — | 0.00 | 1 | 33% |
| T6 | 3 | 1 | 1 | 1.02 | 0 | 0.32 | 1.12 | 0.00 | 0 | — | 0.00 | 1 | 33% |

## NASA-08 (`stream_summary`)

| T | attempted | executable | schema valid | valid | median best-seen precision | median front precision | median d2h percentile | median arrivals shortlisted | median summary fidelity | future rows | successful | delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 3 | 2 | 2 | 2 | 1.00 | 1.00 | 0.494 | 0 | 1.000 | 0 | 2 | 67% |
| T2 | 3 | 3 | 1 | 1 | 1.00 | 1.00 | 0.257 | 0 | 1.000 | 0 | 1 | 33% |
| T3 | 3 | 0 | 0 | 0 | — | — | — | — | — | 0 | 0 | 0% |
| T4 | 3 | 2 | 2 | 2 | 1.00 | 0.97 | 0.042 | 1 | 1.000 | 0 | 2 | 67% |
| T5 | 3 | 1 | 1 | 1 | 1.00 | 1.00 | 0.048 | 0 | 1.000 | 0 | 1 | 33% |
| T6 | 3 | 1 | 0 | 0 | — | — | — | — | — | 0 | 0 | 0% |

## APACHE-07 (`stream_alert_poor`)

| T | attempted | executable | valid | median alerts | median precision | median recall | zero-alert answers | successful | delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 3 | 3 | 3 | 4 | 1.00 | 0.14 | 0 | 3 | 100% |
| T2 | 3 | 3 | 3 | 6 | 1.00 | 0.14 | 0 | 2 | 67% |
| T3 | 3 | 2 | 2 | 4 | 1.00 | 0.14 | 0 | 2 | 67% |
| T4 | 3 | 2 | 2 | 3 | 1.00 | 0.11 | 0 | 2 | 67% |
| T5 | 3 | 3 | 2 | 2 | 1.00 | 0.07 | 1 | 2 | 67% |
| T6 | 3 | 1 | 1 | 1 | 1.00 | 0.04 | 0 | 1 | 33% |

## APACHE-08 (`stream_summary`)

| T | attempted | executable | schema valid | valid | median best-seen precision | median front precision | median d2h percentile | median arrivals shortlisted | median summary fidelity | future rows | successful | delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 3 | 3 | 3 | 3 | 1.00 | 0.40 | 0.012 | 0 | 1.000 | 0 | 3 | 100% |
| T2 | 3 | 3 | 3 | 3 | 1.00 | 0.40 | 0.012 | 0 | 1.000 | 0 | 3 | 100% |
| T3 | 3 | 2 | 1 | 1 | 1.00 | 0.40 | 0.012 | 0 | 1.000 | 0 | 1 | 33% |
| T4 | 3 | 2 | 1 | 1 | 1.00 | 0.40 | 0.012 | 0 | 1.000 | 0 | 1 | 33% |
| T5 | 3 | 3 | 2 | 2 | 1.00 | 0.40 | 0.012 | 0 | 1.000 | 0 | 2 | 67% |
| T6 | 3 | 1 | 1 | 1 | 1.00 | 0.40 | 0.012 | 0 | 1.000 | 0 | 1 | 33% |

## By domain and request

| domain | request | attempted | executable | schema valid | integrity valid | successful | delivery | conditional | future fail / unverified | future rows | crashes | schema errors | invalid JSON | missing |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| cars | CAR-07 | 18 | 12 | 12 | 12 | 12 | 67% | 100% | 0 / 0 | 0 | 6 | 0 | 0 | 0 |
| cars | CAR-08 | 18 | 12 | 11 | 11 | 10 | 56% | 91% | 0 / 0 | 0 | 6 | 1 | 0 | 0 |
| nasa | NASA-07 | 18 | 9 | 6 | 6 | 5 | 28% | 83% | 0 / 0 | 0 | 9 | 3 | 0 | 0 |
| nasa | NASA-08 | 18 | 9 | 6 | 6 | 6 | 33% | 100% | 0 / 0 | 0 | 9 | 3 | 0 | 0 |
| apache | APACHE-07 | 18 | 14 | 13 | 13 | 12 | 67% | 92% | 0 / 0 | 0 | 4 | 1 | 0 | 0 |
| apache | APACHE-08 | 18 | 14 | 11 | 11 | 11 | 61% | 100% | 0 / 0 | 0 | 4 | 3 | 0 | 0 |

## Limitations

- **References are oracle definitions, not customer outcomes.** 'Best seen', 'high' and 'worse than the median so far' follow the frozen checker's rules applied to recorded values; they are not verified inventory value, project overruns or service incidents.
- **High base rates.** Later arrivals are mostly better (Cars) or worse (Apache) than the history, and above-median effort or defects covers most NASA arrivals, so precision is close to the base rate by default.
- **CAR-07 and APACHE-07 are not prediction.** They judge records whose outcomes are already in the file.
- **The future rerun is partial.** It changes only later arrivals' goals; a program that reads an arrival's own recorded outcome is not caught. NASA-07's self-match share is a post-hoc indicator for that.
- **No APACHE-07 arrival is a worst-ever**, so the frozen required-alert rule never applies.
- **Silence passes the frozen alert rules.** NASA-07 and APACHE-07 checks examine only the alerts that were raised (plus the worst-ever rule, which never applies), so an answer with few or no alerts passes; recall shows what such answers miss.
- **Apache's Pareto front is the tied best rows of a single goal** (2 rows), so APACHE-08 front precision cannot exceed 2/5; the best-seen set is the meaningful reference there.
- **Arrivals and checkpoints are not independent trials.** Three trials per cell, often fewer usable; no statistical test is applied.
- **T7 is not scored in this phase.**
  - NASA and Apache have Flow adapters (`t7_validation/adapters/{nasa,apache}/flow.py`); they could be scored later with this extractor's metric functions on their printed ANSWER lines, after an approved adapter run and a check that they use the same arrival rule (last round(0.2 n) rows) and checkpoints (0, k//2, k).
  - Cars has no Flow adapter (only CAR-03/CAR-20 adapters exist), so CAR-07/08 cannot be scored for T7 without new orchestration code.

## Provenance

- Archive: `harness/results/v2/gpt-oss-20b-3trials.zip` (sha256 `d1a51a3fa3711829ede726dd08e1f77da070fc3e3064fdcd1971d24974eb7683`)
- Code commit: `3e9e0f0969eaf2752af9ad70df2b118dcdb165ec` on `analysis/external-metrics` (uncommitted changes present)
- Command: `python -m harness.external`
