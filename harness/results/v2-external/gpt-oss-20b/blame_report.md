# External metric, phase 5H: Blame — do the programs name characteristics really associated with poor results?

**Status: post-observation analysis.** Added after the Task Talk; not part of the pre-registered plan in `docs/hypotheses.md`. Associations are recomputed with the frozen oracle's own measures.

**Association, not causation.** Every measure here is an association in recorded rows. None says that changing a setting or characteristic would change an outcome.

**Two references.** *delta*: |Cliff's delta| of each feature between the program's poor/bad rows and the rest/good rows (`harness.oracle.association`; it depends on the program's own groups). *rho*: |Spearman| of each feature with the outcome over all rows (`harness.oracle.correlation`; d2h for CAR-10's overall outcome); it does not depend on the program. Symbolic columns that are not ratings have no score.

**Ranking metrics.** Top-3 sets include ties. *Top-3 overlap* = share of the program's first three features in the oracle top 3. *Strongest hit* = an oracle-strongest feature is among the first three. *First in top 3* is the frozen checker's own pass rule (oracle agreement, not independent truth). *Spearman* is computed only with at least 4 comparable ranked features (post-hoc minimum). *Chance* = the overlap expected from 3 random candidates.

**Groups and combinations.** Membership is checked against the CSV. *Outcome delta* = Cliff's delta of the outcome score between bad and good/rest rows (positive = bad worse). Contrast values and combination shares are recomputed; wrong ones are false claims. For APACHE-10, *lift* = bad share - good share, compared with the best 2-setting combination (post-hoc omission check), and each combination is checked against all rows.

**Integrity and success.** Integrity-valid = ran and passed the frozen format check; the shuffled rerun is recorded only. *Success* = all frozen base checks pass; *delivery* = successes / all attempts.

## Program-independent reference (|Spearman| with the outcome, all rows)

| request | outcome | scored features | strongest five (all non-goal columns) | strongest five (inputs only) |
|---|---|---:|---|---|
| CAR-09 | Mpg+ | 5 | HpX 0.80; Volume 0.79; Clndrs 0.74; Model 0.53; origin 0.52 | Volume 0.79; Clndrs 0.74; Model 0.53; origin 0.52 |
| CAR-10 | overall (d2h) | 5 | HpX 0.92; Volume 0.88; Clndrs 0.83; origin 0.55; Model 0.45 | Volume 0.88; Clndrs 0.83; origin 0.55; Model 0.45 |
| NASA-09 | Effort- | 26 | MonthsX 0.91; centerX 0.63; idX 0.56; YearX 0.51; stor 0.39 | stor 0.39; time 0.38; data 0.37; acap 0.35; pcap 0.35 |
| NASA-10 | Defects- | 26 | MonthsX 0.95; centerX 0.58; YearX 0.49; idX 0.43; pcap 0.41 | pcap 0.41; acap 0.39; data 0.37; plex 0.28; apex 0.25 |
| APACHE-09 | Performance- | 9 | keepAlive 0.70; handle 0.38; inMemory 0.34; hostnameLookups 0.32; accessLog 0.19 | keepAlive 0.70; handle 0.38; inMemory 0.34; hostnameLookups 0.32; accessLog 0.19 |
| APACHE-10 | Performance- | 9 | keepAlive 0.70; handle 0.38; inMemory 0.34; hostnameLookups 0.32; accessLog 0.19 | keepAlive 0.70; handle 0.38; inMemory 0.34; hostnameLookups 0.32; accessLog 0.19 |

Columns ending in X (identifiers, dates, durations) are not model inputs under the protocol. In NASA they dominate the all-column ranking (project duration, centre, id), so a ranking that leads with them agrees with the oracle while naming characteristics that are not usable planning inputs.

## CAR-09 (`blame_rank`)

| T | attempted | executable | valid | candidates | chance | delta top-3 overlap | delta strongest | rho top-3 overlap | rho strongest | first in top 3 (delta/rho) | spearman (delta/rho, n) | bad percentile | outcome delta | successful | delivery |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| T1 | 3 | 3 | 3 | 4/5 | 0.75 | 1.00 | 3 | 1.00 | 3 | 3/3 | 0.80/0.80, 1 | 0.87 | 1.00 | 3 | 100% |
| T2 | 3 | 3 | 3 | 4/5 | 0.75 | 1.00 | 3 | 1.00 | 3 | 3/3 | 1.00/1.00, 1 | 0.87 | 1.00 | 3 | 100% |
| T3 | 3 | 3 | 3 | 4/5 | 0.60 | 0.67 | 3 | 1.00 | 3 | 3/3 | 0.80/0.80, 1 | 0.87 | 1.00 | 3 | 100% |
| T4 | 3 | 3 | 2 | 4 | 0.75 | 1.00 | 2 | 1.00 | 2 | 2/2 | —/—, 0 | 0.87 | 0.88 | 2 | 67% |
| T5 | 3 | 2 | 2 | 4 | 0.75 | 1.00 | 2 | 1.00 | 2 | 2/2 | 1.00/1.00, 1 | 0.42 | 0.78 | 1 | 33% |
| T6 | 3 | 1 | 1 | 4 | 0.75 | 1.00 | 1 | 1.00 | 1 | 1/1 | —/—, 0 | 0.87 | 1.00 | 1 | 33% |

## CAR-10 (`blame_contrast`)

| T | attempted | executable | valid | candidates | chance | delta top-3 overlap | delta strongest | rho top-3 overlap | rho strongest | first in top 3 (delta/rho) | spearman (delta/rho, n) | bad percentile | outcome delta | false values / checked | successful | delivery |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|
| T1 | 3 | 3 | 3 | 4/5 | 0.75 | 0.67 | 3 | 0.67 | 3 | 3/3 | 0.00/0.40, 1 | 0.87 | 1.00 | 0 / 20 | 3 | 100% |
| T2 | 3 | 3 | 3 | 4/5 | 0.75 | 1.00 | 3 | 0.67 | 3 | 3/3 | —/—, 0 | 0.87 | 1.00 | 0 / 18 | 2 | 67% |
| T3 | 3 | 3 | 3 | 4/5 | 0.60 | 0.67 | 3 | 0.67 | 3 | 3/3 | 0.30/0.60, 1 | 0.90 | 1.00 | 0 / 22 | 3 | 100% |
| T4 | 3 | 3 | 2 | 4 | 0.88 | 1.00 | 2 | 0.67 | 2 | 2/2 | —/—, 0 | 0.97 | 1.00 | 0 / 12 | 2 | 67% |
| T5 | 3 | 2 | 2 | 4 | 0.88 | 1.00 | 2 | 0.67 | 2 | 2/2 | 0.40/0.40, 1 | 0.90 | 0.81 | 0 / 14 | 2 | 67% |
| T6 | 3 | 1 | 1 | 4 | 0.75 | 0.67 | 1 | 0.67 | 1 | 1/1 | —/—, 0 | 0.95 | 1.00 | 0 / 6 | 1 | 33% |

## NASA-09 (`blame_rank`)

| T | attempted | executable | valid | candidates | chance | delta top-3 overlap | delta strongest | rho top-3 overlap | rho strongest | first in top 3 (delta/rho) | spearman (delta/rho, n) | bad percentile | outcome delta | successful | delivery |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| T1 | 3 | 3 | 2 | 22/26 | 0.13 | 0.67 | 2 | 0.50 | 1 | 1/2 | —/—, 0 | 0.97 | 1.00 | 2 | 67% |
| T2 | 3 | 3 | 1 | 22 | 0.14 | 0.67 | 0 | 0.67 | 1 | 1/1 | —/—, 0 | 0.97 | 1.00 | 1 | 33% |
| T3 | 3 | 2 | 1 | 22 | 0.14 | 1.00 | 1 | 0.33 | 0 | 1/1 | 0.52/0.58, 1 | 0.96 | 1.00 | 1 | 33% |
| T4 | 3 | 0 | 0 | — | — | — | 0 | — | 0 | 0/0 | —/—, 0 | — | — | 0 | 0% |
| T5 | 3 | 3 | 3 | 22/26 | 0.14 | 0.33 | 2 | 0.00 | 1 | 1/1 | 0.76/0.76, 2 | 0.97 | 1.00 | 2 | 67% |
| T6 | 3 | 1 | 1 | 22 | 0.14 | 0.33 | 1 | 0.00 | 0 | 1/0 | —/—, 0 | 0.97 | 1.00 | 1 | 33% |

## NASA-10 (`blame_contrast`)

| T | attempted | executable | valid | candidates | chance | delta top-3 overlap | delta strongest | rho top-3 overlap | rho strongest | first in top 3 (delta/rho) | spearman (delta/rho, n) | bad percentile | outcome delta | false values / checked | successful | delivery |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|
| T1 | 3 | 3 | 3 | 22 | 0.18 | 0.67 | 2 | 0.33 | 0 | 2/0 | 0.57/0.39, 1 | 0.97 | 1.00 | 0 / 56 | 2 | 67% |
| T2 | 3 | 3 | 1 | 22 | 0.18 | 0.00 | 0 | 0.00 | 0 | 0/0 | —/—, 0 | 0.97 | 1.00 | 0 / 6 | 0 | 0% |
| T3 | 3 | 2 | 1 | 22 | 0.18 | 0.67 | 1 | 0.67 | 0 | 1/0 | 0.57/0.38, 1 | 0.97 | 1.00 | 0 / 44 | 1 | 33% |
| T4 | 3 | 0 | 0 | — | — | — | 0 | — | 0 | 0/0 | —/—, 0 | — | — | 0 / 0 | 0 | 0% |
| T5 | 3 | 3 | 2 | 22/26 | 0.15 | 0.67 | 2 | 0.67 | 1 | 1/0 | —/—, 0 | 0.93 | 1.00 | 0 / 12 | 1 | 33% |
| T6 | 3 | 1 | 0 | — | — | — | 0 | — | 0 | 0/0 | —/—, 0 | — | — | 0 / 0 | 0 | 0% |

## APACHE-09 (`blame_rank`)

| T | attempted | executable | valid | candidates | chance | delta top-3 overlap | delta strongest | rho top-3 overlap | rho strongest | first in top 3 (delta/rho) | spearman (delta/rho, n) | bad percentile | outcome delta | successful | delivery |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| T1 | 3 | 1 | 0 | — | — | — | 0 | — | 0 | 0/0 | —/—, 0 | — | — | 0 | 0% |
| T2 | 3 | 3 | 3 | 9 | 0.33 | 1.00 | 3 | 0.67 | 3 | 3/3 | 0.98/0.85, 2 | 0.94 | 1.00 | 3 | 100% |
| T3 | 3 | 2 | 2 | 9 | 0.39 | 0.50 | 1 | 0.67 | 1 | 1/1 | 0.70/0.98, 1 | 0.93 | 1.00 | 1 | 33% |
| T4 | 3 | 3 | 2 | 9 | 0.39 | 1.00 | 2 | 0.67 | 2 | 2/2 | —/—, 0 | 0.96 | 1.00 | 2 | 67% |
| T5 | 3 | 3 | 3 | 9 | 0.33 | 0.67 | 3 | 0.67 | 3 | 3/3 | 0.05/0.05, 1 | 0.94 | 1.00 | 3 | 100% |
| T6 | 3 | 2 | 2 | 9 | 0.39 | 1.00 | 2 | 0.67 | 2 | 2/2 | 0.99/0.88, 1 | 0.98 | 1.00 | 2 | 67% |

## APACHE-10 (`blame_combos`)

| T | attempted | executable | valid | combinations | false share claims | first typical of bad | median lift / best | worse over all rows | bad percentile | outcome delta | successful | delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 3 | 1 | 1 | 3 | 0 | 1 | 1.00 | 3 | 0.94 | 1.00 | 1 | 33% |
| T2 | 3 | 3 | 3 | 38 | 0 | 3 | 1.00 | 34 | 0.94 | 1.00 | 3 | 100% |
| T3 | 3 | 2 | 2 | 2 | 0 | 2 | 1.00 | 2 | 0.93 | 1.00 | 2 | 67% |
| T4 | 3 | 3 | 2 | 2 | 0 | 2 | 1.00 | 2 | 0.96 | 1.00 | 2 | 67% |
| T5 | 3 | 3 | 3 | 74 | 0 | 3 | 1.00 | 62 | 0.94 | 1.00 | 3 | 100% |
| T6 | 3 | 2 | 2 | 10 | 0 | 2 | 0.95 | 10 | 0.98 | 1.00 | 2 | 67% |

## By domain and request

| domain | request | attempted | executable | schema valid | integrity valid | successful | delivery | conditional | crashes | schema errors | invalid JSON | missing | shuffled failures (valid) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cars | CAR-09 | 18 | 15 | 14 | 14 | 13 | 72% | 93% | 3 | 1 | 0 | 0 | 1 |
| cars | CAR-10 | 18 | 15 | 14 | 14 | 13 | 72% | 93% | 3 | 1 | 0 | 0 | 1 |
| nasa | NASA-09 | 18 | 12 | 8 | 8 | 7 | 39% | 88% | 6 | 4 | 0 | 0 | 1 |
| nasa | NASA-10 | 18 | 12 | 7 | 7 | 4 | 22% | 57% | 6 | 5 | 0 | 0 | 3 |
| apache | APACHE-09 | 18 | 14 | 12 | 12 | 11 | 61% | 92% | 4 | 2 | 0 | 0 | 1 |
| apache | APACHE-10 | 18 | 14 | 13 | 13 | 13 | 72% | 100% | 4 | 1 | 0 | 0 | 0 |

## Limitations

- **Association, not causation.** A strongly associated feature may be a proxy (for example model year or cylinders for weight); nothing here supports changing it to improve an outcome.
- **Oracle agreement, not expert ground truth.** The references are the frozen oracle's own measures; no expert ranking exists. 'First in top 3' repeats the frozen pass rule.
- **Few candidates make overlap easy.** Cars has 5 non-goal columns (4 inputs), so 3 random features already overlap most of the top 3; compare every overlap with the chance column.
- **The delta reference depends on the program's own groups**; the rho reference does not, but it measures association with the outcome over all rows rather than with the program's 'poor' definition.
- **Short rankings.** Most rankings list only 3 features, so rank agreement is rarely computable.
- **Identifiers pass.** A ranking that leads with X columns such as a project id or duration (`idX`, `MonthsX`) agrees with the all-column oracle and passes the frozen rule, though it names nothing a planner could act on; x_columns_ranked records it.
- **Tiny groups.** Several NASA answers use the minimum 5 poor/bad rows, so their delta reference rests on 5 rows.
- **Three trials per cell, often fewer usable.** No statistical test is applied.
- **T7 is not executed in this phase.**
  - NASA and Apache have Blame adapters (`t7_validation/adapters/{nasa,apache}/blame.py`) that print the common ANSWER lines; they could be scored later with blame_metrics after an approved adapter run (the definitions use only the answer and the CSV).
  - Cars has no Blame adapter, so CAR-09/10 cannot be scored for T7 without new orchestration code.

## Provenance

- Archive: `harness/results/v2/gpt-oss-20b-3trials.zip` (sha256 `d1a51a3fa3711829ede726dd08e1f77da070fc3e3064fdcd1971d24974eb7683`)
- Code commit: `77dd4321306489cf77ea5640d96d5571453abb85` on `analysis/external-metrics` (uncommitted changes present)
- Command: `python -m harness.external`
