# External metric, phase 5F: Certify — do the programs' group-difference verdicts agree with the oracle?

**Status: post-observation analysis.** Added after the Task Talk; not part of the pre-registered plan in `docs/hypotheses.md`. The decision rule is the frozen oracle's own (`harness.oracle.clear_verdict`); no threshold is added.

**What this measures: oracle consistency, not effect detection.** A verdict is scored against the oracle's decision on the same data. That is not ground truth about real differences between vehicles, projects or server settings, and it is **not an A/A false-alarm rate**: the archive has no repeated same-treatment measurements and no simulation protocol for one.

**The oracle's rule.** 'different' when |Cliff's delta| >= 0.474 *and* a two-sample KS test rejects at 95%; 'same' when |delta| < 0.147 *and* KS does not reject; otherwise undecidable, and the frozen checker accepts either verdict. Groups under 5 rows are never decidable. CAR-06 compares the program's own two model-year ranges on three overall scores (d2h, zsum, ranksum); it is decidable only when all three are decidable and agree.

**Outcomes.** *False positive* = said 'different' where the oracle says 'same'. *False negative* = said 'same' where the oracle says 'different'. On undecidable pairs either answer is accepted; 'same' there is the schema's 'not reliably different', the closest thing to an honest 'no clear difference'. The schema allows only 'different' or 'same', so an abstention label (such as 'inconclusive') is a schema error, counted separately.

**Counting.** Comparisons inside one program are not independent trials. Each attempt gets one agreement rate (agree / decidable); cells report the median over attempts. Pooled counts are descriptive totals only. *Delivery* = oracle-consistent answers (no false positive or negative) over all attempts.

## Reference decisions (fixed by the data)

| request | pair | n | Cliff's delta | KS rejects | oracle | why |
|---|---|---:|---:|---|---|---|
| CAR-05 | 3 vs 4 | 4/204 | -0.779 | True | undecidable | group under 5 rows |
| CAR-05 | 3 vs 6 | 4/87 | -0.103 | False | undecidable | group under 5 rows |
| CAR-05 | 3 vs 8 | 4/103 | +0.505 | False | undecidable | group under 5 rows |
| CAR-05 | 4 vs 6 | 204/87 | +0.675 | True | different | large effect and KS rejects |
| CAR-05 | 4 vs 8 | 204/103 | +0.885 | True | different | large effect and KS rejects |
| CAR-05 | 6 vs 8 | 87/103 | +0.555 | True | different | large effect and KS rejects |
| NASA-05 | h vs n | 39/44 | +0.364 | True | undecidable | intermediate evidence |
| NASA-05 | h vs vh | 39/10 | -0.131 | False | same | negligible effect and KS does not reject |
| NASA-05 | n vs vh | 44/10 | -0.520 | True | different | large effect and KS rejects |
| NASA-06 | h vs l | 58/3 | -0.943 | True | undecidable | group under 5 rows |
| NASA-06 | h vs n | 58/10 | -0.248 | False | undecidable | intermediate evidence |
| NASA-06 | h vs vh | 58/17 | -0.258 | False | undecidable | intermediate evidence |
| NASA-06 | h vs xh | 58/5 | -0.503 | False | undecidable | intermediate evidence |
| NASA-06 | l vs n | 3/10 | +0.800 | True | undecidable | group under 5 rows |
| NASA-06 | l vs vh | 3/17 | +0.843 | True | undecidable | group under 5 rows |
| NASA-06 | l vs xh | 3/5 | +0.467 | False | undecidable | group under 5 rows |
| NASA-06 | n vs vh | 10/17 | -0.024 | False | same | negligible effect and KS does not reject |
| NASA-06 | n vs xh | 10/5 | -0.240 | False | undecidable | intermediate evidence |
| NASA-06 | vh vs xh | 17/5 | -0.200 | False | undecidable | intermediate evidence |
| APACHE-05 | 0 vs 1 | 96/96 | -0.813 | True | different | large effect and KS rejects |
| APACHE-06 | 0 vs 1 | 96/96 | +0.216 | True | undecidable | intermediate evidence |

Decidable pairs: CAR-05 3 of 6, NASA-05 2 of 3, NASA-06 1 of 10, APACHE-05 1 of 1, APACHE-06 0 of 1. APACHE-06 has none, so every valid APACHE-06 answer is consistent by default. CAR-06 depends on the ranges each program chose (see below).

## CAR-05 (`groups`)

| T | attempted | executable | valid | decidable per answer | median agreement | FP | FN | undecidable: said same / different | direction agree | stats exact | consistent | delivery |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|---:|
| T1 | 3 | 3 | 3 | 3 | 1.00 | 0 | 0 | 0 / 9 | 9/9 | 3 | 3 | 100% |
| T2 | 3 | 2 | 2 | 3 | 1.00 | 0 | 0 | 0 / 6 | 6/6 | 2 | 2 | 67% |
| T3 | 3 | 3 | 2 | 3 | 1.00 | 0 | 0 | 0 / 6 | 6/6 | 2 | 2 | 67% |
| T4 | 3 | 1 | 1 | 3 | 1.00 | 0 | 0 | 0 / 3 | 3/3 | 1 | 1 | 33% |
| T5 | 3 | 2 | 2 | 3 | 1.00 | 0 | 0 | 2 / 4 | 6/6 | 2 | 2 | 67% |
| T6 | 3 | 2 | 2 | 3 | 1.00 | 0 | 0 | 2 / 4 | 6/6 | 2 | 2 | 67% |

## CAR-06 (`year_groups`)

| T | attempted | executable | valid | decidable per answer | median agreement | FP | FN | undecidable: said same / different | direction agree | stats exact | consistent | delivery |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|---:|
| T1 | 3 | 3 | 3 | 0 | — | 0 | 0 | 0 / 3 | — | 3 | 3 | 100% |
| T2 | 3 | 2 | 2 | 0 | — | 0 | 0 | 0 / 2 | — | 2 | 2 | 67% |
| T3 | 3 | 3 | 3 | 0 | — | 0 | 0 | 0 / 3 | — | 3 | 3 | 100% |
| T4 | 3 | 1 | 1 | 0 | — | 0 | 0 | 0 / 1 | — | 1 | 1 | 33% |
| T5 | 3 | 2 | 2 | 0 | — | 0 | 0 | 0 / 2 | — | 1 | 2 | 67% |
| T6 | 3 | 2 | 2 | 0 | — | 0 | 0 | 0 / 2 | — | 2 | 2 | 67% |

Ranges the valid answers chose (identical choices grouped):

- `70-76;77-82` (n 216;182), 10 answers (T1/1, T1/2, T2/1, T2/3, T3/1, T3/2, T3/3, T5/3, T6/1, T6/2): d2h: n=216/182 delta=+0.446 ks=True oracle=undecidable; zsum: n=216/182 delta=+0.422 ks=True oracle=undecidable; ranksum: n=216/182 delta=+0.409 ks=True oracle=undecidable; combined=undecidable (an overall score is undecidable) said=different -> undecidable_said_different
- `70-75;76-82` (n 182;216), 2 answers (T1/3, T4/2): d2h: n=182/216 delta=+0.420 ks=True oracle=undecidable; zsum: n=182/216 delta=+0.402 ks=True oracle=undecidable; ranksum: n=182/216 delta=+0.390 ks=True oracle=undecidable; combined=undecidable (an overall score is undecidable) said=different -> undecidable_said_different
- `70-76;76-82` (n 216;216) **ranges overlap**, 1 answer (T5/2): d2h: n=216/216 delta=+0.399 ks=True oracle=undecidable; zsum: n=216/216 delta=+0.379 ks=True oracle=undecidable; ranksum: n=216/216 delta=+0.367 ks=True oracle=undecidable; combined=undecidable (an overall score is undecidable) said=different -> undecidable_said_different

## NASA-05 (`groups`)

| T | attempted | executable | valid | decidable per answer | median agreement | FP | FN | undecidable: said same / different | direction agree | stats exact | consistent | delivery |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|---:|
| T1 | 3 | 3 | 3 | 2 | 0.50 | 0 | 3 | 2 / 1 | 3/3 | 3 | 0 | 0% |
| T2 | 3 | 3 | 3 | 2 | 0.50 | 1 | 1 | 0 / 3 | 3/3 | 3 | 1 | 33% |
| T3 | 3 | 3 | 3 | 2 | 0.50 | 0 | 2 | 1 / 2 | 3/3 | 3 | 1 | 33% |
| T4 | 3 | 3 | 2 | 2 | 0.50 | 0 | 2 | 1 / 1 | 2/2 | 2 | 0 | 0% |
| T5 | 3 | 3 | 2 | 2 | 1.00 | 0 | 0 | 0 / 2 | 2/2 | 2 | 2 | 67% |
| T6 | 3 | 1 | 1 | 2 | 1.00 | 0 | 0 | 0 / 1 | 1/1 | 1 | 1 | 33% |

## NASA-06 (`groups`)

| T | attempted | executable | valid | decidable per answer | median agreement | FP | FN | undecidable: said same / different | direction agree | stats exact | consistent | delivery |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|---:|
| T1 | 3 | 3 | 3 | 1 | 1.00 | 0 | 0 | 18 / 9 | — | 3 | 3 | 100% |
| T2 | 3 | 3 | 3 | 1 | 1.00 | 1 | 0 | 8 / 19 | — | 3 | 2 | 67% |
| T3 | 3 | 3 | 3 | 1 | 1.00 | 0 | 0 | 12 / 15 | — | 3 | 3 | 100% |
| T4 | 3 | 3 | 2 | 1 | 1.00 | 0 | 0 | 12 / 6 | — | 2 | 2 | 67% |
| T5 | 3 | 3 | 2 | 1 | 1.00 | 0 | 0 | 0 / 18 | — | 2 | 2 | 67% |
| T6 | 3 | 1 | 1 | 1 | 1.00 | 0 | 0 | 0 / 9 | — | 1 | 1 | 33% |

## APACHE-05 (`groups`)

| T | attempted | executable | valid | decidable per answer | median agreement | FP | FN | undecidable: said same / different | direction agree | stats exact | consistent | delivery |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|---:|
| T1 | 3 | 3 | 3 | 1 | 1.00 | 0 | 0 | 0 / 0 | 3/3 | 3 | 3 | 100% |
| T2 | 3 | 3 | 3 | 1 | 1.00 | 0 | 0 | 0 / 0 | 3/3 | 3 | 3 | 100% |
| T3 | 3 | 3 | 3 | 1 | 1.00 | 0 | 0 | 0 / 0 | 3/3 | 3 | 3 | 100% |
| T4 | 3 | 3 | 2 | 1 | 1.00 | 0 | 0 | 0 / 0 | 2/2 | 2 | 2 | 67% |
| T5 | 3 | 2 | 2 | 1 | 1.00 | 0 | 0 | 0 / 0 | 2/2 | 2 | 2 | 67% |
| T6 | 3 | 2 | 2 | 1 | 0.50 | 0 | 1 | 0 / 0 | 2/2 | 2 | 1 | 33% |

## APACHE-06 (`groups`)

| T | attempted | executable | valid | decidable per answer | median agreement | FP | FN | undecidable: said same / different | direction agree | stats exact | consistent | delivery |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|---:|
| T1 | 3 | 3 | 3 | 0 | — | 0 | 0 | 2 / 1 | — | 3 | 3 | 100% |
| T2 | 3 | 3 | 3 | 0 | — | 0 | 0 | 1 / 2 | — | 3 | 3 | 100% |
| T3 | 3 | 3 | 3 | 0 | — | 0 | 0 | 1 / 2 | — | 3 | 3 | 100% |
| T4 | 3 | 3 | 2 | 0 | — | 0 | 0 | 1 / 1 | — | 2 | 2 | 67% |
| T5 | 3 | 2 | 2 | 0 | — | 0 | 0 | 0 / 2 | — | 2 | 2 | 67% |
| T6 | 3 | 2 | 2 | 0 | — | 0 | 0 | 0 / 2 | — | 2 | 2 | 67% |

## By domain and request

| domain | request | attempted | executable | schema valid | integrity valid | consistent | delivery | conditional | FP | FN | pooled agree / decidable | abstention labels | missing pairs | shuffled verdict failures (valid) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| cars | CAR-05 | 18 | 13 | 12 | 12 | 12 | 67% | 100% | 0 | 0 | 36 / 36 | 0 | 0 | 0 |
| cars | CAR-06 | 18 | 13 | 13 | 13 | 13 | 72% | 100% | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| nasa | NASA-05 | 18 | 16 | 14 | 14 | 5 | 28% | 36% | 1 | 8 | 19 / 28 | 0 | 0 | 9 |
| nasa | NASA-06 | 18 | 16 | 14 | 14 | 13 | 72% | 93% | 1 | 0 | 13 / 14 | 0 | 0 | 1 |
| apache | APACHE-05 | 18 | 16 | 15 | 15 | 14 | 78% | 93% | 0 | 1 | 14 / 15 | 0 | 0 | 1 |
| apache | APACHE-06 | 18 | 16 | 15 | 15 | 15 | 83% | 100% | 0 | 0 | 0 / 0 | 0 | 0 | 0 |

## Limitations

- **Oracle consistency only.** Agreement means a verdict matches `clear_verdict` on the recorded rows. It does not show that a difference is real, causal or useful for inventory, planning or configuration decisions.
- **No A/A false-alarm rate.** That needs repeated measurements of the same treatment (or a specified simulation); neither exists in the frozen archive.
- **Few decidable comparisons.** Most pairs fall between the oracle's 'same' and 'different' bands or involve a group under 5 rows, where any verdict is accepted. Such answers are consistent by construction.
- **No effect-size field.** The schema reports only n, median and mean per group, so effect-size error cannot be measured; group-statistic accuracy and the direction of the reported means are reported instead.
- **Verdicts are not independent.** Pairs within one answer share groups and code; one trial is one attempt.
- **Three trials per cell, often fewer usable.** No statistical test is applied.
- **T7 is not executed in this phase.** Existing adapters and their protocol differences:
  - NASA: `t7_validation/adapters/nasa/certify.py` keeps the original T7 grouping and permutation test, and adds a separate Cliff's delta + KS verdict as an adapter policy that the original T7 demo did not have.
  - Apache: `t7_validation/adapters/apache/certify.py` says 'different' when the original T7 permutation p < 0.05 and |Cliff's delta| >= 0.474: a different rule from the oracle's (no KS step, no 'same' band).
  - Cars: no Certify adapter; `human_t7/cars/certify.py` compares each cylinder count only against 4 cylinders (not every pair) with the shared library's compare() (its own Cohen 0.35, KS 1.36 and Cliff's 0.195 thresholds) and does not print the common ANSWER format.

## Provenance

- Archive: `harness/results/v2/gpt-oss-20b-3trials.zip` (sha256 `d1a51a3fa3711829ede726dd08e1f77da070fc3e3064fdcd1971d24974eb7683`)
- Code commit: `3e9e0f0969eaf2752af9ad70df2b118dcdb165ec` on `analysis/external-metrics` (uncommitted changes present)
- Command: `python -m harness.external`
