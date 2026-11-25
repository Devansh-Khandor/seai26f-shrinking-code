# Statistical validity of the treatment comparisons (phase 6A)

Tools: the professor's `same()` and `bestRanks()` from `stats.py` (vendored unchanged in `harness/vendor_stats.py`; `the.stats` = cliffs 0.195, conf 1.36, eps 0.35), called through `harness/analyze.py` (eps = 0.35 × sd of the lower-median sample).

## How same() decides

1. If the two medians differ by at most eps, the samples are *same*.
2. Otherwise, if |Cliff's delta| > 0.195, they are *different*.
3. Otherwise the KS statistic is compared with 1.36 × sqrt((n+m)/(nm)).

**KS cannot reject at these sizes.** The statistic is at most 1, and the critical value is above 1 whenever both samples have 3 or fewer values (n = m = 3: 1.11; 2 vs 3: 1.24; 1 vs 3: 1.57). So with n ≤ 3 every 'different' comes from step 2 alone. With 3 vs 3 values, |delta| > 0.195 means only two of the nine pairs need to be out of order. With a single value whose median is the lower one, its sd is 0, so eps = 0 and any median gap moves on to the Cliff step, where one value below all three others gives |delta| = 1.

## Matched pairs, trial totals of new LOC over skills 2-10 (the primary frozen sample)

| domain | pair | n reuse/base | reuse median | base median | eps | median gap | abs Cliff | KS / critical | decided by | verdict | informative? |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| cars | T2 vs T1 | 2/3 | 1548.5 | 1700 | 49.7 | 51.0 | 0.67 | 0.67 / 1.24 | Cliff's delta above 0.195 -> different | reuse lower (certified) | no |
| cars | T4 vs T3 | 3/2 | 1413 | 1576 | 49.6 | 244.0 | 0.67 | 0.67 / 1.24 | Cliff's delta above 0.195 -> different | reuse lower (certified) | no |
| cars | T6 vs T5 | 2/3 | 1270 | 1755 | 6.4 | 472.0 | 1.00 | 1.00 / 1.24 | Cliff's delta above 0.195 -> different | reuse lower (certified) | no |
| nasa | T2 vs T1 | 2/3 | 1487 | 1917 | 88.1 | 252.0 | 0.67 | 0.67 / 1.24 | Cliff's delta above 0.195 -> different | reuse lower (certified) | no |
| nasa | T4 vs T3 | 1/3 | 1774 | 1755 | 19.9 | 19.0 | 1.00 | 1.00 / 1.57 | medians within eps -> same | no certified difference | no |
| nasa | T6 vs T5 | 1/3 | 1341 | 1665 | 0.0 | 324.0 | 1.00 | 1.00 / 1.57 | Cliff's delta above 0.195 -> different | reuse lower (certified) | no |
| apache | T2 vs T1 | 3/3 | 1284 | 1343 | 105.3 | 59.0 | 0.11 | 0.33 / 1.11 | medians within eps -> same | no certified difference | no |
| apache | T4 vs T3 | 3/3 | 1398 | 1424 | 71.6 | 26.0 | 0.33 | 0.33 / 1.11 | medians within eps -> same | no certified difference | no |
| apache | T6 vs T5 | 1/3 | 1227 | 1661 | 0.0 | 434.0 | 1.00 | 1.00 / 1.57 | Cliff's delta above 0.195 -> different | reuse lower (certified) | no |

The frozen tables report 6 of 9 trial-total pairs as 'reuse lower (certified)'. All of them are decided by the Cliff step with at most 3 values per side, and several rest on one complete reuse trial (n = 1). They show that every complete reuse trial was smaller than most baseline trials, not a statistically certified effect.

## Pooled per-step LOC (secondary frozen sample)

Pooling the 9 steps of each trial gives n ≈ 24-27 per side, where KS can reject. But the steps of one trial share one development history and differ by skill, so they are not independent development trials; these verdicts are exploratory at best.

| domain | pair | n reuse/base | reuse median | base median | decided by | verdict |
|---|---|---|---:|---:|---|---|
| cars | T2 vs T1 | 26/27 | 164 | 178 | medians within eps -> same | no certified difference |
| cars | T4 vs T3 | 27/26 | 155 | 166 | medians within eps -> same | no certified difference |
| cars | T6 vs T5 | 25/27 | 135 | 196 | Cliff's delta above 0.195 -> different | reuse lower (certified) |
| nasa | T2 vs T1 | 26/27 | 174 | 193 | Cliff's delta above 0.195 -> different | reuse lower (certified) |
| nasa | T4 vs T3 | 24/27 | 174.5 | 200 | KS decides | no certified difference |
| nasa | T6 vs T5 | 25/27 | 133 | 185 | Cliff's delta above 0.195 -> different | reuse lower (certified) |
| apache | T2 vs T1 | 27/27 | 160 | 150 | medians within eps -> same | no certified difference |
| apache | T4 vs T3 | 27/27 | 152 | 150 | medians within eps -> same | no certified difference |
| apache | T6 vs T5 | 25/27 | 130 | 180 | Cliff's delta above 0.195 -> different | reuse lower (certified) |

## bestRanks()

bestRanks sorts treatments by median and keeps those `same()` as the lowest. With 1-3 values per treatment, 'tied for best' inherits every limitation above, and treatments with fewer complete trials (survivors) enter with fewer values. Correctness ranks use skills passing per trial (negated so lower is better).

| domain | LOC: tied for best | complete trials | correctness: tied for best |
|---|---|---|---|
| cars | T6 | T1 3, T2 2, T3 2, T4 3, T5 3, T6 2 | T3 |
| nasa | T6, T2 | T1 3, T2 2, T3 3, T4 1, T5 3, T6 1 | T1, T5 |
| apache | T6 | T1 3, T2 3, T3 3, T4 3, T5 3, T6 1 | T5 |

## Exploratory cross-domain comparison

Unit = one complete trial (domain × trial), pooled over the three domains, so n ≤ 9 per side and KS can reject. Domains differ in size and difficulty, the pairs are balanced only by design, and incomplete trials drop out, so this is exploratory and not a pre-registered test.

| pair | n reuse/base | reuse median | base median | abs Cliff | KS / critical | decided by | same? |
|---|---|---:|---:|---:|---|---|---|
| T2 vs T1 | 7/9 | 1448 | 1626 | 0.37 | 0.35 / 0.69 | Cliff's delta above 0.195 -> different | False |
| T4 vs T3 | 7/8 | 1413 | 1654 | 0.39 | 0.48 / 0.70 | Cliff's delta above 0.195 -> different | False |
| T6 vs T5 | 4/9 | 1270 | 1665 | 1.00 | 1.00 / 0.82 | Cliff's delta above 0.195 -> different | False |

## What must stay descriptive

- Every external metric: at most 3 attempts per cell, often 0-2 valid; no same() is applied.
- T7: one implementation per domain (n = 1), so same() and bestRanks() are not applied to T7.
- Guess and Race predictions: identical in 10 of 17 trials where both are valid, so they are not independent evidence.
- Skill requests within a trial are not independent development trials; neither are held-out rows, arrivals or checkpoints.
