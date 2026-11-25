# Consolidated results, phase 6A — internal and external metrics, T1-T7

**Status:** post-observation synthesis of frozen data and post-hoc external metrics. Nothing was re-run: 0 LLM calls, 0 generated programs, 0 T7 adapters. External metrics were added after the Task Talk and were not pre-registered.

## 1. Experimental design (verified from frozen sources)

**H1 (pre-registered, `docs/hypotheses.md`):** "Neuro-symbolic systems improve when their symbolic half is factored into small reusable structures. The ten skills are synonyms: one machine (cluster, tree, leaf) wearing ten hats. Hence an LLM, or a human, extends a well-factored library with less new code, and fewer errors, than a monolithic one."

**H2:** "The ten skills are ten separate problems. Factoring buys nothing: skill i+1 costs about the same whoever writes it, and whatever library sits underneath."

| item | frozen plan | actual (v2) |
|---|---|---|
| treatments | T1-T6 LLM (plain Python / scikit-learn / ezr × no reuse / reuse); T7 human/team | as planned |
| domains | Cars (auto93, 398 rows), NASA (nasa93dem, 93), Apache (192) | as planned |
| skills | 10 per domain, 20 frozen requests each (60) | as planned |
| LLM trials per domain × treatment | target 10 (course minimum 3, ideal 20) | **3** (54 trials, 540 steps) |
| T7 | one human-directed implementation per domain | 1 per domain (Cars on the LLM-assisted shared library) |
| generation model | recorded per row | `gpt-oss-20b` via `llamacpp`, `gpt-oss-20b-MXFP4.gguf@b12242-3b4d6ab3f`, quantization MXFP4, temperature 1.0, think medium |
| prompt template | recorded per row | `c98c01d01869` (all rows) |
| run date | — | 2026-11-18 (protocol v2, adopted 2026-11-17) |
| budget | none in dollars documented | local open-weights model on an RTX 4080; no API spend is recorded. Recorded usage: 5,540,095 prompt + 2,075,168 response tokens, 4.2 GPU hours |

Sample-size shortfall: 3 of the planned 10 trials per cell. `harness/RUN_ENV.md` documents the v1 setup (2026-11-11, template `2d507109b49a`); the v2 rows record their own model digest, temperature and template hash, which are used here.

## 2. Internal metrics (frozen harness data, failures kept in every denominator)

LOC = newly written nonblank, noncomment lines over skills 2-10, counted only for trials where all nine steps have a LOC value (the frozen complete-trial rule). Semantic = steps whose frozen semantic check passed, out of all 30 steps. Requests = frozen request checks passed, out of 60.

| domain | T | complete LOC trials | LOC 2-10 median [min-max] | steps generated / executable / semantic pass (of 30) | requests pass (of 60) | crashes | schema failures | leakage | budget peeking | LOC share in crashed steps | LOC trend rho |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| cars | T1 | 3/3 | 1700 [1536-1762] | 30 / 27 / 14 | 36 | 3 | 3 | 0 | 5 | 12% | +0.19 |
| cars | T2 | 2/3 | 1548.5 [1448-1649] | 29 / 21 / 10 | 25 | 8 | 1 | 0 | 6 | 31% | -0.15 |
| cars | T3 | 2/3 | 1576 [1495-1657] | 29 / 24 / 15 | 36 | 5 | 2 | 0 | 5 | 20% | +0.40 |
| cars | T4 | 3/3 | 1413 [1261-1544] | 30 / 22 / 10 | 27 | 8 | 5 | 3 | 1 | 30% | +0.17 |
| cars | T5 | 3/3 | 1755 [1591-1988] | 30 / 23 / 11 | 28 | 7 | 2 | 0 | 5 | 29% | +0.31 |
| cars | T6 | 2/3 | 1270 [1257-1283] | 28 / 7 / 5 | 10 | 21 | 4 | 0 | 0 | 76% | +0.27 |
| nasa | T1 | 3/3 | 1917 [1602-2018] | 30 / 23 / 9 | 25 | 7 | 4 | 0 | 5 | 30% | +0.44 |
| nasa | T2 | 2/3 | 1487 [1309-1665] | 29 / 26 / 5 | 17 | 3 | 20 | 0 | 3 | 11% | +0.42 |
| nasa | T3 | 3/3 | 1755 [1667-1773] | 30 / 19 / 5 | 16 | 11 | 9 | 0 | 4 | 43% | +0.20 |
| nasa | T4 | 1/3 | 1774 [1774-1774] | 27 / 20 / 4 | 14 | 7 | 8 | 0 | 3 | 34% | +0.11 |
| nasa | T5 | 3/3 | 1665 [1507-1806] | 30 / 23 / 8 | 26 | 7 | 7 | 0 | 4 | 26% | +0.26 |
| nasa | T6 | 1/3 | 1341 [1341-1341] | 28 / 15 / 3 | 13 | 13 | 4 | 0 | 3 | 49% | +0.45 |
| apache | T1 | 3/3 | 1343 [1256-1626] | 30 / 27 / 10 | 34 | 3 | 11 | 1 | 3 | 12% | +0.41 |
| apache | T2 | 3/3 | 1284 [1202-1759] | 30 / 28 / 13 | 36 | 2 | 10 | 0 | 4 | 9% | +0.17 |
| apache | T3 | 3/3 | 1424 [1258-1651] | 30 / 26 / 19 | 44 | 4 | 2 | 1 | 1 | 16% | +0.51 |
| apache | T4 | 3/3 | 1398 [1158-1565] | 30 / 27 / 16 | 40 | 3 | 9 | 0 | 1 | 3% | +0.40 |
| apache | T5 | 3/3 | 1661 [1640-1687] | 30 / 26 / 22 | 47 | 4 | 2 | 0 | 1 | 16% | +0.43 |
| apache | T6 | 1/3 | 1227 [1227-1227] | 28 / 13 / 11 | 23 | 15 | 2 | 0 | 0 | 57% | +0.28 |

## 3. Matched reuse pairs (reuse vs no reuse, same foundation)

| domain | pair | LOC median reuse / base (complete trials) | same() verdict (decided by) | step semantic reuse / base | requests reuse / base | external delivery reuse / base |
|---|---|---|---|---|---|---|
| cars | T2 vs T1 | 1548.5 / 1700 (2/3) | reuse lower (certified) (Cliff's delta above 0.195) | 33% / 47% | 42% / 60% | 48% / 67% |
| cars | T4 vs T3 | 1413 / 1576 (3/2) | reuse lower (certified) (Cliff's delta above 0.195) | 33% / 50% | 45% / 60% | 48% / 64% |
| cars | T6 vs T5 | 1270 / 1755 (2/3) | reuse lower (certified) (Cliff's delta above 0.195) | 17% / 37% | 17% / 47% | 21% / 45% |
| nasa | T2 vs T1 | 1487 / 1917 (2/3) | reuse lower (certified) (Cliff's delta above 0.195) | 17% / 30% | 28% / 42% | 28% / 47% |
| nasa | T4 vs T3 | 1774 / 1755 (1/3) | no certified difference (medians within eps) | 13% / 17% | 23% / 27% | 25% / 28% |
| nasa | T6 vs T5 | 1341 / 1665 (1/3) | reuse lower (certified) (Cliff's delta above 0.195) | 10% / 27% | 22% / 43% | 19% / 39% |
| apache | T2 vs T1 | 1284 / 1343 (3/3) | no certified difference (medians within eps) | 43% / 33% | 60% / 57% | 64% / 55% |
| apache | T4 vs T3 | 1398 / 1424 (3/3) | no certified difference (medians within eps) | 53% / 63% | 67% / 73% | 69% / 69% |
| apache | T6 vs T5 | 1227 / 1661 (1/3) | reuse lower (certified) (Cliff's delta above 0.195) | 37% / 73% | 38% / 78% | 43% / 76% |

- Reuse has the lower LOC median in **8 of 9** pairs, lower step-level semantic correctness in **8 of 9**, and lower external delivery in **7 of 9**.
- No 'certified' verdict is informative: with n ≤ 3 per side the KS step cannot reject, so every 'different' comes from the Cliff's-delta shortcut (see statistical_validity.md).

## 4. External (business-facing) metrics, all post-hoc

Each request keeps its own success rule; rows are never combined into a composite score. *Delivery* = successes / all attempts (crashes included); *conditional* = successes / integrity-valid answers. The gap between them is the survivorship effect.

| skill | customer question | measured outcome | reference type | attempted | integrity valid | successful | delivery | conditional |
|---|---|---|---|---:|---:|---:|---:|---:|
| Remember (recommended shortlist (-02)) | Which records are our most desirable options? | shortlist precision against the declared-goal truth set | declared goal (Pareto front / best Performance- quartile); -01 is fidelity only | 54 | 43 | 37 | 69% | 86% |
| Guess (prediction (-03); intervals (-04) not measured) | How accurate are our estimates for new records? | MAE on held-out rows vs the historical-mean baseline | observed outcome | 54 | 27 | 24 | 44% | 89% |
| Certify (group comparisons (-05, -06)) | Are observed group differences reliable? | agreement of verdicts with the frozen oracle | statistical oracle (clear_verdict) | 108 | 83 | 72 | 67% | 87% |
| Flow (streaming decisions/predictions (-07), checkpoints (-08)) | Do streaming assessments stay useful as records arrive? | alert/decision precision-recall, streaming MAE ratio, checkpoint shortlist quality | observed outcome (NASA-07 effort) / oracle 'best seen' set / frozen median threshold | 108 | 59 | 56 | 52% | 95% |
| Blame (ranking (-09), contrast (-10), combinations (APACHE-10)) | Which characteristics are associated with poor results? | top-3 overlap with the program-independent association ranking | statistical oracle (|Spearman|, |Cliff's delta|); association, not causation | 108 | 68 | 61 | 56% | 90% |
| Justify (explanations (-11, -12)) | Are explanations truthful about the data? | share of explanatory claims that are true | observed values and declared goals | 108 | 73 | 43 | 40% | 59% |
| Choose (ordered selection (-14); best-row choice (-13) not measured) | Does guided selection find good options within the inspection budget? | normalized regret of the best row inspected within the budget | declared goal (d2h) | 54 | 19 | 11 | 20% | 58% |
| Fix (observed alternatives (Cars/Apache -15, -16); NASA not measurable) | Which observed changes would improve a weak record? | improvement of the proposed observed alternative | observed outcome / declared goal | 72 | 48 | 40 | 56% | 83% |
| Spot | Which records are anomalous? | — | none: not measurable: the datasets carry no anomaly labels, so verified anomaly precision cannot be computed; the frozen checker's internal rules are the only evidence | — | — | — | — | — |
| Race (guided vs random selection (-19/-20 selection request)) | Does guided search beat random sampling? | best d2h found by guided vs random selection | declared goal (d2h) | 54 | 4 | 4 | 7% | 100% |
| Race (model vs historical mean (race_predict request)) | Does the model beat the historical mean? | MAE ratio on held-out rows | observed outcome | 54 | 27 | 20 | 37% | 74% |

Remember -01 (summary fidelity) and Fix for NASA (hypothetical predictions, no observed outcome) are excluded from these counts; Guess -04 and Choose -13 have no external metric.

### External delivery by treatment (successes / attempts over the measured requests above)

Race prediction is left out of these totals because it repeats the Guess predictions in 10 of 17 trials where both are valid; it stays in the per-skill tables.

| domain | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---:|---:|---:|---:|---:|---:|
| cars | 28/42 (67%; cond. 85%) | 20/42 (48%; cond. 80%) | 27/42 (64%; cond. 90%) | 20/42 (48%; cond. 74%) | 19/42 (45%; cond. 73%) | 9/42 (21%; cond. 100%) |
| nasa | 17/36 (47%; cond. 77%) | 10/36 (28%; cond. 59%) | 10/36 (28%; cond. 67%) | 9/36 (25%; cond. 60%) | 14/36 (39%; cond. 74%) | 7/36 (19%; cond. 64%) |
| apache | 23/42 (55%; cond. 82%) | 27/42 (64%; cond. 82%) | 29/42 (69%; cond. 91%) | 29/42 (69%; cond. 100%) | 32/42 (76%; cond. 94%) | 18/42 (43%; cond. 95%) |

### Per-skill delivery by treatment (all domains; successes / attempts)

| skill | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---:|---:|---:|---:|---:|---:|
| Remember | 7/9 | 6/9 | 8/9 | 5/9 | 7/9 | 4/9 |
| Guess | 5/9 | 2/9 | 4/9 | 7/9 | 5/9 | 1/9 |
| Certify | 15/18 | 13/18 | 15/18 | 8/18 | 12/18 | 9/18 |
| Flow | 15/18 | 8/18 | 7/18 | 11/18 | 10/18 | 5/18 |
| Blame | 11/18 | 12/18 | 11/18 | 8/18 | 12/18 | 7/18 |
| Justify | 9/18 | 8/18 | 7/18 | 7/18 | 9/18 | 3/18 |
| Choose | 0/9 | 1/9 | 3/9 | 3/9 | 1/9 | 3/9 |
| Fix | 6/12 | 7/12 | 10/12 | 6/12 | 9/12 | 2/12 |
| Race | 4/18 | 3/18 | 4/18 | 7/18 | 4/18 | 2/18 |

### Quality conditional on valid outcomes (medians over the valid answers of each cell)

| skill / measure | domain | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---|---:|---:|---:|---:|---:|---:|
| Guess CAR-03: median MAE ratio (model/baseline) | cars | 0.50 (1) | 0.64 (2) | 0.56 (2) | 0.56 (1) | 1.45 (2) | — (0) |
| Guess NASA-03: median MAE ratio (model/baseline) | nasa | 0.81 (2) | — (0) | — (0) | 0.83 (3) | 0.83 (1) | 0.88 (2) |
| Guess APACHE-03: median MAE ratio (model/baseline) | apache | 0.69 (2) | 1.00 (1) | 0.45 (2) | 0.45 (3) | 0.48 (3) | — (0) |
| Remember CAR-02: median shortlist precision | cars | 0.80 (3) | 0.80 (3) | 0.80 (3) | 0.80 (3) | 0.80 (3) | 0.80 (1) |
| Remember NASA-02: median shortlist precision | nasa | 0.90 (3) | 0.90 (2) | 0.90 (3) | 0.60 (3) | 0.75 (2) | 0.90 (1) |
| Remember APACHE-02: median shortlist precision | apache | 1.00 (2) | 1.00 (2) | 1.00 (3) | 1.00 (1) | 1.00 (3) | 1.00 (2) |
| Choose CAR-14: median normalized regret (0 = best) | cars | — (0) | — (0) | — (0) | 0.19 (1) | — (0) | — (0) |
| Choose NASA-14: median normalized regret (0 = best) | nasa | — (0) | 0.00 (1) | — (0) | — (0) | 0.39 (1) | 0.39 (1) |
| Choose APACHE-14: median normalized regret (0 = best) | apache | 0.07 (2) | 0.07 (2) | 0.00 (3) | 0.00 (3) | 0.03 (2) | 0.00 (3) |
| Race CAR-19: median advantage over random | cars | — (0) | — (0) | — (0) | 0.10 (1) | — (0) | — (0) |
| Race NASA-20: median advantage over random | nasa | — (0) | — (0) | — (0) | — (0) | — (0) | — (0) |
| Race APACHE-19: median advantage over random | apache | — (0) | — (0) | 0.03 (1) | 0.03 (2) | — (0) | — (0) |
| Blame CAR-09: median rho top-3 overlap | cars | 1.00 (3) | 1.00 (3) | 1.00 (3) | 1.00 (2) | 1.00 (2) | 1.00 (1) |
| Blame NASA-09: median rho top-3 overlap | nasa | 0.50 (2) | 0.67 (1) | 0.33 (1) | — (0) | 0.00 (3) | 0.00 (1) |
| Blame APACHE-09: median rho top-3 overlap | apache | — (0) | 0.67 (3) | 0.67 (2) | 0.67 (2) | 0.67 (3) | 0.67 (2) |

Parentheses give the number of valid answers behind each median (of 3 attempts).

## 5. T7 (one human-directed implementation per domain; n = 1, not an independent trial sample)

| domain | foundation | library baseline LOC | new LOC 2-10 (1-10) | adapter LOC (files) | Guess MAE ratio | T1-T6 Guess MAE ratio medians (valid/3) | T1-T6 LOC 2-10 medians |
|---|---|---:|---|---|---:|---|---|
| cars | human-directed scripts on the LLM-assisted shared shrinking_code/ library | 2255 | 585 (620) | 105 (3) | 0.605 | T1:0.502 [1/3]; T2:0.635 [2/3]; T3:0.563 [2/3]; T4:0.555 [1/3]; T5:1.453 [2/3]; T6:— [0/3] | T1:1700;T2:1548.5;T3:1576;T4:1413;T5:1755;T6:1270 |
| nasa | standalone human-directed modules (human_t7/nasa) | 0 | 1108 (1235) | 829 (10) | 0.819 | T1:0.814 [2/3]; T2:— [0/3]; T3:— [0/3]; T4:0.827 [3/3]; T5:0.827 [1/3]; T6:0.883 [2/3] | T1:1917;T2:1487;T3:1755;T4:1774;T5:1665;T6:1341 |
| apache | standalone human-directed modules (human_t7/apache) | 0 | 1713 (1875) | 624 (10) | 0.450 | T1:0.689 [2/3]; T2:1.000 [1/3]; T3:0.451 [2/3]; T4:0.451 [3/3]; T5:0.484 [3/3]; T6:— [0/3] | T1:1343;T2:1284;T3:1424;T4:1398;T5:1661;T6:1227 |

- **Cars T7 is not independently human-written end to end.** Its 585 new lines (steps 2-10) sit on the 2,255-line `shrinking_code/` library, which was developed with LLM assistance (README, Disclosure). NASA and Apache T7 are standalone modules; their LOC were recomputed here from the `t7-<domain>-stepN` tags with `harness.loc.measure_step` and match the recorded 1,108 and 1,713.
- **Cars LOC uses a different counter** (`experiments/measure_loc.py`, tokenization) from NASA/Apache and T1-T6 (`harness/loc.py`); the counts are close in convention but not identical.
- **Race prediction duplicates Guess** for T7 in every domain (identical predictions), so each domain contributes one prediction result, not two.
- Adapter LOC are orchestration written for the common protocol; they are not part of T7's original development LOC.
- Descriptively (n = 1, no test): NASA T7 wrote fewer new lines than every LLM median (1,108 vs 1,341-1,917); Apache T7 wrote more (1,713 vs 1,227-1,661); Cars T7's 585 lines are not comparable because they extend a 2,255-line library.

**T7 external coverage under the common protocol:**

- Guess: evaluated: common-protocol adapter, all three domains (CAR-03, NASA-03, APACHE-03)
- Race (prediction): evaluated, but the same predictions as Guess in every domain; counted once
- Remember: adapters exist for NASA and Apache only; not evaluated under the common external protocol
- Certify: adapters exist for NASA and Apache (different decision rules from the oracle); not evaluated
- Flow: adapters exist for NASA and Apache (same arrival rule); not evaluated
- Blame: adapters exist for NASA and Apache; not evaluated
- Justify: adapters exist for NASA and Apache; not evaluated
- Choose: adapters exist for NASA and Apache; the honeypot gate is structurally vacuous for T7; not evaluated
- Fix: adapters exist for NASA and Apache; not evaluated
- Race (selection): adapters exist for NASA and Apache; not evaluated
- Spot: not measurable for any treatment (no anomaly labels)

## 6. Hypothesis assessment against the pre-registered win conditions

| H1 win condition | evidence | met? |
|---|---|---|
| 1. Reuse treatments show a downward trend in new LOC across skills | Spearman(new LOC, skill index) is positive in 8 of 9 reuse cells (17 of 18 cells overall): absolute new LOC per skill rises | **no** |
| 2. At least 2 of 3 matched pairs lower for reuse, certified by same() | reuse median lower in 8 of 9 pairs; same() reports 'certified' in 6 of 9, but every one comes from the Cliff shortcut with n ≤ 3 (KS cannot reject), several with one complete reuse trial | **not established** (direction consistent, certification uninformative) |
| 3. Comparable correctness, no savings by omission | reuse has lower step semantic correctness in 8 of 9 pairs; T6 puts 76%, 49%, 57% (Cars, NASA, Apache) of its counted LOC in steps that crashed, and has the lowest external delivery in every domain | **no** |
| 4. Measurable cross-domain reuse, including the third domain | T1-T6 start every domain from scratch, so the experiment never exercises cross-domain transfer for the LLM treatments. Only the LLM-assisted reference library records transfer (410 / 357 implementation lines added for NASA / Apache) | **not tested** for T1-T6 |

**H2** needs flat per-step cost, no certified reuse improvement, and little transferable functionality. Per-step cost is not flat (it rises), reuse is directionally cheaper in LOC, and transfer was not tested, so H2 is not established either. A lack of significance is not evidence of equivalence.

**Verdict: mixed / inconclusive, leaning against H1's 'less code and fewer errors'.** Reuse trajectories that complete tend to write less code, but they fail more often, so the LOC savings coincide with lost functionality. Neither hypothesis is established at n = 3.

**What the external metrics add.** Among programs that work, external quality converges across treatments (many answers are identical), so the treatments differ mainly in how often they deliver a working, integrity-valid answer. Reuse lowers delivery in most pairs; T6 is lowest everywhere.

**Exploratory cross-domain check (not pre-registered).** Pooling complete trials over the three domains (unit = trial): T6 vs T5 4/9 trials, every T6 total below every T5 total (|delta| = 1.00, KS 1.00 > 0.82); T2 vs T1 and T4 vs T3 differ only by the Cliff shortcut (KS does not reject). The T6 result rests on the 4 of 9 T6 trials that completed, so it is survivorship-biased.

## 7. Slide-ready findings (each traceable to the tables above)

1. 54 LLM trials (3 per cell; planned 10), 540 steps, one local model; T7 one implementation per domain.
2. Step-level semantic correctness ranges 10%-73% across the 18 LLM cells.
3. Reuse writes less code in 8/9 matched pairs but is less correct in 8/9.
4. New LOC per skill rises with the skill index in 17 of 18 cells: no shrinking-code trend.
5. same() cannot certify anything at n ≤ 3: KS's critical value exceeds 1; 'certified' verdicts come from Cliff's delta alone.
6. External delivery over all measured requests (Race prediction counted once, with Guess): 348/720 attempts (48%); conditional on a valid answer: 348/424 (82%); the other 296 attempts produced no integrity-valid answer (generation failure, crash, schema error or a failed integrity gate).
7. Among working programs, outputs converge (identical shortlists, rankings, predictions across treatments).
8. T7 beats the historical-mean baseline in all three domains (MAE ratio Cars 0.60, NASA 0.82, Apache 0.45); n = 1 each, no statistics, and the best LLM cells reach similar ratios (Apache T3/T4 0.45).
9. Integrity gates matter: 54 request checks failed the budget-peeking (honeypot) rerun and 5 the held-out leakage rerun; crashed steps account for 131 of 540 steps.
10. Spot cannot be scored externally: the data have no anomaly labels.

## 8. The professor's five instructions

| instruction | status | what remains |
|---|---|---|
| 1. Updated slides (≤ 15; hypotheses first, results, lessons; public GitHub URL on slide 1) | **not done** (out of scope for 6A) | build the deck from sections 1-7; repository origin is `https://github.com/Devansh-Khandor/seai26f-shrinking-code` (check it is public before submitting) |
| 2. Complete T7 | **done for development** (10 skills × 3 domains, tagged); **partial for evaluation** (common-protocol external scoring only for Guess/Race prediction) | decide whether to score the existing NASA/Apache adapters for the other skills; Cars has adapters only for CAR-03/20 |
| 3. Respect the LLM budget | **done as recorded**: one local open-weights model, 3 trials, no paid API; no dollar budget is documented | state the budget assumption explicitly in the slides/paper |
| 4. Internal and external metrics per task | **done for 9 of 10 skills** (Spot not measurable; Guess -04, Choose -13 and NASA Fix not measured) | present the per-skill table and its reference types |
| 5. Compare T1-T7 with the class's statistical methods | **done, with limits**: same()/bestRanks() run and diagnosed; uninformative at n ≤ 3; T7 n = 1 not tested | present as descriptive; cite statistical_validity.md |

## 9. Numerical inconsistencies found

- Phase 3 chat summary said 18 Race predictions beat the baseline; the CSV gives 20 (already flagged in 5A; CSV is correct).
- `harness/RUN_ENV.md` records the v1 environment (template `2d507109b49a`, harness `41f74ac`); the v2 rows use template `c98c01d01869`. Not edited (frozen documentation); v2 provenance is taken from runs.csv.
- An earlier project summary cited '6/9 LOC wins' and '471 tests'; the archive gives reuse lower in 8/9 pairs (median trial totals) and the offline suite has over 900 tests.
- T7 LOC counters differ (Cars: `experiments/measure_loc.py`; NASA/Apache and T1-T6: `harness/loc.py`).
- All other totals re-derived here (540 steps, 1,080 requests, same() verdicts, bestRanks, T7 NASA/Apache LOC 1,108/1,713, T7 MAE ratios) match the frozen or recorded values.
