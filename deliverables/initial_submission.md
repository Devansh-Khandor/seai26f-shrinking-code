# The Shrinking-Code Experiment

**NC State · CSC 591 — AI & Software Engineering · Fall 2026**  
Initial submission · November 9, 2026 · Verified November 8, 2026  
Repository: [https://github.com/Devansh-Khandor/seai26f-shrinking-code](https://github.com/Devansh-Khandor/seai26f-shrinking-code)

## Preregistered hypotheses

**H1 — The Timm Hypothesis**

> Neuro-symbolic systems improve when their symbolic half is factored into small reusable structures. The ten skills are synonyms: one machine (cluster, tree, leaf) wearing ten hats. Hence an LLM, or a human, extends a well-factored library with less new code, and fewer errors, than a monolithic one.

**H2 — The Rival Hypothesis**

> The ten skills are ten separate problems. Factoring buys nothing: skill i+1 costs about the same whoever writes it, and whatever library sits underneath.

## Preregistered win conditions

**Evidence Supporting H1**

H1 will receive support if:

1. Reuse treatments demonstrate an overall downward trend in new LOC across successive skills.
2. At least two of the three matched LLM treatment pairs (T2 versus T1, T4 versus T3, T6 versus T5) exhibit lower new LOC for reuse, with differences certified by `same`.
3. Reuse treatments maintain comparable correctness and do not achieve smaller code sizes by omitting required functionality.
4. The shared framework demonstrates measurable reuse across domains, including the third domain.

Condition 2 is our proposed decision threshold, not a threshold specified by the professor. All comparisons, including unsuccessful ones, will be reported.

**Evidence Supporting H2**

H2 will receive support if new-code-per-step trends remain approximately flat, the reuse treatments do not show certified improvements over the no-reuse baselines, and the shared architecture provides little transferable functionality across domains.

A lack of statistical significance alone will not be treated as proof of equivalence.

**Mixed or Inconclusive Findings**

If results vary across domains, libraries, or skill categories, we will report mixed findings.

We will not change the win conditions after examining the experimental outcomes.

## Three clients, datasets, and frozen requirements

Our three invented clients are an Independent Used-Car Dealership selecting automobiles from `auto93.csv` (398 records), a Software Engineering Project Manager estimating software outcomes from `nasa93dem.csv` (93 records), and an Apache Web Server Administrator optimizing observed configurations from `Apache_AllMeasurements.csv` (192 records). All datasets are in `data/raw/`. The frozen documents [docs/requirements_cars.md](../docs/requirements_cars.md), [docs/requirements_nasa.md](../docs/requirements_nasa.md), and [docs/requirements_apache.md](../docs/requirements_apache.md) each contain 20 skill-tagged customer requests, two per skill. Requirements and hypotheses were frozen November 6, 2026.

## Working implementation and reproducibility

We have implemented all 60 customer requests at commit `ab8b791`, with working demonstrations: ten each for Cars, NASA, and Apache. We verified **229 tests passed, 0 failed** with the full suite, including all 30 demos and unchanged outputs for the previous 25. Our Cars ladder tags: [`step1`](https://github.com/Devansh-Khandor/seai26f-shrinking-code/tree/step1), [`step2`](https://github.com/Devansh-Khandor/seai26f-shrinking-code/tree/step2), [`step3`](https://github.com/Devansh-Khandor/seai26f-shrinking-code/tree/step3), [`step4`](https://github.com/Devansh-Khandor/seai26f-shrinking-code/tree/step4), [`step5`](https://github.com/Devansh-Khandor/seai26f-shrinking-code/tree/step5), [`step6`](https://github.com/Devansh-Khandor/seai26f-shrinking-code/tree/step6), [`step7`](https://github.com/Devansh-Khandor/seai26f-shrinking-code/tree/step7), [`step8`](https://github.com/Devansh-Khandor/seai26f-shrinking-code/tree/step8), [`step9`](https://github.com/Devansh-Khandor/seai26f-shrinking-code/tree/step9), [`step10`](https://github.com/Devansh-Khandor/seai26f-shrinking-code/tree/step10). Transfer completion tags: `nasa-complete`, `apache-complete`.

From the repository root (Python 3, standard library):

```sh
python3 -m shrinking_code.demo --domain <domain> --skill <skill>
python3 -m unittest discover -s tests -v
```

Replace `<domain>` with `cars`, `nasa`, or `apache`; replace `<skill>` with `remember`, `guess`, `certify`, `flow`, `blame`, `justify`, `choose`, `fix`, `spot`, or `race` (the ladder order). For example: `python3 -m shrinking_code.demo --domain apache --skill fix`.
