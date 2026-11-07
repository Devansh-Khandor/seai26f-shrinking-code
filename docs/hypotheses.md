# Pre-Registered Hypotheses — The Shrinking-Code Experiment

**Course:** CSC 591 — AI & Software Engineering  
**Project:** The Shrinking-Code Experiment  
**Version:** 1.0  
**Pre-registration Date:** November 6, 2026  
**Status:** Frozen Experimental Plan — Before Implementation

## 1. Research Question

Does factoring AI capabilities into small, reusable symbolic structures reduce the amount of new code and the number of implementation errors required to add new functionality?

Can the same reusable architecture transfer across three different application domains?

## 2. Competing Hypotheses

### H1 — The Timm Hypothesis

> Neuro-symbolic systems improve when their symbolic half is factored into small reusable structures. The ten skills are synonyms: one machine (cluster, tree, leaf) wearing ten hats. Hence an LLM, or a human, extends a well-factored library with less new code, and fewer errors, than a monolithic one.

### H2 — The Rival Hypothesis

> The ten skills are ten separate problems. Factoring buys nothing: skill i+1 costs about the same whoever writes it, and whatever library sits underneath.

Both hypotheses are reproduced from the course project specification.

## 3. Experimental Domains

We will evaluate the hypotheses across three MOOT datasets.

| Domain | Dataset | Client | Records |
|---|---|---|---:|
| Automobile selection | `auto93.csv` | Used-Car Dealership | 398 |
| Software project estimation | `nasa93dem.csv` | Software Project Manager | 93 |
| Apache server optimization | `Apache_AllMeasurements.csv` | Apache Server Administrator | 192 |

Each domain has 20 frozen customer requirements covering ten skills: Remember, Guess, Certify, Flow, Blame, Justify, Choose, Fix, Spot, and Race.

We will build one shared framework and measure how effectively it transfers between domains.

## 4. Experimental Treatments

Seven implementations will be compared.

| Treatment | Developer | Foundation | Reuse |
|---|---|---|---|
| T1 | LLM | Plain Python | No |
| T2 | LLM | Plain Python | Yes |
| T3 | LLM | scikit-learn | No |
| T4 | LLM | scikit-learn | Yes |
| T5 | LLM | ezr | No |
| T6 | LLM | ezr | Yes |
| T7 | Human/team | Previously developed skills | Yes |

Each treatment must address equivalent customer requirements and pass the same applicable acceptance tests.

The human-written treatment will be kept distinct from LLM-generated implementations. Any LLM involvement will be disclosed.

## 5. Primary Measurements

### 5.1 New Lines of Code

For each verified implementation milestone, record the newly added implementation lines relative to the preceding milestone.

The primary measure is nonblank, noncomment implementation LOC added at that step.

Deleted lines, modifications, test code, generated artifacts, and third-party dependencies will be recorded separately where applicable.

### 5.2 Percentage of New Code

For step i:

**% New Code(i) = 100 × New LOC(i) / Total Implementation LOC(i)**

We will report both the absolute LOC and percentage for every step.

### 5.3 Correctness

Record:

- Number of acceptance tests passed and failed.
- Whether the skill produces the required functionality.
- Runtime errors or unsupported cases.
- Any manual verification findings.

LOC comparisons will distinguish working implementations from incomplete or broken ones.

### 5.4 Cross-Domain Reuse

Measure how much additional implementation code is necessary to support NASA after Cars, and Apache after the previous domains.

Separate changes to the shared framework from domain-specific adapter code.

### 5.5 Statistical Comparisons

Use the professor's `same` procedure, incorporating Cohen, Kolmogorov–Smirnov, and Cliff's delta, to compare treatment results.

Do not describe a numerical difference as statistically meaningful unless it passes the prescribed procedure.

## 6. Pre-Registered Win Conditions

### Evidence Supporting H1

H1 will receive support if:

1. Reuse treatments demonstrate an overall downward trend in new LOC across successive skills.
2. At least two of the three matched LLM treatment pairs (T2 versus T1, T4 versus T3, T6 versus T5) exhibit lower new LOC for reuse, with differences certified by `same`.
3. Reuse treatments maintain comparable correctness and do not achieve smaller code sizes by omitting required functionality.
4. The shared framework demonstrates measurable reuse across domains, including the third domain.

Condition 2 is our proposed decision threshold, not a threshold specified by the professor. All comparisons, including unsuccessful ones, will be reported.

### Evidence Supporting H2

H2 will receive support if new-code-per-step trends remain approximately flat, the reuse treatments do not show certified improvements over the no-reuse baselines, and the shared architecture provides little transferable functionality across domains.

A lack of statistical significance alone will not be treated as proof of equivalence.

### Mixed or Inconclusive Findings

If results vary across domains, libraries, or skill categories, we will report mixed findings.

We will not change the win conditions after examining the experimental outcomes.

## 7. Repetitions and Reset Protocol

The target is **10 independent LLM repetitions** for the experimental treatments.

The professor specifies 3 as the minimum and 20 as the ideal.

Each independent trial must use:

- A fresh execution directory.
- An LLM session with no history from preceding trials.
- Memory and persistent trial-specific context disabled.
- The same frozen requirements for comparable treatments.
- A consistent evaluation procedure.
- Recorded prompts, model identifiers, library versions, and execution settings.

The exact number of completed repetitions will be reported transparently.

## 8. Implementation and Reproducibility

Every completed skill milestone will be committed and Git-tagged as `step1`, `step2`, and so on.

The repository will include automated tests, runnable demonstrations, experiment scripts, code measurements, and raw results.

For ezr-based treatments, the experiment will document the oracle connection and any self-tuning, including the tuning data and parameters.

The implementation must preserve the original MOOT datasets.

## 9. Change Control

This document establishes the experimental plan before implementation.

Any subsequent substantive changes will be documented with a date, rationale, and Git commit reference.

Changes introduced after observing experimental results will be identified separately from this original plan.

## 10. Expected Outcome

We predict that the reusable symbolic architecture will generally require less new code as capabilities accumulate and that at least part of this benefit will transfer across all three domains.

This is a hypothesis, not an experimental result. We will report the actual evidence even if it contradicts our prediction.