# NASA Software Project Estimation — Experiment and Reproducibility

**Project:** The Shrinking-Code Experiment  
**Course:** CSC 591 — AI & Software Engineering  
**Dataset:** `data/raw/nasa93dem.csv`  
**Implementation directory:** `human_t7/nasa/`  
**Development branch:** `human-t7-nasa`  
**Frozen requirements:** [`docs/requirements_nasa.md`](../../docs/requirements_nasa.md), version 1.0 (November 6, 2026)  
**Final implementation checkpoint:** `t7-nasa-step10` (`ddb3e0b`)  
**Results and interpretation:** [`RESULTS.md`](RESULTS.md)

> **Scope.** This document records the implementation setup, algorithms, requirement mapping, experiment controls, Git checkpoints, line-counting procedure, and verification commands. Numerical findings and charts are maintained in `RESULTS.md` rather than duplicated here.

## 1. Research objective and scope

The NASA domain uses 20 frozen customer requirements, grouped into ten sequential skills. Each skill introduces a new capability while allowing previously implemented modules to be reused. The experiment records the amount of newly introduced implementation code at successive checkpoints and evaluates selected capabilities against simple baselines.

The implementation provides a research prototype, not a production project-management service. Its output is exploratory and depends on the recorded sample, feature representation, and evaluation setup.

### Requirement-to-module mapping

| Step | Skill | Frozen requirements | Implementation file | Primary responsibility |
|---:|---|---|---|---|
| 1 | Remember | NASA-01, NASA-02 | `remember.py` | Load and summarize historical records; rank outcomes |
| 2 | Guess | NASA-03, NASA-04 | `guess.py` | Neighbor-based effort and defect estimates |
| 3 | Certify | NASA-05, NASA-06 | `certify.py` | Statistical comparison of selected rating groups |
| 4 | Flow | NASA-07, NASA-08 | `flow.py` | Sequential proposal assessment and history updates |
| 5 | Blame | NASA-09, NASA-10 | `blame.py` | Identify attributes associated with outcome extremes |
| 6 | Justify | NASA-11, NASA-12 | `justify.py` | Neighbor explanations and proposal comparisons |
| 7 | Choose | NASA-13, NASA-14 | `choose.py` | Historical examples and budget-limited inspections |
| 8 | Fix | NASA-15, NASA-16 | `fix.py` | Search hypothetical adjustments to selected settings |
| 9 | Spot | NASA-17, NASA-18 | `spot.py` | Input-distance and outcome-residual anomaly detection |
| 10 | Race | NASA-19, NASA-20 | `race.py` | Prediction and project-selection baseline comparisons |

The authoritative wording and acceptance intent remain in the frozen requirements file; the descriptions above are implementation summaries, not replacements for that specification.

## 2. Data and experimental partitions

- **Source:** `data/raw/nasa93dem.csv`, resolved relative to the repository root by `remember.py`.
- **Records:** 93 software projects; dataset row numbers in program output are 1-based and refer to the original data order.
- **Inputs:** 22 categorical project characteristics used by the neighbor-based methods. Columns ending in `X` are ignored by feature selection; goals and the generated row identifier are not predictors.
- **Goals:** `Kloc+` (maximize), `Effort-` (minimize), and `Defects-` (minimize), following the frozen dataset's header convention.
- **Training/history partition:** first 74 records.
- **Held-out partition:** final 19 records (rows 75–93).

For the proposal-evaluation methods, held-out goals are withheld during ranking and prediction, then revealed only when an evaluation requires observed outcomes. This is a *fixed ordered split*, not randomized cross-validation. Some descriptive historical analyses, such as the final Spot analysis, operate across all 93 completed records rather than claiming an unseen evaluation sample.

### Data-flow boundaries

```mermaid
flowchart TD
    D[93 historical NASA records] --> H[First 74: initial history]
    D --> T[Last 19: held-out proposals]
    H --> M[Model fitting and historical statistics]
    T --> X[Mask goal columns for proposals]
    M --> P[Predict or rank using input features]
    X --> P
    P --> E[Evaluate selected proposals]
    T --> Y[Observed outcomes, accessed at evaluation]
    Y --> E
    D --> S[Completed-record analyses: e.g. Spot leave-one-out]
```

## 3. Implementation conventions and method choices

The modules use standard Python and import shared functions from the preceding skill files. Execute them from the repository root with the commands in Section 6. No external machine-learning package is required by the ten demonstrated module implementations.

### Shared computations

**Categorical distance.** For two projects `a` and `b`, the feature distance is the proportion of selected input columns where their recorded categorical values differ:

\[
 d_X(a,b)=\frac{1}{|X|}\sum_{j\in X}\mathbf{1}[a_j\ne b_j].
\]

**Nearest-neighbor prediction.** Use the five closest historical projects, sorted deterministically by distance and then row number. The predicted outcome is their arithmetic mean. The minimum and maximum neighbor outcomes are descriptive bounds, **not calibrated confidence or prediction intervals**.

**Distance to heaven (d2h).** Normalize the three goals with historical minimum and maximum values, then compute root mean squared distance from their ideal normalized values (1 for `Kloc+`, 0 for `Effort-` and `Defects-`). Lower scores are preferred. The relevant normalization population matters: historical training bounds are used for held-out selection evaluation, while full-dataset descriptive rankings may use all 93 records.

### Skill-specific decisions

| Skill | Main implementation choice | Important scope note |
|---|---|---|
| Remember | CSV parsing, summary statistics and d2h ranking | Ranking reflects the three specified goal directions |
| Guess | k=5, normalized categorical mismatch, neighbor-mean prediction | All X attributes receive equal weight |
| Certify | Two-sided 9,999-permutation test of difference in means | This is not the course's separate `same` method; non-significance does not prove equivalence |
| Flow | Predict each arrival, compare with history's 75th-percentile alert thresholds, then append its observed record | The sequential simulation is not an external streaming system |
| Blame | Compare bottom/top 23 records per goal using categorical total-variation distance | A high separation score is an association, not a cause |
| Justify | Show five supporting neighbors and compare two proposals' predicted risks and differing X fields | Explanations describe model evidence, not causal contributions |
| Choose | Rank held-out proposals once using X-only predictions of all three goals; inspect budgets 3, 5, 10 | This is fixed-order selection, not adaptive search |
| Fix | Explore observed values of `docu`, `tool`, `site`, `sced`, changing one or two settings | Require at least 5% improvement; for defect repair cap predicted effort increase at 5%; hypothetical only |
| Spot | Average distance to five other records; leave-one-out absolute residuals | Projects under inspection are excluded from their own neighbor sets |
| Race | Compare kNN MAE to historical mean; compare fixed ranking against 30 seeded random orders | The random-order comparison is descriptive rather than a significance test |

The Fix candidate generator only tries values observed for individual attributes in training. It does not guarantee every proposed combination is common or feasible in practice.

## 4. Git checkpoint history

Each checkpoint corresponds to a sequential skill implementation. The final tag is the reference point for the complete ten-module code state.

| Step | Git tag | Commit | Skill |
|---:|---|---|---|
| 1 | `t7-nasa-step1` | `2fd2e53` | Remember |
| 2 | `t7-nasa-step2` | `4ab4582` | Guess |
| 3 | `t7-nasa-step3` | `e37e5b1` | Certify |
| 4 | `t7-nasa-step4` | `973b9f2` | Flow |
| 5 | `t7-nasa-step5` | `43ba989` | Blame |
| 6 | `t7-nasa-step6` | `d5ecf52` | Justify |
| 7 | `t7-nasa-step7` | `a6a23af` | Choose |
| 8 | `t7-nasa-step8` | `3d7ba92` | Fix |
| 9 | `t7-nasa-step9` | `fb56aee` | Spot |
| 10 | `t7-nasa-step10` | `ddb3e0b` | Race |

These hashes and tags were checked against the terminal Git history shared during the implementation. A documentation commit after Step 10 will not alter the tagged source states.

Verify locally:

```bash
git switch human-t7-nasa
git tag --list 't7-nasa-step*' | sort -V
git log --oneline --decorate -12
git status --short
```

The module import structure illustrates reuse across skills:

```mermaid
flowchart LR
    R[Remember] --> G[Guess]
    R --> C[Certify]
    G --> F[Flow]
    G --> J[Justify]
    R --> CH[Choose]
    G --> CH
    G --> FI[Fix]
    G --> SP[Spot]
    G --> RA[Race]
    CH --> RA
```

The diagram emphasizes selected reusable dependencies, not an exhaustive import graph.

## 5. Lines-of-code measurement

### Source of truth: treatment harness

LOC follows the algorithm in the `treatment-harness` branch's `harness/loc.py` (`code_lines`, `is_test_path`, and `measure_step`). The analysis compares **complete code snapshots at consecutive Git checkpoints**, not just the file introduced for that skill.

- **Scope:** Python (`.py`) and Lua (`.lua`) files under `human_t7/nasa/`; other file extensions, including Markdown and CSV, are excluded.
- **Filtering:** For Python, discard blank lines and lines whose stripped text begins with `#`; for Lua, discard blank lines and lines whose stripped text begins with `--`. Keep other stripped lines, including docstrings and lines containing inline comments.
- **Change detection:** Apply `difflib.SequenceMatcher(None, previous_lines, current_lines, autojunk=False)` to each file separately. `insert` and `replace` contribute current lines to **new LOC**; `delete` and `replace` contribute prior lines to **deleted LOC**.
- **Tests:** Files in a path component beginning with `test`, or with a stem ending `_test`, contribute their added lines to **test LOC**, separately from implementation LOC.
- **Total LOC:** Count the filtered lines in all current non-test implementation files.
- **% new:** `round(100 × new_loc / total_loc, 2)`; use 0% if total LOC is zero.
- **Initial comparison:** For Step 1, compare `t7-nasa-step1^` with `t7-nasa-step1`; for later steps, compare consecutive `t7-nasa-stepN` tags. This baseline definition follows the recorded NASA audit.

The primary cross-treatment comparison described by the harness uses **new LOC from Steps 2–10**, excluding the initial skill. The full Steps 1–10 sum is also reported for completeness. The two totals must not be conflated.

### Reproduce the harness-compatible audit

From the repository root, run the checked NASA audit script (if present in the local checkout):

```bash
python3 human_t7/nasa/audit_nasa_harness_loc.py
```

The script reads Git snapshots and performs the same filtering and `SequenceMatcher` comparison as `harness/loc.py`; it does not alter tags, commits or source code. If using another checkout, make sure this helper has been committed or supplied separately. The authoritative algorithm remains `harness/loc.py` on the `treatment-harness` branch.

To inspect the actual source differences independently:

```bash
git diff --numstat t7-nasa-step7 t7-nasa-step8 -- 'human_t7/nasa/*.py'
```

This Git command provides **physical-line** changes, so its figures need not match the harness's filtered LOC calculation.

### Recorded harness-compatible LOC results

| Step | Skill | New LOC | Deleted LOC | Test LOC | Total LOC | % new |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Remember | 127 | 0 | 0 | 127 | 100.00% |
| 2 | Guess | 103 | 0 | 0 | 230 | 44.78% |
| 3 | Certify | 96 | 0 | 0 | 326 | 29.45% |
| 4 | Flow | 112 | 0 | 0 | 438 | 25.57% |
| 5 | Blame | 107 | 0 | 0 | 545 | 19.63% |
| 6 | Justify | 111 | 0 | 0 | 656 | 16.92% |
| 7 | Choose | 145 | 0 | 0 | 801 | 18.10% |
| 8 | Fix | 161 | 0 | 0 | 962 | 16.74% |
| 9 | Spot | 114 | 0 | 0 | 1,076 | 10.59% |
| 10 | Race | 159 | 0 | 0 | 1,235 | 12.87% |

**Full implementation (Steps 1–10): 1,235 new LOC.**  **Primary harness comparison (Steps 2–10): 1,108 new LOC.**  No deleted LOC or separately classified test LOC were detected in the ten-step audit. The mean is **123.5 new LOC per skill** across all ten steps.

### Reconciliation with the earlier count

An earlier file-at-a-time script recorded **1,232 LOC** using Python's `tokenize` comment filtering. The harness audit measured **1,235 LOC**: the entire **three-line difference belongs to Step 1** (`remember.py`, 124 originally versus 127 in the harness). All **Steps 2–10 matched exactly** because each checkpoint added only its own skill module and did not edit prior modules. The harness uses stripped-prefix comment filtering rather than tokenized comment detection. Previous code snapshots and Git tags were preserved without modification.

The graphs and interpretation are maintained in [`RESULTS.md`](RESULTS.md#11-the-shrinking-code-question-did-additions-become-smaller).

## 6. Execution and reproduction

### Prerequisites

- A local clone of this repository with the NASA CSV under `data/raw/`.
- Python 3 with the standard library modules used by the source (`csv`, `math`, `statistics`, `pathlib`, `itertools`, `random`, etc.).
- Git with access to tags `t7-nasa-step1` through `t7-nasa-step10`.

The exact Python interpreter version, OS build, and machine specifications were **not recorded** in the retained experiment outputs and should not be fabricated as if they had been captured. They can be added later by running `python3 --version` and recording the environment.

### Run every module

From the repository root:

```bash
for skill in remember guess certify flow blame justify choose fix spot race; do
    echo "===== $skill ====="
    python3 "human_t7/nasa/$skill.py"
done
```

### Repeat the recorded entry-point smoke test

```bash
failed=0
for skill in remember guess certify flow blame justify choose fix spot race; do
    printf '%-10s ' "$skill"
    if python3 "human_t7/nasa/$skill.py" > "/tmp/nasa_${skill}.txt" 2>&1; then
        echo PASS
    else
        echo FAIL
        cat "/tmp/nasa_${skill}.txt"
        failed=1
    fi
done
exit "$failed"
```

The recorded terminal run produced **PASS for all ten programs**. This checks execution without an unhandled error; it does not independently validate each customer requirement.

### Verification performed during development

The following checks were reported as passing during individual steps. These were run as development assertions, not established as a persistent, independently maintained regression suite.

| Area | Recorded verification |
|---|---|
| Remember | Dataset loading, summary and ranking checks during initial implementation |
| Guess | Held-out predictions, neighbor selection and evaluation checks |
| Certify | Group comparison and permutation-test checks |
| Flow | Sequential arrival counts, updates and no held-out-outcome use before assessment |
| Blame | Outcome-group splitting, frequency comparisons and association rankings |
| Justify | Neighbor explanations and proposal-risk comparisons |
| Choose | Historical ranking, inspection budgets and hidden-goal masking |
| Fix | Adjustable-field restrictions, repair thresholds and hidden-goal leakage |
| Spot | Leave-one-out exclusion, input anomaly ordering and residual arithmetic |
| Race | Prediction baseline, seeded random reproducibility, budget ordering and hidden-goal leakage |

Detailed console outputs and interpreted findings are in `RESULTS.md` and the development run records. **Ten passing smoke tests plus targeted assertions do not establish formal acceptance of all 20 frozen requirements.**

## 7. Experimental boundaries and limitations

1. **Small, ordered sample:** 93 records, including a fixed last-19 holdout; different splits may produce different accuracy and rankings.
2. **Simple distance model:** All 22 categorical inputs contribute equally; the approach does not learn feature weights.
3. **Descriptive uncertainty:** Neighbor ranges are not calibrated prediction intervals; Certify's selected-group p-values are not evidence of equivalence.
4. **Fixed selection order:** Choose does not retrain after each inspected candidate. Race evaluates the same fixed ranking at larger budgets.
5. **Repair feasibility:** The Fix search explores limited, hypothetical setting alternatives; predicted gains do not establish real-world improvements.
6. **Anomalies versus errors:** Large residuals can arise from genuine variation, insufficient predictors or inaccurate measurement. No record is automatically invalidated.
7. **Baseline breadth:** Race uses the historical mean and 30 seeded random orderings, not a broad algorithm benchmark or formal statistical comparison.
8. **Code-growth measurement:** Harness new LOC depends on requirement complexity, refactoring activity and the defined source filtering and diff rules. The single sequence does not isolate a causal effect of reuse.
9. **Verification depth:** Execution and targeted checks support implementation confidence, but a full requirement-by-requirement acceptance suite and independent replication remain separate work.

## 8. Final handoff checklist

- [x] 20 frozen requirements mapped to ten implementation files.
- [x] Ten sequential Git tags created through `t7-nasa-step10`.
- [x] Ten module entry points executed with successful exit codes.
- [x] Targeted correctness checks performed during implementation.
- [x] Harness-compatible new LOC measured at every skill tag (1,235 for Steps 1–10; 1,108 for Steps 2–10).
- [x] Numerical findings and diagrams organized in `RESULTS.md`.
- [ ] Formal independent acceptance tests for all 20 customer requirements, if required by evaluation criteria.
- [ ] Capture exact interpreter/environment versions for archival reproducibility, if needed.
- [ ] Apply the same harness counting and checkpoint rules to other experimental branches before cross-treatment comparisons.

For the experimental observations, charts and narrative conclusions, see **[`RESULTS.md`](RESULTS.md)**. For the frozen customer acceptance objectives, see **[`docs/requirements_nasa.md`](../../docs/requirements_nasa.md)**.
