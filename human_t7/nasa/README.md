# NASA Software Project Estimation

**The Shrinking-Code Experiment · CSC 591 — AI & Software Engineering**

This directory contains a ten-step Python prototype for analyzing historical NASA software projects. Each step adds a capability while reusing functions from earlier modules. The implementation addresses **20 frozen requirements** (two per step), from describing historical records to comparing project-selection strategies against baselines.

## Start here

| Document | Purpose |
|---|---|
| [RESULTS.md](RESULTS.md) | Results for all ten skills, charts, observations, comparisons, and the code-growth analysis |
| [EXPERIMENT.md](EXPERIMENT.md) | Dataset and evaluation setup, implementation methods, Git checkpoints, LOC measurement, testing, and reproduction instructions |
| [Frozen NASA requirements](../../docs/requirements_nasa.md) | The 20 original customer requirements (NASA-01–NASA-20) |

## Dataset and approach

- **Dataset:** [`data/raw/nasa93dem.csv`](../../data/raw/nasa93dem.csv) — 93 historical projects.
- **Inputs:** 22 categorical project attributes.
- **Goals:** `Kloc+` (maximize), `Effort-` (minimize), and `Defects-` (minimize), following the dataset's header conventions.
- **Evaluation split:** First 74 records as initial history; last 19 as held-out proposals.
- **Prediction:** Five-nearest-neighbor categorical mismatch distance, with historical neighbor outcomes averaged for the estimate.

The prototype uses Python's standard library; the scripts do not require an external machine-learning package.

## Skills and source files

| Step | Skill | Requirements | Module | Purpose |
|---:|---|---|---|---|
| 1 | Remember | NASA-01/02 | [`remember.py`](remember.py) | Load, summarize, and rank historical projects |
| 2 | Guess | NASA-03/04 | [`guess.py`](guess.py) | Predict effort and defects from comparable projects |
| 3 | Certify | NASA-05/06 | [`certify.py`](certify.py) | Compare selected project groups statistically |
| 4 | Flow | NASA-07/08 | [`flow.py`](flow.py) | Assess sequential arrivals and update history |
| 5 | Blame | NASA-09/10 | [`blame.py`](blame.py) | Identify attributes associated with outcome extremes |
| 6 | Justify | NASA-11/12 | [`justify.py`](justify.py) | Explain estimates and compare proposals |
| 7 | Choose | NASA-13/14 | [`choose.py`](choose.py) | Rank candidates for limited-budget inspection |
| 8 | Fix | NASA-15/16 | [`fix.py`](fix.py) | Evaluate hypothetical adjustments to planning settings |
| 9 | Spot | NASA-17/18 | [`spot.py`](spot.py) | Flag unusual inputs and prediction residuals |
| 10 | Race | NASA-19/20 | [`race.py`](race.py) | Compare prediction and selection against baselines |

## Quick start

From the **repository root**, with Python 3 available:

```bash
python3 human_t7/nasa/remember.py
python3 human_t7/nasa/guess.py
python3 human_t7/nasa/race.py
```

Run all ten demonstrations:

```bash
for skill in remember guess certify flow blame justify choose fix spot race; do
  echo "===== $skill ====="
  python3 "human_t7/nasa/$skill.py" || break
done
```

Each script has its own entry point and prints its results. Execute the commands from the repository root so the dataset and module paths resolve as documented.

## Recorded outcomes

- **10/10** skill entry points passed the recorded execution smoke test; additional targeted assertions were run during development.
- **1,235 harness-compatible new implementation LOC** across Steps 1–10; the primary cross-treatment figure for **Steps 2–10 is 1,108 LOC**. The audit found zero deleted LOC and zero separately counted test LOC.
- On the 19 held-out records, the **historical-mean baseline had lower mean absolute error** than kNN for both effort and defects.
- Model-guided selection beat the **average** of 30 seeded random trials at inspection budgets **5 and 10**, but not at budget **3**.
- Absolute new LOC **did not consistently decrease**; the percentage of new code generally declined as the total implementation grew.

These are exploratory results, not formal acceptance or statistical-significance claims. See [RESULTS.md](RESULTS.md) for the full tables, diagrams, numerical evidence, and limitations.

## Reproduction and checkpoints

The implementation is preserved at Git tags `t7-nasa-step1` through `t7-nasa-step10`; the final implementation checkpoint is **`t7-nasa-step10`** (`ddb3e0b`). The development branch is **`human-t7-nasa`**.

```bash
git tag --list 't7-nasa-step*' | sort -V
git show --stat t7-nasa-step10
```

For the harness LOC calculation, test procedure, feature definitions, and experimental limitations, see [EXPERIMENT.md](EXPERIMENT.md).
