# Apache Server Configuration Optimization

**The Shrinking-Code Experiment — CSC 591**

This directory contains the Apache T7 cumulative Python implementation.

Dataset: `data/raw/Apache_AllMeasurements.csv`

Frozen requirements: `docs/requirements_apache.md`

## Dataset

- 192 historical server configurations
- 8 adjustable binary settings
- 1 constant setting (`base`)
- 1 optimization goal: `Performance-` (minimize)

## Implementation

| Step | Skill | Requirements | Module | Status |
|---|---|---|---|---|
| 1 | Remember | APACHE-01/02 | remember.py | Implemented |

Later skills will reuse earlier functionality.

## Step 1: Remember

APACHE-01 summarizes historical performance and identifies
the best and worst configurations.

APACHE-02 identifies favorable configurations using the
lowest 25% of recorded performance values, including ties,
and compares setting frequencies against the remaining records.

This threshold is an implementation choice, not a statistical
significance test.

## Reproduce

From the repository root:

    python3 human_t7/apache/remember.py

Run the tests:

    python3 -m unittest discover -s human_t7/apache/tests -v

## Evaluation

The implementation uses Python's standard library.

Source snapshots will be preserved using Git tags
`t7-apache-step1` through `t7-apache-step10`.

One cumulative T7 implementation run is required.

The tests provide behavioral checks but are not proof
of universal semantic correctness.
