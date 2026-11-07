# The Shrinking-Code Experiment — Requirements Baseline

**Course:** CSC 591 — AI & Software Engineering  
**Version:** 1.0  
**Status:** Frozen requirements baseline  
**Baseline date:** November 6, 2026 (before implementation)

The Shrinking-Code Experiment examines incremental implementation and code reuse across three decision-support domains. Each domain expresses customer needs through the same ten skills, providing a common requirements structure for comparing experimental treatments. Whether reuse reduces implementation effort or code size remains an experimental question; this baseline reports no results.

## Domains and customer requirements

- **Cars — Automobile selection:** Independent Used-Car Dealership; dataset: [`data/raw/auto93.csv`](../data/raw/auto93.csv). See [Cars customer requirements](requirements_cars.md).
- **NASA — Software project estimation:** Software Engineering Project Manager; dataset: [`data/raw/nasa93dem.csv`](../data/raw/nasa93dem.csv). See [NASA customer requirements](requirements_nasa.md).
- **Apache — Server configuration optimization:** Apache Web Server Administrator; dataset: [`data/raw/Apache_AllMeasurements.csv`](../data/raw/Apache_AllMeasurements.csv). See [Apache customer requirements](requirements_apache.md).

Each domain contains **20 customer requests**, with **two requests per skill**, for **60 requirements** overall. The linked domain documents contain the authoritative request text; this document serves as their index and overview.

## Prescribed skill order

1. Remember
2. Guess
3. Certify
4. Flow
5. Blame
6. Justify
7. Choose
8. Fix
9. Spot
10. Race

## Baseline and change control

Version 1.0 was frozen on **November 6, 2026, before implementation**. These four requirements documents establish the official, fixed requirements baseline for **all experimental treatments**.

Any subsequent substantive change must be documented and versioned rather than silently altering the Version 1.0 baseline. Record the affected requirement IDs, the change and its rationale, the date, and the new version, while preserving the Version 1.0 text for comparison. Changes must explicitly state their effect on comparability across experimental treatments.
