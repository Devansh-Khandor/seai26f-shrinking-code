# Software Project Estimation — Customer Requirements

**Course:** CSC 591 — AI & Software Engineering  
**Project:** The Shrinking-Code Experiment  
**Dataset:** `data/raw/nasa93dem.csv`  
**Client:** Software Engineering Project Manager  
**Status:** Draft — Pending Lecturer Approval

## Client Background

We manage software development projects and want to make better decisions about development effort, software quality, and project planning.

We have historical data from 93 NASA software projects, including development process characteristics, personnel capability ratings, technical requirements, project size, effort, and defects.

We want an AI-assisted decision-support system that helps us estimate outcomes, identify risky project proposals, explain historical patterns, and explore potential improvements before making planning decisions.

## Customer Requirements

### 1. Remember — Representation

**NASA-01:** Show me a summary of our historical software projects, including typical project size, development effort, and defect counts, along with the best and worst observed outcomes.

**NASA-02:** Identify projects that achieved unusually favorable combinations of software size, low development effort, and fewer defects, so we can study what made them stand out.

### 2. Guess — Prediction

**NASA-03:** Before approving a new software project, estimate the development effort it is likely to require based on its known characteristics and our historical records.

**NASA-04:** Estimate the likely number of software defects for a proposed project before development begins, and indicate how uncertain that estimate is.

### 3. Certify — Certification

**NASA-05:** Compare projects with different programmer capability ratings and tell me whether their development-effort differences are meaningful enough to inform future planning.

**NASA-06:** Compare projects with different complexity ratings and determine whether their defect outcomes differ reliably or whether the available evidence is insufficient.

### 4. Flow — Streaming

**NASA-07:** As new project proposals arrive, assess each proposal using its available characteristics and alert me when its predicted effort or defects appear unusually high.

**NASA-08:** As additional completed-project records become available, update our historical summaries and project estimates so that future assessments reflect the latest information.

### 5. Blame — Diagnosis

**NASA-09:** For projects that required unusually high development effort, identify which recorded project characteristics are most strongly associated with those outcomes.

**NASA-10:** Examine projects with unusually high defect counts and identify the characteristics that distinguish them from projects with better quality outcomes.

### 6. Justify — Explanation

**NASA-11:** When the system estimates the effort required for a proposed project, explain the estimate using relevant historical projects and their recorded characteristics.

**NASA-12:** When two project proposals receive different risk assessments, explain the main differences that contributed to those assessments and identify the important tradeoffs.

### 7. Choose — Optimization

**NASA-13:** Help me identify promising combinations of project characteristics that historically delivered more software with lower development effort and fewer defects.

**NASA-14:** When detailed project evaluations are expensive, recommend which project proposals should be evaluated first to discover promising alternatives with a limited evaluation budget.

### 8. Fix — Repair

**NASA-15:** For a project proposal predicted to require excessive effort, identify plausible changes to adjustable planning or process characteristics that might reduce its predicted effort.

**NASA-16:** For a project proposal with a high predicted defect count, identify the smallest plausible combination of adjustable planning changes that could improve its predicted quality without creating unacceptable tradeoffs.

### 9. Spot — Anomaly Detection

**NASA-17:** Flag historical projects whose combinations of project characteristics are unusually different from the rest of our records so that we can review them.

**NASA-18:** Identify project records whose reported effort or defect counts differ substantially from what comparable projects would lead us to expect, so we can investigate possible measurement or recording issues.

### 10. Race — Baseline Comparison

**NASA-19:** Compare our project-effort estimation system against a simple historical-average estimate and show whether it makes meaningfully better predictions on previously unseen projects.

**NASA-20:** Compare our project-selection approach with randomly selecting proposals under the same evaluation budget, and show whether our system identifies more desirable project outcomes.

## Notes

- All requirements must use information available in the historical dataset.
- The dataset's goal directions are `Kloc+`, `Effort-`, and `Defects-`.
- Any new-project proposals or incoming project records may be simulated from held-out historical observations.
- Estimates must be evaluated against data not used to train the corresponding prediction model.
- Explanations and diagnoses must distinguish associations from causal conclusions.
- Suggested project changes are hypothetical and must be restricted to plausible, explicitly designated adjustable characteristics.
- The relatively small dataset may limit predictive accuracy and statistical certainty.
- The requirements are intended to support incremental demonstrations and cross-domain reuse without assuming that reuse will necessarily reduce implementation effort.

**Approval:** Pending lecturer review.