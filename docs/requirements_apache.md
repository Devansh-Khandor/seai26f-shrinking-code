# Apache Server Configuration Optimization — Customer Requirements

**Course:** CSC 591 — AI & Software Engineering  
**Project:** The Shrinking-Code Experiment  
**Dataset:** `data/raw/Apache_AllMeasurements.csv`  
**Client:** Apache Web Server Administrator  
**Status:** Version 1.0 — Frozen requirements baseline  
**Baseline date:** November 6, 2026 (before implementation)

## Client Background

We manage Apache web servers and want a decision-support system that helps us evaluate and improve server configurations.

We have historical performance measurements for 192 server configurations, described by settings related to connection handling, logging, file handling, and other Apache options.

Our goal is to identify configurations with lower recorded performance-metric values, understand which settings are associated with poor outcomes, and reduce the number of expensive configuration evaluations.

We want recommendations that are explainable, supported by historical evidence, and practical to evaluate.

## Customer Requirements

### 1. Remember — Representation

**APACHE-01:** Show me a summary of our previously tested server configurations, including the typical, best, and worst recorded performance values.

**APACHE-02:** Identify the configurations with unusually favorable performance values and show which settings distinguish them from the rest of our tested configurations.

### 2. Guess — Prediction

**APACHE-03:** Before running a benchmark on a configuration, estimate its likely performance value from its settings and our historical benchmark records.

**APACHE-04:** When comparing two historically observed valid configurations that differ by one setting, estimate their potential performance difference and indicate how uncertain that estimate is.

### 3. Certify — Certification

**APACHE-05:** Compare configurations with KeepAlive enabled and disabled, and tell me whether the observed differences in performance are meaningful enough to guide our decisions.

**APACHE-06:** Compare configurations with access logging enabled and disabled, and tell me whether one group consistently performs better or whether our available evidence is inconclusive.

### 4. Flow — Streaming

**APACHE-07:** As new benchmark results arrive, assess each configuration immediately and alert me when its measured performance is unusually poor compared with previously tested configurations.

**APACHE-08:** Keep our performance summaries and recommended-configuration shortlist updated as additional benchmark records become available, without requiring a complete manual restart of the analysis.

### 5. Blame — Diagnosis

**APACHE-09:** When a configuration performs poorly, identify which of its enabled or disabled settings are most strongly associated with unfavorable performance in our historical measurements.

**APACHE-10:** Identify combinations of server settings that frequently appear among poorly performing configurations and show how they differ from better-performing combinations.

### 6. Justify — Explanation

**APACHE-11:** When the system recommends a configuration, explain its recommendation by comparing the configuration's settings and performance with relevant previously tested configurations.

**APACHE-12:** When two configurations receive different performance assessments, explain which settings differ between them and how those differences relate to the predicted or recorded outcomes.

### 7. Choose — Optimization

**APACHE-13:** Help me discover a configuration with a low performance-metric value while evaluating as few candidate configurations as reasonably possible.

**APACHE-14:** When I have a limited benchmarking budget, recommend which configurations should be evaluated first so that we can discover promising server settings early.

### 8. Fix — Repair

**APACHE-15:** For a poorly performing configuration, recommend a small change to its adjustable settings that could lead to a lower predicted performance value, using historically observed valid configurations as alternatives.

**APACHE-16:** Find the smallest number of configuration-setting changes that could bring a poorly performing configuration closer to a specified performance target, while avoiding unsupported or invalid combinations.

### 9. Spot — Anomaly Detection

**APACHE-17:** Identify configurations whose recorded performance values are unusually different from what their settings and comparable historical configurations would lead us to expect.

**APACHE-18:** Flag configurations whose combinations of settings differ substantially from the most similar configurations in our historical benchmark records, even when their measured performance is not obviously poor.

### 10. Race — Baseline Comparison

**APACHE-19:** Compare our configuration-selection system against random configuration selection under the same benchmarking budget, and show which approach discovers better-performing configurations.

**APACHE-20:** Compare our performance predictions against a simple historical-average estimate and show whether our system provides a meaningful improvement on previously unseen configurations.

## Notes

- All customer requests are based on information available in `Apache_AllMeasurements.csv`.
- `Performance-` is the sole optimization goal and must be minimized.
- The physical meaning and units of `Performance-` are not established by the CSV.
- The `base` setting is constant and must not be treated as an adjustable option.
- Configuration changes must be restricted to historically observed valid combinations unless external documentation establishes additional valid configurations.
- Incoming benchmark records may be simulated using withheld historical observations.
- Comparisons between settings are observational and do not establish causal effects.
- The dataset contains no repeated measurements of identical configurations, limiting conclusions about measurement noise.
- Predictions must be evaluated using records excluded from their training data.
- Suggested improvements are predictions or what-if analyses, not guarantees of real-world performance.
- The requirements are intended to support incremental demonstrations and code reuse across domains, without assuming that reuse will produce a particular result.