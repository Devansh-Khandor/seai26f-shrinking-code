# Automobile Selection — Customer Requirements

**Course:** CSC 591 — AI & Software Engineering  
**Project:** The Shrinking-Code Experiment  
**Dataset:** `data/raw/auto93.csv`  
**Client:** Independent Used-Car Dealership  
**Status:** Draft — Pending Lecturer Approval

## Client Background

We operate an independent used-car dealership and want a decision-support system that helps us understand vehicle performance, evaluate incoming inventory, and recommend desirable cars.

Our available historical dataset contains 398 vehicle records with information about engine cylinders, engine volume, model-year codes, origin categories, vehicle weight, acceleration measurements, and fuel economy.

We want the system to help our staff make informed decisions while explaining uncertainty and avoiding unnecessary vehicle evaluations.

## Customer Requirements

### 1. Remember — Representation

**CAR-01:** Show me a summary of our inventory, including typical vehicle weight, fuel efficiency, and acceleration measurements, along with the best and worst observed values.

**CAR-02:** Identify the vehicles that offer the most attractive overall combination of low weight, high fuel efficiency, and favorable acceleration measurements, rather than judging them on a single characteristic.

### 2. Guess — Prediction

**CAR-03:** When a vehicle's fuel-efficiency measurement is unavailable, estimate it using the vehicle characteristics we already know and our historical inventory records.

**CAR-04:** Before physically weighing a newly acquired vehicle, estimate its weight from the available vehicle information and show how uncertain that estimate might be.

### 3. Certify — Certification

**CAR-05:** Compare the fuel efficiency of vehicles with different cylinder counts and tell me whether the observed differences are reliable enough to inform inventory decisions.

**CAR-06:** When comparing two groups of vehicles from different model-year ranges, tell me whether their overall performance differs meaningfully or whether the evidence is too weak to make that conclusion.

### 4. Flow — Streaming

**CAR-07:** As new vehicle records arrive, assess each one immediately and tell me whether it belongs among our more desirable inventory options.

**CAR-08:** Keep our inventory summaries and recommended-vehicle shortlist up to date as additional vehicle records are received, without requiring staff to restart the entire analysis manually.

### 5. Blame — Diagnosis

**CAR-09:** When a vehicle has unusually poor fuel efficiency, identify which of its recorded characteristics are most strongly associated with that outcome.

**CAR-10:** Find the common characteristics of vehicles with unfavorable overall performance and help me understand how they differ from our better-performing vehicles.

### 6. Justify — Explanation

**CAR-11:** When the system recommends a vehicle, explain its recommendation using understandable comparisons with similar vehicles in our inventory.

**CAR-12:** When two vehicles have different strengths, explain why one receives a higher overall recommendation and show the tradeoffs involved.

### 7. Choose — Optimization

**CAR-13:** Help me identify a vehicle with a strong balance of fuel efficiency, weight, and acceleration while evaluating as few candidate vehicles as reasonably possible.

**CAR-14:** When we have a limited inspection budget, recommend which vehicles should be evaluated first so that we can discover promising inventory options early.

### 8. Fix — Repair

**CAR-15:** For a poorly rated vehicle, show which feasible changes to its recorded characteristics could hypothetically lead to a better predicted performance assessment.

**CAR-16:** Find the smallest plausible change to a vehicle's adjustable characteristics that would move its predicted assessment closer to our preferred performance targets.

### 9. Spot — Anomaly Detection

**CAR-17:** Flag vehicle records whose combinations of characteristics are unusually different from the rest of our inventory so that staff can review them.

**CAR-18:** Identify vehicles whose reported fuel efficiency or weight differs substantially from what similar vehicles would lead us to expect, so we can check for possible recording mistakes.

### 10. Race — Baseline Comparison

**CAR-19:** Show whether our vehicle-selection system finds better candidates than simply choosing vehicles at random when both approaches have the same evaluation budget.

**CAR-20:** Compare our vehicle-performance estimates against a straightforward historical-average estimate and show whether the new system provides a meaningful improvement on previously unseen records.

## Notes

- The system uses only information supported by the available historical dataset.
- Predictions, diagnoses, and hypothetical changes must communicate uncertainty where relevant.
- Unusual records are flagged for review, not automatically labeled as errors or fraud.
- The requirements are intended to support incremental demonstrations with opportunities to reuse existing functionality.
- The effectiveness of that reuse will be evaluated experimentally rather than assumed.
