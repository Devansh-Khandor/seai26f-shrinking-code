#!/usr/bin/env python3
"""
Skill 08: Reduce effort / improve quality for high‑risk proposals

NASA-15: For a proposal predicted to require excessive effort, identify plausible
adjustments to categorical characteristics that could lower the predicted effort.
NASA-16: For a proposal with a high predicted defect count, suggest the smallest
combination of changes to categorical characteristics that could improve predicted
quality, explaining potential trade‑offs.

The script reuses functions from earlier skills: data loading, statistics,
and linear regression estimation.  Adjustments are based on mean effort or
defect outcomes for each category of a characteristic.
"""

import csv
import sys
import numpy as np
from typing import List, Dict, Any, Tuple

# Reuse helpers from prior skills
try:
    import skill_01 as s01
    import skill_02 as s02
except ImportError:
    # Minimal stubs if modules are not importable
    import csv
    def load_data(csv_path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    r['Kloc+'] = float(r['Kloc+'])
                    r['Effort-'] = float(r['Effort-'])
                    r['Defects-'] = float(r['Defects-'])
                except (ValueError, KeyError):
                    continue
                rows.append(r)
        return rows
    s01 = type('s01', (), {'load_data': staticmethod(load_data),
                           'compute_stats': staticmethod(
                               lambda rows: {'Kloc+': {'mean': 0, 'std': 0},
                                             'Effort-': {'mean': 0, 'std': 0},
                                             'Defects-': {'mean': 0, 'std': 0}}),
                           'best_and_worst': staticmethod(
                               lambda rows: {'best': {}, 'worst': {}})})

    def estimate_effort(rows):
        return 0, 0, 0, 0, 0
    def estimate_defects(rows):
        return 0, 0, 0, 0, 0
    s02 = type('s02', (), {'estimate_effort': staticmethod(estimate_effort),
                           'estimate_defects': staticmethod(estimate_defects)})

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def parse_rows(csv_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split the CSV into historical (complete numeric fields) and proposals
    (missing Effort- or Defects-).
    """
    historical: List[Dict[str, Any]] = []
    proposals: List[Dict[str, Any]] = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r_copy = r.copy()
            is_historical = True
            for key in ('Kloc+', 'Effort-', 'Defects-'):
                try:
                    r_copy[key] = float(r_copy[key])
                except (ValueError, KeyError):
                    is_historical = False
            if is_historical:
                historical.append(r_copy)
            else:
                proposals.append(r_copy)
    return historical, proposals


def train_and_predict(rows: List[Dict[str, Any]],
                      target: str,
                      value: float) -> float:
    """
    Train a simple linear regression on historical Kloc+ vs target metric
    and predict the value for the given Kloc+.
    """
    from sklearn.linear_model import LinearRegression
    X = np.array([r['Kloc+'] for r in rows]).reshape(-1, 1)
    y = np.array([r[target] for r in rows])
    model = LinearRegression().fit(X, y)
    return model.predict([[value]])[0]


def is_numeric_column(rows: List[Dict[str, Any]], col: str) -> bool:
    """Heuristically detect if a column is numeric."""
    for r in rows[:10]:  # sample a few rows
        try:
            float(r[col])
        except (ValueError, TypeError, KeyError):
            return False
    return True


def get_category_stats(rows: List[Dict[str, Any]],
                       feature: str,
                       target: str) -> Dict[str, float]:
    """
    Return a dict mapping category value -> mean of target metric.
    """
    stats: Dict[str, List[float]] = {}
    for r in rows:
        val = r[feature]
        stats.setdefault(val, []).append(r[target])
    return {k: np.mean(vs) for k, vs in stats.items()}


# ----------------------------------------------------------------------
# Suggestion logic
# ----------------------------------------------------------------------
def suggest_effort_reduction(proposal: Dict[str, Any],
                             historical: List[Dict[str, Any]],
                             cat_stats: Dict[str, Dict[str, float]],
                             overall_mean: float,
                             num_suggestions: int = 3) -> List[str]:
    """
    Return up to `num_suggestions` lines suggesting categorical changes
    that could reduce effort for the proposal.
    """
    suggestions = []
    for feature, stats in cat_stats.items():
        current = proposal.get(feature)
        if current is None or current not in stats:
            continue
        current_mean = stats[current]
        if current_mean <= overall_mean:
            continue  # no benefit in this category
        # Find best (lowest) category
        best_cat, best_mean = min(stats.items(), key=lambda kv: kv[1])
        if best_mean >= current_mean:
            continue
        diff = current_mean - best_mean
        suggestions.append(
            f"• Change {feature} from '{current}' to '{best_cat}' "
            f"could reduce effort by {diff:.2f} PM."
        )
    # If fewer than requested, pad with generic note
    if len(suggestions) < num_suggestions:
        suggestions.append("• No further categorical changes identified.")
    return suggestions[:num_suggestions]


def suggest_defect_improvement(proposal: Dict[str, Any],
                               historical: List[Dict[str, Any]],
                               cat_stats: Dict[str, Dict[str, float]],
                               overall_mean: float,
                               num_suggestions: int = 3) -> List[str]:
    """
    Return up to `num_suggestions` lines suggesting categorical changes
    that could improve predicted defect count for the proposal.
    """
    suggestions = []
    for feature, stats in cat_stats.items():
        current = proposal.get(feature)
        if current is None or current not in stats:
            continue
        current_mean = stats[current]
        if current_mean <= overall_mean:
            continue  # this category already better
        # Find best (lowest) category
        best_cat, best_mean = min(stats.items(), key=lambda kv: kv[1])
        if best_mean >= current_mean:
            continue
        diff = current_mean - best_mean
        suggestions.append(
            f"• Change {feature} from '{current}' to '{best_cat}' "
            f"could reduce defects by {diff:.2f} cases."
        )
    if len(suggestions) < num_suggestions:
        suggestions.append("• No further categorical changes identified.")
    return suggestions[:num_suggestions]


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    historical, proposals = parse_rows(csv_path)

    if not historical:
        print("No historical project data available.")
        sys.exit(1)

    # Overall statistics
    hist_stats = s01.compute_stats(historical)
    eff_mean = hist_stats['Effort-']['mean']
    eff_std = hist_stats['Effort-']['std']
    def_mean = hist_stats['Defects-']['mean']
    def_std = hist_stats['Defects-']['std']

    effort_thresh = eff_mean + 2 * eff_std
    defect_thresh = def_mean + 2 * def_std

    # Identify adjustable categorical columns
    sample = historical[0]
    exclude = {'idX', 'Kloc+', 'Effort-', 'Defects-', 'MonthsX'}
    cat_features = [c for c in sample.keys()
                    if c not in exclude and not is_numeric_column(historical, c)]

    # Pre‑compute category statistics
    eff_cat_stats = {feat: get_category_stats(historical, feat, 'Effort-')
                     for feat in cat_features}
    def_cat_stats = {feat: get_category_stats(historical, feat, 'Defects-')
                     for feat in cat_features}

    # ------------------------------------------------------------------
    # NASA-15: effort reduction suggestions
    # ------------------------------------------------------------------
    print("== NASA-15 ==")
    effort_flagged = False
    for p in proposals:
        # Predict effort for proposal
        try:
            kloc = float(p['Kloc+'])
        except (KeyError, ValueError, TypeError):
            continue
        predicted_eff = train_and_predict(historical, 'Effort-', kloc)
        if predicted_eff <= effort_thresh:
            continue

        effort_flagged = True
        print(f"Proposal ID {int(p['idX'])}: Predicted effort {predicted_eff:.2f} PM "
              f"(threshold {effort_thresh:.2f})")
        print("Suggested categorical changes to reduce effort:")
        suggestions = suggest_effort_reduction(p, historical, eff_cat_stats, eff_mean)
        for s in suggestions:
            print(f"  {s}")
        print()

    if not effort_flagged:
        print("No proposals exceed the excessive‑effort threshold.")
        print("Consider reviewing project parameters such as team skill or tooling.")
        print("Adjusting these categorical factors may help keep effort within range.")

    # ------------------------------------------------------------------
    # NASA-16: defect improvement suggestions
    # ------------------------------------------------------------------
    print("\n== NASA-16 ==")
    defect_flagged = False
    for p in proposals:
        # Predict defects for proposal
        try:
            kloc = float(p['Kloc+'])
        except (KeyError, ValueError, TypeError):
            continue
        predicted_def = train_and_predict(historical, 'Defects-', kloc)
        if predicted_def <= defect_thresh:
            continue

        defect_flagged = True
        print(f"Proposal ID {int(p['idX'])}: Predicted defects {predicted_def:.2f} "
              f"(threshold {defect_thresh:.2f})")
        print("Suggested categorical changes to improve quality:")
        suggestions = suggest_defect_improvement(p, historical, def_cat_stats, def_mean)
        for s in suggestions:
            print(f"  {s}")
        print("Trade‑off: reducing defects may require tighter process controls "
              "or additional review time, potentially affecting schedule.")
        print()

    if not defect_flagged:
        print("No proposals exceed the high‑defect threshold.")
        print("Project appears to be within acceptable quality ranges.")
        print("Maintain current process settings to preserve this outcome.")


if __name__ == "__main__":
    main()
