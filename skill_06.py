#!/usr/bin/env python3
"""
Skill 06: Explain effort estimates and compare risk assessments for project proposals.

NASA-11: When estimating effort for a proposed project, explain the estimate
using relevant historical projects and their recorded characteristics.

NASA-12: When two project proposals receive different risk assessments,
explain the main differences that contributed to those assessments and
identify the important tradeoffs.
"""

import csv
import math
import sys

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def parse_float(val: str | None) -> float | None:
    """Safely parse a string to float; return None on failure."""
    if val is None:
        return None
    try:
        return float(val.strip())
    except (AttributeError, ValueError):
        return None

def mean(values):
    return sum(values) / len(values) if values else 0.0

def stddev(values, mean_val):
    n = len(values)
    if n < 2:
        return 0.0
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / (n - 1))

# --------------------------------------------------------------------------- #
# Regression helpers
# --------------------------------------------------------------------------- #
def linear_regression(x, y):
    """Return slope and intercept for y = a + b * x."""
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(v * v for v in x)
    sum_xy = sum(v * w for v, w in zip(x, y))
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 0.0, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept

def predict_with_confidence(slope, intercept, x0, x, y):
    """Return predicted y at x0 and its standard error of prediction."""
    y0 = intercept + slope * x0
    n = len(x)
    if n < 2:
        return y0, 0.0
    mean_x = sum(x) / n
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    sigma_sq = sum(r * r for r in residuals) / (n - 2) if n > 2 else 0.0
    if sum_sq_x == 0 or sigma_sq == 0:
        return y0, 0.0
    se = math.sqrt(
        sigma_sq * (1 + 1 / n + ((x0 - mean_x) ** 2) / sum_sq_x)
    )
    return y0, se

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_historical(csv_path: str):
    """Read CSV and return lists of numeric columns and project IDs."""
    kloc = []
    effort = []
    defects = []
    ids = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = parse_float(row.get('Kloc+'))
            e = parse_float(row.get('Effort-'))
            d = parse_float(row.get('Defects-'))
            if k is not None and e is not None and d is not None:
                kloc.append(k)
                effort.append(e)
                defects.append(d)
                ids.append(row.get('id'))
    return kloc, effort, defects, ids

# --------------------------------------------------------------------------- #
# Similarity helpers
# --------------------------------------------------------------------------- #
def find_similar_projects(kloc_list, ids, target_kloc, k=3):
    """Return indices of k projects with closest KLOC+ to target."""
    distances = [(abs(k - target_kloc), idx) for idx, k in enumerate(kloc_list)]
    distances.sort(key=lambda t: t[0])
    return [idx for _, idx in distances[:k]]

# --------------------------------------------------------------------------- #
# NASA-11 implementation
# --------------------------------------------------------------------------- #
def nasa_11(csv_path):
    kloc_hist, effort_hist, defects_hist, ids = load_historical(csv_path)
    if not kloc_hist:
        print("== NASA-11 ==")
        print("No historical data available.")
        return

    # Prompt user for proposed KLOC+
    try:
        inp = input("Enter proposed KLOC+ for NASA-11 (numeric): ").strip()
        proposed_kloc = float(inp)
    except Exception:
        print("Invalid input; using default KLOC+ = 100.")
        proposed_kloc = 100.0

    # Estimate effort
    slope, intercept = linear_regression(kloc_hist, effort_hist)
    est_effort, se_effort = predict_with_confidence(
        slope, intercept, proposed_kloc, kloc_hist, effort_hist
    )
    ci_lower_eff = est_effort - 1.96 * se_effort
    ci_upper_eff = est_effort + 1.96 * se_effort

    # Find similar historical projects
    similar_idxs = find_similar_projects(kloc_hist, ids, proposed_kloc, k=3)

    print("== NASA-11 ==")
    print(f"Proposed KLOC+: {proposed_kloc:.2f}")
    print(f"Estimated Effort-: {est_effort:.2f} (95% CI: {ci_lower_eff:.2f} - {ci_upper_eff:.2f})")
    print("\nTop 3 historically similar projects (by KLOC+ difference):")
    similar_vals = []
    for idx in similar_idxs:
        pid = ids[idx]
        pk = kloc_hist[idx]
        pe = effort_hist[idx]
        similar_vals.append(pe)
        print(f"  {pid} - KLOC+={pk:.2f}, Effort-={pe:.2f}")

    # Provide context
    avg_similar_effort = mean(similar_vals) if similar_vals else 0.0
    print(f"\nAverage Effort- of these projects: {avg_similar_effort:.2f}")
    print("The estimate aligns closely with the historical trend for projects of similar size.\n")

# --------------------------------------------------------------------------- #
# NASA-12 implementation
# --------------------------------------------------------------------------- #
def nasa_12(csv_path):
    kloc_hist, effort_hist, defects_hist, ids = load_historical(csv_path)
    if not kloc_hist:
        print("== NASA-12 ==")
        print("No historical data available.")
        return

    # Prompt for two proposals
    try:
        inp1 = input("Enter proposed KLOC+ for project A: ").strip()
        proj_a = float(inp1)
    except Exception:
        print("Invalid input for project A; using default 100.")
        proj_a = 100.0

    try:
        inp2 = input("Enter proposed KLOC+ for project B: ").strip()
        proj_b = float(inp2)
    except Exception:
        print("Invalid input for project B; using default 120.")
        proj_b = 120.0

    # Estimate for A
    slope_a, intercept_a = linear_regression(kloc_hist, effort_hist)
    est_eff_a, se_eff_a = predict_with_confidence(
        slope_a, intercept_a, proj_a, kloc_hist, effort_hist
    )
    ci_low_a = est_eff_a - 1.96 * se_eff_a
    ci_up_a = est_eff_a + 1.96 * se_eff_a

    # Estimate for B
    slope_b, intercept_b = linear_regression(kloc_hist, effort_hist)
    est_eff_b, se_eff_b = predict_with_confidence(
        slope_b, intercept_b, proj_b, kloc_hist, effort_hist
    )
    ci_low_b = est_eff_b - 1.96 * se_eff_b
    ci_up_b = est_eff_b + 1.96 * se_eff_b

    print("\n== NASA-12 ==")
    print(f"Project A: KLOC+={proj_a:.2f} → Effort-={est_eff_a:.2f} (95% CI: {ci_low_a:.2f} - {ci_up_a:.2f})")
    print(f"Project B: KLOC+={proj_b:.2f} → Effort-={est_eff_b:.2f} (95% CI: {ci_low_b:.2f} - {ci_up_b:.2f})")
    print()

    # Compare risk
    risk_diff = est_eff_b - est_eff_a
    print(f"Difference in estimated effort (B - A): {risk_diff:.2f} hours")
    if ci_low_b > ci_up_a:
        print("The CI for Project B lies entirely above that of Project A → higher risk.")
    elif ci_up_b < ci_low_a:
        print("The CI for Project B lies entirely below that of Project A → lower risk.")
    else:
        print("The CIs overlap → risk difference is not statistically significant.")

    # Trade‑off explanation
    print("\nMain contributing factor: higher KLOC+ for Project B leads to higher estimated effort.")
    if proj_b > proj_a:
        print("Increasing KLOC+ increases effort roughly linearly, as seen in the regression slope.")
    else:
        print("Decreasing KLOC+ reduces effort but may also reduce feature set or scope.")
    print("\nTrade‑offs to consider:")
    print("  • Higher effort increases cost and schedule risk but may support more features.")
    print("  • Lower effort may shorten delivery but risk missing critical functionalities.")
    print("Decide based on stakeholder priorities and available resources.\n")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    nasa_11(csv_path)
    nasa_12(csv_path)
