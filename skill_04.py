#!/usr/bin/env python3
"""
Skill 04: Continuous assessment and update of project estimates.

NASA-07: As new project proposals arrive, assess each proposal using its
available characteristics and alert when its predicted effort or defects
appear unusually high.

NASA-08: As additional completed-project records become available,
update our historical summaries and project estimates so that future
assessments reflect the latest information.
"""

import csv
import math
import sys

# --------------------------------------------------------------------------- #
# Utility helpers
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
# Core analysis
# --------------------------------------------------------------------------- #
def load_historical(csv_path):
    """Load historical project records from CSV."""
    kloc = []
    effort = []
    defects = []
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
    return kloc, effort, defects


def print_nasa_07(proposal_kloc, est_effort, se_effort, est_defects, se_defects,
                  mean_effort, std_effort, mean_defects, std_defects):
    print("== NASA-07 ==")
    print(f"Proposed KLOC+: {proposal_kloc:.2f}")
    ci_low_eff = est_effort - 1.96 * se_effort
    ci_up_eff = est_effort + 1.96 * se_effort
    print(f"Estimated Effort-: {est_effort:.2f} (95% CI: {ci_low_eff:.2f} - {ci_up_eff:.2f})")
    ci_low_def = est_defects - 1.96 * se_defects
    ci_up_def = est_defects + 1.96 * se_defects
    print(f"Estimated Defects-: {est_defects:.2f} (95% CI: {ci_low_def:.2f} - {ci_up_def:.2f})")

    # Alert thresholds: > mean + 2*std
    effort_high = est_effort > mean_effort + 2 * std_effort
    defects_high = est_defects > mean_defects + 2 * std_defects
    alert_parts = []
    if effort_high:
        alert_parts.append("unusually high Effort")
    if defects_high:
        alert_parts.append("unusually high Defects")
    if alert_parts:
        print("Alert: " + " and ".join(alert_parts) + " detected.")
    else:
        print("No alert.")
    print()  # blank line for readability


def print_nasa_08(kloc, effort, defects):
    print("== NASA-08 ==")
    n_proj = len(kloc)
    print(f"Updated historical summary: {n_proj} projects")
    if n_proj == 0:
        print("  No data available.")
        return
    mean_effort = mean(effort)
    std_effort = stddev(effort, mean_effort)
    mean_defects = mean(defects)
    std_defects = stddev(defects, mean_defects)
    print(f"  Mean Effort-: {mean_effort:.2f}")
    print(f"  Std Dev Effort-: {std_effort:.2f}")
    print(f"  Mean Defects-: {mean_defects:.2f}")
    print(f"  Std Dev Defects-: {std_defects:.2f}")

    # Show regression coefficients for reference
    slope_eff, intercept_eff = linear_regression(kloc, effort)
    slope_def, intercept_def = linear_regression(kloc, defects)
    print(f"  Effort regression: Effort- = {intercept_eff:.2f} + {slope_eff:.4f} * KLOC+")
    print(f"  Defects regression: Defects- = {intercept_def:.2f} + {slope_def:.4f} * KLOC+")
    print()  # blank line for readability


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main(csv_path):
    # Load initial historical data
    kloc, effort, defects = load_historical(csv_path)

    if not kloc:
        print("No valid historical data found. Exiting.")
        return

    mean_effort = mean(effort)
    std_effort = stddev(effort, mean_effort)
    mean_defects = mean(defects)
    std_defects = stddev(defects, mean_defects)

    # --- NASA-07: Proposal assessment loop ---
    while True:
        print("Enter proposed KLOC+ (numeric) or 'done' to finish proposals:")
        inp = input().strip()
        if inp.lower() == 'done':
            break
        try:
            prop_kloc = float(inp)
        except Exception:
            print("Invalid input. Try again.")
            continue

        # Predict effort
        slope_eff, intercept_eff = linear_regression(kloc, effort)
        est_effort, se_effort = predict_with_confidence(
            slope_eff, intercept_eff, prop_kloc, kloc, effort
        )

        # Predict defects
        slope_def, intercept_def = linear_regression(kloc, defects)
        est_defects, se_defects = predict_with_confidence(
            slope_def, intercept_def, prop_kloc, kloc, defects
        )

        print_nasa_07(
            prop_kloc,
            est_effort, se_effort,
            est_defects, se_defects,
            mean_effort, std_effort,
            mean_defects, std_defects
        )

    # --- NASA-08: Update with new completed projects ---
    print("Enter new completed project records (KLOC+, Effort-, Defects-) one per line.")
    print("Format: <KLOC+>,<Effort->,<Defects->")
    print("Type 'done' when finished.")
    while True:
        line = input().strip()
        if line.lower() == 'done':
            break
        parts = line.split(',')
        if len(parts) != 3:
            print("Incorrect format. Expected 3 comma-separated values.")
            continue
        k = parse_float(parts[0])
        e = parse_float(parts[1])
        d = parse_float(parts[2])
        if None in (k, e, d):
            print("One or more values could not be parsed as float. Skipping.")
            continue
        kloc.append(k)
        effort.append(e)
        defects.append(d)

    # Recompute and print updated summary
    print_nasa_08(kloc, effort, defects)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
