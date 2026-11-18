#!/usr/bin/env python3
"""
Skill 02 – NASA-03 & NASA-04
"""
import csv
import sys
import json
import statistics

# Reuse the same constants and helper as in skill_01
GOALS = ["Kloc+", "Effort-", "Defects-"]  # order matters for output

def parse_value(v):
    """Parse a CSV cell to float, return None for missing."""
    if v == "" or v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None

# ----------------------------------------------------------------------
# Simple linear regression with a single predictor (YearX)
# ----------------------------------------------------------------------
def train_linreg(years, values):
    """
    Train y = a + b * year
    years and values are lists of floats
    """
    n = len(years)
    if n == 0:
        return None, None
    mean_x = sum(years) / n
    mean_y = sum(values) / n
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(years, values))
    var_x = sum((x - mean_x) ** 2 for x in years)
    if var_x == 0:
        b = 0.0
    else:
        b = cov_xy / var_x
    a = mean_y - b * mean_x
    return a, b

def predict_linreg(a, b, year):
    """Predict y for a given year. If year is None, return mean_y (a)."""
    if year is None:
        return a  # intercept
    return a + b * year

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]

    # Load rows
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for col in reader.fieldnames:
                if col in GOALS:
                    parsed[col] = parse_value(row[col])
                else:
                    parsed[col] = row[col]
            rows.append(parsed)

    # Identify held‑out rows (row number divisible by 5)
    held_out = [idx + 1 for idx in range(len(rows)) if (idx + 1) % 5 == 0]
    training_rows = [row for idx, row in enumerate(rows) if (idx + 1) not in held_out]

    # Use YearX as the single predictor
    # Collect training data for Effort-
    train_years_effort = []
    train_values_effort = []
    for r in training_rows:
        y_val = r["Effort-"]
        year_val = parse_value(r["YearX"])
        if y_val is not None and year_val is not None:
            train_years_effort.append(year_val)
            train_values_effort.append(y_val)
    a_eff, b_eff = train_linreg(train_years_effort, train_values_effort)

    # Collect training data for Defects-
    train_years_defect = []
    train_values_defect = []
    for r in training_rows:
        d_val = r["Defects-"]
        year_val = parse_value(r["YearX"])
        if d_val is not None and year_val is not None:
            train_years_defect.append(year_val)
            train_values_defect.append(d_val)
    a_def, b_def = train_linreg(train_years_defect, train_values_defect)

    # Compute residual std for Defects- (used for uncertainty)
    residuals_def = []
    for x, y in zip(train_years_defect, train_values_defect):
        pred = predict_linreg(a_def, b_def, x)
        residuals_def.append(y - pred)
    if len(residuals_def) > 1:
        std_def = statistics.stdev(residuals_def)
    else:
        std_def = 0.0

    # ------------------------------------------------------------------
    # NASA-03: Predict Effort-
    # ------------------------------------------------------------------
    predictions_effort = {}
    mae_values = []
    for h in held_out:
        year_val = parse_value(rows[h - 1]["YearX"])
        pred = predict_linreg(a_eff, b_eff, year_val)
        predictions_effort[str(h)] = pred
        actual = rows[h - 1]["Effort-"]
        if actual is not None:
            mae_values.append(abs(pred - actual))

    mae = sum(mae_values) / len(mae_values) if mae_values else None

    # Print results
    print("== NASA-03 ==")
    # at least three lines of results
    for i, (row_str, val) in enumerate(predictions_effort.items()[:3], start=1):
        print(f"Prediction {i} – row {row_str}: Effort- ≈ {val:.3f}")

    answer_03 = {"predictions": predictions_effort, "mae": mae}
    print("ANSWER NASA-03 " + json.dumps(answer_03, separators=(',',':')))

    # ------------------------------------------------------------------
    # NASA-04: Predict Defects- with uncertainty
    # ------------------------------------------------------------------
    predictions_defects = {}
    for h in held_out:
        year_val = parse_value(rows[h - 1]["YearX"])
        estimate = predict_linreg(a_def, b_def, year_val)
        low = estimate - 2 * std_def
        high = estimate + 2 * std_def
        # Clamp to non‑negative
        if low < 0:
            low = 0.0
        if high < 0:
            high = 0.0
        predictions_defects[str(h)] = {
            "estimate": estimate,
            "low": low,
            "high": high
        }

    print("\n== NASA-04 ==")
    for i, (row_str, d) in enumerate(predictions_defects.items()[:3], start=1):
        print(f"Prediction {i} – row {row_str}: Defects- estimate {d['estimate']:.3f} "
              f"(± {2*std_def:.3f}) -> [{d['low']:.3f}, {d['high']:.3f}]")

    answer_04 = {"predictions": predictions_defects}
    print("ANSWER NASA-04 " + json.dumps(answer_04, separators=(',',':')))

if __name__ == "__main__":
    main()
