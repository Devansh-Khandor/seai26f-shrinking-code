#!/usr/bin/env python3
"""
skill_04.py – NASA-07 & NASA-08

NASA-07: For each new proposal (arrival) predict Effort- and Defects- from Kloc+.
An alert is raised when the prediction is unusually high ( > mean + 2*std of previous actuals ).
NASA-08: As new completed projects arrive, produce checkpoints (after 0, 9, 19 arrivals) showing
summary statistics for each goal column and a shortlist of the 5 best rows seen so far.

The program prints a human‑readable report for each NASA ID followed by a machine‑checked
JSON answer line.  The JSON values are derived purely from the supplied CSV file.
"""

import csv
import json
import math
import sys


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV into list of dicts (preserve order)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(val):
    """Convert a string to float, handling '?' and empty."""
    if val is None:
        return None
    val = val.strip()
    if val == "" or val == "?":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def linear_regression(x_vals, y_vals):
    """Least‑squares linear regression: returns slope, intercept."""
    n = len(x_vals)
    if n == 0:
        return 0.0, 0.0
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
    var_x = sum((x - mean_x) ** 2 for x in x_vals)
    if var_x == 0.0:
        # All x the same; return zero slope, mean of y as intercept
        return 0.0, mean_y
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def std_dev(lst):
    if not lst:
        return 0.0
    m = mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))


def score_row(row, goal_cols, mins, maxs):
    """Compute composite score for a row based on normalized goal values."""
    score = 0.0
    for col in goal_cols:
        v = to_float(row[col])
        if v is None:
            continue
        mn = mins[col]
        mx = maxs[col]
        if mn == mx:
            norm = 0.0
        else:
            if col.endswith("+"):
                norm = (v - mn) / (mx - mn)
            else:  # ends with '-'
                norm = (mx - v) / (mx - mn)
        score += norm
    return score


# ----------------------------------------------------------------------
# NASA-07 logic
# ----------------------------------------------------------------------
def nasa07(rows, history_rows, arrivals):
    """Generate predictions for each arrival."""
    assessments = []
    # Pre‑compute the list of goal columns for regression
    goal_cols = [c for c in rows[0].keys() if c.endswith(("+", "-"))]
    # We'll use only Kloc+, Effort- and Defects- for predictions
    kloc_col = "Kloc+"
    effort_col = "Effort-"
    defects_col = "Defects-"

    for arrival_index, arrival_row in enumerate(arrivals, start=1):
        # Training data: history + earlier arrivals (excluding this one)
        train_rows = history_rows + arrivals[:arrival_index - 1]

        # Gather training data for regression
        x_eff = []
        y_eff = []
        x_def = []
        y_def = []
        for r in train_rows:
            x = to_float(r[kloc_col])
            y_e = to_float(r[effort_col])
            y_d = to_float(r[defects_col])
            if x is not None and y_e is not None:
                x_eff.append(x)
                y_eff.append(y_e)
            if x is not None and y_d is not None:
                x_def.append(x)
                y_def.append(y_d)

        # Compute regression parameters
        eff_slope, eff_intercept = linear_regression(x_eff, y_eff)
        def_slope, def_intercept = linear_regression(x_def, y_def)

        # Handle missing Kloc+ in arrival
        kloc_val = to_float(arrival_row[kloc_col])
        if kloc_val is None:
            # Impute with mean Kloc+ from training data
            kloc_vals = [to_float(r[kloc_col]) for r in train_rows]
            kloc_vals = [v for v in kloc_vals if v is not None]
            kloc_val = mean(kloc_vals) if kloc_vals else 0.0

        # Predictions
        pred_effort = eff_slope * kloc_val + eff_intercept
        pred_defects = def_slope * kloc_val + def_intercept

        # Determine alert thresholds using previous actual values
        actual_efforts = [to_float(r[effort_col]) for r in train_rows if to_float(r[effort_col]) is not None]
        actual_defects = [to_float(r[defects_col]) for r in train_rows if to_float(r[defects_col]) is not None]

        # Compute alert for effort
        alert_effort = False
        if actual_efforts:
            m = mean(actual_efforts)
            s = std_dev(actual_efforts)
            threshold = m + 2 * s
            if pred_effort > threshold:
                alert_effort = True
        # Compute alert for defects
        alert_defects = False
        if actual_defects:
            m = mean(actual_defects)
            s = std_dev(actual_defects)
            threshold = m + 2 * s
            if pred_defects > threshold:
                alert_defects = True

        alert = alert_effort or alert_defects

        row_number = len(history_rows) + arrival_index  # 1‑based index in original rows
        assessments.append(
            {
                "row": row_number,
                "predicted_effort": pred_effort,
                "predicted_defects": pred_defects,
                "alert": alert,
            }
        )

    return assessments


# ----------------------------------------------------------------------
# NASA-08 logic
# ----------------------------------------------------------------------
def nasa08(rows, history_rows, arrivals, checkpoint_m_list):
    """Generate checkpoint summaries and shortlist."""
    checkpoints = []
    goal_cols = [c for c in rows[0].keys() if c.endswith(("+", "-"))]

    for m in checkpoint_m_list:
        # Rows considered up to this checkpoint
        considered_rows = history_rows + arrivals[:m]
        after = m  # arrivals processed

        # Summary statistics per goal column
        summary = {}
        for col in goal_cols:
            vals = [to_float(r[col]) for r in considered_rows]
            vals = [v for v in vals if v is not None]
            n = len(vals)
            if n == 0:
                mean_val = None
                min_val = None
                max_val = None
            else:
                mean_val = mean(vals)
                min_val = min(vals)
                max_val = max(vals)
            summary[col] = {
                "n": n,
                "mean": mean_val,
                "min": min_val,
                "max": max_val,
            }

        # Shortlist of top 5 rows overall so far
        if not considered_rows:
            shortlist = []
        else:
            # Pre‑compute mins and maxs for normalization
            mins = {}
            maxs = {}
            for col in goal_cols:
                vals = [to_float(r[col]) for r in considered_rows]
                vals = [v for v in vals if v is not None]
                if vals:
                    mins[col] = min(vals)
                    maxs[col] = max(vals)
                else:
                    mins[col] = maxs[col] = 0.0
            # Compute scores
            scored = []
            for r in considered_rows:
                idx = rows.index(r) + 1  # 1‑based
                s = score_row(r, goal_cols, mins, maxs)
                scored.append((idx, s))
            scored.sort(key=lambda x: x[1], reverse=True)
            shortlist = [idx for idx, _ in scored[:5]]

        checkpoints.append(
            {
                "after": after,
                "summary": summary,
                "shortlist": shortlist,
            }
        )

    return checkpoints


# ----------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        sys.exit(1)

    total_rows = len(rows)
    arrivals_count = round(0.2 * total_rows)
    history_rows = rows[: total_rows - arrivals_count]
    arrivals = rows[total_rows - arrivals_count :]

    # Determine checkpoints: after 0, 9, 19 arrivals (cap by actual arrivals)
    checkpoint_ms = []
    for m in [0, 9, 19]:
        if m <= arrivals_count:
            checkpoint_ms.append(m)

    # --------------------------------------------------------------
    # NASA-07
    # --------------------------------------------------------------
    print("== NASA-07 ==")
    assessments = nasa07(rows, history_rows, arrivals)
    # Print first three arrivals as human‑readable lines
    for a in assessments[:3]:
        print(
            f"Row {a['row']}: Effort={a['predicted_effort']:.2f}, "
            f"Defects={a['predicted_defects']:.2f}, Alert={a['alert']}"
        )
    # Machine‑checked answer
    answer_07 = {"assessments": assessments}
    print("ANSWER NASA-07", json.dumps(answer_07, separators=(",", ":")))

    # --------------------------------------------------------------
    # NASA-08
    # --------------------------------------------------------------
    print("\n== NASA-08 ==")
    checkpoints = nasa08(rows, history_rows, arrivals, checkpoint_ms)
    for cp in checkpoints:
        print(f"\nCheckpoint after {cp['after']} arrivals:")
        # Human‑readable summary lines (first 3 goal columns if available)
        displayed = 0
        for col, stats in cp["summary"].items():
            if displayed >= 3:
                break
            print(
                f"  {col}: n={stats['n']}, "
                f"mean={stats['mean'] if stats['mean'] is not None else 'None'}, "
                f"min={stats['min'] if stats['min'] is not None else 'None'}, "
                f"max={stats['max'] if stats['max'] is not None else 'None'}"
            )
            displayed += 1
        print(f"  Shortlist top rows: {cp['shortlist'][:5]}")

    # Machine‑checked answer
    answer_08 = {"checkpoints": checkpoints}
    print("ANSWER NASA-08", json.dumps(answer_08, separators=(",", ":")))


if __name__ == "__main__":
    main()
