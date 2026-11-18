#!/usr/bin/env python3
"""
Skill 04 – NASA-07 & NASA-08
"""
import csv
import json
import sys
import math
import statistics
from typing import List, Dict, Any

# Import helpers from the previous skills
# These functions are pure and work on the parsed rows we will build here.
from skill_01 import compute_stats, normalise_composite
from skill_02 import train_linreg, predict_linreg

GOALS = ["Kloc+", "Effort-", "Defects-"]  # order matters for output


def parse_value(v: str) -> Any:
    """Parse a CSV cell to float, return None for missing."""
    if v == "" or v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Load rows with parsed values
    # ------------------------------------------------------------------
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for col in reader.fieldnames:
                if col in GOALS:
                    parsed[col] = parse_value(row[col])
                else:
                    # For non-goal columns, keep raw string; treat '?' or '' as None
                    val = row[col]
                    parsed[col] = None if val in ("", "?") else val
            rows.append(parsed)

    n_rows = len(rows)
    # ------------------------------------------------------------------
    # Identify arrivals and history
    # ------------------------------------------------------------------
    arrivals_count = int(round(0.2 * n_rows))
    arrivals = rows[-arrivals_count:]          # list of arrival rows
    history = rows[:-arrivals_count]           # list of history rows

    # Compute history statistics for alert threshold
    hist_effort_vals = [r["Effort-"] for r in history if r["Effort-"] is not None]
    hist_defect_vals = [r["Defects-"] for r in history if r["Defects-"] is not None]

    if hist_effort_vals:
        effort_mean = sum(hist_effort_vals) / len(hist_effort_vals)
        effort_std = statistics.stdev(hist_effort_vals) if len(hist_effort_vals) > 1 else 0.0
    else:
        effort_mean = effort_std = 0.0

    if hist_defect_vals:
        defect_mean = sum(hist_defect_vals) / len(hist_defect_vals)
        defect_std = statistics.stdev(hist_defect_vals) if len(hist_defect_vals) > 1 else 0.0
    else:
        defect_mean = defect_std = 0.0

    # Budget: at most min(30, number of rows // 5) rows may be evaluated
    budget = min(30, n_rows // 5)
    processed_arrivals = arrivals[: min(budget, arrivals_count)]

    # ------------------------------------------------------------------
    # NASA-07: Assess new proposals
    # ------------------------------------------------------------------
    assessments = []

    # For incremental training, we keep a growing training set
    training_rows = history.copy()

    for arrival_row in processed_arrivals:
        # Train on current training_rows
        # Collect training data for Effort-
        train_years_effort = []
        train_vals_effort = []
        for r in training_rows:
            y_val = r["Effort-"]
            year_val = parse_value(r["YearX"])
            if y_val is not None and year_val is not None:
                train_years_effort.append(year_val)
                train_vals_effort.append(y_val)
        a_eff, b_eff = train_linreg(train_years_effort, train_vals_effort)

        # Collect training data for Defects-
        train_years_defect = []
        train_vals_defect = []
        for r in training_rows:
            d_val = r["Defects-"]
            year_val = parse_value(r["YearX"])
            if d_val is not None and year_val is not None:
                train_years_defect.append(year_val)
                train_vals_defect.append(d_val)
        a_def, b_def = train_linreg(train_years_defect, train_vals_defect)

        # Prediction for this arrival
        year_val = parse_value(arrival_row["YearX"])
        pred_effort = predict_linreg(a_eff, b_eff, year_val)
        pred_defect = predict_linreg(a_def, b_def, year_val)

        # Alert logic: unusually high if > mean + 2*std of historical data
        alert = False
        if pred_effort is not None and (pred_effort > effort_mean + 2 * effort_std):
            alert = True
        if pred_defect is not None and (pred_defect > defect_mean + 2 * defect_std):
            alert = True

        assessment = {
            "row": rows.index(arrival_row) + 1,  # 1‑based
            "predicted_effort": pred_effort,
            "predicted_defects": pred_defect,
            "alert": alert,
        }
        assessments.append(assessment)

        # Add this arrival to training set for next iteration
        training_rows.append(arrival_row)

    # ------------------------------------------------------------------
    # NASA-08: Update summaries at checkpoints
    # ------------------------------------------------------------------
    checkpoints = []
    # checkpoints after m = 0, 9, 19 arrivals
    for m in (0, 9, 19):
        # Limit m to available processed arrivals
        m = min(m, len(processed_arrivals))
        # Rows considered: history + first m arrivals (not just processed ones)
        considered_rows = history + arrivals[:m]

        # Compute per‑goal stats
        summary = {}
        for goal in GOALS:
            vals = [r[goal] for r in considered_rows if r[goal] is not None]
            n = len(vals)
            if n:
                mean_val = sum(vals) / n
                min_val = min(vals)
                max_val = max(vals)
            else:
                mean_val = None
                min_val = None
                max_val = None
            summary[goal] = {"n": n, "mean": mean_val, "min": min_val, "max": max_val}

        # Shortlist best 5 rows seen so far by composite score
        scores = normalise_composite(considered_rows)  # list of (row_num, score)
        scores.sort(key=lambda x: x[1], reverse=True)
        shortlist = [row_num for row_num, _ in scores[:5]]

        checkpoints.append({"after": m, "summary": summary, "shortlist": shortlist})

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print("== NASA-07 ==")
    # At least three lines of results
    print(f"Processed {len(processed_arrivals)} arrivals (budget {budget}).")
    if assessments:
        first = assessments[0]
        print(f"First assessment: row {first['row']} predicted Effort- {first['predicted_effort']:.3f} "
              f"predicted Defects- {first['predicted_defects']:.3f} alert={first['alert']}")
        print(f"Last assessment: row {assessments[-1]['row']} alert={assessments[-1]['alert']}")
    print(f"Total alerts: {sum(1 for a in assessments if a['alert'])}")

    answer_07 = {"assessments": assessments}
    print("ANSWER NASA-07 " + json.dumps(answer_07, separators=(",", ":")))

    print("\n== NASA-08 ==")
    # Print at least three lines
    print(f"Total checkpoints: {len(checkpoints)}")
    for cp in checkpoints[:3]:
        print(f"Checkpoint after {cp['after']} arrivals: shortlist {cp['shortlist']}")
    # Full answer
    answer_08 = {"checkpoints": checkpoints}
    print("ANSWER NASA-08 " + json.dumps(answer_08, separators=(",", ":")))


if __name__ == "__main__":
    main()
