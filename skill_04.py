#!/usr/bin/env python3
"""
skill_04.py: Assess new proposals and update historical summaries.
Usage: python skill_04.py <csv-file>
"""

import json
import sys
import math
from typing import List, Dict, Tuple

import ezr


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def read_rows(path: str) -> List[ezr.ROW]:
    """Read CSV and return rows as list."""
    return list(ezr.csv(path))


def split_history_arrivals(rows: List[ezr.ROW]) -> Tuple[List[ezr.ROW], List[ezr.ROW]]:
    """Split rows into history and arrivals according to spec."""
    n = len(rows)
    arrivals_count = round(n * 0.2)
    arrivals_start = n - arrivals_count
    return rows[:arrivals_start], rows[arrivals_start:]


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (n - 1)) ** 0.5


def ci_bounds(mean_val: float, sd_val: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Return lower and upper bounds of a 95% confidence interval for a mean."""
    if n == 0:
        return mean_val, mean_val
    margin = z * sd_val / (n ** 0.5)
    return mean_val - margin, mean_val + margin


def compute_goal_stats(rows: List[ezr.ROW], goal_cols: List[ezr.COL]) -> Dict[str, Dict[str, float]]:
    """Compute n, mean, min, max for each goal column over given rows."""
    stats: Dict[str, Dict[str, float]] = {}
    for col in goal_cols:
        values = [float(r[col.at]) for r in rows if r[col.at] != "?"]
        n = len(values)
        if n == 0:
            stats[col.txt] = {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        else:
            stats[col.txt] = {
                "n": n,
                "mean": mean(values),
                "min": min(values),
                "max": max(values),
            }
    return stats


def compute_shortlist(rows: List[ezr.ROW], tbl: ezr.TBL, top_n: int = 5) -> List[int]:
    """Return the row numbers (1‑based) of the best rows according to ydist."""
    # Use full table stats for ydist calculation
    ydists = [(idx + 1, ezr.ydist(tbl, r)) for idx, r in enumerate(rows)]
    ydists.sort(key=lambda x: x[1])  # ascending
    return [num for num, _ in ydists[:top_n]]


# ------------------------------------------------------------------
# NASA-07
# ------------------------------------------------------------------
def nasa07_assess(rows: List[ezr.ROW], history: List[ezr.ROW],
                  arrivals: List[ezr.ROW], tbl: ezr.TBL) -> List[Dict]:
    """Predict effort and defects for each arrival and flag alerts."""
    # Determine budget: at most min(30, n//5)
    n_rows = len(rows)
    budget = min(30, n_rows // 5)
    # Only evaluate up to budget arrivals
    arrivals_to_use = arrivals[:budget]

    # Identify goal columns
    goal_cols = [col for col in tbl.cols.values() if col.txt.endswith(("+", "-"))]
    # Map column names to col objects for Effort- and Defects-
    effort_col = next((c for c in goal_cols if c.txt == "Effort-"), None)
    defects_col = next((c for c in goal_cols if c.txt == "Defects-"), None)

    assessments: List[Dict] = []

    # Training set evolves as we process arrivals
    training_rows = history.copy()
    for arrival_idx, arrival in enumerate(arrivals_to_use):
        # Update training set
        training_rows.append(arrival)

        # Gather training values for effort and defects
        effort_vals = [float(r[effort_col.at]) for r in training_rows if r[effort_col.at] != "?"]
        defects_vals = [float(r[defects_col.at]) for r in training_rows if r[defects_col.at] != "?"]

        # Means
        effort_mean = mean(effort_vals)
        defects_mean = mean(defects_vals)

        # Confidence bounds
        eff_n = len(effort_vals)
        def_n = len(defects_vals)
        eff_sd = std(effort_vals)
        def_sd = std(defects_vals)
        _, eff_high = ci_bounds(effort_mean, eff_sd, eff_n)
        _, def_high = ci_bounds(defects_mean, def_sd, def_n)

        # Alerts if predicted > upper bound
        alert_eff = effort_mean > eff_high
        alert_def = defects_mean > def_high
        alert = alert_eff or alert_def

        # Row number (1-based in original file)
        row_num = len(history) + arrival_idx + 1  # arrivals start after history

        assessments.append({
            "row": row_num,
            "predicted_effort": effort_mean,
            "predicted_defects": defects_mean,
            "alert": alert,
        })

    return assessments


# ------------------------------------------------------------------
# NASA-08
# ------------------------------------------------------------------
def nasa08_checkpoints(rows: List[ezr.ROW], history: List[ezr.ROW],
                       arrivals: List[ezr.ROW], tbl: ezr.TBL) -> List[Dict]:
    """Compute summaries and shortlist after 0, 9 and 19 arrivals."""
    checkpoints = []
    for m in [0, 9, 19]:
        # Use history + first m arrivals
        seen_rows = history + arrivals[:m]
        # Compute goal stats
        goal_cols = [col for col in tbl.cols.values() if col.txt.endswith(("+", "-"))]
        summary = compute_goal_stats(seen_rows, goal_cols)
        # Compute shortlist
        shortlist = compute_shortlist(seen_rows, tbl, top_n=5)
        checkpoints.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist,
        })
    return checkpoints


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv-file>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_rows(csv_path)

    # Build the full table (for stats, ydist etc.)
    tbl = ezr.Tbl([rows[0].__class__([x for x in rows[0]] if isinstance(rows[0], list) else rows[0]] if False else rows)

    # Actually create proper table: header is first row of original csv, which is part of rows? Wait read_rows gives all rows, not header.
    # Need to read csv again to get header separately
    # Let's redo reading properly
    with open(csv_path, encoding="utf-8-sig") as f:
        header = next(f).strip().split(",")
    rows_data = [row for row in rows]  # already parsed
    # Build tbl using header and rows_data
    tbl = ezr.Tbl([header] + rows_data)

    # Split into history and arrivals
    history, arrivals = split_history_arrivals(rows_data)

    # NASA-07
    assessments = nasa07_assess(rows_data, history, arrivals, tbl)

    print("== NASA-07 ==")
    for a in assessments:
        print(f"Row {a['row']}: predicted Effort- = {a['predicted_effort']:.3g}, "
              f"predicted Defects- = {a['predicted_defects']:.3g}, alert = {a['alert']}")
    print("ANSWER NASA-07", json.dumps({"assessments": assessments}))

    # NASA-08
    checkpoints = nasa08_checkpoints(rows_data, history, arrivals, tbl)

    print("\n== NASA-08 ==")
    for cp in checkpoints:
        print(f"After {cp['after']} arrivals:")
        for goal, stats in cp["summary"].items():
            print(f"  {goal}: n={stats['n']}, mean={stats['mean']:.3g}, "
                  f"min={stats['min']:.3g}, max={stats['max']:.3g}")
        print("  Shortlist:", ", ".join(map(str, cp["shortlist"])))
    print("ANSWER NASA-08", json.dumps({"checkpoints": checkpoints}))


if __name__ == "__main__":
    main()
