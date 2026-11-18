#!/usr/bin/env python3
"""
Skill 04 – Apache performance analytics (APACHE-07 & APACHE-08)

Usage:
    python skill_04.py <csv_path>
"""

import json
import math
import sys
import ezr

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def goal_columns(tbl):
    """Return list of (col_at, col_txt) for columns that end with '+' or '-'."""
    return [(c.at, c.txt) for c in tbl.cols.values() if c.txt.endswith(('+', '-'))]

def compute_summary(rows, goal_at):
    """Compute n, mean, min, max for a goal column over a list of rows."""
    values = [float(r[goal_at]) for r in rows if r[goal_at] != "?"]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": n, "mean": sum(values) / n, "min": min(values), "max": max(values)}

def shortlist(rows, goal_at, count=5):
    """Return the row numbers of the best (lowest) rows for a goal column."""
    scored = [(float(r[goal_at]), i + 1)  # row number is 1‑based
              for i, r in enumerate(rows) if r[goal_at] != "?"]
    scored.sort()  # ascending on performance
    return [num for _, num in scored[:count]]

# --------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    tbl = ezr.Tbl(ezr.csv(csv_path))
    all_rows = tbl.rows
    total = len(all_rows)

    # Goal columns
    goals = goal_columns(tbl)
    if not goals:
        raise RuntimeError("No goal column found.")
    # Assume single goal for simplicity; use the first one
    goal_at, goal_txt = goals[0]

    # Split history and arrivals
    arrivals_len = round(0.2 * total)
    arrivals = all_rows[-arrivals_len:]
    history = all_rows[:-arrivals_len]

    # Budget: at most min(30, total // 5) rows may be evaluated
    budget = min(30, total // 5)

    # --------------------
    # APACHE-07
    # --------------------
    print("== APACHE-07 ==")
    assessments = []

    prev_vals = []
    eval_count = 0
    for i, row in enumerate(arrivals):
        row_num = len(history) + i + 1  # 1‑based
        perf_cell = row[goal_at]
        alert = False

        if eval_count < budget and perf_cell != "?":
            # Compute alert threshold: mean + 2*std of previous values
            eval_count += 1
            prev_vals.append(float(perf_cell))
            if len(prev_vals) > 1:
                mu = sum(prev_vals[:-1]) / (len(prev_vals) - 1)
                var = sum((x - mu) ** 2 for x in prev_vals[:-1]) / (len(prev_vals) - 1)
                std = math.sqrt(var)
                threshold = mu + 2 * std
                alert = float(perf_cell) > threshold
            else:
                # Only one previous point, no threshold
                alert = False
        else:
            alert = False  # not evaluated

        assessments.append({"row": row_num, "alert": alert})
        # Print at least three lines
        print(f"Row {row_num}: alert={alert}")

    # Machine‑checked answer
    print("ANSWER APACHE-07", json.dumps({"assessments": assessments}))

    # --------------------
    # APACHE-08
    # --------------------
    print("\n== APACHE-08 ==")

    # Checkpoints after 0, 19, 38 arrivals
    checkpoint_ms = [0, 19, 38]
    checkpoints = []

    for m in checkpoint_ms:
        # Cap m at arrivals_len
        m_capped = min(m, arrivals_len)
        subset = history + arrivals[:m_capped]

        # Summaries per goal column
        summary = {}
        for at, txt in goals:
            stats = compute_summary(subset, at)
            # Replace None with 0 for JSON compatibility? keep as None
            summary[txt] = stats

        # Shortlist of 5 best rows (lowest performance)
        short = shortlist(subset, goal_at, 5)

        checkpoints.append({"after": m_capped, "summary": summary, "shortlist": short})

        # Print at least three lines
        print(f"Checkpoint after {m_capped} arrivals:")
        for txt, stats in summary.items():
            print(f"  {txt}: n={stats['n']} mean={stats['mean']} min={stats['min']} max={stats['max']}")
        print(f"  Shortlist: {short}")

    # Machine‑checked answer
    print("ANSWER APACHE-08", json.dumps({"checkpoints": checkpoints}))

if __name__ == "__main__":
    main()
