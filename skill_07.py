#!/usr/bin/env python3
"""
Skill 07: Identify promising combinations of project characteristics
and recommend proposals to evaluate under a limited budget.

NASA-13: Find historical characteristic combinations that delivered
        more software (higher KLOC+) while keeping effort and defects low.

NASA-14: Given a budget of how many proposals can be evaluated,
        rank the remaining proposals by a benefit‑cost score
        (high KLOC+, low Effort‑, low Defects-).

Usage:
    python skill_07.py <csv_path>
"""

import csv
import math
import sys
import itertools

# --------------------------------------------------------------------------- #
# Parsing utilities
# --------------------------------------------------------------------------- #
def parse_float(val: str | None) -> float | None:
    """Safely parse a string to float; return None on failure."""
    if val is None:
        return None
    try:
        return float(val.strip())
    except (AttributeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Data ingestion
# --------------------------------------------------------------------------- #
def read_csv(csv_path: str):
    """
    Read the CSV file and return a list of project rows.
    Each row is a dict with numeric fields parsed as floats.
    """
    numeric_cols = ["Kloc+", "Effort-", "Defects-",
                    "MonthsX", "YearX", "centerX"]
    rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = raw.copy()
            # Parse numeric fields
            for col in numeric_cols:
                row[col] = parse_float(row.get(col))
            rows.append(row)

    return rows


# --------------------------------------------------------------------------- #
# Helper for selecting categorical columns
# --------------------------------------------------------------------------- #
def get_categorical_columns(rows):
    """Return a list of column names that are categorical."""
    numeric_cols = {"Kloc+", "Effort-", "Defects-",
                    "MonthsX", "YearX", "centerX"}
    all_keys = set(rows[0].keys())
    cats = [k for k in all_keys if k not in numeric_cols and k != "id"]
    return cats


# --------------------------------------------------------------------------- #
# Group statistics
# --------------------------------------------------------------------------- #
def group_by_pair(rows, col1, col2):
    """Return a dict mapping (v1, v2) to list of rows."""
    groups = {}
    for r in rows:
        key = (r.get(col1), r.get(col2))
        groups.setdefault(key, []).append(r)
    return groups


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stddev(values, mean_val):
    n = len(values)
    if n < 2:
        return 0.0
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / (n - 1))


# --------------------------------------------------------------------------- #
# Composite scoring
# --------------------------------------------------------------------------- #
def z_score(value, mean_val, std_val):
    """Return z‑score; guard against zero std."""
    return (value - mean_val) / std_val if std_val > 0 else 0.0


def compute_group_score(mean_kloc, mean_effort, mean_defects,
                       overall_stats):
    """
    Composite score = (KLOC z) - (Effort z) - (Defects z)
    Higher is better: large size, low cost, low defects.
    """
    kloc_z = z_score(mean_kloc,
                     overall_stats["mean_kloc"],
                     overall_stats["std_kloc"])
    eff_z = z_score(mean_effort,
                    overall_stats["mean_effort"],
                    overall_stats["std_effort"])
    def_z = z_score(mean_defects,
                    overall_stats["mean_defects"],
                    overall_stats["std_defects"])
    return kloc_z - eff_z - def_z


# --------------------------------------------------------------------------- #
# NASA‑13: promising characteristic combinations
# --------------------------------------------------------------------------- #
def nasa_13(rows):
    cats = get_categorical_columns(rows)
    if len(cats) < 2:
        print("== NASA-13 ==")
        print("Not enough categorical columns to form pairs.")
        return

    # Compute overall statistics for z‑scoring
    all_kloc = [r["Kloc+"] for r in rows if r["Kloc+"] is not None]
    all_effort = [r["Effort-"] for r in rows if r["Effort-"] is not None]
    all_defects = [r["Defects-"] for r in rows if r["Defects-"] is not None]
    overall_stats = {
        "mean_kloc": mean(all_kloc),
        "std_kloc": stddev(all_kloc, mean(all_kloc)),
        "mean_effort": mean(all_effort),
        "std_effort": stddev(all_effort, mean(all_effort)),
        "mean_defects": mean(all_defects),
        "std_defects": stddev(all_defects, mean(all_defects)),
    }

    best_groups = []

    # Iterate all unordered pairs of categorical columns
    for col1, col2 in itertools.combinations(cats, 2):
        groups = group_by_pair(rows, col1, col2)
        for (v1, v2), g_rows in groups.items():
            # Skip groups too small
            if len(g_rows) < 3:
                continue
            kloc_vals = [r["Kloc+"] for r in g_rows if r["Kloc+"] is not None]
            effort_vals = [r["Effort-"] for r in g_rows if r["Effort-"] is not None]
            defect_vals = [r["Defects-"] for r in g_rows if r["Defects-"] is not None]
            if not kloc_vals or not effort_vals or not defect_vals:
                continue
            m_kloc = mean(kloc_vals)
            m_effort = mean(effort_vals)
            m_defect = mean(defect_vals)
            score = compute_group_score(m_kloc, m_effort, m_defect, overall_stats)
            best_groups.append(((col1, v1, col2, v2), m_kloc, m_effort, m_defect, score))

    # Pick top 3 groups by score
    best_groups.sort(key=lambda x: x[4], reverse=True)
    top_n = best_groups[:3]

    print("== NASA-13 ==")
    if not top_n:
        print("No groups found with sufficient data.")
        return

    for (col1, v1, col2, v2), m_kloc, m_effort, m_defect, score in top_n:
        print(f"{col1}={v1}, {col2}={v2} => "
              f"KLOC+= {m_kloc:.2f}, Effort-={m_effort:.2f}, "
              f"Defects-={m_defect:.2f}, Score={score:.3f}")


# --------------------------------------------------------------------------- #
# NASA‑14: propose evaluation order with limited budget
# --------------------------------------------------------------------------- #
def nasa_14(rows):
    # Compute overall z‑scoring stats as in NASA‑13
    all_kloc = [r["Kloc+"] for r in rows if r["Kloc+"] is not None]
    all_effort = [r["Effort-"] for r in rows if r["Effort-"] is not None]
    all_defects = [r["Defects-"] for r in rows if r["Defects-"] is not None]
    overall_stats = {
        "mean_kloc": mean(all_kloc),
        "std_kloc": stddev(all_kloc, mean(all_kloc)),
        "mean_effort": mean(all_effort),
        "std_effort": stddev(all_effort, mean(all_effort)),
        "mean_defects": mean(all_defects),
        "std_defects": stddev(all_defects, mean(all_defects)),
    }

    # Ask user for evaluation budget
    try:
        inp = input("Enter number of proposals you can evaluate (budget): ").strip()
        budget = int(inp)
        if budget <= 0:
            raise ValueError
    except Exception:
        print("Invalid input; default budget = 3.")
        budget = 3

    # Compute score for each proposal
    scored_rows = []
    for r in rows:
        k = r["Kloc+"]
        e = r["Effort-"]
        d = r["Defects-"]
        if k is None or e is None or d is None:
            continue
        score = compute_group_score(k, e, d, overall_stats)
        scored_rows.append((score, r))

    scored_rows.sort(key=lambda x: x[0], reverse=True)

    print("== NASA-14 ==")
    print(f"Top {budget} proposals to evaluate first (based on score):")
    for i in range(min(budget, len(scored_rows))):
        score, r = scored_rows[i]
        print(f"  {r.get('id', 'N/A')}: KLOC+= {r['Kloc+']:.2f}, "
              f"Effort-={r['Effort-']:.2f}, Defects-={r['Defects-']:.2f}, "
              f"Score={score:.3f}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    if not rows:
        print("No data loaded.")
        sys.exit(1)

    nasa_13(rows)
    print()
    nasa_14(rows)


if __name__ == "__main__":
    main()
