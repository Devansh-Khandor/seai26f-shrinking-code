#!/usr/bin/env python3
"""
Skill 01: Inventory summary and balanced performance ranking.

Usage:
    python skill_01.py <path-to-auto93.csv>
"""

import csv
import sys
import statistics
from pathlib import Path


def load_data(csv_path: str):
    """
    Load CSV data into a list of dictionaries with numeric conversion.
    """
    numeric_fields = {"Clndrs", "Volume", "HpX", "Lbs-", "Acc+", "Mpg+"}
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {}
            for key, value in raw.items():
                if key in numeric_fields:
                    # Convert to float (some values are integers, but float covers both)
                    try:
                        row[key] = float(value)
                    except ValueError:
                        # In case of unexpected non-numeric value, skip this row
                        row[key] = None
                else:
                    row[key] = value
            rows.append(row)
    return rows


def compute_stats(rows):
    """
    Compute mean, min, max for weight, mpg, acceleration.
    """
    weights = [r["Lbs-"] for r in rows]
    mpgs = [r["Mpg+"] for r in rows]
    accs = [r["Acc+"] for r in rows]

    stats = {
        "weight": {
            "mean": statistics.mean(weights),
            "min": min(weights),
            "max": max(weights),
        },
        "mpg": {
            "mean": statistics.mean(mpgs),
            "min": min(mpgs),
            "max": max(mpgs),
        },
        "acc": {
            "mean": statistics.mean(accs),
            "min": min(accs),
            "max": max(accs),
        },
    }
    return stats


def compute_balanced_scores(rows):
    """
    Compute a balanced score for each vehicle based on:
        - low weight (better)
        - high mpg (better)
        - low acceleration time (better)
    Scores are normalized to [0,1] per metric, then summed.
    """
    weights = [r["Lbs-"] for r in rows]
    mpgs = [r["Mpg+"] for r in rows]
    accs = [r["Acc+"] for r in rows]

    min_w, max_w = min(weights), max(weights)
    min_m, max_m = min(mpgs), max(mpgs)
    min_a, max_a = min(accs), max(accs)

    def norm(val, min_val, max_val, higher_is_better=True):
        if max_val == min_val:
            return 1.0  # No variation: treat as optimal
        if higher_is_better:
            return (val - min_val) / (max_val - min_val)
        else:
            return (max_val - val) / (max_val - min_val)

    scored_rows = []
    for r in rows:
        w_norm = norm(r["Lbs-"], min_w, max_w, higher_is_better=False)
        m_norm = norm(r["Mpg+"], min_m, max_m, higher_is_better=True)
        a_norm = norm(r["Acc+"], min_a, max_a, higher_is_better=False)
        score = w_norm + m_norm + a_norm
        scored = r.copy()
        scored["score"] = score
        scored_rows.append(scored)

    # Sort descending by score
    scored_rows.sort(key=lambda x: x["score"], reverse=True)
    return scored_rows


def format_value(value, fmt="{:.2f}"):
    """Format a numeric value with two decimal places, or return as is for None."""
    if value is None:
        return "N/A"
    return fmt.format(value)


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).is_file():
        print(f"Error: file '{csv_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    rows = load_data(csv_path)
    if not rows:
        print("Error: no data loaded.", file=sys.stderr)
        sys.exit(1)

    # CAR-01: Inventory summary
    print("== CAR-01 ==")
    stats = compute_stats(rows)
    print(
        f"Weight: mean={format_value(stats['weight']['mean'])} lbs, "
        f"min={format_value(stats['weight']['min'])} lbs, "
        f"max={format_value(stats['weight']['max'])} lbs"
    )
    print(
        f"Fuel efficiency (mpg): mean={format_value(stats['mpg']['mean'])}, "
        f"min={format_value(stats['mpg']['min'])}, "
        f"max={format_value(stats['mpg']['max'])}"
    )
    print(
        f"Acceleration time (s): mean={format_value(stats['acc']['mean'])}, "
        f"min={format_value(stats['acc']['min'])}, "
        f"max={format_value(stats['acc']['max'])}"
    )

    # CAR-02: Balanced performance ranking
    print("\n== CAR-02 ==")
    scored_rows = compute_balanced_scores(rows)
    top_n = min(3, len(scored_rows))
    print(f"Top {top_n} vehicles with balanced performance:")
    for idx in range(top_n):
        r = scored_rows[idx]
        print(
            f"{idx + 1}. Model: {r['Model']}, "
            f"Weight: {format_value(r['Lbs-'])} lbs, "
            f"MPG: {format_value(r['Mpg+'])}, "
            f"Acc: {format_value(r['Acc+'])} s, "
            f"Score: {format_value(r['score'], '{:.3f}')}"
        )


if __name__ == "__main__":
    main()
