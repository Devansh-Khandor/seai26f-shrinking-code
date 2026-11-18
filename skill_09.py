#!/usr/bin/env python3
"""
skill_09.py

Usage:
    python skill_09.py <csv_path>
"""

import csv
import json
import sys
from statistics import mean, stdev
from collections import defaultdict

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    print("scikit-learn is required for this script.", file=sys.stderr)
    sys.exit(1)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def is_goal_col(col_name):
    """Goal columns end with '+' or '-'."""
    return col_name.endswith('+') or col_name.endswith('-')

def is_input_col(col_name):
    """Input columns are those that are not goal columns and do not end with 'X'."""
    return not col_name.endswith('X') and not is_goal_col(col_name)

def parse_int(val):
    """Parse binary 0/1, missing '?' -> None."""
    if val == '?':
        return None
    return int(val)

def parse_float(val):
    """Parse float or missing '?' -> None."""
    if val == '?':
        return None
    return float(val)

def hamming_distance(row_a, row_b, input_cols):
    """Count differing values over input columns. Treat None as a distinct value."""
    diff = 0
    for col in input_cols:
        va = row_a[col]
        vb = row_b[col]
        if va != vb:
            diff += 1
    return diff

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        raw_rows = list(reader)

    n_rows = len(raw_rows)

    # Identify columns
    goal_cols = [c for c in fieldnames if is_goal_col(c)]
    if not goal_cols:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)

    # Pick minimization target if present, else first goal
    target_col = None
    for c in goal_cols:
        if c.endswith('-'):
            target_col = c
            break
    if not target_col:
        target_col = goal_cols[0]

    input_cols = [c for c in fieldnames if is_input_col(c)]

    # Parse rows into dict of parsed values
    parsed_rows = []
    for r in raw_rows:
        parsed = {}
        for c in fieldnames:
            if is_goal_col(c):
                parsed[c] = parse_float(r[c])
            else:
                parsed[c] = parse_int(r[c])
        parsed_rows.append(parsed)

    # -------------------------------------------------------------------- #
    # APACHE-17: Flag anomalous performance
    # -------------------------------------------------------------------- #
    print("== APACHE-17 ==")

    # Build training data: all rows with target present
    X_train = []
    y_train = []
    for r in parsed_rows:
        val = r[target_col]
        if val is not None:
            # Replace missing inputs with 0 for training
            x = [cval if cval is not None else 0 for cval in [r[c] for c in input_cols]]
            X_train.append(x)
            y_train.append(val)

    if X_train:
        # Train RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            min_samples_split=2,
            max_depth=None,
        )
        model.fit(X_train, y_train)

        # Predict and compute residuals
        residuals = []
        for idx, r in enumerate(parsed_rows, start=1):
            val = r[target_col]
            if val is not None:
                x = [cval if cval is not None else 0 for cval in [r[c] for c in input_cols]]
                pred = model.predict([x])[0]
                resid = abs(pred - val)
                residuals.append((resid, idx, val, pred))
        # Sort by residual descending
        residuals.sort(key=lambda x: x[0], reverse=True)
        flagged17 = []
        for resid, idx, val, pred in residuals[:20]:
            flagged17.append({
                "row": idx,
                "column": target_col,
                "actual": round(val, 3),
                "expected": round(pred, 3)
            })
    else:
        flagged17 = []

    # Print summary
    print(f"Total rows processed: {n_rows}")
    print(f"Rows with valid performance: {len([r for r in parsed_rows if r[target_col] is not None])}")
    print(f"Flagged rows (up to 20): {len(flagged17)}")
    # Show first few flagged rows
    for f in flagged17[:3]:
        print(f"Row {f['row']}: actual={f['actual']}, expected={f['expected']}")

    # Machine‑checked answer
    answer_17 = {"flagged": flagged17}
    print("ANSWER APACHE-17", json.dumps(answer_17, separators=(',', ':')))

    # -------------------------------------------------------------------- #
    # APACHE-18: Flag unusual input combinations
    # -------------------------------------------------------------------- #
    print("\n== APACHE-18 ==")

    # Compute minimal Hamming distance to any other row
    max_possible = len(input_cols)
    min_distances = {}
    for i, r in enumerate(parsed_rows, start=1):
        min_d = max_possible + 1  # start with large
        for j, r2 in enumerate(parsed_rows, start=1):
            if i == j:
                continue
            d = hamming_distance(r, r2, input_cols)
            if d < min_d:
                min_d = d
                if min_d == 0:  # can't get lower
                    break
        min_distances[i] = min_d

    # Compute unusualness score: larger distance = more unusual
    scores = {row: round(dist / max_possible, 3) if max_possible > 0 else 0.0
              for row, dist in min_distances.items()}

    # Select top 20 rows with highest score
    sorted_rows = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    flagged18_rows = [row for row, score in sorted_rows[:20]]

    # Print summary
    print(f"Total rows: {n_rows}")
    print(f"Rows flagged for unusual input combination: {len(flagged18_rows)}")
    for row in flagged18_rows[:3]:
        print(f"Row {row} unusualness score: {scores[row]}")

    # Machine‑checked answer
    answer_18 = {
        "flagged": flagged18_rows,
        "scores": {str(k): v for k, v in scores.items() if k in flagged18_rows}
    }
    print("ANSWER APACHE-18", json.dumps(answer_18, separators=(',', ':')))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
