#!/usr/bin/env python3
"""
Skill 09: NASA-17 and NASA-18

NASA-17 – Flag historical projects whose combinations of project characteristics
          are unusually different from the rest of our records.

NASA-18 – Identify project records whose reported effort or defect counts differ
          substantially from what comparable projects would lead us to expect.

The program prints a human‑readable summary and a machine‑checked JSON answer for
each request.  It is deterministic and uses only the standard library.
"""

import csv
import json
import math
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Utility – atom conversion
# ----------------------------------------------------------------------
def atom(s: str) -> Any:
    """Coerce a CSV cell to a Python value."""
    s = s.strip()
    if s == "?":
        return s
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return {"True": True, "False": False}.get(s, s)

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
def load_csv(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [dict(zip(header, [atom(v) for v in r])) for r in reader]
    return header, rows

# ----------------------------------------------------------------------
# Helper – identify column types
# ----------------------------------------------------------------------
def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")

def is_input(col: str) -> bool:
    return not (col.endswith("X") or is_goal(col))

# ----------------------------------------------------------------------
# Numeric statistics per column
# ----------------------------------------------------------------------
def numeric_stats(rows: List[Dict[str, Any]], cols: List[str]) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for col in cols:
        vals = [row[col] for row in rows if isinstance(row[col], (int, float))]
        if not vals:
            continue
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sd = math.sqrt(var)
        stats[col] = (mu, sd)
    return stats

# ----------------------------------------------------------------------
# Distance to centroid – used for NASA-17
# ----------------------------------------------------------------------
def compute_distances(rows: List[Dict[str, Any]], input_cols: List[str],
                      stats: Dict[str, Tuple[float, float]]) -> List[float]:
    distances = []
    for row in rows:
        sq = 0.0
        cnt = 0
        for col in input_cols:
            v = row[col]
            if isinstance(v, (int, float)):
                mu, sd = stats[col]
                if sd == 0:
                    d = 0.0
                else:
                    d = (v - mu) / sd
                sq += d * d
                cnt += 1
            elif isinstance(v, str) and v != "?":
                # categorical: 0 if same as mode, 1 otherwise
                # mode is the most common value in the column
                # we will compute it lazily below
                pass
            # ignore missing
        # Categorical differences will be added after mode calculation
        distances.append((sq, cnt, row))
    # Now compute categorical part
    # compute mode per categorical column
    cat_modes: Dict[str, Any] = {}
    for col in input_cols:
        vals = [row[col] for row in rows if isinstance(row[col], str) and row[col] != "?"]
        if vals:
            cat_modes[col] = Counter(vals).most_common(1)[0][0]
    final_dist = []
    for sq, cnt, row in distances:
        # add categorical differences
        for col, mode in cat_modes.items():
            v = row[col]
            if isinstance(v, str) and v != "?":
                if v != mode:
                    sq += 1.0
                cnt += 1
        dist = math.sqrt(sq) if cnt > 0 else 0.0
        final_dist.append(dist)
    return final_dist

# ----------------------------------------------------------------------
# KNN prediction – used for NASA-18
# ----------------------------------------------------------------------
def knn_predict(target_row: Dict[str, Any], all_rows: List[Dict[str, Any]],
                 input_cols: List[str], target: str, k: int = 3) -> float:
    def dist(r1: Dict[str, Any], r2: Dict[str, Any]) -> float:
        sq = 0.0
        cnt = 0
        for col in input_cols:
            v1 = r1[col]
            v2 = r2[col]
            if v1 == "?" or v2 == "?" or (isinstance(v1, str) and isinstance(v2, str) and v1 == v2):
                continue
            if isinstance(v1, str) or isinstance(v2, str):
                sq += 1.0
                cnt += 1
            else:
                try:
                    d = float(v1) - float(v2)
                    sq += d * d
                    cnt += 1
                except Exception:
                    sq += 1.0
                    cnt += 1
        return math.sqrt(sq) if cnt > 0 else float("inf")

    dists = []
    for r in all_rows:
        if r is target_row:
            continue
        d = dist(target_row, r)
        dists.append((d, r))
    dists.sort(key=lambda x: x[0])
    neigh_vals = []
    for d, r in dists[:k]:
        val = r[target]
        if val != "?":
            neigh_vals.append(val)
    if not neigh_vals:
        return float("nan")
    return sum(neigh_vals) / len(neigh_vals)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = load_csv(csv_path)
    nrows = len(rows)

    # Identify columns
    input_cols = [c for c in header if is_input(c)]
    goal_cols = [c for c in header if is_goal(c)]

    # ------------------------------------------------------------
    # NASA-17
    # ------------------------------------------------------------
    # Compute stats for numeric columns
    numeric_cols = [c for c in input_cols if all(isinstance(r[c], (int, float)) for r in rows if r[c] != "?")]
    num_stats = numeric_stats(rows, numeric_cols)

    # Distances to centroid
    distances = compute_distances(rows, input_cols, num_stats)
    mean_dist = sum(distances) / len(distances)
    var_dist = sum((d - mean_dist) ** 2 for d in distances) / len(distances)
    sd_dist = math.sqrt(var_dist) if var_dist > 0 else 1.0

    # Score each row
    scores = {}
    for idx, d in enumerate(distances, start=1):
        scores[idx] = abs(d - mean_dist) / sd_dist

    # Flag top 10 rows
    flagged17 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
    flagged_nums17 = [row for row, _ in flagged17]

    # Human‑readable section
    print("== NASA-17 ==")
    print(f"Total rows: {nrows}")
    print(f"Flagged rows (up to 10): {flagged_nums17}")
    print(f"Scores of flagged rows: {{")
    for row, score in flagged17:
        print(f"  {row}: {score:.4f}")
    print("}")

    answer17 = {
        "flagged": flagged_nums17,
        "scores": {str(row): round(score, 6) for row, score in flagged17}
    }
    print(f"ANSWER NASA-17 {json.dumps(answer17)}")

    # ------------------------------------------------------------
    # NASA-18
    # ------------------------------------------------------------
    # Predict for each row
    errors_effort = []
    errors_defects = []
    predictions = []

    for idx, row in enumerate(rows, start=1):
        eff_actual = row.get("Effort-")
        def_actual = row.get("Defects-")
        # Skip rows missing at least one target
        if eff_actual == "?" or def_actual == "?" or eff_actual == "" or def_actual == "":
            continue

        eff_pred = knn_predict(row, rows, input_cols, "Effort-")
        def_pred = knn_predict(row, rows, input_cols, "Defects-")
        if math.isnan(eff_pred) or math.isnan(def_pred):
            continue

        err_eff = abs(eff_pred - eff_actual)
        err_def = abs(def_pred - def_actual)
        errors_effort.append((idx, err_eff))
        errors_defects.append((idx, err_def))
        predictions.append((idx, eff_pred, def_pred))

    def error_stats(errors: List[Tuple[int, float]]) -> Tuple[float, float]:
        vals = [e for _, e in errors]
        if not vals:
            return 0.0, 1.0
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sd = math.sqrt(var) if var > 0 else 1.0
        return mu, sd

    mu_eff, sd_eff = error_stats(errors_effort)
    mu_def, sd_def = error_stats(errors_defects)

    flagged18 = []
    for idx, eff_actual in errors_effort:
        err = abs(knn_predict(rows[idx-1], rows, input_cols, "Effort-") - eff_actual)
        score = abs(err - mu_eff) / sd_eff
        flagged18.append((idx, "Effort-", eff_actual,
                         knn_predict(rows[idx-1], rows, input_cols, "Effort-"), score))
    for idx, def_actual in errors_defects:
        err = abs(knn_predict(rows[idx-1], rows, input_cols, "Defects-") - def_actual)
        score = abs(err - mu_def) / sd_def
        flagged18.append((idx, "Defects-", def_actual,
                         knn_predict(rows[idx-1], rows, input_cols, "Defects-"), score))

    # Sort by score descending, keep top 10
    flagged18.sort(key=lambda kv: kv[4], reverse=True)
    flagged18 = flagged18[:10]

    # Human‑readable section
    print("\n== NASA-18 ==")
    print(f"Total rows: {nrows}")
    print(f"Flagged rows (up to 10):")
    for row, col, actual, expected, score in flagged18:
        print(f"  Row {row}: {col} actual={actual} expected={expected:.4f} score={score:.4f}")

    answer18 = {
        "flagged": [
            {"row": row,
             "column": col,
             "actual": actual,
             "expected": round(expected, 6)}
            for row, col, actual, expected, _ in flagged18
        ]
    }
    print(f"ANSWER NASA-18 {json.dumps(answer18)}")

if __name__ == "__main__":
    main()
