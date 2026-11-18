#!/usr/bin/env python3
"""
CAR-07: As new vehicle records arrive, assess each one immediately and tell me whether it belongs among our more desirable inventory options.
CAR-08: Keep our inventory summaries and recommended‑vehicle shortlist up to date as additional vehicle records are received, without requiring staff to restart the entire analysis manually.
"""

import csv
import json
import sys
from math import isfinite
from typing import Dict, List, Tuple, Optional, Iterable, Any

# ----------------------------------------------------------------------
# Utility parsing
# ----------------------------------------------------------------------
def parse_csv(path: str) -> List[Dict[str, Optional[float]]]:
    rows: List[Dict[str, Optional[float]]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed: Dict[str, Optional[float]] = {}
            for k, v in r.items():
                if v is None or v.strip() == "" or v.strip() == "?":
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None  # treat non‑numeric as missing
            rows.append(parsed)
    return rows


# ----------------------------------------------------------------------
# Goal column identification
# ----------------------------------------------------------------------
def goal_columns(headers: List[str]) -> List[str]:
    return [c for c in headers if c.endswith(("+", "-"))]


# ----------------------------------------------------------------------
# Min/Max handling
# ----------------------------------------------------------------------
def init_minmax(goals: List[str]) -> Dict[str, Dict[str, float]]:
    return {g: {"min": float("inf"), "max": float("-inf")} for g in goals}


def update_minmax(mm: Dict[str, Dict[str, float]], row: Dict[str, Optional[float]]) -> None:
    for g, m in mm.items():
        v = row.get(g)
        if v is not None and isfinite(v):
            if v < m["min"]:
                m["min"] = v
            if v > m["max"]:
                m["max"] = v


# ----------------------------------------------------------------------
# Normalisation and scoring
# ----------------------------------------------------------------------
def score_row(row: Dict[str, Optional[float]],
              mm: Dict[str, Dict[str, float]],
              goals: List[str]) -> Optional[float]:
    vals: List[float] = []
    for g in goals:
        v = row.get(g)
        if v is None or not isfinite(v):
            return None
        m = mm[g]
        if m["min"] == m["max"]:  # avoid div by zero
            norm = 0.0
        else:
            if g.endswith("+"):
                norm = (v - m["min"]) / (m["max"] - m["min"])
            else:  # g.endswith("-")
                norm = (m["max"] - v) / (m["max"] - m["min"])
        vals.append(norm)
    return sum(vals) / len(vals) if vals else None


# ----------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------
def compute_summary(seen: List[Tuple[int, Dict[str, Optional[float]]]],
                    mm: Dict[str, Dict[str, float]],
                    goals: List[str]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for g in goals:
        m = mm[g]
        if m["min"] == float("inf"):  # no data
            result[g] = {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
            continue
        # compute mean over seen rows (ignore missing)
        vals = [row[g] for _, row in seen if row.get(g) is not None and isfinite(row[g])]
        n = len(vals)
        mean_val = sum(vals) / n if n else 0.0
        result[g] = {
            "n": n,
            "mean": round(mean_val, 6),
            "min": round(m["min"], 6),
            "max": round(m["max"], 6),
        }
    return result


# ----------------------------------------------------------------------
# Best‑5 shortlist
# ----------------------------------------------------------------------
def best_five(seen: List[Tuple[int, Dict[str, Optional[float]]]],
              mm: Dict[str, Dict[str, float]],
              goals: List[str],
              k: int = 5) -> List[int]:
    scores: List[Tuple[float, int]] = []
    for idx, row in seen:
        sc = score_row(row, mm, goals)
        if sc is not None:
            scores.append((sc, idx))
    # sort descending score, then ascending row number
    scores.sort(key=lambda x: (-x[0], x[1]))
    return [idx for _, idx in scores[:k]]


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows = parse_csv(csv_path)
    if not rows:
        print("No data")
        return

    headers = list(rows[0].keys())
    goals = goal_columns(headers)

    total_rows = len(rows)
    arrival_count = int(round(0.2 * total_rows))
    if arrival_count > total_rows:
        arrival_count = total_rows

    history_rows = rows[:total_rows - arrival_count]
    arrivals_rows = rows[total_rows - arrival_count:]

    # Store seen rows as tuples (row_number, row_dict)
    seen: List[Tuple[int, Dict[str, Optional[float]]]] = []
    for i, r in enumerate(history_rows, start=1):
        seen.append((i, r))

    mm = init_minmax(goals)
    for _, r in seen:
        update_minmax(mm, r)

    # Prepare checkpoints
    checkpoints: List[Dict[str, Any]] = []

    # Checkpoint 0
    checkpoints.append({
        "after": 0,
        "summary": compute_summary(seen, mm, goals),
        "shortlist": best_five(seen, mm, goals),
    })

    assessments: List[Dict[str, Any]] = []

    # Process arrivals
    for step, r in enumerate(arrivals_rows, start=1):
        row_number = len(history_rows) + step  # 1‑based index

        # Score for current row using current mm (excluding this row)
        sc_current = score_row(r, mm, goals)

        # Determine desirability based on current best‑5
        current_best = best_five(seen, mm, goals, k=5)
        if sc_current is not None:
            if not current_best:
                desirable = True
            else:
                # lowest score in current best‑5
                lowest = score_row(
                    next((sr for rn, sr in seen if rn == current_best[-1]), mm, goals)
                if lowest is None:
                    desirable = True
                else:
                    desirable = sc_current > lowest
        else:
            desirable = False

        assessments.append({"row": row_number, "desirable": desirable})

        # Append row to seen and update minmax
        seen.append((row_number, r))
        update_minmax(mm, r)

        # Checkpoint at 40 and 80 arrivals
        if step in (40, 80):
            checkpoints.append({
                "after": step,
                "summary": compute_summary(seen, mm, goals),
                "shortlist": best_five(seen, mm, goals),
            })

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------
    # CAR-07
    print("== CAR-07 ==")
    for a in assessments:
        print(f"Arrival row {a['row']}: desirable={str(a['desirable']).lower()}")
    print(json.dumps({"assessments": assessments}, separators=(",", ":")))

    # CAR-08
    print("\n== CAR-08 ==")
    for cp in checkpoints:
        # format summary numbers to 6 decimals
        formatted_summary = {
            g: {
                "n": v["n"],
                "mean": round(v["mean"], 6),
                "min": round(v["min"], 6),
                "max": round(v["max"], 6),
            }
            for g, v in cp["summary"].items()
        }
        print(f"After {cp['after']} arrivals:")
        for g, stats in formatted_summary.items():
            print(f"  {g}: n={stats['n']}, mean={stats['mean']}, min={stats['min']}, max={stats['max']}")
        print(f"  Shortlist: {cp['shortlist']}")
    # Final JSON
    print(json.dumps(
        {
            "checkpoints": [
                {
                    "after": cp["after"],
                    "summary": {
                        g: {
                            "n": v["n"],
                            "mean": round(v["mean"], 6),
                            "min": round(v["min"], 6),
                            "max": round(v["max"], 6),
                        }
                        for g, v in cp["summary"].items()
                    },
                    "shortlist": cp["shortlist"],
                }
                for cp in checkpoints
            ]
        },
        separators=(",", ":")
    ))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
