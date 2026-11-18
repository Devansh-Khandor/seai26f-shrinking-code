#!/usr/bin/env python3
"""
Skill 06: Apache configuration recommendation explanation and comparison

This program implements two analyses for the Apache configuration data:

* **APACHE-11** – When the system recommends a configuration, explain its
  recommendation by comparing the configuration's settings and performance
  with relevant previously tested configurations.

* **APACHE-12** – When two configurations receive different performance
  assessments, explain which settings differ between them and how those
  differences relate to the predicted or recorded outcomes.

The program prints a header line for each analysis, a few human‑readable
lines, and ends each section with a machine‑checked answer line
containing a single‑line JSON object as required.
"""

import csv
import json
import math
import sys
from typing import Any, Dict, List, Tuple

# ------------------------------------------------------------
# Utility – reading CSV into list of dicts (values are parsed)
# ------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Return header and list of rows as dicts; '?' -> None."""
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows: List[Dict[str, Any]] = []
        for row in rdr:
            d: Dict[str, Any] = {}
            for h, val in zip(header, row):
                if val == "?":
                    d[h] = None
                else:
                    try:
                        if "." in val or "e" in val.lower():
                            d[h] = float(val)
                        else:
                            d[h] = int(val)
                    except ValueError:
                        d[h] = val
            rows.append(d)
    return header, rows

# ------------------------------------------------------------
# Helpers – column classification
# ------------------------------------------------------------
def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")

def is_input(col: str) -> bool:
    return not col.endswith("X") and not is_goal(col)

# ------------------------------------------------------------
# Column statistics (mean, sd or entropy) needed for normalization
# ------------------------------------------------------------
class NumStat:
    def __init__(self):
        self.n = 0
        self.mu = 0.0
        self.m2 = 0.0

    def add(self, v: float) -> None:
        self.n += 1
        d = v - self.mu
        self.mu += d / self.n
        self.m2 += d * (v - self.mu)

    @property
    def sd(self) -> float:
        return 0.0 if self.n < 2 else math.sqrt(self.m2 / (self.n - 1))

class SymStat:
    def __init__(self):
        self.counts: Dict[Any, int] = {}

    def add(self, v: Any) -> None:
        self.counts[v] = self.counts.get(v, 0) + 1

    def mode(self) -> Any:
        return max(self.counts.items(), key=lambda kv: kv[1])[0] if self.counts else None

def compute_stats(header: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for col in header:
        if col.endswith("X"):  # skip
            continue
        if is_goal(col):
            stats[col] = NumStat()
        elif is_input(col):
            stats[col] = SymStat()
        else:  # treat as numeric
            stats[col] = NumStat()
        for r in rows:
            v = r[col]
            if v is None:
                continue
            stats[col].add(v)
    return stats

# ------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------
def norm(col: str, value: Any, stats: Dict[str, Any]) -> float:
    """Return normalized value in [0,1] for numeric columns."""
    s = stats[col]
    if isinstance(s, SymStat):
        return value
    if s.n == 0:
        return 0.0
    z = (value - s.mu) / (1e-32 + s.sd)
    # logistic scaling to 0..1
    z = max(-3, min(3, z))
    return 1 / (1 + math.exp(-1.7 * z))

# ------------------------------------------------------------
# Distance functions
# ------------------------------------------------------------
def gap(col: str, a: Any, b: Any, stats: Dict[str, Any]) -> float:
    if a == "?":
        a = None
    if b == "?":
        b = None
    if a is None or b is None:
        return 1.0
    s = stats[col]
    if isinstance(s, SymStat):
        return 0.0 if a == b else 1.0
    # numeric
    return abs(norm(col, a, stats) - norm(col, b, stats))

def xdist(row1: Dict[str, Any], row2: Dict[str, Any],
          input_cols: List[str], stats: Dict[str, Any]) -> float:
    n = len(input_cols)
    if n == 0:
        return 0.0
    s = 0.0
    for col in input_cols:
        v = gap(col, row1[col], row2[col], stats)
        s += v * v
    return math.sqrt(s / n)

def ydist(row: Dict[str, Any], goal_cols: List[str], stats: Dict[str, Any]) -> float:
    n = len(goal_cols)
    if n == 0:
        return 0.0
    s = 0.0
    for col in goal_cols:
        v = row[col]
        if v is None:
            return math.inf
        s += gap(col, v, v, stats) * gap(col, v, v, stats)  # gap of value with itself
    # For numeric goals, gap of value with itself is zero; for symmetry
    # we use direct distance: |norm - goal_goal| but goal_goal is same => 0
    # Instead compute deviation from goal optimal:
    # Here we just compute difference from goal optimum: we approximate by norm difference to optimal
    # optimal is min for '-' and max for '+'. For simplicity use abs(norm - 0) for '-' and abs(norm - 1) for '+'
    s = 0.0
    for col in goal_cols:
        v = row[col]
        if v is None:
            return math.inf
        s += (abs(norm(col, v, stats) - (0 if col.endswith("-") else 1))) ** 2
    return math.sqrt(s / n)

# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
def main(csv_path: str) -> None:
    header, rows = read_csv(csv_path)
    goal_cols = [c for c in header if is_goal(c)]
    input_cols = [c for c in header if is_input(c)]
    stats = compute_stats(header, rows)

    # ------------------------------------------------------------
    # APACHE-11 – recommendation explanation
    # ------------------------------------------------------------
    print("== APACHE-11 ==")
    # Find best row (min ydist)
    best_row = None
    best_dist = math.inf
    for idx, r in enumerate(rows, start=1):
        d = ydist(r, goal_cols, stats)
        if d < best_dist:
            best_dist = d
            best_row = (idx, r)
    if best_row is None:
        print("No valid rows for recommendation.")
        sys.exit(1)

    best_idx, best_dict = best_row
    # Find 3 nearest neighbors (excluding itself)
    neighbors = []
    for idx, r in enumerate(rows, start=1):
        if idx == best_idx:
            continue
        d = xdist(best_dict, r, input_cols, stats)
        neighbors.append((d, idx, r))
    neighbors.sort(key=lambda kv: kv[0])
    top_neighbors = neighbors[:3]
    print(f"Recommended row: {best_idx}  ydist={best_dist:.3f}")
    print(f"Closest neighbors:")
    for d, idx, r in top_neighbors:
        print(f"  row {idx} dist={d:.3f}")
    # Machine‑checked answer
    def serialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
        return {k: row[k] for k in header}

    answer_11 = {
        "recommended": best_idx,
        "neighbors": [
            {"row": nidx, "values": serialize_row(rdict)}
            for _, nidx, rdict in top_neighbors
        ],
    }
    print(f"ANSWER APACHE-11 {json.dumps(answer_11)}")

    # ------------------------------------------------------------
    # APACHE-12 – comparison explanation
    # ------------------------------------------------------------
    print("\n== APACHE-12 ==")
    # Find best and worst rows by ydist
    best = None
    best_val = math.inf
    worst = None
    worst_val = -math.inf
    for idx, r in enumerate(rows, start=1):
        d = ydist(r, goal_cols, stats)
        if d < best_val:
            best_val, best = d, (idx, r)
        if d > worst_val:
            worst_val, worst = d, (idx, r)
    if not best or not worst:
        print("Insufficient rows for comparison.")
        sys.exit(1)

    best_idx, best_dict = best
    worst_idx, worst_dict = worst

    preferred = best_idx if best_val < worst_val else worst_idx
    scores = {str(best_idx): best_val, str(worst_idx): worst_val}
    print(f"Row {best_idx} (ydist={best_val:.3f}) vs Row {worst_idx} (ydist={worst_val:.3f})")
    print(f"Preferred row: {preferred}")

    # Differences
    differences = []
    for col in header:
        a = best_dict[col]
        b = worst_dict[col]
        if a != b:
            differences.append({"feature": col, "a": a, "b": b})
    print("Differences:")
    for d in differences[:5]:
        print(f"  {d['feature']}: {d['a']} vs {d['b']}")
    # Tradeoffs: goals where non-preferred is better
    non_preferred_idx = worst_idx if preferred == best_idx else best_idx
    non_preferred = worst_dict if preferred == best_idx else best_dict
    tradeoffs = []
    for col in goal_cols:
        a = best_dict[col]
        b = worst_dict[col]
        if a is None or b is None:
            continue
        better = False
        if col.endswith("-"):  # minimize
            better = a < b
        else:  # maximize
            better = a > b
        if better and non_preferred_idx != preferred:
            tradeoffs.append(col)
    print("Tradeoffs (non-preferred better in):", tradeoffs)

    answer_12 = {
        "a": best_idx,
        "b": worst_idx,
        "preferred": preferred,
        "scores": scores,
        "differences": differences,
        "tradeoffs": tradeoffs,
    }
    print(f"ANSWER APACHE-12 {json.dumps(answer_12)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
