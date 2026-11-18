#!/usr/bin/env python3
import csv
import json
import sys
from statistics import median
from typing import List, Any, Tuple

def read_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    """
    Read CSV, convert numeric cells, keep '?' as None.
    Return header list and list of rows.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            converted = []
            for v in row:
                if v == "?":
                    converted.append(None)
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            converted.append(float(v))
                        else:
                            converted.append(int(v))
                    except ValueError:
                        converted.append(v)
            rows.append(converted)
        return header, rows

def col_indices(header: List[str]) -> List[int]:
    """Return indices of goal columns (ending with + or -)."""
    return [i for i, name in enumerate(header) if name.endswith(('+', '-'))]

def compute_stats(rows: List[List[Any]], header: List[str], goal_idx: int) -> dict:
    """Compute mean, median, min, max, best, worst for a single column."""
    values = [r[goal_idx] for r in rows if r[goal_idx] is not None]
    if not values:
        # all missing: return NaNs
        return {"mean": None, "median": None, "min": None, "max": None, "best": None, "worst": None}
    mean = sum(values) / len(values)
    med = median(values)
    mn = min(values)
    mx = max(values)
    name = header[goal_idx]
    if name.endswith('+'):   # maximize
        best, worst = mx, mn
    else:                    # minimize
        best, worst = mn, mx
    return {"mean": mean, "median": med, "min": mn, "max": mx, "best": best, "worst": worst}

def compute_all_stats(rows: List[List[Any]], header: List[str]) -> dict:
    stats = {}
    for idx in col_indices(header):
        stats[header[idx]] = compute_stats(rows, header, idx)
    return stats

def compute_normalized_score(rows: List[List[Any]], header: List[str]) -> List[Tuple[int, float]]:
    """Return list of (row_number, score). Row numbers are 1-based."""
    goal_indices = col_indices(header)
    mins = {}
    maxs = {}
    for idx in goal_indices:
        vals = [r[idx] for r in rows if r[idx] is not None]
        if vals:
            mins[idx] = min(vals)
            maxs[idx] = max(vals)
        else:
            mins[idx] = maxs[idx] = None  # all missing
    scores = []
    for i, r in enumerate(rows, start=1):
        total = 0.0
        for idx in goal_indices:
            val = r[idx]
            if val is None:
                norm = 0.0
            else:
                mn = mins[idx]
                mx = maxs[idx]
                if mn is None or mx is None or mx == mn:
                    norm = 0.5
                else:
                    name = header[idx]
                    if name.endswith('+'):   # maximize
                        norm = (val - mn) / (mx - mn)
                    else:                    # minimize
                        norm = (mx - val) / (mx - mn)
            total += norm
        scores.append((i, total))
    return scores

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    header, rows = read_csv(path)

    # CAR-01
    stats = compute_all_stats(rows, header)
    print("== CAR-01 ==")
    for col in header:
        if col in stats:
            s = stats[col]
            print(f"{col}: mean={s['mean']}, median={s['median']}, min={s['min']}, max={s['max']}, best={s['best']}, worst={s['worst']}")
    print(f"ANSWER CAR-01 {json.dumps(stats)}")

    # CAR-02
    scores = compute_normalized_score(rows, header)
    # Sort descending by score, then by row number ascending for tie-breaking
    top = sorted(scores, key=lambda x: (-x[1], x[0]))[:10]
    top_indices = [r for r, _ in top]
    print("\n== CAR-02 ==")
    print("Top 10 rows by overall combination:", top_indices)
    print(f"ANSWER CAR-02 {json.dumps({'top': top_indices})}")

if __name__ == "__main__":
    main()
