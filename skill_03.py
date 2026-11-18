#!/usr/bin/env python3
import csv
import json
import sys
import math

# ----------------- Data utilities -----------------

def read_data(path):
    """
    Read CSV and return list of dict rows.
    '?' or '' -> None, numeric -> float, else keep string.
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, val in row.items():
                if val == '?' or val == '':
                    parsed[key] = None
                else:
                    try:
                        parsed[key] = float(val)
                    except ValueError:
                        parsed[key] = val
            rows.append(parsed)
    return rows

# ----------------- Stats helpers -----------------

def mean(values):
    return sum(values) / len(values)

def median(values):
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def std(values):
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))

# ----------------- CAR-05 -----------------

def car_05(rows):
    goal = "Mpg+"
    groups_raw = {}
    for r in rows:
        c = r.get("Clndrs")
        v = r.get(goal)
        if c is None or v is None:
            continue
        groups_raw.setdefault(c, []).append(v)

    # Build summary dict
    groups = {}
    for c, vals in groups_raw.items():
        groups[str(c)] = {
            "n": len(vals),
            "median": round(median(vals), 2),
            "mean": round(mean(vals), 2)
        }

    # Pairwise comparisons
    keys = sorted(groups.keys(), key=lambda x: float(x))
    comparisons = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            vals_a = groups_raw[float(a)]
            vals_b = groups_raw[float(b)]
            mean_a = mean(vals_a)
            mean_b = mean(vals_b)
            diff = mean_a - mean_b
            s_a = std(vals_a)
            s_b = std(vals_b)
            denom = math.sqrt(s_a ** 2 / len(vals_a) + s_b ** 2 / len(vals_b))
            if denom == 0:
                verdict = "same"
            else:
                t = abs(diff) / denom
                verdict = "different" if t > 1.96 else "same"
            comparisons.append({"a": float(a), "b": float(b), "verdict": verdict})

    return {"groups": groups, "comparisons": comparisons}

# ----------------- CAR-06 -----------------

def car_06(rows):
    goal_cols = [k for k in rows[0].keys() if k.endswith('-') or k.endswith('+')]

    # Compute min/max per goal across all rows
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [r[col] for r in rows if r[col] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0

    # Define performance score per row
    def perf_score(r):
        score = 0.0
        for col in goal_cols:
            v = r[col]
            if v is None:
                return None
            mn = mins[col]
            mx = maxs[col]
            if mx == mn:
                norm = 0.0
            else:
                if col.endswith('-'):  # minimize
                    norm = (mx - v) / (mx - mn)
                else:
                    norm = (v - mn) / (mx - mn)
            score += norm
        return score

    # Determine two groups by Model year
    model_vals = sorted(set(r["Model"] for r in rows if r["Model"] is not None))
    if not model_vals:
        return {"groups": [], "verdict": "same"}
    mid = len(model_vals) // 2
    group1_models = set(model_vals[:mid + 1])
    group2_models = set(model_vals[mid + 1:])

    groups = [{}, {}]  # 0: group1, 1: group2
    perf_data = [[], []]  # performance scores per group

    for r in rows:
        m = r.get("Model")
        if m is None:
            continue
        if m in group1_models:
            grp = 0
        elif m in group2_models:
            grp = 1
        else:
            continue  # shouldn't happen
        groups[grp].setdefault("n", 0)
        groups[grp]["n"] += 1
        s = perf_score(r)
        if s is not None:
            perf_data[grp].append(s)

    # Compute mean performance and std
    means = []
    stds = []
    for p in perf_data:
        if p:
            means.append(mean(p))
            stds.append(std(p))
        else:
            means.append(0.0)
            stds.append(0.0)

    # T-test
    if len(perf_data[0]) == 0 or len(perf_data[1]) == 0:
        verdict = "same"
    else:
        diff = means[0] - means[1]
        denom = math.sqrt(stds[0] ** 2 / len(perf_data[0]) + stds[1] ** 2 / len(perf_data[1]))
        if denom == 0:
            verdict = "same"
        else:
            t = abs(diff) / denom
            verdict = "different" if t > 1.96 else "same"

    # Build group descriptions
    group_descs = []
    for grp, m_set in [(0, group1_models), (1, group2_models)]:
        if not m_set:
            continue
        mn = min(m_set)
        mx = max(m_set)
        group_descs.append({
            "from": mn,
            "to": mx,
            "n": groups[grp]["n"]
        })

    return {"groups": group_descs, "verdict": verdict}

# ----------------- Main -----------------

def main(csv_path):
    rows = read_data(csv_path)

    # CAR-05
    print("== CAR-05 ==")
    res05 = car_05(rows)
    # Some simple printed lines
    print(f"Number of cylinder groups: {len(res05['groups'])}")
    print(f"Number of comparisons: {len(res05['comparisons'])}")
    print("Example group stats:")
    for k, v in list(res05['groups'].items())[:3]:
        print(f"  {k}: n={v['n']}, mean={v['mean']}, median={v['median']}")
    answer_json05 = json.dumps(res05, ensure_ascii=False)
    print(f"ANSWER CAR-05 {answer_json05}")

    # CAR-06
    print("\n== CAR-06 ==")
    res06 = car_06(rows)
    print(f"Groups: {res06['groups']}")
    print(f"Verdict: {res06['verdict']}")
    answer_json06 = json.dumps(res06, ensure_ascii=False)
    print(f"ANSWER CAR-06 {answer_json06}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
