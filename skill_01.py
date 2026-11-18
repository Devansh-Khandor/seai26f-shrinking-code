#!/usr/bin/env python3
import sys
import csv
import json

def parse_value(v):
    v = v.strip()
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def median(lst):
    n = len(lst)
    if n == 0:
        return None
    s = sorted(lst)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    else:
        return (s[mid - 1] + s[mid]) / 2.0

def main(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        idx_kloc = header.index('Kloc+')
        idx_effort = header.index('Effort-')
        idx_defects = header.index('Defects-')

        # Data containers
        kloc_vals = []
        effort_vals = []
        defects_vals = []
        row_nums = []          # row numbers (1-based)
        row_kloc = {}
        row_effort = {}
        row_defects = {}

        # For top list
        top_rows = []

        for rnum, row in enumerate(reader, start=1):
            row_nums.append(rnum)

            k = parse_value(row[idx_kloc])
            e = parse_value(row[idx_effort])
            d = parse_value(row[idx_defects])

            if k is not None:
                kloc_vals.append(k)
                row_kloc[rnum] = k
            if e is not None:
                effort_vals.append(e)
                row_effort[rnum] = e
            if d is not None:
                defects_vals.append(d)
                row_defects[rnum] = d

            # Only consider rows with all three values for top list
            if k is not None and e is not None and d is not None:
                top_rows.append(rnum)

    # Compute stats per column
    def compute_stats(vals, name, maximize=True):
        if not vals:
            return None
        mean = sum(vals) / len(vals)
        med = median(vals)
        mn = min(vals)
        mx = max(vals)
        best = mx if maximize else mn
        worst = mn if maximize else mx
        return {
            "mean": mean,
            "median": med,
            "min": mn,
            "max": mx,
            "best": best,
            "worst": worst
        }

    stats_kloc = compute_stats(kloc_vals, 'Kloc+', maximize=True)
    stats_effort = compute_stats(effort_vals, 'Effort-', maximize=False)
    stats_defects = compute_stats(defects_vals, 'Defects-', maximize=False)

    # Print NASA-01 section
    print("== NASA-01 ==")
    # One line per column
    print(f"Kloc+: mean={stats_kloc['mean']:.3f} median={stats_kloc['median']:.3f} min={stats_kloc['min']:.3f} max={stats_kloc['max']:.3f} best={stats_kloc['best']:.3f} worst={stats_kloc['worst']:.3f}")
    print(f"Effort-: mean={stats_effort['mean']:.3f} median={stats_effort['median']:.3f} min={stats_effort['min']:.3f} max={stats_effort['max']:.3f} best={stats_effort['best']:.3f} worst={stats_effort['worst']:.3f}")
    print(f"Defects-: mean={stats_defects['mean']:.3f} median={stats_defects['median']:.3f} min={stats_defects['min']:.3f} max={stats_defects['max']:.3f} best={stats_defects['best']:.3f} worst={stats_defects['worst']:.3f}")
    # JSON answer
    nasa01_json = {
        "Kloc+": stats_kloc,
        "Effort-": stats_effort,
        "Defects-": stats_defects
    }
    print("ANSWER NASA-01 " + json.dumps(nasa01_json, separators=(',', ':')))

    # Compute composite score for top list
    k_min = stats_kloc['min']
    k_max = stats_kloc['max']
    e_min = stats_effort['min']
    e_max = stats_effort['max']
    d_min = stats_defects['min']
    d_max = stats_defects['max']

    scored_rows = []
    for r in top_rows:
        k = row_kloc[r]
        e = row_effort[r]
        d = row_defects[r]
        # Normalize
        if k_max - k_min == 0:
            nk = 0.5
        else:
            nk = (k - k_min) / (k_max - k_min)
        if e_max - e_min == 0:
            ne = 0.5
        else:
            ne = (e_max - e) / (e_max - e_min)
        if d_max - d_min == 0:
            nd = 0.5
        else:
            nd = (d_max - d) / (d_max - d_min)
        score = nk + ne + nd
        scored_rows.append((score, r))

    scored_rows.sort(reverse=True, key=lambda x: x[0])
    top10 = [r for _, r in scored_rows[:10]]

    # Print NASA-02 section
    print("\n== NASA-02 ==")
    print("Top projects (row numbers):")
    for r in top10:
        print(r)
    nasa02_json = {"top": top10}
    print("ANSWER NASA-02 " + json.dumps(nasa02_json, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
