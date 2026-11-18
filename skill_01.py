#!/usr/bin/env python3
import csv
import json
import sys
import math

def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            cleaned = {}
            for k, v in row.items():
                if v == '?':
                    cleaned[k] = None
                else:
                    # Try to convert to int/float
                    try:
                        if '.' in v:
                            cleaned[k] = float(v)
                        else:
                            cleaned[k] = int(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append((idx, cleaned))
    return rows, reader.fieldnames

def compute_stats(rows, goal_columns):
    stats = {}
    for col in goal_columns:
        vals = [v[col] for _, v in rows if v[col] is not None]
        if not vals:
            continue
        count = len(vals)
        mean = sum(vals) / count
        sorted_vals = sorted(vals)
        median = sorted_vals[count // 2] if count % 2 == 1 else (sorted_vals[count // 2 - 1] + sorted_vals[count // 2]) / 2
        min_val = min(vals)
        max_val = max(vals)
        # Determine best/worst based on goal direction
        if col.endswith('+'):
            best = max_val
            worst = min_val
        else:  # ends with '-'
            best = min_val
            worst = max_val
        stats[col] = {
            "mean": mean,
            "median": median,
            "min": min_val,
            "max": max_val,
            "best": best,
            "worst": worst
        }
    return stats

def get_top_rows(rows, goal_col, n_top):
    # For '-' goal, lower is better
    sorted_rows = sorted(rows, key=lambda r: r[1][goal_col] if r[1][goal_col] is not None else math.inf)
    top = sorted_rows[:n_top]
    return [r[0] for r in top]

def compute_setting_shares(rows, top_rows, setting_columns):
    shares = []
    top_set = set(top_rows)
    for col in setting_columns:
        top_vals = [r[col] for r, _ in rows if r in top_set and r[1][col] is not None]
        rest_vals = [r[col] for r, _ in rows if r not in top_set and r[1][col] is not None]
        if not top_vals or not rest_vals:
            continue
        top_share = sum(1 for v in top_vals if v == 1) / len(top_vals)
        rest_share = sum(1 for v in rest_vals if v == 1) / len(rest_vals)
        diff = abs(top_share - rest_share)
        shares.append({"setting": col, "top_share": top_share, "rest_share": rest_share})
    # Sort by difference descending
    shares.sort(key=lambda d: abs(d["top_share"] - d["rest_share"]), reverse=True)
    return shares

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)

    # Identify goal columns (ending with '+' or '-')
    goal_columns = [col for col in fieldnames if col.endswith(('+', '-'))]
    # Settings are all other columns except goal columns
    setting_columns = [col for col in fieldnames if col not in goal_columns and col not in ('X',)]

    # APACHE-01
    print("== APACHE-01 ==")
    stats = compute_stats(rows, goal_columns)
    for col, s in stats.items():
        print(f"{col}: mean {s['mean']:.2f}, median {s['median']:.2f}, min {s['min']:.2f}, max {s['max']:.2f}, best {s['best']:.2f}, worst {s['worst']:.2f}")
    # JSON answer
    json_ans = {}
    for col, s in stats.items():
        json_ans[col] = {k: round(v, 6) for k, v in s.items()}
    print(f"ANSWER APACHE-01 {json.dumps(json_ans)}")

    # APACHE-02
    print("\n== APACHE-02 ==")
    # Determine number of top rows: 10% of total, clamped 5-20
    n_top = max(5, min(20, max(1, int(0.10 * len(rows)))))
    top_rows = get_top_rows(rows, goal_columns[0], n_top)  # only one goal column
    print(f"Top {len(top_rows)} configurations: {top_rows}")
    # Compute setting shares
    setting_shares = compute_setting_shares(rows, top_rows, setting_columns)
    # Output at least two most distinguishing
    print("Top distinguishing settings:")
    for setting in setting_shares[:2]:
        print(f"{setting['setting']}: top_share={setting['top_share']:.3f}, rest_share={setting['rest_share']:.3f}")
    # JSON answer
    top_json = {"top": top_rows}
    settings_json = [{"setting": s["setting"],
                      "top_share": round(s["top_share"], 6),
                      "rest_share": round(s["rest_share"], 6)}
                     for s in setting_shares[:2]]
    top_json["settings"] = settings_json
    print(f"ANSWER APACHE-02 {json.dumps(top_json)}")

if __name__ == "__main__":
    main()
