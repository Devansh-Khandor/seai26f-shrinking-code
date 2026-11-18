#!/usr/bin/env python3
import sys
import json
import ezr


def compute_goal_stats(rows, goal_col):
    """Return mean, median, min, max, best, worst for the goal column."""
    goal_vals = [row[goal_col.at] for row in rows if row[goal_col.at] != "?"]
    n = len(goal_vals)
    if n == 0:
        raise ValueError("No goal values found.")

    mean = sum(goal_vals) / n
    sorted_vals = sorted(goal_vals)
    if n % 2:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

    min_val = min(goal_vals)
    max_val = max(goal_vals)
    # The goal ends with '-' meaning we want to minimize it.
    best = min_val
    worst = max_val
    return mean, median, min_val, max_val, best, worst


def top_rows(rows, goal_col, n_top):
    """Return top n_top row numbers with the best (smallest) goal values."""
    # Sort by goal value, treating missing values as worst.
    sorted_rows = sorted(
        enumerate(rows, start=1),
        key=lambda x: x[1][goal_col.at] if x[1][goal_col.at] != "?" else float("inf"),
    )
    return sorted_rows[:n_top]


def setting_shares(rows, goal_col, top_set, n_top):
    """Return distinguishing stats for each setting column."""
    setting_cols = [
        col
        for col in goal_col.tbl.cols.values()
        if col.txt != goal_col.txt and col.txt[-1] != "X"
    ]
    stats = []
    for col in setting_cols:
        top_count = sum(
            1 for idx, r in enumerate(rows, start=1)
            if idx in top_set and r[col.at] == 1
        )
        rest_count = sum(
            1 for idx, r in enumerate(rows, start=1)
            if idx not in top_set and r[col.at] == 1
        )
        top_share = top_count / n_top
        rest_share = (
            rest_count / (len(rows) - n_top) if len(rows) - n_top else 0.0
        )
        diff = abs(top_share - rest_share)
        stats.append(
            {
                "setting": col.txt,
                "top_share": top_share,
                "rest_share": rest_share,
                "diff": diff,
            }
        )
    # Sort by how distinguishing the setting is.
    stats.sort(key=lambda s: s["diff"], reverse=True)
    return stats


def main(csv_path):
    rows = list(ezr.csv(csv_path))
    tbl = ezr.Tbl(rows)
    # Identify the goal column (ends with '-').
    goal_col = None
    for col in tbl.cols.values():
        if col.txt.endswith("-"):
            goal_col = col
            break
    if goal_col is None:
        raise RuntimeError("No goal column found in dataset.")
    goal_col.tbl = tbl  # store reference for later use

    # --- APACHE-01 ---
    mean, median, min_val, max_val, best, worst = compute_goal_stats(rows, goal_col)

    print("== APACHE-01 ==")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Min: {min_val:.2f}")
    print(f"Max: {max_val:.2f}")
    print(f"Best: {best:.2f}")
    print(f"Worst: {worst:.2f}")

    ans_01 = {
        "Performance-": {
            "mean": mean,
            "median": median,
            "min": min_val,
            "max": max_val,
            "best": best,
            "worst": worst,
        }
    }
    print("ANSWER APACHE-01", json.dumps(ans_01, separators=(",", ":")))

    # --- APACHE-02 ---
    # Determine how many top rows to consider: 5 to 20 rows, roughly 10% of data.
    n_top = max(5, min(20, len(rows) // 10))
    top_sorted = top_rows(rows, goal_col, n_top)
    top_indices = [idx for idx, _ in top_sorted]
    top_set = set(top_indices)

    stats = setting_shares(rows, goal_col, top_set, n_top)
    # Keep at least two settings, most distinguishing first.
    top_settings = stats[:max(2, len(stats))]

    # Prepare output list without the 'diff' field.
    output_settings = [
        {"setting": s["setting"], "top_share": s["top_share"], "rest_share": s["rest_share"]}
        for s in top_settings
    ]

    print("== APACHE-02 ==")
    print(f"Top {n_top} rows: {top_indices}")
    for s in output_settings:
        print(
            f"{s['setting']}: top {s['top_share']:.3f}, rest {s['rest_share']:.3f}"
        )

    ans_02 = {"top": top_indices, "settings": output_settings}
    print("ANSWER APACHE-02", json.dumps(ans_02, separators=(",", ":")))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
