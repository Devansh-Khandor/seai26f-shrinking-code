#!/usr/bin/env python3
"""
Skill 05: Advanced analysis of poorly performing configurations.

APACHE-09 – Identify settings most strongly associated with poor performance.
APACHE-10 – Find combinations of settings that are common among poor
            configurations and how they differ from good ones.
"""

import json
import sys
import itertools
import math
from typing import List, Dict, Any

import ezr

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def goal_col(tbl: ezr.Tbl) -> ezr.COL:
    """Return the goal column (ends with '-' or '+')."""
    for col in tbl.cols.values():
        if col.txt.endswith("-") or col.txt.endswith("+"):
            return col
    raise RuntimeError("No goal column found.")


def compute_stats(rows: List[ezr.ROW], col: ezr.COL) -> Dict[str, Any]:
    """Compute mean and std for a column over given rows (ignore missing)."""
    vals = [row[col.at] for row in rows if row[col.at] != "?"]
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {"mean": mean, "std": std}


def poor_and_good_rows(rows: List[ezr.ROW], col: ezr.COL) -> tuple:
    """Return two lists: poor rows (worst) and good rows (best)."""
    stats = compute_stats(rows, col)
    threshold_poor = stats["mean"] + 2 * stats["std"]
    # Poor rows: performance worse than threshold
    poor = [row for row in rows if row[col.at] != "?" and row[col.at] > threshold_poor]
    if len(poor) < 5:
        # fallback: take the 5 worst rows
        sorted_by_perf = sorted(
            [row for row in rows if row[col.at] != "?"],
            key=lambda r: r[col.at],
            reverse=True,
        )
        poor = sorted_by_perf[:5]

    # Good rows: the 5 best rows
    good = sorted(
        [row for row in rows if row[col.at] != "?"],
        key=lambda r: r[col.at],
    )[:5]

    return poor, good


def setting_differences(poor_rows, good_rows, cols: List[ezr.COL]) -> List[str]:
    """Return setting names sorted by absolute difference in proportion of 1."""
    diffs = []
    for col in cols:
        # skip goal or X columns
        if col.txt.endswith(("X", "+", "-")):
            continue
        # compute proportion of 1 among bad and good
        bad_vals = [row[col.at] for row in poor_rows if row[col.at] != "?"]
        good_vals = [row[col.at] for row in good_rows if row[col.at] != "?"]
        bad_p = sum(1 for v in bad_vals if v == 1 or v == "1") / max(1, len(bad_vals))
        good_p = sum(1 for v in good_vals if v == 1 or v == "1") / max(1, len(good_vals))
        diffs.append((col.txt, abs(bad_p - good_p)))
    # sort by diff descending
    diffs.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in diffs[:3]]  # at least 3


def combination_shares(poor_rows, good_rows, cols: List[ezr.COL]) -> List[Dict[str, Any]]:
    """Return list of combos (at least 2 settings) with bad/good shares."""
    combos = {}
    # consider all unordered pairs of setting columns
    setting_cols = [c for c in cols if c.txt.endswith(("0", "1", "X", "Y", "Z", "")) and not c.txt.endswith(("X", "+", "-"))]
    # Actually we need all non-goal columns
    setting_cols = [c for c in cols if not c.txt.endswith(("X", "+", "-"))]
    for c1, c2 in itertools.combinations(setting_cols, 2):
        for row_set, key, label in (
            (poor_rows, "bad", "bad_count"),
            (good_rows, "good", "good_count"),
        ):
            for row in row_set:
                v1, v2 = row[c1.at], row[c2.at]
                if v1 == "?" or v2 == "?" or v1 not in (0, 1, "0", "1") or v2 not in (0, 1, "0", "1"):
                    continue
                # normalize to int
                iv1 = int(v1)
                iv2 = int(v2)
                combo_key = (c1.txt, iv1, c2.txt, iv2)
                if combo_key not in combos:
                    combos[combo_key] = {"bad_count": 0, "good_count": 0}
                combos[combo_key][label] += 1

    # Build result list
    res = []
    total_bad = len(poor_rows)
    total_good = len(good_rows)
    for key, counts in combos.items():
        bad_share = counts["bad_count"] / max(1, total_bad)
        good_share = counts["good_count"] / max(1, total_good)
        if counts["bad_count"] == 0:
            continue
        settings = {key[0]: key[1], key[2]: key[3]}
        res.append(
            {
                "settings": settings,
                "bad_share": round(bad_share, 3),
                "good_share": round(good_share, 3),
            }
        )
    # Sort by bad_share descending
    res.sort(key=lambda x: x["bad_share"], reverse=True)
    return res[:5]  # at least 1, show up to 5


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows: List[ezr.ROW] = list(ezr.csv(csv_path))
    tbl: ezr.Tbl = ezr.Tbl(rows)

    gcol = goal_col(tbl)

    # ------------------------------------------------------------------
    # APACHE-09
    # ------------------------------------------------------------------
    poor_rows, good_rows = poor_and_good_rows(rows, gcol)
    poor_indices = [rows.index(r) + 1 for r in poor_rows]
    ranking = setting_differences(poor_rows, good_rows, tbl.cols.values())

    print("== APACHE-09 ==")
    # at least three lines
    print("Poor rows:", poor_indices)
    print("Ranking of settings most associated with poor performance:")
    for name in ranking:
        print(f"  {name}")
    # JSON answer
    answer_09 = {"poor": poor_indices, "ranking": ranking}
    print("ANSWER APACHE-09", json.dumps(answer_09, separators=(",", ":")))

    # ------------------------------------------------------------------
    # APACHE-10
    # ------------------------------------------------------------------
    combos = combination_shares(poor_rows, good_rows, tbl.cols.values())
    bad_indices = poor_indices
    good_indices = [rows.index(r) + 1 for r in good_rows]

    print("\n== APACHE-10 ==")
    print("Bad rows:", bad_indices)
    print("Good rows:", good_indices)
    print("Top combinations:")
    for combo in combos:
        settings_str = " ".join(f"{k}={v}" for k, v in combo["settings"].items())
        print(
            f"  {settings_str}: bad_share={combo['bad_share']:.3f} good_share={combo['good_share']:.3f}"
        )
    # JSON answer
    answer_10 = {
        "bad": bad_indices,
        "good": good_indices,
        "combinations": combos,
    }
    print("ANSWER APACHE-10", json.dumps(answer_10, separators=(",", ":")))


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
