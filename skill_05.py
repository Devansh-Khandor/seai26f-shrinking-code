#!/usr/bin/env python3
"""
Skill 05: Identify characteristics associated with unusually high effort and defects.

NASA-09: For projects that required unusually high development effort, identify which
recorded project characteristics are most strongly associated with those outcomes.

NASA-10: Examine projects with unusually high defect counts and identify the
characteristics that distinguish them from projects with better quality outcomes.

The implementation reuses the data‑loading helper from skill_01 and builds on the
existing functions for statistics calculation.  It reports the top three
characteristics for each NASA sub‑task, along with a quantitative measure of
association.

Usage:
    python skill_05.py <csv_path>
"""
import sys
import collections
import numpy as np

try:
    import skill_01 as s01
except ImportError:
    # Minimal fallback if skill_01 is unavailable
    import csv
    def load_data(csv_path: str):
        rows = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    r['Kloc+'] = float(r['Kloc+'])
                    r['Effort-'] = float(r['Effort-'])
                    r['Defects-'] = float(r['Defects-'])
                except (ValueError, KeyError):
                    continue
                rows.append(r)
        return rows
    s01 = type('s01', (), {'load_data': staticmethod(load_data)})

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def _is_numeric(val: str) -> bool:
    """Return True if val can be cast to float."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def analyze_characteristics(rows, target_key: str, n_top: int = 3):
    """
    For a target metric (e.g. 'Effort-' or 'Defects-'), identify the
    characteristics most strongly associated with unusually high values.
    Returns a list of dictionaries containing the column name, effect size,
    and detail information.
    """
    if not rows:
        return []

    # Overall statistics for numeric columns
    overall_numeric_stats = {}
    for col in rows[0].keys():
        if col in ('idX', 'Kloc+', 'Effort-', 'Defects-', 'MonthsX'):
            continue
        if _is_numeric(rows[0][col]):
            vals = np.array([float(r[col]) for r in rows])
            overall_numeric_stats[col] = {
                'mean': vals.mean(),
                'std': vals.std(ddof=1),
                'arr': vals,
            }

    # High‑outlier threshold (75th percentile)
    target_arr = np.array([float(r[target_key]) for r in rows])
    high_thr = np.percentile(target_arr, 75)
    high_rows = [r for r in rows if float(r[target_key]) > high_thr]
    overall_prop = len(high_rows) / len(rows)

    # Collect effects
    effects = []

    for col in rows[0].keys():
        if col in ('idX', 'Kloc+', 'Effort-', 'Defects-', 'MonthsX'):
            continue

        if col in overall_numeric_stats:
            # Numeric characteristic
            stats = overall_numeric_stats[col]
            overall_mean = stats['mean']
            overall_std = stats['std']
            if overall_std == 0:  # skip degenerate
                continue
            high_vals = np.array([float(r[col]) for r in high_rows])
            high_mean = high_vals.mean()
            effect = (high_mean - overall_mean) / overall_std  # Cohen's d
            effects.append({
                'col': col,
                'effect': effect,
                'detail': (high_mean, overall_mean),
                'type': 'numeric',
            })
        else:
            # Categorical characteristic
            cat_counts = collections.Counter(r[col] for r in rows)
            high_counts = collections.Counter(r[col] for r in high_rows)
            best_cat = None
            best_diff = -1
            for cat, total in cat_counts.items():
                prop_high = high_counts[cat] / total if total > 0 else 0
                diff = prop_high - overall_prop
                if abs(diff) > best_diff:
                    best_diff = abs(diff)
                    best_cat = cat
                    best_prop_high = prop_high
            if best_cat is not None:
                effects.append({
                    'col': col,
                    'effect': best_diff,
                    'detail': {
                        'cat': best_cat,
                        'prop_high': best_prop_high,
                        'overall_prop': overall_prop,
                    },
                    'type': 'categorical',
                })

    # Sort by absolute effect size
    effects.sort(key=lambda x: abs(x['effect']), reverse=True)
    return effects[:n_top], high_thr


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = s01.load_data(csv_path)

    if not rows:
        print("No valid data found.")
        sys.exit(1)

    # NASA-09: high effort characteristics
    effort_results, effort_thr = analyze_characteristics(rows, 'Effort-')
    print("== NASA-09 ==")
    print(f"High effort threshold (75th percentile): {effort_thr:.2f}")
    if effort_results:
        print("Top characteristics associated with unusually high effort:")
        for i, res in enumerate(effort_results, 1):
            col = res['col']
            effect = res['effect']
            if res['type'] == 'numeric':
                high_mean, overall_mean = res['detail']
                print(f"{i}. {col}: effect {effect:+.2f} "
                      f"(high mean {high_mean:.2f} vs overall {overall_mean:.2f})")
            else:
                cat = res['detail']['cat']
                prop_high = res['detail']['prop_high']
                overall_prop = res['detail']['overall_prop']
                print(f"{i}. {col} ({cat}): effect {effect:+.2f} "
                      f"(prop high {prop_high:.3f} vs overall {overall_prop:.3f})")
    else:
        print("No significant characteristics found for high effort.")

    # NASA-10: high defects characteristics
    defect_results, defect_thr = analyze_characteristics(rows, 'Defects-')
    print("\n== NASA-10 ==")
    print(f"High defects threshold (75th percentile): {defect_thr:.2f}")
    if defect_results:
        print("Top characteristics distinguishing projects with high defect counts:")
        for i, res in enumerate(defect_results, 1):
            col = res['col']
            effect = res['effect']
            if res['type'] == 'numeric':
                high_mean, overall_mean = res['detail']
                print(f"{i}. {col}: effect {effect:+.2f} "
                      f"(high mean {high_mean:.2f} vs overall {overall_mean:.2f})")
            else:
                cat = res['detail']['cat']
                prop_high = res['detail']['prop_high']
                overall_prop = res['detail']['overall_prop']
                print(f"{i}. {col} ({cat}): effect {effect:+.2f} "
                      f"(prop high {prop_high:.3f} vs overall {overall_prop:.3f})")
    else:
        print("No significant characteristics found for high defects.")


if __name__ == "__main__":
    main()
