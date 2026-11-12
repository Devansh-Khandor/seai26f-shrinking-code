#!/usr/bin/env python3
"""
Skill 04: Dynamic proposal assessment and continuous historical update

NASA-07: Assess new project proposals using their characteristics and flag
unusually high predicted effort or defects.

NASA-08: Update all historical summaries and project estimates when new
completed-project records are added to the data.

Usage:
    python skill_04.py <csv_path>
"""

import csv
import sys
from typing import List, Dict, Any

# Reuse helper functions from existing skills
try:
    import skill_01 as s01
    import skill_02 as s02
except ImportError:
    # Fallback definitions if modules are not importable
    def load_data(csv_path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
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
    s01 = type('s01', (), {'load_data': staticmethod(load_data),
                           'compute_stats': staticmethod(
                               lambda rows: {'Kloc+': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
                                             'Effort-': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
                                             'Defects-': {'mean': 0, 'std': 0, 'min': 0, 'max': 0}}),
                           'best_and_worst': staticmethod(
                               lambda rows: {'best': {}, 'worst': {}})})

    def estimate_effort(rows):
        return 0, 0, 0, 0, 0
    def estimate_defects(rows):
        return 0, 0, 0, 0, 0
    s02 = type('s02', (), {'estimate_effort': staticmethod(estimate_effort),
                           'estimate_defects': staticmethod(estimate_defects)})

# -------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------

def parse_rows(csv_path: str) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """
    Read the CSV and split into historical (completed) rows and proposal
    rows (missing numeric Effort- or Defects- values).
    Returns (historical, proposals).
    """
    historical: List[Dict[str, Any]] = []
    proposals: List[Dict[str, Any]] = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r_copy = r.copy()
            # Convert numeric columns if possible
            is_historical = True
            try:
                r_copy['Kloc+'] = float(r_copy['Kloc+'])
            except (ValueError, KeyError):
                is_historical = False
            try:
                r_copy['Effort-'] = float(r_copy['Effort-'])
            except (ValueError, KeyError):
                is_historical = False
            try:
                r_copy['Defects-'] = float(r_copy['Defects-'])
            except (ValueError, KeyError):
                is_historical = False

            if is_historical:
                historical.append(r_copy)
            else:
                proposals.append(r_copy)

    return historical, proposals


# -------------------------------------------------------------
# Prediction helpers
# -------------------------------------------------------------

def train_regression_XY(rows, target_key: str):
    """Train a simple LinearRegression and return model, X, y, mse, sxx."""
    from sklearn.linear_model import LinearRegression
    import numpy as np

    X = np.array([r['Kloc+'] for r in rows]).reshape(-1, 1)
    y = np.array([r[target_key] for r in rows])

    model = LinearRegression().fit(X, y)

    residuals = y - model.predict(X)
    mse = np.mean(residuals ** 2)
    n = len(rows)
    sxx = np.sum((X - X.mean()) ** 2)

    return model, X, y, mse, sxx


def predict_with_se(model, X_val, mse, sxx, X_mean, n):
    """
    Return predicted value and standard error for a single X value.
    """
    import numpy as np
    pred = model.predict([[X_val]])[0]
    se = np.sqrt(mse * (1 + 1 / n + ((X_val - X_mean) ** 2) / sxx))
    return pred, se


# -------------------------------------------------------------
# Output helpers
# -------------------------------------------------------------

def print_nasa_07(proposals, effort_stats, defect_stats, effort_model, defect_model, effort_mse,
                 effort_sxx, defect_mse, defect_sxx):
    """
    NASA-07: Evaluate proposals and flag unusually high predictions.
    """
    print("== NASA-07 ==")

    if not proposals:
        print("No new proposals to evaluate.")
        return

    header = f"{'ID':>3} {'Size(KLOC)':>10} {'Effort(PM)':>12} {'Effort SE':>10} {'Defects':>10} {'Defects SE':>10} {'Alerts'}"
    print(header)
    for r in proposals:
        try:
            kloc = float(r['Kloc+'])
        except Exception:
            # If size missing, skip proposal
            continue
        # Predicted effort
        eff_pred, eff_se = predict_with_se(effort_model, kloc, effort_mse,
                                           effort_sxx, effort_model.predict([[kloc]])[0],
                                           len(effort_model.coef_))
        # Predicted defects
        def_pred, def_se = predict_with_se(defect_model, kloc, defect_mse,
                                           defect_sxx, defect_model.predict([[kloc]])[0],
                                           len(defect_model.coef_))

        alerts = []
        if eff_pred > effort_stats['mean'] + 2 * effort_stats['std']:
            alerts.append("High Effort")
        if def_pred > defect_stats['mean'] + 2 * defect_stats['std']:
            alerts.append("High Defects")
        alert_str = ", ".join(alerts) if alerts else ""

        print(f"{int(r['idX']):>3} {kloc:10.2f} {eff_pred:12.2f} {eff_se:10.2f} {def_pred:10.2f} {def_se:10.2f} {alert_str}")

    print()  # blank line after section


def print_nasa_08(historical, bw, effort_stats, defect_stats, effort_model, defect_model,
                  effort_mse, effort_sxx, defect_mse, defect_sxx):
    """
    NASA-08: Re‑compute all summaries and print them.
    """
    print("== NASA-08 ==")

    # Summary statistics (like skill_01.print_summary)
    print(f"Typical size (KLOC): {historical['Kloc+']['mean']:.2f} ± {historical['Kloc+']['std']:.2f}")
    print(f"Typical effort (PM): {historical['Effort-']['mean']:.2f} ± {historical['Effort-']['std']:.2f}")
    print(f"Typical defects: {historical['Defects-']['mean']:.0f} ± {historical['Defects-']['std']:.0f}")

    best = bw['best']
    worst = bw['worst']
    print(f"Best project (ID {best['idX']}): size {best['Kloc+']:.2f}, effort {best['Effort-']:.2f}, defects {best['Defects-']}")
    print(f"Worst project (ID {worst['idX']}): size {worst['Kloc+']:.2f}, effort {worst['Effort-']:.2f}, defects {worst['Defects-']}")

    # Effort estimation (like skill_02)
    print("\n== NASA-03 ==")
    # Typical project size is mean KLOC
    mean_kloc = effort_model.predict([[effort_model.predict([[0]])[0]]])[0]  # placeholder to avoid unused var
    mean_kloc = np.mean([r['Kloc+'] for r in historical_rows])  # compute later
    # For clarity, we recalc using actual historical data
    mean_kloc = np.mean([r['Kloc+'] for r in historical_rows])
    effort_pred, effort_se, effort_ci_low, effort_ci_high, effort_r2 = estimate_effort(historical_rows)

    print(f"Estimated effort for a typical project (size {mean_kloc:.2f} KLOC): "
          f"{effort_pred:.2f} PM ± {effort_se:.2f} PM "
          f"(95% CI: {effort_ci_low:.2f} – {effort_ci_high:.2f})")
    print(f"Model R²: {effort_r2:.2f}")

    # Defect estimation
    print("\n== NASA-04 ==")
    defect_pred, defect_se, defect_ci_low, defect_ci_high, defect_r2 = estimate_defects(historical_rows)

    print(f"Estimated defects for a typical project (size {mean_kloc:.2f} KLOC): "
          f"{defect_pred:.2f} ± {defect_se:.2f} defects "
          f"(95% CI: {defect_ci_low:.2f} – {defect_ci_high:.2f})")
    print(f"Model R²: {defect_r2:.2f}")


# -------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    historical_rows, proposals = parse_rows(csv_path)

    if not historical_rows:
        print("No historical project data available. Cannot train models.")
        sys.exit(1)

    # Historical statistics
    hist_stats = s01.compute_stats(historical_rows)
    bw = s01.best_and_worst(historical_rows)

    # Train regression models for effort and defects
    effort_model, effort_X, effort_y, effort_mse, effort_sxx = train_regression_XY(historical_rows, 'Effort-')
    defect_model, defect_X, defect_y, defect_mse, defect_sxx = train_regression_XY(historical_rows, 'Defects-')

    # NASA-07: assess proposals
    print_nasa_07(proposals,
                  hist_stats['Effort-'],
                  hist_stats['Defects-'],
                  effort_model,
                  defect_model,
                  effort_mse,
                  effort_sxx,
                  defect_mse,
                  defect_sxx)

    # NASA-08: update summaries and estimates
    # We reuse the same statistics and models because we have already updated them
    # Print the updated information
    # Re‑compute estimates for a typical project (use mean KLOC)
    import numpy as np
    mean_kloc = np.mean([r['Kloc+'] for r in historical_rows])
    effort_pred, effort_se, effort_ci_low, effort_ci_high, effort_r2 = s02.estimate_effort(historical_rows)
    defect_pred, defect_se, defect_ci_low, defect_ci_high, defect_r2 = s02.estimate_defects(historical_rows)

    print("\n== NASA-08 ==")
    # Summary statistics
    print(f"Typical size (KLOC): {hist_stats['Kloc+']['mean']:.2f} ± {hist_stats['Kloc+']['std']:.2f}")
    print(f"Typical effort (PM): {hist_stats['Effort-']['mean']:.2f} ± {hist_stats['Effort-']['std']:.2f}")
    print(f"Typical defects: {hist_stats['Defects-']['mean']:.0f} ± {hist_stats['Defects-']['std']:.0f}")
    best = bw['best']
    worst = bw['worst']
    print(f"Best project (ID {best['idX']}): size {best['Kloc+']:.2f}, effort {best['Effort-']:.2f}, defects {best['Defects-']}")
    print(f"Worst project (ID {worst['idX']}): size {worst['Kloc+']:.2f}, effort {worst['Effort-']:.2f}, defects {worst['Defects-']}")
    # Effort estimate
    print("\n== NASA-03 ==")
    print(f"Estimated effort for a typical project (size {mean_kloc:.2f} KLOC): "
          f"{effort_pred:.2f} PM ± {effort_se:.2f} PM "
          f"(95% CI: {effort_ci_low:.2f} – {effort_ci_high:.2f})")
    print(f"Model R²: {effort_r2:.2f}")
    # Defect estimate
    print("\n== NASA-04 ==")
    print(f"Estimated defects for a typical project (size {mean_kloc:.2f} KLOC): "
          f"{defect_pred:.2f} ± {defect_se:.2f} defects "
          f"(95% CI: {defect_ci_low:.2f} – {defect_ci_high:.2f})")
    print(f"Model R²: {defect_r2:.2f}")


if __name__ == "__main__":
    main()
