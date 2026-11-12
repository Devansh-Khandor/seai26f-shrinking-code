#!/usr/bin/env python3
"""
skill_04.py - NASA-07 and NASA-08: Real‑time project assessment and summary updates.

Usage:
    python skill_04.py <csv_path>
"""

import sys
import numpy as np

# ----------------------------------------------------------------------
# Reuse helpers from skill_01
# ----------------------------------------------------------------------
try:
    from skill_01 import load_csv, get_numeric_column
except Exception:
    import csv
    def load_csv(csv_path: str):
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = [row for row in reader]
        return rows, header

    def get_numeric_column(rows, col):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (ValueError, TypeError):
                pass
        return vals

# ----------------------------------------------------------------------
# Reuse data preparation and modelling from skill_02
# ----------------------------------------------------------------------
try:
    from skill_02 import prepare_data, train_regression, predict_with_uncertainty
except Exception:
    # Minimal local implementation if skill_02 is not importable
    from sklearn.linear_model import LinearRegression

    def identify_numeric_columns(rows, header):
        numeric = set()
        categorical = set()
        for col in header:
            if col == 'idX':
                continue
            sample = None
            for r in rows:
                v = r.get(col, '')
                if v != '':
                    sample = v
                    break
            if sample is None:
                categorical.add(col)
            else:
                try:
                    float(sample)
                    numeric.add(col)
                except ValueError:
                    categorical.add(col)
        return numeric, categorical

    def build_mappings(rows, categorical):
        mappings = {}
        for col in categorical:
            mapping = {}
            idx = 0
            for r in rows:
                val = r.get(col, '')
                if val == '':
                    continue
                if val not in mapping:
                    mapping[val] = idx
                    idx += 1
            mappings[col] = mapping
        return mappings

    def encode_row(r, numeric, categorical, mappings):
        vec = []
        for col in numeric:
            try:
                vec.append(float(r.get(col, '0')))
            except ValueError:
                vec.append(0.0)
        for col in categorical:
            mapping = mappings[col]
            vec.append(float(mapping.get(r.get(col, ''), -1)))  # unseen -> -1
        return vec

    def prepare_data(rows, header, target_col):
        numeric, categorical = identify_numeric_columns(rows, header)
        numeric.discard(target_col)
        categorical.discard(target_col)
        numeric.discard('idX')
        categorical.discard('idX')
        mappings = build_mappings(rows, categorical)

        X = []
        y = []
        for r in rows:
            try:
                y_val = float(r.get(target_col, '0'))
            except ValueError:
                continue
            X.append(encode_row(r, numeric, categorical, mappings))
            y.append(y_val)
        return np.array(X, dtype=float), np.array(y, dtype=float), numeric, categorical, mappings

    def train_regression(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    def predict_with_uncertainty(model, X_train, y_train, X_new):
        y_pred_train = model.predict(X_train)
        residuals = y_train - y_pred_train
        std_resid = np.sqrt(np.mean(residuals ** 2))
        y_pred_new = model.predict(X_new)[0]
        uncertainty = 2 * std_resid
        return y_pred_new, uncertainty, std_resid

# ----------------------------------------------------------------------
# Reuse grouping and statistics from skill_03
# ----------------------------------------------------------------------
try:
    from skill_03 import group_values_by_category, compute_stats
except Exception:
    from statistics import mean, stdev, StatisticsError

    def group_values_by_category(rows, category_col, value_col):
        groups = {}
        for r in rows:
            cat = r.get(category_col)
            if not cat:
                continue
            try:
                val = float(r.get(value_col, ''))
            except (ValueError, TypeError):
                continue
            groups.setdefault(cat, []).append(val)
        return groups

    def compute_stats(values):
        if not values:
            return float('nan'), float('nan')
        try:
            m = mean(values)
            sd = stdev(values) if len(values) > 1 else 0.0
        except StatisticsError:
            m, sd = float('nan'), float('nan')
        return m, sd

# ----------------------------------------------------------------------
# NASA-07: assess new proposals
# ----------------------------------------------------------------------
def report_nasa_07(rows, header):
    print("\n== NASA-07 ==")

    # Prepare data for both targets
    X_effort, y_effort, _, _, _ = prepare_data(rows, header, 'Effort-')
    X_defect, y_defect, _, _, _ = prepare_data(rows, header, 'Defects-')

    # Historical thresholds (median + 2*std)
    effort_vals = get_numeric_column(rows, 'Effort-')
    defect_vals = get_numeric_column(rows, 'Defects-')
    if not effort_vals or not defect_vals:
        print("Insufficient data for NASA-07.")
        return

    effort_mean, effort_std = compute_stats(effort_vals)
    defect_mean, defect_std = compute_stats(defect_vals)

    effort_thresh = effort_mean + 2 * effort_std
    defect_thresh = defect_mean + 2 * defect_std

    print(f"Historical Effort: mean={effort_mean:.2f} std={effort_std:.2f} threshold={effort_thresh:.2f}")
    print(f"Historical Defects: mean={defect_mean:.2f} std={defect_std:.2f} threshold={defect_thresh:.2f}")

    alerts = []
    n = len(rows)
    for i in range(n):
        # Leave-one-out for Effort
        X_train_eff = np.delete(X_effort, i, axis=0)
        y_train_eff = np.delete(y_effort, i)
        model_eff = train_regression(X_train_eff, y_train_eff)
        pred_eff, _, _ = predict_with_uncertainty(
            model_eff, X_train_eff, y_train_eff, X_effort[i].reshape(1, -1)
        )
        # Leave-one-out for Defects
        X_train_def = np.delete(X_defect, i, axis=0)
        y_train_def = np.delete(y_defect, i)
        model_def = train_regression(X_train_def, y_train_def)
        pred_def, _, _ = predict_with_uncertainty(
            model_def, X_train_def, y_train_def, X_defect[i].reshape(1, -1)
        )

        id_proj = rows[i].get('idX', str(i+1))
        if pred_eff > effort_thresh or pred_def > defect_thresh:
            msg = f"Alert: Project {id_proj}: PredEff={pred_eff:.2f} (th={effort_thresh:.2f}); PredDef={pred_def:.2f} (th={defect_thresh:.2f})"
            alerts.append(msg)

    if alerts:
        print(f"\n{len(alerts)} alert(s) found:")
        for a in alerts:
            print(a)
    else:
        print("\nNo alerts: all projects within normal bounds.")

# ----------------------------------------------------------------------
# NASA-08: update summaries and models
# ----------------------------------------------------------------------
def report_nasa_08(rows, header):
    print("\n== NASA-08 ==")

    n = len(rows)
    if n == 0:
        print("No data to summarize.")
        return

    effort_vals = get_numeric_column(rows, 'Effort-')
    defect_vals = get_numeric_column(rows, 'Defects-')

    eff_mean, eff_std = compute_stats(effort_vals)
    def_mean, def_std = compute_stats(defect_vals)

    print(f"Total projects: {n}")
    print(f"Effort: mean={eff_mean:.2f} std={eff_std:.2f}")
    print(f"Defects: mean={def_mean:.2f} std={def_std:.2f}")

    # Full‑data models
    X_effort, y_effort, _, _, _ = prepare_data(rows, header, 'Effort-')
    X_defect, y_defect, _, _, _ = prepare_data(rows, header, 'Defects-')

    model_eff = train_regression(X_effort, y_effort)
    model_def = train_regression(X_defect, y_defect)

    r2_eff = model_eff.score(X_effort, y_effort)
    r2_def = model_def.score(X_defect, y_defect)

    print(f"Effort model R²: {r2_eff:.3f}")
    print(f"Defects model R²: {r2_def:.3f}")

    # Predict for the most recent project (last row)
    pred_eff, unc_eff, _ = predict_with_uncertainty(
        model_eff, X_effort, y_effort, X_effort[-1].reshape(1, -1)
    )
    pred_def, unc_def, _ = predict_with_uncertainty(
        model_def, X_defect, y_defect, X_defect[-1].reshape(1, -1)
    )
    id_last = rows[-1].get('idX', str(n))
    print(f"Predicted effort for project {id_last}: {pred_eff:.2f} +/- {unc_eff:.2f}")
    print(f"Predicted defects for project {id_last}: {pred_def:.2f} +/- {unc_def:.2f}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, header = load_csv(csv_path)

    report_nasa_07(rows, header)
    report_nasa_08(rows, header)

if __name__ == "__main__":
    main()
