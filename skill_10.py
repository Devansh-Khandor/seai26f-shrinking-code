#!/usr/bin/env python3
"""
skill_10.py - NASA-19 and NASA-20: Evaluation of estimation and selection strategies.

Usage:
    python skill_10.py <csv_path>
"""

import sys
import random
import numpy as np

# ----------------------------------------------------------------------
# Import utilities from earlier skill modules
# ----------------------------------------------------------------------
try:
    from skill_01 import load_csv, get_numeric_column, favorable_projects
    from skill_02 import prepare_data, train_regression
except Exception as e:
    # Fallback minimal implementations (should not be required in the test environment)
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

    def favorable_projects(rows, size_col, effort_col, defect_col, effort_quantile=0.25, defect_quantile=0.25):
        efforts = get_numeric_column(rows, effort_col)
        defects = get_numeric_column(rows, defect_col)
        if not efforts or not defects:
            return []
        effort_thresh = sorted(efforts)[int(len(efforts)*effort_quantile)]
        defect_thresh = sorted(defects)[int(len(defects)*defect_quantile)]
        fav = []
        for r in rows:
            try:
                eff = float(r[effort_col])
                defc = float(r[defect_col])
            except (ValueError, TypeError):
                continue
            if eff <= effort_thresh and defc <= defect_thresh:
                fav.append(r)
        return fav

    def identify_numeric_columns(rows, header):
        numeric = set()
        categorical = set()
        for col in header:
            if col == 'idX':
                continue
            sample = None
            for r in rows:
                val = r.get(col, '')
                if val != '':
                    sample = val
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
            val = r.get(col, '')
            mapping = mappings[col]
            vec.append(float(mapping.get(val, -1)))  # unseen -> -1
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
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        return model

# ----------------------------------------------------------------------
# NASA-19: Compare model vs historical average
# ----------------------------------------------------------------------
def nasa_19(rows):
    """Compare project‑effort estimation system against a simple historical‑average estimate."""
    print("\n== NASA-19 ==")

    n = len(rows)
    if n < 5:
        print("Insufficient data to perform a train/test split.")
        return

    indices = list(range(n))
    random.shuffle(indices)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    train_rows = [rows[i] for i in train_idx]
    test_rows  = [rows[i] for i in test_idx]

    header = list(rows[0].keys())

    # Prepare data
    X_train, y_train, _, _, _ = prepare_data(train_rows, header, 'Effort-')
    X_test,  y_test,  _, _, _ = prepare_data(test_rows,  header, 'Effort-')

    # Train linear regression model
    model = train_regression(X_train, y_train)

    # Predictions
    pred_model = model.predict(X_test)
    # Historical‑average prediction
    mean_train = np.mean(y_train)
    pred_hist  = np.full_like(pred_model, mean_train)

    # Error metrics
    mae_model = np.mean(np.abs(pred_model - y_test))
    mae_hist  = np.mean(np.abs(pred_hist  - y_test))
    mse_model = np.mean((pred_model - y_test)**2)
    mse_hist  = np.mean((pred_hist  - y_test)**2)

    print(f"Test set size: {len(y_test)} projects")
    print(f"Historical‑average MAE: {mae_hist:.2f}, MSE: {mse_hist:.2f}")
    print(f"Model MAE: {mae_model:.2f}, MSE: {mse_model:.2f}")
    if mae_hist:
        improvement = (mae_hist - mae_model) / mae_hist * 100
        print(f"Model reduces MAE by {improvement:.1f}% compared to historical average.")
    else:
        print("Historical MAE is zero, cannot compute improvement.")


# ----------------------------------------------------------------------
# NASA-20: Compare selection vs random under same budget
# ----------------------------------------------------------------------
def nasa_20(rows):
    """Compare our project‑selection approach with random selection under the same evaluation budget."""
    print("\n== NASA-20 ==")

    # Determine favorable projects using the existing helper
    fav_projects = favorable_projects(rows, 'Kloc+', 'Effort-', 'Defects-')
    num_fav = len(fav_projects)
    if num_fav == 0:
        print("No favorable projects identified; cannot compare selection.")
        return

    # Random selection with the same count
    all_ids = [r['idX'] for r in rows]
    random_ids = random.sample(all_ids, num_fav)

    # Helper to compute average size/(effort+defects) metric
    def avg_metric(selected_ids):
        metrics = []
        for r in rows:
            if r['idX'] in selected_ids:
                try:
                    size   = float(r['Kloc+'])
                    effort = float(r['Effort-'])
                    defects= float(r['Defects-'])
                    denom = effort + defects
                    if denom > 0:
                        metrics.append(size / denom)
                except (ValueError, TypeError):
                    continue
        return np.mean(metrics) if metrics else 0.0

    fav_metric    = avg_metric({r['idX'] for r in fav_projects})
    random_metric = avg_metric(set(random_ids))

    print(f"Number of projects selected: {num_fav}")
    print(f"Average size/(effort+defects) for favorable selection: {fav_metric:.3f}")
    print(f"Average size/(effort+defects) for random selection:    {random_metric:.3f}")
    diff = fav_metric - random_metric
    if diff > 0:
        print(f"Difference: {diff:.3f} (favorable selection yields a higher metric, i.e., more desirable outcomes)")
    elif diff < 0:
        print(f"Difference: {diff:.3f} (random selection yields a higher metric, unexpected)")
    else:
        print("No difference between selection strategies.")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, _ = load_csv(csv_path)

    nasa_19(rows)
    nasa_20(rows)


if __name__ == "__main__":
    main()
