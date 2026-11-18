#!/usr/bin/env python3
import csv
import json
import sys
import math
from sklearn.linear_model import LinearRegression

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
                    try:
                        if '.' in v:
                            cleaned[k] = float(v)
                        else:
                            cleaned[k] = int(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append((idx, cleaned))
    return rows, reader.fieldnames

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)

    # Identify goal and setting columns
    goal_columns = [col for col in fieldnames if col.endswith(('+', '-'))]
    # We are focusing on Performance- goal
    if 'Performance-' not in goal_columns:
        print("Error: 'Performance-' column not found.")
        sys.exit(1)
    setting_columns = [col for col in fieldnames if col not in goal_columns and not col.endswith('X')]

    # Held-out rows: numbers divisible by 5
    held_out_ids = {idx for idx, _ in rows if idx % 5 == 0}
    # Training rows: all others
    train_rows = [(idx, r) for idx, r in rows if idx not in held_out_ids]

    # Compute means for each setting (train only, ignoring missing)
    setting_means = {}
    for col in setting_columns:
        vals = [r[col] for _, r in train_rows if r[col] is not None]
        if vals:
            setting_means[col] = sum(vals) / len(vals)
        else:
            setting_means[col] = 0.0

    # Prepare training data: only rows with non-missing goal and non-missing settings (after impute)
    X_train = []
    y_train = []
    for idx, r in train_rows:
        if r['Performance-'] is None:
            continue
        # Build feature vector with imputed means
        vec = []
        skip = False
        for col in setting_columns:
            val = r[col]
            if val is None:
                val = setting_means[col]
            vec.append(val)
        X_train.append(vec)
        y_train.append(r['Performance-'])

    if not X_train:
        print("Error: No training data available.")
        sys.exit(1)

    # Fit linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Predict on held-out rows
    predictions = {}
    errors = []
    for idx, r in rows:
        if idx not in held_out_ids:
            continue
        # Build feature vector with imputed means
        vec = []
        for col in setting_columns:
            val = r[col]
            if val is None:
                val = setting_means[col]
            vec.append(val)
        pred = lr.predict([vec])[0]
        predictions[str(idx)] = pred
        true = r['Performance-']
        if true is not None:
            errors.append(abs(true - pred))

    mae = sum(errors) / len(errors) if errors else 0.0

    # === APACHE-03 ===
    print("== APACHE-03 ==")
    print(f"Total held-out rows: {len(held_out_ids)}")
    print(f"MAE on held-out predictions: {mae:.6f}")
    print(f"First prediction example: row {next(iter(predictions))} -> {predictions[next(iter(predictions))]:.6f}")

    # Answer JSON
    ans_apache_03 = {
        "predictions": {k: round(v, 6) for k, v in predictions.items()},
        "mae": round(mae, 6)
    }
    print(f"ANSWER APACHE-03 {json.dumps(ans_apache_03)}")

    # === APACHE-04 ===
    print("\n== APACHE-04 ==")
    # Find pairs differing in exactly one setting
    pairs = []
    # Precompute prediction for all rows (using imputed means)
    all_preds = {}
    for idx, r in rows:
        # Skip if goal missing
        if r['Performance-'] is None:
            continue
        vec = []
        skip = False
        for col in setting_columns:
            val = r[col]
            if val is None:
                val = setting_means[col]
            vec.append(val)
        all_preds[idx] = lr.predict([vec])[0]

    # Helper to compare two rows
    def diff_setting(idx_a, idx_b):
        r_a = rows_dict[idx_a]
        r_b = rows_dict[idx_b]
        diff_cols = []
        for col in setting_columns:
            va = r_a[col]
            vb = r_b[col]
            if va is None or vb is None:
                return None  # skip if missing
            if va != vb:
                diff_cols.append(col)
        if len(diff_cols) == 1:
            return diff_cols[0]
        return None

    # Build dict for quick access
    rows_dict = {idx: r for idx, r in rows}

    for i, (idx_a, r_a) in enumerate(rows):
        if r_a['Performance-'] is None:
            continue
        for idx_b, r_b in rows[i+1:]:
            if r_b['Performance-'] is None:
                continue
            setting = diff_setting(idx_a, idx_b)
            if setting:
                diff = all_preds[idx_b] - all_preds[idx_a]
                low = diff - mae
                high = diff + mae
                pairs.append({
                    "a": idx_a,
                    "b": idx_b,
                    "setting": setting,
                    "difference": round(diff, 6),
                    "low": round(low, 6),
                    "high": round(high, 6)
                })
                if len(pairs) >= 5:
                    break
        if len(pairs) >= 5:
            break

    # Show at least 5 pairs
    for p in pairs:
        print(f"Pair {p['a']} vs {p['b']}: setting={p['setting']}, diff={p['difference']:.6f}")

    # Answer JSON
    ans_apache_04 = {"pairs": pairs[:5]}
    print(f"ANSWER APACHE-04 {json.dumps(ans_apache_04)}")

if __name__ == "__main__":
    main()
