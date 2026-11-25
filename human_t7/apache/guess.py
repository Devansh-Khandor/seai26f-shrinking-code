"""Apache T7, Step 2: Guess (APACHE-03 and APACHE-04)."""

import math
import random
import statistics

from remember import (
    GOAL,
    adjustable_settings,
    load_configurations,
)

SEED = 42
K = 5
TRAIN_FRACTION = 0.80


def train_test_split(configurations, seed=SEED):
    """Split records reproducibly without modifying the input."""
    if len(configurations) < 2:
        raise ValueError("At least two records are required")

    indices = list(range(len(configurations)))
    random.Random(seed).shuffle(indices)

    n_train = max(
        1, min(len(configurations) - 1,
               round(len(configurations) * TRAIN_FRACTION))
    )

    train_indices = set(indices[:n_train])

    train = [
        row for i, row in enumerate(configurations)
        if i in train_indices
    ]
    test = [
        row for i, row in enumerate(configurations)
        if i not in train_indices
    ]

    return train, test


def configuration_distance(a, b, settings):
    """Normalized Hamming distance over adjustable settings."""
    if not settings:
        raise ValueError("No adjustable settings")

    return sum(
        a[name] != b[name] for name in settings
    ) / len(settings)


def nearest_neighbors(query, history, settings, k=K):
    """Return the nearest historical observations."""
    if not history or k < 1:
        raise ValueError("History must be nonempty and k positive")

    return sorted(
        history,
        key=lambda row: (
            configuration_distance(query, row, settings),
            row["row_number"],
        ),
    )[:k]


def predict_performance(query, history, settings, k=K):
    """Predict from observed neighbors; report their dispersion."""
    neighbors = nearest_neighbors(query, history, settings, k)

    values = [row[GOAL] for row in neighbors]

    return {
        "prediction": statistics.mean(values),
        "neighbor_min": min(values),
        "neighbor_max": max(values),
        "neighbor_sd": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "neighbor_rows": [
            row["row_number"] for row in neighbors
        ],
    }


def mean_absolute_error(actual, predicted):
    """Calculate mean absolute prediction error."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Mismatched or empty prediction lists")

    return statistics.mean(
        abs(a - p) for a, p in zip(actual, predicted)
    )


def evaluate_predictions(train, test, settings, k=K):
    """Evaluate against held-out outcomes and a training-mean baseline."""
    if not train or not test:
        raise ValueError("Train and test must be nonempty")

    actual = [row[GOAL] for row in test]
    predicted = [
        predict_performance(row, train, settings, k)["prediction"]
        for row in test
    ]

    baseline = statistics.mean(row[GOAL] for row in train)
    baseline_predictions = [baseline] * len(test)

    return {
        "mae": mean_absolute_error(actual, predicted),
        "baseline_mae": mean_absolute_error(
            actual, baseline_predictions
        ),
        "baseline_prediction": baseline,
        "actual": actual,
        "predicted": predicted,
    }


def differing_settings(a, b, settings):
    """Find settings on which two configurations differ."""
    return [
        name for name in settings
        if a[name] != b[name]
    ]


def one_setting_pairs(configurations, settings):
    """Find observed pairs that differ in exactly one setting."""
    pairs = []

    for i, a in enumerate(configurations):
        for b in configurations[i + 1:]:
            changes = differing_settings(a, b, settings)
            if len(changes) == 1:
                pairs.append((a, b, changes[0]))

    return pairs


def estimate_pair_difference(a, b, train, settings, k=K):
    """Estimate B minus A without using either held-out outcome."""
    changes = differing_settings(a, b, settings)

    if len(changes) != 1:
        raise ValueError(
            "Configurations must differ in exactly one setting"
        )

    prediction_a = predict_performance(a, train, settings, k)
    prediction_b = predict_performance(b, train, settings, k)

    return {
        "changed_setting": changes[0],
        "estimate_a": prediction_a,
        "estimate_b": prediction_b,
        "predicted_difference": (
            prediction_b["prediction"] -
            prediction_a["prediction"]
        ),
    }


def main():
    configurations = load_configurations()
    settings = adjustable_settings(configurations)

    train, test = train_test_split(configurations)
    results = evaluate_predictions(train, test, settings)

    print("APACHE T7 — STEP 2: GUESS")
    print(f"\nTraining configurations: {len(train)}")
    print(f"Held-out configurations: {len(test)}")
    print(f"Random seed: {SEED}")
    print(f"Nearest neighbors: {K}")

    print("\nAPACHE-03: PREDICTION EVALUATION")
    print(f"kNN MAE: {results['mae']:.2f}")
    print(f"Historical-mean MAE: {results['baseline_mae']:.2f}")
    print(
        f"Training-mean prediction: "
        f"{results['baseline_prediction']:.2f}"
    )

    print("\nFirst five held-out predictions:")
    for row in test[:5]:
        outcome = predict_performance(row, train, settings)
        print(
            f"Row {row['row_number']:>3}: "
            f"actual={row[GOAL]:>8.2f}, "
            f"predicted={outcome['prediction']:>8.2f}, "
            f"neighbor_sd={outcome['neighbor_sd']:>8.2f}"
        )

    print("\nAPACHE-04: ONE-SETTING CONFIGURATION COMPARISON")

    pairs = one_setting_pairs(test, settings)
    print(f"Eligible held-out pairs: {len(pairs)}")

    if not pairs:
        print(
            "No qualifying held-out pair exists for this split. "
            "APACHE-04 demonstration unavailable."
        )
        return

    a, b, setting = pairs[0]
    comparison = estimate_pair_difference(
        a, b, train, settings
    )

    pred_a = comparison["estimate_a"]
    pred_b = comparison["estimate_b"]

    print(f"Configuration A: row {a['row_number']}")
    print(f"Configuration B: row {b['row_number']}")
    print(
        f"Changed setting: {setting} "
        f"({a[setting]} -> {b[setting]})"
    )

    print(
        f"Estimated performance A: "
        f"{pred_a['prediction']:.2f}"
    )
    print(
        f"Estimated performance B: "
        f"{pred_b['prediction']:.2f}"
    )
    print(
        f"Estimated difference (B - A): "
        f"{comparison['predicted_difference']:+.2f}"
    )

    print(
        f"A neighbor range: "
        f"{pred_a['neighbor_min']:.2f} to "
        f"{pred_a['neighbor_max']:.2f}; "
        f"SD={pred_a['neighbor_sd']:.2f}"
    )
    print(
        f"B neighbor range: "
        f"{pred_b['neighbor_min']:.2f} to "
        f"{pred_b['neighbor_max']:.2f}; "
        f"SD={pred_b['neighbor_sd']:.2f}"
    )

    print(
        "\nNeighbor ranges and SDs are descriptive, "
        "not calibrated confidence intervals."
    )

    print(
        f"Observed held-out difference (B - A): "
        f"{b[GOAL] - a[GOAL]:+.2f}"
    )
    print(
        "The observed difference is shown only for retrospective "
        "evaluation, not used to make either prediction."
    )


if __name__ == "__main__":
    main()
