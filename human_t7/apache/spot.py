"""Apache T7, Step 9: Spot (APACHE-17 and APACHE-18)."""

import math
import statistics

from remember import GOAL, adjustable_settings, load_configurations
from guess import (
    K,
    configuration_distance,
    nearest_neighbors,
    predict_performance,
    train_test_split,
)
from flow import historical_quantile

ANOMALY_QUANTILE = 0.90
TOP_N = 5


def outcome_anomaly(row, history, settings, k=K):
    """Compare a measured outcome with a history-only prediction."""
    if any(
        item["row_number"] == row["row_number"]
        for item in history
    ):
        raise ValueError("Evaluated row must not be in its history")

    prediction = predict_performance(row, history, settings, k)
    measured = row[GOAL]

    if not math.isfinite(measured):
        raise ValueError("Measured performance must be finite")

    return {
        "row_number": row["row_number"],
        "measured": measured,
        "predicted": prediction["prediction"],
        "residual": measured - prediction["prediction"],
        "absolute_residual": abs(
            measured - prediction["prediction"]
        ),
        "neighbor_sd": prediction["neighbor_sd"],
        "neighbor_rows": prediction["neighbor_rows"],
    }


def configuration_anomaly(row, history, settings, k=K):
    """Measure input unusualness without using performance values."""
    if any(
        item["row_number"] == row["row_number"]
        for item in history
    ):
        raise ValueError("Evaluated row must not be in its history")

    neighbors = nearest_neighbors(row, history, settings, k)
    distances = [
        configuration_distance(row, neighbor, settings)
        for neighbor in neighbors
    ]

    return {
        "row_number": row["row_number"],
        "isolation_score": statistics.mean(distances),
        "nearest_distance": min(distances),
        "neighbor_rows": [
            neighbor["row_number"] for neighbor in neighbors
        ],
    }


def training_thresholds(
    history, settings, k=K, quantile=ANOMALY_QUANTILE
):
    """Calibrate both scores using leave-one-out training records."""
    if len(history) < 2:
        raise ValueError("At least two training rows required")

    residuals = []
    isolation_scores = []

    for index, row in enumerate(history):
        reference = history[:index] + history[index + 1:]

        residuals.append(
            outcome_anomaly(
                row, reference, settings, k
            )["absolute_residual"]
        )

        isolation_scores.append(
            configuration_anomaly(
                row, reference, settings, k
            )["isolation_score"]
        )

    return {
        "residual_threshold": historical_quantile(
            sorted(residuals), quantile
        ),
        "isolation_threshold": historical_quantile(
            sorted(isolation_scores), quantile
        ),
        "training_residuals": residuals,
        "training_isolation_scores": isolation_scores,
    }


def analyze_anomalies(test, history, settings, k=K):
    """Apply training-calibrated thresholds to held-out rows."""
    thresholds = training_thresholds(history, settings, k)

    outcome_results = []
    configuration_results = []

    for row in test:
        performance = outcome_anomaly(
            row, history, settings, k
        )
        configuration = configuration_anomaly(
            row, history, settings, k
        )

        performance["flagged"] = (
            performance["absolute_residual"]
            > thresholds["residual_threshold"]
        )
        configuration["flagged"] = (
            configuration["isolation_score"]
            > thresholds["isolation_threshold"]
        )

        outcome_results.append(performance)
        configuration_results.append(configuration)

    outcome_results.sort(
        key=lambda item: (
            -item["absolute_residual"],
            item["row_number"],
        )
    )
    configuration_results.sort(
        key=lambda item: (
            -item["isolation_score"],
            item["row_number"],
        )
    )

    return {
        "thresholds": thresholds,
        "outcome_results": outcome_results,
        "configuration_results": configuration_results,
    }


def main():
    rows = load_configurations()
    settings = adjustable_settings(rows)
    train, test = train_test_split(rows)

    result = analyze_anomalies(test, train, settings)

    outcome = result["outcome_results"]
    configuration = result["configuration_results"]
    thresholds = result["thresholds"]

    print("APACHE T7 — STEP 9: SPOT")
    print(f"\nHistorical training records: {len(train)}")
    print(f"Held-out evaluation records: {len(test)}")
    print(f"Nearest neighbors: {K}")
    print("Thresholds: training leave-one-out 90th percentiles")

    print("\nAPACHE-17: UNEXPECTED PERFORMANCE")
    print(
        f"Absolute residual threshold: "
        f"{thresholds['residual_threshold']:.2f}"
    )
    print(
        f"Flagged held-out records: "
        f"{sum(item['flagged'] for item in outcome)}"
    )

    print("\nFive largest prediction residuals:")
    print(
        f"{'Row':>5} {'Measured':>10} {'Predicted':>11}"
        f" {'Residual':>11} {'Abs Error':>11} {'Flag':>6}"
    )

    for item in outcome[:TOP_N]:
        print(
            f"{item['row_number']:>5}"
            f" {item['measured']:>10.2f}"
            f" {item['predicted']:>11.2f}"
            f" {item['residual']:>+11.2f}"
            f" {item['absolute_residual']:>11.2f}"
            f" {'YES' if item['flagged'] else 'NO':>6}"
        )

    print("\nAPACHE-18: UNUSUAL CONFIGURATION SETTINGS")
    print(
        f"Mean-neighbor-distance threshold: "
        f"{thresholds['isolation_threshold']:.4f}"
    )
    print(
        f"Flagged held-out records: "
        f"{sum(item['flagged'] for item in configuration)}"
    )

    print("\nFive most isolated configurations:")
    print(
        f"{'Row':>5} {'Isolation':>11}"
        f" {'Nearest':>10} {'Flag':>6}  Neighbor rows"
    )

    for item in configuration[:TOP_N]:
        print(
            f"{item['row_number']:>5}"
            f" {item['isolation_score']:>11.4f}"
            f" {item['nearest_distance']:>10.4f}"
            f" {'YES' if item['flagged'] else 'NO':>6}"
            f"  {item['neighbor_rows']}"
        )

    outcome_ids = {
        item["row_number"] for item in outcome
        if item["flagged"]
    }
    configuration_ids = {
        item["row_number"] for item in configuration
        if item["flagged"]
    }

    print("\nOVERLAP BETWEEN ANOMALY TYPES")
    print(f"Outcome anomalies: {len(outcome_ids)}")
    print(f"Configuration anomalies: {len(configuration_ids)}")
    print(f"Flagged by both: {len(outcome_ids & configuration_ids)}")

    print(
        "\nPerformance anomalies use held-out measurements "
        "only after prediction."
        "\nConfiguration anomaly scores never use performance."
        "\nStrictly exceeding the training threshold triggers "
        "a flag; ties do not."
        "\nAnomaly flags are descriptive screening indicators."
    )


if __name__ == "__main__":
    main()
