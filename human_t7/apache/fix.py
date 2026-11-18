"""Apache T7, Step 8: Fix (APACHE-15 and APACHE-16)."""

import math

from remember import GOAL, adjustable_settings, load_configurations
from guess import (
    differing_settings,
    predict_performance,
    train_test_split,
)

MAX_SMALL_CHANGES = 2
DEFAULT_TARGET = 1200.0


def predict_without_self(query, history, settings, k=5):
    """Predict without letting a historical row predict itself."""
    reference = [
        row for row in history
        if row["row_number"] != query["row_number"]
    ]

    if not reference:
        raise ValueError("No independent historical neighbors")

    return predict_performance(query, reference, settings, k)


def valid_alternatives(source, candidates, history, settings, k=5):
    """Score observed alternatives by edits and predicted performance."""
    if not settings:
        raise ValueError("No adjustable settings")

    source_prediction = predict_without_self(
        source, history, settings, k
    )["prediction"]

    seen = set()
    alternatives = []

    for candidate in candidates:
        row_id = candidate["row_number"]

        if row_id in seen:
            raise ValueError("Duplicate candidate identifier")
        seen.add(row_id)

        if row_id == source["row_number"]:
            continue

        # These rows must be original observed configurations.
        # The constant setting cannot be adjusted.
        if candidate["base"] != source["base"]:
            continue

        changes = differing_settings(source, candidate, settings)
        if not changes:
            continue

        estimate = predict_without_self(
            candidate, history, settings, k
        )

        alternatives.append({
            "row_number": row_id,
            "changes": len(changes),
            "setting_changes": [
                {
                    "setting": name,
                    "from": source[name],
                    "to": candidate[name],
                }
                for name in changes
            ],
            "prediction": estimate["prediction"],
            "neighbor_sd": estimate["neighbor_sd"],
            "predicted_improvement": (
                source_prediction - estimate["prediction"]
            ),
        })

    return source_prediction, alternatives


def recommend_small_repair(
    source, candidates, history, settings,
    max_changes=MAX_SMALL_CHANGES, k=5
):
    """Prefer the fewest edits among predicted improvements."""
    if max_changes < 1:
        raise ValueError("max_changes must be positive")

    source_prediction, alternatives = valid_alternatives(
        source, candidates, history, settings, k
    )

    eligible = [
        item for item in alternatives
        if item["changes"] <= max_changes
        and item["predicted_improvement"] > 1e-9
    ]

    eligible.sort(
        key=lambda item: (
            item["changes"],
            item["prediction"],
            item["row_number"],
        )
    )

    return {
        "source_prediction": source_prediction,
        "recommendation": eligible[0] if eligible else None,
        "eligible_count": len(eligible),
    }


def find_minimum_repair(
    source, candidates, history, settings,
    target, k=5
):
    """Find the fewest edits needed to meet a predicted target."""
    if not math.isfinite(target):
        raise ValueError("Target must be finite")

    source_prediction, alternatives = valid_alternatives(
        source, candidates, history, settings, k
    )

    eligible = [
        item for item in alternatives
        if item["prediction"] <= target
    ]

    eligible.sort(
        key=lambda item: (
            item["changes"],
            item["prediction"],
            item["row_number"],
        )
    )

    return {
        "target": target,
        "source_prediction": source_prediction,
        "recommendation": eligible[0] if eligible else None,
        "eligible_count": len(eligible),
    }


def print_repair(result):
    print(f"Source predicted performance: "
          f"{result['source_prediction']:.2f}")

    recommendation = result["recommendation"]

    if recommendation is None:
        print("No qualifying historically observed alternative found.")
        return

    print(f"Recommended CSV row: {recommendation['row_number']}")
    print(f"Setting changes: {recommendation['changes']}")

    for change in recommendation["setting_changes"]:
        print(
            f"  {change['setting']}: "
            f"{change['from']} -> {change['to']}"
        )

    print(
        f"Alternative predicted performance: "
        f"{recommendation['prediction']:.2f}"
    )
    print(
        f"Predicted improvement: "
        f"{recommendation['predicted_improvement']:+.2f}"
    )
    print(
        f"Alternative neighbor SD: "
        f"{recommendation['neighbor_sd']:.2f}"
    )


def main():
    rows = load_configurations()
    settings = adjustable_settings(rows)
    history, held_out = train_test_split(rows)

    # This source has already been benchmarked: its outcome
    # is known to be poor. Other held-out outcomes remain hidden
    # from the repair recommendation algorithm.
    source = max(
        held_out,
        key=lambda row: (row[GOAL], -row["row_number"])
    )

    print("APACHE T7 — STEP 8: FIX")
    print(f"\nHistorical training records: {len(history)}")
    print(f"Held-out records: {len(held_out)}")
    print(f"Observed valid configurations: {len(rows)}")
    print(f"Source row: {source['row_number']}")
    print(f"Source measured performance: {source[GOAL]:.2f}")
    print("Only the eight adjustable settings may change.")

    print("\nAPACHE-15: SMALL PREDICTED REPAIR")
    small = recommend_small_repair(
        source, rows, history, settings
    )
    print(f"Maximum setting changes: {MAX_SMALL_CHANGES}")
    print(f"Qualifying alternatives: {small['eligible_count']}")
    print_repair(small)

    print("\nAPACHE-16: MINIMUM CHANGES TO TARGET")
    target_result = find_minimum_repair(
        source, rows, history, settings,
        target=DEFAULT_TARGET
    )
    print(f"Specified target: {DEFAULT_TARGET:.2f}")
    print(
        f"Predicted-to-meet-target alternatives: "
        f"{target_result['eligible_count']}"
    )
    print_repair(target_result)

    print(
        "\nRepairs are selected only from observed configurations."
        "\nTraining examples exclude themselves during prediction."
        "\nPredicted improvements are not performance guarantees."
        "\nThe target is a demonstration parameter."
    )


if __name__ == "__main__":
    main()
