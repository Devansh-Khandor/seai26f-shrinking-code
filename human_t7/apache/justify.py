"""Apache T7, Step 6: Justify (APACHE-11 and APACHE-12)."""

import math

from remember import (
    GOAL,
    adjustable_settings,
    load_configurations,
)
from guess import (
    K,
    configuration_distance,
    differing_settings,
    nearest_neighbors,
    one_setting_pairs,
    predict_performance,
    train_test_split,
)


def explain_prediction(query, history, settings, k=K):
    """Explain a prediction using comparable historical records."""
    outcome = predict_performance(query, history, settings, k)
    neighbors = nearest_neighbors(query, history, settings, k)

    evidence = []

    for neighbor in neighbors:
        different = differing_settings(query, neighbor, settings)

        evidence.append({
            "row_number": neighbor["row_number"],
            "performance": neighbor[GOAL],
            "distance": configuration_distance(
                query, neighbor, settings
            ),
            "differing_settings": different,
        })

    return {
        "row_number": query["row_number"],
        "settings": {
            name: query[name] for name in settings
        },
        "prediction": outcome["prediction"],
        "neighbor_min": outcome["neighbor_min"],
        "neighbor_max": outcome["neighbor_max"],
        "neighbor_sd": outcome["neighbor_sd"],
        "evidence": evidence,
    }


def recommend_configuration(candidates, history, settings, k=K):
    """Choose lowest predicted performance without inspecting targets."""
    if not candidates:
        raise ValueError("Candidates cannot be empty")

    ranked = []

    for candidate in candidates:
        explanation = explain_prediction(
            candidate, history, settings, k
        )
        ranked.append((explanation["prediction"],
                       candidate["row_number"],
                       candidate, explanation))

    _, _, selected, explanation = min(
        ranked, key=lambda item: item[:2]
    )

    return selected, explanation


def compare_configurations(a, b, history, settings, k=K):
    """Explain a predicted difference and identify setting changes."""
    if a["row_number"] == b["row_number"]:
        raise ValueError("Cannot compare a record with itself")

    explanation_a = explain_prediction(a, history, settings, k)
    explanation_b = explain_prediction(b, history, settings, k)

    differences = [
        {
            "setting": name,
            "a_value": a[name],
            "b_value": b[name],
        }
        for name in differing_settings(a, b, settings)
    ]

    return {
        "row_a": a["row_number"],
        "row_b": b["row_number"],
        "differences": differences,
        "prediction_a": explanation_a,
        "prediction_b": explanation_b,
        "predicted_difference": (
            explanation_b["prediction"]
            - explanation_a["prediction"]
        ),
    }


def validate_explanation(explanation, history, settings, k=K):
    """Check evidence is from history and prediction is reproducible."""
    history_by_id = {
        row["row_number"]: row for row in history
    }

    evidence = explanation["evidence"]
    if len(evidence) != min(k, len(history)):
        return False

    values = []

    for item in evidence:
        row = history_by_id.get(item["row_number"])
        if row is None:
            return False

        values.append(row[GOAL])

        if not math.isclose(item["performance"], row[GOAL]):
            return False

        if item["differing_settings"] != [
            name for name in settings
            if explanation["settings"][name] != row[name]
        ]:
            return False

        distance = sum(
            explanation["settings"][name] != row[name]
            for name in settings
        ) / len(settings)

        if not math.isclose(distance, item["distance"]):
            return False

    return math.isclose(
        explanation["prediction"], sum(values) / len(values)
    )


def print_explanation(explanation):
    print(f"Configuration row: {explanation['row_number']}")
    print(f"Predicted performance: {explanation['prediction']:.2f}")

    print("Settings:")
    for name, value in explanation["settings"].items():
        print(f"  {name}={value}")

    print("\nComparable historical configurations:")
    print(
        f"{'Row':>5} {'Performance':>12} {'Distance':>10}"
        "  Differing settings"
    )

    for item in explanation["evidence"]:
        differences = ", ".join(item["differing_settings"]) or "none"
        print(
            f"{item['row_number']:>5}"
            f" {item['performance']:>12.2f}"
            f" {item['distance']:>10.3f}"
            f"  {differences}"
        )

    print(
        f"Neighbor performance range: "
        f"{explanation['neighbor_min']:.2f} to "
        f"{explanation['neighbor_max']:.2f}"
    )
    print(
        f"Neighbor SD: {explanation['neighbor_sd']:.2f}"
    )


def main():
    configurations = load_configurations()
    settings = adjustable_settings(configurations)
    history, candidates = train_test_split(configurations)

    history_ids = {row["row_number"] for row in history}
    candidate_ids = {row["row_number"] for row in candidates}

    assert not history_ids & candidate_ids

    print("APACHE T7 — STEP 6: JUSTIFY")
    print(f"\nHistorical training configurations: {len(history)}")
    print(f"Held-out candidate configurations: {len(candidates)}")

    print("\nAPACHE-11: EXPLAIN A RECOMMENDED CONFIGURATION")

    selected, explanation = recommend_configuration(
        candidates, history, settings
    )

    assert validate_explanation(
        explanation, history, settings
    )

    print("Selection rule: lowest predicted performance")
    print_explanation(explanation)

    # The actual outcome is revealed only after selection.
    print(
        f"\nRetrospective measured performance: "
        f"{selected[GOAL]:.2f}"
    )
    print(
        f"Prediction error: "
        f"{explanation['prediction'] - selected[GOAL]:+.2f}"
    )

    print("\nAPACHE-12: EXPLAIN A ONE-SETTING DIFFERENCE")

    pairs = one_setting_pairs(candidates, settings)

    if not pairs:
        raise AssertionError(
            "No one-setting pair in held-out candidates"
        )

    a, b, changed_setting = pairs[0]
    comparison = compare_configurations(
        a, b, history, settings
    )

    assert validate_explanation(
        comparison["prediction_a"], history, settings
    )
    assert validate_explanation(
        comparison["prediction_b"], history, settings
    )

    print(f"Configuration A: row {comparison['row_a']}")
    print(f"Configuration B: row {comparison['row_b']}")

    for change in comparison["differences"]:
        print(
            f"Different setting: {change['setting']} "
            f"({change['a_value']} -> {change['b_value']})"
        )

    print(
        f"Predicted performance A: "
        f"{comparison['prediction_a']['prediction']:.2f}"
    )
    print(
        f"Predicted performance B: "
        f"{comparison['prediction_b']['prediction']:.2f}"
    )
    print(
        f"Predicted difference (B - A): "
        f"{comparison['predicted_difference']:+.2f}"
    )

    print("\nEvidence for configuration A:")
    print_explanation(comparison["prediction_a"])

    print("\nEvidence for configuration B:")
    print_explanation(comparison["prediction_b"])

    print(
        f"\nRetrospective recorded difference (B - A): "
        f"{b[GOAL] - a[GOAL]:+.2f}"
    )
    print(
        "\nExplanations reflect historical similarity and "
        "predicted outcomes, not proven causal effects."
    )


if __name__ == "__main__":
    main()
