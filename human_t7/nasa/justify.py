
"""NASA T7 - Step 6: Justify (NASA-11 and NASA-12)."""

import statistics

from remember import load_projects
from guess import get_features, distance, nearest_neighbors, predict

K = 5


def explain_effort(proposal, history, features):
    """Explain an effort prediction with similar historical projects."""
    result = predict(proposal, history, features, "Effort-", k=K)

    print(f"\nNASA-11: EFFORT EXPLANATION FOR ROW {proposal['row_number']}")
    print(f"Predicted effort: {result['estimate']:.2f}")
    print(f"Neighbor outcome range: {result['low']:.2f} - {result['high']:.2f}")

    print("\nSupporting historical projects:")
    print(f"{'Row':>5} {'Distance':>10} {'Effort':>12} {'Defects':>12}")

    for neighbor in result["neighbors"]:
        print(
            f"{neighbor['row_number']:>5}"
            f" {distance(proposal, neighbor, features):>10.3f}"
            f" {neighbor['Effort-']:>12.2f}"
            f" {neighbor['Defects-']:>12.0f}"
        )

    print(
        "\nExplanation: The estimate is the average effort "
        "of these five similar historical projects."
    )
    print("Similarity is measured using the 22 input characteristics.")

    return result


def risk_estimates(proposal, history, features):
    """Predict effort and defects without accessing actual outcomes."""
    effort = predict(proposal, history, features, "Effort-")
    defects = predict(proposal, history, features, "Defects-")

    return {
        "effort": effort["estimate"],
        "defects": defects["estimate"],
    }


def different_features(first, second, features):
    """List the input characteristics that differ."""
    return [
        (feature, first[feature], second[feature])
        for feature in features
        if first[feature] != second[feature]
    ]


def compare_proposals(first, second, history, features):
    """Explain predicted differences and observed input tradeoffs."""
    a = risk_estimates(first, history, features)
    b = risk_estimates(second, history, features)

    print("\nNASA-12: COMPARING TWO PROJECT PROPOSALS")
    print(
        f"{'Prediction':<20}"
        f" {'Row ' + str(first['row_number']):>14}"
        f" {'Row ' + str(second['row_number']):>14}"
    )
    print(f"{'Effort':<20} {a['effort']:>14.2f} {b['effort']:>14.2f}")
    print(f"{'Defects':<20} {a['defects']:>14.2f} {b['defects']:>14.2f}")

    effort_better = first if a["effort"] < b["effort"] else second
    defects_better = first if a["defects"] < b["defects"] else second

    print(
        f"\nLower predicted effort: "
        f"row {effort_better['row_number']}"
    )
    print(
        f"Lower predicted defects: "
        f"row {defects_better['row_number']}"
    )

    if effort_better["row_number"] != defects_better["row_number"]:
        print("Tradeoff: Neither proposal is better on both outcomes.")
    else:
        print("One proposal is better on both predicted outcomes.")

    differences = different_features(first, second, features)

    print(f"\nInput characteristics that differ: {len(differences)}")
    for feature, first_value, second_value in differences:
        print(
            f"  {feature:<8} "
            f"{str(first_value):>4} vs {str(second_value):<4}"
        )

    print(
        "\nThese characteristics distinguish the proposals; "
        "this does not establish their individual causal effects."
    )

    return a, b, differences


def choose_proposals(proposals, history, features):
    """Find contrasting proposals using predictions only."""
    mean_effort = statistics.mean(p["Effort-"] for p in history)
    mean_defects = statistics.mean(p["Defects-"] for p in history)

    ranked = []

    for proposal in proposals:
        estimates = risk_estimates(proposal, history, features)

        # Both effort and defects are minimized.
        score = (
            estimates["effort"] / mean_effort
            + estimates["defects"] / mean_defects
        )

        ranked.append((score, proposal["row_number"], proposal))

    ranked.sort(key=lambda item: (item[0], item[1]))

    return ranked[0][2], ranked[-1][2]


def main():
    projects = load_projects()
    history = projects[:74]
    proposals = projects[74:]
    features = get_features(projects)

    print("NASA T7 - STEP 6: JUSTIFY")
    print(f"Historical training projects: {len(history)}")
    print(f"Held-out project proposals: {len(proposals)}")

    # First demonstrate the explanation for a single new proposal.
    explain_effort(proposals[0], history, features)

    # Select two contrasting proposals using predicted risk,
    # never using their hidden effort or defect outcomes.
    lower_risk, higher_risk = choose_proposals(
        proposals, history, features
    )

    compare_proposals(
        lower_risk, higher_risk, history, features
    )

    print("\nJustification demonstration completed.")


if __name__ == "__main__":
    main()
