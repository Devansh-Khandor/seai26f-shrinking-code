
"""NASA T7 - Step 8: Fix (NASA-15 and NASA-16).

Explore hypothetical changes to adjustable project settings.
These predictions are not evidence of causal improvements.
"""

from itertools import combinations, product

from remember import load_projects
from guess import get_features, predict


ADJUSTABLE = ("docu", "tool", "site", "sced")
MIN_IMPROVEMENT = 0.05
MAX_EFFORT_INCREASE = 0.05


def predict_outcomes(proposal, history, features):
    """Predict effort and defects using the existing kNN model."""
    return {
        goal: predict(proposal, history, features, goal)["estimate"]
        for goal in ("Effort-", "Defects-")
    }


def possible_values(history):
    """Use only settings that have appeared in historical projects."""
    return {
        feature: sorted({row[feature] for row in history})
        for feature in ADJUSTABLE
    }


def candidate_changes(proposal, values, max_changes=2):
    """Generate hypothetical proposals with one or two changed settings."""
    for count in range(1, max_changes + 1):
        for features_to_change in combinations(ADJUSTABLE, count):
            choices = [
                [value for value in values[feature]
                 if value != proposal[feature]]
                for feature in features_to_change
            ]

            for selected_values in product(*choices):
                candidate = proposal.copy()

                for feature, value in zip(
                    features_to_change, selected_values
                ):
                    candidate[feature] = value

                changes = {
                    feature: (proposal[feature], candidate[feature])
                    for feature in features_to_change
                }

                yield candidate, changes


def find_repair(
    proposal,
    history,
    features,
    target,
    max_effort_increase=None,
):
    """Find the best qualifying repair using the fewest setting changes."""
    baseline = predict_outcomes(proposal, history, features)
    values = possible_values(history)

    tested = 0
    passing = []

    for candidate, changes in candidate_changes(proposal, values):
        tested += 1
        predicted = predict_outcomes(candidate, history, features)

        original = baseline[target]
        improvement = (original - predicted[target]) / original

        if improvement < MIN_IMPROVEMENT:
            continue

        if max_effort_increase is not None:
            effort_limit = (
                baseline["Effort-"] * (1 + max_effort_increase)
            )
            if predicted["Effort-"] > effort_limit:
                continue

        passing.append({
            "changes": changes,
            "predicted": predicted,
            "improvement": improvement,
        })

    # Prioritize fewer changes, then greater improvement.
    passing.sort(
        key=lambda result: (
            len(result["changes"]),
            -result["improvement"],
        )
    )

    return baseline, passing[0] if passing else None, tested


def highest_risk_proposal(proposals, history, features, goal):
    """Select a proposal using predicted outcomes, never hidden goals."""
    return max(
        proposals,
        key=lambda project: (
            predict(project, history, features, goal)["estimate"],
            -project["row_number"],
        ),
    )


def print_repair(title, proposal, history, features, target, limit=None):
    print(f"\n{title}")
    print(f"Selected proposal: row {proposal['row_number']}")

    baseline, repair, tested = find_repair(
        proposal, history, features, target, limit
    )

    print(f"Original predicted effort: {baseline['Effort-']:.2f}")
    print(f"Original predicted defects: {baseline['Defects-']:.2f}")
    print(f"Hypothetical alternatives tested: {tested}")

    if repair is None:
        print("No qualifying improvement found.")
        print("The available historical evidence does not support")
        print("a recommendation under the chosen thresholds.")
        return

    print(f"\nSettings changed: {len(repair['changes'])}")
    for feature, (old, new) in repair["changes"].items():
        print(f"  {feature}: {old} -> {new}")

    new = repair["predicted"]
    print(f"\nNew predicted effort: {new['Effort-']:.2f}")
    print(f"New predicted defects: {new['Defects-']:.2f}")
    print(
        f"Target improvement: {repair['improvement'] * 100:.1f}%"
    )

    effort_change = (
        (new["Effort-"] - baseline["Effort-"])
        / baseline["Effort-"] * 100
    )
    defect_change = (
        (new["Defects-"] - baseline["Defects-"])
        / baseline["Defects-"] * 100
    )

    print(f"Predicted effort change: {effort_change:+.1f}%")
    print(f"Predicted defect change: {defect_change:+.1f}%")


def main():
    projects = load_projects()
    features = get_features(projects)

    history = projects[:74]
    proposals = projects[74:]

    # Remove hidden outcomes from incoming proposals.
    masked = [
        {
            key: value
            for key, value in row.items()
            if key not in ("Kloc+", "Effort-", "Defects-")
        }
        for row in proposals
    ]

    print("NASA T7 - STEP 8: FIX")
    print(f"Training projects: {len(history)}")
    print(f"Candidate proposals: {len(masked)}")
    print(f"Adjustable settings: {', '.join(ADJUSTABLE)}")

    effort_risk = highest_risk_proposal(
        masked, history, features, "Effort-"
    )

    defect_risk = highest_risk_proposal(
        masked, history, features, "Defects-"
    )

    print_repair(
        "NASA-15: REDUCE PREDICTED EFFORT",
        effort_risk, history, features, "Effort-"
    )

    print_repair(
        "NASA-16: REDUCE PREDICTED DEFECTS",
        defect_risk, history, features, "Defects-",
        MAX_EFFORT_INCREASE
    )

    print("\nIMPORTANT LIMITATIONS")
    print("- Changes are hypothetical, not proven interventions.")
    print("- Only previously observed individual setting values are tried.")
    print("- Combinations are not guaranteed to have existed historically.")
    print("- Model predictions can be inaccurate.")
    print("- No improvement is guaranteed in real-world projects.")


if __name__ == "__main__":
    main()
