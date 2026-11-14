
"""NASA T7 - Step 2: Guess (NASA-03 and NASA-04)."""

import statistics
from remember import load_projects

GOALS = ("Kloc+", "Effort-", "Defects-")
K = 5


def get_features(projects):
    """Find the input columns, excluding goals and ignored fields."""
    columns = projects[0].keys()

    return [
        name for name in columns
        if name not in GOALS
        and name != "row_number"
        and not name.endswith("X")
    ]


def distance(a, b, features):
    """Fraction of categorical features that differ."""
    different = sum(a[name] != b[name] for name in features)
    return different / len(features)


def nearest_neighbors(proposal, training, features, k=K):
    """Find k historically similar projects."""
    ordered = sorted(
        training,
        key=lambda row: (
            distance(proposal, row, features),
            row["row_number"],
        ),
    )
    return ordered[:k]


def predict(proposal, training, features, goal, k=K):
    """Predict a goal from the k nearest historical projects."""
    neighbors = nearest_neighbors(proposal, training, features, k)
    values = [row[goal] for row in neighbors]

    return {
        "estimate": statistics.mean(values),
        "low": min(values),
        "high": max(values),
        "neighbors": neighbors,
    }


def evaluate(training, testing, features, goal):
    """Evaluate kNN against a simple training-mean baseline."""
    actual = []
    predicted = []
    baseline = statistics.mean(row[goal] for row in training)

    for project in testing:
        result = predict(project, training, features, goal)
        actual.append(project[goal])
        predicted.append(result["estimate"])

    model_mae = statistics.mean(
        abs(a - p) for a, p in zip(actual, predicted)
    )
    baseline_mae = statistics.mean(
        abs(a - baseline) for a in actual
    )

    return model_mae, baseline_mae


def main():
    projects = load_projects()
    features = get_features(projects)

    training = projects[:74]
    testing = projects[74:]

    print("NASA T7 - STEP 2: GUESS")
    print(f"Training projects: {len(training)}")
    print(f"Testing projects:  {len(testing)}")
    print(f"Input features:    {len(features)}")
    print(f"Nearest neighbors: {K}")

    # The first held-out project acts as a new project proposal.
    # Only its X columns are used by the predictor.
    proposal = testing[0]

    print("\nNASA-03: DEVELOPMENT EFFORT ESTIMATE")
    effort = predict(proposal, training, features, "Effort-")

    print(f"Proposal row: {proposal['row_number']}")
    print(f"Predicted effort: {effort['estimate']:.2f}")
    print(
        f"Neighbor outcome range: "
        f"{effort['low']:.2f} to {effort['high']:.2f}"
    )

    print("\nNASA-04: DEFECT ESTIMATE AND UNCERTAINTY")
    defects = predict(proposal, training, features, "Defects-")

    print(f"Predicted defects: {defects['estimate']:.2f}")
    print(
        f"Neighbor outcome range: "
        f"{defects['low']:.2f} to {defects['high']:.2f}"
    )
    print("Note: Range is descriptive, not a confidence interval.")

    print("\nFIVE MOST SIMILAR HISTORICAL PROJECTS")
    for row in effort["neighbors"]:
        d = distance(proposal, row, features)
        print(
            f"Row {row['row_number']:>2}: "
            f"distance={d:.3f}, "
            f"effort={row['Effort-']:.1f}, "
            f"defects={row['Defects-']:.0f}"
        )

    print("\nHELD-OUT PREDICTION EVALUATION")
    print(f"{'Goal':<12} {'kNN MAE':>12} {'Mean MAE':>12}")

    for goal in ("Effort-", "Defects-"):
        model_mae, baseline_mae = evaluate(
            training, testing, features, goal
        )
        print(
            f"{goal:<12} "
            f"{model_mae:>12.2f} "
            f"{baseline_mae:>12.2f}"
        )


if __name__ == "__main__":
    main()
