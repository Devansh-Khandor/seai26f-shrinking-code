
"""NASA T7 - Step 9: Spot (NASA-17 and NASA-18).

Identify unusual input profiles and unexpectedly large
prediction errors in historical NASA projects.
"""

from remember import load_projects
from guess import get_features, distance, nearest_neighbors, predict


K = 5


def other_projects(project, projects):
    """Exclude the current project from its own comparison set."""
    return [
        row for row in projects
        if row["row_number"] != project["row_number"]
    ]


def input_anomalies(projects, features):
    """Rank projects by distance from their nearest neighbors."""
    results = []

    for project in projects:
        others = other_projects(project, projects)

        neighbors = nearest_neighbors(
            project, others, features, k=K
        )

        distances = [
            distance(project, neighbor, features)
            for neighbor in neighbors
        ]

        score = sum(distances) / len(distances)

        results.append({
            "row_number": project["row_number"],
            "score": score,
            "neighbors": [
                neighbor["row_number"] for neighbor in neighbors
            ],
        })

    results.sort(
        key=lambda result: (-result["score"], result["row_number"])
    )

    return results


def outcome_anomalies(projects, features, goal):
    """Find records with unusually large prediction errors."""
    results = []

    for project in projects:
        others = other_projects(project, projects)

        # Predict using other projects, never the row itself.
        prediction = predict(
            project, others, features, goal, k=K
        )["estimate"]

        actual = project[goal]
        residual = actual - prediction

        results.append({
            "row_number": project["row_number"],
            "actual": actual,
            "predicted": prediction,
            "residual": residual,
            "absolute_error": abs(residual),
        })

    results.sort(
        key=lambda result: (
            -result["absolute_error"],
            result["row_number"],
        )
    )

    return results


def print_input_anomalies(results, count=5):
    print("\nNASA-17: UNUSUAL PROJECT CHARACTERISTICS")
    print("Higher average X-distance means a more unusual profile.")

    print(
        f"{'Row':>5} {'Isolation':>12} "
        f"{'Nearest historical rows':>28}"
    )

    for result in results[:count]:
        neighbors = ", ".join(
            str(row) for row in result["neighbors"]
        )

        print(
            f"{result['row_number']:>5}"
            f" {result['score']:>12.4f}"
            f" {neighbors:>28}"
        )


def print_outcome_anomalies(results, goal, count=5):
    print(f"\nNASA-18: UNUSUAL {goal} OUTCOMES")
    print("Residual = actual - predicted")

    print(
        f"{'Row':>5} {'Actual':>12}"
        f" {'Predicted':>12} {'Residual':>12}"
        f" {'Abs. Error':>12}"
    )

    for result in results[:count]:
        print(
            f"{result['row_number']:>5}"
            f" {result['actual']:>12.2f}"
            f" {result['predicted']:>12.2f}"
            f" {result['residual']:>+12.2f}"
            f" {result['absolute_error']:>12.2f}"
        )


def main():
    projects = load_projects()
    features = get_features(projects)

    print("NASA T7 - STEP 9: SPOT")
    print(f"Historical projects: {len(projects)}")
    print(f"Input features: {len(features)}")
    print(f"Nearest neighbors: {K}")

    # NASA-17: unusual combinations of input characteristics.
    unusual_inputs = input_anomalies(projects, features)
    print_input_anomalies(unusual_inputs)

    # NASA-18: unexpectedly high or low recorded outcomes.
    for goal in ("Effort-", "Defects-"):
        unusual_outcomes = outcome_anomalies(
            projects, features, goal
        )
        print_outcome_anomalies(unusual_outcomes, goal)

    print("\nIMPORTANT LIMITATIONS")
    print("- An anomaly is not necessarily a data error.")
    print("- Large residuals may reflect genuine project variation.")
    print("- Predictions exclude the project being evaluated.")
    print("- Results are exploratory and depend on the kNN model.")


if __name__ == "__main__":
    main()
