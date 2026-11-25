
"""NASA T7 - Step 7: Choose (NASA-13 and NASA-14)."""

from remember import (
    load_projects,
    rank_projects,
    goal_bounds,
    distance_to_heaven,
)
from guess import get_features, predict


GOALS = ("Kloc+", "Effort-", "Defects-")
BUDGETS = (3, 5, 10)


def promising_history(history, features, count=5):
    """Find favorable observed projects and their characteristics."""
    ranked = rank_projects(history)

    print("\nNASA-13: PROMISING HISTORICAL PROJECTS")
    print("These are observed projects, not hypothetical changes.")
    print(
        f"{'Row':>5} {'KLOC':>9} {'Effort':>10}"
        f" {'Defects':>10} {'d2h':>9}"
    )

    for project in ranked[:count]:
        print(
            f"{project['row_number']:>5}"
            f" {project['Kloc+']:>9.1f}"
            f" {project['Effort-']:>10.1f}"
            f" {project['Defects-']:>10.0f}"
            f" {project['d2h']:>9.4f}"
        )

        settings = ", ".join(
            f"{feature}={project[feature]}"
            for feature in features
        )
        print(f"      Characteristics: {settings}")

    return ranked[:count]


def mask_proposals(proposals, features):
    """Keep only information known before evaluation."""
    return [
        {
            "row_number": project["row_number"],
            **{
                feature: project[feature]
                for feature in features
            },
        }
        for project in proposals
    ]


def select_candidates(proposals, history, features):
    """Prioritize proposals using predicted goals only."""
    bounds = goal_bounds(history)
    ranked = []

    for proposal in proposals:
        estimates = {
            goal: predict(
                proposal, history, features, goal
            )["estimate"]
            for goal in GOALS
        }

        estimated_score = distance_to_heaven(
            estimates, bounds
        )

        ranked.append({
            "row_number": proposal["row_number"],
            "predictions": estimates,
            "predicted_d2h": estimated_score,
        })

    ranked.sort(
        key=lambda result: (
            result["predicted_d2h"],
            result["row_number"],
        )
    )

    return ranked


def evaluate_budget(ranking, observed, bounds, budget):
    """Reveal outcomes only for proposals already selected."""
    inspected = ranking[:budget]
    best = None

    for candidate in inspected:
        project = observed[candidate["row_number"]]
        actual_score = distance_to_heaven(
            project, bounds
        )

        if best is None or actual_score < best["d2h"]:
            best = {
                "row_number": project["row_number"],
                "d2h": actual_score,
                "project": project,
            }

    return best


def main():
    projects = load_projects()
    features = get_features(projects)

    history = projects[:74]
    held_out = projects[74:]

    print("NASA T7 - STEP 7: CHOOSE")
    print(f"Historical projects: {len(history)}")
    print(f"Candidate proposals: {len(held_out)}")
    print(f"Input features: {len(features)}")

    promising_history(history, features)

    # Hide all actual goal outcomes before ranking proposals.
    proposals = mask_proposals(held_out, features)
    ranking = select_candidates(
        proposals, history, features
    )

    print("\nNASA-14: BUDGET-LIMITED SELECTION")
    print("Proposals ranked by predicted distance-to-heaven.")
    print(
        f"{'Rank':>5} {'Row':>5}"
        f" {'Predicted d2h':>15}"
    )

    for position, candidate in enumerate(
        ranking, start=1
    ):
        print(
            f"{position:>5}"
            f" {candidate['row_number']:>5}"
            f" {candidate['predicted_d2h']:>15.4f}"
        )

    # Actual outcomes are accessed only for scoring inspections.
    observed = {
        project["row_number"]: project
        for project in held_out
    }

    bounds = goal_bounds(history)

    print("\nRESULTS AFTER EVALUATING SELECTED PROJECTS")
    print(
        f"{'Budget':>7} {'Best row':>10}"
        f" {'Observed d2h':>15}"
    )

    for budget in BUDGETS:
        best = evaluate_budget(
            ranking, observed, bounds, budget
        )
        print(
            f"{budget:>7}"
            f" {best['row_number']:>10}"
            f" {best['d2h']:>15.4f}"
        )

    print("\nLower distance-to-heaven is more favorable.")
    print(
        "Actual proposal outcomes were not used "
        "to choose the inspection order."
    )
    print(
        "Scores use bounds learned from the "
        "74 historical projects."
    )


if __name__ == "__main__":
    main()
