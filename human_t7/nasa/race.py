
"""NASA T7 - Step 10: Race (NASA-19 and NASA-20).

Compare kNN prediction with the historical mean, and
model-guided selection with random selection.
"""

import random
import statistics

from remember import load_projects, goal_bounds
from guess import get_features, evaluate, predict
from choose import (
    mask_proposals,
    select_candidates,
    evaluate_budget,
)


BUDGETS = (3, 5, 10)
RANDOM_TRIALS = 30
SEED = 42


def prediction_race(history, testing, features):
    """Compare kNN against the training-mean baseline."""
    print("\nNASA-19: PREDICTION BASELINE COMPARISON")
    print(f"{'Goal':<12} {'kNN MAE':>12} {'Mean MAE':>12}")

    for goal in ("Effort-", "Defects-"):
        model_mae, mean_mae = evaluate(
            history, testing, features, goal
        )

        print(
            f"{goal:<12}"
            f" {model_mae:>12.2f}"
            f" {mean_mae:>12.2f}"
        )

        if model_mae < mean_mae:
            print("  kNN has lower average error.")
        elif model_mae > mean_mae:
            print("  Historical mean has lower average error.")
        else:
            print("  Both methods have equal average error.")

    # The fraction of held-out projects on which
    # kNN makes the smaller absolute error.
    for goal in ("Effort-", "Defects-"):
        baseline = statistics.mean(
            project[goal] for project in history
        )

        wins = 0
        ties = 0

        for project in testing:
            prediction = predict(
                project, history, features, goal
            )["estimate"]

            model_error = abs(project[goal] - prediction)
            mean_error = abs(project[goal] - baseline)

            if model_error < mean_error:
                wins += 1
            elif model_error == mean_error:
                ties += 1

        print(
            f"{goal}: kNN lower error on {wins}/{len(testing)} "
            f"projects; ties={ties}"
        )


def random_rankings(candidate_ids, trials=RANDOM_TRIALS, seed=SEED):
    """Generate independent seeded random candidate orders."""
    rng = random.Random(seed)
    orders = []

    for _ in range(trials):
        shuffled = list(candidate_ids)
        rng.shuffle(shuffled)

        orders.append([
            {"row_number": row_number}
            for row_number in shuffled
        ])

    return orders


def selection_race(history, testing, features):
    """Compare model-guided and random budget-limited inspection."""
    masked = mask_proposals(testing, features)
    model_ranking = select_candidates(
        masked, history, features
    )

    # Only the evaluation stage can access observed goals.
    observed = {
        project["row_number"]: project
        for project in testing
    }

    bounds = goal_bounds(history)

    random_orders = random_rankings(
        [project["row_number"] for project in masked]
    )

    print("\nNASA-20: MODEL-GUIDED VS RANDOM SELECTION")
    print(f"Candidate proposals: {len(masked)}")
    print(f"Random trials: {len(random_orders)}")
    print(f"Random seed: {SEED}")
    print("Lower observed d2h is better.")

    print(
        f"\n{'Budget':>6} {'Model d2h':>12}"
        f" {'Random mean':>13} {'Random best':>13}"
        f" {'Random <= model':>17}"
    )

    results = []

    for budget in BUDGETS:
        model_best = evaluate_budget(
            model_ranking, observed, bounds, budget
        )

        random_scores = [
            evaluate_budget(
                order, observed, bounds, budget
            )["d2h"]
            for order in random_orders
        ]

        random_mean = statistics.mean(random_scores)
        random_best = min(random_scores)

        # Number of random trials that matched or beat the model.
        random_wins = sum(
            score <= model_best["d2h"] + 1e-12
            for score in random_scores
        )

        print(
            f"{budget:>6}"
            f" {model_best['d2h']:>12.4f}"
            f" {random_mean:>13.4f}"
            f" {random_best:>13.4f}"
            f" {random_wins:>13}/{len(random_scores)}"
        )

        results.append({
            "budget": budget,
            "model_row": model_best["row_number"],
            "model_d2h": model_best["d2h"],
            "random_mean": random_mean,
            "random_best": random_best,
            "random_wins": random_wins,
            "random_scores": random_scores,
        })

    return model_ranking, random_orders, results


def main():
    projects = load_projects()
    features = get_features(projects)

    history = projects[:74]
    testing = projects[74:]

    print("NASA T7 - STEP 10: RACE")
    print(f"Historical training projects: {len(history)}")
    print(f"Held-out testing projects: {len(testing)}")
    print(f"Input features: {len(features)}")

    prediction_race(history, testing, features)

    _, _, results = selection_race(
        history, testing, features
    )

    print("\nINTERPRETATION")
    for result in results:
        budget = result["budget"]

        if result["model_d2h"] < result["random_mean"]:
            print(
                f"Budget {budget}: model beats the "
                "average random outcome."
            )
        else:
            print(
                f"Budget {budget}: model does not beat "
                "the average random outcome."
            )

    print("\nLIMITATIONS")
    print("- The test set contains only 19 projects.")
    print("- Thirty random trials estimate a comparison distribution.")
    print("- MAE and random comparisons are descriptive here.")
    print("- This does not establish statistical significance.")
    print("- No held-out goals are used for candidate ranking.")


if __name__ == "__main__":
    main()
