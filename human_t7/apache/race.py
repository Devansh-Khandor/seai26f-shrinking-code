"""Apache T7, Step 10: Race (APACHE-19 and APACHE-20)."""

import math
import random
import statistics

from remember import GOAL, adjustable_settings, load_configurations
from guess import (
    evaluate_predictions,
    mean_absolute_error,
    train_test_split,
)
from choose import (
    BUDGETS,
    oracle_best,
    plan_inspections,
    simulate_inspections,
)

RANDOM_TRIALS = 30
SEED = 42
BOOTSTRAP_REPEATS = 2000


def random_inspection_trials(
    candidates, budgets=BUDGETS,
    trials=RANDOM_TRIALS, seed=SEED
):
    """Simulate independent seeded random inspection orders."""
    if not candidates or trials < 1:
        raise ValueError("Candidates and trials must be nonempty")

    ids = [row["row_number"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate identifiers")

    if not budgets or any(
        not 1 <= budget <= len(candidates)
        for budget in budgets
    ):
        raise ValueError("Invalid inspection budget")

    outcomes = {budget: [] for budget in budgets}

    for trial in range(trials):
        order = list(candidates)
        random.Random(seed + trial).shuffle(order)

        # Every trial evaluates all budgets using the SAME
        # randomized inspection order.
        running_best = math.inf
        best_at = {}

        for inspection, row in enumerate(order, start=1):
            running_best = min(running_best, row[GOAL])

            if inspection in outcomes:
                best_at[inspection] = running_best

        for budget in budgets:
            outcomes[budget].append(best_at[budget])

    return outcomes


def compare_selection(
    plan, candidates, budgets=BUDGETS,
    trials=RANDOM_TRIALS, seed=SEED
):
    """Compare model-guided and random selection at equal budgets."""
    random_outcomes = random_inspection_trials(
        candidates, budgets, trials, seed
    )

    comparisons = []

    for budget in budgets:
        guided = simulate_inspections(
            plan, candidates, budget
        )["best_performance"]

        values = random_outcomes[budget]

        comparisons.append({
            "budget": budget,
            "guided_best": guided,
            "random_mean": statistics.mean(values),
            "random_median": statistics.median(values),
            "random_sd": (
                statistics.stdev(values)
                if len(values) > 1 else 0.0
            ),
            "random_min": min(values),
            "random_max": max(values),
            "guided_wins": sum(guided < x for x in values),
            "ties": sum(guided == x for x in values),
            "random_wins": sum(guided > x for x in values),
            "random_values": values,
        })

    return comparisons


def paired_error_summary(actual, model_pred, baseline_pred):
    """Summarize paired errors on the same held-out observations."""
    if (
        not actual
        or len(actual) != len(model_pred)
        or len(actual) != len(baseline_pred)
    ):
        raise ValueError("Prediction lists must be aligned and nonempty")

    model_errors = [
        abs(a - p) for a, p in zip(actual, model_pred)
    ]
    baseline_errors = [
        abs(a - p) for a, p in zip(actual, baseline_pred)
    ]

    differences = [
        baseline - model
        for baseline, model in zip(
            baseline_errors, model_errors
        )
    ]

    model_mae = mean_absolute_error(actual, model_pred)
    baseline_mae = mean_absolute_error(actual, baseline_pred)

    return {
        "count": len(actual),
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "mae_improvement": baseline_mae - model_mae,
        "relative_improvement_pct": (
            100 * (baseline_mae - model_mae) / baseline_mae
            if baseline_mae > 0 else None
        ),
        "model_better": sum(d > 0 for d in differences),
        "equal": sum(d == 0 for d in differences),
        "baseline_better": sum(d < 0 for d in differences),
        "paired_improvements": differences,
    }


def bootstrap_mean_interval(
    values, repeats=BOOTSTRAP_REPEATS, seed=SEED
):
    """Exploratory paired bootstrap percentile interval for the mean."""
    if not values or repeats < 2:
        raise ValueError("Need values and at least two resamples")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("All values must be finite")

    rng = random.Random(seed)
    n = len(values)
    resampled_means = []

    for _ in range(repeats):
        sample = [
            values[rng.randrange(n)] for _ in range(n)
        ]
        resampled_means.append(statistics.mean(sample))

    resampled_means.sort()

    return (
        resampled_means[
            math.floor(0.025 * (repeats - 1))
        ],
        resampled_means[
            math.ceil(0.975 * (repeats - 1))
        ],
    )


def evaluate_prediction_race(history, test, settings):
    """Compare kNN against the training-mean predictor."""
    evaluation = evaluate_predictions(
        history, test, settings
    )

    actual = evaluation["actual"]
    model_pred = evaluation["predicted"]
    baseline_pred = [
        evaluation["baseline_prediction"]
    ] * len(actual)

    result = paired_error_summary(
        actual, model_pred, baseline_pred
    )

    result["bootstrap_interval"] = bootstrap_mean_interval(
        result["paired_improvements"]
    )

    return result


def main():
    rows = load_configurations()
    settings = adjustable_settings(rows)
    history, test = train_test_split(rows)

    print("APACHE T7 — STEP 10: RACE")
    print(f"\nHistorical training records: {len(history)}")
    print(f"Held-out candidates: {len(test)}")
    print(f"Random selection trials: {RANDOM_TRIALS}")
    print(f"Random seed: {SEED}")
    print(f"Inspection budgets: {BUDGETS}")

    print("\nAPACHE-19: MODEL-GUIDED VS RANDOM SELECTION")

    plan = plan_inspections(test, history, settings)
    comparisons = compare_selection(plan, test)
    oracle = oracle_best(test)

    print(f"Retrospective pool optimum: {oracle['performance']:.2f}")
    print(
        f"{'Budget':>7} {'Guided':>9} {'Random mean':>12}"
        f" {'Random median':>14} {'Random SD':>10}"
        f" {'W/T/L':>10}"
    )

    for item in comparisons:
        outcomes = (
            f"{item['guided_wins']}/"
            f"{item['ties']}/"
            f"{item['random_wins']}"
        )

        print(
            f"{item['budget']:>7}"
            f" {item['guided_best']:>9.2f}"
            f" {item['random_mean']:>12.2f}"
            f" {item['random_median']:>14.2f}"
            f" {item['random_sd']:>10.2f}"
            f" {outcomes:>10}"
        )

    print("\nRandom results across 30 trials:")
    for item in comparisons:
        print(
            f"Budget {item['budget']:>2}: "
            f"min={item['random_min']:.2f}, "
            f"max={item['random_max']:.2f}, "
            f"guided={item['guided_best']:.2f}"
        )

    print(
        "\nW/T/L = trials where guided selection was "
        "better / tied / worse than random."
    )

    print("\nAPACHE-20: PREDICTION BASELINE COMPARISON")

    prediction = evaluate_prediction_race(
        history, test, settings
    )

    low, high = prediction["bootstrap_interval"]

    print(f"Held-out predictions: {prediction['count']}")
    print(f"5-NN MAE: {prediction['model_mae']:.2f}")
    print(
        f"Training-mean baseline MAE: "
        f"{prediction['baseline_mae']:.2f}"
    )
    print(
        f"MAE improvement (baseline - kNN): "
        f"{prediction['mae_improvement']:+.2f}"
    )

    relative = prediction["relative_improvement_pct"]
    if relative is not None:
        print(f"Relative MAE improvement: {relative:+.1f}%")

    print(
        f"kNN better on: {prediction['model_better']} records"
    )
    print(
        f"Tied errors on: {prediction['equal']} records"
    )
    print(
        f"Baseline better on: "
        f"{prediction['baseline_better']} records"
    )

    print(
        f"Exploratory paired bootstrap interval for "
        f"MAE improvement: [{low:+.2f}, {high:+.2f}]"
    )

    print(
        "\nAll comparisons use the same held-out candidate pool."
        "\nNo held-out performance is used for model training"
        " or to construct the guided inspection plan."
        "\nRandom trials vary inspection order, not model training."
        "\nBootstrap uncertainty reflects the fixed test split"
        " and should not be generalized to independent splits."
    )


if __name__ == "__main__":
    main()
