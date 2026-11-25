"""Apache T7, Step 7: Choose (APACHE-13 and APACHE-14)."""

import math

from remember import (
    GOAL,
    adjustable_settings,
    load_configurations,
)
from guess import predict_performance, train_test_split

BUDGETS = (5, 10, 20)


def plan_inspections(candidates, history, settings, k=5):
    """Rank candidate IDs using predicted performance only."""
    if not candidates or not history:
        raise ValueError("Candidates and history must be nonempty")
    if not settings:
        raise ValueError("No adjustable settings available")

    candidate_ids = [r["row_number"] for r in candidates]
    history_ids = {r["row_number"] for r in history}

    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Duplicate candidate identifiers")
    if set(candidate_ids) & history_ids:
        raise ValueError("Candidates overlap training history")

    plan = []

    for candidate in candidates:
        estimate = predict_performance(
            candidate, history, settings, k
        )

        plan.append({
            "row_number": candidate["row_number"],
            "predicted_performance": estimate["prediction"],
            "neighbor_sd": estimate["neighbor_sd"],
        })

    plan.sort(
        key=lambda item: (
            item["predicted_performance"],
            item["row_number"],
        )
    )

    for rank, item in enumerate(plan, start=1):
        item["priority"] = rank

    return plan


def simulate_inspections(plan, candidates, budget):
    """Reveal actual outcomes only in the prescribed inspection order."""
    if not 1 <= budget <= len(plan):
        raise ValueError("Budget must be between 1 and plan length")

    candidates_by_id = {
        row["row_number"]: row for row in candidates
    }

    if len(candidates_by_id) != len(candidates):
        raise ValueError("Duplicate candidate identifiers")

    inspected = []
    best = None

    for entry in plan[:budget]:
        row_id = entry["row_number"]

        if row_id not in candidates_by_id:
            raise ValueError("Plan references unknown candidate")

        observed = candidates_by_id[row_id][GOAL]

        # The measured outcome is used only after this inspection.
        if best is None or (
            observed, row_id
        ) < (
            best["performance"], best["row_number"]
        ):
            best = {
                "row_number": row_id,
                "performance": observed,
            }

        inspected.append({
            "inspection": len(inspected) + 1,
            "row_number": row_id,
            "predicted_performance": entry["predicted_performance"],
            "measured_performance": observed,
            "best_so_far": best["performance"],
            "best_row_so_far": best["row_number"],
        })

    return {
        "budget": budget,
        "best_row": best["row_number"],
        "best_performance": best["performance"],
        "inspected": inspected,
    }


def evaluate_budgets(plan, candidates, budgets=BUDGETS):
    """Evaluate a fixed inspection plan at multiple budgets."""
    return [
        simulate_inspections(plan, candidates, budget)
        for budget in budgets
    ]


def oracle_best(candidates):
    """Retrospective full-pool optimum; never used to choose candidates."""
    if not candidates:
        raise ValueError("Candidates cannot be empty")

    best = min(
        candidates,
        key=lambda row: (row[GOAL], row["row_number"])
    )

    return {
        "row_number": best["row_number"],
        "performance": best[GOAL],
    }


def first_discovery(plan, candidates, target):
    """First inspection reaching an observed performance target."""
    if not math.isfinite(target):
        raise ValueError("Target must be finite")

    outcome = simulate_inspections(
        plan, candidates, len(plan)
    )

    for record in outcome["inspected"]:
        if record["best_so_far"] <= target:
            return record

    return None


def main():
    rows = load_configurations()
    settings = adjustable_settings(rows)
    history, candidates = train_test_split(rows)

    plan = plan_inspections(candidates, history, settings)
    results = evaluate_budgets(plan, candidates)
    oracle = oracle_best(candidates)

    print("APACHE T7 — STEP 7: CHOOSE")
    print(f"\nHistorical training records: {len(history)}")
    print(f"Held-out candidate configurations: {len(candidates)}")
    print("Ranking: lowest predicted performance first")
    print("Prediction model: 5-nearest-neighbor regression")
    print("Only historically observed configurations are considered.")

    print("\nAPACHE-14: PRIORITIZED BENCHMARKING PLAN")
    print(
        f"{'Priority':>8} {'Row':>5}"
        f" {'Predicted':>11} {'Neighbor SD':>12}"
    )

    for entry in plan[:20]:
        print(
            f"{entry['priority']:>8}"
            f" {entry['row_number']:>5}"
            f" {entry['predicted_performance']:>11.2f}"
            f" {entry['neighbor_sd']:>12.2f}"
        )

    print("\nAPACHE-13: BEST CONFIGURATION FOUND BY BUDGET")
    print(
        f"{'Budget':>7} {'Best Row':>9}"
        f" {'Best Measured':>15}"
    )

    for result in results:
        print(
            f"{result['budget']:>7}"
            f" {result['best_row']:>9}"
            f" {result['best_performance']:>15.2f}"
        )

    print("\nDISCOVERY PROGRESSION")
    print(
        f"{'Inspection':>10} {'Row':>5}"
        f" {'Measured':>11} {'Best So Far':>12}"
    )

    for record in results[-1]["inspected"]:
        print(
            f"{record['inspection']:>10}"
            f" {record['row_number']:>5}"
            f" {record['measured_performance']:>11.2f}"
            f" {record['best_so_far']:>12.2f}"
        )

    # Retrospective analysis only: the oracle was not
    # consulted when creating the inspection plan.
    print("\nRETROSPECTIVE EVALUATION")
    print(f"Full candidate-pool best row: {oracle['row_number']}")
    print(
        f"Full candidate-pool best performance: "
        f"{oracle['performance']:.2f}"
    )

    discovery = first_discovery(
        plan, candidates, oracle["performance"]
    )

    if discovery is None:
        print("The full-pool optimum was not discovered.")
    else:
        print(
            f"First inspection reaching the full-pool optimum: "
            f"{discovery['inspection']}"
        )

    print(
        "\nNo candidate's measured performance was used to "
        "build the inspection plan."
    )
    print(
        "The full-pool optimum is retrospective evaluation "
        "information, not an available selection oracle."
    )


if __name__ == "__main__":
    main()
