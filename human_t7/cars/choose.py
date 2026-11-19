from pathlib import Path

from shrinking_code.data import Table
from shrinking_code.evaluation import train_test_split
from shrinking_code.selection import HistoricalOracle, select_candidates
from shrinking_code.ranking import distance_to_heaven
from shrinking_code.stats import summarize

DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "auto93.csv"
)

def main():
    table = Table.load(DATA_FILE)

    training, candidates = train_test_split(
        table,
        seed=42,
        train_fraction=0.60,
    )

    budget = 30
    checkpoints = (5, 10, 20, 30)

    oracle = HistoricalOracle(candidates, budget)

    print("CARS T7 - STEP 7: CHOOSE")

    result = select_candidates(
        training,
        oracle.visible,
        oracle.inspect,
        budget=budget,
        k=5,
    )

    print("\nCAR-13: Balanaced vehicle selection")

    for cost in checkpoints:
        checkpoint = result.checkpoint(cost)
        best = checkpoint.best_row

        print(f"\nAfter {cost} inspections:")

        if best is None:
            print("  No vehicle with complete observed goals.")
        else:
            print(f"  Best vehicle: {best.number}")
            print(f"  Actual score: {checkpoint.best_score:.4f}")

    goals = training.role("y")
    summaries = {col.index: summarize(training, col) for col in goals}

    candidate_scores = [
        (row, distance_to_heaven(row, goals, summaries))
        for row in candidates.rows
    ]
    candidate_scores = [
        (row, score) for row, score in candidate_scores
        if score is not None
    ]
    candidate_best = min(
        candidate_scores,
        key=lambda item: (item[1], item[0].number)
    )

    print("\nOffline candidate-pool benchmark:")
    print(f"  Best candidate: {candidate_best[0].number}")
    print(f"  Best possible score: {candidate_best[1]:.4f}")

    print("\nCAR-14: Inspection priority")

    for position, inspection in enumerate(result.inspections, 1):
        print(
            f"  {position}. Vehicle {inspection.row.number}: "
            f"Predicted={inspection.predicted_score:.4f}, "
            f"Observed={inspection.observed_score:.4f}"
            if inspection.observed_score is not None
            else
            f"  {position}. Vehicle {inspection.row.number}: "
            f"Predicted={inspection.predicted_score:.4f}, "
            "Observed=unavailable"
        )

    print("\nSummary:")
    print(f"  Inspections performed: {len(result.inspections)}")
    print(f"  Predictions made: {result.prediction_count}")
    print(f"  Oracle inspections: {len(oracle.inspected)}")


if __name__ == "__main__":
    main()