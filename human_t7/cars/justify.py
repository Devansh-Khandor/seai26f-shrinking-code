
from pathlib import Path

from shrinking_code.data import Table
from shrinking_code.explanation import Explainer


DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "auto93.csv"
)


def print_explanation(explanation):
    print(f"Vehicle ID: {explanation.row.number}")

    print("\nGoal contributions:")
    for goal in explanation.goals:
        print(
            f"  {goal.name}: {goal.raw} "
            f"({goal.direction}, ideal={goal.ideal}, "
            f"normalized={goal.normalized:.3f}, "
            f"squared distance={goal.squared:.3f})"
        )

def main():
    table = Table.load(DATA_FILE)
    explainer = Explainer(table)

    print("CARS T7 - STEP 6: JUSTIFY")

    print("\nCAR-11: Recommendation explanation")

    best_row, best_score = explainer.ranked[0]
    best = explainer.explain(best_row.number)

    print("\nRecommended vehicle:")
    print_explanation(best)

    print("\nThree most similar historical vehicles:")
    neighbors = explainer.neighbors(best_row.number, k=3)

    for neighbor in neighbors:
        explanation = neighbor.explanation
        print(
            f"\n  Vehicle {explanation.row.number}: "
            f"distance={neighbor.distance:.4f}, "
            f"rank={explanation.rank}, "
            f"score={explanation.score:.4f}"
        )
        for goal in explanation.goals:
            print(f"    {goal.name}: {goal.raw}")

    print("\nCAR-12: Strengths and trade-offs")

    pair = explainer.tradeoff_pair()

    if pair is None:
        print("No qualifying trade-off pair was found.")
    else:
        print("\nVehicle A:")
        print_explanation(pair.a)

        print("\nVehicle B:")
        print_explanation(pair.b)

        print("\nComparison:")
        print(f"  Goals favoring A: {pair.favors_a or 'None'}")
        print(f"  Goals favoring B: {pair.favors_b or 'None'}")
        print(f"  Equal goals: {pair.equal or 'None'}")
        print(f"  Unknown goals: {pair.unknown or 'None'}")

        if pair.winner is not None:
            print(f"  Overall winner: Vehicle {pair.winner}")
            print(
                "  The winner has the lower combined "
                "distance-to-heaven score."
            )
        else:
            print("  No overall winner could be established.")


if __name__ == "__main__":
    main()
