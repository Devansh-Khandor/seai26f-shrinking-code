from pathlib import Path

from shrinking_code.data import Table
from shrinking_code.baselines import selection_race, prediction_race
from shrinking_code.evaluation import train_test_split


DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "auto93.csv"
)


def main():
    table = Table.load(DATA_FILE)

    print("CARS T7 - STEP 10: RACE")
    print(f"Historical records: {len(table.rows)}")

    print("\nCAR-19: Model selection vs. random selection")

    race = selection_race(
        table,
        budgets=(10, 20, 30),
        seeds=tuple(range(30))
    )

    for checkpoint in race.checkpoints:
        print(f"\nInspection budget: {checkpoint.budget}")
        print(f"  Model best score: {checkpoint.model_score:.4f}")
        print(f"  Random mean score: {checkpoint.random_summary.mean:.4f}")
        print(f"  Random median score: {checkpoint.random_summary.median:.4f}")
        print(
            f"  Random trials matching or beating model: "
            f"{checkpoint.match_or_beat_fraction:.1%}"
        )
        print(
            f"  Valid random trials: "
            f"{checkpoint.valid_trials}/{checkpoint.total_trials}"
        )
        print(f"  Model predictions: {checkpoint.model_predictions}")

    print("\nCAR-20: kNN vs. historical-average prediction")

    training, test = train_test_split(
        table, seed=42, train_fraction=0.80
    )

    print(f"Training vehicles: {len(training.rows)}")
    print(f"Testing vehicles: {len(test.rows)}")

    for target in ("Mpg+", "Lbs-"):
        result = prediction_race(training, test, target)
        evaluation = result.evaluation

        print(f"\n{target}: Prediction comparison")
        print(f"  Test records: {evaluation.test_count}")
        print(f"  kNN MAE: {evaluation.mae:.2f}")
        print(f"  Historical-average MAE: {evaluation.baseline_mae:.2f}")
        print(f"  Absolute improvement: {result.improvement:.2f}")

        if result.percentage_improvement is not None:
            print(
                f"  Percentage improvement: "
                f"{result.percentage_improvement:.2f}%"
            )

        print(f"  kNN wins: {result.wins}")
        print(f"  kNN losses: {result.losses}")
        print(f"  Ties: {result.ties}")

        comparison = result.comparison
        print(
            f"  Paired comparison: "
            f"{'within thresholds' if comparison.same else 'exceeds at least one threshold' if comparison.same is False else 'insufficient data'}"
        )

        if comparison.same is not None:
            print(
                f"  Cohen's d: {comparison.cohen:.3f}"
                if comparison.cohen is not None else
                "  Cohen's d: unavailable"
            )
            print(
                f"  KS statistic: {comparison.ks_d:.3f}"
                if comparison.ks_d is not None else
                "  KS statistic: unavailable"
            )
            print(
                f"  Cliff's delta: {comparison.cliffs:.3f}"
                if comparison.cliffs is not None else
                "  Cliff's delta: unavailable"
            )


if __name__ == "__main__":
    main()