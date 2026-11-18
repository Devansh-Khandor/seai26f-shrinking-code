from pathlib import Path

from shrinking_code.data import Row, Table
from shrinking_code.prediction import KNNRegressor
from shrinking_code.evaluation import train_test_split, evaluate

DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "auto93.csv"
)

def main():
    table = Table.load(DATA_FILE)
    training, testing = train_test_split(table, seed=42, train_fraction=0.8)

    print("CARS T7 — STEP 2: GUESS")
    print(f"Historical records: {len(table.rows)}")
    print(f"Training records: {len(training.rows)}")
    print(f"Testing records: {len(testing.rows)}")

    model = KNNRegressor.fit(training, k=5)

    for target, description in (
        ("Mpg+", "CARS-03: Fuel efficiency estimation and uncertaintyc"),
        ("Lbs-", "CARS-04: Vehicle weight estimation and uncertainty"),
    ):
        
        result = evaluate(model, testing, target)

        print(f"\n{description}")
        print(f"  MAE:          {result.mae:.2f}")
        print(f"  Baseline MAE: {result.baseline_mae:.2f}")
        print(f"  Test records: {result.test_count}")

    example = testing.rows[0]
    targets = ("Mpg+", "Lbs-")

    query = Row(
        number=example.number,
        cells=tuple(
            None if column.name in targets else value
            for column, value in zip(testing.columns, example.cells)
        ),
    )

    print("\nVEHICLE ESTIMATION EXAMPLE")
    print(f"Vehicle ID: {query.number}")
    print("Known characteristics:")

    for column, value in zip(testing.columns, query.cells):
        if column.role == "x":
            print(f"  {column.name}: {value}")

    print("\nEstimated values:")

    for target, label, unit in (
        ("Mpg+", "Fuel efficiency", "MPG"),
        ("Lbs-", "Vehicle weight", "lbs"),
    ):
        prediction = model.predict(query, target)

        print(f"  {label}: {prediction.value:.2f} {unit}")
        print(
            f"    Neighbor range: "
            f"{prediction.minimum:.2f}-{prediction.maximum:.2f} {unit}"
        )
        print(f"    Neighbor StdDev: {prediction.stddev:.2f}")

if __name__ == "__main__":
    main()

 