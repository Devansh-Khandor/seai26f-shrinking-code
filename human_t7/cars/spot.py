from pathlib import Path

from shrinking_code.data import Table
from shrinking_code.evaluation import train_test_split
from shrinking_code.anomalies import input_isolation, outcome_residuals


DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "auto93.csv"
)


def main():
    table = Table.load(DATA_FILE)

    print("CARS T7 - STEP 9: SPOT")
    print(f"Historical records: {len(table.rows)}")

    print("\nCAR-17: Unusual vehicle characteristics")

    isolation = input_isolation(table, k=5, top=5)

    for finding in isolation:
        if not finding.flagged:
            continue

        status = "FLAGGED" if finding.flagged else "Review"
        print(f"\nVehicle {finding.row.number}: {status}")
        if finding.score is not None:
            print(f"  Mean neighbor distance: {finding.score:.4f}")
        print(f"  Reason: {finding.reason}")
        print("  Nearest neighbors:")
        for number, distance in finding.neighbors:
            print(f"    Vehicle {number}: distance={distance:.4f}")

    training, test = train_test_split(
        table, seed=42, train_fraction=0.80
    )

    print("\nCAR-18: Unexpected vehicle outcomes")
    print(f"Training vehicles: {len(training.rows)}")
    print(f"Testing vehicles: {len(test.rows)}")

    for target in ("Mpg+", "Lbs-"):
        findings = outcome_residuals(
            training, test, target, k=5, top=5
        )

        print(f"\n{target}: Largest prediction errors")

        for finding in findings:
            if not finding.flagged:
                continue
            
            status = "FLAGGED" if finding.flagged else "Review"
            print(f"\nVehicle {finding.row.number}: {status}")

            if finding.score is None:
                print(f"  {finding.reason}")
                continue

            print(f"  Actual: {finding.actual:.2f}")
            print(f"  Predicted: {finding.prediction.value:.2f}")
            print(f"  Absolute error: {finding.score:.2f}")
            print(f"  Signed error: {finding.signed:.2f}")
            print(f"  Reason: {finding.reason}")


if __name__ == "__main__":
    main()