from shrinking_code.data import Table
from shrinking_code.stats import summarize
from shrinking_code.ranking import rank_rows

from pathlib import Path

DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "auto93.csv"
)

def main():
    table = Table.load(DATA_FILE)

    print("CARS T7 — STEP 1: REMEMBER")
    print(f"\nLoaded {len(table.rows)} historical vehicle records")

    print("CARS-01: Historical vehicle records summary")
    for name in ("Lbs-", "Mpg+", "Acc+"):
        col = next(c for c in table.columns if c.name == name)
        summary = summarize(table, col)

        print(f"\n{name}")
        print(f"  Count:  {summary.count}")
        print(f"  Mean:   {summary.mean:.2f}")
        print(f"  Median: {summary.median:.2f}")
        print(f"  Best:   {summary.minimum if col.ideal == 0 else summary.maximum}")
        print(f"  Worst:  {summary.maximum if col.ideal == 0 else summary.minimum}")

    print("\nCARS-02: 10 Most attractive overall combination\n")
    ranked = rank_rows(table)
    cols = {c.name: c for c in table.columns}

    for row, score in ranked[:10]:
        print(
            f"Vehicle {row.number}: "
            f"Weight={row.cells[cols["Lbs-"].index]}, "
            f"Acceleration={row.cells[cols["Acc+"].index]}, "
            f"MPG={row.cells[cols["Mpg+"].index]}, "
            f"Score={score:.3f}"
        )

if __name__ == "__main__":
    main()
