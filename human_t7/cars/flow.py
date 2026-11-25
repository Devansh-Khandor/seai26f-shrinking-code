
from pathlib import Path

from shrinking_code.data import Table
from shrinking_code.streaming import Inventory

DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "auto93.csv"
)


def print_shortlist(inventory):
    print("\n  Current top-five vehicles:")

    for position, (row, score) in enumerate(inventory.shortlist, 1):
        mpg = row.cells[7]
        weight = row.cells[5]

        print(
            f"    {position}. Vehicle {row.number}: "
            f"MPG={mpg}, Weight={weight}, Score={score:.3f}"
        )


def main():
    table = Table.load(DATA_FILE)

    initial_rows = table.rows[:50]
    initial = Table(table.columns, initial_rows)
    inventory = Inventory(initial)

    print("CARS T7 - STEP 4: FLOW")
    print(f"Initial inventory: {len(initial.rows)} vehicles")
    print_shortlist(inventory)

    print("\nCAR-07: Assess each arriving vehicle")

    for row in table.rows[50:55]:
        arrival = inventory.accept(row.cells, number=row.number)

        print(f"\n  New vehicle: {arrival.row.number}")
        print(f"    Previous inventory: {arrival.previous_count}")
        print(f"    Overall score: {arrival.score:.3f}" if arrival.score is not None
              else "    Overall score: unavailable")
        print(f"    Current rank: {arrival.rank}" if arrival.rank is not None
              else "    Current rank: unavailable")
        print(f"    Top-five vehicle: {arrival.top_five}")

    print("\nCAR-08: Updated inventory")
    print(f"Accepted new vehicles: {inventory.accepted_count}")
    print(f"Current inventory size: {len(inventory.table.rows)}")

    print("\nUpdated summaries:")

    for column, summary in zip(inventory.table.columns, inventory.summaries):
        if column.role == "y":
            print(
                f"  {column.name}: "
                f"Count={summary.count}, "
                f"Mean={summary.mean:.2f}, "
                f"Median={summary.median:.2f}"
            )

    print_shortlist(inventory)


if __name__ == "__main__":
    main()
