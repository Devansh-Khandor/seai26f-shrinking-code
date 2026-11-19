from pathlib import Path

from shrinking_code.data import Table 
from shrinking_code.alternatives import Alternatives
from shrinking_code.stats import summarize

DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "auto93.csv"
)

def print_changes(changes):
    if not changes:
        print("  No known characteristic changes")
        return

    for change in changes:
        print(f"  {change.name}: {change.before} -> {change.after}")

def main():
    table = Table.load(DATA_FILE)
    alternatives = Alternatives(table)

    source = alternatives.worst()

    if source is None:
        print("No complete historical vehicles available")
        return

    source_result = alternatives.explainer.explain(source.number)

    print("CARS T7 - STEP 8: FIX")
    print(f"Historical records: {len(table.rows)}")
    print(f"Source vehicle: {source.number}")
    print(f"Source performance score: {source_result.score:.4f}")

    print("\nCAR-15: Better-performing historical alternatives")

    better = alternatives.better(source.number, limit=5)

    if not better:
        print("  No better historical alternatives found.")
    else:
        for position, alternative in enumerate(better, 1):
            pair = alternative.comparison
            print(f"\n{position} Vehicle {pair.b.row.number}")
            print(f"  Performance score: {pair.b.score:.4f}")
            print(f"  Improvement: {alternative.improvement:.4f}")
            print(f"  Characteristic distance: {alternative.distance:.4f}")
            print(f"  Pareto dominates source: {alternative.pareto_dominates}")
            print("  Characteristic changes:")
            print_changes(alternative.changed)

    goals = table.role("y")
    targets = {
        column.name: summarize(table, column).median
        for column in goals
    }

    print("\nCAR-16: Closest alternative meeting more targets")
    print("Target thresholds:")
    for name, value in targets.items():
        direction = "at most" if next(
            c for c in goals if c.name == name
        ).ideal == 0 else "at least"
        print(f"  {name}: {direction} {value:.2f}")

    source_status = alternatives.explainer.reference
    repair = alternatives.meet_more(source.number, targets)

    if repair is None:
        print("\n No historical alternative meets more targets")
    else:
        pair = repair.comparison
        print(f"\n  Alternative vehicle: {pair.b.row.number}")
        print(f"  Performance score: {pair.b.score:.4f}")
        print(f"  Characteristic distance: {repair.distance:.4f}")
        print(f"  Targets satisfied by source: "
              f"{len(repair.source_targets.satisfied)}")
        print(f"  Targets satisfied by alternative: "
              f"{len(repair.alternative_targets.satisfied)}")
        print(f"  Number of characteristic changes: {len(repair.changed)}")
        print("  Characteristic changes:")
        print_changes(repair.changed)

        print("  Newly satisfied targets:")
        newly_satisfied = (
            set(repair.alternative_targets.satisfied)
            - set(repair.source_targets.satisfied)
        )
        for name in sorted(newly_satisfied):
            print(f"    {name}")


if __name__ == "__main__":
    main()