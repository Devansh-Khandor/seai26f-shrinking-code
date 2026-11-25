
from pathlib import Path

from shrinking_code.data import Table
from shrinking_code.diagnosis import outcome_groups, compare_groups

DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "auto93.csv"
)


def print_findings(findings):
    for finding in findings:
        print(f"\n  {finding.column.name}")

        if finding.column.numeric:
            print(f"    Poor mean: {finding.poor.mean:.2f}")
            print(f"    Poor median: {finding.poor.median:.2f}")

            if finding.separation is not None:
                print(f"    Cliff's delta: {finding.separation:.3f}")
                print(f"    Direction: {finding.direction}")

        else:
            print(f"    Separation: {finding.separation:.3f}"
                  if finding.separation is not None
                  else "    Separation: unavailable")

            for category in finding.categories:
                if category.difference is not None:
                    print(
                        f"    {category.value}: "
                        f"Poor={category.poor_proportion:.1%}, "
                        f"Favorable={category.favorable_proportion:.1%}"
                    )

        for limitation in finding.limitations:
            print(f"    Note: {limitation}")


def main():
    table = Table.load(DATA_FILE)

    print("CARS T7 - STEP 5: BLAME")
    print(f"Historical records: {len(table.rows)}")

    print("\nCAR-09: Characteristics associated with poor MPG (fuel efficiency)")

    mpg_groups = outcome_groups(
        table,
        outcome="Mpg+",
        higher_better=True,
    )

    mpg_findings = compare_groups(
        mpg_groups.poor,
        mpg_groups.favorable,
    )

    print("\nCharacteristics (listed in order of how strongly they affect):")
    print_findings(mpg_findings)

    print("\nCAR-10: Characteristics associated with poor overall performance")

    performance_groups = outcome_groups(table)

    performance_findings = compare_groups(
        performance_groups.poor,
        performance_groups.favorable,
    )

    print("\nCharacteristics (listed in order of how strongly they affect):")
    print_findings(performance_findings)

if __name__ == "__main__":
    main()
