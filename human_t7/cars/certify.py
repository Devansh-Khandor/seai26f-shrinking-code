
from pathlib import Path
from statistics import mean, pstdev

from shrinking_code.data import Table
from shrinking_code.certification import compare
from shrinking_code.ranking import rank_rows

DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "auto93.csv"
)


def print_comparison(label, xs, ys):
    result = compare(xs, ys)

    print(f"\n  {label}")
    print(f"    Group sizes: {result.left.count} vs {result.right.count}")

    if result.same is None:
        print("    Insufficient observations for comparison")
        return

    print(f"    Cohen:       {result.cohen:.3f}")
    print(f"    KS:          {result.ks:.3f}")
    print(f"    Cliff's:     {result.cliffs:.3f}")
    print(f"    Within thresholds: {result.same}")

    if result.exceeded:
        print(f"    Thresholds exceeded: {', '.join(result.exceeded)}")


def main():
    table = Table.load(DATA_FILE)

    mpg_col = next(c for c in table.columns if c.name == "Mpg+")
    cylinders_col = next(c for c in table.columns if c.name == "Clndrs")
    year_col = next(c for c in table.columns if c.name == "Model")

    print("CARS T7 - STEP 3: CERTIFY")
    print(f"Historical records: {len(table.rows)}")

    print("\nCAR-05: Fuel efficiency by cylinder count")

    mpg_by_cylinders = {}

    for row in table.rows:
        cylinders = row.cells[cylinders_col.index]
        mpg = row.cells[mpg_col.index]

        if cylinders is not None and mpg is not None:
            mpg_by_cylinders.setdefault(int(cylinders), []).append(mpg)

    reference = mpg_by_cylinders.get(4, [])

    for cylinders in sorted(mpg_by_cylinders):
        values = mpg_by_cylinders[cylinders]

        print(f"\n  {cylinders} cylinders:")
        print(f"    Vehicles: {len(values)}")
        print(f"    Average MPG: {mean(values):.2f}")
        print(f"    MPG StdDev:  {pstdev(values):.2f}")

        if cylinders != 4:
            print_comparison(
                f"{cylinders} cylinders vs 4 cylinders",
                values,
                reference,
            )
        else:
            print("    Reference group for comparisons")

    print("\nCAR-06: Overall performance by model-year period")

    periods = {
        "1970-1973": (70, 73),
        "1974-1977": (74, 77),
        "1978-1982": (78, 82),
    }

    ranked = rank_rows(table)
    scores_by_period = {name: [] for name in periods}

    for row, score in ranked:
        year = row.cells[year_col.index]

        if year is None:
            continue

        for name, (start, end) in periods.items():
            if start <= year <= end:
                scores_by_period[name].append(score)
                break

    for name, scores in scores_by_period.items():
        print(f"\n  {name}:")
        print(f"    Vehicles: {len(scores)}")

        if scores:
            print(f"    Average performance score: {mean(scores):.4f}")
            print(f"    Score StdDev: {pstdev(scores):.4f}")
            print("    Lower scores indicate better overall performance")

    names = list(periods)

    print("\n  Model-year period comparisons:")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            first = names[i]
            second = names[j]

            print_comparison(
                f"{first} vs {second}",
                scores_by_period[first],
                scores_by_period[second],
            )


if __name__ == "__main__":
    main()
