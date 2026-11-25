"""Apache T7, Step 1: Remember (APACHE-01 and APACHE-02)."""

import csv
import math
import statistics
from pathlib import Path

DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "Apache_AllMeasurements.csv"
)

GOAL = "Performance-"
CONSTANT = "base"


def load_configurations(path=DATA_FILE):
    """Load historically observed Apache configurations."""
    configurations = []

    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or GOAL not in reader.fieldnames:
            raise ValueError("Missing Performance- goal column")

        settings = [
            name for name in reader.fieldnames
            if name not in (GOAL, CONSTANT)
        ]

        if not settings:
            raise ValueError("No adjustable settings found")

        for row_number, row in enumerate(reader, start=1):
            configuration = {"row_number": row_number}

            for name in reader.fieldnames:
                value = row[name]
                if value is None or not value.strip():
                    raise ValueError(
                        f"Missing {name} at row {row_number}"
                    )

                if name == GOAL:
                    configuration[name] = float(value)
                    if not math.isfinite(configuration[name]):
                        raise ValueError("Nonfinite performance value")
                else:
                    configuration[name] = int(value)
                    if configuration[name] not in (0, 1):
                        raise ValueError(
                            f"Nonbinary setting {name} at row {row_number}"
                        )

            configurations.append(configuration)

    if not configurations:
        raise ValueError("Dataset contains no configurations")

    if len({row[CONSTANT] for row in configurations}) != 1:
        raise ValueError("base is expected to be constant")

    return configurations


def adjustable_settings(configurations):
    """Return variable settings, excluding the constant base."""
    return [
        key for key in configurations[0]
        if key not in ("row_number", GOAL, CONSTANT)
    ]


def summarize(configurations):
    """Compute descriptive performance statistics."""
    values = [row[GOAL] for row in configurations]

    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "maximum": max(values),
    }


def rank_configurations(configurations):
    """Rank from best (lowest performance value) to worst."""
    return sorted(
        configurations,
        key=lambda row: (row[GOAL], row["row_number"]),
    )


def favorable_configurations(configurations, fraction=0.25):
    """Select the lowest fraction, including ties at the cutoff."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    ranked = rank_configurations(configurations)
    cutoff_index = math.ceil(len(ranked) * fraction) - 1
    cutoff = ranked[cutoff_index][GOAL]

    favorable = [
        row for row in ranked if row[GOAL] <= cutoff
    ]
    remaining = [
        row for row in ranked if row[GOAL] > cutoff
    ]

    return favorable, remaining, cutoff


def setting_differences(favorable, remaining, settings):
    """Compare setting=1 frequency in the two groups."""
    if not favorable or not remaining:
        return []

    comparisons = []

    for setting in settings:
        good_rate = sum(
            row[setting] == 1 for row in favorable
        ) / len(favorable)

        other_rate = sum(
            row[setting] == 1 for row in remaining
        ) / len(remaining)

        comparisons.append({
            "setting": setting,
            "favorable_rate": good_rate,
            "remaining_rate": other_rate,
            "difference": good_rate - other_rate,
        })

    return sorted(
        comparisons,
        key=lambda item: (
            -abs(item["difference"]),
            item["setting"],
        ),
    )


def print_configurations(configurations, settings, limit=5):
    """Display configuration identifiers, settings, and performance."""
    header = f"{'Row':>5} {'Performance':>13}"
    header += "".join(f" {name}" for name in settings)
    print(header)

    for row in configurations[:limit]:
        line = f"{row['row_number']:>5} {row[GOAL]:>13.2f}"
        line += "".join(f" {row[name]:>{len(name)}}" for name in settings)
        print(line)


def main():
    configurations = load_configurations()
    settings = adjustable_settings(configurations)
    summary = summarize(configurations)

    ranked = rank_configurations(configurations)
    favorable, remaining, cutoff = favorable_configurations(
        configurations
    )

    differences = setting_differences(
        favorable, remaining, settings
    )

    print("APACHE T7 — STEP 1: REMEMBER")
    print(f"\nLoaded {len(configurations)} configurations")
    print(f"Adjustable settings: {len(settings)}")

    print("\nAPACHE-01: PERFORMANCE SUMMARY")
    for name, value in summary.items():
        print(f"{name:<12}: {value:.2f}")

    print("\nAPACHE-01: FIVE BEST CONFIGURATIONS")
    print_configurations(ranked[:5], settings)

    print("\nAPACHE-01: FIVE WORST CONFIGURATIONS")
    print_configurations(ranked[-5:][::-1], settings)

    print("\nAPACHE-02: UNUSUALLY FAVORABLE CONFIGURATIONS")
    print("Definition: lowest 25% by recorded performance")
    print(f"Cutoff: {cutoff:.2f}")
    print(f"Favorable records: {len(favorable)}")
    print(f"Remaining records: {len(remaining)}")
    print_configurations(favorable, settings)

    print("\nAPACHE-02: DISTINGUISHING SETTINGS")
    print(
        f"{'Setting':<20} {'Favorable':>11}"
        f" {'Others':>11} {'Difference':>12}"
    )

    for item in differences:
        print(
            f"{item['setting']:<20}"
            f" {item['favorable_rate']:>11.1%}"
            f" {item['remaining_rate']:>11.1%}"
            f" {item['difference']:>+12.1%}"
        )

    print(
        "\nThese are descriptive associations, not causal effects."
    )


if __name__ == "__main__":
    main()
