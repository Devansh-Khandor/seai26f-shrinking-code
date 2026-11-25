"""Apache T7, Step 5: Blame (APACHE-09 and APACHE-10)."""

import math
from itertools import combinations

from remember import (
    GOAL,
    adjustable_settings,
    load_configurations,
    rank_configurations,
)

POOR_FRACTION = 0.25
MIN_SUPPORT = 5


def split_poor_performers(rows, fraction=POOR_FRACTION):
    """Select the worst fraction, including ties at the cutoff."""
    if not rows:
        raise ValueError("Dataset cannot be empty")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")

    ranked = rank_configurations(rows)
    cutoff_index = len(ranked) - math.ceil(len(rows) * fraction)
    cutoff = ranked[cutoff_index][GOAL]

    poor = [row for row in rows if row[GOAL] >= cutoff]
    other = [row for row in rows if row[GOAL] < cutoff]

    if not poor or not other:
        raise ValueError("Both comparison groups are required")

    return poor, other, cutoff


def prevalence(rows, conditions):
    """Fraction of records satisfying all specified setting values."""
    if not rows:
        raise ValueError("Cannot compute prevalence on empty records")

    return sum(
        all(row[setting] == value for setting, value in conditions)
        for row in rows
    ) / len(rows)


def analyze_individual_settings(poor, other, settings):
    """Find setting values enriched among poor performers."""
    results = []

    for setting in settings:
        for value in (0, 1):
            conditions = ((setting, value),)

            poor_rate = prevalence(poor, conditions)
            other_rate = prevalence(other, conditions)

            results.append({
                "setting": setting,
                "value": value,
                "poor_count": round(poor_rate * len(poor)),
                "other_count": round(other_rate * len(other)),
                "poor_rate": poor_rate,
                "other_rate": other_rate,
                "difference": poor_rate - other_rate,
            })

    # For binary settings, values 0 and 1 are complementary.
    # Keep the value more common among poor performers.
    enriched = [
        item for item in results
        if item["difference"] > 0
    ]

    return sorted(
        enriched,
        key=lambda item: (
            -item["difference"],
            item["setting"],
            item["value"],
        ),
    )


def analyze_setting_pairs(
    poor, other, settings, min_support=MIN_SUPPORT
):
    """Identify two-setting patterns enriched among poor records."""
    if min_support < 1:
        raise ValueError("min_support must be positive")

    results = []

    for first, second in combinations(settings, 2):
        for first_value in (0, 1):
            for second_value in (0, 1):
                conditions = (
                    (first, first_value),
                    (second, second_value),
                )

                poor_count = sum(
                    all(row[name] == value for name, value in conditions)
                    for row in poor
                )
                other_count = sum(
                    all(row[name] == value for name, value in conditions)
                    for row in other
                )

                if poor_count < min_support:
                    continue

                poor_rate = poor_count / len(poor)
                other_rate = other_count / len(other)

                difference = poor_rate - other_rate

                if difference <= 0:
                    continue

                results.append({
                    "first_setting": first,
                    "first_value": first_value,
                    "second_setting": second,
                    "second_value": second_value,
                    "poor_count": poor_count,
                    "other_count": other_count,
                    "poor_rate": poor_rate,
                    "other_rate": other_rate,
                    "difference": difference,
                })

    return sorted(
        results,
        key=lambda item: (
            -item["difference"],
            -item["poor_count"],
            item["first_setting"],
            item["second_setting"],
            item["first_value"],
            item["second_value"],
        ),
    )


def main():
    rows = load_configurations()
    settings = adjustable_settings(rows)

    poor, other, cutoff = split_poor_performers(rows)

    individual = analyze_individual_settings(
        poor, other, settings
    )
    pairs = analyze_setting_pairs(
        poor, other, settings
    )

    print("APACHE T7 — STEP 5: BLAME")
    print(f"\nTotal configurations: {len(rows)}")
    print(f"Poor-performance cutoff: {cutoff:.2f}")
    print(f"Poor performers: {len(poor)}")
    print(f"Remaining configurations: {len(other)}")
    print("Definition: highest 25% Performance-, including ties.")

    print("\nAPACHE-09: SETTINGS ASSOCIATED WITH POOR PERFORMANCE")
    print(
        f"{'Setting':<20} {'Value':>5} {'Poor':>9}"
        f" {'Others':>9} {'Diff':>10} {'Support':>8}"
    )

    for item in individual:
        print(
            f"{item['setting']:<20}"
            f" {item['value']:>5}"
            f" {item['poor_rate']:>9.1%}"
            f" {item['other_rate']:>9.1%}"
            f" {item['difference']:>+10.1%}"
            f" {item['poor_count']:>8}"
        )

    print("\nAPACHE-10: SETTINGS COMBINATIONS")
    print("Top 10 enriched two-setting patterns")
    print(f"Minimum poor-group support: {MIN_SUPPORT}")

    print(
        f"{'Pattern':<46} {'Poor':>8}"
        f" {'Others':>8} {'Diff':>9} {'Count':>7}"
    )

    for item in pairs[:10]:
        pattern = (
            f"{item['first_setting']}={item['first_value']}, "
            f"{item['second_setting']}={item['second_value']}"
        )

        print(
            f"{pattern:<46}"
            f" {item['poor_rate']:>8.1%}"
            f" {item['other_rate']:>8.1%}"
            f" {item['difference']:>+9.1%}"
            f" {item['poor_count']:>7}"
        )

    print(
        "\nPrevalence differences indicate associations only."
        "\nThey do not establish causal effects of Apache settings."
    )


if __name__ == "__main__":
    main()
