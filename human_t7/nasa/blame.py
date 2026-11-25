
"""NASA T7 - Step 5: Blame (NASA-09 and NASA-10)."""

from collections import Counter

from remember import load_projects
from guess import get_features


def split_projects(projects, goal):
    """Select equally sized low- and high-outcome groups."""
    ordered = sorted(
        projects,
        key=lambda row: (row[goal], row["row_number"]),
    )

    # Approximately one quarter at either end.
    size = len(projects) // 4

    low = ordered[:size]
    high = ordered[-size:]

    return low, high


def frequencies(projects, feature):
    """Return the frequency distribution of a categorical feature."""
    counts = Counter(row[feature] for row in projects)
    total = len(projects)

    return {
        value: count / total
        for value, count in counts.items()
    }


def total_variation(first, second):
    """Calculate distribution difference (0 = same, 1 = disjoint)."""
    values = set(first) | set(second)

    return 0.5 * sum(
        abs(first.get(value, 0) - second.get(value, 0))
        for value in values
    )


def diagnose(projects, goal):
    """Rank X features by association with high versus low outcomes."""
    features = get_features(projects)
    low, high = split_projects(projects, goal)

    results = []

    for feature in features:
        low_freq = frequencies(low, feature)
        high_freq = frequencies(high, feature)

        score = total_variation(low_freq, high_freq)

        # Find the feature value most overrepresented
        # among the high-outcome projects.
        values = set(low_freq) | set(high_freq)

        high_value = max(
            sorted(values),
            key=lambda value: (
                high_freq.get(value, 0)
                - low_freq.get(value, 0)
            ),
        )

        results.append({
            "feature": feature,
            "tv": score,
            "high_value": high_value,
            "high_frequency": high_freq.get(high_value, 0),
            "low_frequency": low_freq.get(high_value, 0),
        })

    results.sort(key=lambda result: (-result["tv"], result["feature"]))

    return low, high, results


def print_diagnosis(projects, goal, requirement):
    """Explain which characteristics differ most."""
    low, high, results = diagnose(projects, goal)

    print(f"\n{requirement}: DIAGNOSIS FOR {goal}")
    print(f"Low-outcome projects:  {len(low)}")
    print(f"High-outcome projects: {len(high)}")

    print(
        f"\n{'Feature':<10} {'TV':>7} {'Value':>8}"
        f" {'High %':>10} {'Low %':>10}"
    )

    for result in results[:10]:
        print(
            f"{result['feature']:<10}"
            f" {result['tv']:>7.4f}"
            f" {result['high_value']:>8}"
            f" {result['high_frequency'] * 100:>9.1f}%"
            f" {result['low_frequency'] * 100:>9.1f}%"
        )

    if results:
        strongest = results[0]
        print(
            f"\nStrongest observed association: "
            f"{strongest['feature']} "
            f"(TV={strongest['tv']:.4f})"
        )

    print(
        "Interpretation: Feature distributions differ "
        "between high and low groups."
    )
    print(
        "These are descriptive associations, "
        "not causal explanations."
    )

    return results


def main():
    projects = load_projects()

    print("NASA T7 - STEP 5: BLAME")
    print(f"Loaded {len(projects)} historical projects")
    print(f"Input features: {len(get_features(projects))}")

    print_diagnosis(
        projects,
        "Effort-",
        "NASA-09",
    )

    print_diagnosis(
        projects,
        "Defects-",
        "NASA-10",
    )

    print("\nDiagnosis completed.")


if __name__ == "__main__":
    main()
