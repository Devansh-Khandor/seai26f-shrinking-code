
"""NASA T7 - Step 3: Certify."""

import random
import statistics

from remember import load_projects


def group_values(projects, feature, goal):
    """Collect outcome values for each feature rating."""
    groups = {}

    for project in projects:
        rating = project[feature]
        groups.setdefault(rating, []).append(project[goal])

    return groups


def permutation_test(a, b, repetitions=9999, seed=42):
    """Two-sided permutation test for a difference in means."""
    observed = abs(statistics.mean(a) - statistics.mean(b))

    combined = list(a) + list(b)
    rng = random.Random(seed)
    n = len(a)
    extreme = 0

    for _ in range(repetitions):
        rng.shuffle(combined)
        difference = abs(
            statistics.mean(combined[:n])
            - statistics.mean(combined[n:])
        )
        if difference >= observed - 1e-12:
            extreme += 1

    # Add-one correction for Monte Carlo p-values.
    p_value = (extreme + 1) / (repetitions + 1)
    return observed, p_value


def compare(projects, feature, goal, first, second):
    groups = group_values(projects, feature, goal)

    print(f"\nComparing {feature}={first} and {feature}={second}")
    print(f"Outcome: {goal}")

    if first not in groups or second not in groups:
        print("One or both ratings are missing.")
        return

    a = groups[first]
    b = groups[second]

    print(
        f"{first}: n={len(a)}, "
        f"mean={statistics.mean(a):.2f}, "
        f"median={statistics.median(a):.2f}"
    )
    print(
        f"{second}: n={len(b)}, "
        f"mean={statistics.mean(b):.2f}, "
        f"median={statistics.median(b):.2f}"
    )

    if min(len(a), len(b)) < 3:
        print("Insufficient observations for a reliable comparison.")
        return

    difference, p_value = permutation_test(a, b)

    print(f"Absolute mean difference: {difference:.2f}")
    print(f"Permutation p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(
            "Result: evidence of a difference in group means "
            "(unadjusted p < 0.05)."
        )
    else:
        print(
            "Result: insufficient evidence of a difference "
            "in group means."
        )

    print(
        "Note: Observational association only; "
        "not evidence of causation."
    )


def show_groups(projects, feature, goal):
    groups = group_values(projects, feature, goal)

    print(f"\nAvailable {feature} groups:")
    for rating in sorted(groups):
        values = groups[rating]
        print(
            f"  {rating:<3} n={len(values):>2} "
            f"mean {goal}={statistics.mean(values):.2f}"
        )


def main():
    projects = load_projects()

    print("NASA T7 - STEP 3: CERTIFY")
    print(f"Loaded {len(projects)} historical projects")

    print("\nNASA-05: PROGRAMMER CAPABILITY VS EFFORT")
    show_groups(projects, "pcap", "Effort-")
    compare(projects, "pcap", "Effort-", "h", "vh")

    print("\nNASA-06: COMPLEXITY VS DEFECTS")
    show_groups(projects, "cplx", "Defects-")
    compare(projects, "cplx", "Defects-", "n", "vh")

    print(
        "\nInterpretation: These are exploratory, "
        "unadjusted group comparisons."
    )
    print(
        "A non-significant result does not establish "
        "that two groups are equivalent."
    )


if __name__ == "__main__":
    main()
