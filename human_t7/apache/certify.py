"""Apache T7, Step 3: Certify (APACHE-05 and APACHE-06)."""

import math
import random
import statistics

from remember import GOAL, load_configurations

PERMUTATIONS = 2000
SEED = 42


def split_by_setting(rows, setting):
    """Separate observations by binary setting value."""
    disabled = [r[GOAL] for r in rows if r[setting] == 0]
    enabled = [r[GOAL] for r in rows if r[setting] == 1]

    if not disabled or not enabled:
        raise ValueError(f"Both groups required for {setting}")

    return disabled, enabled


def cohens_d(a, b):
    """Signed standardized mean difference: b minus a."""
    if len(a) < 2 or len(b) < 2:
        return 0.0

    variance = (
        (len(a) - 1) * statistics.variance(a)
        + (len(b) - 1) * statistics.variance(b)
    ) / (len(a) + len(b) - 2)

    if variance == 0:
        return 0.0 if statistics.mean(a) == statistics.mean(b) else math.inf

    return (statistics.mean(b) - statistics.mean(a)) / math.sqrt(variance)


def cliffs_delta(a, b):
    """Signed dominance effect, comparing b against a."""
    greater = sum(y > x for x in a for y in b)
    less = sum(y < x for x in a for y in b)
    return (greater - less) / (len(a) * len(b))


def ks_statistic(a, b):
    """Two-sample maximum empirical CDF difference."""
    xs, ys = sorted(a), sorted(b)
    i = j = 0
    maximum = 0.0

    for value in sorted(set(xs + ys)):
        while i < len(xs) and xs[i] <= value:
            i += 1
        while j < len(ys) and ys[j] <= value:
            j += 1

        maximum = max(
            maximum,
            abs(i / len(xs) - j / len(ys)),
        )

    return maximum


def permutation_pvalue(a, b, repeats=PERMUTATIONS, seed=SEED):
    """Two-sided permutation test of difference in means."""
    observed = abs(statistics.mean(b) - statistics.mean(a))
    combined = list(a) + list(b)
    n_a = len(a)

    rng = random.Random(seed)
    extreme = 0

    for _ in range(repeats):
        shuffled = combined.copy()
        rng.shuffle(shuffled)

        difference = abs(
            statistics.mean(shuffled[n_a:])
            - statistics.mean(shuffled[:n_a])
        )

        if difference >= observed - 1e-12:
            extreme += 1

    return (extreme + 1) / (repeats + 1)


def compare_groups(rows, setting):
    disabled, enabled = split_by_setting(rows, setting)

    mean0 = statistics.mean(disabled)
    mean1 = statistics.mean(enabled)
    median0 = statistics.median(disabled)
    median1 = statistics.median(enabled)

    d = cohens_d(disabled, enabled)
    cliff = cliffs_delta(disabled, enabled)
    ks = ks_statistic(disabled, enabled)
    p = permutation_pvalue(disabled, enabled)

    return {
        "setting": setting,
        "n0": len(disabled),
        "n1": len(enabled),
        "mean0": mean0,
        "mean1": mean1,
        "median0": median0,
        "median1": median1,
        "mean_difference": mean1 - mean0,
        "cohens_d": d,
        "cliffs_delta": cliff,
        "ks": ks,
        "permutation_p": p,
    }


def print_comparison(result):
    name = result["setting"]

    print(f"\n{name}: DISABLED VS ENABLED")
    print(f"Disabled records: {result['n0']}")
    print(f"Enabled records : {result['n1']}")

    print(f"Mean disabled   : {result['mean0']:.2f}")
    print(f"Mean enabled    : {result['mean1']:.2f}")
    print(f"Median disabled : {result['median0']:.2f}")
    print(f"Median enabled  : {result['median1']:.2f}")

    print(
        f"Mean difference (enabled - disabled): "
        f"{result['mean_difference']:+.2f}"
    )

    print(f"Cohen's d       : {result['cohens_d']:+.4f}")
    print(f"Cliff's delta   : {result['cliffs_delta']:+.4f}")
    print(f"KS statistic    : {result['ks']:.4f}")
    print(f"Permutation p   : {result['permutation_p']:.4f}")

    if result["mean_difference"] < 0:
        print("Observed direction: enabled has lower mean performance.")
    elif result["mean_difference"] > 0:
        print("Observed direction: disabled has lower mean performance.")
    else:
        print("Observed direction: equal group means.")

    if result["permutation_p"] < 0.05:
        print("Evidence: difference in means is distinguishable")
        print("under this exploratory permutation test.")
    else:
        print("Evidence: insufficient to distinguish the")
        print("mean difference from sampling variation.")

    print("Interpretation: observational, not causal.")


def main():
    rows = load_configurations()

    print("APACHE T7 — STEP 3: CERTIFY")
    print(f"Loaded {len(rows)} historical configurations")

    print("\nAPACHE-05: KEEPALIVE")
    print_comparison(compare_groups(rows, "keepAlive"))

    print("\nAPACHE-06: ACCESS LOGGING")
    print_comparison(compare_groups(rows, "accessLog"))

    print(
        "\nCaution: These are exploratory group comparisons."
        "\nMultiple testing, dependence between settings, "
        "and confounding limit strong conclusions."
        "\nThis is not a causal experiment or a direct "
        "replication of ezr's treatment-ranking procedure."
    )


if __name__ == "__main__":
    main()
