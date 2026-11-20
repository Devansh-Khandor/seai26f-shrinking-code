
"""Measure the human T7 Cars milestones using the official LOC methodology."""

import argparse
import csv
from pathlib import Path

import measure_loc as base


SKILLS = (
    "Remember", "Guess", "Certify", "Flow", "Blame",
    "Justify", "Choose", "Fix", "Spot", "Race",
)

COMMITS = (
    "91eb95b",
    "36ba9bf",
    "b3b558f",
    "38765b1",
    "c722fe6",
    "ff8e07b",
    "ba3c893",
    "0bd2c33",
    "7d9c40b",
    "c996b0e",
)


# Keep the official categories, but include the T7 implementation.
original_category = base.category


def category(path):
    if path.startswith("human_t7/cars/") and path.endswith(".py"):
        return "impl"
    return original_category(path)


base.category = category


def t7_history(git, skills=SKILLS):
    if len(skills) != len(COMMITS):
        raise base.MeasurementError("Skill and commit counts do not match")

    milestones = []
    previous = None

    for step, (skill, short_commit) in enumerate(zip(skills, COMMITS), 1):
        commit = git.text("rev-parse", "--verify", short_commit + "^{commit}")

        if step == 1:
            parents = git.text(
                "rev-list", "--parents", "-n", "1", commit
            ).split()[1:]

            if len(parents) != 1:
                raise base.MeasurementError(
                    "The first T7 commit must have exactly one parent"
                )

            previous = parents[0]

        if previous == commit:
            raise base.MeasurementError(
                f"T7 step {step}: repeated milestone commit"
            )

        git.run("merge-base", "--is-ancestor", previous, commit)

        if git.text("rev-list", "--merges", f"{previous}..{commit}"):
            raise base.MeasurementError(
                f"T7 step {step}: merge found in milestone interval"
            )

        milestones.append({
            "step": step,
            "skill": skill,
            "tag": f"human_t7_step{step}",
            "tag_object": "",
            "commit": commit,
            "base_commit": previous,
        })

        previous = commit

    return milestones


# Reuse the official snapshot, tokenization, diff, and LOC calculations.
base.history = t7_history


def write_results(rows, destination):
    if not rows:
        raise base.MeasurementError("No T7 milestone results to write")
    
    destination.mkdir(parents=True, exist_ok=True)

    csv_path = destination / "human_t7_cars_loc.csv"
    md_path = destination / "human_t7_cars_loc.md"

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Human T7 Cars LOC measurements",
        "",
        "Measured from the ten T7 Git commits using the methodology in "
        "`experiments/measure_loc.py`.",
        "",
        "Implementation LOC includes both `shrinking_code/*.py` "
        "and `human_t7/cars/*.py`.",
        "",
        "| Step | Skill | Commit | Added impl | Removed impl | Total impl | New % | "
        "Added tests | Added docs |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        keys = (
            "step", "skill", "commit", "added_impl_loc",
            "removed_impl_loc", "total_impl_loc", "percent_new_code",
            "added_test_loc", "added_docs_loc",
        )
        lines.append("| " + " | ".join(str(row[k]) for k in keys) + " |")

    first, last = rows[0], rows[-1]
    added = sum(row["added_impl_loc"] for row in rows)
    removed = sum(row["removed_impl_loc"] for row in rows)

    lines += [
        "",
        "## Summary",
        "",
        f"- Baseline implementation LOC: {first['base_impl_loc']}",
        f"- Final implementation LOC: {last['total_impl_loc']}",
        f"- Total added implementation LOC: {added}",
        f"- Total removed implementation LOC: {removed}",
        "",
        "## Methodology",
        "",
        "This script reuses the official measurement implementation for "
        "Python tokenization, snapshots, Git diffs, and LOC calculations. "
        "The only category extension is that `human_t7/cars/*.py` is "
        "counted as implementation alongside `shrinking_code/*.py`.",
        "",
        "The first T7 commit is compared with its parent. Each later commit "
        "is compared with the preceding T7 commit. The shared "
        "`shrinking_code/` library is therefore included in the baseline "
        "and cumulative implementation totals.",
        "",
        "Added LOC counts eligible added lines, not net growth. Removed "
        "LOC is reported separately. New percentage is added implementation "
        "LOC divided by total implementation LOC at that milestone.",
        "",
        "These measurements describe one sequential development history; "
        "they are not independent experimental repetitions or statistical "
        "evidence for H1 or H2.",
        "",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: <repo>/results)",
    )
    args = parser.parse_args()

    try:
        rows = base.measure(args.repo)
        write_results(rows, args.output or args.repo / "results")
    except (base.MeasurementError, OSError) as error:
        parser.exit(1, f"LOC measurement failed: {error}\n")

    print(
        f"Measured {len(rows)} T7 milestones; "
        f"final implementation LOC: {rows[-1]['total_impl_loc']}"
    )


if __name__ == "__main__":
    main()
