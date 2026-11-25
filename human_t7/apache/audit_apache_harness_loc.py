#!/usr/bin/env python3
"""Read-only Apache T7 LOC audit matching the NASA harness audit."""

import difflib
import subprocess
from pathlib import PurePosixPath

SKILLS = (
    "Remember", "Guess", "Certify", "Flow", "Blame",
    "Justify", "Choose", "Fix", "Spot", "Race",
)

PREFIX = "human_t7/apache/"
COMMENT = {".py": "#", ".lua": "--"}


def git(*args, as_bytes=False):
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        check=True,
    )
    return (
        result.stdout
        if as_bytes
        else result.stdout.decode("utf-8").strip()
    )


def code_lines(content, suffix):
    """Apply exactly the same source filter as the NASA auditor."""
    marker = COMMENT.get(suffix)

    if marker is None:
        return []

    return [
        line.strip()
        for line in content.splitlines()
        if line.strip()
        and not line.strip().startswith(marker)
    ]


def snapshot(ref):
    """Read Apache source files directly from a Git checkpoint."""
    paths = git(
        "ls-tree", "-r", "--name-only", ref
    ).splitlines()

    result = {}

    for path in paths:
        if (
            not path.startswith(PREFIX)
            or PurePosixPath(path).suffix not in COMMENT
        ):
            continue

        relative_path = path[len(PREFIX):]

        result[relative_path] = git(
            "show",
            f"{ref}:{path}",
            as_bytes=True,
        ).decode("utf-8")

    return result


def is_test_path(path):
    p = PurePosixPath(path)

    return (
        any(part.startswith("test") for part in p.parts)
        or p.stem.endswith("_test")
    )


def measure(previous, current):
    """Replicate NASA's snapshot-diff LOC measurement."""
    added_total = 0
    deleted_total = 0
    test_added_total = 0
    cumulative_total = 0
    changed = []

    for path in sorted(set(previous) | set(current)):
        suffix = PurePosixPath(path).suffix

        old = code_lines(previous.get(path, ""), suffix)
        now = code_lines(current.get(path, ""), suffix)

        added = 0
        removed = 0

        matcher = difflib.SequenceMatcher(
            None, old, now, autojunk=False
        )

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ("insert", "replace"):
                added += j2 - j1

            if tag in ("delete", "replace"):
                removed += i2 - i1

        if added or removed:
            changed.append((path, added, removed))

        if is_test_path(path):
            test_added_total += added
            continue

        added_total += added
        deleted_total += removed
        cumulative_total += len(now)

    percentage = (
        round(100 * added_total / cumulative_total, 2)
        if cumulative_total else 0.0
    )

    return (
        added_total,
        deleted_total,
        test_added_total,
        cumulative_total,
        percentage,
        changed,
    )


def main():
    git("rev-parse", "--show-toplevel")

    all_added = 0
    all_deleted = 0
    all_tests = 0
    primary_added = 0

    print("APACHE T7 — HARNESS-COMPATIBLE LOC AUDIT")
    print()
    print(
        f"{'Step':>4} {'Skill':<12} {'New LOC':>9}"
        f" {'Deleted':>9} {'Test LOC':>9}"
        f" {'Total LOC':>10} {'% New':>8}"
    )
    print("-" * 72)

    for step, skill in enumerate(SKILLS, start=1):
        before = (
            f"t7-apache-step{step - 1}"
            if step > 1 else
            "t7-apache-step1^"
        )
        after = f"t7-apache-step{step}"

        previous = snapshot(before)
        current = snapshot(after)

        new, deleted, tests, total, pct, changed = measure(
            previous, current
        )

        all_added += new
        all_deleted += deleted
        all_tests += tests

        if step >= 2:
            primary_added += new

        print(
            f"{step:>4} {skill:<12} {new:>9}"
            f" {deleted:>9} {tests:>9}"
            f" {total:>10} {pct:>7.2f}%"
        )

        changes = ", ".join(
            f"{path} (+{added}/-{removed})"
            for path, added, removed in changed
        )

        print("     Changes: " + (changes or "(none)"))

    print("-" * 72)
    print(f"All ten skills — new implementation LOC: {all_added}")
    print(f"All ten skills — deleted implementation LOC: {all_deleted}")
    print(f"All ten skills — separately counted new test LOC: {all_tests}")
    print()
    print("PRIMARY RESEARCH METRIC")
    print(f"New implementation LOC, Steps 2–10: {primary_added}")
    print()
    print("Measurement matches the NASA T7 auditor:")
    print("- Filter blank and comment-only lines")
    print("- Compare filtered Git source snapshots")
    print("- Count inserted and replacement lines as additions")
    print("- Count deleted and replacement lines as deletions")
    print("- Exclude test LOC from the implementation metric")
    print()
    print("No repository files or Git tags were modified.")


if __name__ == "__main__":
    main()
