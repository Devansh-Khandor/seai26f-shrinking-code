"""New-code measurement per step (pre-registration, docs/hypotheses.md §5.1-5.2).

Primary: non-blank, non-comment implementation lines ADDED versus the
previous step. Deleted lines and test code are counted separately.
"""

from dataclasses import dataclass
import difflib
from pathlib import PurePosixPath

COMMENT = {".py": "#", ".lua": "--"}


@dataclass(frozen=True)
class StepLoc:
    new_loc: int
    deleted_loc: int
    test_loc: int
    total_loc: int
    pct_new: float


def is_test_path(path):
    p = PurePosixPath(path)
    if any(part.startswith("test") for part in p.parts):
        return True
    return p.stem.endswith("_test")


def code_lines(text, suffix):
    marker = COMMENT.get(suffix)
    if marker is None:
        return []
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith(marker):
            out.append(s)
    return out


def measure_step(prev, cur):
    new = deleted = tests = total = 0
    for path in sorted(set(prev) | set(cur)):
        suffix = PurePosixPath(path).suffix
        a = code_lines(prev.get(path, ""), suffix)
        b = code_lines(cur.get(path, ""), suffix)
        added = removed = 0
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if tag in ("insert", "replace"):
                added += j2 - j1
            if tag in ("delete", "replace"):
                removed += i2 - i1
        if is_test_path(path):
            tests += added
            continue
        new += added
        deleted += removed
        total += len(b)
    pct = round(100 * new / total, 2) if total else 0.0
    return StepLoc(new, deleted, tests, total, pct)
