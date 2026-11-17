"""Name the reason a step failed, from evidence only (status, exit, stderr, semantic result).

Each category is `<who>:<what>`, where who is
  system      the machine or harness (backend down, context too small, missing dependency)
  generation  the reply could not be turned into code (cut off, no parsable files)
  code        the generated program itself (syntax, bad import, API misuse, crash, wrong answers)
  evaluator   the semantic checker could not judge it
so system and evaluator problems are never blamed on the model.
"""

import re

ALLOWED_MODULES = {"plain": set(), "sklearn": {"sklearn", "numpy", "scipy", "joblib", "threadpoolctl"},
                   "ezr": {"ezr"}}
_EXC = re.compile(r"^(\w+(?:\.\w+)*(?:Error|Exception|Interrupt|Exit|Warning)|\w+Error)\b:?\s*(.*)$")
_MISSING = re.compile(r"No module named '([\w.]+)'")
_CANT_IMPORT = re.compile(r"cannot import name '(\w+)' from '([\w.]+)'")
_LIB_FRAME = re.compile(r'File ".*[\\/](sklearn|numpy|scipy|ezr)(?:[\\/]|\.py")')

STATUS_CATEGORY = {
    "backend_error": "system:backend",
    "context_overflow": "system:context_overflow",
    "output_truncated": "generation:truncated",
    "parse_failed": "generation:extraction",
}


def last_exception(stderr):
    """(exception name, message) from the last traceback line, or (None, last line)."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = _EXC.match(ln)
        if m:
            return m.group(1).split(".")[-1], m.group(2)
    return None, (lines[-1] if lines else "")


def crash_category(stderr, background):
    exc, msg = last_exception(stderr)
    allowed = ALLOWED_MODULES.get(background, set())
    if exc in ("SyntaxError", "IndentationError", "TabError"):
        return "code:syntax", f"{exc}: {msg}"
    if exc == "ModuleNotFoundError":
        m = _MISSING.search(msg)
        mod = m.group(1).split(".")[0] if m else "?"
        if mod in allowed:
            return "system:dependency", f"allowed module {mod} is not installed"
        return "code:forbidden_import", f"imports {mod}, which this treatment does not provide"
    if exc == "ImportError":
        m = _CANT_IMPORT.search(msg)
        if m and m.group(2).split(".")[0] in allowed:
            return "code:api_misuse", f"{m.group(2)} has no {m.group(1)}"
        return "code:forbidden_import", f"ImportError: {msg}"
    if exc == "MemoryError":
        return "system:resource", "MemoryError"
    if exc in ("AttributeError", "TypeError", "ValueError", "KeyError", "NameError") and (
            _LIB_FRAME.search(stderr or "") or re.search(r"\b(ezr|sklearn|numpy)\b", msg)):
        return "code:api_misuse", f"{exc}: {msg}"
    if exc:
        return "code:runtime", f"{exc}: {msg}"
    return "code:runtime", f"exited non-zero: {msg}"[:200]


def categorize(status, check=None, semantic=None, background=None):
    """(category, detail) for one step; ('', '') when it fully passed."""
    if status in STATUS_CATEGORY:
        return STATUS_CATEGORY[status], status
    if check is None:
        return "evaluator:unknown", f"status {status}"
    if check.timed_out:
        return "runtime:timeout", f"no exit within the time limit ({check.seconds:.0f}s)"
    if check.output_capped:
        return "runtime:output_flood", "printed more than the output cap"
    if not check.runs:
        if check.stderr.startswith("missing entry point"):
            return "code:missing_entry", check.stderr.strip()
        return crash_category(check.stderr, background)
    if semantic is not None:
        if semantic.status == "checker_error":
            return "evaluator:checker_error", "the semantic checker raised an exception"
        if semantic.status == "fail":
            failed = [f for r in semantic.requests for f in r.failed]
            no_answer = [f for f in failed if "answer present" in f or "answer format" in f]
            crashed = [f for f in failed if "program runs" in f]
            if no_answer and len(no_answer) == len(failed):
                return "code:no_answer", no_answer[0][:200]
            if crashed:
                return "code:fragile", crashed[0][:200]
            return "code:semantic", (failed[0] if failed else "")[:200]
        if semantic.status == "unverified":
            notes = [n for r in semantic.requests for n in r.notes]
            return "evaluator:unverified", (notes[0] if notes else "")[:200]
    if not check.contract:
        return "code:format", "output sections missing or too short (original contract)"
    return "", ""
