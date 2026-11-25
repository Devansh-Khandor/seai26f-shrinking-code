"""Semantic (requirement) checks for one step: does the program actually do what each request asks?

The program already ran once on the real CSV (checks.run_check). Here its
ANSWER lines are checked against harness/answers.py, then it is run again on
controlled copies of the data, so an answer cannot pass by being hard-coded,
by peeking at hidden values, or by reading the future:

  shuffled  rows in a fixed different order; every check is repeated on it
  mask      held-out target values replaced by absurd ones (Guess, Race)
  future    later arrivals' goals replaced (Flow); earlier decisions must not move
  honeypot  six planted rows with ideal goals but mediocre inputs (Choose, Race)

Per request: pass | fail | unverified | execution_failure | checker_error.
The step passes only if both requests pass. Nothing here calls an LLM.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import tempfile
import traceback

from harness import answers as A
from harness import oracle as O

SEED = 20261117
STATUSES = ("pass", "fail", "unverified", "execution_failure", "checker_error")


@dataclass
class RequestResult:
    request_id: str
    status: str
    passed: int = 0
    total: int = 0
    failed: list = field(default_factory=list)     # "run:check: detail"
    notes: list = field(default_factory=list)       # unverified reasons, checker errors


@dataclass
class StepSemantic:
    status: str
    requests: list
    runs: dict = field(default_factory=dict)        # run name -> short outcome


# ---- controlled copies of the data ------------------------------------------

def _fmt(v):
    return str(int(v)) if float(v).is_integer() else f"{v:.6g}"


def shuffled(data):
    raw = [list(r) for r in data.raw]
    random.Random(SEED).shuffle(raw)
    return data.with_raw(raw), {}


def masked(data, target):
    j = data.col(target)
    vs = data.values(target)
    span = (max(vs) - min(vs)) or 1.0
    raw = [list(r) for r in data.raw]
    for r in O.held_out(data.n):
        v = data.rows[r - 1][j]
        if v is not None:
            raw[r - 1][j] = _fmt(3 * v + 3 * span)
    return data.with_raw(raw), {}


def future(data):
    """Later arrivals get goals far better than anything observed: a program that looks
    ahead (global bounds, ranks over the whole file) changes its earlier decisions."""
    later = O.arrivals(data.n)[O.arrivals_count(data.n) // 2:]
    raw = [list(r) for r in data.raw]
    for g, w in data.goals():
        j = data.col(g)
        vs = data.values(g)
        span = (max(vs) - min(vs)) or 1.0
        extreme = max(vs) + 2 * span if w > 0 else max(0.0, min(vs) - 2 * span) if min(vs) >= 0 else min(vs) - 2 * span
        for r in later:
            raw[r - 1][j] = _fmt(extreme)
    return data.with_raw(raw), {}


def honeypot(data):
    """Six planted rows: inputs copied from middling rows, goals set to the best observed values."""
    s = O.d2h(data)
    mid = O.median(list(s.values()))
    donors = sorted(s, key=lambda r: (abs(s[r] - mid), r))[:A.HONEYPOTS]
    raw = [list(r) for r in data.raw]
    planted = []
    for r in donors:
        cells = list(data.raw[r - 1])
        for g, w in data.goals():
            vs = data.values(g)
            cells[data.col(g)] = _fmt(max(vs) if w > 0 else min(vs))
        planted.append(cells)
    n = len(raw) + len(planted)
    positions = [round(n * (i + 1) / (len(planted) + 1)) for i in range(len(planted))]
    out, pots, k = [], set(), 0
    for pos in range(1, n + 1):
        if k < len(planted) and pos == positions[k]:
            out.append(planted[k])
            pots.add(pos)
            k += 1
        else:
            out.append(raw.pop(0))
    return data.with_raw(out), {"honeypots": pots}


def _make(run, data, rids):
    if run == "shuffled":
        return shuffled(data)
    if run == "future":
        return future(data)
    if run == "honeypot":
        return honeypot(data)
    if run == "mask":
        targets = {A.REQUESTS[r][1]["target"] for r in rids if "mask" in A.KINDS[A.REQUESTS[r][0]][2]}
        d = data
        for t in sorted(targets):
            d, _ = masked(d, t)
        return d, {}
    raise ValueError(run)


# ---- evaluation ----------------------------------------------------------------

def _status(items, notes):
    if any(not ok for _, ok, _ in items):
        return "fail"
    if notes:
        return "unverified"
    return "pass"


def evaluate(request_ids, csv_path, base_result, rerun, evidence=None):
    """Judge REQUEST_IDS from BASE_RESULT (a CheckResult on CSV_PATH).

    RERUN(csv_path) runs the same program on another CSV and returns a CheckResult.
    EVIDENCE, if given, is a path where a JSON record of every check is written.
    """
    if not base_result.runs:
        why = "timed out" if base_result.timed_out else "crashed or exited non-zero"
        res = [RequestResult(r, "execution_failure", notes=[f"program {why}"]) for r in request_ids]
        return _finish(StepSemantic("execution_failure", res), evidence)
    try:
        return _finish(_evaluate(request_ids, csv_path, base_result, rerun), evidence)
    except Exception:
        tb = traceback.format_exc()
        res = [RequestResult(r, "checker_error", notes=[tb]) for r in request_ids]
        return _finish(StepSemantic("checker_error", res), evidence)


def _evaluate(rids, csv_path, base_result, rerun):
    base_data = O.load(csv_path)
    base_ans = A.parse_answers(base_result.stdout)
    items = {r: [] for r in rids}
    notes = {r: [] for r in rids}
    base_ctx = {}
    for r in rids:
        a = base_ans.get(r)
        if a is None:
            items[r].append(("base:answer present", False, f"no `ANSWER {r} {{...}}` line"))
            continue
        if isinstance(a, str):
            items[r].append(("base:answer format", False, a))
            continue
        c = A.check_request(r, base_data, a, "base")
        items[r] += c.items
        if not any(name.endswith("answer format") and not ok for name, ok, _ in c.items):
            base_ctx[r] = c
    runs = {"base": "ran"}
    jobs = []   # (run kind, key in `runs`, requests judged, data maker)
    for run in ["shuffled"] + A.runs_needed(rids):
        wanted = [r for r in rids if run == "shuffled" or run in A.KINDS[A.REQUESTS[r][0]][2]]
        wanted = [r for r in wanted if r in base_ctx]
        if not wanted:
            continue
        if run == "mask":
            # One masked copy per target: a model for one target may legitimately use the
            # held-out rows' *other* goal values as inputs, so those must stay untouched.
            by_target = {}
            for r in wanted:
                by_target.setdefault(A.REQUESTS[r][1]["target"], []).append(r)
            for t, rs in sorted(by_target.items()):
                key = "mask" if len(by_target) == 1 else f"mask {t}"
                jobs.append((run, key, rs, lambda t=t: masked(base_data, t)))
        else:
            jobs.append((run, run, wanted, lambda run=run, wanted=wanted: _make(run, base_data, wanted)))
    for run, key, wanted, make in jobs:
        data, extra = make()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / Path(csv_path).name
            path.write_text(data.text(), encoding="utf-8")
            res = rerun(path)
        if res.timed_out:
            runs[key] = "timed out"
            for r in wanted:
                notes[r].append(f"{run}: program timed out on the controlled copy")
            continue
        if not res.runs:
            runs[key] = "crashed"
            for r in wanted:
                items[r].append((f"{run}:program runs", False,
                                 f"crashed on the {run} copy of the data: {res.stderr.strip()[-300:]}"))
            continue
        runs[key] = "ran"
        new_ans = A.parse_answers(res.stdout)
        for r in wanted:
            a = new_ans.get(r)
            if not isinstance(a, dict):
                items[r].append((f"{run}:answer present", False, f"no valid ANSWER {r} line on the {run} copy"))
                continue
            if run in ("shuffled", "honeypot"):
                items[r] += A.check_request(r, data, a, run, extra).items
            else:
                items[r] += A.compare_request(r, base_ctx[r], data, a, run).items
    out = []
    for r in rids:
        failed = [f"{n}: {d}" for n, ok, d in items[r] if not ok]
        out.append(RequestResult(r, _status(items[r], notes[r]), sum(ok for _, ok, _ in items[r]),
                                 len(items[r]), failed, notes[r]))
    return StepSemantic(step_status([x.status for x in out]), out, runs)


def step_status(statuses):
    if all(s == "pass" for s in statuses):
        return "pass"
    for s in ("execution_failure", "checker_error", "fail", "unverified"):
        if s in statuses:
            return s
    return "fail"


def _finish(sem, evidence):
    if evidence is not None:
        Path(evidence).write_text(json.dumps({
            "status": sem.status, "runs": sem.runs,
            "requests": [vars(r) for r in sem.requests]}, indent=1), encoding="utf-8")
    return sem
