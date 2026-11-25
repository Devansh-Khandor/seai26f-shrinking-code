"""External (business-facing) metrics recovered offline from the frozen v2 archive.

    python -m harness.external            # writes harness/results/v2-external/gpt-oss-20b/

Phase 1 covers one metric only: prediction MAE against the historical-mean
baseline for the Guess requests CAR-03, NASA-03 and APACHE-03.

What it does, for every attempted (domain, treatment, trial) of step 2:

1. Reads the archived program output (stepNN.check.txt) and semantic evidence
   (stepNN.semantic.json) straight from the v2 ZIP, in memory. Nothing in the
   archive or in harness/results/v2/ is written, and no generated program is run.
2. Parses ANSWER lines with harness.answers.parse_answers and validates the
   answer with the frozen checker (harness.answers.check_request), exactly as
   the semantic check did.
3. Recomputes the model's MAE from its recorded held-out predictions and the
   CSV, and the historical-mean baseline MAE on the same held-out rows. The
   protocol is the frozen one (harness.oracle.held_out: every row whose number
   is divisible by 5 is held out; the baseline mean uses only the other rows).
4. Applies the leakage gate already recorded by the semantic check (the `mask`
   rerun). A schema-valid answer whose leakage check failed, or could not be
   verified, is kept in the output but excluded from the primary metric.

These metrics were added after the Task Talk, after the v2 results had been
seen. They are not part of the pre-registration (docs/hypotheses.md).
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from math import comb
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile

from harness import answers as A
from harness import oracle as O

HARNESS_DIR = Path(__file__).resolve().parent
REPO = HARNESS_DIR.parent
FROZEN_DIR = HARNESS_DIR / "results" / "v2"
ARCHIVE = FROZEN_DIR / "gpt-oss-20b-3trials.zip"
RUNS_CSV = FROZEN_DIR / "runs.csv"
REQUIREMENTS_CSV = FROZEN_DIR / "requirements.csv"
DATA_DIR = REPO / "data" / "raw"
OUT_ROOT = HARNESS_DIR / "results" / "v2-external"
MODEL = "gpt-oss-20b"

DOMAINS = ("cars", "nasa", "apache")
TREATMENTS = ("T1", "T2", "T3", "T4", "T5", "T6")
TRIALS = (1, 2, 3)
DATASETS = {"cars": "auto93.csv", "nasa": "nasa93dem.csv", "apache": "Apache_AllMeasurements.csv"}

# Phase 1: the Guess prediction request of each domain (skill 2, the first request of the step).
GUESS_STEP = 2
PREDICTION = {
    "cars": ("CAR-03", "Mpg+", +1),
    "nasa": ("NASA-03", "Effort-", -1),
    "apache": ("APACHE-03", "Performance-", -1),
}

ENTRY = re.compile(
    r"^(?P<root>[^/]+)/runs/(?P<model>[^/]+)/(?P<domain>cars|nasa|apache)/(?P<treatment>T[1-6])/"
    r"trial(?P<trial>\d+)/step(?P<step>\d{2})\.(?P<kind>check\.txt|semantic\.json)$")

# Why a row is not in the primary metric. Empty string = included.
REASONS = (
    "not_attempted",            # no runs.csv row for the step
    "generation_failed",        # runs.csv status is not ok (e.g. parse_failed): no program to run
    "execution_failure",        # the program exited non-zero on the real CSV
    "execution_timeout",        # the program timed out on the real CSV
    "answer_missing",           # no ANSWER line for the request
    "answer_invalid_json",      # ANSWER line present but not a JSON object
    "answer_schema_error",      # the frozen checker rejected the answer's format
    "leakage_failed",           # mask rerun: changing held-out targets moved the predictions
    "leakage_unverified",       # mask rerun crashed, timed out, gave no valid answer, or never ran
    "baseline_zero",            # baseline MAE is 0, so the ratio is undefined
    "honeypot_failed",          # honeypot rerun: planted ideal rows were picked within the budget (peeking)
    "honeypot_unverified",      # honeypot rerun crashed, timed out, gave no valid answer, or never ran
    "budget_mismatch",          # race_select: the reported budget is not the frozen oracle budget
)

CSV_FIELDS = [
    "model", "domain", "treatment", "trial", "skill_index", "request_id", "target", "target_direction",
    "step_status", "step_runs", "step_semantic", "request_status",
    "check_runs", "check_timed_out", "answer_state", "answer_detail",
    "leakage_gate", "leakage_detail", "shuffled_gate",
    "n_heldout", "model_mae", "baseline_mean", "baseline_mae", "mae_ratio", "beats_baseline",
    "reported_mae", "reported_mae_matches",
    "primary", "exclusion_reason",
    "check_entry", "check_crc32", "check_sha256", "semantic_entry", "semantic_sha256",
]

SUMMARY_FIELDS = [
    "scope", "domain", "treatment", "request_id",
    "attempted", "generation_failed", "execution_failed", "answer_missing", "answer_invalid_json",
    "answer_schema_error", "schema_valid", "leakage_failed", "leakage_unverified",
    "primary_n", "primary_rate", "beats_baseline_n", "conditional_beat_rate", "delivery_rate",
    "median_mae_ratio", "min_mae_ratio", "max_mae_ratio", "median_model_mae", "baseline_mae",
]


# ---- archive paths -------------------------------------------------------------

def parse_entry(name):
    """{root, model, domain, treatment, trial, step, kind} for a run entry, else None."""
    m = ENTRY.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["trial"], d["step"] = int(d["trial"]), int(d["step"])
    return d


def entry_name(root, model, domain, treatment, trial, step, kind):
    return f"{root}/runs/{model}/{domain}/{treatment}/trial{trial}/step{step:02d}.{kind}"


def archive_root(names, model):
    roots = {p["root"] for p in map(parse_entry, names) if p and p["model"] == model}
    if len(roots) != 1:
        raise ValueError(f"expected one archive root for model {model}, found {sorted(roots)}")
    return roots.pop()


def request_ids(domain, step):
    prefix = {"cars": "CAR", "nasa": "NASA", "apache": "APACHE"}[domain]
    return f"{prefix}-{2 * step - 1:02d}", f"{prefix}-{2 * step:02d}"


# ---- check.txt -----------------------------------------------------------------

def split_check(text):
    """(header flags, stdout, stderr) of an archived check.txt.

    The stdout is everything between the first '--- stdout ---' line and the
    last '--- stderr ---' line, so program output that happens to contain the
    marker text cannot cut the stdout short.
    """
    lines = text.splitlines()
    flags = {}
    if lines:
        for tok in lines[0].split():
            k, _, v = tok.partition("=")
            flags[k] = v == "True"
    try:
        start = lines.index("--- stdout ---")
    except ValueError:
        return flags, "", ""
    ends = [i for i, ln in enumerate(lines) if ln == "--- stderr ---" and i > start]
    end = ends[-1] if ends else len(lines)
    return flags, "\n".join(lines[start + 1:end]), "\n".join(lines[end + 1:])


# ---- semantic evidence -----------------------------------------------------------

def leakage_gate(sem, request_id, target):
    """('pass' | 'fail' | 'unverified' | 'not_run', detail) from the archived mask rerun.

    The semantic check runs one masked copy per target; its key in `runs` is
    'mask' when only one target needed masking, else 'mask <target>'.
    """
    req = next((r for r in sem.get("requests", []) if r.get("request_id") == request_id), None)
    if req is None:
        return "not_run", "request missing from semantic evidence"
    runs = sem.get("runs", {})
    key = f"mask {target}" if f"mask {target}" in runs else "mask"
    outcome = runs.get(key)
    masked = [f for f in req.get("failed", []) if f.startswith("mask:")]
    leak = [f for f in masked if f.startswith("mask:held-out targets not used")]
    if leak:
        return "fail", leak[0][:200]
    if outcome is None:
        return "not_run", "no mask rerun recorded"
    if outcome != "ran":
        return "unverified", f"mask rerun {outcome}"
    if masked:
        return "unverified", masked[0][:200]
    return "pass", ""


def shuffled_gate(sem, request_id):
    req = next((r for r in sem.get("requests", []) if r.get("request_id") == request_id), None)
    if req is None or "shuffled" not in sem.get("runs", {}):
        return "not_run"
    return "fail" if any(f.startswith("shuffled:") for f in req.get("failed", [])) else "pass"


# ---- the metric ------------------------------------------------------------------

def baseline(data, target):
    """(historical mean, baseline MAE, held-out rows): the mean of the non-held-out rows' TARGET,
    scored on the held-out rows. Mirrors harness.answers._baseline_mae; computed independently."""
    held = O.held_out(data.n)
    held_set = set(held)
    train = [data.value(r, target) for r in range(1, data.n + 1)
             if r not in held_set and data.value(r, target) is not None]
    mean = O.mean(train)
    truth = [data.value(r, target) for r in held]
    if any(v is None for v in truth):
        raise ValueError(f"missing {target} value in a held-out row")
    return mean, O.mean([abs(mean - y) for y in truth]), held


def model_mae(data, target, preds):
    """MAE of PREDS {row: estimate} against the CSV's TARGET; PREDS must cover exactly the held-out rows."""
    held = O.held_out(data.n)
    if set(preds) != set(held):
        raise ValueError("predictions do not cover exactly the held-out rows")
    return O.mean([abs(preds[r] - data.value(r, target)) for r in held])


def mae_ratio(model, base):
    """(ratio, reason): model/base; undefined when the baseline MAE is 0."""
    if base == 0:
        return None, "baseline_zero"
    return model / base, ""


# ---- extraction -------------------------------------------------------------------

def _index(path, keys):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = tuple(r[c] for c in keys)
            if k in out:
                raise ValueError(f"duplicate row {k} in {path}")
            out[k] = r
    return out


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def extract(archive=ARCHIVE, runs_csv=RUNS_CSV, requirements_csv=REQUIREMENTS_CSV, data_dir=DATA_DIR, model=MODEL,
            domains=DOMAINS):
    """One row per attempted (domain, treatment, trial) for the Guess prediction request."""
    runs = _index(runs_csv, ("model", "domain", "treatment", "trial", "skill_index"))
    reqs = _index(requirements_csv, ("model", "domain", "treatment", "trial", "request_id"))
    rows = []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        root = archive_root(names, model)
        for domain in domains:
            rid, target, direction = PREDICTION[domain]
            if rid not in request_ids(domain, GUESS_STEP) or A.REQUESTS[rid] != ("predict", {"target": target}):
                raise ValueError(f"{rid} is not the frozen predict request for {target}")
            data = O.load(Path(data_dir) / DATASETS[domain])
            if (target, direction) not in data.goals():
                raise ValueError(f"{target} is not a goal with direction {direction:+d} in {DATASETS[domain]}")
            base_mean, base_mae, held = baseline(data, target)
            for t in TREATMENTS:
                for k in TRIALS:
                    rows.append(_row(z, names, root, model, domain, t, k, rid, target, direction, data,
                                     base_mean, base_mae, len(held), runs, reqs))
    keys = [(r["domain"], r["treatment"], r["trial"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate trial records")
    return rows


def _attempt(z, names, root, model, domain, t, k, step, rid, data, runs, reqs, fields, defaults):
    """Common part of every metric: locate the attempt, read its archived output, validate the answer.

    Returns (row, ctx, answer, semantic evidence). ctx is None when the attempt has no schema-valid
    answer; the row then already carries its exclusion reason."""
    row = {f: "" for f in fields}
    row.update(model=model, domain=domain, treatment=t, trial=k, skill_index=step, request_id=rid, primary=False)
    row.update(defaults)
    run = runs.get((model, domain, t, str(k), str(step)))
    req = reqs.get((model, domain, t, str(k), rid))
    row["request_status"] = req["status"] if req else ""
    if run is None:
        row["exclusion_reason"] = "not_attempted"
        return row, None, None, {}
    row.update(step_status=run["status"], step_runs=run["runs"], step_semantic=run["semantic"])
    check = entry_name(root, model, domain, t, k, step, "check.txt")
    sem_name = entry_name(root, model, domain, t, k, step, "semantic.json")
    if run["status"] != "ok" or check not in names:
        row["answer_state"] = "no_output"
        row["exclusion_reason"] = "generation_failed"
        row["answer_detail"] = f"step status {run['status']}"
        return row, None, None, {}
    raw = z.read(check)
    info = z.getinfo(check)
    row.update(check_entry=check, check_crc32=f"{info.CRC:08x}", check_sha256=_sha(raw))
    sem = {}
    if sem_name in names:
        sb = z.read(sem_name)
        sem = json.loads(sb)
        row.update(semantic_entry=sem_name, semantic_sha256=_sha(sb))
    flags, stdout, _ = split_check(raw.decode("utf-8", "replace"))
    row.update(check_runs=flags.get("runs", False), check_timed_out=flags.get("timed_out", False))
    if not flags.get("runs", False):
        row["answer_state"] = "not_checked"
        row["exclusion_reason"] = "execution_timeout" if flags.get("timed_out") else "execution_failure"
        return row, None, None, sem
    ans = A.parse_answers(stdout).get(rid)
    if ans is None:
        row.update(answer_state="missing", exclusion_reason="answer_missing")
        return row, None, None, sem
    if isinstance(ans, str):
        row.update(answer_state="invalid_json", answer_detail=ans[:200], exclusion_reason="answer_invalid_json")
        return row, None, None, sem
    ctx = A.check_request(rid, data, ans, "base")
    bad = [d for name, ok, d in ctx.items if name.endswith("answer format") and not ok]
    if bad:
        row.update(answer_state="schema_error", answer_detail=bad[0][:200], exclusion_reason="answer_schema_error")
        return row, None, None, sem
    row["answer_state"] = "valid"
    return row, ctx, ans, sem


def _row(z, names, root, model, domain, t, k, rid, target, direction, data, base_mean, base_mae, n_held, runs, reqs):
    row, ctx, ans, sem = _attempt(
        z, names, root, model, domain, t, k, GUESS_STEP, rid, data, runs, reqs, CSV_FIELDS,
        dict(target=target, target_direction=direction, baseline_mean=base_mean, baseline_mae=base_mae,
             n_heldout=n_held, leakage_gate="not_run", shuffled_gate="not_run"))
    if ctx is None:
        return row
    preds = A._heldout_preds(ctx)            # the checker's own reading of the predictions
    mae = model_mae(data, target, preds)
    ratio, why = mae_ratio(mae, base_mae)
    row.update(model_mae=mae, mae_ratio="" if ratio is None else ratio,
               beats_baseline="" if ratio is None else mae < base_mae)
    try:
        rep = A.num(ans.get("mae"))
        row.update(reported_mae=rep, reported_mae_matches=O.close(rep, mae, rel=0.01))
    except A.SchemaError:
        row.update(reported_mae="", reported_mae_matches=False)
    gate, detail = leakage_gate(sem, rid, target)
    row.update(leakage_gate=gate, leakage_detail=detail, shuffled_gate=shuffled_gate(sem, rid))
    if gate == "fail":
        row["exclusion_reason"] = "leakage_failed"
    elif gate != "pass":
        row["exclusion_reason"] = "leakage_unverified"
    elif why:
        row["exclusion_reason"] = why
    else:
        row["primary"] = True
    return row


# ---- phase 2: selection quality (Choose, choose_order) ------------------------------

def honeypot_gate(sem, request_id):
    """('pass' | 'fail' | 'unverified' | 'not_run', detail) from the archived honeypot rerun.

    The honeypot copy plants six rows with ideal goals but middling inputs. A program that
    picks them within its budget is reading goal values it has not paid to inspect."""
    req = next((r for r in sem.get("requests", []) if r.get("request_id") == request_id), None)
    if req is None:
        return "not_run", "request missing from semantic evidence"
    outcome = sem.get("runs", {}).get("honeypot")
    hp = [f for f in req.get("failed", []) if f.startswith("honeypot:")]
    peek = [f for f in hp if f.startswith("honeypot:respects the budget")]
    if peek:
        return "fail", peek[0][:200]
    if outcome is None:
        return "not_run", "no honeypot rerun recorded"
    if outcome != "ran":
        return "unverified", f"honeypot rerun {outcome}"
    if hp:
        return "unverified", hp[0][:200]
    return "pass", ""


def expected_random_best(scores, k):
    """Exact expected minimum score of K distinct rows drawn uniformly at random from SCORES.

    With scores sorted ascending s[0] <= ... <= s[n-1], the minimum of a random K-subset is s[i]
    with probability C(n-1-i, k-1) / C(n, k). No random draws are simulated."""
    s = sorted(scores)
    n = len(s)
    if not 1 <= k <= n:
        raise ValueError(f"k={k} outside 1..{n}")
    total = comb(n, k)
    return sum(s[i] * (comb(n - 1 - i, k - 1) / total) for i in range(n - k + 1))


def p_random_at_least_as_good(scores, k, best):
    """Probability that K uniformly random rows include one scoring <= BEST."""
    worse = sum(v > best for v in scores)
    return 1 - comb(worse, k) / comb(len(scores), k)


def selection_metrics(scores, order, k=None):
    """Score a recorded inspection ORDER with oracle d2h SCORES {row: d2h} (lower is better).

    Only the first K rows of the order count (default: all of it). Returns a dict of metrics.
    The scores are used only to grade rows the program already chose; they never build an order."""
    k = len(order) if k is None else k
    if not 1 <= k <= len(order):
        raise ValueError(f"cannot score the first {k} rows of an order of {len(order)}")
    picked = order[:k]
    if len(set(picked)) != len(picked) or any(r not in scores for r in picked):
        raise ValueError("order must list distinct candidate rows")
    vals = list(scores.values())
    best_row = min(picked, key=lambda r: (scores[r], picked.index(r)))
    best, lo, hi = scores[best_row], min(vals), max(vals)
    rng = hi - lo
    exp_rand = expected_random_best(vals, k)
    return {
        "k": k, "best_row": best_row, "best_d2h": best, "pool_best_d2h": lo, "pool_worst_d2h": hi,
        "d2h_range": rng, "regret": best - lo,
        "normalized_regret": (best - lo) / rng if rng > 0 else None,
        "best_rank": 1 + sum(v < best for v in vals),
        "expected_random_best": exp_rand, "advantage_vs_random": exp_rand - best,
        "p_random_at_least_as_good": p_random_at_least_as_good(vals, k, best),
    }


SELECTION_STEP = 7
SELECTION = {"cars": "CAR-14", "nasa": "NASA-14", "apache": "APACHE-14"}
COMMON_K = 10      # the smallest order length the frozen answer format allows, shared by all domains

SELECTION_FIELDS = [
    "model", "domain", "treatment", "trial", "skill_index", "request_id",
    "step_status", "step_runs", "step_semantic", "request_status",
    "check_runs", "check_timed_out", "answer_state", "answer_detail",
    "honeypot_gate", "honeypot_detail", "shuffled_gate",
    "n_candidates", "budget", "n_inspected", "within_budget",
    "best_row", "best_d2h", "pool_best_d2h", "pool_worst_d2h", "d2h_range",
    "regret", "normalized_regret", "best_rank", "expected_random_best", "advantage_vs_random",
    "p_random_at_least_as_good",
    "best_d2h_at_10", "regret_at_10", "expected_random_best_at_10", "advantage_at_10",
    "beats_random", "primary", "exclusion_reason", "notes",
    "check_entry", "check_crc32", "check_sha256", "semantic_entry", "semantic_sha256",
]

SELECTION_SUMMARY_FIELDS = [
    "scope", "domain", "treatment", "request_id", "budget",
    "attempted", "generation_failed", "execution_failed", "answer_missing", "answer_invalid_json",
    "answer_schema_error", "schema_valid", "honeypot_failed", "honeypot_unverified",
    "primary_n", "primary_rate", "beats_random_n", "conditional_beat_rate", "delivery_rate",
    "median_best_d2h", "median_regret", "median_normalized_regret", "median_advantage_vs_random",
    "median_advantage_at_10", "median_p_random_at_least_as_good", "median_n_inspected", "pool_best_d2h",
    "excluded_median_normalized_regret",
]

SELECTION_PROTOCOL = {
    "requests": {d: {"request_id": SELECTION[d], "kind": "choose_order", "step": SELECTION_STEP} for d in DOMAINS},
    "answer": "ANSWER <ID> {\"order\": [row numbers, first evaluated first]}; 10..budget distinct rows "
              "(validated by harness.answers.check_request)",
    "candidate_pool": "every row of the domain CSV (1-based file order)",
    "budget": "harness.oracle.budget(n) = min(30, n // 5): cars 30, nasa 18, apache 30",
    "score": "harness.oracle.d2h over all rows (min-max bounds over all rows, goal direction from the column "
             "suffix); lower is better; used only to grade rows the program already listed",
    "best_d2h": "lowest d2h among the rows in the recorded order",
    "regret": "best_d2h - lowest d2h in the candidate pool",
    "normalized_regret": "regret / (highest - lowest d2h in the pool); undefined when that range is 0",
    "expected_random_best": "exact expectation of the lowest d2h among k rows drawn uniformly without replacement, "
                            "k = number of rows in the order (no simulated draws)",
    "advantage_vs_random": "expected_random_best - best_d2h (positive = better than random)",
    "p_random_at_least_as_good": "probability that k random rows include one with d2h <= best_d2h",
    "common_k": f"the same metrics on the first {COMMON_K} rows of every order (the minimum length allowed), "
                "for comparison across domains and order lengths",
    "beats_random": "advantage_vs_random > 0 (analysis choice)",
    "primary_inclusion": "program ran; ANSWER accepted by harness.answers.check_request; honeypot rerun recorded "
                         "as passed",
    "honeypot_unverified_policy": "excluded from primary (analysis choice)",
    "shuffled_rerun": "recorded, not gating",
}


def extract_selection(archive=ARCHIVE, runs_csv=RUNS_CSV, requirements_csv=REQUIREMENTS_CSV, data_dir=DATA_DIR,
                      model=MODEL, domains=DOMAINS):
    """One row per attempted (domain, treatment, trial) for the Choose inspection-order request."""
    runs = _index(runs_csv, ("model", "domain", "treatment", "trial", "skill_index"))
    reqs = _index(requirements_csv, ("model", "domain", "treatment", "trial", "request_id"))
    rows = []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        root = archive_root(names, model)
        for domain in domains:
            rid = SELECTION[domain]
            if rid not in request_ids(domain, SELECTION_STEP) or A.REQUESTS[rid][0] != "choose_order":
                raise ValueError(f"{rid} is not the frozen choose_order request")
            data = O.load(Path(data_dir) / DATASETS[domain])
            scores = O.d2h(data)
            if len(scores) != data.n:
                raise ValueError(f"some {domain} rows have no d2h (missing goal values)")
            for t in TREATMENTS:
                for k in TRIALS:
                    rows.append(_selection_row(z, names, root, model, domain, t, k, rid, data, scores, runs, reqs))
    keys = [(r["domain"], r["treatment"], r["trial"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate trial records")
    return rows


def _selection_row(z, names, root, model, domain, t, k, rid, data, scores, runs, reqs):
    budget = O.budget(data.n)
    row, ctx, ans, sem = _attempt(
        z, names, root, model, domain, t, k, SELECTION_STEP, rid, data, runs, reqs, SELECTION_FIELDS,
        dict(n_candidates=data.n, budget=budget, honeypot_gate="not_run", shuffled_gate="not_run",
             pool_best_d2h=min(scores.values()), pool_worst_d2h=max(scores.values()),
             d2h_range=max(scores.values()) - min(scores.values())))
    if ctx is None:
        return row
    order = A.rows(data, A.need(ans, "order", "list"), "order")       # the checker's own reading
    row.update(n_inspected=len(order), within_budget=len(order) <= budget)
    if len(order) > budget:                                           # the frozen schema already forbids this
        row.update(exclusion_reason="answer_schema_error", answer_state="schema_error",
                   answer_detail=f"order has {len(order)} rows, budget is {budget}")
        return row
    m = selection_metrics(scores, order)
    m10 = selection_metrics(scores, order, COMMON_K)
    notes = []
    if m["normalized_regret"] is None:
        notes.append("d2h range is zero: normalized regret undefined")
    row.update(best_row=m["best_row"], best_d2h=m["best_d2h"], regret=m["regret"],
               normalized_regret="" if m["normalized_regret"] is None else m["normalized_regret"],
               best_rank=m["best_rank"], expected_random_best=m["expected_random_best"],
               advantage_vs_random=m["advantage_vs_random"],
               p_random_at_least_as_good=m["p_random_at_least_as_good"],
               best_d2h_at_10=m10["best_d2h"], regret_at_10=m10["regret"],
               expected_random_best_at_10=m10["expected_random_best"], advantage_at_10=m10["advantage_vs_random"],
               beats_random=m["advantage_vs_random"] > 0, notes="; ".join(notes))
    gate, detail = honeypot_gate(sem, rid)
    row.update(honeypot_gate=gate, honeypot_detail=detail, shuffled_gate=shuffled_gate(sem, rid))
    if gate == "fail":
        row["exclusion_reason"] = "honeypot_failed"
    elif gate != "pass":
        row["exclusion_reason"] = "honeypot_unverified"
    else:
        row["primary"] = True
    return row


def _med(xs):
    xs = [x for x in xs if x != "" and x is not None]
    return O.median(xs) if xs else ""


def _selection_cell(scope, domain, treatment, rid, sel):
    prim = [r for r in sel if r["primary"]]
    beats = sum(bool(r["beats_random"]) for r in prim)
    reasons = [r["exclusion_reason"] for r in sel]
    budgets, bests = {r["budget"] for r in sel}, {r["pool_best_d2h"] for r in sel}
    gated = [r for r in sel if r["exclusion_reason"] in ("honeypot_failed", "honeypot_unverified")]
    return {
        "scope": scope, "domain": domain, "treatment": treatment, "request_id": rid,
        "budget": budgets.pop() if len(budgets) == 1 else "",
        "attempted": len(sel),
        "generation_failed": reasons.count("generation_failed") + reasons.count("not_attempted"),
        "execution_failed": reasons.count("execution_failure") + reasons.count("execution_timeout"),
        "answer_missing": reasons.count("answer_missing"),
        "answer_invalid_json": reasons.count("answer_invalid_json"),
        "answer_schema_error": reasons.count("answer_schema_error"),
        "schema_valid": sum(r["answer_state"] == "valid" for r in sel),
        "honeypot_failed": reasons.count("honeypot_failed"),
        "honeypot_unverified": reasons.count("honeypot_unverified"),
        "primary_n": len(prim), "primary_rate": _rate(len(prim), len(sel)),
        "beats_random_n": beats, "conditional_beat_rate": _rate(beats, len(prim)),
        "delivery_rate": _rate(beats, len(sel)),
        "median_best_d2h": _med([r["best_d2h"] for r in prim]),
        "median_regret": _med([r["regret"] for r in prim]),
        "median_normalized_regret": _med([r["normalized_regret"] for r in prim]),
        "median_advantage_vs_random": _med([r["advantage_vs_random"] for r in prim]),
        "median_advantage_at_10": _med([r["advantage_at_10"] for r in prim]),
        "median_p_random_at_least_as_good": _med([r["p_random_at_least_as_good"] for r in prim]),
        "median_n_inspected": _med([r["n_inspected"] for r in prim]),
        "pool_best_d2h": bests.pop() if len(bests) == 1 else "",
        "excluded_median_normalized_regret": _med([r["normalized_regret"] for r in gated]),
    }


def summarize_selection(rows):
    out = []
    for d in DOMAINS:
        for t in TREATMENTS:
            sel = [r for r in rows if r["domain"] == d and r["treatment"] == t]
            if sel:
                out.append(_selection_cell("domain_treatment", d, t, SELECTION[d], sel))
    for d in DOMAINS:
        sel = [r for r in rows if r["domain"] == d]
        if sel:
            out.append(_selection_cell("domain", d, "all", SELECTION[d], sel))
    for t in TREATMENTS:
        sel = [r for r in rows if r["treatment"] == t]
        if sel:
            out.append(_selection_cell("treatment_all_domains", "all", t, "*-14", sel))
    return out


# ---- phase 3: Race (race_select and race_predict, skill 10) -------------------------

RACE_STEP = 10
# NASA lists its two Race requests in the opposite order to Cars and Apache (docs/requirements_nasa.md).
RACE = {
    "cars": {"select": "CAR-19", "predict": "CAR-20"},
    "nasa": {"select": "NASA-20", "predict": "NASA-19"},
    "apache": {"select": "APACHE-19", "predict": "APACHE-20"},
}


def check_failed(sem, request_id, prefix):
    """First failed-check text of REQUEST_ID that starts with PREFIX (e.g. 'base:verdict follows'), else ''."""
    req = next((r for r in sem.get("requests", []) if r.get("request_id") == request_id), None)
    if req is None:
        return ""
    return next((f[:200] for f in req.get("failed", []) if f.startswith(prefix)), "")


def race_selection_metrics(scores, ours, rand, budget):
    """Score a recorded race_select answer: OURS[i] and RAND[i] are the best rows found by the guided
    and the random arm in repeat i. Lower d2h is better. Repeats are summarised within the trial."""
    if len(ours) != len(rand) or not ours:
        raise ValueError("ours and random must have one row per repeat")
    g = [scores[r] for r in ours]
    x = [scores[r] for r in rand]
    wins = sum(a < b for a, b in zip(g, x))
    ties = sum(a == b for a, b in zip(g, x))
    n = len(g)
    exp_rand = expected_random_best(list(scores.values()), budget)
    mg, mr = O.median(g), O.median(x)
    return {
        "n_repeats": n, "median_guided_d2h": mg, "median_random_d2h": mr, "median_advantage": mr - mg,
        "mean_guided_d2h": O.mean(g), "mean_random_d2h": O.mean(x),
        "guided_wins": wins, "ties": ties, "random_wins": n - wins - ties,
        "win_fraction": wins / n, "tie_fraction": ties / n,
        "expected_random_best": exp_rand, "advantage_vs_analytic": exp_rand - mg,
        "random_arm_vs_analytic": O.mean(x) - exp_rand,
        "median_guided_regret": mg - min(scores.values()),
    }


RACE_SELECTION_FIELDS = [
    "model", "domain", "treatment", "trial", "skill_index", "request_id",
    "step_status", "step_runs", "step_semantic", "request_status",
    "check_runs", "check_timed_out", "answer_state", "answer_detail",
    "honeypot_gate", "honeypot_detail", "random_arm_gate", "random_arm_detail", "verdict_gate", "shuffled_gate",
    "n_candidates", "budget", "reported_budget", "budget_matches", "n_repeats",
    "median_guided_d2h", "median_random_d2h", "median_advantage", "mean_guided_d2h", "mean_random_d2h",
    "guided_wins", "ties", "random_wins", "win_fraction", "tie_fraction",
    "expected_random_best", "advantage_vs_analytic", "random_arm_vs_analytic",
    "pool_best_d2h", "median_guided_regret", "reported_verdict",
    "beats_random_analytic", "beats_random_empirical",
    "primary", "empirical_primary", "exclusion_reason", "notes",
    "check_entry", "check_crc32", "check_sha256", "semantic_entry", "semantic_sha256",
]

RACE_PREDICTION_FIELDS = [
    "model", "domain", "treatment", "trial", "skill_index", "request_id", "target", "target_direction",
    "step_status", "step_runs", "step_semantic", "request_status",
    "check_runs", "check_timed_out", "answer_state", "answer_detail",
    "leakage_gate", "leakage_detail", "verdict_gate", "shuffled_gate",
    "n_heldout", "model_mae", "baseline_mean", "baseline_mae", "mae_ratio", "mae_advantage", "beats_baseline",
    "rows_model_better", "rows_model_better_frac",
    "reported_baseline", "reported_baseline_matches", "reported_mae_model", "reported_mae_model_matches",
    "reported_mae_baseline", "reported_mae_baseline_matches", "reported_verdict",
    "primary", "exclusion_reason",
    "check_entry", "check_crc32", "check_sha256", "semantic_entry", "semantic_sha256",
]

RACE_PROTOCOL = {
    "requests": {d: dict(v, step=RACE_STEP) for d, v in RACE.items()},
    "note": "NASA-19 is the prediction race and NASA-20 the selection race; Cars and Apache are the other way round",
    "selection": {
        "answer": "{\"budget\": b, \"repeats\": >= 20, \"ours\": [best row per repeat], \"random\": [best of b random "
                  "rows per repeat], \"verdict\": ...}; schema validated by harness.answers.check_request",
        "score": "harness.oracle.d2h over all rows of the CSV (lower is better)",
        "empirical": "archived per-repeat rows of both arms: medians, median_advantage = median random d2h - median "
                     "guided d2h (positive = guided better), win_fraction = repeats where guided d2h < random d2h "
                     "(pairs taken by repeat index as recorded)",
        "analytical": "expected_random_best = exact expected best d2h of `budget` rows drawn uniformly without "
                      "replacement from all rows; advantage_vs_analytic = expected_random_best - median guided d2h",
        "primary_inclusion": "program ran; schema valid; honeypot rerun passed; reported budget == "
                             "harness.oracle.budget(n)",
        "empirical_inclusion": "primary and the frozen check 'random baseline is really random' passed on the base run",
        "delivered": "primary and advantage_vs_analytic > 0 (analysis choice)",
        "unverifiable": "the rows each arm evaluated are not archived, so it cannot be checked that `ours` and "
                        "`random` are really the best of what each arm saw",
    },
    "prediction": {
        "metric": "same definitions as phase 1 (every-fifth-row hold-out, training-mean baseline, mae_ratio)",
        "verification": "model and baseline MAE recomputed from the recorded predictions and the CSV; the program's "
                        "reported baseline, mae_model and mae_baseline are compared (rel. 0.5%, 1%, 1% as the frozen "
                        "checker does) but never used",
        "primary_inclusion": "program ran; schema valid; mask leakage rerun passed",
        "separate_from_phase1": "the Guess request (skill 2) and the Race request (skill 10) are never pooled",
    },
}


def _race_attempts(archive, runs_csv, requirements_csv, data_dir, model, domains, kind, make_row):
    runs = _index(runs_csv, ("model", "domain", "treatment", "trial", "skill_index"))
    reqs = _index(requirements_csv, ("model", "domain", "treatment", "trial", "request_id"))
    rows = []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        root = archive_root(names, model)
        for domain in domains:
            rid = RACE[domain][kind]
            want = {"select": "race_select", "predict": "race_predict"}[kind]
            if rid not in request_ids(domain, RACE_STEP) or A.REQUESTS[rid][0] != want:
                raise ValueError(f"{rid} is not the frozen {want} request")
            data = O.load(Path(data_dir) / DATASETS[domain])
            for t in TREATMENTS:
                for k in TRIALS:
                    rows.append(make_row(z, names, root, model, domain, t, k, rid, data, runs, reqs))
    keys = [(r["domain"], r["treatment"], r["trial"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate trial records")
    return rows


def _race_selection_row(z, names, root, model, domain, t, k, rid, data, runs, reqs):
    scores = O.d2h(data)
    budget = O.budget(data.n)
    row, ctx, ans, sem = _attempt(
        z, names, root, model, domain, t, k, RACE_STEP, rid, data, runs, reqs, RACE_SELECTION_FIELDS,
        dict(n_candidates=data.n, budget=budget, pool_best_d2h=min(scores.values()),
             expected_random_best=expected_random_best(list(scores.values()), budget),
             honeypot_gate="not_run", random_arm_gate="not_run", verdict_gate="not_run", shuffled_gate="not_run",
             empirical_primary=False))
    if ctx is None:
        return row
    ours = A.rows(data, A.need(ans, "ours", "list"), "ours", distinct=False)     # the checker's own reading
    rand = A.rows(data, A.need(ans, "random", "list"), "random", distinct=False)
    rb = A.num(A.need(ans, "budget"))
    m = race_selection_metrics(scores, ours, rand, budget)
    row.update(m)
    row.update(reported_budget=rb, budget_matches=rb == budget, reported_verdict=ans.get("verdict", ""),
               beats_random_analytic=m["advantage_vs_analytic"] > 0, beats_random_empirical=m["median_advantage"] > 0)
    gate, detail = honeypot_gate(sem, rid)
    arm = check_failed(sem, rid, "base:random baseline is really random")
    verdict = check_failed(sem, rid, "base:verdict follows from the results")
    row.update(honeypot_gate=gate, honeypot_detail=detail, shuffled_gate=shuffled_gate(sem, rid),
               random_arm_gate="fail" if arm else "pass", random_arm_detail=arm,
               verdict_gate="fail" if verdict else "pass")
    if gate == "fail":
        row["exclusion_reason"] = "honeypot_failed"
    elif gate != "pass":
        row["exclusion_reason"] = "honeypot_unverified"
    elif rb != budget:
        row["exclusion_reason"] = "budget_mismatch"
    else:
        row["primary"] = True
        row["empirical_primary"] = not arm
    if row["primary"] and arm:
        row["notes"] = "random arm failed the frozen plausibility check: excluded from the empirical comparison"
    return row


def _race_prediction_row(z, names, root, model, domain, t, k, rid, data, runs, reqs):
    target = A.REQUESTS[rid][1]["target"]
    direction = dict(data.goals())[target]
    base_mean, base_mae, held = baseline(data, target)
    row, ctx, ans, sem = _attempt(
        z, names, root, model, domain, t, k, RACE_STEP, rid, data, runs, reqs, RACE_PREDICTION_FIELDS,
        dict(target=target, target_direction=direction, baseline_mean=base_mean, baseline_mae=base_mae,
             n_heldout=len(held), leakage_gate="not_run", verdict_gate="not_run", shuffled_gate="not_run"))
    if ctx is None:
        return row
    preds = A._heldout_preds(ctx)
    mae = model_mae(data, target, preds)
    ratio, why = mae_ratio(mae, base_mae)
    better = sum(abs(preds[r] - data.value(r, target)) < abs(base_mean - data.value(r, target)) for r in held)
    rep_b, rep_m, rep_mb = (A.num(A.need(ans, key)) for key in ("baseline", "mae_model", "mae_baseline"))
    row.update(model_mae=mae, mae_ratio="" if ratio is None else ratio, mae_advantage=base_mae - mae,
               beats_baseline="" if ratio is None else mae < base_mae,
               rows_model_better=better, rows_model_better_frac=better / len(held),
               reported_baseline=rep_b, reported_baseline_matches=O.close(rep_b, base_mean, rel=0.005),
               reported_mae_model=rep_m, reported_mae_model_matches=O.close(rep_m, mae, rel=0.01),
               reported_mae_baseline=rep_mb, reported_mae_baseline_matches=O.close(rep_mb, base_mae, rel=0.01),
               reported_verdict=ans.get("verdict", ""))
    gate, detail = leakage_gate(sem, rid, target)
    verdict = check_failed(sem, rid, "base:verdict follows from the errors")
    row.update(leakage_gate=gate, leakage_detail=detail, shuffled_gate=shuffled_gate(sem, rid),
               verdict_gate="fail" if verdict else "pass")
    if gate == "fail":
        row["exclusion_reason"] = "leakage_failed"
    elif gate != "pass":
        row["exclusion_reason"] = "leakage_unverified"
    elif why:
        row["exclusion_reason"] = why
    else:
        row["primary"] = True
    return row


def extract_race_selection(archive=ARCHIVE, runs_csv=RUNS_CSV, requirements_csv=REQUIREMENTS_CSV, data_dir=DATA_DIR,
                           model=MODEL, domains=DOMAINS):
    return _race_attempts(archive, runs_csv, requirements_csv, data_dir, model, domains, "select",
                          _race_selection_row)


def extract_race_prediction(archive=ARCHIVE, runs_csv=RUNS_CSV, requirements_csv=REQUIREMENTS_CSV, data_dir=DATA_DIR,
                            model=MODEL, domains=DOMAINS):
    return _race_attempts(archive, runs_csv, requirements_csv, data_dir, model, domains, "predict",
                          _race_prediction_row)


RACE_SUMMARY_FIELDS = [
    "metric", "scope", "domain", "treatment", "request_id",
    "attempted", "generation_failed", "execution_failed", "answer_missing", "answer_invalid_json",
    "answer_schema_error", "schema_valid", "gate_failed", "gate_unverified", "budget_mismatch",
    "primary_n", "primary_rate", "beats_n", "conditional_beat_rate", "delivery_rate",
    # selection
    "budget", "expected_random_best", "median_guided_d2h", "median_random_d2h", "median_advantage",
    "median_win_fraction", "median_advantage_vs_analytic", "empirical_n", "empirical_beats_n",
    # prediction
    "baseline_mae", "median_model_mae", "median_mae_ratio", "min_mae_ratio", "max_mae_ratio", "median_mae_advantage",
]


def _race_counts(sel, gate_fail, gate_unv):
    reasons = [r["exclusion_reason"] for r in sel]
    return {
        "attempted": len(sel),
        "generation_failed": reasons.count("generation_failed") + reasons.count("not_attempted"),
        "execution_failed": reasons.count("execution_failure") + reasons.count("execution_timeout"),
        "answer_missing": reasons.count("answer_missing"),
        "answer_invalid_json": reasons.count("answer_invalid_json"),
        "answer_schema_error": reasons.count("answer_schema_error"),
        "schema_valid": sum(r["answer_state"] == "valid" for r in sel),
        "gate_failed": reasons.count(gate_fail), "gate_unverified": reasons.count(gate_unv),
        "budget_mismatch": reasons.count("budget_mismatch"),
    }


def _one(values):
    vals = {v for v in values}
    return vals.pop() if len(vals) == 1 else ""


def _race_sel_cell(scope, domain, treatment, rid, sel):
    prim = [r for r in sel if r["primary"]]
    emp = [r for r in prim if r["empirical_primary"]]
    beats = sum(bool(r["beats_random_analytic"]) for r in prim)
    out = {f: "" for f in RACE_SUMMARY_FIELDS}
    out.update(metric="selection", scope=scope, domain=domain, treatment=treatment, request_id=rid)
    out.update(_race_counts(sel, "honeypot_failed", "honeypot_unverified"))
    out.update(primary_n=len(prim), primary_rate=_rate(len(prim), len(sel)), beats_n=beats,
               conditional_beat_rate=_rate(beats, len(prim)), delivery_rate=_rate(beats, len(sel)),
               budget=_one(r["budget"] for r in sel), expected_random_best=_one(r["expected_random_best"] for r in sel),
               median_guided_d2h=_med([r["median_guided_d2h"] for r in prim]),
               median_advantage_vs_analytic=_med([r["advantage_vs_analytic"] for r in prim]),
               empirical_n=len(emp), empirical_beats_n=sum(bool(r["beats_random_empirical"]) for r in emp),
               median_random_d2h=_med([r["median_random_d2h"] for r in emp]),
               median_advantage=_med([r["median_advantage"] for r in emp]),
               median_win_fraction=_med([r["win_fraction"] for r in emp]))
    return out


def _race_pred_cell(scope, domain, treatment, rid, sel):
    prim = [r for r in sel if r["primary"]]
    ratios = [r["mae_ratio"] for r in prim]
    beats = sum(bool(r["beats_baseline"]) for r in prim)
    out = {f: "" for f in RACE_SUMMARY_FIELDS}
    out.update(metric="prediction", scope=scope, domain=domain, treatment=treatment, request_id=rid)
    out.update(_race_counts(sel, "leakage_failed", "leakage_unverified"))
    out.update(primary_n=len(prim), primary_rate=_rate(len(prim), len(sel)), beats_n=beats,
               conditional_beat_rate=_rate(beats, len(prim)), delivery_rate=_rate(beats, len(sel)),
               baseline_mae=_one(r["baseline_mae"] for r in sel),
               median_model_mae=_med([r["model_mae"] for r in prim]),
               median_mae_ratio=_med(ratios), min_mae_ratio=min(ratios) if ratios else "",
               max_mae_ratio=max(ratios) if ratios else "",
               median_mae_advantage=_med([r["mae_advantage"] for r in prim]))
    return out


def summarize_race(sel_rows, pred_rows):
    out = []
    for kind, rows, cell in (("select", sel_rows, _race_sel_cell), ("predict", pred_rows, _race_pred_cell)):
        for d in DOMAINS:
            for t in TREATMENTS:
                s = [r for r in rows if r["domain"] == d and r["treatment"] == t]
                if s:
                    out.append(cell("domain_treatment", d, t, RACE[d][kind], s))
        for d in DOMAINS:
            s = [r for r in rows if r["domain"] == d]
            if s:
                out.append(cell("domain", d, "all", RACE[d][kind], s))
        for t in TREATMENTS:
            s = [r for r in rows if r["treatment"] == t]
            if s:
                out.append(cell("treatment_all_domains", "all", t, "race_" + kind, s))
    return out


# ---- summaries ---------------------------------------------------------------------

def _rate(a, b):
    return a / b if b else ""


def _cell(scope, domain, treatment, rid, sel):
    prim = [r for r in sel if r["primary"]]
    ratios = [r["mae_ratio"] for r in prim]
    beats = sum(bool(r["beats_baseline"]) for r in prim)
    reasons = [r["exclusion_reason"] for r in sel]
    bases = {r["baseline_mae"] for r in sel}
    return {
        "scope": scope, "domain": domain, "treatment": treatment, "request_id": rid,
        "attempted": len(sel),
        "generation_failed": reasons.count("generation_failed") + reasons.count("not_attempted"),
        "execution_failed": reasons.count("execution_failure") + reasons.count("execution_timeout"),
        "answer_missing": reasons.count("answer_missing"),
        "answer_invalid_json": reasons.count("answer_invalid_json"),
        "answer_schema_error": reasons.count("answer_schema_error"),
        "schema_valid": sum(r["answer_state"] == "valid" for r in sel),
        "leakage_failed": reasons.count("leakage_failed"),
        "leakage_unverified": reasons.count("leakage_unverified"),
        "primary_n": len(prim),
        "primary_rate": _rate(len(prim), len(sel)),
        "beats_baseline_n": beats,
        "conditional_beat_rate": _rate(beats, len(prim)),
        "delivery_rate": _rate(beats, len(sel)),
        "median_mae_ratio": O.median(ratios) if ratios else "",
        "min_mae_ratio": min(ratios) if ratios else "",
        "max_mae_ratio": max(ratios) if ratios else "",
        "median_model_mae": O.median([r["model_mae"] for r in prim]) if prim else "",
        "baseline_mae": bases.pop() if len(bases) == 1 else "",
    }


def summarize(rows):
    out = []
    for d in DOMAINS:
        for t in TREATMENTS:
            sel = [r for r in rows if r["domain"] == d and r["treatment"] == t]
            if sel:
                out.append(_cell("domain_treatment", d, t, PREDICTION[d][0], sel))
    for d in DOMAINS:
        sel = [r for r in rows if r["domain"] == d]
        if sel:
            out.append(_cell("domain", d, "all", PREDICTION[d][0], sel))
    for t in TREATMENTS:
        sel = [r for r in rows if r["treatment"] == t]
        if sel:
            out.append(_cell("treatment_all_domains", "all", t, "*-03", sel))
    return out


# ---- writing -----------------------------------------------------------------------

def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def to_csv(records, fields):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in records:
        w.writerow({f: _fmt(r.get(f, "")) for f in fields})
    return buf.getvalue()


def guard_out(out_dir, frozen=FROZEN_DIR):
    """Refuse any output location inside the frozen v2 results."""
    out, frozen = Path(out_dir).resolve(), Path(frozen).resolve()
    if out == frozen or out.is_relative_to(frozen):
        raise ValueError(f"refusing to write inside the frozen results directory {frozen}")
    return out


def _num(v, spec=".3f"):
    return "—" if v == "" or v is None else format(v, spec)


def _pct(v):
    return "—" if v == "" else f"{100 * v:.0f}%"


def report(rows, summary, manifest):
    cells = [s for s in summary if s["scope"] == "domain_treatment"]
    lines = [
        "# External metric, phase 1: prediction MAE vs the historical-mean baseline",
        "",
        "**Status: post-observation analysis.** This metric was added after the Task Talk, after the v2 "
        "T1–T6 results had been seen. It is not part of the pre-registered plan in `docs/hypotheses.md` "
        "and does not change its win conditions.",
        "",
        "## What is measured",
        "",
        "- **Requests:** CAR-03 (`Mpg+`), NASA-03 (`Effort-`), APACHE-03 (`Performance-`): the Guess step "
        "(skill 2) of every v2 trial of T1–T6, model `gpt-oss-20b`.",
        "- **Held-out rows:** the frozen protocol (`harness.oracle.held_out`): every row whose number is "
        "divisible by 5. Cars 79 rows, NASA 18, Apache 38.",
        "- **Model MAE:** recomputed from the program's recorded held-out predictions against the CSV. "
        "The program's own reported `mae` is not used (it is kept in `reported_mae` for reference).",
        "- **Baseline MAE:** the mean target value of the non-held-out rows, scored on the same held-out "
        "rows. This is the baseline the frozen checker uses for \"beats the historical mean\".",
        "- **MAE ratio** = model MAE / baseline MAE. Below 1 means the program beats the baseline. MAE is "
        "an error, so lower is better whatever the goal's direction.",
        "- **Source:** archived base-run output only. No generated program was rerun or retrained, and no "
        "LLM was called.",
        "",
        "## Inclusion rules",
        "",
        "Every attempted trial is a row in `prediction_mae.csv`. A row enters the primary metric only if:",
        "",
        "1. the step produced a program and it ran on the real CSV,",
        "2. it printed an `ANSWER` line for the request that the frozen checker accepts (predictions for "
        "exactly the held-out rows), and",
        "3. the archived `mask` rerun shows the predictions did not move when only the held-out target values "
        "were changed (`leakage_gate = pass`).",
        "",
        "A failed leakage check excludes the row (`leakage_failed`). So does a mask rerun that crashed, "
        "timed out or gave no valid answer (`leakage_unverified`): leakage could not be ruled out. This "
        "second rule is a choice made for this analysis. Schema-valid rows excluded by the gate still "
        "show their MAE in the CSV for transparency, but never count in the summaries.",
        "",
        "The shuffled-rows rerun is recorded (`shuffled_gate`) but does not gate this metric.",
        "",
        "Two rates are reported, and they answer different questions:",
        "",
        "- **Conditional accuracy:** the median MAE ratio over primary rows only. It says how good the "
        "predictions are *when* a treatment delivers them.",
        "- **Delivery rate:** primary rows that beat the baseline ÷ all attempted trials. Failed, missing and "
        "excluded attempts count as not delivered, so a treatment cannot look better by crashing on hard "
        "trials.",
        "",
        "## Results by domain and treatment",
        "",
        "n = trials. Each cell has 3 attempted trials. \"—\" means no primary value.",
        "",
        "| domain | treatment | attempted | gen. failed | exec. failed | no / bad answer | schema valid | "
        "leakage failed | leakage unverified | primary n | median MAE ratio | min–max ratio | beats baseline "
        "| delivery rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for s in cells:
        bad = s["answer_missing"] + s["answer_invalid_json"] + s["answer_schema_error"]
        rng = "—" if s["min_mae_ratio"] == "" else f"{s['min_mae_ratio']:.3f}–{s['max_mae_ratio']:.3f}"
        lines.append(
            f"| {s['domain']} | {s['treatment']} | {s['attempted']} | {s['generation_failed']} | "
            f"{s['execution_failed']} | {bad} | {s['schema_valid']} | {s['leakage_failed']} | "
            f"{s['leakage_unverified']} | {s['primary_n']} | {_num(s['median_mae_ratio'])} | {rng} | "
            f"{s['beats_baseline_n']}/{s['primary_n']} | {_pct(s['delivery_rate'])} |")
    lines += ["", "## By domain", "",
              "| domain | request | baseline MAE | attempted | primary n | median MAE ratio | median model MAE "
              "| delivery rate |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in [s for s in summary if s["scope"] == "domain"]:
        lines.append(f"| {s['domain']} | {s['request_id']} | {_num(s['baseline_mae'])} | {s['attempted']} | "
                     f"{s['primary_n']} | {_num(s['median_mae_ratio'])} | {_num(s['median_model_mae'])} | "
                     f"{_pct(s['delivery_rate'])} |")
    lines += ["", "## By treatment, all three domains", "",
              "MAE ratios are unitless, so they can be listed together, but the three domains have different "
              "targets and sample sizes. Treat this table as descriptive.", "",
              "| treatment | attempted | primary n | median MAE ratio | beats baseline | delivery rate |",
              "|---|---:|---:|---:|---:|---:|"]
    for s in [s for s in summary if s["scope"] == "treatment_all_domains"]:
        lines.append(f"| {s['treatment']} | {s['attempted']} | {s['primary_n']} | {_num(s['median_mae_ratio'])} | "
                     f"{s['beats_baseline_n']}/{s['primary_n']} | {_pct(s['delivery_rate'])} |")
    lines += ["", "## Per-trial values", "",
              "| domain | treatment | trial | exclusion reason | model MAE | baseline MAE | MAE ratio | leakage | "
              "shuffled |",
              "|---|---|---:|---|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['domain']} | {r['treatment']} | {r['trial']} | {r['exclusion_reason'] or 'included'} | "
                     f"{_num(r['model_mae'])} | {_num(r['baseline_mae'])} | {_num(r['mae_ratio'])} | "
                     f"{r['leakage_gate']} | {r['shuffled_gate']} |")
    lines += [
        "", "## Limitations", "",
        "- **Three trials per cell, often fewer usable.** No statistical comparison between treatments is made "
        "in this phase; with n ≤ 3, `same()` would have very little power.",
        "- **One fixed split.** The every-fifth-row hold-out is the frozen protocol shared by all treatments, so "
        "the comparison is fair, but there is no resampling of the split.",
        "- **Baseline is per domain, not per trial.** It depends only on the CSV, so it is the same for every "
        "treatment in a domain.",
        "- **Skill 2 only.** The Race request (CAR-20, NASA-19, APACHE-20) also reports MAE; it comes from the same "
        "trials and is not pooled with this one.",
        "- **T7 is not included** in this phase.",
        "", "## Provenance", "",
        f"- Archive: `{manifest['inputs']['archive']['path']}` (sha256 `{manifest['inputs']['archive']['sha256']}`)",
        f"- Code commit: `{manifest['git']['commit']}` on `{manifest['git']['branch']}`"
        f"{' (uncommitted changes present)' if manifest['git']['dirty'] else ''}",
        f"- Command: `{manifest['command']}`",
        "- Full input and output hashes: `manifest.json`.",
        "",
    ]
    return "\n".join(lines)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _file_info(path):
    b = Path(path).read_bytes()
    try:
        shown = str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        shown = str(path)
    return {"path": shown, "sha256": _sha(b), "bytes": len(b)}


def build_manifest(archive, runs_csv, requirements_csv, data_dir, argv):
    with zipfile.ZipFile(archive) as z:
        root = archive_root(z.namelist(), MODEL)
        copies = {}
        for name, path in (("runs.csv", runs_csv), ("requirements.csv", requirements_csv)):
            inside = f"{root}/results/{name}"
            copies[name] = (inside in z.namelist()
                            and _sha(z.read(inside)) == _sha(Path(path).read_bytes()))
    return {
        "analysis": "external metrics phase 1: prediction MAE vs historical-mean baseline; phase 2: selection quality under a fixed inspection budget; "
                    "phase 3: Race (guided vs random selection, model vs historical-mean prediction)",
        "status": "post-observation (added after the Task Talk; not pre-registered)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m harness.external" + (" " + " ".join(argv) if argv else ""),
        "model": MODEL,
        "protocol": {
            "requests": {d: {"request_id": p[0], "target": p[1], "direction": p[2]} for d, p in PREDICTION.items()},
            "held_out": "rows whose 1-based number is divisible by 5 (harness.oracle.held_out)",
            "baseline": "mean target of the non-held-out rows, scored on the held-out rows",
            "metric": "mae_ratio = model_mae / baseline_mae; undefined (excluded) when baseline_mae == 0",
            "beats_baseline": "model_mae < baseline_mae (strict, as in the frozen checker)",
            "primary_inclusion": "program ran; ANSWER accepted by harness.answers.check_request; "
                                 "mask leakage rerun recorded as passed",
            "leakage_unverified_policy": "excluded from primary (analysis choice, documented in the report)",
            "shuffled_rerun": "recorded, not gating",
            "llm_calls": 0,
            "programs_rerun": 0,
            "selection": SELECTION_PROTOCOL,
            "race": RACE_PROTOCOL,
        },
        "inputs": {
            "archive": _file_info(archive),
            "runs_csv": _file_info(runs_csv),
            "requirements_csv": _file_info(requirements_csv),
            "archive_csv_copies_identical": copies,
            "datasets": {d: _file_info(Path(data_dir) / DATASETS[d]) for d in DOMAINS},
            "code": {m: _file_info(HARNESS_DIR / f"{m}.py") for m in ("external", "answers", "oracle", "semantic")},
        },
        "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
                "dirty": bool(_git("status", "--porcelain"))},
        "environment": {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
                        "platform": platform.platform()},
    }


def selection_report(rows, summary, manifest):
    cells = [s for s in summary if s["scope"] == "domain_treatment"]
    valid = [r for r in rows if r["answer_state"] == "valid"]
    peek = [r for r in valid if r["honeypot_gate"] == "fail"]
    lines = [
        "# External metric, phase 2: selection quality under a fixed inspection budget",
        "",
        "**Status: post-observation analysis.** This metric was added after the Task Talk, after the v2 "
        "T1–T6 results had been seen. It is not part of the pre-registered plan in `docs/hypotheses.md` "
        "and does not change its win conditions.",
        "",
        "## What is measured",
        "",
        "- **Requests:** CAR-14, NASA-14, APACHE-14 (Choose, skill 7): \"recommend which rows should be evaluated "
        "first\" under a limited inspection budget. Every v2 trial of T1–T6, model `gpt-oss-20b`.",
        "- **Recorded answer:** `{\"order\": [row numbers]}`, the rows the program would inspect, first one first. "
        "The frozen checker requires 10 to `budget` distinct rows, numbered 1..n in file order.",
        "- **Budget and candidate pool:** the frozen `harness.oracle.budget(n)` = min(30, n // 5): Cars 30 of 398 "
        "rows, NASA 18 of 93, Apache 30 of 192. Every row of the CSV is a candidate.",
        "- **Score:** the frozen `harness.oracle.d2h` (distance to heaven over all goals, goal direction from the "
        "column suffix; lower is better). It only grades rows the program already listed; it never builds or "
        "changes an order.",
        "- **Best selected d2h:** the lowest d2h among the rows in the order.",
        "- **Regret:** best selected d2h − best d2h in the whole pool. **Normalized regret:** regret ÷ (worst − "
        "best d2h in the pool); 0 = found the best row, 1 = found only the worst.",
        "- **Advantage over random:** the exact expected best d2h of the same number of rows drawn uniformly at "
        "random, minus the best selected d2h. Positive = better than random. It is computed in closed form, so "
        "no random draws are simulated and none are counted as trials.",
        "- **P(random ≥ as good):** the probability that the same number of random rows would include one at "
        "least as good as the program's best.",
        f"- **At {COMMON_K} rows:** the same metrics for the first {COMMON_K} rows of every order. Programs "
        "chose different order lengths (10 to the budget), and budgets differ by domain; this column compares "
        "everyone at the same inspection count.",
        "",
        "## Inclusion rules",
        "",
        "Every attempted trial is a row in `selection_quality.csv`. A row enters the primary metric only if the "
        "program ran, its `ANSWER` passed the frozen checker's format, and the archived **honeypot** rerun "
        "passed. The honeypot copy plants six rows with ideal goals but middling inputs; a program that puts "
        "them early in its order is reading goal values it has not paid to inspect (budget peeking).",
        "",
        "- `honeypot_failed`: the frozen check \"respects the budget\" failed. Excluded.",
        "- `honeypot_unverified`: the honeypot rerun crashed, timed out or gave no valid answer. Excluded "
        "(analysis choice: peeking could not be ruled out).",
        "- The shuffled-rows rerun is recorded (`shuffled_gate`) but does not gate this metric.",
        "",
        "**Delivery rate** = primary trials whose selection beats random (advantage > 0) ÷ all 3 attempted "
        "trials. \"Beats random\" (advantage > 0) is a threshold chosen for this analysis.",
        "",
        "## Results by domain and treatment",
        "",
        "Each cell has 3 attempted trials. \"—\" means no primary value. Medians are over primary trials only.",
        "",
        "| domain | treatment | budget | attempted | exec. failed | no / bad answer | schema valid | "
        "honeypot failed | honeypot unverified | primary n | median best d2h | median regret | median "
        "normalized regret | median advantage vs random | median advantage at 10 | delivery rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in cells:
        bad = s["answer_missing"] + s["answer_invalid_json"] + s["answer_schema_error"]
        failed = s["execution_failed"] + s["generation_failed"]
        lines.append(
            f"| {s['domain']} | {s['treatment']} | {s['budget']} | {s['attempted']} | {failed} | {bad} | "
            f"{s['schema_valid']} | {s['honeypot_failed']} | {s['honeypot_unverified']} | {s['primary_n']} | "
            f"{_num(s['median_best_d2h'])} | {_num(s['median_regret'])} | {_num(s['median_normalized_regret'])} | "
            f"{_num(s['median_advantage_vs_random'], '+.3f')} | {_num(s['median_advantage_at_10'], '+.3f')} | "
            f"{_pct(s['delivery_rate'])} |")
    lines += ["", "\"exec. failed\" includes steps that produced no program (`parse_failed`).", "",
              "## By domain", "",
              "| domain | request | budget | pool best d2h | attempted | schema valid | honeypot failed | primary n | "
              "median normalized regret | median advantage vs random | delivery rate |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in [s for s in summary if s["scope"] == "domain"]:
        lines.append(f"| {s['domain']} | {s['request_id']} | {s['budget']} | {_num(s['pool_best_d2h'])} | "
                     f"{s['attempted']} | {s['schema_valid']} | {s['honeypot_failed']} | {s['primary_n']} | "
                     f"{_num(s['median_normalized_regret'])} | {_num(s['median_advantage_vs_random'], '+.3f')} | "
                     f"{_pct(s['delivery_rate'])} |")
    lines += ["", "## By treatment, all three domains", "",
              "Normalized regret and the advantage are on the d2h scale of each domain; the domains differ in "
              "size, goals and budget. Treat this table as descriptive.", "",
              "| treatment | attempted | schema valid | honeypot failed | primary n | median normalized regret | "
              "median advantage at 10 | delivery rate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in [s for s in summary if s["scope"] == "treatment_all_domains"]:
        lines.append(f"| {s['treatment']} | {s['attempted']} | {s['schema_valid']} | {s['honeypot_failed']} | "
                     f"{s['primary_n']} | {_num(s['median_normalized_regret'])} | "
                     f"{_num(s['median_advantage_at_10'], '+.3f')} | {_pct(s['delivery_rate'])} |")
    lines += [
        "", "## Budget peeking", "",
        f"{len(peek)} of the {len(valid)} schema-valid orders failed the honeypot check. Their selections are kept "
        "in the CSV (`primary = false`) but excluded from every summary above. For context only, the table "
        "compares their scores on the real data with the primary orders. A program that reads unpaid goal "
        "values can look better than an honest one, so the excluded values do not measure selection skill.", "",
        "| domain | schema valid | honeypot failed | median normalized regret, excluded | median normalized regret, "
        "primary | median advantage vs random, excluded | median advantage vs random, primary |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for d in DOMAINS:
        v = [r for r in valid if r["domain"] == d]
        if not v:
            continue
        ex = [r for r in v if r["honeypot_gate"] == "fail"]
        pr = [r for r in v if r["primary"]]
        lines.append(f"| {d} | {len(v)} | {len(ex)} | {_num(_med([r['normalized_regret'] for r in ex]))} | "
                     f"{_num(_med([r['normalized_regret'] for r in pr]))} | "
                     f"{_num(_med([r['advantage_vs_random'] for r in ex]), '+.3f')} | "
                     f"{_num(_med([r['advantage_vs_random'] for r in pr]), '+.3f')} |")
    lines += [
        "", "## Per-trial values", "",
        "| domain | treatment | trial | exclusion reason | inspected | best row | best d2h | regret | normalized "
        "regret | expected random best | advantage | P(random ≥ as good) | honeypot |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(
            f"| {r['domain']} | {r['treatment']} | {r['trial']} | {r['exclusion_reason'] or 'included'} | "
            f"{r['n_inspected'] or '—'} | {r['best_row'] or '—'} | {_num(r['best_d2h'])} | {_num(r['regret'])} | "
            f"{_num(r['normalized_regret'])} | {_num(r['expected_random_best'])} | "
            f"{_num(r['advantage_vs_random'], '+.3f')} | {_num(r['p_random_at_least_as_good'], '.2f')} | "
            f"{r['honeypot_gate']} |")
    lines += [
        "", "## Limitations", "",
        "- **Three trials per cell, often fewer usable.** No statistical comparison between treatments is made; "
        "with n ≤ 3, `same()` would have very little power.",
        "- **Budgets differ by domain** (30, 18, 30 rows) and pools differ in size (398, 93, 192), so a regret of "
        f"the same size is not equally hard in every domain. The at-{COMMON_K} columns and normalized regret "
        "reduce, but do not remove, this difference.",
        "- **The order is scored, not replayed.** The archive records only the final order. Whether each choice "
        "used only goals of rows already placed is known only through the honeypot gate.",
        "- **The honeypot gate is a test, not a proof.** It checks the first budget(n − 6) rows of the order on a "
        "planted copy of the data. Passing it does not prove the program never peeked.",
        "- **d2h is one overall score.** The frozen checker also accepts z-score sums and rank sums for "
        "\"overall\"; on NASA, whose goals conflict, they can disagree. This metric uses d2h only.",
        "- **The random baseline is analytical.** It is the exact expectation for uniform sampling without "
        "replacement, not a simulation, and it says nothing about variance between LLM trials.",
        "- **T7 is not included** in this phase.",
        "", "## Provenance", "",
        f"- Archive: `{manifest['inputs']['archive']['path']}` (sha256 `{manifest['inputs']['archive']['sha256']}`)",
        f"- Code commit: `{manifest['git']['commit']}` on `{manifest['git']['branch']}`"
        f"{' (uncommitted changes present)' if manifest['git']['dirty'] else ''}",
        f"- Command: `{manifest['command']}`",
        "- Full input and output hashes: `manifest.json`.",
        "",
    ]
    return "\n".join(lines)


def race_report(sel_rows, pred_rows, summary, manifest):
    cells = lambda metric: [s for s in summary if s["metric"] == metric and s["scope"] == "domain_treatment"]
    lines = [
        "# External metric, phase 3: Race — guided vs random selection, model vs historical-mean prediction",
        "",
        "**Status: post-observation analysis.** These metrics were added after the Task Talk, after the v2 "
        "T1–T6 results had been seen. They are not part of the pre-registered plan in `docs/hypotheses.md` "
        "and do not change its win conditions.",
        "",
        "## Requests",
        "",
        "Race is skill 10. Each attempted step holds two requests, and NASA numbers them the other way round:",
        "",
        "| domain | selection race | prediction race |",
        "|---|---|---|",
    ]
    lines += [f"| {d} | {RACE[d]['select']} | {RACE[d]['predict']} |" for d in DOMAINS]
    lines += [
        "",
        "The prediction race is kept separate from the phase 1 Guess metric (CAR-03, NASA-03, APACHE-03), "
        "even where the target and the trial are the same.",
        "",
        "## A. Guided vs random selection",
        "",
        "**Recorded answer:** `{\"budget\", \"repeats\", \"ours\": [best row per repeat], \"random\": [best of "
        "`budget` random rows per repeat], \"verdict\"}`. Every row is scored with the frozen `harness.oracle.d2h` "
        "(lower is better). Two comparisons are reported, and they are different kinds of evidence:",
        "",
        "- **Empirical (archived repeats):** median d2h of each arm over the program's own repeats (20 in every "
        "valid answer); *median advantage* = median random − median guided (positive = guided better); *win "
        "fraction* = share of repeats where the guided row has lower d2h than the random row of the same repeat "
        "index. This relies on the program's random arm, so it is used only when the frozen check \"random "
        "baseline is really random\" passed.",
        "- **Analytical:** the exact expected best d2h of `budget` rows drawn uniformly at random from the whole "
        "CSV, computed in closed form; *advantage vs analytical* = that expectation − the median guided d2h. This "
        "does not depend on the program's random arm.",
        "",
        "Repeats are summarised within each trial; they are never counted as LLM trials. The archive records only "
        "the best row of each repeat, not the rows each arm evaluated, so it cannot be checked that those rows "
        "really are the best each arm saw.",
        "",
        "**Primary:** the program ran, the answer passed the frozen format check, the honeypot rerun passed (no "
        "budget peeking) and the reported budget equals the frozen budget. **Delivered** = primary and advantage "
        "vs analytical > 0 (analysis choice), over all 3 attempted trials.",
        "",
        "| domain | T | budget | attempted | exec. failed | bad answer | schema valid | honeypot failed | "
        "honeypot unverified | primary n | median guided d2h | analytical random best | median advantage vs "
        "analytical | empirical n | median random-arm d2h | median advantage (empirical) | median win fraction "
        "| delivery rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in cells("selection"):
        bad = s["answer_missing"] + s["answer_invalid_json"] + s["answer_schema_error"]
        lines.append(
            f"| {s['domain']} | {s['treatment']} | {s['budget']} | {s['attempted']} | "
            f"{s['execution_failed'] + s['generation_failed']} | {bad} | {s['schema_valid']} | {s['gate_failed']} | "
            f"{s['gate_unverified']} | {s['primary_n']} | {_num(s['median_guided_d2h'])} | "
            f"{_num(s['expected_random_best'])} | {_num(s['median_advantage_vs_analytic'], '+.3f')} | "
            f"{s['empirical_n']} | {_num(s['median_random_d2h'])} | {_num(s['median_advantage'], '+.3f')} | "
            f"{_num(s['median_win_fraction'], '.2f')} | {_pct(s['delivery_rate'])} |")
    lines += ["", "\"exec. failed\" includes steps that produced no program (`parse_failed`).", "",
              "### By domain", "",
              "| domain | request | attempted | schema valid | honeypot failed | primary n | median advantage vs "
              "analytical | delivery rate |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in [s for s in summary if s["metric"] == "selection" and s["scope"] == "domain"]:
        lines.append(f"| {s['domain']} | {s['request_id']} | {s['attempted']} | {s['schema_valid']} | "
                     f"{s['gate_failed']} | {s['primary_n']} | {_num(s['median_advantage_vs_analytic'], '+.3f')} | "
                     f"{_pct(s['delivery_rate'])} |")
    valid = [r for r in sel_rows if r["answer_state"] == "valid"]
    peek = [r for r in valid if r["honeypot_gate"] == "fail"]
    lines += ["", "### Budget peeking", "",
              f"{len(peek)} of the {len(valid)} schema-valid selection answers failed the honeypot check and are "
              "excluded. For context only (these values do not measure honest selection):", "",
              "| domain | schema valid | honeypot failed | median advantage vs analytical, excluded | median advantage "
              "vs analytical, primary |", "|---|---:|---:|---:|---:|"]
    for d in DOMAINS:
        v = [r for r in valid if r["domain"] == d]
        if v:
            ex = [r for r in v if r["honeypot_gate"] == "fail"]
            pr = [r for r in v if r["primary"]]
            lines.append(f"| {d} | {len(v)} | {len(ex)} | "
                         f"{_num(_med([r['advantage_vs_analytic'] for r in ex]), '+.3f')} | "
                         f"{_num(_med([r['advantage_vs_analytic'] for r in pr]), '+.3f')} |")
    arm = [r for r in valid if r["random_arm_gate"] == "fail"]
    lines += ["", "### Is the programs' random arm really random?", "",
              f"{len(arm)} of the {len(valid)} schema-valid answers failed the frozen check \"random baseline is really "
              "random\". The table compares each program's random arm (mean d2h of its recorded best-of-budget rows) "
              "with the exact expectation for uniform random sampling. A gap far from 0 means the reported random "
              "arm is not plain best-of-budget random sampling scored by d2h.", "",
              "| domain | schema valid | analytical random best | median of the programs' random-arm mean d2h | "
              "median gap | min–max gap |", "|---|---:|---:|---:|---:|---|"]
    for d in DOMAINS:
        v = [r for r in valid if r["domain"] == d]
        if v:
            gaps = [r["random_arm_vs_analytic"] for r in v]
            lines.append(f"| {d} | {len(v)} | {_num(v[0]['expected_random_best'])} | "
                         f"{_num(_med([r['mean_random_d2h'] for r in v]))} | {_num(_med(gaps), '+.3f')} | "
                         f"{min(gaps):+.3f}–{max(gaps):+.3f} |")
    lines.append("")
    lines += [
        "## B. Model vs historical-mean prediction", "",
        "Same definitions as phase 1: the every-fifth-row hold-out, the training-mean baseline, `mae_ratio` = model "
        "MAE ÷ baseline MAE (below 1 = better than the baseline). Both MAEs are recomputed from the recorded "
        "predictions and the CSV; the program's own `baseline`, `mae_model` and `mae_baseline` are compared but "
        "never used. **Primary:** ran, schema valid, mask leakage rerun passed.", "",
        "| domain | T | attempted | exec. failed | bad answer | schema valid | leakage failed | leakage unverified | "
        "primary n | median MAE ratio | min–max ratio | median MAE advantage | beats baseline | delivery rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for s in cells("prediction"):
        bad = s["answer_missing"] + s["answer_invalid_json"] + s["answer_schema_error"]
        rng = "—" if s["min_mae_ratio"] == "" else f"{s['min_mae_ratio']:.3f}–{s['max_mae_ratio']:.3f}"
        lines.append(
            f"| {s['domain']} | {s['treatment']} | {s['attempted']} | {s['execution_failed'] + s['generation_failed']} "
            f"| {bad} | {s['schema_valid']} | {s['gate_failed']} | {s['gate_unverified']} | {s['primary_n']} | "
            f"{_num(s['median_mae_ratio'])} | {rng} | {_num(s['median_mae_advantage'])} | "
            f"{s['beats_n']}/{s['primary_n']} | {_pct(s['delivery_rate'])} |")
    lines += ["", "### By domain", "",
              "| domain | request | baseline MAE | attempted | primary n | median MAE ratio | delivery rate |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for s in [s for s in summary if s["metric"] == "prediction" and s["scope"] == "domain"]:
        lines.append(f"| {s['domain']} | {s['request_id']} | {_num(s['baseline_mae'])} | {s['attempted']} | "
                     f"{s['primary_n']} | {_num(s['median_mae_ratio'])} | {_pct(s['delivery_rate'])} |")
    pv = [r for r in pred_rows if r["answer_state"] == "valid"]
    mism = {k: sum(r[k] is False for r in pv) for k in
            ("reported_baseline_matches", "reported_mae_model_matches", "reported_mae_baseline_matches")}
    lines += ["", "### Reported vs recomputed values", "",
              f"Among {len(pv)} schema-valid prediction answers, the program's reported value disagreed with the "
              f"recomputed one for: baseline mean {mism['reported_baseline_matches']}, model MAE "
              f"{mism['reported_mae_model_matches']}, baseline MAE {mism['reported_mae_baseline_matches']}. "
              "Only recomputed values are used above.", ""]
    lines += ["## Per-trial values", "",
              "| domain | T | trial | selection | guided d2h | advantage vs analytical | win fraction | prediction | "
              "MAE ratio |", "|---|---|---:|---|---:|---:|---:|---|---:|"]
    pmap = {(r["domain"], r["treatment"], r["trial"]): r for r in pred_rows}
    for r in sel_rows:
        p = pmap.get((r["domain"], r["treatment"], r["trial"]), {})
        lines.append(f"| {r['domain']} | {r['treatment']} | {r['trial']} | {r['exclusion_reason'] or 'included'} | "
                     f"{_num(r['median_guided_d2h'])} | {_num(r['advantage_vs_analytic'], '+.3f')} | "
                     f"{_num(r['win_fraction'], '.2f')} | {p.get('exclusion_reason') or 'included'} | "
                     f"{_num(p.get('mae_ratio', ''))} |")
    lines += [
        "", "## Limitations", "",
        "- **Three trials per cell, often fewer usable.** No statistical comparison between treatments is made.",
        "- **Repeats are not trials.** The 20 repeats inside one answer come from one program run; the win fraction "
        "describes that program, not variation between LLM trials.",
        "- **Pairing is as recorded.** Repeat i of `ours` is compared with repeat i of `random`; the archive does not "
        "show whether the program used the same seed or candidate pool for both arms.",
        "- **The arms' evaluated rows are not archived**, so the reported best rows cannot be checked against what "
        "each arm actually saw. The analytical baseline avoids relying on the program's random arm.",
        "- **\"Best\" is judged by d2h here.** The frozen checker also accepts z-score sums and rank sums; the programs "
        "may have used another overall score to pick their best rows.",
        "- **Budgets differ by domain** (Cars 30, NASA 18, Apache 30) and so do pool sizes.",
        "- **The prediction race uses the same split and target as Guess** for Cars and Apache (and NASA's Effort); "
        "it comes from a later step of the same trials, so it is not independent evidence.",
        "- **T7 is not included** in this phase.",
        "", "## Provenance", "",
        f"- Archive: `{manifest['inputs']['archive']['path']}` (sha256 `{manifest['inputs']['archive']['sha256']}`)",
        f"- Code commit: `{manifest['git']['commit']}` on `{manifest['git']['branch']}`"
        f"{' (uncommitted changes present)' if manifest['git']['dirty'] else ''}",
        f"- Command: `{manifest['command']}`",
        "- Full input and output hashes: `manifest.json`.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(out_dir, rows, summary, manifest, sel_rows=None, sel_summary=None, race=None):
    out = guard_out(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "prediction_mae.csv": to_csv(rows, CSV_FIELDS),
        "prediction_summary.csv": to_csv(summary, SUMMARY_FIELDS),
        "prediction_report.md": report(rows, summary, manifest),
    }
    if sel_rows is not None:
        files.update({
            "selection_quality.csv": to_csv(sel_rows, SELECTION_FIELDS),
            "selection_summary.csv": to_csv(sel_summary, SELECTION_SUMMARY_FIELDS),
            "selection_report.md": selection_report(sel_rows, sel_summary, manifest),
        })
    if race is not None:
        rs, rp, rsum = race
        files.update({
            "race_selection.csv": to_csv(rs, RACE_SELECTION_FIELDS),
            "race_prediction.csv": to_csv(rp, RACE_PREDICTION_FIELDS),
            "race_summary.csv": to_csv(rsum, RACE_SUMMARY_FIELDS),
            "race_report.md": race_report(rs, rp, rsum, manifest),
        })
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    manifest = dict(manifest, outputs={n: {"sha256": _sha(t.encode("utf-8"))} for n, t in files.items()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--runs", type=Path, default=RUNS_CSV)
    p.add_argument("--requirements", type=Path, default=REQUIREMENTS_CSV)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out", type=Path, default=OUT_ROOT / MODEL)
    p.add_argument("--domains", default=",".join(DOMAINS), help="comma-separated subset of cars,nasa,apache")
    a = p.parse_args(argv)
    domains = tuple(d for d in DOMAINS if d in a.domains.split(","))
    if not domains:
        p.error("no known domain in --domains")
    guard_out(a.out)
    rows = extract(a.archive, a.runs, a.requirements, a.data_dir, domains=domains)
    summary = summarize(rows)
    sel = extract_selection(a.archive, a.runs, a.requirements, a.data_dir, domains=domains)
    sel_summary = summarize_selection(sel)
    manifest = build_manifest(a.archive, a.runs, a.requirements, a.data_dir, argv)
    rsel = extract_race_selection(a.archive, a.runs, a.requirements, a.data_dir, domains=domains)
    rpred = extract_race_prediction(a.archive, a.runs, a.requirements, a.data_dir, domains=domains)
    out = write_outputs(a.out, rows, summary, manifest, sel, sel_summary, (rsel, rpred, summarize_race(rsel, rpred)))
    print(f"prediction: {len(rows)} attempts, {sum(r['primary'] for r in rows)} in the primary metric")
    print(f"selection:  {len(sel)} attempts, {sum(r['primary'] for r in sel)} in the primary metric")
    print(f"race selection:  {len(rsel)} attempts, {sum(r['primary'] for r in rsel)} in the primary metric")
    print(f"race prediction: {len(rpred)} attempts, {sum(r['primary'] for r in rpred)} in the primary metric")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
