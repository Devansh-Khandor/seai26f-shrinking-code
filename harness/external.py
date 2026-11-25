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


def _row(z, names, root, model, domain, t, k, rid, target, direction, data, base_mean, base_mae, n_held, runs, reqs):
    row = {f: "" for f in CSV_FIELDS}
    row.update(model=model, domain=domain, treatment=t, trial=k, skill_index=GUESS_STEP, request_id=rid,
               target=target, target_direction=direction, baseline_mean=base_mean, baseline_mae=base_mae,
               n_heldout=n_held, primary=False, leakage_gate="not_run", shuffled_gate="not_run")
    run = runs.get((model, domain, t, str(k), str(GUESS_STEP)))
    req = reqs.get((model, domain, t, str(k), rid))
    row["request_status"] = req["status"] if req else ""
    if run is None:
        row["exclusion_reason"] = "not_attempted"
        return row
    row.update(step_status=run["status"], step_runs=run["runs"], step_semantic=run["semantic"])
    check = entry_name(root, model, domain, t, k, GUESS_STEP, "check.txt")
    sem_name = entry_name(root, model, domain, t, k, GUESS_STEP, "semantic.json")
    if run["status"] != "ok" or check not in names:
        row["answer_state"] = "no_output"
        row["exclusion_reason"] = "generation_failed"
        row["answer_detail"] = f"step status {run['status']}"
        return row
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
        return row
    ans = A.parse_answers(stdout).get(rid)
    if ans is None:
        row.update(answer_state="missing", exclusion_reason="answer_missing")
        return row
    if isinstance(ans, str):
        row.update(answer_state="invalid_json", answer_detail=ans[:200], exclusion_reason="answer_invalid_json")
        return row
    ctx = A.check_request(rid, data, ans, "base")
    bad = [d for name, ok, d in ctx.items if name.endswith("answer format") and not ok]
    if bad:
        row.update(answer_state="schema_error", answer_detail=bad[0][:200], exclusion_reason="answer_schema_error")
        return row
    row["answer_state"] = "valid"
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
        "analysis": "external metrics phase 1: prediction MAE vs historical-mean baseline",
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


def write_outputs(out_dir, rows, summary, manifest):
    out = guard_out(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "prediction_mae.csv": to_csv(rows, CSV_FIELDS),
        "prediction_summary.csv": to_csv(summary, SUMMARY_FIELDS),
        "prediction_report.md": report(rows, summary, manifest),
    }
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
    manifest = build_manifest(a.archive, a.runs, a.requirements, a.data_dir, argv)
    out = write_outputs(a.out, rows, summary, manifest)
    prim = sum(r["primary"] for r in rows)
    print(f"{len(rows)} attempts, {prim} in the primary metric; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
