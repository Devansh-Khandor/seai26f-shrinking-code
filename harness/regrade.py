"""Re-grade finished steps with the current semantic checker. No LLM calls.

Each selected step's saved program is run again on the domain CSV and on the controlled copies,
exactly as during the run, and its semantic verdict, failure category and request rows are replaced.
Prompts, replies, code and LOC are untouched. Run it only while no harness run is writing the CSVs.

    python -m harness.regrade --model gpt-oss-20b --skills 5,6

Only steps whose program ran (status ok, runs True) are re-graded; crashes, timeouts and parse
failures keep their rows. The previous CSVs are kept as *.pre-regrade.csv, each step's previous
evidence as stepNN.semantic.pre-regrade.json, and every changed verdict is listed in regrade.txt.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil

from harness.checks import run_check
from harness.failures import categorize
from harness.requirements import load_domain
from harness.results import REQ_FIELDS
from harness.run import RESULTS, RUNS_ROOT, _rel, parse_list, requirements_path
from harness.semantic import evaluate
from harness.treatments import TREATMENTS
from harness.workspace import step_dir, trial_dir

STEP = ("model", "domain", "treatment", "trial", "skill_index")


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def _write(path, fields, rows):
    tmp = Path(path).with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def _key(row):
    return tuple(str(row[k]) for k in STEP)


def regrade_step(row, runs_root, timeout):
    """(new runs.csv fields, new requirement rows) for one finished step."""
    domain = load_domain(row["domain"])
    skill = domain.skills[int(row["skill_index"]) - 1]
    treatment = TREATMENTS[row["treatment"]]
    t = trial_dir(runs_root, row["model"], row["domain"], row["treatment"], row["trial"])
    i = int(row["skill_index"])
    step = step_dir(t, i)
    evidence = t / f"step{i:02d}.semantic.json"
    if evidence.exists():
        shutil.copyfile(evidence, evidence.with_name(f"step{i:02d}.semantic.pre-regrade.json"))
    check = run_check(step, skill, treatment, domain.csv_path, timeout)
    sem = evaluate([r.id for r in skill.requests], domain.csv_path, check,
                   lambda path: run_check(step, skill, treatment, path, timeout), evidence=evidence)
    cat, detail = categorize(row["status"], check, sem, treatment.background)
    new = {"semantic": sem.status, "failure_category": cat, "failure_detail": detail}
    by_id = {r.request_id: r for r in sem.requests}
    reqs = []
    for req in skill.requests:
        r = by_id[req.id]
        out = {k: row[k] for k in ("model", "domain", "treatment", "trial", "skill_index", "skill")}
        out.update(request_id=req.id, status=r.status, checks_passed=r.passed, checks_total=r.total,
                   failed_checks=" | ".join(r.failed)[:2000], notes=" | ".join(r.notes)[:500],
                   evidence=_rel(evidence))
        reqs.append(out)
    return new, reqs, check


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--domains", default="cars,nasa,apache")
    ap.add_argument("--treatments", default="T1-T6")
    ap.add_argument("--trials", default="1-100")
    ap.add_argument("--skills", default="1-10")
    ap.add_argument("--check-timeout", type=float, default=120.0)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    domains, treatments = set(parse_list(a.domains)), set(parse_list(a.treatments, "T"))
    trials, skills = set(parse_list(a.trials)), set(parse_list(a.skills))

    run_fields, run_rows = _read(a.results)
    req_path = requirements_path(a.results)
    req_fields, req_rows = _read(req_path)
    assert tuple(req_fields) == REQ_FIELDS, "requirements.csv has unexpected columns"
    chosen = [r for r in run_rows if r["model"] == a.model and r["domain"] in domains
              and r["treatment"] in treatments and r["trial"] in trials and r["skill_index"] in skills
              and r["status"] == "ok" and r["runs"] == "True"]
    print(f"re-grading {len(chosen)} steps")

    old_req = {}
    for r in req_rows:
        old_req.setdefault(_key(r), {})[r["request_id"]] = r
    new_req, log = {}, []
    for n, row in enumerate(chosen, 1):
        new, reqs, check = regrade_step(row, a.runs_root, a.check_timeout)
        tag = " ".join(row[k] for k in ("domain", "treatment", "trial", "skill_index"))
        if not check.runs:
            log.append(f"{tag}: WARNING the program did not run this time; verdict now {new['semantic']}")
        if new["semantic"] != row["semantic"]:
            log.append(f"{tag}: step {row['semantic']} -> {new['semantic']}")
        for q in reqs:
            before = old_req.get(_key(row), {}).get(q["request_id"], {})
            if before.get("status") != q["status"]:
                log.append(f"{tag} {q['request_id']}: {before.get('status')} -> {q['status']}"
                           f"  was: {before.get('failed_checks', '')[:200]}  now: {q['failed_checks'][:200]}")
        row.update(new)
        new_req[_key(row)] = reqs
        print(f"[{n}/{len(chosen)}] {tag}: {new['semantic']}", flush=True)

    shutil.copyfile(a.results, Path(a.results).with_suffix(".pre-regrade.csv"))
    shutil.copyfile(req_path, Path(req_path).with_suffix(".pre-regrade.csv"))
    _write(a.results, run_fields, run_rows)
    out, emitted = [], set()
    for r in req_rows:
        k = _key(r)
        if k in new_req:
            if k not in emitted:
                out += new_req[k]
                emitted.add(k)
        else:
            out.append(r)
    _write(req_path, REQ_FIELDS, out)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = [f"regrade {stamp}: {len(chosen)} steps ({a.domains}; {a.treatments}; trials {a.trials}; skills {a.skills})",
              *log, f"{len(log)} changes", ""]
    with open(Path(a.results).parent / "regrade.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
