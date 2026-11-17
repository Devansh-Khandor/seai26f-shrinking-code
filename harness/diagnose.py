"""Ask the LLM why each failed step failed: a SUSPECTED cause, kept beside the confirmed one.

    python -m harness.diagnose --backend llamacpp --model gpt-oss-20b

runs.csv already holds a failure_category derived from evidence (harness/failures.py).
This step adds a second opinion: for every failed step it sends the request, the
generated code, the error output and the failed checks to the same backend, and
asks whether the cause looks like a system problem or a fundamental code error.
Answers go to diagnoses.csv next to runs.csv, marked as suspected; they never
change a pass/fail verdict. Resumable: steps already diagnosed are skipped.
"""

import argparse
import json
from pathlib import Path
import re
import sys

from harness.backends import BackendError, make_backend, with_retries
from harness.requirements import load_domain
from harness.results import append_row, latest_rows, load_rows
from harness.workspace import step_dir, trial_dir

HARNESS_DIR = Path(__file__).resolve().parent
FIELDS = ("model", "domain", "treatment", "trial", "skill_index", "confirmed_category", "suspected_cause",
          "suspected_category", "explanation", "basis")
KEY = ("model", "domain", "treatment", "trial", "skill_index")
CATEGORIES = ("system:dependency", "system:resource", "system:harness", "generation:extraction", "generation:truncated",
              "code:syntax", "code:forbidden_import", "code:api_misuse", "code:runtime", "code:timeout",
              "code:no_answer", "code:format", "code:semantic", "code:fragile", "unclear")
MAX_CODE = 12_000
_JSON = re.compile(r"\{.*\}", re.S)


def _tail(path, n):
    p = Path(path)
    if not p.is_file():
        return "(none)"
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[-n:]


def build_prompt(row, runs_root):
    domain = load_domain(row["domain"])
    skill = domain.skills[int(row["skill_index"]) - 1]
    t = trial_dir(runs_root, row["model"], row["domain"], row["treatment"], int(row["trial"]))
    i = int(row["skill_index"])
    code_dir = step_dir(t, i)
    files = sorted(p for p in code_dir.rglob("*") if p.is_file() and p.suffix in (".py", ".lua")) \
        if code_dir.is_dir() else []
    code = "\n\n".join(f"FILE: {p.relative_to(code_dir).as_posix()}\n{p.read_text(encoding='utf-8', errors='replace')}"
                       for p in files)[-MAX_CODE:] or "(no code was extracted)"
    sem = _tail(t / f"step{i:02d}.semantic.json", 3000)
    requests = "\n".join(f"- {r.id}: {r.text}" for r in skill.requests)
    return f"""You are reviewing why one step of an automated code-generation experiment failed.
Decide whether the failure is a SYSTEM problem (the machine, environment, missing dependency, harness, or time
limit too small for correct code) or a fundamental CODE problem (the generated program is wrong).

Treatment {row['treatment']}, domain {row['domain']}, skill {i} ({skill.name}). The requests were:
{requests}

Harness status: {row['status']}. Program ran: {row['runs']}. Original format contract: {row['contract']}.
Semantic check: {row.get('semantic')}. Category assigned from the evidence: {row.get('failure_category')}
({row.get('failure_detail')}).

Generated code (possibly truncated):
{code}

Program output and errors (tail):
{_tail(t / f"step{i:02d}.check.txt", 4000)}

Semantic check record (tail):
{sem}

Model reply (tail, for steps whose reply could not be used):
{_tail(t / f"step{i:02d}.response.txt", 2000) if row['status'] in ('parse_failed', 'output_truncated') else '(not needed)'}

Reply with only one JSON object: {{"cause": "system" or "code" or "unclear", "category": one of
{list(CATEGORIES)}, "explanation": "one or two sentences citing the evidence"}}"""


def parse_diagnosis(text):
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or d.get("cause") not in ("system", "code", "unclear"):
        return None
    if d.get("category") not in CATEGORIES:
        d["category"] = "unclear"
    return d


def failed_rows(rows, model):
    return [r for r in latest_rows(rows) if r["model"] == model and r.get("failure_category")
            and r["status"] != "backend_error"]


def diagnose(backend, results_path, runs_root, out=print):
    results_path = Path(results_path)
    out_path = results_path.parent / "diagnoses.csv"
    done = {tuple(str(r[k]) for k in KEY) for r in load_rows(out_path)}
    todo = [r for r in failed_rows(load_rows(results_path), backend.model)
            if tuple(str(r[k]) for k in KEY) not in done]
    out(f"{len(todo)} failed step(s) to diagnose ({len(done)} already done)")
    for r in todo:
        try:
            reply = with_retries(lambda: backend.complete(build_prompt(r, runs_root)))
        except BackendError as e:
            out(f"backend error, stopping: {e}")
            return
        d = parse_diagnosis(getattr(reply, "text", reply))
        row = {k: r[k] for k in KEY}
        row["confirmed_category"] = r.get("failure_category")
        if d is None:
            row.update(suspected_cause="unparsed", suspected_category="", explanation="", basis="suspected (LLM)")
        else:
            row.update(suspected_cause=d["cause"], suspected_category=d["category"],
                       explanation=str(d.get("explanation", ""))[:500], basis="suspected (LLM)")
        append_row(out_path, row, FIELDS)
        out(f"{r['domain']} {r['treatment']} trial {r['trial']} skill {r['skill_index']}: "
            f"confirmed {row['confirmed_category']}, suspected {row['suspected_cause']}/{row['suspected_category']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m harness.diagnose", description=__doc__.split("\n")[0])
    ap.add_argument("--backend", required=True, choices=["ollama", "llamacpp", "openai", "claude-cli", "fake"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url")
    ap.add_argument("--context", type=int)
    ap.add_argument("--think", choices=["low", "medium", "high"])
    ap.add_argument("--runs-root", type=Path, default=HARNESS_DIR / "runs" / "v2")
    ap.add_argument("--results", type=Path, default=HARNESS_DIR / "results" / "v2" / "runs.csv")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    diagnose(make_backend(a.backend, a.model, a.base_url, a.context, None, a.think), a.results, a.runs_root)


if __name__ == "__main__":
    main()
