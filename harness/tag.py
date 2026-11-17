"""Git-tag every finished trial so the histograms are derivable from tags. Never calls an LLM.

    python -m harness.tag --model gpt-oss-20b                 # protocol v2 (default)
    python -m harness.tag --model gpt-oss-20b --protocol v1   # the original 3-trial runs
    git push origin --tags

Each trial becomes its own small history, off every branch: an empty start
commit, then one commit per step whose tree is exactly the harness snapshot
runs/<model>/<domain>/<T#>/trial<k>/stepNN/. A step that failed carries the
previous snapshot forward, as the harness does. Tags are named

    v2-<t#>-<domain>-r<trial>-step<k>   e.g. v2-t1-cars-r1-step1  (protocol v2)
    <t#>-<domain>-r<trial>-step<k>      e.g. t1-cars-r4-step1     (protocol v1)

Only trials whose ten steps are all finished are tagged. Existing tags are
never moved: a trial that is already tagged is checked, not rewritten. After
tagging, every tagged trial is re-measured from its tags alone with
harness.loc, and any difference from runs.csv is an error.
"""

import argparse
from datetime import datetime
import subprocess
import sys
from pathlib import Path

from harness.loc import measure_step
from harness.results import DONE_STATUSES, latest_rows, load_rows
from harness.workspace import read_code, step_dir, trial_dir

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent
STEPS = range(1, 11)
TMP_REF = "refs/harness-tag/tmp"


class TagError(Exception):
    pass


PROTOCOLS = {  # name: (tag prefix, runs root, results csv)
    "v1": ("", HARNESS_DIR / "runs", HARNESS_DIR / "results" / "runs.csv"),
    "v2": ("v2-", HARNESS_DIR / "runs" / "v2", HARNESS_DIR / "results" / "v2" / "runs.csv"),
}


def tag_name(treatment, domain, trial, step, prefix=""):
    return f"{prefix}{treatment.lower()}-{domain}-r{trial}-step{step}"


def git(repo, *args, data=None):
    p = subprocess.run(["git", "-C", str(repo), *args], input=data, capture_output=True)
    if p.returncode != 0:
        raise TagError(f"git {' '.join(args)} failed: {p.stderr.decode(errors='replace').strip()}")
    return p.stdout


def existing_tags(repo):
    return set(git(repo, "tag", "-l").decode().split())


def tree_files(repo, rev):
    """{path: text} for the tree at rev, read the same way as workspace.read_code."""
    entries = []
    for line in git(repo, "ls-tree", "-r", "-z", rev).split(b"\0"):
        if line:
            meta, path = line.split(b"\t", 1)
            if meta.split()[1] == b"blob":
                entries.append((path.decode(), meta.split()[2].decode()))
    if not entries:
        return {}
    raw = git(repo, "cat-file", "--batch", data="".join(f"{sha}\n" for _, sha in entries).encode())
    out, pos = {}, 0
    for path, _ in entries:
        header_end = raw.index(b"\n", pos)
        size = int(raw[pos:header_end].split()[2])
        body = raw[header_end + 1:header_end + 1 + size]
        pos = header_end + 1 + size + 1
        try:
            out[path] = body.decode("utf-8")
        except UnicodeDecodeError:
            continue
    return out


def trials_in(rows, model):
    """{(domain, treatment, trial): {step: row}} for this model's latest rows."""
    out = {}
    for r in latest_rows(rows):
        if r["model"] == model:
            out.setdefault((r["domain"], r["treatment"], int(r["trial"])), {})[int(r["skill_index"])] = r
    return out


def is_finished(steps):
    return set(steps) >= set(STEPS) and all(steps[i]["status"] in DONE_STATUSES for i in STEPS)


def snapshots(runs_root, model, domain, treatment, trial, steps):
    """The code for each step, rebuilding carried-forward steps the zip/cleanup may have dropped."""
    t = trial_dir(runs_root, model, domain, treatment, trial)
    out, prev = {}, {}
    for i in STEPS:
        d = step_dir(t, i)
        if d.is_dir():
            code = read_code(d)
        elif steps[i]["status"] == "ok":
            raise TagError(f"{d} is missing but runs.csv says step {i} is ok")
        else:
            code = prev
        out[i], prev = code, code
    return out


def epoch(timestamp):
    return int(datetime.fromisoformat(timestamp).timestamp())


def message(model, domain, treatment, trial, row):
    i = int(row["skill_index"])
    return (f"{treatment} {domain} trial {trial} step {i}: {row['skill']} ({model})\n\n"
            f"Harness snapshot step{i:02d}, status {row['status']}, "
            f"new_loc {row['new_loc'] or '-'}, total_loc {row['total_loc'] or '-'}, "
            f"prompt template {row['prompt_template_hash'] or '-'}.\n")


def fast_import_stream(model, ident, trials, prefix=""):
    """trials: [(domain, treatment, trial, steps, code)] -> bytes for git fast-import."""
    parts = []

    def w(s):
        parts.append(s.encode() if isinstance(s, str) else s)

    def data(text):
        b = text.encode() if isinstance(text, str) else text
        w(f"data {len(b)}\n")
        w(b)
        w("\n")

    mark = 0
    for domain, treatment, trial, steps, code in trials:
        mark += 1
        w(f"reset {TMP_REF}\ncommit {TMP_REF}\nmark :{mark}\n")
        w(f"committer {ident} {epoch(steps[1]['timestamp']) - 1} +0000\n")
        data(f"{treatment} {domain} trial {trial}: empty start ({model})\n")
        w("deleteall\n\n")
        for i in STEPS:
            when = epoch(steps[i]["timestamp"])
            msg = message(model, domain, treatment, trial, steps[i])
            mark += 1
            w(f"commit {TMP_REF}\nmark :{mark}\ncommitter {ident} {when} +0000\n")
            data(msg)
            w("deleteall\n")
            for path in sorted(code[i]):
                if any(c in path for c in '"\\\n') or path.startswith(" "):
                    raise TagError(f"unsupported file name in {treatment} {domain} trial {trial}: {path!r}")
                w(f"M 100644 inline {path}\n")
                data(code[i][path])
            w("\n")
            w(f"tag {tag_name(treatment, domain, trial, i, prefix)}\nfrom :{mark}\ntagger {ident} {when} +0000\n")
            data(msg)
    return b"".join(parts)


def verify_trial(repo, model, domain, treatment, trial, steps, prefix=""):
    """Re-measure one trial from its tags alone; return a list of problems."""
    problems = []
    for i in STEPS:
        tag = tag_name(treatment, domain, trial, i, prefix)
        subject = git(repo, "log", "-1", "--format=%s", tag).decode().strip()
        if not subject.endswith(f"({model})"):
            problems.append(f"{tag}: belongs to another model ({subject})")
            continue
        cur, prev = tree_files(repo, tag), tree_files(repo, f"{tag}^")
        row = steps[i]
        if row["status"] != "ok":
            if cur != prev:
                problems.append(f"{tag}: status {row['status']} but the code changed")
            continue
        loc = measure_step(prev, cur)
        got = (loc.new_loc, loc.deleted_loc, loc.test_loc, loc.total_loc, loc.pct_new)
        want = (int(row["new_loc"]), int(row["deleted_loc"]), int(row["test_loc"] or 0),
                int(row["total_loc"]), float(row["pct_new"]))
        if got != want:
            problems.append(f"{tag}: tags give {got}, runs.csv has {want}")
    return problems


def tag_runs(model, runs_root, results_path, repo=REPO_ROOT, out=print, prefix=""):
    trials = trials_in(load_rows(results_path), model)
    if not trials:
        raise TagError(f"no rows for model {model!r} in {results_path}")
    have = existing_tags(repo)
    ident = "{} <{}>".format(git(repo, "config", "user.name").decode().strip(),
                             git(repo, "config", "user.email").decode().strip())
    new, already, unfinished = [], [], []
    for key in sorted(trials):
        domain, treatment, trial = key
        steps = trials[key]
        names = [tag_name(treatment, domain, trial, i, prefix) for i in STEPS]
        present = [n for n in names if n in have]
        if len(present) == len(names):
            already.append(key)
        elif present:
            raise TagError(f"{treatment} {domain} trial {trial} is partly tagged ({', '.join(present)}); "
                           "fix by hand, tags are never moved")
        elif not is_finished(steps):
            unfinished.append(key)
        else:
            new.append((domain, treatment, trial, steps,
                        snapshots(runs_root, model, domain, treatment, trial, steps)))

    if new:
        git(repo, "fast-import", "--quiet", data=fast_import_stream(model, ident, new, prefix))
        git(repo, "update-ref", "-d", TMP_REF)

    problems = []
    for domain, treatment, trial in already + [n[:3] for n in new]:
        problems += verify_trial(repo, model, domain, treatment, trial, trials[(domain, treatment, trial)], prefix)

    out(f"tagged {len(new)} new trial(s) ({10 * len(new)} tags), "
        f"{len(already)} already tagged, {len(unfinished)} not finished")
    for domain, treatment, trial in unfinished:
        out(f"  not finished, skipped: {treatment} {domain} trial {trial}")
    if problems:
        for p in problems:
            out(f"  MISMATCH {p}")
        raise TagError(f"{len(problems)} tag(s) do not match runs.csv")
    out(f"all {len(already) + len(new)} tagged trial(s) re-measure from tags exactly as in runs.csv")
    if new:
        out("push with: git push origin --tags")
    return [n[:3] for n in new]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m harness.tag", description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="v2")
    ap.add_argument("--runs-root", type=Path)
    ap.add_argument("--results", type=Path)
    a = ap.parse_args(argv)
    prefix, runs_root, results = PROTOCOLS[a.protocol]
    try:
        tag_runs(a.model, a.runs_root or runs_root, a.results or results, prefix=prefix)
    except TagError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
