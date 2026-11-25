"""Run treatments x domains x trials x skills against one LLM backend. Resumable.

    python -m harness --backend ollama --model qwen2.5-coder:14b --domains cars --trials 1-3

Protocol v2: every step is also judged by the semantic checker (harness/semantic.py)
and given a failure category (harness/failures.py). Outputs go under v2/ so the
v1 results are never touched.
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

from harness.backends import (BackendError, ContextOverflow, estimate_tokens, fits,
                              make_backend, with_retries)
from harness.checks import lua_executable, python_has_sklearn, run_check
from harness.failures import categorize
from harness.loc import measure_step
from harness.prompts import EZR_FILES, build_prompt, template_hash
from harness.reply import Reply, ReplyParseError, parse_reply
from harness.requirements import REPO_ROOT, load_domain
from harness.results import REQ_FIELDS, append_row, completed_keys, key_of
from harness.semantic import evaluate
from harness.treatments import TREATMENTS
from harness.workspace import apply_step, read_code, save_exchange, step_dir, trial_dir

HARNESS_DIR = Path(__file__).resolve().parent
LOC_FIELDS = ("new_loc", "deleted_loc", "test_loc", "total_loc", "pct_new")
PROTOCOL = "v2"
RUNS_ROOT = HARNESS_DIR / "runs" / PROTOCOL
RESULTS = HARNESS_DIR / "results" / PROTOCOL / "runs.csv"
# steps whose reply never became runnable code
NOT_RUN = {"parse_failed": "parse_failure", "output_truncated": "parse_failure", "context_overflow": "parse_failure"}


def requirements_path(results_path):
    return Path(results_path).parent / "requirements.csv"


@dataclass
class RunConfig:
    backend: object
    domains: list
    treatments: list
    trials: list
    skills: list
    runs_root: Path
    results_path: Path
    check_timeout: float = 120.0
    dry_run: bool = False
    backend_info: dict = field(default_factory=dict)


def parse_list(spec, prefix=""):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (p.strip()[len(prefix):] for p in part.split("-", 1))
            if lo.isdigit() and hi.isdigit():
                out += [f"{prefix}{i}" for i in range(int(lo), int(hi) + 1)]
                continue
        if part:
            out.append(part)
    return out


def run_step(cfg, domain, treatment, trial, skill):
    backend, i = cfg.backend, skill.index
    t = trial_dir(cfg.runs_root, backend.model, domain.name, treatment.name, trial)
    start = time.monotonic()
    prev = read_code(step_dir(t, i - 1)) if i > 1 else {}
    prompt = build_prompt(domain, skill, treatment, prev)
    save_exchange(t, i, prompt, None)
    row = {"model": backend.model, "backend": backend.name, "domain": domain.name,
           "treatment": treatment.name, "trial": trial, "skill_index": i, "skill": skill.name,
           "runs": False, "contract": False, "prompt_tokens_est": estimate_tokens(prompt),
           "prompt_template_hash": template_hash(), **cfg.backend_info, **{k: "" for k in LOC_FIELDS}}

    def finish(status, check=None, sem=None):
        row.update(status=status, seconds=round(time.monotonic() - start, 1),
                   timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if status != "backend_error":
            row["failure_category"], row["failure_detail"] = categorize(status, check, sem, treatment.background)
            if sem is None:
                row["semantic"] = NOT_RUN.get(status, "execution_failure")
            _requirement_rows(cfg, row, skill, sem, t)
        append_row(cfg.results_path, row)
        return row

    completion = None
    if fits(prompt, backend.context_tokens):
        try:
            completion = with_retries(lambda: backend.complete(prompt))
        except ContextOverflow:
            completion = None
        except BackendError:
            return finish("backend_error")
    if completion is None:
        apply_step(t, i, None)
        return finish("context_overflow")

    row.update(prompt_tokens=completion.prompt_tokens, response_tokens=completion.response_tokens,
               finish_reason=completion.finish_reason)
    save_exchange(t, i, prompt, completion.text)
    if completion.finish_reason == "length":
        apply_step(t, i, None)
        return finish("output_truncated")
    try:
        reply = parse_reply(completion.text)
        if treatment.background == "ezr":  # the provided library is not the LLM's code
            reply = Reply({k: v for k, v in reply.files.items() if k not in EZR_FILES}, reply.deletes)
        step = apply_step(t, i, reply)
    except (ReplyParseError, OSError, ValueError):
        apply_step(t, i, None)
        return finish("parse_failed")

    check = run_check(step, skill, treatment, domain.csv_path, cfg.check_timeout)
    sem = evaluate([r.id for r in skill.requests], domain.csv_path, check,
                   lambda path: run_check(step, skill, treatment, path, cfg.check_timeout),
                   evidence=t / f"step{i:02d}.semantic.json")
    row["semantic"] = sem.status
    (t / f"step{i:02d}.check.txt").write_text(
        f"runs={check.runs} contract={check.contract} timed_out={check.timed_out} "
        f"output_capped={check.output_capped}\n"
        f"files written at runtime (discarded): {', '.join(check.created) or 'none'}\n"
        f"--- stdout ---\n{check.stdout}\n--- stderr ---\n{check.stderr}\n", encoding="utf-8")
    loc = measure_step(prev, read_code(step))
    row.update(runs=check.runs, contract=check.contract, output_capped=check.output_capped, new_loc=loc.new_loc,
               deleted_loc=loc.deleted_loc, test_loc=loc.test_loc,
               total_loc=loc.total_loc, pct_new=loc.pct_new)
    return finish("timeout" if check.timed_out else "ok", check, sem)


def _requirement_rows(cfg, row, skill, sem, trial_path):
    """One requirements.csv row per request, for every finished step (failures included)."""
    i = int(row["skill_index"])
    evidence = _rel(trial_path / f"step{i:02d}.semantic.json") if sem else ""
    by_id = {r.request_id: r for r in sem.requests} if sem else {}
    for req in skill.requests:
        r = by_id.get(req.id)
        out = {k: row.get(k) for k in ("model", "domain", "treatment", "trial", "skill_index", "skill")}
        out["request_id"] = req.id
        if r is None:
            out.update(status=row["semantic"], checks_passed=0, checks_total=0, failed_checks="",
                       notes=row["status"], evidence="")
        else:
            out.update(status=r.status, checks_passed=r.passed, checks_total=r.total,
                       failed_checks=" | ".join(r.failed)[:2000], notes=" | ".join(r.notes)[:500], evidence=evidence)
        append_row(requirements_path(cfg.results_path), out, REQ_FIELDS)


def preflight(cfg):
    """Refuse to start a night whose failures would be the machine's, not the model's."""
    chosen = [TREATMENTS[n] for n in cfg.treatments]
    if any(t.lang == "lua" for t in chosen) and lua_executable() is None:
        raise SystemExit("T5/T6 need Lua: install lua/luajit or set HARNESS_LUA")
    if any(t.background == "ezr" for t in chosen) and sys.version_info < (3, 12):
        raise SystemExit(f"T5/T6 need Python 3.12+ for ezr.py; this is {sys.version.split()[0]}")
    if any(t.background == "sklearn" for t in chosen) and not python_has_sklearn():
        raise SystemExit(f"T3/T4 need scikit-learn in {sys.executable}: pip install -r requirements.txt")


def _pkg_version(name):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _lua_version():
    exe = lua_executable()
    if exe is None:
        return "not found"
    try:
        import subprocess
        p = subprocess.run([exe, "-v"], capture_output=True, text=True, timeout=10)
        return ((p.stdout or p.stderr).strip().split("  ")[0]) or os.path.basename(exe)
    except Exception:
        return os.path.basename(exe)


def _rel(path):
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def _banner(cfg, domains, done, wanted, out):
    """Print what is about to run, so a wrong flag shows up before the night starts."""
    b, info = cfg.backend, cfg.backend_info
    total = remaining = 0
    for domain in domains:
        for trial in cfg.trials:
            for name in cfg.treatments:
                t = trial_dir(cfg.runs_root, b.model, domain.name, name, trial)
                for skill in domain.skills:
                    if skill.index not in wanted:
                        continue
                    total += 1
                    key = key_of({"model": b.model, "domain": domain.name, "treatment": name,
                                  "trial": trial, "skill_index": skill.index})
                    if not (key in done and step_dir(t, skill.index).is_dir()):
                        remaining += 1
    chosen = [TREATMENTS[n] for n in cfg.treatments]
    venv = " (venv)" if sys.prefix != sys.base_prefix else " (no venv)"
    url = getattr(b, "base_url", None)
    out(f"harness start {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out(f"  backend   {b.name}  model {b.model}" + (f"  at {url}" if url else ""))
    out(f"  model     quantization {info.get('quantization', '?')}  file/digest {info.get('model_digest', '?')}")
    out(f"  sampling  temperature {info.get('temperature', 'backend default')}  think {info.get('think', 'off')}")
    out(f"  context   {b.context_tokens} tokens" + (" (matches server)" if hasattr(b, "verify") else ""))
    out(f"  python    {sys.version.split()[0]}{venv}" + (
        f"  scikit-learn {_pkg_version('scikit-learn')}  numpy {_pkg_version('numpy')}"
        if any(t.background == "sklearn" for t in chosen) else ""))
    if any(t.lang == "lua" for t in chosen):
        out(f"  lua       {_lua_version()}")
    out(f"  protocol  {PROTOCOL}  template {template_hash()}  check timeout {cfg.check_timeout:g}s")
    out(f"  plan      {','.join(d.name for d in domains)} x {','.join(cfg.treatments)} x trials "
        f"{','.join(map(str, cfg.trials))} x skills {','.join(str(s) for s in sorted(wanted))} = {total} steps, "
        f"{total - remaining} already done, {remaining} to run")
    out(f"  output    {_rel(cfg.results_path)}, {_rel(cfg.runs_root)}/{b.model}/")


def _progress(row):
    head = f"{row['domain']} {row['treatment']} trial {row['trial']} skill {row['skill_index']}/10"
    if row["status"] != "ok":
        return f"{head} {row['status']} ({row['seconds']:.0f}s)"
    mark = lambda ok: "✓" if ok else "✗"
    return (f"{head} {mark(row['runs'])} runs {mark(row['contract'])} contract "
            f"{mark(row.get('semantic') == 'pass')} semantic ({row['seconds']:.0f}s)"
            + (f"  [{row['failure_category']}]" if row.get("failure_category") else ""))


def _dry_run(cfg, domains, out):
    folder = cfg.runs_root / "_dryrun"
    folder.mkdir(parents=True, exist_ok=True)
    for domain in domains:
        for name in cfg.treatments:
            p = build_prompt(domain, domain.skills[0], TREATMENTS[name], {})
            (folder / f"{domain.name}_{name}.txt").write_text(p, encoding="utf-8")
            out(f"{domain.name} {name} skill1 est {estimate_tokens(p)} tokens "
                f"(context {cfg.backend.context_tokens})")


def run(cfg, out=print):
    domains = [load_domain(d) for d in cfg.domains]
    if cfg.dry_run:
        return _dry_run(cfg, domains, out)
    preflight(cfg)
    if hasattr(cfg.backend, "verify"):
        try:
            cfg.backend.verify()
        except BackendError as e:
            raise SystemExit(str(e))
    if not cfg.backend_info and hasattr(cfg.backend, "describe"):
        try:
            cfg.backend_info = {k: v for k, v in cfg.backend.describe().items() if v is not None}
        except Exception:
            cfg.backend_info = {}
    done = completed_keys(cfg.results_path)
    wanted = {int(s) for s in cfg.skills}
    _banner(cfg, domains, done, wanted, out)
    # trial is outside treatment so the matched pairs fill evenly if a night runs short
    for domain in domains:
        for trial in cfg.trials:
            for name in cfg.treatments:
                treatment = TREATMENTS[name]
                t = trial_dir(cfg.runs_root, cfg.backend.model, domain.name, name, trial)
                for skill in domain.skills:
                    if skill.index not in wanted:
                        continue
                    key = key_of({"model": cfg.backend.model, "domain": domain.name,
                                  "treatment": name, "trial": trial, "skill_index": skill.index})
                    if key in done and step_dir(t, skill.index).is_dir():
                        continue
                    row = run_step(cfg, domain, treatment, trial, skill)
                    out(_progress(row))
                    if row["status"] == "backend_error":
                        break


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m harness", description=__doc__.split("\n")[0])
    ap.add_argument("--backend", required=True, choices=["ollama", "llamacpp", "openai", "claude-cli", "fake"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url")
    ap.add_argument("--context", type=int, help="context window in tokens")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--think", choices=["low", "medium", "high"],
                    help="reasoning effort for thinking models (ollama, llamacpp), e.g. gpt-oss")
    ap.add_argument("--domains", default="cars,nasa,apache")
    ap.add_argument("--treatments", default="T1-T6")
    ap.add_argument("--trials", default="1-3")
    ap.add_argument("--skills", default="1-10")
    ap.add_argument("--check-timeout", type=float, default=120.0)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    treatments = parse_list(a.treatments, "T")
    unknown = [t for t in treatments if t not in TREATMENTS]
    if unknown:
        ap.error(f"unknown treatments {unknown}")
    cfg = RunConfig(make_backend(a.backend, a.model, a.base_url, a.context, a.temperature, a.think),
                    parse_list(a.domains), treatments, [int(x) for x in parse_list(a.trials)],
                    [int(x) for x in parse_list(a.skills)], a.runs_root, a.results,
                    a.check_timeout, a.dry_run)
    try:
        run(cfg)
    except KeyboardInterrupt:
        print("\nstopped. Finished steps are saved; rerun the same command to resume.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
